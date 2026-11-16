import os
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split, KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error
from scipy.optimize import minimize
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

# Reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Create required output directories
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# 1. Load Raw Data
train_raw = pd.read_parquet("./input/train.parquet")
test_raw = pd.read_parquet("./input/test.parquet")

target_col = "NADAC Rate"
id_col = "id" if "id" in test_raw.columns else None

# Clean missing target rows if present
train_raw = train_raw.dropna(subset=[target_col]).reset_index(drop=True)

# 2. Strict Train / Validation Split Before Feature Processing
train_df, val_df = train_test_split(train_raw, test_size=0.2, random_state=42)
train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)
test_df = test_raw.reset_index(drop=True)

y_train = train_df[target_col].values.astype(np.float32)
y_val = val_df[target_col].values.astype(np.float32)

y_train_log = np.log1p(np.maximum(0, y_train))
y_val_log = np.log1p(np.maximum(0, y_val))

# 3. Column Partitioning & Feature Engineering
feature_candidates = [c for c in train_df.columns if c not in [target_col, "id"]]

date_cols = []
text_cols = []
cat_cols = []
num_cols = []

for c in feature_candidates:
    c_lower = c.lower()
    if (
        "date" in c_lower
        or "time" in c_lower
        or pd.api.types.is_datetime64_any_dtype(train_df[c])
    ):
        date_cols.append(c)
    elif train_df[c].dtype == "object" or pd.api.types.is_string_dtype(train_df[c]):
        sample_str = train_df[c].dropna().astype(str)
        avg_words = (
            sample_str.apply(lambda x: len(x.split())).mean()
            if len(sample_str) > 0
            else 0
        )
        if avg_words > 2 or "desc" in c_lower or "name" in c_lower:
            text_cols.append(c)
        else:
            cat_cols.append(c)
    elif pd.api.types.is_numeric_dtype(train_df[c]):
        num_cols.append(c)
    else:
        cat_cols.append(c)

X_train_parts, X_val_parts, X_test_parts = [], [], []

# A. Temporal Features
for col in date_cols:
    for df, parts in [
        (train_df, X_train_parts),
        (val_df, X_val_parts),
        (test_df, X_test_parts),
    ]:
        dt = pd.to_datetime(df[col], errors="coerce")
        valid_dates = dt.dropna()
        min_date = (
            valid_dates.min() if not valid_dates.empty else pd.Timestamp("2020-01-01")
        )
        date_df = pd.DataFrame(
            {
                f"{col}_year": dt.dt.year.fillna(-1),
                f"{col}_month": dt.dt.month.fillna(-1),
                f"{col}_day": dt.dt.day.fillna(-1),
                f"{col}_dayofweek": dt.dt.dayofweek.fillna(-1),
                f"{col}_quarter": dt.dt.quarter.fillna(-1),
                f"{col}_days_since_start": (dt - min_date).dt.days.fillna(-1),
            },
            index=df.index,
        )
        parts.append(date_df)

# B. Text Features (RegEx Concentration Parsing, Multi-Unit & Word/Char TF-IDF SVD)
for col in text_cols:
    for df, parts in [
        (train_df, X_train_parts),
        (val_df, X_val_parts),
        (test_df, X_test_parts),
    ]:
        text_series = df[col].fillna("").astype(str)
        mg_vals = text_series.str.extract(r'(?i)\b(\d+(?:\.\d+)?)\s*mg\b', expand=False).astype(float).fillna(0.0)
        ml_vals = text_series.str.extract(r'(?i)\b(\d+(?:\.\d+)?)\s*ml\b', expand=False).astype(float).fillna(0.0)
        mcg_vals = text_series.str.extract(r'(?i)\b(\d+(?:\.\d+)?)\s*mcg\b', expand=False).astype(float).fillna(0.0)
        gm_vals = text_series.str.extract(r'(?i)\b(\d+(?:\.\d+)?)\s*(?:gm|gram)\b', expand=False).astype(float).fillna(0.0)
        iu_vals = text_series.str.extract(r'(?i)\b(\d+(?:\.\d+)?)\s*iu\b', expand=False).astype(float).fillna(0.0)
        pct_vals = text_series.str.extract(r'(?i)\b(\d+(?:\.\d+)?)\s*%', expand=False).astype(float).fillna(0.0)
        act_vals = text_series.str.extract(r'(?i)\b(\d+(?:\.\d+)?)\s*(?:actuation|act)\b', expand=False).astype(float).fillna(0.0)

        # Parse concentration ratio (e.g. mg/ml, mg/actuation, mcg/actuation)
        ratio_extract = text_series.str.extract(
            r'(?i)(\d+(?:\.\d+)?)\s*(?:mg|mcg|gm|iu)?\s*/\s*(\d+(?:\.\d+)?|\b)\s*(?:ml|actuation|act|gm|dose|tab|cap)?',
            expand=True
        )
        ratio_num = pd.to_numeric(ratio_extract[0], errors='coerce').fillna(0.0)
        ratio_den = pd.to_numeric(ratio_extract[1], errors='coerce').fillna(1.0)
        ratio_den = ratio_den.replace(0.0, 1.0)
        conc_ratio = np.log1p(np.maximum(0, ratio_num / ratio_den))

        txt_df = pd.DataFrame(
            {
                f"{col}_len": text_series.str.len(),
                f"{col}_word_count": text_series.apply(lambda s: len(s.split())),
                f"{col}_mg_val": mg_vals,
                f"{col}_ml_val": ml_vals,
                f"{col}_mcg_val": mcg_vals,
                f"{col}_gm_val": gm_vals,
                f"{col}_iu_val": iu_vals,
                f"{col}_pct_val": pct_vals,
                f"{col}_act_val": act_vals,
                f"{col}_conc_ratio": conc_ratio,
            },
            index=df.index,
        )
        parts.append(txt_df)

    # Word TF-IDF + TruncatedSVD
    tfidf_word = TfidfVectorizer(
        max_features=2500, ngram_range=(1, 2), token_pattern=r"(?u)\b\w+\b"
    )
    svd_word = TruncatedSVD(n_components=min(12, len(train_df)), random_state=42)

    tr_txt = train_df[col].fillna("").astype(str)
    va_txt = val_df[col].fillna("").astype(str)
    te_txt = test_df[col].fillna("").astype(str)

    tr_vec_word = svd_word.fit_transform(tfidf_word.fit_transform(tr_txt))
    va_vec_word = svd_word.transform(tfidf_word.transform(va_txt))
    te_vec_word = svd_word.transform(tfidf_word.transform(te_txt))

    svd_word_cols = [f"{col}_svd_{i}" for i in range(tr_vec_word.shape[1])]
    X_train_parts.append(pd.DataFrame(tr_vec_word, columns=svd_word_cols, index=train_df.index))
    X_val_parts.append(pd.DataFrame(va_vec_word, columns=svd_word_cols, index=val_df.index))
    X_test_parts.append(pd.DataFrame(te_vec_word, columns=svd_word_cols, index=test_df.index))

    # Character N-gram TF-IDF + TruncatedSVD (N-grams 3 to 5)
    tfidf_char = TfidfVectorizer(
        max_features=2500, analyzer='char', ngram_range=(3, 5)
    )
    svd_char = TruncatedSVD(n_components=min(12, len(train_df)), random_state=42)

    tr_vec_char = svd_char.fit_transform(tfidf_char.fit_transform(tr_txt))
    va_vec_char = svd_char.transform(tfidf_char.transform(va_txt))
    te_vec_char = svd_char.transform(tfidf_char.transform(te_txt))

    svd_char_cols = [f"{col}_char_svd_{i}" for i in range(tr_vec_char.shape[1])]
    X_train_parts.append(pd.DataFrame(tr_vec_char, columns=svd_char_cols, index=train_df.index))
    X_val_parts.append(pd.DataFrame(va_vec_char, columns=svd_char_cols, index=val_df.index))
    X_test_parts.append(pd.DataFrame(te_vec_char, columns=svd_char_cols, index=test_df.index))

# C. Categorical Features (Frequency & Smoothed Out-of-Fold Target Encodings)
for col in cat_cols:
    freq_map = (
        train_df[col]
        .fillna("MISSING")
        .astype(str)
        .value_counts(normalize=True)
        .to_dict()
    )

    global_target_mean = float(np.mean(y_train_log))
    m_weight = 10.0

    cat_train_str = train_df[col].fillna("MISSING").astype(str)
    cat_val_str = val_df[col].fillna("MISSING").astype(str)
    cat_test_str = test_df[col].fillna("MISSING").astype(str)

    # Full train smoothed target encoding for val and test
    stats = pd.DataFrame({'cat': cat_train_str, 'y': y_train_log}).groupby('cat').agg(
        count=('y', 'count'),
        mean=('y', 'mean')
    )
    smoothed_map = ((stats['count'] * stats['mean'] + m_weight * global_target_mean) / (stats['count'] + m_weight)).to_dict()

    # Out-of-fold smoothed target encoding for train
    oof_target_enc = np.full(len(train_df), global_target_mean, dtype=np.float32)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for tr_idx, val_idx in kf.split(train_df):
        tr_cat = cat_train_str.iloc[tr_idx]
        tr_y = y_train_log[tr_idx]
        val_cat = cat_train_str.iloc[val_idx]

        fold_stats = pd.DataFrame({'cat': tr_cat, 'y': tr_y}).groupby('cat').agg(
            count=('y', 'count'),
            mean=('y', 'mean')
        )
        fold_smoothed = ((fold_stats['count'] * fold_stats['mean'] + m_weight * global_target_mean) / (fold_stats['count'] + m_weight)).to_dict()
        oof_target_enc[val_idx] = val_cat.map(fold_smoothed).fillna(global_target_mean).values

    train_cat_df = pd.DataFrame({
        f"{col}_freq": cat_train_str.map(freq_map).fillna(0.0),
        f"{col}_target_enc": oof_target_enc,
    }, index=train_df.index)

    val_cat_df = pd.DataFrame({
        f"{col}_freq": cat_val_str.map(freq_map).fillna(0.0),
        f"{col}_target_enc": cat_val_str.map(smoothed_map).fillna(global_target_mean),
    }, index=val_df.index)

    test_cat_df = pd.DataFrame({
        f"{col}_freq": cat_test_str.map(freq_map).fillna(0.0),
        f"{col}_target_enc": cat_test_str.map(smoothed_map).fillna(global_target_mean),
    }, index=test_df.index)

    X_train_parts.append(train_cat_df)
    X_val_parts.append(val_cat_df)
    X_test_parts.append(test_cat_df)

# D. Numerical Features
for col in num_cols:
    med_val = train_df[col].median()
    med_val = 0.0 if np.isnan(med_val) else float(med_val)
    for df, parts in [
        (train_df, X_train_parts),
        (val_df, X_val_parts),
        (test_df, X_test_parts),
    ]:
        num_df = pd.DataFrame(
            {
                f"{col}_isna": df[col].isna().astype(int),
                col: df[col].fillna(med_val),
            },
            index=df.index,
        )
        parts.append(num_df)

# Concatenate & Scale Features
X_train = (
    pd.concat(X_train_parts, axis=1)
    if X_train_parts
    else pd.DataFrame(index=train_df.index)
)
X_val = (
    pd.concat(X_val_parts, axis=1) if X_val_parts else pd.DataFrame(index=val_df.index)
)
X_test = (
    pd.concat(X_test_parts, axis=1)
    if X_test_parts
    else pd.DataFrame(index=test_df.index)
)

X_train = X_train.loc[:, ~X_train.columns.duplicated()].replace([np.inf, -np.inf], np.nan).fillna(0.0)
X_val = X_val.loc[:, ~X_val.columns.duplicated()].replace([np.inf, -np.inf], np.nan).fillna(0.0)
X_test = X_test.loc[:, ~X_test.columns.duplicated()].replace([np.inf, -np.inf], np.nan).fillna(0.0)

scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)


# 4. PyTorch Architecture: Deep Tabular Gated ResNet
class TabularGatedBlock(nn.Module):
    def __init__(self, dim, dropout_rate=0.25):
        super().__init__()
        self.norm = nn.BatchNorm1d(dim)
        self.fc1 = nn.Linear(dim, dim * 2)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(dim * 2, dim)
        self.gate = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        res = x
        x_norm = self.norm(x)
        h = self.act(self.fc1(x_norm))
        h = self.fc2(h)
        g = torch.sigmoid(self.gate(x_norm))
        out = res + self.dropout(h * g)
        return out


class TabularGatedResNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_blocks=3, dropout_rate=0.25):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.SiLU()
        )
        self.blocks = nn.ModuleList(
            [TabularGatedBlock(hidden_dim, dropout_rate) for _ in range(num_blocks)]
        )
        self.head = nn.Sequential(
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        h = self.input_layer(x)
        for block in self.blocks:
            h = block(h)
        out = self.head(h)
        return out.squeeze(-1)


# Train PyTorch Deep Learning Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim = X_train_scaled.shape[1]
model_nn = TabularGatedResNet(
    input_dim=input_dim, hidden_dim=128, num_blocks=3, dropout_rate=0.25
).to(device)

criterion = nn.MSELoss()
optimizer = optim.AdamW(model_nn.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40)

X_tr_tensor = torch.tensor(X_train_scaled.values, dtype=torch.float32)
y_tr_tensor = torch.tensor(y_train_log, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val_scaled.values, dtype=torch.float32)
X_te_tensor = torch.tensor(X_test_scaled.values, dtype=torch.float32)

train_dataset = TensorDataset(X_tr_tensor, y_tr_tensor)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, drop_last=False)

best_val_loss = float("inf")
best_nn_weights = None

for epoch in range(1, 41):
    model_nn.train()
    for bx, by in train_loader:
        if bx.shape[0] <= 1:
            continue
        bx, by = bx.to(device), by.to(device)
        optimizer.zero_grad()
        preds = model_nn(bx)
        loss = criterion(preds, by)
        loss.backward()
        optimizer.step()
    scheduler.step()

    model_nn.eval()
    with torch.no_grad():
        val_preds_nn_log = model_nn(X_val_tensor.to(device)).cpu().numpy()
        val_preds_nn = np.nan_to_num(np.clip(np.expm1(val_preds_nn_log), 0, None), nan=float(np.nanmedian(y_train)))
        val_rmse = root_mean_squared_error(y_val, val_preds_nn)
        if val_rmse < best_val_loss:
            best_val_loss = val_rmse
            best_nn_weights = {
                k: v.cpu().clone() for k, v in model_nn.state_dict().items()
            }

if best_nn_weights is not None:
    model_nn.load_state_dict({k: v.to(device) for k, v in best_nn_weights.items()})

model_nn.eval()
with torch.no_grad():
    nn_val_preds = np.nan_to_num(np.clip(np.expm1(model_nn(X_val_tensor.to(device)).cpu().numpy()), 0, None), nan=float(np.nanmedian(y_train)))
    nn_test_preds = np.nan_to_num(np.clip(np.expm1(model_nn(X_te_tensor.to(device)).cpu().numpy()), 0, None), nan=float(np.nanmedian(y_train)))

# 5. Train LightGBM Regressor
lgb_model = lgb.LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.03,
    num_leaves=15,
    min_child_samples=10,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbosity=-1,
)
lgb_model.fit(
    X_train_scaled,
    y_train_log,
    eval_set=[(X_val_scaled, y_val_log)],
    callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
)
lgb_val_preds = np.nan_to_num(np.clip(np.expm1(lgb_model.predict(X_val_scaled)), 0, None), nan=float(np.nanmedian(y_train)))
lgb_test_preds = np.nan_to_num(np.clip(np.expm1(lgb_model.predict(X_test_scaled)), 0, None), nan=float(np.nanmedian(y_train)))

# 6. Train XGBoost Regressor
xgb_model = xgb.XGBRegressor(
    n_estimators=1000,
    learning_rate=0.03,
    max_depth=4,
    reg_lambda=1.0,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",
    random_state=42,
    early_stopping_rounds=50,
    eval_metric="rmse",
)
xgb_model.fit(X_train_scaled, y_train_log, eval_set=[(X_val_scaled, y_val_log)], verbose=False)
xgb_val_preds = np.nan_to_num(np.clip(np.expm1(xgb_model.predict(X_val_scaled)), 0, None), nan=float(np.nanmedian(y_train)))
xgb_test_preds = np.nan_to_num(np.clip(np.expm1(xgb_model.predict(X_test_scaled)), 0, None), nan=float(np.nanmedian(y_train)))

# 7. Train CatBoost Regressor
cb_model = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.03,
    depth=4,
    l2_leaf_reg=5.0,
    random_seed=42,
    early_stopping_rounds=50,
    verbose=0,
)
cb_model.fit(X_train_scaled, y_train_log, eval_set=(X_val_scaled, y_val_log))
cb_val_preds = np.nan_to_num(np.clip(np.expm1(cb_model.predict(X_val_scaled)), 0, None), nan=float(np.nanmedian(y_train)))
cb_test_preds = np.nan_to_num(np.clip(np.expm1(cb_model.predict(X_test_scaled)), 0, None), nan=float(np.nanmedian(y_train)))

# 8. Ensemble Optimization via SLSQP
val_matrix = np.column_stack([nn_val_preds, lgb_val_preds, xgb_val_preds, cb_val_preds])
test_matrix = np.column_stack(
    [nn_test_preds, lgb_test_preds, xgb_test_preds, cb_test_preds]
)


def blend_objective(weights):
    w = weights / (np.sum(weights) + 1e-12)
    pred = np.clip(val_matrix @ w, 0, None)
    return root_mean_squared_error(y_val, pred)


init_w = np.ones(4) / 4.0
bounds = [(0, 1)] * 4
res = minimize(blend_objective, init_w, bounds=bounds, method="SLSQP")
opt_weights = res.x / np.sum(res.x)

final_val_preds = np.clip(val_matrix @ opt_weights, 0, None)
final_test_preds = np.clip(test_matrix @ opt_weights, 0, None)

score = root_mean_squared_error(y_val, final_val_preds)

# 9. Format & Save Final Submission CSV
if "id" in test_raw.columns:
    ids = test_raw["id"]
else:
    ids = np.arange(len(final_test_preds))

submission_df = pd.DataFrame({"id": ids, "NADAC Rate": final_test_preds})
submission_df.to_csv("./submission/submission.csv", index=False)

# Final Line Output Required for Parser
print(f"Final Validation Score: {score}")
