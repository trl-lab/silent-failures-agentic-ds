import os
import warnings

import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb

warnings.filterwarnings("ignore")

os.makedirs("./submission", exist_ok=True)
os.makedirs("./working", exist_ok=True)

# 1. Load Data
train_path = "./input/train.parquet"
test_path = "./input/test.parquet"

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = "Gross Issued Amount (US$)"
assert target_col in train_df.columns, f"Target column {target_col} not found!"

# Save raw target and log-transformed target
y_train_raw = train_df[target_col].values
y_train_log = np.log1p(np.maximum(0, y_train_raw))

# Identify test set IDs
if "id" in test_df.columns:
    test_ids = test_df["id"].values
elif "ID" in test_df.columns:
    test_ids = test_df["ID"].values
else:
    test_ids = np.arange(len(test_df))

train_len = len(train_df)
df_all = pd.concat(
    [train_df.drop(columns=[target_col], errors="ignore"), test_df],
    axis=0,
    ignore_index=True,
)

# Column Categorizations
num_cols = []
cat_cols = []
text_cols = []
date_cols = []

for col in df_all.columns:
    if col.lower() in ["id", "index"]:
        continue

    if pd.api.types.is_datetime64_any_dtype(df_all[col]):
        date_cols.append(col)
        continue

    if any(kw in col.lower() for kw in ["date", "year", "dt"]):
        try:
            parsed = pd.to_datetime(df_all[col], errors="coerce")
            if parsed.notna().sum() > 0.5 * len(df_all):
                df_all[col] = parsed
                date_cols.append(col)
                continue
        except Exception:
            pass

    if pd.api.types.is_numeric_dtype(df_all[col]):
        num_cols.append(col)
    else:
        str_vals = df_all[col].astype(str)
        avg_len = str_vals.str.len().mean()
        n_unique = str_vals.nunique()
        if avg_len > 30 or (n_unique > 100 and avg_len > 15):
            text_cols.append(col)
        else:
            cat_cols.append(col)

# Datetime Features
for dcol in date_cols:
    dt_series = pd.to_datetime(df_all[dcol], errors="coerce")
    df_all[f"{dcol}_year"] = dt_series.dt.year
    df_all[f"{dcol}_month"] = dt_series.dt.month
    df_all[f"{dcol}_quarter"] = dt_series.dt.quarter
    df_all[f"{dcol}_day"] = dt_series.dt.day
    df_all[f"{dcol}_dayofweek"] = dt_series.dt.dayofweek

# Categorical & Frequency Encoding
for ccol in cat_cols:
    df_all[ccol] = df_all[ccol].fillna("Missing").astype(str)
    freq = df_all.iloc[:train_len][ccol].value_counts(normalize=True)
    df_all[f"{ccol}_freq"] = df_all[ccol].map(freq).fillna(0)

# Categorical Pairwise Interaction Features
top_cats = [c for c in cat_cols if df_all[c].nunique() < 200][:3]
new_cat_cols = []
for i in range(len(top_cats)):
    for j in range(i + 1, len(top_cats)):
        c1, c2 = top_cats[i], top_cats[j]
        inter_col = f"{c1}_X_{c2}"
        df_all[inter_col] = df_all[c1].astype(str) + "_" + df_all[c2].astype(str)
        freq_inter = df_all.iloc[:train_len][inter_col].value_counts(normalize=True)
        df_all[f"{inter_col}_freq"] = df_all[inter_col].map(freq_inter).fillna(0)
        new_cat_cols.append(inter_col)

cat_cols.extend(new_cat_cols)

# Text Feature Extraction (TF-IDF + SVD)
if text_cols:
    df_all["combined_text"] = ""
    for tcol in text_cols:
        df_all[tcol] = df_all[tcol].fillna("")
        df_all[f"{tcol}_len"] = df_all[tcol].str.len()
        df_all[f"{tcol}_word_count"] = df_all[tcol].apply(lambda x: len(x.split()))
        df_all["combined_text"] = (
            df_all["combined_text"] + " " + df_all[tcol].astype(str)
        )

    combined_text = df_all["combined_text"].tolist()
    train_text = combined_text[:train_len]

    tfidf = TfidfVectorizer(
        max_features=8000,
        ngram_range=(1, 2),
        stop_words="english",
        sublinear_tf=True,
    )
    tfidf.fit(train_text)
    text_tfidf = tfidf.transform(combined_text)

    svd = TruncatedSVD(n_components=20, random_state=42)
    svd.fit(text_tfidf[:train_len])
    text_svd = svd.transform(text_tfidf)

    for idx in range(20):
        df_all[f"tfidf_svd_{idx}"] = text_svd[:, idx]

    df_all.drop(columns=["combined_text"], inplace=True, errors="ignore")

# Numerical Transformations
for ncol in num_cols:
    med = df_all.iloc[:train_len][ncol].median()
    df_all[ncol] = df_all[ncol].fillna(med)
    if (df_all[ncol] >= 0).all():
        df_all[f"{ncol}_log1p"] = np.log1p(df_all[ncol])

# Split back to Train & Test
df_train = df_all.iloc[:train_len].copy()
df_test = df_all.iloc[train_len:].copy()

# Out-of-fold Smoothed Target Encoding
kf = KFold(n_splits=5, shuffle=True, random_state=42)

te_cols = [c for c in cat_cols if df_all[c].nunique() > 2]
for col in te_cols:
    te_train = np.zeros(train_len)
    global_mean = y_train_log.mean()

    for tr_idx, val_idx in kf.split(df_train):
        tr_cat = df_train[col].iloc[tr_idx]
        tr_y = pd.Series(y_train_log[tr_idx], index=tr_cat.index)
        val_cat = df_train[col].iloc[val_idx]

        counts = tr_cat.value_counts()
        means = tr_y.groupby(tr_cat).mean()
        smooth_means = (counts * means + 10 * global_mean) / (counts + 10)
        te_train[val_idx] = val_cat.map(smooth_means).fillna(global_mean).values

    df_train[f"{col}_te"] = te_train

    counts_full = df_train[col].value_counts()
    y_full_series = pd.Series(y_train_log, index=df_train.index)
    means_full = y_full_series.groupby(df_train[col]).mean()
    smooth_means_full = (counts_full * means_full + 10 * global_mean) / (
        counts_full + 10
    )
    df_test[f"{col}_te"] = (
        df_test[col].map(smooth_means_full).fillna(global_mean).values
    )

# Encode Categoricals to Integer Codes consistently
for col in cat_cols:
    cat_type = pd.CategoricalDtype(categories=df_all[col].unique(), ordered=False)
    df_train[col] = df_train[col].astype(cat_type).cat.codes
    df_test[col] = df_test[col].astype(cat_type).cat.codes

df_train.drop(columns=date_cols, inplace=True, errors="ignore")
df_test.drop(columns=date_cols, inplace=True, errors="ignore")

X_train = df_train.select_dtypes(include=[np.number])
X_test = df_test[X_train.columns]
X_train = X_train.fillna(0)
X_test = X_test.fillna(0)

input_dim = X_train.shape[1]
test_len = len(X_test)


# 2. PyTorch TabularResNet Architecture
class ResidualBlock(nn.Module):

    def __init__(self, dim: int, dropout: float = 0.15):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.act = nn.SiLU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.drop(self.act(self.bn1(self.fc1(x))))
        out = self.bn2(self.fc2(out))
        return self.act(out + residual)


class TabularResNet(nn.Module):

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_blocks: int = 3,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim, dropout) for _ in range(num_blocks)]
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x).squeeze(-1)


# 3. Ensemble Setup & K-Fold Cross-Validation
model_names = [
    "lgb_mse",
    "lgb_huber",
    "xgb",
    "catboost",
    "extratrees",
    "pytorch_resnet",
]

oof_log_preds = {m: np.zeros(train_len) for m in model_names}
test_log_preds = {m: np.zeros(test_len) for m in model_names}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train)):
    X_tr, y_tr = X_train.iloc[tr_idx], y_train_log[tr_idx]
    X_va, y_va = X_train.iloc[val_idx], y_train_log[val_idx]

    # Model 1: LightGBM MSE
    m_lgb_mse = lgb.LGBMRegressor(
        objective="regression",
        metric="rmse",
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=31,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=1,
        min_child_samples=15,
        random_state=42 + fold,
        verbose=-1,
    )
    m_lgb_mse.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
    )
    oof_log_preds["lgb_mse"][val_idx] = m_lgb_mse.predict(X_va)
    test_log_preds["lgb_mse"] += m_lgb_mse.predict(X_test) / 5.0

    # Model 2: LightGBM Huber
    m_lgb_huber = lgb.LGBMRegressor(
        objective="huber",
        alpha=0.9,
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=31,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=1,
        min_child_samples=15,
        random_state=42 + fold,
        verbose=-1,
    )
    m_lgb_huber.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
    )
    oof_log_preds["lgb_huber"][val_idx] = m_lgb_huber.predict(X_va)
    test_log_preds["lgb_huber"] += m_lgb_huber.predict(X_test) / 5.0

    # Model 3: XGBoost
    m_xgb = xgb.XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42 + fold,
        early_stopping_rounds=50,
        n_jobs=-1,
    )
    m_xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    oof_log_preds["xgb"][val_idx] = m_xgb.predict(X_va)
    test_log_preds["xgb"] += m_xgb.predict(X_test) / 5.0

    # Model 4: CatBoost
    m_cb = cb.CatBoostRegressor(
        loss_function="RMSE",
        eval_metric="RMSE",
        iterations=1000,
        learning_rate=0.04,
        depth=6,
        l2_leaf_reg=3.0,
        random_seed=42 + fold,
        verbose=False,
    )
    m_cb.fit(X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=50, verbose=False)
    oof_log_preds["catboost"][val_idx] = m_cb.predict(X_va)
    test_log_preds["catboost"] += m_cb.predict(X_test) / 5.0

    # Model 5: ExtraTrees
    m_et = ExtraTreesRegressor(
        n_estimators=300,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42 + fold,
        n_jobs=-1,
    )
    m_et.fit(X_tr, y_tr)
    oof_log_preds["extratrees"][val_idx] = m_et.predict(X_va)
    test_log_preds["extratrees"] += m_et.predict(X_test) / 5.0

    # Model 6: PyTorch TabularResNet
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_va_sc = scaler.transform(X_va)
    X_test_sc = scaler.transform(X_test)

    train_ds = TensorDataset(
        torch.tensor(X_tr_sc, dtype=torch.float32),
        torch.tensor(y_tr, dtype=torch.float32),
    )
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)

    net = TabularResNet(
        input_dim=input_dim, hidden_dim=256, num_blocks=3, dropout=0.15
    ).to(device)
    criterion = nn.SmoothL1Loss(beta=1.0)
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)

    best_val_loss = float("inf")
    best_weights = None

    for epoch in range(40):
        net.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = net(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()

        net.eval()
        with torch.no_grad():
            va_x = torch.tensor(X_va_sc, dtype=torch.float32).to(device)
            va_y = torch.tensor(y_va, dtype=torch.float32).to(device)
            va_out = net(va_x)
            va_loss = criterion(va_out, va_y).item()
            if va_loss < best_val_loss:
                best_val_loss = va_loss
                best_weights = {k: v.cpu().clone() for k, v in net.state_dict().items()}

    net.load_state_dict(best_weights)
    net.to(device)
    net.eval()
    with torch.no_grad():
        va_tensor = torch.tensor(X_va_sc, dtype=torch.float32).to(device)
        test_tensor = torch.tensor(X_test_sc, dtype=torch.float32).to(device)
        oof_log_preds["pytorch_resnet"][val_idx] = net(va_tensor).cpu().numpy()
        test_log_preds["pytorch_resnet"] += net(test_tensor).cpu().numpy() / 5.0

# 4. Inverse Transformation to Raw Dollar Scale
oof_raw_preds = {m: np.expm1(np.maximum(0, oof_log_preds[m])) for m in model_names}
test_raw_preds = {m: np.expm1(np.maximum(0, test_log_preds[m])) for m in model_names}


# 5. SLSQP Constrained Optimization for Ensemble Blending
def objective(weights):
    blend_oof = np.zeros(train_len)
    for idx, m in enumerate(model_names):
        blend_oof += weights[idx] * oof_raw_preds[m]
    return np.sqrt(mean_squared_error(y_train_raw, blend_oof))


init_weights = np.ones(len(model_names)) / len(model_names)
bounds = [(0.0, 1.0) for _ in model_names]
constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

res = minimize(
    objective,
    init_weights,
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
)
opt_weights = res.x

# 6. Final Blended Predictions
final_oof = np.zeros(train_len)
final_test = np.zeros(test_len)

for idx, m in enumerate(model_names):
    final_oof += opt_weights[idx] * oof_raw_preds[m]
    final_test += opt_weights[idx] * test_raw_preds[m]

final_oof = np.maximum(0, final_oof)
final_test = np.maximum(0, final_test)

val_rmse = np.sqrt(mean_squared_error(y_train_raw, final_oof))

# 7. Save Submission
submission = pd.DataFrame({"id": test_ids, "Gross Issued Amount (US$)": final_test})
submission.to_csv("./submission/submission.csv", index=False)

print(f"Final Validation Score: {val_rmse}")