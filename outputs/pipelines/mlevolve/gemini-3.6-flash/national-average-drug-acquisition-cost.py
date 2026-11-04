import os
import random
import re
import numpy as np
import pandas as pd
import scipy.optimize as opt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import KFold
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Ensure output directories exist
os.makedirs("./submission", exist_ok=True)
os.makedirs("./working", exist_ok=True)


# Deterministic Random Seed Setup
def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


seed_everything(42)

# --- 1. Load Input Data ---
train_df_raw = pd.read_parquet("./input/train.parquet")
test_df_raw = pd.read_parquet("./input/test.parquet")

target_col = "NADAC Per Unit"

# Ensure 0-based integer row IDs
if "id" not in train_df_raw.columns:
    train_df_raw["id"] = np.arange(len(train_df_raw))
if "id" not in test_df_raw.columns:
    test_df_raw["id"] = np.arange(len(test_df_raw))

# Identify column types
text_cols = [
    c
    for c in train_df_raw.columns
    if train_df_raw[c].dtype == "object" and c != target_col
]
date_cols = [
    c for c in train_df_raw.columns if "date" in c.lower() or "year" in c.lower()
]


# --- 2. Domain Regex & Text Feature Extraction ---
def extract_text_features(df, cols):
    feats = {}
    for col in cols:
        if col in df.columns:
            s = df[col].astype(str).fillna("")
            char_len = s.str.len()
            word_count = s.str.split().str.len()
            digit_count = s.apply(lambda x: sum(c.isdigit() for c in x))
            first_num = s.apply(
                lambda x: (
                    float(re.findall(r"[-+]?\d*\.\d+|\d+", x)[0])
                    if re.findall(r"[-+]?\d*\.\d+|\d+", x)
                    else 0.0
                )
            )

            feats[f"{col}_char_len"] = char_len
            feats[f"{col}_word_count"] = word_count
            feats[f"{col}_digit_count"] = digit_count
            feats[f"{col}_first_num"] = first_num

            # Numerical interaction ratios relative to text length and composition
            feats[f"{col}_num_char_ratio"] = first_num / (char_len + 1.0)
            feats[f"{col}_num_word_ratio"] = first_num / (word_count + 1.0)
            feats[f"{col}_digit_char_ratio"] = digit_count / (char_len + 1.0)

            feats[f"{col}_has_mg"] = s.str.contains("mg", case=False).astype(int)
            feats[f"{col}_has_ml"] = s.str.contains("ml", case=False).astype(int)
            feats[f"{col}_has_mcg"] = s.str.contains("mcg", case=False).astype(int)
            feats[f"{col}_has_pct"] = s.str.contains("%", case=False).astype(int)
            feats[f"{col}_has_cap"] = s.str.contains("cap", case=False).astype(int)
            feats[f"{col}_has_tab"] = s.str.contains("tab", case=False).astype(int)
            feats[f"{col}_has_soln"] = s.str.contains("sol", case=False).astype(int)
            feats[f"{col}_has_inj"] = s.str.contains("inj", case=False).astype(int)
    return pd.DataFrame(feats, index=df.index)


train_text_feats = extract_text_features(train_df_raw, text_cols)
test_text_feats = extract_text_features(test_df_raw, text_cols)

# --- 3. Text Vectorization via TF-IDF + Truncated SVD ---
combined_train_text = (
    train_df_raw[text_cols].fillna("").astype(str).agg(" ".join, axis=1)
    if text_cols
    else pd.Series([""] * len(train_df_raw))
)
combined_test_text = (
    test_df_raw[text_cols].fillna("").astype(str).agg(" ".join, axis=1)
    if text_cols
    else pd.Series([""] * len(test_df_raw))
)

tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 3), stop_words="english")
train_tfidf = tfidf.fit_transform(combined_train_text)
test_tfidf = tfidf.transform(combined_test_text)

svd_dim = 64
svd = TruncatedSVD(n_components=svd_dim, random_state=42)
train_svd = pd.DataFrame(
    svd.fit_transform(train_tfidf),
    columns=[f"svd_{i}" for i in range(svd_dim)],
    index=train_df_raw.index,
)
test_svd = pd.DataFrame(
    svd.transform(test_tfidf),
    columns=[f"svd_{i}" for i in range(svd_dim)],
    index=test_df_raw.index,
)


# --- 4. Temporal Features Handling ---
def parse_date_features(df, cols):
    d_df = pd.DataFrame(index=df.index)
    for col in cols:
        if col in df.columns:
            dt = pd.to_datetime(df[col], errors="coerce")
            d_df[f"{col}_year"] = dt.dt.year.fillna(-1)
            d_df[f"{col}_month"] = dt.dt.month.fillna(-1)
            d_df[f"{col}_day"] = dt.dt.day.fillna(-1)
            d_df[f"{col}_dayofweek"] = dt.dt.dayofweek.fillna(-1)
            d_df[f"{col}_quarter"] = dt.dt.quarter.fillna(-1)
    return d_df


train_date_feats = parse_date_features(train_df_raw, date_cols)
test_date_feats = parse_date_features(test_df_raw, date_cols)

# --- 5. Categorical Encoding (Frequency & Factorization) ---
cat_cols = [
    c
    for c in train_df_raw.columns
    if train_df_raw[c].dtype in ["object", "category"] and c not in date_cols
]
train_cat_feats = pd.DataFrame(index=train_df_raw.index)
test_cat_feats = pd.DataFrame(index=test_df_raw.index)

for col in cat_cols:
    freq_map = train_df_raw[col].value_counts(normalize=True).to_dict()
    train_cat_feats[f"{col}_freq"] = train_df_raw[col].map(freq_map).fillna(0.0)
    test_cat_feats[f"{col}_freq"] = test_df_raw[col].map(freq_map).fillna(0.0)

    codes, _ = pd.factorize(
        pd.concat([train_df_raw[col], test_df_raw[col]], axis=0).fillna("MISSING")
    )
    train_cat_feats[f"{col}_code"] = codes[: len(train_df_raw)]
    test_cat_feats[f"{col}_code"] = codes[len(train_df_raw) :]

# --- 6. Numerical Features & Out-of-Fold Target Encoding ---
num_cols = [
    c
    for c in train_df_raw.columns
    if train_df_raw[c].dtype in ["int64", "float64"]
    and c not in ["id", target_col]
    and c not in date_cols
]
train_num_feats = train_df_raw[num_cols].fillna(0.0)
test_num_feats = test_df_raw[num_cols].fillna(0.0)

train_target = train_df_raw[target_col].values
train_target_log = np.log1p(np.maximum(0, train_target))

# Create 5-Fold Cross Validation Split
kf = KFold(n_splits=5, shuffle=True, random_state=42)
train_folds = np.zeros(len(train_df_raw), dtype=int)
for fold, (train_idx, val_idx) in enumerate(kf.split(train_df_raw)):
    train_folds[val_idx] = fold

# Out-of-Fold Target Statistics with Smoothing
smoothing = 10.0
te_train_df = pd.DataFrame(index=train_df_raw.index)
te_test_df = pd.DataFrame(index=test_df_raw.index)

for col in cat_cols:
    te_train_df[f"{col}_te_mean"] = 0.0
    te_train_df[f"{col}_te_std"] = 0.0
    te_train_df[f"{col}_te_min"] = 0.0
    te_train_df[f"{col}_te_max"] = 0.0

    test_te_mean_acc = np.zeros(len(test_df_raw))
    test_te_std_acc = np.zeros(len(test_df_raw))
    test_te_min_acc = np.zeros(len(test_df_raw))
    test_te_max_acc = np.zeros(len(test_df_raw))

    for fold in range(5):
        tr_mask = train_folds != fold
        val_mask = train_folds == fold

        tr_cat = train_df_raw.loc[tr_mask, col].fillna("MISSING")
        tr_y = train_target_log[tr_mask]

        val_cat = train_df_raw.loc[val_mask, col].fillna("MISSING")
        test_cat = test_df_raw[col].fillna("MISSING")

        global_mean = tr_y.mean()
        global_std = tr_y.std() if len(tr_y) > 1 else 0.0
        global_min = tr_y.min()
        global_max = tr_y.max()

        df_tr = pd.DataFrame({"cat": tr_cat, "y": tr_y})
        grp = df_tr.groupby("cat")["y"].agg(["count", "mean", "std", "min", "max"])

        grp["smooth_mean"] = (grp["count"] * grp["mean"] + smoothing * global_mean) / (
            grp["count"] + smoothing
        )
        grp["std"] = grp["std"].fillna(global_std)
        grp["min"] = grp["min"].fillna(global_min)
        grp["max"] = grp["max"].fillna(global_max)

        mean_map = grp["smooth_mean"].to_dict()
        std_map = grp["std"].to_dict()
        min_map = grp["min"].to_dict()
        max_map = grp["max"].to_dict()

        te_train_df.loc[val_mask, f"{col}_te_mean"] = val_cat.map(mean_map).fillna(
            global_mean
        )
        te_train_df.loc[val_mask, f"{col}_te_std"] = val_cat.map(std_map).fillna(
            global_std
        )
        te_train_df.loc[val_mask, f"{col}_te_min"] = val_cat.map(min_map).fillna(
            global_min
        )
        te_train_df.loc[val_mask, f"{col}_te_max"] = val_cat.map(max_map).fillna(
            global_max
        )

        test_te_mean_acc += test_cat.map(mean_map).fillna(global_mean).values / 5.0
        test_te_std_acc += test_cat.map(std_map).fillna(global_std).values / 5.0
        test_te_min_acc += test_cat.map(min_map).fillna(global_min).values / 5.0
        test_te_max_acc += test_cat.map(max_map).fillna(global_max).values / 5.0

    te_test_df[f"{col}_te_mean"] = test_te_mean_acc
    te_test_df[f"{col}_te_std"] = test_te_std_acc
    te_test_df[f"{col}_te_min"] = test_te_min_acc
    te_test_df[f"{col}_te_max"] = test_te_max_acc

# Compute group statistics for key categorical features strictly on train
grp_train_dict = {}
grp_test_dict = {}

for cat_col in cat_cols:
    for num_col in num_cols:
        grp = train_df_raw.groupby(cat_col)[num_col]
        mean_map = grp.mean().to_dict()
        std_map = grp.std().fillna(0.0).to_dict()
        min_map = grp.min().to_dict()
        max_map = grp.max().to_dict()

        global_mean = train_df_raw[num_col].mean()
        global_std = train_df_raw[num_col].std()
        global_min = train_df_raw[num_col].min()
        global_max = train_df_raw[num_col].max()

        grp_train_dict[f"{cat_col}_{num_col}_grp_mean"] = train_df_raw[cat_col].map(mean_map).fillna(global_mean).values
        grp_train_dict[f"{cat_col}_{num_col}_grp_std"] = train_df_raw[cat_col].map(std_map).fillna(global_std).values
        grp_train_dict[f"{cat_col}_{num_col}_grp_min"] = train_df_raw[cat_col].map(min_map).fillna(global_min).values
        grp_train_dict[f"{cat_col}_{num_col}_grp_max"] = train_df_raw[cat_col].map(max_map).fillna(global_max).values

        grp_test_dict[f"{cat_col}_{num_col}_grp_mean"] = test_df_raw[cat_col].map(mean_map).fillna(global_mean).values
        grp_test_dict[f"{cat_col}_{num_col}_grp_std"] = test_df_raw[cat_col].map(std_map).fillna(global_std).values
        grp_test_dict[f"{cat_col}_{num_col}_grp_min"] = test_df_raw[cat_col].map(min_map).fillna(global_min).values
        grp_test_dict[f"{cat_col}_{num_col}_grp_max"] = test_df_raw[cat_col].map(max_map).fillna(global_max).values

group_stat_train = pd.DataFrame(grp_train_dict, index=train_df_raw.index)
group_stat_test = pd.DataFrame(grp_test_dict, index=test_df_raw.index)

# Assemble Feature DataFrames
X_train = pd.concat(
    [
        train_num_feats,
        train_text_feats,
        train_svd,
        train_date_feats,
        train_cat_feats,
        te_train_df,
        group_stat_train,
    ],
    axis=1,
)
X_test = pd.concat(
    [
        test_num_feats,
        test_text_feats,
        test_svd,
        test_date_feats,
        test_cat_feats,
        te_test_df,
        group_stat_test,
    ],
    axis=1,
)

X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0.0)
X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(0.0)

train_df = X_train.copy()
train_df["id"] = train_df_raw["id"].values
train_df["fold"] = train_folds
train_df[target_col] = train_target
train_df["target_log"] = train_target_log

test_df = X_test.copy()
test_df["id"] = test_df_raw["id"].values

ignore_cols = ["id", "fold", target_col, "target_log"]
feature_cols = [c for c in train_df.columns if c not in ignore_cols]


# --- 7. PyTorch Architecture Definition ---
class GatedResidualBlock(nn.Module):
    def __init__(self, hidden_dim, dropout_rate=0.1):
        super().__init__()
        self.fc_gate = nn.Linear(hidden_dim, hidden_dim * 2)
        self.fc_out = nn.Linear(hidden_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.act = nn.GELU()

    def forward(self, x):
        residual = x
        gated = self.fc_gate(x)
        val, gate = gated.chunk(2, dim=-1)
        x = self.act(val) * torch.sigmoid(gate)
        x = self.dropout(self.fc_out(x))
        return self.layer_norm(residual + x)


class TabularGatedResNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_blocks=2, dropout_rate=0.1):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()
        )
        self.blocks = nn.ModuleList(
            [GatedResidualBlock(hidden_dim, dropout_rate) for _ in range(num_blocks)]
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.GELU(),
            nn.Dropout(dropout_rate / 2.0),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x).squeeze(-1)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 8. Multi-Model K-Fold Cross-Validation ---
num_folds = train_df["fold"].nunique()
n_samples = len(train_df)
n_test = len(test_df)

oof_lgb = np.zeros(n_samples)
oof_xgb = np.zeros(n_samples)
oof_cat = np.zeros(n_samples)
oof_nn = np.zeros(n_samples)

test_lgb = np.zeros(n_test)
test_xgb = np.zeros(n_test)
test_cat = np.zeros(n_test)
test_nn = np.zeros(n_test)

X_test_raw = test_df[feature_cols].values.astype(np.float32)

for fold in range(num_folds):
    train_mask = train_df["fold"] != fold
    val_mask = train_df["fold"] == fold

    X_train_f = train_df.loc[train_mask, feature_cols].values.astype(np.float32)
    y_train_log = train_df.loc[train_mask, "target_log"].values.astype(np.float32)

    X_val_f = train_df.loc[val_mask, feature_cols].values.astype(np.float32)
    y_val_log = train_df.loc[val_mask, "target_log"].values.astype(np.float32)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_f)
    X_val_scaled = scaler.transform(X_val_f)
    X_test_scaled = scaler.transform(X_test_raw)

    # 1) LightGBM Model
    lgb_model = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.04,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        n_jobs=-1,
    )
    lgb_model.fit(
        X_train_f,
        y_train_log,
        eval_set=[(X_val_f, y_val_log)],
        callbacks=[lgb.early_stopping(stopping_rounds=40, verbose=False)],
    )
    oof_lgb[val_mask] = lgb_model.predict(X_val_f)
    test_lgb += lgb_model.predict(X_test_raw) / num_folds

    # 2) XGBoost Model
    xgb_model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.04,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        early_stopping_rounds=40,
        random_state=42 + fold,
        n_jobs=-1,
    )
    xgb_model.fit(
        X_train_f,
        y_train_log,
        eval_set=[(X_val_f, y_val_log)],
        verbose=False,
    )
    oof_xgb[val_mask] = xgb_model.predict(X_val_f)
    test_xgb += xgb_model.predict(X_test_raw) / num_folds

    # 3) CatBoost Model
    cat_model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.04,
        depth=6,
        random_seed=42 + fold,
        early_stopping_rounds=40,
        verbose=0,
    )
    cat_model.fit(X_train_f, y_train_log, eval_set=(X_val_f, y_val_log))
    oof_cat[val_mask] = cat_model.predict(X_val_f)
    test_cat += cat_model.predict(X_test_raw) / num_folds

    # 4) PyTorch Tabular Gated ResNet
    train_dataset = TensorDataset(
        torch.tensor(X_train_scaled), torch.tensor(y_train_log)
    )
    val_dataset = TensorDataset(torch.tensor(X_val_scaled), torch.tensor(y_val_log))

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)

    nn_model = TabularGatedResNet(input_dim=len(feature_cols)).to(device)
    optimizer = optim.AdamW(nn_model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_nn_preds = None
    best_model_state = None

    for epoch in range(20):
        nn_model.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = nn_model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
        scheduler.step()

        nn_model.eval()
        val_loss = 0.0
        val_preds_list = []
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                out = nn_model(bx)
                val_loss += criterion(out, by).item() * len(bx)
                val_preds_list.append(out.cpu().numpy())

        val_loss /= len(X_val_scaled)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_nn_preds = np.concatenate(val_preds_list)
            best_model_state = {k: v.cpu().clone() for k, v in nn_model.state_dict().items()}

    oof_nn[val_mask] = best_nn_preds

    if best_model_state is not None:
        nn_model.load_state_dict(best_model_state)
    nn_model.eval()
    test_tensor = torch.tensor(X_test_scaled).to(device)
    with torch.no_grad():
        test_nn += nn_model(test_tensor).cpu().numpy() / num_folds

    val_rmse_lgb = np.sqrt(mean_squared_error(y_val_log, oof_lgb[val_mask]))
    print(f"Fold {fold+1}/{num_folds} - LGB Log RMSE: {val_rmse_lgb:.4f}")

# --- 9. Reconstruct Original Scale & Blending Optimization ---
y_true_orig = train_df[target_col].values

oof_orig_lgb = np.expm1(np.maximum(0, oof_lgb))
oof_orig_xgb = np.expm1(np.maximum(0, oof_xgb))
oof_orig_cat = np.expm1(np.maximum(0, oof_cat))
oof_orig_nn = np.expm1(np.maximum(0, oof_nn))

test_orig_lgb = np.expm1(np.maximum(0, test_lgb))
test_orig_xgb = np.expm1(np.maximum(0, test_xgb))
test_orig_cat = np.expm1(np.maximum(0, test_cat))
test_orig_nn = np.expm1(np.maximum(0, test_nn))


def ensemble_loss(params):
    w1, w2, w3, w4, alpha, beta = params
    base_pred = (
        w1 * oof_orig_lgb
        + w2 * oof_orig_xgb
        + w3 * oof_orig_cat
        + w4 * oof_orig_nn
    )
    pred = alpha * base_pred + beta
    return np.sqrt(mean_squared_error(y_true_orig, pred))


init_params = [0.25, 0.25, 0.25, 0.25, 1.0, 0.0]
bounds = [
    (0.0, 1.0),
    (0.0, 1.0),
    (0.0, 1.0),
    (0.0, 1.0),
    (0.01, 10.0),
    (-500.0, 500.0),
]
constraints = {"type": "eq", "fun": lambda p: 1.0 - sum(p[:4])}

opt_res = opt.minimize(
    ensemble_loss,
    init_params,
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
)
w1, w2, w3, w4, alpha, beta = opt_res.x

final_oof_preds = alpha * (
    w1 * oof_orig_lgb + w2 * oof_orig_xgb + w3 * oof_orig_cat + w4 * oof_orig_nn
) + beta
final_score = np.sqrt(mean_squared_error(y_true_orig, final_oof_preds))

# --- 10. Generate Final Submission File ---
final_test_preds = alpha * (
    w1 * test_orig_lgb + w2 * test_orig_xgb + w3 * test_orig_cat + w4 * test_orig_nn
) + beta
final_test_preds = np.maximum(0.0, final_test_preds)

submission_df = pd.DataFrame(
    {"id": test_df["id"].astype(int).values, "NADAC Per Unit": final_test_preds}
)
submission_df.to_csv("./submission/submission.csv", index=False)

print(f"Final Validation Score: {final_score}")