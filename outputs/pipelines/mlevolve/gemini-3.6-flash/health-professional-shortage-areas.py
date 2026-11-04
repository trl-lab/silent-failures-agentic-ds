import json
import os
import random
import warnings
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb

try:
    from catboost import CatBoostRegressor
except ImportError:
    CatBoostRegressor = None

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# 1. Environment & Reproducibility Setup
# -----------------------------------------------------------------------------
SEED = 42


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


seed_everything(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_dir = "./input"
working_dir = "./working"
sub_dir = "./submission"

os.makedirs(working_dir, exist_ok=True)
os.makedirs(sub_dir, exist_ok=True)

# -----------------------------------------------------------------------------
# 2. Data Loading & Feature Engineering Pipeline
# -----------------------------------------------------------------------------
train_df = pd.read_parquet(os.path.join(input_dir, "train.parquet"))
test_df = pd.read_parquet(os.path.join(input_dir, "test.parquet"))

target_col = "HPSA Score"

if "id" in test_df.columns:
    test_ids = test_df["id"].values
elif "ID" in test_df.columns:
    test_ids = test_df["ID"].values
else:
    test_ids = np.arange(len(test_df))

y = train_df[target_col].values.astype(np.float32)
y_min, y_max = float(y.min()), float(y.max())

drop_cols = [target_col, "id", "ID"]
X_train_raw = train_df.drop(
    columns=[c for c in drop_cols if c in train_df.columns]
).copy()
X_test_raw = test_df.drop(columns=[c for c in drop_cols if c in test_df.columns]).copy()

n_train = len(X_train_raw)
n_test = len(X_test_raw)

combined_df = pd.concat([X_train_raw, X_test_raw], axis=0, ignore_index=True)

# Date/Time Parsing and Feature Extraction
date_cols = [c for c in combined_df.columns if "date" in c.lower() or "dt" in c.lower()]
for col in date_cols:
    try:
        dt_series = pd.to_datetime(combined_df[col], errors="coerce")
        if dt_series.notnull().sum() > 0.1 * len(combined_df):
            combined_df[f"{col}_year"] = dt_series.dt.year.astype("float32")
            combined_df[f"{col}_month"] = dt_series.dt.month.astype("float32")
            combined_df[f"{col}_day"] = dt_series.dt.day.astype("float32")
            combined_df[f"{col}_dayofweek"] = dt_series.dt.dayofweek.astype("float32")

            max_dt = dt_series.max()
            combined_df[f"{col}_days_diff"] = (max_dt - dt_series).dt.days.astype(
                "float32"
            )
            combined_df.drop(columns=[col], inplace=True)
    except Exception:
        pass

# Dynamic Column Categorization
num_cols = []
cat_cols = []
text_cols = []

for c in combined_df.columns:
    if pd.api.types.is_numeric_dtype(combined_df[c]):
        num_cols.append(c)
    else:
        nunique = combined_df[c].nunique(dropna=True)
        avg_len = combined_df[c].astype(str).str.len().mean()
        if nunique > 100 and avg_len > 30:
            text_cols.append(c)
        else:
            cat_cols.append(c)

# Numerical Feature Transformations
for col in num_cols:
    missing_pct = combined_df[col].isna().mean()
    if missing_pct > 0.02:
        combined_df[f"{col}_isna"] = combined_df[col].isna().astype("float32")

    skewness = combined_df[col].skew()
    if abs(skewness) > 1.5:
        min_val = combined_df[col].min()
        shift = abs(min_val) if min_val < 0 else 0
        combined_df[f"{col}_log1p"] = np.log1p(combined_df[col] + shift + 1.0).astype(
            "float32"
        )

# Domain-Specific Interaction Ratios & Numerical Interactions
if len(num_cols) >= 2:
    top_num = (
        combined_df[num_cols].var().sort_values(ascending=False).index[:6].tolist()
    )
    for i in range(len(top_num)):
        for j in range(i + 1, len(top_num)):
            c1, c2 = top_num[i], top_num[j]
            combined_df[f"{c1}_ratio_{c2}"] = (
                combined_df[c1] / (combined_df[c2].abs() + 1e-5)
            ).astype("float32")
            combined_df[f"{c1}_diff_{c2}"] = (combined_df[c1] - combined_df[c2]).astype(
                "float32"
            )
            combined_df[f"{c1}_mult_{c2}"] = (combined_df[c1] * combined_df[c2]).astype(
                "float32"
            )

# Categorical Encodings & Multi-Granularity Group Aggregations
for col in cat_cols:
    freq = combined_df[col].value_counts(normalize=True, dropna=False)
    combined_df[f"{col}_freq"] = combined_df[col].map(freq).fillna(0).astype("float32")
    combined_df[col] = combined_df[col].astype("category").cat.codes.astype("int32")

if len(cat_cols) > 0 and len(num_cols) > 0:
    major_cats = cat_cols
    key_nums = num_cols[:6]
    train_part = combined_df.iloc[:n_train]
    for cat in major_cats:
        for num in key_nums:
            stats = train_part.groupby(cat)[num].agg(["mean", "median", "std", "min", "max"])
            for stat_name in ["mean", "median", "std", "min", "max"]:
                feat_name = f"agg_{cat}_{num}_{stat_name}"
                mapped = combined_df[cat].map(stats[stat_name]).astype("float32")
                combined_df[feat_name] = mapped.fillna(0) if stat_name == "std" else mapped

# Text Embeddings (TF-IDF + Truncated SVD fitted on train only)
for t_col in text_cols:
    train_text = combined_df.iloc[:n_train][t_col].fillna("").astype(str)
    full_text = combined_df[t_col].fillna("").astype(str)
    tfidf = TfidfVectorizer(max_features=100, stop_words="english")
    tfidf.fit(train_text)
    tfidf_mat = tfidf.transform(full_text)
    svd = TruncatedSVD(n_components=4, random_state=SEED)
    svd.fit(tfidf.transform(train_text))
    svd_feats = svd.transform(tfidf_mat)

    for k in range(4):
        combined_df[f"{t_col}_svd_{k}"] = svd_feats[:, k].astype("float32")
    combined_df.drop(columns=[t_col], inplace=True)

# Split Processed Features Back to Train and Test
X_train = combined_df.iloc[:n_train].copy().reset_index(drop=True)
X_test = combined_df.iloc[n_train:].copy().reset_index(drop=True)

# Robust Out-of-Fold Target Encoding with Multi-Scale Smoothing
kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
global_mean = y.mean()

target_enc_cols = cat_cols
for col in target_enc_cols:
    for m_smoothing in [5.0, 15.0]:
        te_train = np.zeros(n_train, dtype=np.float32)

        for train_idx, val_idx in kf.split(X_train):
            X_tr_fold = X_train.iloc[train_idx]
            y_tr_fold = y[train_idx]
            df_tr = pd.DataFrame({"cat": X_tr_fold[col], "target": y_tr_fold})
            stats = df_tr.groupby("cat")["target"].agg(["count", "mean"])
            smooth = (stats["count"] * stats["mean"] + m_smoothing * global_mean) / (
                stats["count"] + m_smoothing
            )
            te_train[val_idx] = (
                X_train.iloc[val_idx][col].map(smooth).fillna(global_mean).values
            )

        X_train[f"{col}_te_{int(m_smoothing)}"] = te_train

        df_full = pd.DataFrame({"cat": X_train[col], "target": y})
        stats_full = df_full.groupby("cat")["target"].agg(["count", "mean"])
        smooth_full = (
            stats_full["count"] * stats_full["mean"] + m_smoothing * global_mean
        ) / (stats_full["count"] + m_smoothing)
        X_test[f"{col}_te_{int(m_smoothing)}"] = (
            X_test[col].map(smooth_full).fillna(global_mean).astype("float32")
        )

# Fill any remaining NaNs
X_train = X_train.fillna(0)
X_test = X_test.fillna(0)

# Save processed datasets
X_train.to_parquet(os.path.join(working_dir, "X_train.parquet"))
X_test.to_parquet(os.path.join(working_dir, "X_test.parquet"))
pd.DataFrame({"y": y}).to_parquet(os.path.join(working_dir, "y_train.parquet"))

n_features = X_train.shape[1]


# -----------------------------------------------------------------------------
# 3. Model Architecture Definitions
# -----------------------------------------------------------------------------
class TabularResBlock(nn.Module):

    def __init__(self, in_features: int, dropout_rate: float = 0.2):
        super(TabularResBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.LayerNorm(in_features),
            nn.Mish(),
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, in_features),
            nn.LayerNorm(in_features),
        )
        self.scale = nn.Parameter(torch.tensor(0.1))
        self.act = nn.Mish()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.scale * self.block(x))


class TabularResNet(nn.Module):

    def __init__(
        self,
        num_features: int,
        hidden_dim: int = 256,
        num_res_blocks: int = 3,
        dropout_rate: float = 0.2,
    ):
        super(TabularResNet, self).__init__()
        self.input_norm = nn.LayerNorm(num_features)
        self.input_proj = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Mish(),
            nn.Dropout(dropout_rate),
        )
        self.res_blocks = nn.ModuleList(
            [TabularResBlock(hidden_dim, dropout_rate) for _ in range(num_res_blocks)]
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.Mish(),
            nn.Dropout(dropout_rate / 2.0),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_norm(x)
        h = self.input_proj(x)
        for res_block in self.res_blocks:
            h = res_block(h)
        out = self.head(h)
        return out.squeeze(-1)


# -----------------------------------------------------------------------------
# 4. Model Training & Cross-Validation Loop
# -----------------------------------------------------------------------------
N_SPLITS = 5
oof_lgb = np.zeros(n_train, dtype=np.float32)
test_preds_lgb = np.zeros(n_test, dtype=np.float32)

oof_xgb = np.zeros(n_train, dtype=np.float32)
test_preds_xgb = np.zeros(n_test, dtype=np.float32)

has_catboost = CatBoostRegressor is not None
if has_catboost:
    oof_cat = np.zeros(n_train, dtype=np.float32)
    test_preds_cat = np.zeros(n_test, dtype=np.float32)

oof_nn = np.zeros(n_train, dtype=np.float32)
test_preds_nn = np.zeros(n_test, dtype=np.float32)

lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "n_estimators": 2000,
    "learning_rate": 0.015,
    "num_leaves": 31,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.75,
    "reg_alpha": 0.2,
    "reg_lambda": 1.0,
    "random_state": SEED,
    "n_jobs": -1,
    "verbose": -1,
}

xgb_params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "n_estimators": 2000,
    "learning_rate": 0.015,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.75,
    "reg_alpha": 0.2,
    "reg_lambda": 1.0,
    "random_state": SEED,
    "n_jobs": -1,
    "tree_method": "hist",
    "early_stopping_rounds": 50,
}

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y)):
    X_tr, y_tr = X_train.iloc[train_idx], y[train_idx]
    X_va, y_va = X_train.iloc[val_idx], y[val_idx]

    # LightGBM Training
    model_lgb = lgb.LGBMRegressor(**lgb_params)
    model_lgb.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )
    oof_lgb[val_idx] = model_lgb.predict(X_va)
    test_preds_lgb += model_lgb.predict(X_test) / N_SPLITS

    # XGBoost Training
    model_xgb = xgb.XGBRegressor(**xgb_params)
    model_xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    oof_xgb[val_idx] = model_xgb.predict(X_va)
    test_preds_xgb += model_xgb.predict(X_test) / N_SPLITS

    # CatBoost Training (Optional if library present)
    if has_catboost:
        model_cat = CatBoostRegressor(
            loss_function="RMSE",
            eval_metric="RMSE",
            iterations=2000,
            learning_rate=0.02,
            depth=6,
            l2_leaf_reg=3.0,
            random_seed=SEED,
            verbose=False,
            early_stopping_rounds=50,
        )
        model_cat.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=False)
        oof_cat[val_idx] = model_cat.predict(X_va)
        test_preds_cat += model_cat.predict(X_test) / N_SPLITS

    # PyTorch TabularResNet Training
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr.values)
    X_va_s = scaler.transform(X_va.values)
    X_te_s = scaler.transform(X_test.values)

    train_ds = TensorDataset(
        torch.tensor(X_tr_s, dtype=torch.float32),
        torch.tensor(y_tr, dtype=torch.float32),
    )
    val_ds = TensorDataset(
        torch.tensor(X_va_s, dtype=torch.float32),
        torch.tensor(y_va, dtype=torch.float32),
    )

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False, drop_last=False)

    nn_model = TabularResNet(
        num_features=n_features,
        hidden_dim=256,
        num_res_blocks=3,
        dropout_rate=0.2,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(nn_model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=40, eta_min=1e-6)

    best_loss = float("inf")
    best_state = None

    for epoch in range(40):
        nn_model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            preds = nn_model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()

        scheduler.step()

        nn_model.eval()
        val_loss = 0.0
        val_count = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                preds = nn_model(batch_x)
                val_loss += criterion(preds, batch_y).item() * len(batch_y)
                val_count += len(batch_y)

        val_rmse = np.sqrt(val_loss / val_count)
        if val_rmse < best_loss:
            best_loss = val_rmse
            best_state = {k: v.clone().cpu() for k, v in nn_model.state_dict().items()}

    nn_model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    nn_model.eval()

    with torch.no_grad():
        va_tensor = torch.tensor(X_va_s, dtype=torch.float32).to(device)
        te_tensor = torch.tensor(X_te_s, dtype=torch.float32).to(device)
        oof_nn[val_idx] = nn_model(va_tensor).cpu().numpy()
        test_preds_nn += nn_model(te_tensor).cpu().numpy() / N_SPLITS

# -----------------------------------------------------------------------------
# 5. Ensemble Weight Optimization & Test Prediction
# -----------------------------------------------------------------------------
models_oof = [oof_lgb, oof_xgb, oof_nn]
models_test = [test_preds_lgb, test_preds_xgb, test_preds_nn]

if has_catboost:
    models_oof.append(oof_cat)
    models_test.append(test_preds_cat)

oof_matrix = np.column_stack(models_oof)
test_matrix = np.column_stack(models_test)
num_models = oof_matrix.shape[1]


def blend_rmse(weights):
    w = np.maximum(0, weights)
    if np.sum(w) > 0:
        w = w / np.sum(w)
    blend = np.dot(oof_matrix, w)
    return np.sqrt(mean_squared_error(y, blend))


init_weights = np.ones(num_models) / num_models
bounds = [(0, 1)] * num_models
res = minimize(
    blend_rmse, init_weights, bounds=bounds, method="SLSQP", options={"maxiter": 500}
)

best_w = np.maximum(0, res.x)
if np.sum(best_w) > 0:
    best_w = best_w / np.sum(best_w)
else:
    best_w = init_weights

oof_blend = np.clip(np.dot(oof_matrix, best_w), y_min, y_max)
test_preds_blend = np.clip(np.dot(test_matrix, best_w), y_min, y_max)

cv_rmse = float(np.sqrt(mean_squared_error(y, oof_blend)))

# Save final predictions to submission file
sub_df = pd.DataFrame({"id": test_ids, "HPSA Score": test_preds_blend})
sub_df.to_csv(os.path.join(sub_dir, "submission.csv"), index=False)

# Final validation score output
print(f"Final Validation Score: {cv_rmse}")