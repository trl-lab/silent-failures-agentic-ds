import json
import math
import os
import re
import warnings
import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb

warnings.filterwarnings("ignore")

# ==============================================================================
# 1. SETUP AND FEATURE ENGINEERING
# ==============================================================================
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

train_path = "./input/train.parquet"
test_path = "./input/test.parquet"

train = pd.read_parquet(train_path)
test = pd.read_parquet(test_path)

target_col = "Amount in USD"
assert (
    target_col in train.columns
), f"Target column '{target_col}' not found in train dataset."

# Standardize ID column
if "id" not in test.columns and "ID" not in test.columns:
    test["id"] = np.arange(len(test))
elif "ID" in test.columns and "id" not in test.columns:
    test["id"] = test["ID"]

if "id" not in train.columns and "ID" not in train.columns:
    train["id"] = np.arange(len(train))
elif "ID" in train.columns and "id" not in train.columns:
    train["id"] = train["ID"]

# Stratified 5-Fold Scheme based on Log-Target Deciles
y = np.maximum(0, train[target_col].values)
y_log = np.log1p(y)

num_folds = 5
skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
bins = pd.qcut(y_log, q=num_folds * 2, labels=False, duplicates="drop")

train["fold"] = -1
for fold, (train_idx, val_idx) in enumerate(skf.split(train, bins)):
    train.loc[val_idx, "fold"] = fold

# Dynamic Column Identification
non_feature_cols = [target_col, "id", "ID", "fold"]
candidate_cols = [c for c in train.columns if c not in non_feature_cols]

num_cols = []
cat_cols = []
text_cols = []
date_cols = []

for c in candidate_cols:
    if pd.api.types.is_datetime64_any_dtype(train[c]):
        date_cols.append(c)
    elif pd.api.types.is_numeric_dtype(train[c]):
        if train[c].nunique() < 10 and train[c].dtype in [np.int64, int]:
            cat_cols.append(c)
        else:
            num_cols.append(c)
    else:
        train_str = train[c].astype(str).fillna("")
        avg_len = train_str.str.len().mean()
        is_date = False
        if "date" in c.lower() or "year" in c.lower() or "time" in c.lower():
            try:
                pd.to_datetime(train_str.iloc[:100], errors="raise")
                is_date = True
            except Exception:
                pass
        if is_date:
            date_cols.append(c)
        elif avg_len > 40:
            text_cols.append(c)
        else:
            cat_cols.append(c)

train_df = train.copy()
test_df = test.copy()

# Temporal Feature Engineering
for dcol in date_cols:
    for df_curr in [train_df, test_df]:
        dt_series = pd.to_datetime(df_curr[dcol], errors="coerce")
        df_curr[f"{dcol}_year"] = dt_series.dt.year.fillna(-1).astype(float)
        df_curr[f"{dcol}_month"] = dt_series.dt.month.fillna(-1).astype(float)
        df_curr[f"{dcol}_quarter"] = dt_series.dt.quarter.fillna(-1).astype(float)
        df_curr[f"{dcol}_day"] = dt_series.dt.day.fillna(-1).astype(float)
        df_curr[f"{dcol}_dayofweek"] = dt_series.dt.dayofweek.fillna(-1).astype(float)

        month_rad = 2 * np.pi * df_curr[f"{dcol}_month"] / 12.0
        df_curr[f"{dcol}_month_sin"] = np.sin(month_rad)
        df_curr[f"{dcol}_month_cos"] = np.cos(month_rad)

        df_curr.drop(columns=[dcol], inplace=True, errors="ignore")

# Text Embeddings & Complexity Metrics
if text_cols:
    train_text = train_df[text_cols].astype(str).fillna("").agg(" ".join, axis=1)
    test_text = test_df[text_cols].astype(str).fillna("").agg(" ".join, axis=1)

    for t_series, df_curr in [(train_text, train_df), (test_text, test_df)]:
        df_curr["text_char_count"] = t_series.str.len()
        df_curr["text_word_count"] = t_series.str.split().str.len()
        df_curr["text_avg_word_len"] = df_curr["text_char_count"] / (
            df_curr["text_word_count"] + 1e-5
        )
        df_curr["text_uppercase_ratio"] = t_series.apply(
            lambda x: sum(1 for c in x if c.isupper()) / (len(x) + 1e-5)
        )
        df_curr["text_digit_ratio"] = t_series.apply(
            lambda x: sum(1 for c in x if c.isdigit()) / (len(x) + 1e-5)
        )

    tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=5000, stop_words="english")
    tfidf_train = tfidf.fit_transform(train_text)
    tfidf_test = tfidf.transform(test_text)

    svd = TruncatedSVD(n_components=16, random_state=42)
    svd_train = svd.fit_transform(tfidf_train)
    svd_test = svd.transform(tfidf_test)

    for i in range(16):
        train_df[f"text_svd_{i}"] = svd_train[:, i]
        test_df[f"text_svd_{i}"] = svd_test[:, i]

    train_df.drop(columns=text_cols, inplace=True, errors="ignore")
    test_df.drop(columns=text_cols, inplace=True, errors="ignore")

# Categorical Frequency & Out-of-Fold Target Encodings
smooth_m = 10.0
global_log_mean = float(y_log.mean())

for ccol in cat_cols:
    freq = train_df[ccol].value_counts(normalize=True).to_dict()
    train_df[f"{ccol}_freq"] = train_df[ccol].map(freq).fillna(0.0).astype(np.float32)
    test_df[f"{ccol}_freq"] = test_df[ccol].map(freq).fillna(0.0).astype(np.float32)

    counts = train_df.groupby(ccol).size()
    means = train_df.assign(log_y=y_log).groupby(ccol)["log_y"].mean()
    smoothed_test = (counts * means + smooth_m * global_log_mean) / (counts + smooth_m)
    test_df[f"{ccol}_te"] = (
        test_df[ccol].map(smoothed_test).fillna(global_log_mean).astype(np.float32)
    )

    train_df[f"{ccol}_te"] = global_log_mean
    for f in range(num_folds):
        idx_tr = train_df["fold"] != f
        idx_val = train_df["fold"] == f

        f_train = train_df[idx_tr]
        f_y_log = y_log[idx_tr]

        f_counts = f_train.groupby(ccol).size()
        f_means = f_train.assign(log_y=f_y_log).groupby(ccol)["log_y"].mean()
        f_global_mean = float(f_y_log.mean())

        f_smoothed = (f_counts * f_means + smooth_m * f_global_mean) / (
            f_counts + smooth_m
        )
        train_df.loc[idx_val, f"{ccol}_te"] = (
            train_df.loc[idx_val, ccol]
            .map(f_smoothed)
            .fillna(f_global_mean)
            .astype(np.float32)
        )

# Group Aggregations
top_cats = sorted(cat_cols, key=lambda c: train_df[c].nunique())[:3]
for ccol in top_cats:
    for ncol in num_cols[:5]:
        grp_means = train_df.groupby(ccol)[ncol].mean()
        train_df[f"{ncol}_by_{ccol}_mean"] = train_df[ccol].map(grp_means)
        train_df[f"{ncol}_ratio_{ccol}"] = train_df[ncol] / (
            train_df[f"{ncol}_by_{ccol}_mean"] + 1e-5
        )

        test_grp_means = test_df[ccol].map(grp_means).fillna(train_df[ncol].mean())
        test_df[f"{ncol}_by_{ccol}_mean"] = test_grp_means
        test_df[f"{ncol}_ratio_{ccol}"] = test_df[ncol] / (test_grp_means + 1e-5)


def sanitize_colname(col):
    return re.sub(r"[^\w\s]", "_", str(col)).replace(" ", "_")


train_df.columns = [sanitize_colname(c) for c in train_df.columns]
test_df.columns = [sanitize_colname(c) for c in test_df.columns]

feature_cols = [
    c
    for c in train_df.columns
    if c not in [sanitize_colname(target_col), "id", "ID", "fold"]
]

for col in feature_cols:
    if pd.api.types.is_numeric_dtype(train_df[col]):
        med = train_df[col].median()
        if np.isnan(med):
            med = 0.0
        train_df[col] = train_df[col].replace([np.inf, -np.inf], np.nan).fillna(med)
        test_df[col] = test_df[col].replace([np.inf, -np.inf], np.nan).fillna(med)
    else:
        categories = train_df[col].astype(str).fillna("Missing").unique()
        cat_type = pd.CategoricalDtype(categories=categories, ordered=False)
        train_df[col] = (
            train_df[col]
            .astype(str)
            .fillna("Missing")
            .astype(cat_type)
            .cat.codes.astype(np.float32)
        )
        test_df[col] = (
            test_df[col]
            .astype(str)
            .fillna("Missing")
            .astype(cat_type)
            .cat.codes.astype(np.float32)
        )

train_df.to_parquet("./working/train_processed.parquet", index=False)
test_df.to_parquet("./working/test_processed.parquet", index=False)

metadata = {
    "target_col": sanitize_colname(target_col),
    "fold_col": "fold",
    "id_col": "id",
    "feature_cols": feature_cols,
}

with open("./working/feature_metadata.json", "w") as f:
    json.dump(metadata, f, indent=4)


# ==============================================================================
# 2. MODEL DEFINITIONS
# ==============================================================================
class TabularResBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.15):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.act = nn.Mish()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.norm1(x)
        out = self.fc1(out)
        out = self.act(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.norm2(out)
        return residual + out


class TabularResMLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_dim: int = 256,
        num_blocks: int = 3,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Mish(),
            nn.Dropout(dropout),
        )

        self.blocks = nn.ModuleList(
            [TabularResBlock(hidden_dim, dropout) for _ in range(num_blocks)]
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Mish(),
            nn.Dropout(dropout / 2.0),
            nn.Linear(hidden_dim // 2, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(x)
        for block in self.blocks:
            x = block(x)
        out = self.head(x)
        return out.squeeze(-1)


class RMSELoss(nn.Module):
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        mse = torch.mean((y_pred - y_true) ** 2)
        return torch.sqrt(mse + self.eps)


def create_optimizer_and_scheduler(
    model: nn.Module,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    num_epochs: int = 35,
):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-6
    )
    return optimizer, scheduler


def get_lgb_params(random_state: int = 42) -> dict:
    return {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "n_estimators": 2000,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": 7,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": random_state,
        "n_jobs": -1,
        "verbose": -1,
    }


def get_xgb_params(random_state: int = 42) -> dict:
    return {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "n_estimators": 2000,
        "learning_rate": 0.03,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": random_state,
        "n_jobs": -1,
        "tree_method": "hist",
    }


def get_catboost_params(random_state: int = 42) -> dict:
    return {
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "iterations": 2000,
        "learning_rate": 0.03,
        "depth": 6,
        "l2_leaf_reg": 3.0,
        "random_seed": random_state,
        "verbose": False,
        "thread_count": -1,
    }


# ==============================================================================
# 3. CROSS-VALIDATION TRAINING AND ENSEMBLE BLENDING
# ==============================================================================
target_col_meta = metadata["target_col"]
fold_col = metadata["fold_col"]
id_col = metadata["id_col"]

y_true = np.maximum(0, train_df[target_col_meta].values.astype(np.float64))
y_log = np.log1p(y_true)

X_train_full = train_df[feature_cols].values.astype(np.float32)
X_test_full = test_df[feature_cols].values.astype(np.float32)
folds = train_df[fold_col].values
num_folds = len(np.unique(folds))

oof_lgb = np.zeros(len(train_df))
oof_xgb = np.zeros(len(train_df))
oof_cat = np.zeros(len(train_df))
oof_nn = np.zeros(len(train_df))

test_lgb = np.zeros(len(test_df))
test_xgb = np.zeros(len(test_df))
test_cat = np.zeros(len(test_df))
test_nn = np.zeros(len(test_df))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for fold in range(num_folds):
    idx_tr = folds != fold
    idx_va = folds == fold

    X_tr, y_tr_log = X_train_full[idx_tr], y_log[idx_tr]
    X_va, y_va_log = X_train_full[idx_va], y_log[idx_va]

    # LightGBM
    lgb_params = get_lgb_params(random_state=42 + fold)
    lgb_model = lgb.LGBMRegressor(**lgb_params)
    lgb_model.fit(
        X_tr,
        y_tr_log,
        eval_set=[(X_va, y_va_log)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    pred_lgb_va = np.expm1(np.maximum(0, lgb_model.predict(X_va)))
    oof_lgb[idx_va] = pred_lgb_va
    test_lgb += np.expm1(np.maximum(0, lgb_model.predict(X_test_full))) / num_folds

    # XGBoost
    xgb_params = get_xgb_params(random_state=42 + fold)
    xgb_params["early_stopping_rounds"] = 50
    xgb_model = xgb.XGBRegressor(**xgb_params)
    xgb_model.fit(X_tr, y_tr_log, eval_set=[(X_va, y_va_log)], verbose=False)
    pred_xgb_va = np.expm1(np.maximum(0, xgb_model.predict(X_va)))
    oof_xgb[idx_va] = pred_xgb_va
    test_xgb += np.expm1(np.maximum(0, xgb_model.predict(X_test_full))) / num_folds

    # CatBoost
    cat_params = get_catboost_params(random_state=42 + fold)
    cat_params["early_stopping_rounds"] = 50
    cat_model = cb.CatBoostRegressor(**cat_params)
    cat_model.fit(
        X_tr,
        y_tr_log,
        eval_set=(X_va, y_va_log),
        verbose=False,
        use_best_model=True,
    )
    pred_cat_va = np.expm1(np.maximum(0, cat_model.predict(X_va)))
    oof_cat[idx_va] = pred_cat_va
    test_cat += np.expm1(np.maximum(0, cat_model.predict(X_test_full))) / num_folds

    # PyTorch TabularResMLP
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_va_scaled = scaler.transform(X_va)
    X_test_scaled = scaler.transform(X_test_full)

    ds_tr = TensorDataset(
        torch.tensor(X_tr_scaled, dtype=torch.float32),
        torch.tensor(y_tr_log, dtype=torch.float32),
    )
    ds_va = TensorDataset(
        torch.tensor(X_va_scaled, dtype=torch.float32),
        torch.tensor(y_va_log, dtype=torch.float32),
    )

    loader_tr = DataLoader(ds_tr, batch_size=128, shuffle=True)
    loader_va = DataLoader(ds_va, batch_size=256, shuffle=False)

    nn_model = TabularResMLP(
        in_features=X_train_full.shape[1],
        hidden_dim=256,
        num_blocks=3,
        dropout=0.15,
    ).to(device)
    criterion = RMSELoss()
    optimizer, scheduler = create_optimizer_and_scheduler(
        nn_model, lr=1e-3, weight_decay=1e-4, num_epochs=35
    )

    best_val_loss = float("inf")
    best_nn_weights = None

    for epoch in range(35):
        nn_model.train()
        for bx, by in loader_tr:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = nn_model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()

        scheduler.step()

        nn_model.eval()
        val_loss = 0.0
        val_count = 0
        with torch.no_grad():
            for bx, by in loader_va:
                bx, by = bx.to(device), by.to(device)
                out = nn_model(bx)
                val_loss += criterion(out, by).item() * len(by)
                val_count += len(by)
        val_loss /= val_count

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_nn_weights = {k: v.cpu() for k, v in nn_model.state_dict().items()}

    nn_model.load_state_dict(best_nn_weights)
    nn_model.to(device)
    nn_model.eval()

    with torch.no_grad():
        va_tensor = torch.tensor(X_va_scaled, dtype=torch.float32).to(device)
        pred_nn_va = np.expm1(np.maximum(0, nn_model(va_tensor).cpu().numpy()))
        oof_nn[idx_va] = pred_nn_va

        test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
        test_nn += (
            np.expm1(np.maximum(0, nn_model(test_tensor).cpu().numpy())) / num_folds
        )

    print(
        f"Fold {fold+1}/{num_folds} Finished. LGB RMSE: {np.sqrt(mean_squared_error(y_true[idx_va], pred_lgb_va)):.2f}"
    )

# SLSQP Ensemble Weight Optimization
oof_matrix = np.column_stack([oof_lgb, oof_xgb, oof_cat, oof_nn])
test_matrix = np.column_stack([test_lgb, test_xgb, test_cat, test_nn])


def obj_func(weights):
    weights = np.maximum(0, weights)
    if np.sum(weights) > 0:
        weights = weights / np.sum(weights)
    blend = np.dot(oof_matrix, weights)
    return np.sqrt(mean_squared_error(y_true, blend))


init_weights = [0.25, 0.25, 0.25, 0.25]
bounds = [(0, 1)] * 4
constraints = {"type": "eq", "fun": lambda w: 1.0 - np.sum(w)}

res = minimize(
    obj_func,
    init_weights,
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
)
opt_weights = res.x / np.sum(res.x)

final_oof_pred = np.dot(oof_matrix, opt_weights)
final_validation_rmse = np.sqrt(mean_squared_error(y_true, final_oof_pred))

# Save Test Submission
final_test_pred = np.maximum(0, np.dot(test_matrix, opt_weights))

sub_df = pd.DataFrame(
    {
        "id": test_df[id_col].astype(int),
        "Amount in USD": np.round(final_test_pred, 4),
    }
)

sub_df.to_csv("./submission/submission.csv", index=False)

print(f"Final Validation Score: {final_validation_rmse}")
