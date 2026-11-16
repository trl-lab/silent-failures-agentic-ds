import math
import os
from typing import Any, Dict, List, Tuple
from catboost import CatBoostRegressor
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import BayesianRidge, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import QuantileTransformer, RobustScaler, StandardScaler
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# -----------------------------------------------------------------------------
# 1. Setup Directories and Load Data
# -----------------------------------------------------------------------------
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

train_path = "./input/train.parquet"
test_path = "./input/test.parquet"

print(f"Loading data from {train_path} and {test_path}...")
df_train = pd.read_parquet(train_path)
df_test = pd.read_parquet(test_path)

print(f"Raw Train Shape: {df_train.shape}, Raw Test Shape: {df_test.shape}")

target_col = "LOANTOASSET"
if target_col not in df_train.columns:
    matched = [c for c in df_train.columns if c.upper() == target_col]
    if matched:
        target_col = matched[0]
    else:
        raise ValueError(f"Target column '{target_col}' not found in train columns.")

y_train = df_train[target_col].copy().astype(np.float64)
df_train_features = df_train.drop(columns=[target_col]).copy()
df_test_features = df_test.copy()

# -----------------------------------------------------------------------------
# 2. Data Cleaning & Modal Separation
# -----------------------------------------------------------------------------
# Drop constant columns across training features
constant_cols = [
    c
    for c in df_train_features.columns
    if df_train_features[c].nunique(dropna=False) <= 1
]
if constant_cols:
    df_train_features.drop(columns=constant_cols, inplace=True)
    df_test_features.drop(columns=constant_cols, inplace=True, errors="ignore")

# Identify ID-like columns
id_cols = [
    c
    for c in df_train_features.columns
    if c.lower() in ["id", "cert", "rssd", "unnamed: 0", "index"]
]
feature_cols = [c for c in df_train_features.columns if c not in id_cols]

cat_cols = [
    c
    for c in feature_cols
    if df_train_features[c].dtype == "object"
    or df_train_features[c].dtype.name == "category"
]
num_cols = [c for c in feature_cols if c not in cat_cols]

print(
    f"Identified {len(num_cols)} numerical features and {len(cat_cols)} categorical features."
)

# -----------------------------------------------------------------------------
# 3. Missingness Engineering & Skewness Treatment (Batch-Constructed)
# -----------------------------------------------------------------------------
new_train_features = {}
new_test_features = {}

sparse_num_cols = [c for c in num_cols if df_train_features[c].isna().mean() > 0.05]
for col in sparse_num_cols:
    new_train_features[f"{col}_isna"] = df_train_features[col].isna().astype(np.float32).values
    new_test_features[f"{col}_isna"] = df_test_features[col].isna().astype(np.float32).values

skewed_cols = []
for c in num_cols:
    val_skew = df_train_features[c].skew()
    if abs(val_skew) > 3.0:
        skewed_cols.append(c)
        new_train_features[f"{c}_log1p"] = (
            np.sign(df_train_features[c].fillna(0).values)
            * np.log1p(np.abs(df_train_features[c].fillna(0).values))
        ).astype(np.float32)
        new_test_features[f"{c}_log1p"] = (
            np.sign(df_test_features[c].fillna(0).values)
            * np.log1p(np.abs(df_test_features[c].fillna(0).values))
        ).astype(np.float32)

# -----------------------------------------------------------------------------
# 4. Domain Financial Ratios & Balance-Sheet Identity Engineering
# -----------------------------------------------------------------------------
asset_cols = [c for c in num_cols if "ASSET" in c.upper()]
loan_cols = [c for c in num_cols if "LN" in c.upper() or "LOAN" in c.upper()]
dep_cols = [c for c in num_cols if "DEP" in c.upper()]
eq_cols = [c for c in num_cols if "EQ" in c.upper() or "CAP" in c.upper()]
inc_cols = [
    c
    for c in num_cols
    if "INC" in c.upper()
    or "NII" in c.upper()
    or "ROE" in c.upper()
    or "ROA" in c.upper()
]
sec_cols = [c for c in num_cols if "SEC" in c.upper() or "INV" in c.upper()]
liab_cols = [c for c in num_cols if "LIAB" in c.upper() or "DEBT" in c.upper()]

ratio_pairs = set()
for a in asset_cols:
    for d in dep_cols:
        ratio_pairs.add((d, a))
    for e in eq_cols:
        ratio_pairs.add((e, a))
    for l in loan_cols:
        ratio_pairs.add((l, a))
    for i in inc_cols:
        ratio_pairs.add((i, a))
    for s in sec_cols:
        ratio_pairs.add((s, a))
    for li in liab_cols:
        ratio_pairs.add((li, a))

for l in loan_cols:
    for d in dep_cols:
        ratio_pairs.add((l, d))
    for e in eq_cols:
        ratio_pairs.add((l, e))

variances = df_train_features[num_cols].var().sort_values(ascending=False)
top_num_cols = variances.head(10).index.tolist()
for i in range(len(top_num_cols)):
    for j in range(i + 1, min(i + 4, len(top_num_cols))):
        ratio_pairs.add((top_num_cols[i], top_num_cols[j]))

for col_a, col_b in ratio_pairs:
    if col_a in df_train_features.columns and col_b in df_train_features.columns:
        ratio_name = f"RATIO_{col_a}__div__{col_b}"
        diff_name = f"DIFF_{col_a}__minus__{col_b}"

        denom_train = df_train_features[col_b].replace(0, np.nan).values
        new_train_features[ratio_name] = (
            df_train_features[col_a].values / (denom_train + 1e-6)
        ).astype(np.float32)
        new_train_features[diff_name] = (
            df_train_features[col_a].values - df_train_features[col_b].values
        ).astype(np.float32)

        denom_test = df_test_features[col_b].replace(0, np.nan).values
        new_test_features[ratio_name] = (
            df_test_features[col_a].values / (denom_test + 1e-6)
        ).astype(np.float32)
        new_test_features[diff_name] = (
            df_test_features[col_a].values - df_test_features[col_b].values
        ).astype(np.float32)

# Non-linear banking interactions (Equity * ROA, Deposit * Assets, etc.)
if eq_cols and inc_cols:
    e_col = eq_cols[0]
    i_col = inc_cols[0]
    new_train_features["INTERACTION_equity_x_inc"] = (
        df_train_features[e_col].values * df_train_features[i_col].values
    ).astype(np.float32)
    new_test_features["INTERACTION_equity_x_inc"] = (
        df_test_features[e_col].values * df_test_features[i_col].values
    ).astype(np.float32)

if dep_cols and asset_cols:
    d_col = dep_cols[0]
    a_col = asset_cols[0]
    new_train_features["INTERACTION_dep_x_asset"] = (
        np.log1p(np.abs(df_train_features[d_col].fillna(0).values))
        * np.log1p(np.abs(df_train_features[a_col].fillna(0).values))
    ).astype(np.float32)
    new_test_features["INTERACTION_dep_x_asset"] = (
        np.log1p(np.abs(df_test_features[d_col].fillna(0).values))
        * np.log1p(np.abs(df_test_features[a_col].fillna(0).values))
    ).astype(np.float32)

# -----------------------------------------------------------------------------
# 5. Categorical Encoding & Leakage-Free 10-Fold Statistics
# -----------------------------------------------------------------------------
for cat in cat_cols:
    freq_map = df_train_features[cat].value_counts(normalize=True).to_dict()
    new_train_features[f"{cat}_freq"] = (
        df_train_features[cat].map(freq_map).fillna(0).astype(np.float32).values
    )
    new_test_features[f"{cat}_freq"] = (
        df_test_features[cat].map(freq_map).fillna(0).astype(np.float32).values
    )

    df_train_features[cat] = (
        df_train_features[cat].astype("category").cat.codes.astype(np.int32)
    )
    df_test_features[cat] = (
        df_test_features[cat].astype("category").cat.codes.astype(np.int32)
    )

N_SPLITS = 10
target_bins = pd.qcut(y_train, q=10, labels=False, duplicates="drop")
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

fold_series = np.full(len(df_train_features), -1, dtype=np.int32)
for fold_idx, (trn_idx, val_idx) in enumerate(skf.split(df_train_features, target_bins)):
    fold_series[val_idx] = fold_idx
df_train_features["fold"] = fold_series

group_key_candidates = [
    c
    for c in cat_cols
    if df_train_features[c].nunique() > 1 and df_train_features[c].nunique() < 500
]

global_mean = float(y_train.mean())
for group_col in group_key_candidates[:4]:
    oof_col_name = f"TE_{group_col}"
    train_te = np.full(len(df_train_features), global_mean, dtype=np.float32)
    test_te = np.zeros(len(df_test_features), dtype=np.float32)

    for fold_idx in range(N_SPLITS):
        trn_mask = df_train_features["fold"] != fold_idx
        val_mask = df_train_features["fold"] == fold_idx

        trn_groups = df_train_features.loc[trn_mask, group_col]
        trn_y = y_train.loc[trn_mask]

        group_stats = trn_y.groupby(trn_groups).agg(["mean", "count"])
        smoothing = 10.0
        smoothed = (
            group_stats["mean"] * group_stats["count"] + global_mean * smoothing
        ) / (group_stats["count"] + smoothing)
        te_map = smoothed.to_dict()

        train_te[val_mask] = (
            df_train_features.loc[val_mask, group_col]
            .map(te_map)
            .fillna(global_mean)
            .astype(np.float32)
            .values
        )

        test_te += (
            df_test_features[group_col]
            .map(te_map)
            .fillna(global_mean)
            .astype(np.float32)
            .values
            / N_SPLITS
        )

    new_train_features[oof_col_name] = train_te
    new_test_features[oof_col_name] = test_te

# Banking Peer-Group Aggregations (Mean, Median, Std, Z-Score, Deviations)
key_peer_num_cols = top_num_cols[:4] if len(top_num_cols) >= 4 else num_cols[:4]
for group_col in group_key_candidates[:3]:
    for num_c in key_peer_num_cols:
        if num_c not in df_train_features.columns:
            continue
        peer_mean_col = f"PEER_MEAN_{num_c}_by_{group_col}"
        peer_median_col = f"PEER_MEDIAN_{num_c}_by_{group_col}"
        peer_zscore_col = f"PEER_ZSCORE_{num_c}_by_{group_col}"

        train_p_mean = np.zeros(len(df_train_features), dtype=np.float32)
        train_p_median = np.zeros(len(df_train_features), dtype=np.float32)
        train_p_zscore = np.zeros(len(df_train_features), dtype=np.float32)
        test_p_mean = np.zeros(len(df_test_features), dtype=np.float32)
        test_p_median = np.zeros(len(df_test_features), dtype=np.float32)
        test_p_zscore = np.zeros(len(df_test_features), dtype=np.float32)

        for fold_idx in range(N_SPLITS):
            trn_mask = df_train_features["fold"] != fold_idx
            val_mask = df_train_features["fold"] == fold_idx

            trn_subset = df_train_features.loc[trn_mask, [group_col, num_c]]
            group_stats = trn_subset.groupby(group_col)[num_c].agg(["mean", "median", "std"])

            mean_map = group_stats["mean"].to_dict()
            med_map = group_stats["median"].to_dict()
            std_map = group_stats["std"].fillna(1.0).to_dict()

            overall_m = df_train_features.loc[trn_mask, num_c].mean()
            overall_med = df_train_features.loc[trn_mask, num_c].median()
            overall_s = df_train_features.loc[trn_mask, num_c].std()
            if pd.isna(overall_s) or overall_s == 0:
                overall_s = 1.0

            val_means = (
                df_train_features.loc[val_mask, group_col]
                .map(mean_map)
                .fillna(overall_m)
                .astype(np.float32)
                .values
            )
            val_meds = (
                df_train_features.loc[val_mask, group_col]
                .map(med_map)
                .fillna(overall_med)
                .astype(np.float32)
                .values
            )
            val_stds = (
                df_train_features.loc[val_mask, group_col]
                .map(std_map)
                .fillna(overall_s)
                .replace(0, 1.0)
                .astype(np.float32)
                .values
            )
            train_p_mean[val_mask] = val_means
            train_p_median[val_mask] = val_meds
            train_p_zscore[val_mask] = (
                (df_train_features.loc[val_mask, num_c].values - val_means) / (val_stds + 1e-5)
            ).astype(np.float32)

            t_means = (
                df_test_features[group_col]
                .map(mean_map)
                .fillna(overall_m)
                .astype(np.float32)
                .values
            )
            t_meds = (
                df_test_features[group_col]
                .map(med_map)
                .fillna(overall_med)
                .astype(np.float32)
                .values
            )
            t_stds = (
                df_test_features[group_col]
                .map(std_map)
                .fillna(overall_s)
                .replace(0, 1.0)
                .astype(np.float32)
                .values
            )
            test_p_mean += t_means / N_SPLITS
            test_p_median += t_meds / N_SPLITS
            test_p_zscore += (
                (df_test_features[num_c].values - t_means) / (t_stds + 1e-5)
            ).astype(np.float32) / N_SPLITS

        new_train_features[peer_mean_col] = train_p_mean
        new_train_features[peer_median_col] = train_p_median
        new_train_features[peer_zscore_col] = train_p_zscore
        new_test_features[peer_mean_col] = test_p_mean
        new_test_features[peer_median_col] = test_p_median
        new_test_features[peer_zscore_col] = test_p_zscore

# Batch concatenate all new features at once to prevent fragmentation
df_train_features = pd.concat(
    [df_train_features, pd.DataFrame(new_train_features, index=df_train_features.index)],
    axis=1,
)
df_test_features = pd.concat(
    [df_test_features, pd.DataFrame(new_test_features, index=df_test_features.index)],
    axis=1,
)

# High correlation pruning to eliminate redundant/degenerate features
feature_candidates = [c for c in df_train_features.columns if c not in ["fold", "id"]]
corr_matrix = df_train_features[feature_candidates].corr().abs()
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
drop_redundant = [col for col in upper_tri.columns if any(upper_tri[col] > 0.998)]
if drop_redundant:
    df_train_features.drop(columns=drop_redundant, inplace=True)
    df_test_features.drop(columns=drop_redundant, inplace=True, errors="ignore")

df_train_features[target_col] = y_train.values

feature_columns = [
    c for c in df_train_features.columns if c not in [target_col, "fold"]
]
print(f"Total engineered features after collinear pruning: {len(feature_columns)}")


# -----------------------------------------------------------------------------
# 6. Tabular Deep Neural Network Architecture & Model Builders
# -----------------------------------------------------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int, dropout_rate: float = 0.15):
        super().__init__()
        self.norm1 = nn.LayerNorm(in_features)
        self.linear1 = nn.Linear(in_features, out_features)
        self.act1 = nn.Mish()
        self.dropout1 = nn.Dropout(dropout_rate)

        self.norm2 = nn.LayerNorm(out_features)
        self.linear2 = nn.Linear(out_features, out_features)
        self.act2 = nn.Mish()
        self.dropout2 = nn.Dropout(dropout_rate)

        if in_features != out_features:
            self.shortcut = nn.Linear(in_features, out_features)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.norm1(x)
        out = self.dropout1(self.act1(self.linear1(out)))
        out = self.norm2(out)
        out = self.dropout2(self.act2(self.linear2(out)))
        return out + residual


class TabularResNet(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_dims: List[int] = [256, 128, 64],
        dropout_rate: float = 0.15,
    ):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Linear(in_features, hidden_dims[0]),
            nn.Mish(),
        )

        blocks = []
        for i in range(len(hidden_dims) - 1):
            blocks.append(
                ResidualBlock(
                    hidden_dims[i], hidden_dims[i + 1], dropout_rate=dropout_rate
                )
            )
        self.res_blocks = nn.ModuleList(blocks)

        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dims[-1]),
            nn.Dropout(dropout_rate / 2.0),
            nn.Linear(hidden_dims[-1], 32),
            nn.Mish(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(x)
        for block in self.res_blocks:
            x = block(x)
        out = self.head(x)
        return out.squeeze(-1)


class RMSELoss(nn.Module):
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.mse = nn.MSELoss(reduction="mean")
        self.eps = eps

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(self.mse(predictions, targets) + self.eps)


def build_neural_pipeline(
    in_features: int,
    lr: float = 2e-3,
    weight_decay: float = 1e-4,
    epochs: int = 50,
    device: str = "cpu",
) -> Tuple[nn.Module, nn.Module, torch.optim.Optimizer, Any]:
    model = TabularResNet(
        in_features=in_features, hidden_dims=[256, 128, 64], dropout_rate=0.15
    ).to(device)
    criterion = RMSELoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    return model, criterion, optimizer, scheduler


def get_lightgbm_configs(seed: int = 42) -> Dict[str, Any]:
    return {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": -1,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "min_child_samples": 20,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": seed,
        "verbosity": -1,
        "n_jobs": -1,
    }


def get_xgboost_configs(seed: int = 42) -> Dict[str, Any]:
    return {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "learning_rate": 0.03,
        "max_depth": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "min_child_weight": 5,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": seed,
        "n_jobs": -1,
    }


def get_catboost_configs(seed: int = 42) -> Dict[str, Any]:
    return {
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "iterations": 1500,
        "learning_rate": 0.03,
        "depth": 6,
        "l2_leaf_reg": 5.0,
        "random_seed": seed,
        "verbose": False,
        "thread_count": -1,
    }


# -----------------------------------------------------------------------------
# 7. Model Training & Cross-Validation
# -----------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training using computation device: {device}")

X = df_train_features[feature_columns].copy()
y = df_train_features[target_col].values.astype(np.float64)
folds = df_train_features["fold"].values
X_test = df_test_features[feature_columns].copy()

n_train = len(df_train_features)
n_test = len(df_test_features)

oof_lgb = np.zeros(n_train, dtype=np.float64)
oof_cb = np.zeros(n_train, dtype=np.float64)
oof_xgb = np.zeros(n_train, dtype=np.float64)
oof_et = np.zeros(n_train, dtype=np.float64)
oof_nn = np.zeros(n_train, dtype=np.float64)

test_preds_lgb = np.zeros(n_test, dtype=np.float64)
test_preds_cb = np.zeros(n_test, dtype=np.float64)
test_preds_xgb = np.zeros(n_test, dtype=np.float64)
test_preds_et = np.zeros(n_test, dtype=np.float64)
test_preds_nn = np.zeros(n_test, dtype=np.float64)

# 7.1 LightGBM Training
print("\n--- Training LightGBM Models ---")
lgb_params = get_lightgbm_configs(seed=42)

for fold_idx in range(N_SPLITS):
    trn_idx = np.where(folds != fold_idx)[0]
    val_idx = np.where(folds == fold_idx)[0]

    X_tr, y_tr = X.iloc[trn_idx], y[trn_idx]
    X_va, y_va = X.iloc[val_idx], y[val_idx]

    dtrain = lgb.Dataset(X_tr, label=y_tr)
    dval = lgb.Dataset(X_va, label=y_va, reference=dtrain)

    callbacks = [
        lgb.early_stopping(stopping_rounds=50, verbose=False),
        lgb.log_evaluation(period=0),
    ]
    model_lgb = lgb.train(
        lgb_params, dtrain, num_boost_round=1500, valid_sets=[dval], callbacks=callbacks
    )

    best_iter = model_lgb.best_iteration
    val_pred = model_lgb.predict(X_va, num_iteration=best_iter)
    oof_lgb[val_idx] = val_pred
    test_preds_lgb += model_lgb.predict(X_test, num_iteration=best_iter) / N_SPLITS

    fold_rmse = np.sqrt(mean_squared_error(y_va, val_pred))
    print(
        f"LightGBM Fold {fold_idx + 1}/{N_SPLITS} - RMSE: {fold_rmse:.5f} (Best Iter: {best_iter})"
    )

lgb_total_rmse = np.sqrt(mean_squared_error(y, oof_lgb))
print(f"--> LightGBM Overall OOF RMSE: {lgb_total_rmse:.5f}")

# 7.2 CatBoost Training
print("\n--- Training CatBoost Models ---")
cb_params = get_catboost_configs(seed=42)

for fold_idx in range(N_SPLITS):
    trn_idx = np.where(folds != fold_idx)[0]
    val_idx = np.where(folds == fold_idx)[0]

    X_tr, y_tr = X.iloc[trn_idx], y[trn_idx]
    X_va, y_va = X.iloc[val_idx], y[val_idx]

    model_cb = CatBoostRegressor(**cb_params)
    model_cb.fit(
        X_tr,
        y_tr,
        eval_set=(X_va, y_va),
        early_stopping_rounds=50,
        verbose=False,
    )

    best_iter = model_cb.get_best_iteration()
    val_pred = model_cb.predict(X_va)
    oof_cb[val_idx] = val_pred
    test_preds_cb += model_cb.predict(X_test) / N_SPLITS

    fold_rmse = np.sqrt(mean_squared_error(y_va, val_pred))
    print(
        f"CatBoost Fold {fold_idx + 1}/{N_SPLITS} - RMSE: {fold_rmse:.5f} (Best Iter: {best_iter})"
    )

cb_total_rmse = np.sqrt(mean_squared_error(y, oof_cb))
print(f"--> CatBoost Overall OOF RMSE: {cb_total_rmse:.5f}")

# 7.3 XGBoost Training
print("\n--- Training XGBoost Models ---")
xgb_params = get_xgboost_configs(seed=42)

for fold_idx in range(N_SPLITS):
    trn_idx = np.where(folds != fold_idx)[0]
    val_idx = np.where(folds == fold_idx)[0]

    X_tr, y_tr = X.iloc[trn_idx], y[trn_idx]
    X_va, y_va = X.iloc[val_idx], y[val_idx]

    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dval = xgb.DMatrix(X_va, label=y_va)
    dtest = xgb.DMatrix(X_test)

    model_xgb = xgb.train(
        xgb_params,
        dtrain,
        num_boost_round=1500,
        evals=[(dval, "val")],
        early_stopping_rounds=50,
        verbose_eval=False,
    )

    best_iter = model_xgb.best_iteration
    val_pred = model_xgb.predict(dval)
    oof_xgb[val_idx] = val_pred
    test_preds_xgb += model_xgb.predict(dtest) / N_SPLITS

    fold_rmse = np.sqrt(mean_squared_error(y_va, val_pred))
    print(
        f"XGBoost Fold {fold_idx + 1}/{N_SPLITS} - RMSE: {fold_rmse:.5f} (Best Iter: {best_iter})"
    )

xgb_total_rmse = np.sqrt(mean_squared_error(y, oof_xgb))
print(f"--> XGBoost Overall OOF RMSE: {xgb_total_rmse:.5f}")

# 7.4 ExtraTrees Training
print("\n--- Training ExtraTrees Models ---")
for fold_idx in range(N_SPLITS):
    trn_idx = np.where(folds != fold_idx)[0]
    val_idx = np.where(folds == fold_idx)[0]

    imputer = SimpleImputer(strategy="median")
    X_tr_imp = np.nan_to_num(imputer.fit_transform(X.iloc[trn_idx]), nan=0.0, posinf=0.0, neginf=0.0)
    X_va_imp = np.nan_to_num(imputer.transform(X.iloc[val_idx]), nan=0.0, posinf=0.0, neginf=0.0)
    X_test_imp = np.nan_to_num(imputer.transform(X_test), nan=0.0, posinf=0.0, neginf=0.0)

    model_et = ExtraTreesRegressor(
        n_estimators=300,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=42 + fold_idx,
        n_jobs=-1,
    )
    model_et.fit(X_tr_imp, y[trn_idx])

    val_pred = model_et.predict(X_va_imp)
    oof_et[val_idx] = val_pred
    test_preds_et += model_et.predict(X_test_imp) / N_SPLITS

    fold_rmse = np.sqrt(mean_squared_error(y[val_idx], val_pred))
    print(
        f"ExtraTrees Fold {fold_idx + 1}/{N_SPLITS} - RMSE: {fold_rmse:.5f}"
    )

et_total_rmse = np.sqrt(mean_squared_error(y, oof_et))
print(f"--> ExtraTrees Overall OOF RMSE: {et_total_rmse:.5f}")

# 7.5 Tabular Neural Network Training
print("\n--- Training TabularResNet Models ---")
NN_EPOCHS = 50
BATCH_SIZE = 128

for fold_idx in range(N_SPLITS):
    trn_idx = np.where(folds != fold_idx)[0]
    val_idx = np.where(folds == fold_idx)[0]

    scaler = StandardScaler()
    X_tr_scaled = np.nan_to_num(X.iloc[trn_idx].values, nan=0.0, posinf=0.0, neginf=0.0)
    X_va_scaled = np.nan_to_num(X.iloc[val_idx].values, nan=0.0, posinf=0.0, neginf=0.0)
    X_test_scaled = np.nan_to_num(X_test.values, nan=0.0, posinf=0.0, neginf=0.0)

    X_tr_norm = np.clip(scaler.fit_transform(X_tr_scaled), -10.0, 10.0).astype(
        np.float32
    )
    X_va_norm = np.clip(scaler.transform(X_va_scaled), -10.0, 10.0).astype(np.float32)
    X_test_norm = np.clip(scaler.transform(X_test_scaled), -10.0, 10.0).astype(
        np.float32
    )

    y_tr_t = torch.tensor(y[trn_idx], dtype=torch.float32)
    y_va_t = torch.tensor(y[val_idx], dtype=torch.float32)

    train_dataset = TensorDataset(torch.tensor(X_tr_norm), y_tr_t)
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False
    )

    model, criterion, optimizer, scheduler = build_neural_pipeline(
        in_features=X.shape[1],
        lr=2e-3,
        weight_decay=1e-4,
        epochs=NN_EPOCHS,
        device=str(device),
    )

    best_val_rmse = float("inf")
    best_weights = None
    patience = 12
    patience_counter = 0

    for epoch in range(NN_EPOCHS):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            va_x_t = torch.tensor(X_va_norm, dtype=torch.float32).to(device)
            va_preds = model(va_x_t).cpu().numpy()
            epoch_val_rmse = np.sqrt(mean_squared_error(y[val_idx], va_preds))

        if epoch_val_rmse < best_val_rmse:
            best_val_rmse = epoch_val_rmse
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    model.load_state_dict({k: v.to(device) for k, v in best_weights.items()})
    model.eval()
    with torch.no_grad():
        va_x_t = torch.tensor(X_va_norm, dtype=torch.float32).to(device)
        test_x_t = torch.tensor(X_test_norm, dtype=torch.float32).to(device)
        best_va_preds = model(va_x_t).cpu().numpy().astype(np.float64)
        best_test_preds = model(test_x_t).cpu().numpy().astype(np.float64)

    oof_nn[val_idx] = best_va_preds
    test_preds_nn += best_test_preds / N_SPLITS
    print(f"TabularResNet Fold {fold_idx + 1}/{N_SPLITS} - RMSE: {best_val_rmse:.5f}")

nn_total_rmse = np.sqrt(mean_squared_error(y, oof_nn))
print(f"--> TabularResNet Overall OOF RMSE: {nn_total_rmse:.5f}")

# -----------------------------------------------------------------------------
# 8. Optimal Ensemble Blending & Validation Evaluation
# -----------------------------------------------------------------------------
print("\n--- Optimizing Ensemble Blend Weights ---")
oof_matrix = np.column_stack([oof_lgb, oof_cb, oof_xgb, oof_et, oof_nn])
test_matrix = np.column_stack([test_preds_lgb, test_preds_cb, test_preds_xgb, test_preds_et, test_preds_nn])


def blend_objective(weights):
    weights = np.array(weights)
    weights = weights / np.sum(weights)
    blended = np.dot(oof_matrix, weights)
    return np.sqrt(mean_squared_error(y, blended))


init_weights = np.array([0.30, 0.30, 0.20, 0.10, 0.10])
bounds = [(0.0, 1.0) for _ in range(5)]
constraints = {"type": "eq", "fun": lambda w: 1.0 - sum(w)}

res = minimize(
    blend_objective,
    init_weights,
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
)
optimal_weights = res.x / np.sum(res.x)

print(f"Optimized Blend Weights [LGBM, CatBoost, XGB, ExtraTrees, TabularResNet]: {optimal_weights.round(4)}")

oof_ensemble = np.dot(oof_matrix, optimal_weights)
test_ensemble = np.dot(test_matrix, optimal_weights)

final_score = np.sqrt(mean_squared_error(y, oof_ensemble))
print(f"Ensemble Out-of-Fold RMSE: {final_score:.5f}")

# -----------------------------------------------------------------------------
# 9. Generate Submission File
# -----------------------------------------------------------------------------
submission_path = "./submission/submission.csv"
df_submission = pd.DataFrame(
    {"id": np.arange(len(test_ensemble), dtype=np.int64), "LOANTOASSET": test_ensemble}
)

assert len(df_submission) == len(df_test_features), "Submission row count mismatch!"
assert df_submission["LOANTOASSET"].isna().sum() == 0, "Submission contains NaN values!"
assert list(df_submission.columns) == [
    "id",
    "LOANTOASSET",
], "Submission column names incorrect!"

df_submission.to_csv(submission_path, index=False)
print(f"Submission file saved successfully to {submission_path}")

# -----------------------------------------------------------------------------
# 10. Final Validation Score Output
# -----------------------------------------------------------------------------
print(f"Final Validation Score: {final_score}")
