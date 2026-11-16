import os
import re
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")
torch.manual_seed(42)
np.random.seed(42)

# =========================================================================
# 1. Setup Directories & Load Data
# =========================================================================
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

train_path = "./input/train.parquet"
test_path = "./input/test.parquet"

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = "Amount in USD"
if target_col not in train_df.columns:
    candidates = [
        c for c in train_df.columns if "amount" in c.lower() or "usd" in c.lower()
    ]
    if candidates:
        target_col = candidates[0]

print(f"Loaded train shape: {train_df.shape}, test shape: {test_df.shape}")
print(f"Target column: {target_col}")

y_train = train_df[target_col].values.astype(np.float64)
train_features = train_df.drop(columns=[target_col], errors="ignore").copy()
test_features = test_df.drop(columns=[target_col], errors="ignore").copy()

feature_cols = [c for c in train_features.columns if c in test_features.columns]
train_features = train_features[feature_cols].copy()
test_features = test_features[feature_cols].copy()


# =========================================================================
# 2. Feature Engineering Pipeline
# =========================================================================
def parse_and_extract_datetime_features(df_in):
    """Extract temporal, cyclical, and fiscal features from all date/time columns."""
    df = df_in.copy()
    date_cols = []

    for col in df.columns:
        if df[col].dtype == "object" or "datetime" in str(df[col].dtype):
            sample_non_null = df[col].dropna().astype(str)
            if len(sample_non_null) > 0 and (
                "date" in col.lower()
                or "year" in col.lower()
                or "month" in col.lower()
                or sample_non_null.str.contains(r"^\d{4}[-/]\d{2}[-/]\d{2}").sum()
                > len(sample_non_null) * 0.5
            ):
                date_cols.append(col)

    extracted_date_series = {}
    for col in date_cols:
        parsed = pd.to_datetime(df[col], errors="coerce")
        if parsed.notna().sum() > 0:
            extracted_date_series[col] = parsed
            prefix = f"dt_{col}_"
            df[f"{prefix}year"] = parsed.dt.year.fillna(-1).astype(int)
            df[f"{prefix}quarter"] = parsed.dt.quarter.fillna(-1).astype(int)
            df[f"{prefix}month"] = parsed.dt.month.fillna(-1).astype(int)
            df[f"{prefix}day"] = parsed.dt.day.fillna(-1).astype(int)
            df[f"{prefix}dayofweek"] = parsed.dt.dayofweek.fillna(-1).astype(int)
            df[f"{prefix}dayofyear"] = parsed.dt.dayofyear.fillna(-1).astype(int)
            df[f"{prefix}is_quarter_end"] = parsed.dt.is_quarter_end.fillna(
                False
            ).astype(int)
            df[f"{prefix}is_year_end"] = parsed.dt.is_year_end.fillna(False).astype(int)

            month_norm = 2 * np.pi * (parsed.dt.month.fillna(1) - 1) / 12.0
            df[f"{prefix}month_sin"] = np.sin(month_norm)
            df[f"{prefix}month_cos"] = np.cos(month_norm)

    date_col_names = list(extracted_date_series.keys())
    for i in range(len(date_col_names)):
        for j in range(i + 1, len(date_col_names)):
            d1, d2 = date_col_names[i], date_col_names[j]
            delta = (
                extracted_date_series[d1] - extracted_date_series[d2]
            ).dt.total_seconds() / (24 * 3600.0)
            df[f"dt_delta_{d1}_minus_{d2}"] = delta.fillna(0.0)

    df = df.drop(columns=date_cols, errors="ignore")
    return df


train_fe = parse_and_extract_datetime_features(train_features)
test_fe = parse_and_extract_datetime_features(test_features)

cat_cols = []
num_cols = []
for c in train_fe.columns:
    if train_fe[c].dtype == "object" or train_fe[c].dtype.name == "category":
        cat_cols.append(c)
    else:
        num_cols.append(c)

# Categorical interaction pairs and cardinality ratios
interaction_pairs = []
if len(cat_cols) >= 2:
    base_cats = list(cat_cols)
    for i in range(len(base_cats)):
        for j in range(i + 1, len(base_cats)):
            c1, c2 = base_cats[i], base_cats[j]
            pair_name = f"inter_{c1}_{c2}"
            train_fe[pair_name] = (
                train_fe[c1].astype(str) + "__" + train_fe[c2].astype(str)
            )
            test_fe[pair_name] = (
                test_fe[c1].astype(str) + "__" + test_fe[c2].astype(str)
            )
            interaction_pairs.append(pair_name)
            cat_cols.append(pair_name)

# Frequency & Cumulative transaction frequency encodings
for c in cat_cols:
    freq_map = train_fe[c].astype(str).value_counts(normalize=True).to_dict()
    train_fe[f"freq_{c}"] = (
        train_fe[c].astype(str).map(freq_map).fillna(0.0).astype(np.float32)
    )
    test_fe[f"freq_{c}"] = (
        test_fe[c].astype(str).map(freq_map).fillna(0.0).astype(np.float32)
    )
    # Cardinality ratio
    n_uniq = float(train_fe[c].nunique())
    train_fe[f"card_ratio_{c}"] = np.float32(n_uniq / max(len(train_fe), 1))
    test_fe[f"card_ratio_{c}"] = np.float32(n_uniq / max(len(train_fe), 1))
    # Cumulative count
    train_fe[f"cumcount_{c}"] = (
        train_fe.groupby(c).cumcount().astype(np.float32)
    )
    test_fe[f"cumcount_{c}"] = (
        test_fe.groupby(c).cumcount().astype(np.float32)
    )

# Clean feature names for tree-based models
clean_col_map = {col: re.sub(r'[\[\]<>,:"\s]', "_", col) for col in train_fe.columns}
train_fe.rename(columns=clean_col_map, inplace=True)
test_fe.rename(columns=clean_col_map, inplace=True)
cat_cols = [clean_col_map[c] for c in cat_cols]

# Category dtype encoding
for c in cat_cols:
    train_fe[c] = train_fe[c].astype(str).fillna("missing")
    test_fe[c] = test_fe[c].astype(str).fillna("missing")
    all_cats = pd.Series(
        list(train_fe[c].unique()) + list(test_fe[c].unique()) + ["missing"]
    ).unique()
    train_fe[c] = pd.Categorical(train_fe[c], categories=all_cats)
    test_fe[c] = pd.Categorical(test_fe[c], categories=all_cats)

# 5-fold cross-validation setup & Leak-free Multi-statistic Out-of-fold target encoding
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
fold_indices = list(kf.split(train_fe, y_train))

global_mean = float(np.mean(y_train))
global_std = float(np.std(y_train))
global_min = float(np.min(y_train))
global_max = float(np.max(y_train))
smoothing_weight = 10.0

# Select target encoding columns: all primary categoricals and interaction pairs
te_cat_candidates = list(cat_cols)

for c in te_cat_candidates:
    for stat in ["mean", "std", "min", "max"]:
        train_fe[f"te_{stat}_{c}"] = np.nan
        test_fe[f"te_{stat}_{c}"] = 0.0

for fold, (trn_idx, val_idx) in enumerate(fold_indices):
    y_trn = y_train[trn_idx]
    for c in te_cat_candidates:
        trn_series = train_fe[c].iloc[trn_idx].astype(str)
        stats = (
            pd.DataFrame({"cat": trn_series, "target": y_trn})
            .groupby("cat")["target"]
            .agg(["count", "mean", "std", "min", "max"])
        )
        stats["std"] = stats["std"].fillna(global_std)
        stats["smoothed_mean"] = (
            stats["count"] * stats["mean"] + smoothing_weight * global_mean
        ) / (stats["count"] + smoothing_weight)
        stats["smoothed_std"] = (
            stats["count"] * stats["std"] + smoothing_weight * global_std
        ) / (stats["count"] + smoothing_weight)

        mean_map = stats["smoothed_mean"].to_dict()
        std_map = stats["smoothed_std"].to_dict()
        min_map = stats["min"].to_dict()
        max_map = stats["max"].to_dict()

        val_series = train_fe[c].iloc[val_idx].astype(str)
        train_fe.loc[train_fe.index[val_idx], f"te_mean_{c}"] = val_series.map(
            mean_map
        ).fillna(global_mean)
        train_fe.loc[train_fe.index[val_idx], f"te_std_{c}"] = val_series.map(
            std_map
        ).fillna(global_std)
        train_fe.loc[train_fe.index[val_idx], f"te_min_{c}"] = val_series.map(
            min_map
        ).fillna(global_min)
        train_fe.loc[train_fe.index[val_idx], f"te_max_{c}"] = val_series.map(
            max_map
        ).fillna(global_max)

# Whole-train statistics for test set mapping
for c in te_cat_candidates:
    full_series = train_fe[c].astype(str)
    stats = (
        pd.DataFrame({"cat": full_series, "target": y_train})
        .groupby("cat")["target"]
        .agg(["count", "mean", "std", "min", "max"])
    )
    stats["std"] = stats["std"].fillna(global_std)
    stats["smoothed_mean"] = (
        stats["count"] * stats["mean"] + smoothing_weight * global_mean
    ) / (stats["count"] + smoothing_weight)
    stats["smoothed_std"] = (
        stats["count"] * stats["std"] + smoothing_weight * global_std
    ) / (stats["count"] + smoothing_weight)

    test_fe[f"te_mean_{c}"] = test_fe[c].astype(str).map(
        stats["smoothed_mean"].to_dict()
    ).fillna(global_mean)
    test_fe[f"te_std_{c}"] = test_fe[c].astype(str).map(
        stats["smoothed_std"].to_dict()
    ).fillna(global_std)
    test_fe[f"te_min_{c}"] = test_fe[c].astype(str).map(
        stats["min"].to_dict()
    ).fillna(global_min)
    test_fe[f"te_max_{c}"] = test_fe[c].astype(str).map(
        stats["max"].to_dict()
    ).fillna(global_max)

# Impute numeric missing values with median
numeric_final_cols = [c for c in train_fe.columns if c not in cat_cols]
for c in numeric_final_cols:
    fill_val = float(train_fe[c].median() if train_fe[c].notna().sum() > 0 else 0.0)
    train_fe[c] = train_fe[c].fillna(fill_val).astype(np.float32)
    test_fe[c] = test_fe[c].fillna(fill_val).astype(np.float32)


# =========================================================================
# 3. Model Architecture Definitions
# =========================================================================
def build_lgb_deep(random_state=42):
    return LGBMRegressor(
        objective="regression",
        metric="rmse",
        boosting_type="gbdt",
        n_estimators=2500,
        learning_rate=0.025,
        num_leaves=63,
        max_depth=8,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.7,
        reg_alpha=0.5,
        reg_lambda=2.0,
        min_child_samples=15,
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
    )


def build_lgb_shallow(random_state=42):
    return LGBMRegressor(
        objective="regression",
        metric="rmse",
        boosting_type="gbdt",
        n_estimators=2500,
        learning_rate=0.035,
        num_leaves=15,
        max_depth=4,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=0.01,
        reg_lambda=0.5,
        min_child_samples=25,
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
    )


def build_xgb_mid(random_state=42):
    return XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        tree_method="hist",
        enable_categorical=True,
        early_stopping_rounds=50,
        n_estimators=2500,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.75,
        reg_alpha=0.2,
        reg_lambda=1.5,
        min_child_weight=4,
        random_state=random_state,
        n_jobs=-1,
    )


def build_xgb_shallow(random_state=42):
    return XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        tree_method="hist",
        enable_categorical=True,
        early_stopping_rounds=50,
        n_estimators=2500,
        learning_rate=0.035,
        max_depth=4,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.05,
        reg_lambda=0.5,
        min_child_weight=2,
        random_state=random_state,
        n_jobs=-1,
    )


def build_cat_deep(random_state=42):
    return CatBoostRegressor(
        loss_function="RMSE",
        eval_metric="RMSE",
        iterations=2500,
        learning_rate=0.03,
        depth=8,
        l2_leaf_reg=5.0,
        random_seed=random_state,
        verbose=False,
        thread_count=-1,
    )


def build_cat_shallow(random_state=42):
    return CatBoostRegressor(
        loss_function="RMSE",
        eval_metric="RMSE",
        iterations=2500,
        learning_rate=0.04,
        depth=4,
        l2_leaf_reg=2.0,
        random_seed=random_state,
        verbose=False,
        thread_count=-1,
    )


class TabularResidualBlock(nn.Module):
    def __init__(self, hidden_dim, dropout_rate=0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x):
        return x + self.block(x)


class TabularResNet(nn.Module):
    def __init__(
        self, input_dim, hidden_dim=256, num_blocks=3, dropout_rate=0.2, output_dim=1
    ):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
        )
        self.res_blocks = nn.ModuleList(
            [
                TabularResidualBlock(hidden_dim, dropout_rate=dropout_rate)
                for _ in range(num_blocks)
            ]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout_rate / 2.0),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, x):
        h = self.input_layer(x)
        for block in self.res_blocks:
            h = block(h)
        return self.head(h).squeeze(-1)


# =========================================================================
# 4. Training, Evaluation & Ensembling
# =========================================================================
def calculate_rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


n_train = len(train_fe)
cat_features_list = [c for c in cat_cols if c in train_fe.columns]

for col in cat_features_list:
    train_fe[col] = train_fe[col].astype("category")
    test_fe[col] = test_fe[col].astype("category")

model_factories = {
    "cat_deep": build_cat_deep,
    "cat_shallow": build_cat_shallow,
    "lgb_deep": build_lgb_deep,
    "lgb_shallow": build_lgb_shallow,
    "xgb_mid": build_xgb_mid,
    "xgb_shallow": build_xgb_shallow,
}

model_names = list(model_factories.keys())
oof_preds = {name: np.zeros(n_train, dtype=np.float64) for name in model_names}
test_preds_folds = {name: [] for name in model_names}

print(f"Training {len(model_names)} diverse models across {len(fold_indices)} folds...")

for fold, (trn_idx, val_idx) in enumerate(fold_indices):
    X_tr, y_tr = train_fe.iloc[trn_idx], y_train[trn_idx]
    X_va, y_va = train_fe.iloc[val_idx], y_train[val_idx]
    fold_scores = {}

    for name, factory in model_factories.items():
        seed = 42 + fold * 10
        model = factory(random_state=seed)

        if name.startswith("lgb"):
            lgb_callbacks = [
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=0),
            ]
            model.fit(
                X_tr,
                y_tr,
                eval_set=[(X_va, y_va)],
                categorical_feature=cat_features_list,
                callbacks=lgb_callbacks,
            )
        elif name.startswith("xgb"):
            model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        elif name.startswith("cat"):
            model.fit(
                X_tr,
                y_tr,
                eval_set=[(X_va, y_va)],
                cat_features=cat_features_list,
                early_stopping_rounds=50,
                verbose=False,
            )

        val_pred = model.predict(X_va)
        oof_preds[name][val_idx] = val_pred
        test_preds_folds[name].append(model.predict(test_fe))
        fold_scores[name] = calculate_rmse(y_va, val_pred)

    score_str = " | ".join([f"{k}: {v:.4f}" for k, v in fold_scores.items()])
    print(f"Fold {fold+1}/{len(fold_indices)} - {score_str}")

for name in model_names:
    print(f"Overall OOF - {name} RMSE: {calculate_rmse(y_train, oof_preds[name]):.4f}")

# Constrained non-negative L2-regularized blending optimization
oof_matrix = np.column_stack([oof_preds[name] for name in model_names])


def blend_loss(w):
    weights = np.array(w)
    weights = weights / (np.sum(weights) + 1e-12)
    pred = np.dot(oof_matrix, weights)
    l2_reg = 1e-4 * np.sum(weights**2)
    return calculate_rmse(y_train, pred) + l2_reg


init_w = np.ones(len(model_names)) / len(model_names)
bounds = [(0.0, 1.0) for _ in range(len(model_names))]
constraints = {"type": "eq", "fun": lambda w: 1.0 - np.sum(w)}

opt_res = minimize(
    blend_loss,
    init_w,
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
)
opt_w = (
    opt_res.x / np.sum(opt_res.x)
    if opt_res.success
    else init_w
)

oof_blend = np.dot(oof_matrix, opt_w)
clip_non_negative = False
if np.min(y_train) >= 0:
    clipped_oof = np.clip(oof_blend, a_min=0.0, a_max=None)
    if calculate_rmse(y_train, clipped_oof) <= calculate_rmse(y_train, oof_blend):
        oof_blend = clipped_oof
        clip_non_negative = True

final_rmse = calculate_rmse(y_train, oof_blend)
weight_summary = ", ".join([f"{name}={opt_w[i]:.3f}" for i, name in enumerate(model_names)])
print(f"Optimal Blend Weights: {weight_summary}")
print(f"Blended Ensemble OOF RMSE: {final_rmse:.4f}")

# =========================================================================
# 5. Final Test Predictions & Submission Generation
# =========================================================================
test_matrix = np.column_stack([np.mean(test_preds_folds[name], axis=0) for name in model_names])
final_test_preds = np.dot(test_matrix, opt_w)

if clip_non_negative:
    final_test_preds = np.clip(final_test_preds, a_min=0.0, a_max=None)

submission_path = "./submission/submission.csv"
submission_df = pd.DataFrame(
    {"id": np.arange(len(final_test_preds)), "Amount in USD": final_test_preds}
)
submission_df.to_csv(submission_path, index=False)

print(f"Submission successfully saved to {submission_path}")
print(f"Final Validation Score: {final_rmse}")
