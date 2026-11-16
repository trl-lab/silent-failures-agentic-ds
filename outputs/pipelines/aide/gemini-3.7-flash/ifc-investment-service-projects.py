import os
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import warnings

warnings.filterwarnings("ignore")

# Set random seeds for reproducibility
np.random.seed(42)

# Load data
train_path = "./input/train.parquet"
test_path = "./input/test.parquet"
train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = "Total IFC investment as approved by Board(Million - USD)"
y = train_df[target_col].copy()

# Combine train and test for consistent feature processing
n_train = len(train_df)
combined = pd.concat([train_df.drop(columns=[target_col]), test_df], ignore_index=True)


# Helper function to clean numeric strings
def clean_numeric_series(series):
    if series.dtype == "object":
        cleaned = (
            series.astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("$", "", regex=False)
            .str.strip()
        )
        numeric = pd.to_numeric(cleaned, errors="coerce")
        if numeric.notna().sum() > 0.4 * len(series):
            return numeric
    return series


# Preprocess numeric and datetime columns
date_cols = []
num_cols = []
cat_cols = []

for col in combined.columns:
    combined[col] = clean_numeric_series(combined[col])

    if combined[col].dtype == "object":
        try:
            parsed_date = pd.to_datetime(combined[col], errors="coerce")
            if parsed_date.notna().sum() > 0.3 * len(combined):
                combined[col] = parsed_date
                date_cols.append(col)
                continue
        except Exception:
            pass
    elif pd.api.types.is_datetime64_any_dtype(combined[col]):
        date_cols.append(col)
        continue

    if pd.api.types.is_numeric_dtype(combined[col]):
        num_cols.append(col)
    else:
        cat_cols.append(col)

# Date features engineering
for col in date_cols:
    combined[f"{col}_year"] = combined[col].dt.year
    combined[f"{col}_month"] = combined[col].dt.month
    combined[f"{col}_day"] = combined[col].dt.day
    combined[f"{col}_dayofweek"] = combined[col].dt.dayofweek
    combined[f"{col}_quarter"] = combined[col].dt.quarter
    combined[f"{col}_timestamp"] = combined[col].astype(np.int64) // 10**9
    combined.loc[combined[col].isna(), f"{col}_timestamp"] = np.nan

# Pairwise date differences
for i in range(len(date_cols)):
    for j in range(i + 1, len(date_cols)):
        col_a, col_b = date_cols[i], date_cols[j]
        diff_col_name = f"diff_days_{col_a}_{col_b}"
        combined[diff_col_name] = (
            combined[col_a] - combined[col_b]
        ).dt.total_seconds() / 86400.0

combined = combined.drop(columns=date_cols)

# Sub-component financial aggregations and interactions
component_cols = [
    c
    for c in num_cols
    if any(
        k in c.lower()
        for k in ["loan", "equity", "guarantee", "risk", "cost", "investment", "amount"]
    )
]

if len(component_cols) > 1:
    combined["sum_components"] = combined[component_cols].fillna(0).sum(axis=1)
    combined["mean_components"] = combined[component_cols].mean(axis=1)
    combined["std_components"] = combined[component_cols].std(axis=1).fillna(0)
    combined["max_components"] = combined[component_cols].max(axis=1).fillna(0)
    combined["min_components"] = combined[component_cols].min(axis=1).fillna(0)
    combined["count_nonzero_components"] = (combined[component_cols].fillna(0) > 0).sum(
        axis=1
    )

    # Log1p transforms for component sums and values
    for c in component_cols:
        combined[f"{c}_log1p"] = np.log1p(np.maximum(0, combined[c].fillna(0)))

    # Key pairwise differences and ratios among top components
    for i in range(min(len(component_cols), 6)):
        for j in range(i + 1, min(len(component_cols), 6)):
            c1, c2 = component_cols[i], component_cols[j]
            combined[f"{c1}_minus_{c2}"] = combined[c1].fillna(0) - combined[c2].fillna(
                0
            )
            combined[f"{c1}_ratio_{c2}"] = combined[c1].fillna(0) / (
                combined[c2].fillna(0).abs() + 1e-5
            )

# Categorical group aggregations on financial components
for cat in cat_cols[:5]:
    if len(component_cols) > 0:
        main_comp = component_cols[0]
        group_mean = combined.groupby(cat)[main_comp].transform("mean")
        group_std = combined.groupby(cat)[main_comp].transform("std").fillna(0)
        combined[f"{cat}_{main_comp}_grp_mean"] = group_mean
        combined[f"{cat}_{main_comp}_grp_std"] = group_std

# Categorical encodings: Frequency encoding + Ordinal label encoding
for col in cat_cols:
    val_counts = combined[col].value_counts(dropna=False)
    combined[f"{col}_freq"] = combined[col].map(val_counts)
    combined[col] = combined[col].astype(str).astype("category").cat.codes

# Clean infinite values and prepare feature datasets
combined = combined.replace([np.inf, -np.inf], np.nan)
X_train = combined.iloc[:n_train].reset_index(drop=True)
X_test = combined.iloc[n_train:].reset_index(drop=True)

# 5-fold cross-validation setup
kf = KFold(n_splits=5, shuffle=True, random_state=42)

oof_xgb = np.zeros(n_train)
test_xgb = np.zeros(len(X_test))

oof_lgb = np.zeros(n_train)
test_lgb = np.zeros(len(X_test))

oof_cat = np.zeros(n_train)
test_cat = np.zeros(len(X_test))

xgb_params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "learning_rate": 0.03,
    "max_depth": 6,
    "subsample": 0.85,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "tree_method": "hist",
    "n_estimators": 2000,
    "n_jobs": -1,
}

lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.03,
    "num_leaves": 31,
    "max_depth": -1,
    "subsample": 0.85,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "n_estimators": 2000,
    "n_jobs": -1,
    "verbose": -1,
}

cat_params = {
    "loss_function": "RMSE",
    "eval_metric": "RMSE",
    "learning_rate": 0.04,
    "depth": 6,
    "iterations": 2000,
    "random_seed": 42,
    "verbose": False,
}

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y)):
    X_tr, y_tr = X_train.iloc[train_idx], y.iloc[train_idx]
    X_va, y_va = X_train.iloc[val_idx], y.iloc[val_idx]

    # 1. XGBoost
    model_xgb = xgb.XGBRegressor(**xgb_params, early_stopping_rounds=50)
    model_xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    oof_xgb[val_idx] = model_xgb.predict(X_va)
    test_xgb += model_xgb.predict(X_test) / kf.n_splits

    # 2. LightGBM
    model_lgb = lgb.LGBMRegressor(**lgb_params)
    model_lgb.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )
    oof_lgb[val_idx] = model_lgb.predict(X_va)
    test_lgb += model_lgb.predict(X_test) / kf.n_splits

    # 3. CatBoost
    model_cat = CatBoostRegressor(**cat_params, early_stopping_rounds=50)
    model_cat.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=False)
    oof_cat[val_idx] = model_cat.predict(X_va)
    test_cat += model_cat.predict(X_test) / kf.n_splits


# Optimal 3-way blend weight calculation
def loss_func(weights):
    w1, w2, w3 = weights
    pred = w1 * oof_xgb + w2 * oof_lgb + w3 * oof_cat
    if (y >= 0).all():
        pred = np.clip(pred, 0, None)
    return np.sqrt(mean_squared_error(y, pred))


res = minimize(
    loss_func,
    [1 / 3, 1 / 3, 1 / 3],
    bounds=[(0, 1), (0, 1), (0, 1)],
    constraints={"type": "eq", "fun": lambda w: sum(w) - 1.0},
)

w_xgb, w_lgb, w_cat = res.x
oof_blend = w_xgb * oof_xgb + w_lgb * oof_lgb + w_cat * oof_cat
test_blend = w_xgb * test_xgb + w_lgb * test_lgb + w_cat * test_cat

if (y >= 0).all():
    oof_blend = np.clip(oof_blend, 0, None)
    test_blend = np.clip(test_blend, 0, None)

cv_rmse = np.sqrt(mean_squared_error(y, oof_blend))
print(f"Validation Metric (5-Fold CV RMSE): {cv_rmse:.4f}")

# Save final predictions
os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame({"id": np.arange(len(test_blend)), target_col: test_blend})
submission.to_csv("./working/submission.csv", index=False)
