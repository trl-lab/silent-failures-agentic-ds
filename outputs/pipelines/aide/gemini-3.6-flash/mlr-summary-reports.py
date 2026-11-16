import os
import re
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

warnings.filterwarnings("ignore")

# 1. Load data
train_path = "./input/train.parquet"
test_path = "./input/test.parquet"

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = "Adjusted MLR"


# Function to parse string numbers/percentages/currency
def parse_numeric_series(s):
    if s.dtype in [np.float64, np.float32, np.int64, np.int32]:
        return s, False

    s_str = s.astype(str).str.strip()
    is_pct = s_str.str.contains("%", na=False).mean() > 0.3

    clean_s = (
        s_str.str.replace("$", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.replace("(", "-", regex=False)
        .str.replace(")", "", regex=False)
        .str.replace("N/A", "", regex=False)
        .str.replace("nan", "", regex=False)
        .str.replace("None", "", regex=False)
    )

    num_s = pd.to_numeric(clean_s, errors="coerce")
    valid_ratio = num_s.notna().sum() / max(len(s), 1)

    if valid_ratio > 0.4:
        if is_pct and num_s.dropna().mean() > 1.5:
            num_s = num_s / 100.0
        return num_s, True
    return s, False


# Clean Target Column
target_raw = train_df[target_col]
target_clean, _ = parse_numeric_series(target_raw)
if target_clean.dropna().mean() > 1.5:
    target_clean = target_clean / 100.0

train_df[target_col] = target_clean
train_df = train_df.dropna(subset=[target_col]).reset_index(drop=True)
y = train_df[target_col].values

# Drop target column from features
X_train_raw = train_df.drop(columns=[target_col])
X_test_raw = test_df.copy()
if target_col in X_test_raw.columns:
    X_test_raw = X_test_raw.drop(columns=[target_col])

# Combine for uniform feature engineering
combined = pd.concat([X_train_raw, X_test_raw], axis=0, ignore_index=True)

num_cols = []
cat_cols = []

for col in combined.columns:
    parsed_s, is_num = parse_numeric_series(combined[col])
    if is_num:
        combined[col] = parsed_s
        num_cols.append(col)
    else:
        if pd.api.types.is_numeric_dtype(combined[col]):
            num_cols.append(col)
        else:
            cat_cols.append(col)

# Format categorical columns
for col in cat_cols:
    combined[col] = combined[col].fillna("Missing").astype(str)
    freq = combined[col].value_counts()
    combined[f"{col}_freq"] = combined[col].map(freq)
    combined[col] = combined[col].astype("category").cat.codes

# Logarithmic transformations for non-negative numerical variables
for col in num_cols:
    valid_vals = combined[col].dropna()
    if len(valid_vals) > 0 and (valid_vals >= 0).all():
        combined[f"{col}_log1p"] = np.log1p(combined[col])

# Complete pairwise interaction features between all numerical variables
if len(num_cols) >= 2:
    for i in range(len(num_cols)):
        for j in range(i + 1, len(num_cols)):
            c1, c2 = num_cols[i], num_cols[j]
            combined[f"{c1}_ratio_{c2}"] = combined[c1] / (combined[c2].abs() + 1e-5)
            combined[f"{c1}_diff_{c2}"] = combined[c1] - combined[c2]
            combined[f"{c1}_mult_{c2}"] = combined[c1] * combined[c2]

# Row-wise summary statistics
numeric_mat = combined[num_cols].values
combined["row_mean"] = np.nanmean(numeric_mat, axis=1)
combined["row_std"] = np.nanstd(numeric_mat, axis=1)
combined["row_min"] = np.nanmin(numeric_mat, axis=1)
combined["row_max"] = np.nanmax(numeric_mat, axis=1)
combined["row_sum"] = np.nansum(numeric_mat, axis=1)

# Full group aggregations across all categorical and numerical columns
for c_col in cat_cols:
    for n_col in num_cols:
        grouped = combined.groupby(c_col)[n_col]
        combined[f"{c_col}_{n_col}_grp_mean"] = grouped.transform("mean")
        combined[f"{c_col}_{n_col}_grp_std"] = grouped.transform("std")

# Clean column names to eliminate special JSON characters for LightGBM/XGBoost
clean_feature_names = []
for idx, col in enumerate(combined.columns):
    sanitized = re.sub(r"[^\w]", "_", str(col))
    clean_feature_names.append(f"f_{idx}_{sanitized}")
combined.columns = clean_feature_names

# Separate train and test
n_train = len(train_df)
X_train = combined.iloc[:n_train].copy().reset_index(drop=True)
X_test = combined.iloc[n_train:].copy().reset_index(drop=True)

# Replace infinite values and NaNs
X_train = X_train.replace([np.inf, -np.inf], np.nan)
X_test = X_test.replace([np.inf, -np.inf], np.nan)

for col in X_train.columns:
    med = X_train[col].median()
    if pd.isna(med):
        med = 0
    X_train[col] = X_train[col].fillna(med)
    X_test[col] = X_test[col].fillna(med)

# 5-Fold Cross Validation Setup
kf = KFold(n_splits=5, shuffle=True, random_state=42)

oof_xgb = np.zeros(n_train)
oof_lgb = np.zeros(n_train)
oof_cat = np.zeros(n_train)

test_preds_xgb = np.zeros(len(X_test))
test_preds_lgb = np.zeros(len(X_test))
test_preds_cat = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y)):
    X_tr, y_tr = X_train.iloc[train_idx], y[train_idx]
    X_va, y_va = X_train.iloc[val_idx], y[val_idx]

    # XGBoost
    model_xgb = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        early_stopping_rounds=50,
        n_jobs=-1,
    )
    model_xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    oof_xgb[val_idx] = model_xgb.predict(X_va)
    test_preds_xgb += model_xgb.predict(X_test) / 5.0

    # LightGBM
    model_lgb = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        verbose=-1,
        n_jobs=-1,
    )
    model_lgb.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    oof_lgb[val_idx] = model_lgb.predict(X_va)
    test_preds_lgb += model_lgb.predict(X_test) / 5.0

    # CatBoost
    model_cat = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.03,
        depth=5,
        early_stopping_rounds=50,
        random_seed=42 + fold,
        verbose=0,
    )
    model_cat.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=False)
    oof_cat[val_idx] = model_cat.predict(X_va)
    test_preds_cat += model_cat.predict(X_test) / 5.0

# Stacking Meta-Learner via Ridge Regression
oof_meta = np.column_stack([oof_xgb, oof_lgb, oof_cat])
test_meta = np.column_stack([test_preds_xgb, test_preds_lgb, test_preds_cat])

scaler = StandardScaler()
oof_meta_scaled = scaler.fit_transform(oof_meta)
test_meta_scaled = scaler.transform(test_meta)

meta_oof = np.zeros(n_train)
meta_test_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(kf.split(oof_meta_scaled, y)):
    X_tr, y_tr = oof_meta_scaled[train_idx], y[train_idx]
    X_va, y_va = oof_meta_scaled[val_idx], y[val_idx]

    meta_model = Ridge(alpha=1.0)
    meta_model.fit(X_tr, y_tr)
    meta_oof[val_idx] = meta_model.predict(X_va)
    meta_test_preds += meta_model.predict(test_meta_scaled) / 5.0

# Evaluate Performance
final_rmse = root_mean_squared_error(y, meta_oof)
print(f"Validation RMSE: {final_rmse:.5f}")

# Save Submission
os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame(
    {"id": np.arange(len(meta_test_preds)), "Adjusted MLR": meta_test_preds}
)
submission.to_csv("./working/submission.csv", index=False)
