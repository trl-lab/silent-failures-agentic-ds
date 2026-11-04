import os
import warnings
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")

# Load train and test data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "NET_GEN"

# Clean target NaNs if present
train_df = train_df.dropna(subset=[target_col]).reset_index(drop=True)

y_train = train_df[target_col].values
X_train_raw = train_df.drop(columns=[target_col])
X_test_raw = test_df.copy()

# Save ID column if present, then drop from input features
train_id = X_train_raw["id"] if "id" in X_train_raw.columns else None
test_id = X_test_raw["id"] if "id" in X_test_raw.columns else np.arange(len(X_test_raw))

if "id" in X_train_raw.columns:
    X_train_raw = X_train_raw.drop(columns=["id"])
if "id" in X_test_raw.columns:
    X_test_raw = X_test_raw.drop(columns=["id"])

# Combine for consistent preprocessing
df_all = pd.concat([X_train_raw, X_test_raw], axis=0).reset_index(drop=True)

# Process datetime features
for col in list(df_all.columns):
    if pd.api.types.is_datetime64_any_dtype(df_all[col]):
        df_all[f"{col}_year"] = df_all[col].dt.year
        df_all[f"{col}_month"] = df_all[col].dt.month
        df_all[f"{col}_day"] = df_all[col].dt.day
        df_all[f"{col}_dayofweek"] = df_all[col].dt.dayofweek
        df_all = df_all.drop(columns=[col])
    elif df_all[col].dtype == "object":
        parsed = pd.to_datetime(df_all[col], errors="coerce")
        if parsed.notna().sum() > 0.5 * len(df_all):
            df_all[f"{col}_year"] = parsed.dt.year
            df_all[f"{col}_month"] = parsed.dt.month
            df_all[f"{col}_day"] = parsed.dt.day
            df_all = df_all.drop(columns=[col])

# Identify categorical and numerical columns
raw_cat_cols = df_all.select_dtypes(include=["object", "category"]).columns.tolist()
num_cols = df_all.select_dtypes(include=[np.number]).columns.tolist()

# 1. Frequency Encoding for Categorical Columns
for col in raw_cat_cols:
    freq = df_all[col].astype(str).value_counts(normalize=True, dropna=False).to_dict()
    df_all[f"{col}_freq"] = df_all[col].astype(str).map(freq).astype(np.float32)

# 2. Comprehensive Group Aggregations + Relative Deviation & Z-score Features
for cat_col in raw_cat_cols:
    n_unique = df_all[cat_col].nunique(dropna=True)
    if 2 <= n_unique <= 500:
        for num_col in num_cols:
            grouped = df_all.groupby(cat_col, observed=False)[num_col]
            mean_s = grouped.transform("mean").astype(np.float32)
            std_s = grouped.transform("std").astype(np.float32)
            min_s = grouped.transform("min").astype(np.float32)
            max_s = grouped.transform("max").astype(np.float32)

            df_all[f"{cat_col}_{num_col}_mean"] = mean_s
            df_all[f"{cat_col}_{num_col}_std"] = std_s
            df_all[f"{cat_col}_{num_col}_min"] = min_s
            df_all[f"{cat_col}_{num_col}_max"] = max_s

            # Calculate relative difference and z-score within group
            diff_s = (df_all[num_col] - mean_s).astype(np.float32)
            zscore_s = (diff_s / (std_s + 1e-5)).astype(np.float32)

            df_all[f"{cat_col}_{num_col}_diff_mean"] = diff_s
            df_all[f"{cat_col}_{num_col}_zscore"] = zscore_s

# Cast raw categorical columns to pandas category type for LightGBM native support
for col in raw_cat_cols:
    df_all[col] = df_all[col].astype("category")

n_train = len(train_df)
X_train = df_all.iloc[:n_train].reset_index(drop=True)
X_test = df_all.iloc[n_train:].reset_index(drop=True)

cat_features = X_train.select_dtypes(include=["category"]).columns.tolist()

# 5-Fold Cross Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(X_train))
test_preds = np.zeros(len(X_test))

lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "n_estimators": 3000,
    "learning_rate": 0.03,
    "num_leaves": 31,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
    X_tr, y_tr = X_train.iloc[train_idx], y_train[train_idx]
    X_va, y_va = X_train.iloc[val_idx], y_train[val_idx]

    model = lgb.LGBMRegressor(**lgb_params)
    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        categorical_feature=cat_features,
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )

    oof_preds[val_idx] = model.predict(X_va)
    test_preds += model.predict(X_test) / kf.n_splits

cv_rmse = np.sqrt(mean_squared_error(y_train, oof_preds))
print(f"5-Fold Cross-Validation RMSE: {cv_rmse:.5f}")

# Save submission
os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame({"id": test_id, "NET_GEN": test_preds})
submission.to_csv("./working/submission.csv", index=False)
