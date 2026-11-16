import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error
import lightgbm as lgb
import warnings

warnings.filterwarnings("ignore")

# 1. Load datasets
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

os.makedirs("./working", exist_ok=True)

target_col = "Amount"
y = train[target_col].values.astype(np.float64)

X_train_raw = train.drop(columns=[target_col]).copy()
X_test_raw = test.copy()

# Combine features for consistent encoding
df_all = pd.concat([X_train_raw, X_test_raw], axis=0, ignore_index=True)

# 2. Feature Engineering & Preprocessing
for col in df_all.columns:
    if df_all[col].dtype == "object":
        try:
            sample = df_all[col].dropna().iloc[:50]
            if len(sample) > 0:
                parsed = pd.to_datetime(sample, errors="coerce")
                if parsed.notna().mean() > 0.8:
                    df_all[col] = pd.to_datetime(df_all[col], errors="coerce")
        except Exception:
            pass

datetime_cols = [
    col for col in df_all.columns if pd.api.types.is_datetime64_any_dtype(df_all[col])
]
for col in datetime_cols:
    df_all[f"{col}_year"] = df_all[col].dt.year
    df_all[f"{col}_month"] = df_all[col].dt.month
    df_all[f"{col}_day"] = df_all[col].dt.day
    df_all[f"{col}_dayofweek"] = df_all[col].dt.dayofweek
    df_all[f"{col}_quarter"] = df_all[col].dt.quarter
    df_all.drop(columns=[col], inplace=True)

cat_cols = [
    col
    for col in df_all.columns
    if df_all[col].dtype == "object" or df_all[col].dtype.name == "category"
]

for col in cat_cols:
    df_all[col] = df_all[col].fillna("Missing").astype(str)
    freq = df_all[col].value_counts()
    df_all[f"{col}_freq"] = df_all[col].map(freq)
    df_all[col] = df_all[col].astype("category")

n_train = len(train)
X_train = df_all.iloc[:n_train].reset_index(drop=True)
X_test = df_all.iloc[n_train:].reset_index(drop=True)

# 3. Cross-Validation Setup & LightGBM Training
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(n_train)
test_preds = np.zeros(len(test))

lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "random_state": 42,
    "n_estimators": 1500,
    "n_jobs": -1,
    "verbose": -1,
}

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y)):
    X_tr, y_tr = X_train.iloc[train_idx], y[train_idx]
    X_va, y_va = X_train.iloc[val_idx], y[val_idx]

    model = lgb.LGBMRegressor(**lgb_params)
    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
    )

    oof_preds[val_idx] = model.predict(X_va)
    test_preds += model.predict(X_test) / kf.n_splits

if np.min(y) >= 0:
    oof_preds = np.clip(oof_preds, 0, None)
    test_preds = np.clip(test_preds, 0, None)

cv_rmse = root_mean_squared_error(y, oof_preds)
print(f"5-Fold Cross-Validation RMSE: {cv_rmse:.4f}")

# 4. Save Test Predictions
submission = pd.DataFrame({"id": np.arange(len(test)), "Amount": test_preds})
submission.to_csv("./working/submission.csv", index=False)
print("Submission successfully saved to ./working/submission.csv")
