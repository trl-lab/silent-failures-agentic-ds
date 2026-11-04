import os
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import lightgbm as lgb
from catboost import CatBoostRegressor
import xgboost as xgb

warnings.filterwarnings("ignore")

train_path = "/path/to/user/workspace/ds_failures/datasets/strable/managed-care-enrollment/data/train.parquet"
test_path = "/path/to/user/workspace/ds_failures/datasets/strable/managed-care-enrollment/data/test.parquet"

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = "Total Enrollment"

if target_col in train_df.columns:
    train_df = train_df.dropna(subset=[target_col])
    y = train_df[target_col].values
    train_df = train_df.drop(columns=[target_col])

if target_col in test_df.columns:
    test_df = test_df.drop(columns=[target_col])

n_train = len(train_df)
combined = pd.concat([train_df, test_df], axis=0, ignore_index=True)

for col in list(combined.columns):
    if combined[col].dtype == "object":
        cleaned = (
            combined[col]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("$", "", regex=False)
            .str.strip()
        )
        num_s = pd.to_numeric(cleaned, errors="coerce")
        if num_s.notna().sum() > 0.5 * len(combined):
            combined[col] = num_s

for col in list(combined.columns):
    if combined[col].dtype == "object":
        try:
            dt_s = pd.to_datetime(combined[col], errors="coerce")
            if dt_s.notna().sum() > 0.5 * len(combined):
                combined[col + "_year"] = dt_s.dt.year
                combined[col + "_month"] = dt_s.dt.month
                combined[col + "_day"] = dt_s.dt.day
                combined[col + "_dayofweek"] = dt_s.dt.dayofweek
        except Exception:
            pass

for col in list(combined.columns):
    if combined[col].dtype == "object" or isinstance(combined[col].dtype, pd.CategoricalDtype):
        freq = combined[col].value_counts(dropna=False)
        combined[col + "_freq"] = combined[col].map(freq)
        combined[col] = combined[col].astype(str).astype("category").cat.codes

combined.columns = [f"feature_{i}" for i in range(combined.shape[1])]
combined = combined.replace([np.inf, -np.inf], np.nan)

X_train = combined.iloc[:n_train].reset_index(drop=True)
X_test = combined.iloc[n_train:].reset_index(drop=True)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

lgb_preds = np.zeros(len(X_test))
cb_preds = np.zeros(len(X_test))
xgb_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y)):
    X_tr, y_tr = X_train.iloc[train_idx], y[train_idx]
    X_val, y_val = X_train.iloc[val_idx], y[val_idx]

    model_lgb = lgb.LGBMRegressor(
        objective="regression",
        metric="rmse",
        learning_rate=0.05,
        num_leaves=63,
        n_estimators=1000,
        random_state=42 + fold,
        verbose=-1,
        n_jobs=-1
    )
    model_lgb.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    lgb_preds += model_lgb.predict(X_test) / 5.0

    model_cb = CatBoostRegressor(
        loss_function="RMSE",
        eval_metric="RMSE",
        learning_rate=0.05,
        depth=6,
        iterations=1000,
        random_seed=42 + fold,
        verbose=0,
        thread_count=-1
    )
    model_cb.fit(
        X_tr, y_tr,
        eval_set=(X_val, y_val),
        early_stopping_rounds=50,
        verbose=False
    )
    cb_preds += model_cb.predict(X_test) / 5.0

    model_xgb = xgb.XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        n_estimators=1000,
        early_stopping_rounds=50,
        random_state=42 + fold,
        n_jobs=-1
    )
    model_xgb.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    xgb_preds += model_xgb.predict(X_test) / 5.0

final_preds = (lgb_preds + cb_preds + xgb_preds) / 3.0

if np.min(y) >= 0:
    final_preds = np.clip(final_preds, 0, None)

output_df = pd.DataFrame({
    "id": np.arange(len(final_preds)),
    "Total Enrollment": final_preds
})
output_df.to_csv("predictions.csv", index=False)