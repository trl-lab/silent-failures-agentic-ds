import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error
from xgboost import XGBRegressor
import lightgbm as lgb
from catboost import CatBoostRegressor

# 1. Load Datasets
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "New Commitments (US$)"

# Combine for uniform feature engineering
train_df["is_train"] = 1
test_df["is_train"] = 0
if target_col not in test_df.columns:
    test_df[target_col] = np.nan

df = pd.concat([train_df, test_df], ignore_index=True)

# 2. Feature Engineering
date_cols = []
cat_cols = []
num_cols = []

for col in df.columns:
    if col in [target_col, "is_train"]:
        continue
    if (
        pd.api.types.is_datetime64_any_dtype(df[col])
        or "date" in col.lower()
        or "year" in col.lower()
    ):
        try:
            parsed = pd.to_datetime(df[col], errors="coerce")
            if parsed.notna().sum() > 0.5 * len(df):
                df[col] = parsed
                date_cols.append(col)
                continue
        except Exception:
            pass
    if pd.api.types.is_numeric_dtype(df[col]):
        num_cols.append(col)
    else:
        cat_cols.append(col)

# Temporal Features
for col in date_cols:
    df[f"{col}_year"] = df[col].dt.year
    df[f"{col}_month"] = df[col].dt.month
    df[f"{col}_quarter"] = df[col].dt.quarter
    df[f"{col}_day"] = df[col].dt.day
    df[f"{col}_dayofweek"] = df[col].dt.dayofweek
    min_date = df[col].min()
    if pd.notna(min_date):
        df[f"{col}_days_since"] = (df[col] - min_date).dt.days
    df.drop(columns=[col], inplace=True)

# Categorical Features
for col in cat_cols:
    df[col] = df[col].astype(str).fillna("missing")
    freq = df[col].value_counts()
    df[f"{col}_freq"] = df[col].map(freq)
    df[col] = df[col].astype("category").cat.codes

# 3. Train / Test Split and Target Transformation
train_clean = df[df["is_train"] == 1].drop(columns=["is_train"]).reset_index(drop=True)
test_clean = df[df["is_train"] == 0].drop(columns=["is_train"]).reset_index(drop=True)

X = train_clean.drop(columns=[target_col])
y = train_clean[target_col].values
X_test = test_clean.drop(columns=[target_col])

min_y = np.nanmin(y)
y_offset = abs(min_y) + 1.0 if min_y < 0 else 0.0
y_trans = np.log1p(y + y_offset)

# 4. Cross-Validation and Ensemble Model Training
kf = KFold(n_splits=5, shuffle=True, random_state=42)

oof_xgb = np.zeros(len(train_clean))
oof_lgb = np.zeros(len(train_clean))
oof_cb = np.zeros(len(train_clean))

test_xgb = np.zeros(len(test_clean))
test_lgb = np.zeros(len(test_clean))
test_cb = np.zeros(len(test_clean))

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y_trans)):
    X_tr, y_tr = X.iloc[train_idx], y_trans[train_idx]
    X_va, y_va = X.iloc[val_idx], y_trans[val_idx]

    # XGBoost
    xgb_model = XGBRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        n_jobs=-1,
        early_stopping_rounds=50,
    )
    xgb_model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

    oof_xgb[val_idx] = xgb_model.predict(X_va)
    test_xgb += xgb_model.predict(X_test) / kf.n_splits

    # LightGBM
    lgb_model = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        n_jobs=-1,
        verbose=-1,
    )
    lgb_model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )

    oof_lgb[val_idx] = lgb_model.predict(X_va)
    test_lgb += lgb_model.predict(X_test) / kf.n_splits

    # CatBoost
    cb_model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.03,
        depth=6,
        random_seed=42 + fold,
        verbose=False,
        early_stopping_rounds=50,
    )
    cb_model.fit(X_tr, y_tr, eval_set=(X_va, y_va))

    oof_cb[val_idx] = cb_model.predict(X_va)
    test_cb += cb_model.predict(X_test) / kf.n_splits

# 5. Blend Predictions and Evaluate
oof_blend = 0.50 * oof_xgb + 0.25 * oof_lgb + 0.25 * oof_cb
test_blend = 0.50 * test_xgb + 0.25 * test_lgb + 0.25 * test_cb

oof_preds_orig = np.expm1(oof_blend) - y_offset
oof_preds_orig = np.clip(oof_preds_orig, 0, None)

val_rmse = root_mean_squared_error(y, oof_preds_orig)
print(f"Validation RMSE: {val_rmse:.4f}")

# 6. Save Test Submission
test_preds_orig = np.expm1(test_blend) - y_offset
test_preds_orig = np.clip(test_preds_orig, 0, None)

os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame(
    {"id": np.arange(len(test_preds_orig)), target_col: test_preds_orig}
)
submission.to_csv("./working/submission.csv", index=False)
