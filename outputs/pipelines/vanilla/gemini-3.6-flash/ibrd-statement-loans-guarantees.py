import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

# 1. Load data
train_path = "/path/to/user/workspace/ds_failures/datasets/strable/ibrd-statement-loans-guarantees/data/train.parquet"
test_path = "/path/to/user/workspace/ds_failures/datasets/strable/ibrd-statement-loans-guarantees/data/test.parquet"

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = "Original Principal Amount (US$)"

# Store target
y_train = train_df[target_col].values
X_train_df = train_df.drop(columns=[target_col])
X_test_df = test_df.copy()

n_train = len(train_df)
combined = pd.concat([X_train_df, X_test_df], axis=0, ignore_index=True)

# 2. Feature Engineering

# Datetime processing
cols_to_check = list(combined.columns)
for col in cols_to_check:
    if combined[col].dtype == 'object' or pd.api.types.is_datetime64_any_dtype(combined[col]):
        try:
            converted = pd.to_datetime(combined[col], errors='coerce')
            if converted.notna().sum() > 0.3 * len(combined):
                combined[col + '_year'] = converted.dt.year
                combined[col + '_month'] = converted.dt.month
                combined[col + '_day'] = converted.dt.day
                combined[col + '_dayofweek'] = converted.dt.dayofweek
                timestamps = converted.view('int64') // 10**9
                timestamps[converted.isna()] = np.nan
                combined[col + '_timestamp'] = timestamps
        except Exception:
            pass

# Financial amount columns row-wise features
num_cols = combined.select_dtypes(include=[np.number]).columns.tolist()
amount_cols = [c for c in num_cols if any(k in c.lower() for k in ['amount', 'us$', 'repaid', 'disbursed', 'due', 'held', 'cancelled', 'obligation'])]

if len(amount_cols) > 1:
    combined['amount_sum'] = combined[amount_cols].sum(axis=1)
    combined['amount_mean'] = combined[amount_cols].mean(axis=1)
    combined['amount_std'] = combined[amount_cols].std(axis=1)
    combined['amount_min'] = combined[amount_cols].min(axis=1)
    combined['amount_max'] = combined[amount_cols].max(axis=1)
    combined['amount_count_nonzero'] = (combined[amount_cols] != 0).sum(axis=1)

# Handle categorical columns
cat_cols = []
for col in combined.columns:
    if combined[col].dtype == 'object' or pd.api.types.is_datetime64_any_dtype(combined[col]) or combined[col].dtype.name == 'category':
        cat_cols.append(col)
        combined[col] = combined[col].astype(str)
        freq = combined[col].value_counts().to_dict()
        combined[col + '_freq'] = combined[col].map(freq)
        combined[col] = combined[col].astype('category')

# Separate back to train and test
X_train = combined.iloc[:n_train].copy()
X_test = combined.iloc[n_train:].copy()

# Prepare for cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

lgb_preds = np.zeros(len(X_test))
xgb_preds = np.zeros(len(X_test))
cat_preds = np.zeros(len(X_test))

# 3. Model Training & Cross Validation
for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
    X_tr, y_tr = X_train.iloc[train_idx], y_train[train_idx]
    X_va, y_va = X_train.iloc[val_idx], y_train[val_idx]

    # LightGBM
    model_lgb = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.04,
        num_leaves=63,
        random_state=42 + fold,
        n_jobs=-1,
        verbose=-1
    )
    model_lgb.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )
    lgb_preds += model_lgb.predict(X_test) / 5.0

    # XGBoost
    model_xgb = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.04,
        max_depth=7,
        random_state=42 + fold,
        n_jobs=-1,
        enable_categorical=True,
        tree_method='hist',
        early_stopping_rounds=50
    )
    model_xgb.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        verbose=False
    )
    xgb_preds += model_xgb.predict(X_test) / 5.0

    # CatBoost
    X_tr_cb = X_tr.copy()
    X_va_cb = X_va.copy()
    X_te_cb = X_test.copy()
    for c in cat_cols:
        X_tr_cb[c] = X_tr_cb[c].astype(str)
        X_va_cb[c] = X_va_cb[c].astype(str)
        X_te_cb[c] = X_te_cb[c].astype(str)

    model_cat = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.04,
        depth=7,
        random_seed=42 + fold,
        cat_features=cat_cols,
        verbose=0
    )
    model_cat.fit(
        X_tr_cb, y_tr,
        eval_set=(X_va_cb, y_va),
        early_stopping_rounds=50
    )
    cat_preds += model_cat.predict(X_te_cb) / 5.0

# 4. Ensemble and Output
final_preds = 0.4 * lgb_preds + 0.3 * xgb_preds + 0.3 * cat_preds

if (y_train >= 0).all():
    final_preds = np.clip(final_preds, 0, None)

submission = pd.DataFrame({
    "id": np.arange(len(final_preds)),
    target_col: final_preds
})

submission.to_csv("predictions.csv", index=False)