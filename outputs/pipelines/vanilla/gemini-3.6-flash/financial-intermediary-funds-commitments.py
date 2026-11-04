import os
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import OrdinalEncoder

# Paths
train_path = "/path/to/user/workspace/ds_failures/datasets/strable/financial-intermediary-funds-commitments/data/train.parquet"
test_path = "/path/to/user/workspace/ds_failures/datasets/strable/financial-intermediary-funds-commitments/data/test.parquet"

# Load data
train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = "Amount in USD"

# Extract target and combine for uniform feature engineering
y = train_df[target_col].values
X_train_raw = train_df.drop(columns=[target_col])
X_test_raw = test_df.drop(columns=[target_col], errors='ignore')

n_train = len(train_df)
n_test = len(test_df)

combined = pd.concat([X_train_raw, X_test_raw], axis=0, ignore_index=True)

# Drop redundant index/id columns if present in raw data
drop_cols = [c for c in combined.columns if c.lower() in ['id', 'unnamed: 0']]
if drop_cols:
    combined = combined.drop(columns=drop_cols)

# Extract date components if date features exist
date_features = {}
for col in list(combined.columns):
    if combined[col].dtype == 'object' or 'date' in col.lower() or 'year' in col.lower():
        try:
            dt_col = pd.to_datetime(combined[col], errors='coerce')
            if dt_col.notna().sum() > 0.3 * len(dt_col):
                date_features[col + '_year'] = dt_col.dt.year
                date_features[col + '_month'] = dt_col.dt.month
                date_features[col + '_day'] = dt_col.dt.day
                date_features[col + '_dayofweek'] = dt_col.dt.dayofweek
                date_features[col + '_quarter'] = dt_col.dt.quarter
        except Exception:
            pass

if date_features:
    date_df = pd.DataFrame(date_features, index=combined.index)
    combined = pd.concat([combined, date_df], axis=1)

# Categorical vs Numeric columns
cat_cols = []
num_cols = []
for col in combined.columns:
    if combined[col].dtype == 'object' or combined[col].dtype.name == 'category':
        cat_cols.append(col)
    else:
        num_cols.append(col)

# Categorical Frequency Encoding
for col in cat_cols:
    freq = combined[col].value_counts(dropna=False)
    freq_col = col + '_freq'
    combined[freq_col] = combined[col].map(freq)
    num_cols.append(freq_col)

# Categorical Ordinal Encoding
for col in cat_cols:
    combined[col] = combined[col].astype(str).fillna('missing')
    oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    combined[col] = oe.fit_transform(combined[[col]])

# Fill missing numeric values with median
for col in num_cols:
    median_val = combined[col].median()
    combined[col] = combined[col].fillna(median_val if not np.isnan(median_val) else 0)

# Split back to train and test
X_train = combined.iloc[:n_train].copy()
X_test = combined.iloc[n_train:].copy()

# Ensembling with K-Fold cross-validation
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

lgb_preds = np.zeros(n_test)
cat_preds = np.zeros(n_test)
xgb_preds = np.zeros(n_test)

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y)):
    X_tr, y_tr = X_train.iloc[train_idx], y[train_idx]
    X_va, y_va = X_train.iloc[val_idx], y[val_idx]

    # LightGBM
    model_lgb = lgb.LGBMRegressor(
        n_estimators=1200,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        n_jobs=-1,
        verbose=-1
    )
    model_lgb.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)]
    )
    lgb_preds += model_lgb.predict(X_test) / n_splits

    # CatBoost
    model_cat = CatBoostRegressor(
        iterations=1200,
        learning_rate=0.04,
        depth=6,
        random_seed=42 + fold,
        verbose=0
    )
    model_cat.fit(
        X_tr, y_tr,
        eval_set=(X_va, y_va),
        early_stopping_rounds=100
    )
    cat_preds += model_cat.predict(X_test) / n_splits

    # XGBoost
    model_xgb = xgb.XGBRegressor(
        n_estimators=1200,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        n_jobs=-1,
        early_stopping_rounds=100
    )
    model_xgb.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        verbose=False
    )
    xgb_preds += model_xgb.predict(X_test) / n_splits

# Blend predictions
final_preds = (lgb_preds + cat_preds + xgb_preds) / 3.0

# Create submission dataframe
sub = pd.DataFrame({
    'id': np.arange(n_test),
    'Amount in USD': final_preds
})

sub.to_csv('predictions.csv', index=False)