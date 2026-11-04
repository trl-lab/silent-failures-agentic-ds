import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import OrdinalEncoder
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# Load datasets
train_path = '/path/to/user/workspace/ds_failures/datasets/strable/aca-federal-upper-limits-wide/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/strable/aca-federal-upper-limits-wide/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'ACA FUL'
y = train_df[target_col].values
X = train_df.drop(columns=[target_col]).copy()
X_test = test_df.copy()
if target_col in X_test.columns:
    X_test = X_test.drop(columns=[target_col])

# Identify categorical and numerical columns
cat_cols = [col for col in X.columns if X[col].dtype == 'object' or pd.api.types.is_categorical_dtype(X[col])]
num_cols = [col for col in X.columns if col not in cat_cols]

# Fill missing values for categorical features with a constant
for col in cat_cols:
    X[col] = X[col].fillna('__MISSING__').astype(str)
    X_test[col] = X_test[col].fillna('__MISSING__').astype(str)

# Fill missing values for numerical features with median
for col in num_cols:
    median_val = X[col].median()
    X[col] = X[col].fillna(median_val)
    X_test[col] = X_test[col].fillna(median_val)

# Ordinal encode categoricals for tree-based models
encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
X[cat_cols] = encoder.fit_transform(X[cat_cols])
X_test[cat_cols] = encoder.transform(X_test[cat_cols])

# Set categorical column types for LightGBM
for col in cat_cols:
    X[col] = X[col].astype('category')
    X_test[col] = X_test[col].astype('category')

# K-Fold Cross Validation
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

lgb_preds = np.zeros(len(X_test))
cat_preds = np.zeros(len(X_test))

# CatBoost data copy (categorical features as integers or strings)
X_cb = X.copy()
X_test_cb = X_test.copy()
for col in cat_cols:
    X_cb[col] = X_cb[col].astype(int)
    X_test_cb[col] = X_test_cb[col].astype(int)

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_tr, y_tr = X.iloc[train_idx], y[train_idx]
    X_va, y_va = X.iloc[val_idx], y[val_idx]

    # LightGBM
    lgb = LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        n_jobs=-1,
        verbose=-1
    )
    lgb.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[]
    )
    lgb_preds += lgb.predict(X_test) / n_splits

    # CatBoost
    cat = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        random_seed=42 + fold,
        verbose=False
    )
    cat.fit(
        X_cb.iloc[train_idx], y_tr,
        eval_set=(X_cb.iloc[val_idx], y_va),
        early_stopping_rounds=50,
        verbose=False
    )
    cat_preds += cat.predict(X_test_cb) / n_splits

# Ensemble predictions
final_preds = 0.5 * lgb_preds + 0.5 * cat_preds

# Save submission
submission = pd.DataFrame({
    'id': np.arange(len(final_preds)),
    'ACA FUL': final_preds
})
submission.to_csv('predictions.csv', index=False)