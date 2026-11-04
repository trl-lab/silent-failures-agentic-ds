import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# Load data
train_path = '/path/to/user/workspace/ds_failures/datasets/strable/disbursements-in-trust-funds/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/strable/disbursements-in-trust-funds/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'Cash Disbursements (US$)'
y = train_df[target_col].values
X_train = train_df.drop(columns=[target_col]).copy()
X_test = test_df.copy()

categorical_cols = [col for col in X_train.columns if X_train[col].dtype == 'object' or X_train[col].dtype.name == 'category']
numeric_cols = [col for col in X_train.columns if col not in categorical_cols]

# Fill missing values
for col in categorical_cols:
    X_train[col] = X_train[col].fillna('Missing').astype(str)
    X_test[col] = X_test[col].fillna('Missing').astype(str)

for col in numeric_cols:
    median_val = X_train[col].median()
    X_train[col] = X_train[col].fillna(median_val)
    X_test[col] = X_test[col].fillna(median_val)

# Frequency encoding for high cardinality categoricals
for col in categorical_cols:
    freq = pd.concat([X_train[col], X_test[col]]).value_counts()
    X_train[f'{col}_freq'] = X_train[col].map(freq).fillna(0)
    X_test[f'{col}_freq'] = X_test[col].map(freq).fillna(0)

# Convert categorical columns to category dtype for LightGBM
for col in categorical_cols:
    X_train[col] = X_train[col].astype('category')
    X_test[col] = X_test[col].astype('category')

# K-Fold Cross Validation and Training
kf = KFold(n_splits=5, shuffle=True, random_state=42)
lgb_preds = np.zeros(len(X_test))
cb_preds = np.zeros(len(X_test))

# Prepare features for CatBoost (convert category to str)
cb_train = X_train.copy()
cb_test = X_test.copy()
for col in categorical_cols:
    cb_train[col] = cb_train[col].astype(str)
    cb_test[col] = cb_test[col].astype(str)

for train_idx, val_idx in kf.split(X_train):
    X_tr, y_tr = X_train.iloc[train_idx], y[train_idx]
    X_va, y_va = X_train.iloc[val_idx], y[val_idx]

    # LightGBM
    lgb = LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        verbosity=-1
    )
    lgb.fit(X_tr, y_tr)
    lgb_preds += lgb.predict(X_test) / kf.n_splits

    # CatBoost
    cb_tr = cb_train.iloc[train_idx]
    cb_va = cb_train.iloc[val_idx]
    cb = CatBoostRegressor(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        cat_features=categorical_cols,
        verbose=False,
        random_seed=42
    )
    cb.fit(cb_tr, y_tr)
    cb_preds += cb.predict(cb_test) / kf.n_splits

# Ensemble predictions
final_predictions = 0.5 * lgb_preds + 0.5 * cb_preds

# Save predictions
output_df = pd.DataFrame({
    'id': np.arange(len(final_predictions)),
    target_col: final_predictions
})
output_df.to_csv('predictions.csv', index=False)