import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# Load datasets
train_path = '/path/to/user/workspace/ds_failures/datasets/strable/contributions-to-financial-intermediary-funds/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/strable/contributions-to-financial-intermediary-funds/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'Amount in USD'
y_train = train_df[target_col].values
X_train = train_df.drop(columns=[target_col]).copy()
X_test = test_df.drop(columns=[target_col], errors='ignore').copy()

# Identify feature types
cat_cols = [col for col in X_train.columns if X_train[col].dtype == 'object' or isinstance(X_train[col].dtype, pd.CategoricalDtype)]
num_cols = [col for col in X_train.columns if col not in cat_cols]

# Fill missing values for categorical features
for col in cat_cols:
    X_train[col] = X_train[col].astype(str).fillna('missing')
    X_test[col] = X_test[col].astype(str).fillna('missing')

for col in num_cols:
    median_val = X_train[col].median()
    X_train[col] = X_train[col].fillna(median_val)
    X_test[col] = X_test[col].fillna(median_val)

# Prepare categorical data for LightGBM
X_train_lgb = X_train.copy()
X_test_lgb = X_test.copy()
for col in cat_cols:
    X_train_lgb[col] = X_train_lgb[col].astype('category')
    X_test_lgb[col] = X_test_lgb[col].astype('category')

# K-Fold Cross-Validation Setup
kf = KFold(n_splits=5, shuffle=True, random_state=42)
lgb_preds = np.zeros(len(X_test))
cb_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    # LightGBM
    X_tr_lgb, y_tr = X_train_lgb.iloc[train_idx], y_train[train_idx]
    X_va_lgb, y_va = X_train_lgb.iloc[val_idx], y_train[val_idx]

    lgb_model = LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=31,
        random_state=42 + fold,
        verbosity=-1
    )
    lgb_model.fit(
        X_tr_lgb, y_tr,
        eval_set=[(X_va_lgb, y_va)],
        callbacks=[]
    )
    lgb_preds += lgb_model.predict(X_test_lgb) / kf.n_splits

    # CatBoost
    X_tr_cb = X_train.iloc[train_idx]
    X_va_cb = X_train.iloc[val_idx]

    cb_model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        cat_features=cat_cols,
        random_seed=42 + fold,
        verbose=0
    )
    cb_model.fit(
        X_tr_cb, y_tr,
        eval_set=(X_va_cb, y_va),
        early_stopping_rounds=50,
        verbose=False
    )
    cb_preds += cb_model.predict(X_test) / kf.n_splits

# Ensemble predictions
final_predictions = 0.5 * lgb_preds + 0.5 * cb_preds

# Save predictions
output_df = pd.DataFrame({
    'id': np.arange(len(final_predictions)),
    'Amount in USD': final_predictions
})
output_df.to_csv('predictions.csv', index=False)