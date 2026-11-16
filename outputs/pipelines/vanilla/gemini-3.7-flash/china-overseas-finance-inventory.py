import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

# Load datasets
train_path = '/path/to/user/workspace/ds_failures/datasets/strable/china-overseas-finance-inventory/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/strable/china-overseas-finance-inventory/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'installed_capacity'
y_train = train_df[target_col].values
X_train = train_df.drop(columns=[target_col]).copy()
X_test = test_df.copy()

# Drop identifier columns with excessive uniqueness or irrelevant text if necessary
id_cols_to_drop = ['unique_id', 'power_plant_name', 'compid', 'unit_id', 'cgp_id', 'gppd_id', 'sais_id', 'r_id', 'ij_id']
cols_to_drop = [c for c in id_cols_to_drop if c in X_train.columns]
X_train.drop(columns=cols_to_drop, inplace=True)
X_test.drop(columns=cols_to_drop, inplace=True)

# Identify column types
cat_cols = []
num_cols = []

for col in X_train.columns:
    if X_train[col].dtype == 'object' or pd.api.types.is_categorical_dtype(X_train[col]):
        cat_cols.append(col)
    else:
        num_cols.append(col)

# Handle categorical features for LightGBM, CatBoost, XGBoost
for col in cat_cols:
    X_train[col] = X_train[col].astype(str).fillna('missing')
    X_test[col] = X_test[col].astype(str).fillna('missing')
    
    # Categorical dtype for LightGBM
    X_train[col] = X_train[col].astype('category')
    X_test[col] = pd.Categorical(X_test[col], categories=X_train[col].cat.categories)

for col in num_cols:
    X_train[col] = pd.to_numeric(X_train[col], errors='coerce')
    X_test[col] = pd.to_numeric(X_test[col], errors='coerce')

# Convert categorical for XGBoost (using category codes)
X_train_xgb = X_train.copy()
X_test_xgb = X_test.copy()
for col in cat_cols:
    X_train_xgb[col] = X_train_xgb[col].cat.codes
    X_test_xgb[col] = X_test_xgb[col].cat.codes

# K-Fold CV
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

lgb_preds = np.zeros(len(X_test))
cat_preds = np.zeros(len(X_test))
xgb_preds = np.zeros(len(X_test))

# Convert string categorical for CatBoost
X_train_cb = X_train.copy()
X_test_cb = X_test.copy()
for col in cat_cols:
    X_train_cb[col] = X_train_cb[col].astype(str)
    X_test_cb[col] = X_test_cb[col].astype(str)

for train_idx, val_idx in kf.split(X_train):
    # LightGBM
    lgb = LGBMRegressor(n_estimators=500, learning_rate=0.03, random_state=42, verbose=-1)
    lgb.fit(X_train.iloc[train_idx], y_train[train_idx])
    lgb_preds += lgb.predict(X_test) / n_splits
    
    # CatBoost
    cb = CatBoostRegressor(iterations=500, learning_rate=0.03, cat_features=cat_cols, verbose=0, random_seed=42)
    cb.fit(X_train_cb.iloc[train_idx], y_train[train_idx])
    cat_preds += cb.predict(X_test_cb) / n_splits

    # XGBoost
    xgb = XGBRegressor(n_estimators=500, learning_rate=0.03, random_state=42, max_depth=6)
    xgb.fit(X_train_xgb.iloc[train_idx], y_train[train_idx])
    xgb_preds += xgb.predict(X_test_xgb) / n_splits

# Ensemble predictions
final_predictions = (lgb_preds + cat_preds + xgb_preds) / 3.0

# Save predictions
output_df = pd.DataFrame({
    'id': range(len(final_predictions)),
    'installed_capacity': final_predictions
})
output_df.to_csv('predictions.csv', index=False)