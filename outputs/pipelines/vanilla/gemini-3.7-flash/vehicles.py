import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# Load data
train_df = pd.read_parquet('/path/to/user/workspace/ds_failures/datasets/strable/vehicles/data/train.parquet')
test_df = pd.read_parquet('/path/to/user/workspace/ds_failures/datasets/strable/vehicles/data/test.parquet')

# Separate target and features
y_train = train_df['youSaveSpend'].values
X_train = train_df.drop(columns=['youSaveSpend']).copy()
X_test = test_df.copy()

# Preprocess date columns
date_cols = ['createdOn', 'modifiedOn']
for col in date_cols:
    if col in X_train.columns:
        dt_train = pd.to_datetime(X_train[col], errors='coerce')
        dt_test = pd.to_datetime(X_test[col], errors='coerce')
        
        X_train[col + '_year'] = dt_train.dt.year.fillna(-1).astype(int)
        X_train[col + '_month'] = dt_train.dt.month.fillna(-1).astype(int)
        X_train[col + '_day'] = dt_train.dt.day.fillna(-1).astype(int)
        
        X_test[col + '_year'] = dt_test.dt.year.fillna(-1).astype(int)
        X_test[col + '_month'] = dt_test.dt.month.fillna(-1).astype(int)
        X_test[col + '_day'] = dt_test.dt.day.fillna(-1).astype(int)
        
        X_train.drop(columns=[col], inplace=True)
        X_test.drop(columns=[col], inplace=True)

# Convert boolean columns to integer
bool_cols = X_train.select_dtypes(include=['bool']).columns.tolist()
for col in bool_cols:
    X_train[col] = X_train[col].astype(int)
    X_test[col] = X_test[col].astype(int)

# Identify and process categorical columns
cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
for col in cat_cols:
    X_train[col] = X_train[col].astype(str).fillna('missing').astype('category')
    X_test[col] = pd.Categorical(X_test[col].astype(str).fillna('missing'), categories=X_train[col].cat.categories)

# Numerical columns
num_cols = [c for c in X_train.columns if c not in cat_cols]
for col in num_cols:
    fill_val = X_train[col].median() if not pd.isna(X_train[col].median()) else 0
    X_train[col] = X_train[col].fillna(fill_val)
    X_test[col] = X_test[col].fillna(fill_val)

# Convert categorical to integer codes for tree models
X_train_encoded = X_train.copy()
X_test_encoded = X_test.copy()
for col in cat_cols:
    X_train_encoded[col] = X_train[col].cat.codes
    X_test_encoded[col] = X_test[col].cat.codes

# 5-fold cross-validation ensemble
kf = KFold(n_splits=5, shuffle=True, random_state=42)
lgb_preds = np.zeros(len(X_test))
cb_preds = np.zeros(len(X_test))

for train_idx, val_idx in kf.split(X_train_encoded):
    X_tr, y_tr = X_train_encoded.iloc[train_idx], y_train[train_idx]
    
    # LightGBM
    lgb_model = LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1
    )
    lgb_model.fit(X_tr, y_tr)
    lgb_preds += lgb_model.predict(X_test_encoded) / kf.n_splits
    
    # CatBoost
    cb_model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.04,
        depth=6,
        random_seed=42,
        verbose=False
    )
    cb_model.fit(X_tr, y_tr)
    cb_preds += cb_model.predict(X_test_encoded) / kf.n_splits

# Blend predictions
final_preds = 0.5 * lgb_preds + 0.5 * cb_preds

# Save predictions
output_df = pd.DataFrame({
    'id': np.arange(len(final_preds)),
    'youSaveSpend': final_preds
})
output_df.to_csv('predictions.csv', index=False)