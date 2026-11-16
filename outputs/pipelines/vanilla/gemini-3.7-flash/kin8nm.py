import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

# Load datasets
train_path = '/path/to/user/workspace/ds_failures/datasets/openml/kin8nm/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/openml/kin8nm/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'y'
feature_cols = [col for col in train_df.columns if col != target_col]

X = train_df[feature_cols].copy()
y = train_df[target_col].copy()
X_test = test_df[feature_cols].copy()

# Handle any missing values if present
for col in feature_cols:
    if X[col].isnull().any() or X_test[col].isnull().any():
        median_val = X[col].median()
        X[col] = X[col].fillna(median_val)
        X_test[col] = X_test[col].fillna(median_val)

# K-Fold Cross Validation Setup
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

test_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
    X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]
    
    # Model 1: LightGBM
    lgb = LGBMRegressor(
        n_estimators=1500,
        learning_rate=0.03,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        verbose=-1
    )
    lgb.fit(X_tr, y_tr)
    lgb_pred_test = lgb.predict(X_test)
    
    # Model 2: XGBoost
    xgb = XGBRegressor(
        n_estimators=1500,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        n_jobs=-1
    )
    xgb.fit(X_tr, y_tr)
    xgb_pred_test = xgb.predict(X_test)
    
    # Model 3: CatBoost
    cb = CatBoostRegressor(
        iterations=1500,
        learning_rate=0.03,
        depth=6,
        random_seed=42 + fold,
        verbose=0
    )
    cb.fit(X_tr, y_tr)
    cb_pred_test = cb.predict(X_test)
    
    # Blend predictions for this fold
    fold_test_pred = (lgb_pred_test + xgb_pred_test + cb_pred_test) / 3.0
    test_preds += fold_test_pred / n_splits

# Save predictions to CSV
output_df = pd.DataFrame({
    'id': np.arange(len(test_preds)),
    'y': test_preds
})
output_df.to_csv('predictions.csv', index=False)