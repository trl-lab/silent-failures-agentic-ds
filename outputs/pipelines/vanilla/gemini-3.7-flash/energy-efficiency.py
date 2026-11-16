import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor

# Load data
train_path = '/path/to/user/workspace/ds_failures/datasets/openml/energy-efficiency/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/openml/energy-efficiency/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'heating_load'
features = [col for col in train_df.columns if col != target_col]

X = train_df[features].copy()
y = train_df[target_col].copy()
X_test = test_df[features].copy()

# Handle any missing values if present
for col in features:
    if X[col].isnull().any() or X_test[col].isnull().any():
        median_val = X[col].median()
        X[col] = X[col].fillna(median_val)
        X_test[col] = X_test[col].fillna(median_val)

# K-Fold Cross Validation & Model Ensembling
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

test_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
    X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]
    
    # Model 1: LightGBM
    lgb = LGBMRegressor(
        n_estimators=500,
        learning_rate=0.03,
        num_leaves=31,
        random_state=42 + fold,
        verbose=-1
    )
    lgb.fit(X_tr, y_tr)
    
    # Model 2: CatBoost
    cb = CatBoostRegressor(
        iterations=600,
        learning_rate=0.04,
        depth=6,
        random_seed=42 + fold,
        verbose=0
    )
    cb.fit(X_tr, y_tr)
    
    # Model 3: XGBoost
    xgb = XGBRegressor(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=5,
        random_state=42 + fold,
        verbosity=0
    )
    xgb.fit(X_tr, y_tr)
    
    # Model 4: ExtraTrees
    et = ExtraTreesRegressor(
        n_estimators=200,
        random_state=42 + fold,
        n_jobs=-1
    )
    et.fit(X_tr, y_tr)
    
    # Model 5: RandomForest
    rf = RandomForestRegressor(
        n_estimators=200,
        random_state=42 + fold,
        n_jobs=-1
    )
    rf.fit(X_tr, y_tr)
    
    # Predict on test
    fold_pred = (
        0.25 * lgb.predict(X_test) +
        0.25 * cb.predict(X_test) +
        0.25 * xgb.predict(X_test) +
        0.15 * et.predict(X_test) +
        0.10 * rf.predict(X_test)
    )
    test_preds += fold_pred / n_splits

# Save predictions
submission = pd.DataFrame({
    'id': np.arange(len(test_preds)),
    'heating_load': test_preds
})

submission.to_csv('predictions.csv', index=False)