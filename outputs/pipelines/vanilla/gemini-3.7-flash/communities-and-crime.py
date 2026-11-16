import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
import lightgbm as lgb
from catboost import CatBoostRegressor

# Load datasets
train_path = '/path/to/user/workspace/ds_failures/datasets/uci/communities-and-crime/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/uci/communities-and-crime/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'ViolentCrimesPerPop'
y_train = train_df[target_col].values

# Drop target and non-predictive metadata
drop_cols = [target_col, 'fold', 'communityname']
feature_cols = [c for c in train_df.columns if c not in drop_cols]

X_train = train_df[feature_cols].copy()
X_test = test_df[feature_cols].copy()

# Replace '?' with NaN and convert all features to numeric
for col in feature_cols:
    X_train[col] = pd.to_numeric(X_train[col].replace('?', np.nan), errors='coerce')
    X_test[col] = pd.to_numeric(X_test[col].replace('?', np.nan), errors='coerce')

# 5-fold Cross-Validation Ensemble
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

lgb_preds = np.zeros(len(X_test))
cb_preds = np.zeros(len(X_test))
ridge_preds = np.zeros(len(X_test))

# Imputer for linear model
imputer = SimpleImputer(strategy='median')
X_train_imp = imputer.fit_transform(X_train)
X_test_imp = imputer.transform(X_test)

for train_idx, val_idx in kf.split(X_train):
    X_tr, y_tr = X_train.iloc[train_idx], y_train[train_idx]
    X_va, y_va = X_train.iloc[val_idx], y_train[val_idx]
    
    # LightGBM
    lgb_model = lgb.LGBMRegressor(
        n_estimators=300,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=-1
    )
    lgb_model.fit(X_tr, y_tr)
    lgb_preds += lgb_model.predict(X_test) / n_splits
    
    # CatBoost
    cb_model = CatBoostRegressor(
        iterations=400,
        learning_rate=0.03,
        depth=6,
        random_seed=42,
        verbose=0
    )
    cb_model.fit(X_tr, y_tr)
    cb_preds += cb_model.predict(X_test) / n_splits
    
    # Ridge Regression
    X_tr_imp, y_tr_imp = X_train_imp[train_idx], y_train[train_idx]
    ridge_model = Ridge(alpha=1.0, random_state=42)
    ridge_model.fit(X_tr_imp, y_tr_imp)
    ridge_preds += ridge_model.predict(X_test_imp) / n_splits

# Blend predictions and clip to valid [0, 1] range
final_preds = (0.4 * lgb_preds + 0.4 * cb_preds + 0.2 * ridge_preds)
final_preds = np.clip(final_preds, 0.0, 1.0)

# Save predictions
output_df = pd.DataFrame({
    'id': np.arange(len(final_preds)),
    'ViolentCrimesPerPop': final_preds
})
output_df.to_csv('predictions.csv', index=False)