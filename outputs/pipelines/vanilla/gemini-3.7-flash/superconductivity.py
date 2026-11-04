import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

# Load datasets
train_path = '/path/to/user/workspace/ds_failures/datasets/uci/superconductivity/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/uci/superconductivity/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'critical_temp'
X = train_df.drop(columns=[target_col])
y = train_df[target_col]
X_test = test_df.copy()

if target_col in X_test.columns:
    X_test = X_test.drop(columns=[target_col])

# Ensure consistent column ordering
feature_cols = list(X.columns)
X_test = X_test[feature_cols]

# K-Fold Ensemble
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

lgb_test_preds = np.zeros(len(X_test))
xgb_test_preds = np.zeros(len(X_test))
cat_test_preds = np.zeros(len(X_test))

lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'n_estimators': 1500,
    'learning_rate': 0.03,
    'num_leaves': 63,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1
}

xgb_params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'n_estimators': 1500,
    'learning_rate': 0.03,
    'max_depth': 8,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'n_jobs': -1
}

cat_params = {
    'loss_function': 'RMSE',
    'iterations': 1500,
    'learning_rate': 0.04,
    'depth': 7,
    'random_seed': 42,
    'verbose': 0
}

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
    X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]
    
    # LightGBM
    lgb_model = lgb.LGBMRegressor(**lgb_params)
    lgb_model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )
    lgb_test_preds += lgb_model.predict(X_test) / n_splits
    
    # XGBoost
    xgb_model = xgb.XGBRegressor(**xgb_params, early_stopping_rounds=50)
    xgb_model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        verbose=False
    )
    xgb_test_preds += xgb_model.predict(X_test) / n_splits
    
    # CatBoost
    cat_model = CatBoostRegressor(**cat_params, early_stopping_rounds=50)
    cat_model.fit(
        X_tr, y_tr,
        eval_set=(X_va, y_va),
        verbose=False
    )
    cat_test_preds += cat_model.predict(X_test) / n_splits

# Ensemble predictions (critical_temp is non-negative)
final_preds = (lgb_test_preds + xgb_test_preds + cat_test_preds) / 3.0
final_preds = np.clip(final_preds, 0, None)

# Save submission
submission = pd.DataFrame({
    'id': np.arange(len(final_preds)),
    'critical_temp': final_preds
})

submission.to_csv('predictions.csv', index=False)