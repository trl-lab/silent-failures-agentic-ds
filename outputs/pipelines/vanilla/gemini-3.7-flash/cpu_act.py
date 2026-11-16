import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
from catboost import CatBoostRegressor
import xgboost as xgb

# 1. Load data
train_path = '/path/to/user/workspace/ds_failures/datasets/openml/cpu_act/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/openml/cpu_act/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'usr'
feature_cols = [col for col in train_df.columns if col != target_col]

X = train_df[feature_cols].copy()
y = train_df[target_col].copy()
X_test = test_df[feature_cols].copy()

# 2. Cross-validation and ensemble training
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_preds = np.zeros(len(train_df))
test_preds = np.zeros(len(test_df))

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_train_f, y_train_f = X.iloc[train_idx], y.iloc[train_idx]
    X_val_f, y_val_f = X.iloc[val_idx], y.iloc[val_idx]
    
    # Model 1: LightGBM
    lgb_model = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        verbose=-1
    )
    lgb_model.fit(
        X_train_f, y_train_f,
        eval_set=[(X_val_f, y_val_f)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )
    
    # Model 2: CatBoost
    cb_model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.03,
        depth=6,
        random_seed=42 + fold,
        verbose=0
    )
    cb_model.fit(
        X_train_f, y_train_f,
        eval_set=(X_val_f, y_val_f),
        early_stopping_rounds=50,
        verbose=False
    )
    
    # Model 3: XGBoost
    xgb_model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        early_stopping_rounds=50
    )
    xgb_model.fit(
        X_train_f, y_train_f,
        eval_set=[(X_val_f, y_val_f)],
        verbose=False
    )
    
    # Validation predictions (fold ensemble)
    val_pred_lgb = lgb_model.predict(X_val_f)
    val_pred_cb = cb_model.predict(X_val_f)
    val_pred_xgb = xgb_model.predict(X_val_f)
    oof_preds[val_idx] = (val_pred_lgb + val_pred_cb + val_pred_xgb) / 3.0
    
    # Test predictions
    test_pred_lgb = lgb_model.predict(X_test)
    test_pred_cb = cb_model.predict(X_test)
    test_pred_xgb = xgb_model.predict(X_test)
    test_preds += (test_pred_lgb + test_pred_cb + test_pred_xgb) / 3.0 / n_splits

# 3. Save submission
submission_df = pd.DataFrame({
    'id': np.arange(len(test_preds)),
    'usr': test_preds
})

submission_df.to_csv('predictions.csv', index=False)