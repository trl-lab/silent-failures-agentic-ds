import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import lightgbm as lgb
from catboost import CatBoostRegressor
import xgboost as xgb

train_path = "/path/to/user/workspace/ds_failures/datasets/openml/house_prices_ames/data/train.parquet"
test_path = "/path/to/user/workspace/ds_failures/datasets/openml/house_prices_ames/data/test.parquet"

train = pd.read_parquet(train_path)
test = pd.read_parquet(test_path)

target_col = "Sale_Price"
y_train = train[target_col].values.astype(float)
X_train = train.drop(columns=[target_col]).copy()
X_test = test.drop(columns=[target_col]).copy() if target_col in test.columns else test.copy()

categorical_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()

for col in categorical_cols:
    X_train[col] = X_train[col].astype(str)
    X_test[col] = X_test[col].astype(str)
    
    categories = sorted(list(set(X_train[col].unique()).union(set(X_test[col].unique()))))
    cat_type = pd.CategoricalDtype(categories=categories)
    
    X_train[col] = X_train[col].astype(cat_type).cat.codes
    X_test[col] = X_test[col].astype(cat_type).cat.codes

for col in X_train.columns:
    X_train[col] = pd.to_numeric(X_train[col], errors="coerce")
    X_test[col] = pd.to_numeric(X_test[col], errors="coerce")

kf = KFold(n_splits=5, shuffle=True, random_state=42)
preds = np.zeros(len(X_test))

for train_idx, val_idx in kf.split(X_train):
    X_tr, y_tr = X_train.iloc[train_idx], y_train[train_idx]
    X_va, y_va = X_train.iloc[val_idx], y_train[val_idx]
    
    y_tr_log = np.log1p(y_tr)
    y_va_log = np.log1p(y_va)
    
    lgb_log = lgb.LGBMRegressor(
        n_estimators=1500, learning_rate=0.03, random_state=42,
        verbosity=-1, subsample=0.8, colsample_bytree=0.8
    )
    lgb_log.fit(X_tr, y_tr_log, eval_set=[(X_va, y_va_log)], callbacks=[lgb.early_stopping(50, verbose=False)])
    p_lgb_log = np.expm1(lgb_log.predict(X_test))
    
    lgb_raw = lgb.LGBMRegressor(
        n_estimators=1500, learning_rate=0.03, random_state=42,
        verbosity=-1, subsample=0.8, colsample_bytree=0.8
    )
    lgb_raw.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=[lgb.early_stopping(50, verbose=False)])
    p_lgb_raw = lgb_raw.predict(X_test)
    
    cb_log = CatBoostRegressor(
        iterations=1500, learning_rate=0.03, random_seed=42, verbose=0, subsample=0.8
    )
    cb_log.fit(X_tr, y_tr_log, eval_set=(X_va, y_va_log), early_stopping_rounds=50)
    p_cb_log = np.expm1(cb_log.predict(X_test))
    
    cb_raw = CatBoostRegressor(
        iterations=1500, learning_rate=0.03, random_seed=42, verbose=0
    )
    cb_raw.fit(X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=50)
    p_cb_raw = cb_raw.predict(X_test)
    
    xgb_log = xgb.XGBRegressor(
        n_estimators=1500, learning_rate=0.03, random_state=42,
        subsample=0.8, colsample_bytree=0.8, early_stopping_rounds=50
    )
    xgb_log.fit(X_tr, y_tr_log, eval_set=[(X_va, y_va_log)], verbose=False)
    p_xgb_log = np.expm1(xgb_log.predict(X_test))
    
    fold_pred = (p_lgb_log * 0.25 + p_lgb_raw * 0.25 + p_cb_log * 0.25 + p_cb_raw * 0.125 + p_xgb_log * 0.125)
    preds += fold_pred / 5.0

sub = pd.DataFrame({
    "id": np.arange(len(preds)),
    "Sale_Price": preds
})

sub.to_csv("predictions.csv", index=False)