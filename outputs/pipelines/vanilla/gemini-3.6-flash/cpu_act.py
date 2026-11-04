import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold

train_path = '/path/to/user/workspace/ds_failures/datasets/openml/cpu_act/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/openml/cpu_act/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'usr'
X = train_df.drop(columns=[target_col])
y = train_df[target_col]

if target_col in test_df.columns:
    X_test = test_df.drop(columns=[target_col])
else:
    X_test = test_df.copy()

X_test = X_test[X.columns]

cat_cols = X.select_dtypes(include=['object', 'category']).columns
X_num = X.copy()
X_test_num = X_test.copy()

for col in cat_cols:
    X_num[col] = X_num[col].astype('category').cat.codes
    X_test_num[col] = X_test_num[col].astype('category').cat.codes

kf = KFold(n_splits=10, shuffle=True, random_state=42)

test_preds_lgb = np.zeros(len(X_test_num))
test_preds_xgb = np.zeros(len(X_test_num))
test_preds_cb = np.zeros(len(X_test_num))

for fold, (train_idx, val_idx) in enumerate(kf.split(X_num, y)):
    X_tr, y_tr = X_num.iloc[train_idx], y.iloc[train_idx]
    X_va, y_va = X_num.iloc[val_idx], y.iloc[val_idx]
    
    model_lgb = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=31,
        random_state=42 + fold,
        verbosity=-1
    )
    model_lgb.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )
    test_preds_lgb += model_lgb.predict(X_test_num) / kf.n_splits
    
    model_xgb = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=6,
        early_stopping_rounds=50,
        random_state=42 + fold,
        verbosity=0
    )
    model_xgb.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        verbose=False
    )
    test_preds_xgb += model_xgb.predict(X_test_num) / kf.n_splits
    
    model_cb = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.03,
        depth=6,
        random_seed=42 + fold,
        verbose=0
    )
    model_cb.fit(
        X_tr, y_tr,
        eval_set=(X_va, y_va),
        early_stopping_rounds=50,
        verbose=False
    )
    test_preds_cb += model_cb.predict(X_test_num) / kf.n_splits

final_preds = (test_preds_lgb + test_preds_xgb + test_preds_cb) / 3.0

submission = pd.DataFrame({
    'id': np.arange(len(final_preds)),
    'usr': final_preds
})
submission.to_csv('predictions.csv', index=False)