import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import OrdinalEncoder
from sklearn.linear_model import Ridge
import lightgbm as lgb
import catboost as cb
import xgboost as xgb

train_path = '/path/to/user/workspace/ds_failures/datasets/uci/student-performance/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/uci/student-performance/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'G3'
y = train_df[target_col].values.astype(float)

feature_cols = [c for c in train_df.columns if c != target_col and c in test_df.columns]
X = train_df[feature_cols].copy()
X_test = test_df[feature_cols].copy()

for col in feature_cols:
    if X[col].dtype == 'object' or X[col].dtype.name == 'category':
        try:
            num_X = pd.to_numeric(X[col])
            num_test = pd.to_numeric(X_test[col])
            if num_X.isna().sum() == X[col].isna().sum():
                X[col] = num_X
                X_test[col] = num_test
        except Exception:
            pass

cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
if cat_cols:
    oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    X[cat_cols] = oe.fit_transform(X[cat_cols].astype(str))
    X_test[cat_cols] = oe.transform(X_test[cat_cols].astype(str))

X = X.fillna(-999).astype(float)
X_test = X_test.fillna(-999).astype(float)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

oof_cb = np.zeros(len(X))
test_cb = np.zeros(len(X_test))

oof_lgb = np.zeros(len(X))
test_lgb = np.zeros(len(X_test))

oof_xgb = np.zeros(len(X))
test_xgb = np.zeros(len(X_test))

oof_ridge = np.zeros(len(X))
test_ridge = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_tr, y_tr = X.iloc[train_idx], y[train_idx]
    X_va, y_va = X.iloc[val_idx], y[val_idx]

    model_cb = cb.CatBoostRegressor(
        iterations=800,
        learning_rate=0.03,
        depth=5,
        random_seed=42 + fold,
        verbose=0
    )
    model_cb.fit(X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=100, verbose=False)
    oof_cb[val_idx] = model_cb.predict(X_va)
    test_cb += model_cb.predict(X_test) / 5.0

    model_lgb = lgb.LGBMRegressor(
        n_estimators=800,
        learning_rate=0.03,
        num_leaves=15,
        max_depth=4,
        random_state=42 + fold,
        verbose=-1
    )
    model_lgb.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)]
    )
    oof_lgb[val_idx] = model_lgb.predict(X_va)
    test_lgb += model_lgb.predict(X_test) / 5.0

    model_xgb = xgb.XGBRegressor(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        verbosity=0
    )
    model_xgb.fit(X_tr, y_tr)
    oof_xgb[val_idx] = model_xgb.predict(X_va)
    test_xgb += model_xgb.predict(X_test) / 5.0

    model_ridge = Ridge(alpha=10.0)
    model_ridge.fit(X_tr, y_tr)
    oof_ridge[val_idx] = model_ridge.predict(X_va)
    test_ridge += model_ridge.predict(X_test) / 5.0

oof_matrix = np.column_stack([oof_cb, oof_lgb, oof_xgb, oof_ridge])
test_matrix = np.column_stack([test_cb, test_lgb, test_xgb, test_ridge])

meta_model = Ridge(alpha=1.0, positive=True)
meta_model.fit(oof_matrix, y)
final_test_preds = meta_model.predict(test_matrix)

final_test_preds = np.clip(final_test_preds, 0, 20)

sub = pd.DataFrame({
    'id': np.arange(len(final_test_preds)),
    'G3': final_test_preds
})
sub.to_csv('predictions.csv', index=False)