import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import lightgbm as lgb
from catboost import CatBoostRegressor

train_path = '/path/to/user/workspace/ds_failures/datasets/strable/hospitals/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/strable/hospitals/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'BEDS'
drop_cols = ['OBJECTID', 'ID', 'ADDRESS', 'TELEPHONE', 'WEBSITE', 'STATE_ID', 'ALT_NAME', 'SOURCE', 'SOURCEDATE', 'VAL_DATE', target_col]
feature_cols = [col for col in train_df.columns if col not in drop_cols]

X = train_df[feature_cols].copy()
y = train_df[target_col].values
X_test = test_df[feature_cols].copy()

categorical_cols = []
for col in feature_cols:
    if X[col].dtype == 'object' or pd.api.types.is_string_dtype(X[col]) or pd.api.types.is_categorical_dtype(X[col]):
        categorical_cols.append(col)
        X[col] = X[col].astype(str).fillna('MISSING')
        X_test[col] = X_test[col].astype(str).fillna('MISSING')
    else:
        X[col] = pd.to_numeric(X[col], errors='coerce')
        X_test[col] = pd.to_numeric(X_test[col], errors='coerce')

for col in categorical_cols:
    X[col] = X[col].astype('category')
    X_test[col] = pd.Categorical(X_test[col], categories=X[col].cat.categories)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

lgb_preds = np.zeros(len(X_test))
cb_preds = np.zeros(len(X_test))

cat_features_idx = [X.columns.get_loc(c) for c in categorical_cols]

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_tr, y_tr = X.iloc[train_idx], y[train_idx]
    X_va, y_va = X.iloc[val_idx], y[val_idx]
    
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
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    lgb_preds += lgb_model.predict(X_test) / kf.n_splits

    X_tr_cb = X_tr.copy()
    X_va_cb = X_va.copy()
    X_test_cb = X_test.copy()
    for col in categorical_cols:
        X_tr_cb[col] = X_tr_cb[col].astype(str)
        X_va_cb[col] = X_va_cb[col].astype(str)
        X_test_cb[col] = X_test_cb[col].astype(str)

    cb_model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.04,
        depth=6,
        cat_features=categorical_cols,
        random_seed=42 + fold,
        verbose=0
    )
    cb_model.fit(
        X_tr_cb, y_tr,
        eval_set=(X_va_cb, y_va),
        early_stopping_rounds=50,
        verbose=False
    )
    cb_preds += cb_model.predict(X_test_cb) / kf.n_splits

final_preds = 0.5 * lgb_preds + 0.5 * cb_preds

output_df = pd.DataFrame({
    'id': np.arange(len(final_preds)),
    'BEDS': final_preds
})

output_df.to_csv('predictions.csv', index=False)