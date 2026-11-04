import pandas as pd
import numpy as np
import lightgbm as lgb
from catboost import CatBoostRegressor
import xgboost as xgb
from sklearn.model_selection import KFold

train_path = "/path/to/user/workspace/ds_failures/datasets/strable/industry-payments-project/data/train.parquet"
test_path = "/path/to/user/workspace/ds_failures/datasets/strable/industry-payments-project/data/test.parquet"

train = pd.read_parquet(train_path)
test = pd.read_parquet(test_path)

if 'projectPayment' in train.columns:
    mask = train['projectPayment'].notna()
    train = train[mask].reset_index(drop=True)
    y = train['projectPayment'].values
    X = train.drop(columns=['projectPayment'])
else:
    raise ValueError("Target projectPayment not in train dataset")

X_test = test.drop(columns=['projectPayment'], errors='ignore').copy()

all_cols = list(X.columns)
for col in all_cols:
    if pd.api.types.is_datetime64_any_dtype(X[col]) or 'date' in col.lower() or 'time' in col.lower():
        try:
            d_train = pd.to_datetime(X[col], errors='coerce')
            d_test = pd.to_datetime(X_test[col], errors='coerce')
            if d_train.notna().sum() > 0:
                X[col + '_year'] = d_train.dt.year
                X[col + '_month'] = d_train.dt.month
                X[col + '_day'] = d_train.dt.day
                X[col + '_dayofweek'] = d_train.dt.dayofweek
                X_test[col + '_year'] = d_test.dt.year
                X_test[col + '_month'] = d_test.dt.month
                X_test[col + '_day'] = d_test.dt.day
                X_test[col + '_dayofweek'] = d_test.dt.dayofweek
                X = X.drop(columns=[col])
                X_test = X_test.drop(columns=[col])
        except Exception:
            pass

cat_cols = []
features = list(X.columns)
for col in features:
    if X[col].dtype == 'object' or isinstance(X[col].dtype, pd.CategoricalDtype):
        cat_cols.append(col)
        train_cats = X[col].astype(str).fillna('missing')
        test_cats = X_test[col].astype(str).fillna('missing')
        all_cats = pd.concat([train_cats, test_cats]).unique()
        X[col] = pd.Categorical(train_cats, categories=all_cats)
        X_test[col] = pd.Categorical(test_cats, categories=all_cats)

X_cb = X.copy()
X_test_cb = X_test.copy()
for col in cat_cols:
    X_cb[col] = X_cb[col].astype(str)
    X_test_cb[col] = X_test_cb[col].astype(str)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

lgb_preds = np.zeros(len(X_test))
cat_preds = np.zeros(len(X_test))
xgb_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_tr, y_tr = X.iloc[train_idx], y[train_idx]
    X_va, y_va = X.iloc[val_idx], y[val_idx]
    
    model_lgb = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        random_state=42 + fold,
        n_jobs=-1,
        verbose=-1
    )
    model_lgb.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    lgb_preds += model_lgb.predict(X_test) / kf.n_splits
    
    X_tr_cb, X_va_cb = X_cb.iloc[train_idx], X_cb.iloc[val_idx]
    model_cat = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.05,
        cat_features=cat_cols,
        random_seed=42 + fold,
        verbose=0
    )
    model_cat.fit(
        X_tr_cb, y_tr,
        eval_set=(X_va_cb, y_va),
        early_stopping_rounds=50,
        verbose=False
    )
    cat_preds += model_cat.predict(X_test_cb) / kf.n_splits
    
    model_xgb = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        enable_categorical=True,
        random_state=42 + fold,
        n_jobs=-1,
        early_stopping_rounds=50
    )
    model_xgb.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        verbose=False
    )
    xgb_preds += model_xgb.predict(X_test) / kf.n_splits

final_preds = 0.4 * lgb_preds + 0.4 * cat_preds + 0.2 * xgb_preds

sub = pd.DataFrame({
    'id': np.arange(len(final_preds)),
    'projectPayment': final_preds
})
sub.to_csv('predictions.csv', index=False)