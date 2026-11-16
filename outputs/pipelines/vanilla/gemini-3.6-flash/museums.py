import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

train_path = '/path/to/user/workspace/ds_failures/datasets/strable/museums/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/strable/museums/data/test.parquet'

train = pd.read_parquet(train_path)
test = pd.read_parquet(test_path)

if 'REVENUE15' in train.columns:
    train = train.dropna(subset=['REVENUE15']).reset_index(drop=True)

y = train['REVENUE15'].values
X = train.drop(columns=['REVENUE15'])
X_test = test.drop(columns=['REVENUE15'], errors='ignore')

X, X_test = X.align(X_test, join='left', axis=1)

cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

kf = KFold(n_splits=5, shuffle=True, random_state=42)
y_series = pd.Series(y, index=X.index)

for c in cat_cols:
    X[c] = X[c].astype(str)
    X_test[c] = X_test[c].astype(str)

    combined = pd.concat([X[c], X_test[c]], axis=0)
    freq = combined.value_counts()
    X[f'{c}_freq'] = X[c].map(freq)
    X_test[f'{c}_freq'] = X_test[c].map(freq)

    te_train = pd.Series(index=X.index, dtype=float)
    for tr_idx, va_idx in kf.split(X):
        tr_cat = X[c].iloc[tr_idx]
        tr_y = y_series.iloc[tr_idx]
        means = tr_y.groupby(tr_cat).mean()
        te_train.iloc[va_idx] = X[c].iloc[va_idx].map(means)

    global_mean = y.mean()
    te_train = te_train.fillna(global_mean)

    full_means = y_series.groupby(X[c]).mean()
    te_test = X_test[c].map(full_means).fillna(global_mean)

    X[f'{c}_te'] = te_train
    X_test[f'{c}_te'] = te_test

    cat_type = pd.CategoricalDtype(categories=combined.unique())
    X[c] = X[c].astype(cat_type).cat.codes
    X_test[c] = X_test[c].astype(cat_type).cat.codes

clean_cols = [str(col).replace('[', '_').replace(']', '_').replace('<', '_').replace('>', '_').replace(':', '_') for col in X.columns]
X.columns = clean_cols
X_test.columns = clean_cols

for c in X.columns:
    X[c] = pd.to_numeric(X[c], errors='coerce')
    X_test[c] = pd.to_numeric(X_test[c], errors='coerce')

lgb_preds = np.zeros(len(X_test))
xgb_preds = np.zeros(len(X_test))
cat_preds = np.zeros(len(X_test))

for fold, (tr_idx, va_idx) in enumerate(kf.split(X)):
    X_tr, y_tr = X.iloc[tr_idx], y[tr_idx]
    
    model_lgb = LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=31,
        random_state=42 + fold,
        verbose=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1
    )
    model_lgb.fit(X_tr, y_tr)
    lgb_preds += model_lgb.predict(X_test) / 5.0

    model_xgb = XGBRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=6,
        random_state=42 + fold,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,
        verbosity=0
    )
    model_xgb.fit(X_tr, y_tr)
    xgb_preds += model_xgb.predict(X_test) / 5.0

    model_cat = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.03,
        depth=6,
        random_seed=42 + fold,
        verbose=0,
        thread_count=-1
    )
    model_cat.fit(X_tr, y_tr)
    cat_preds += model_cat.predict(X_test) / 5.0

final_preds = 0.4 * lgb_preds + 0.3 * xgb_preds + 0.3 * cat_preds
final_preds = np.clip(final_preds, 0, None)

submission = pd.DataFrame({
    'id': np.arange(len(final_preds)),
    'REVENUE15': final_preds
})
submission.to_csv('predictions.csv', index=False)