import pandas as pd
import numpy as np
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold

train_path = "/path/to/user/workspace/ds_failures/datasets/strable/beer-ratings/data/train.parquet"
test_path = "/path/to/user/workspace/ds_failures/datasets/strable/beer-ratings/data/test.parquet"

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = "review_overall"

y = train_df[target_col].values
X = train_df.drop(columns=[target_col])
X_test = test_df.drop(columns=[target_col], errors='ignore')

for id_candidate in ['id', 'ID', 'Id', 'index', 'Index']:
    if id_candidate in X.columns and id_candidate not in test_df.columns:
        X = X.drop(columns=[id_candidate])

X, X_test = X.align(X_test, join='inner', axis=1)

cat_cols = []
num_cols = []

for col in X.columns:
    if pd.api.types.is_numeric_dtype(X[col]):
        num_cols.append(col)
    else:
        cat_cols.append(col)

for col in cat_cols:
    X[col] = X[col].astype(str).fillna('missing')
    X_test[col] = X_test[col].astype(str).fillna('missing')

    freq = X[col].value_counts().to_dict()
    X[f'{col}_freq'] = X[col].map(freq).fillna(0)
    X_test[f'{col}_freq'] = X_test[col].map(freq).fillna(0)

review_sub_cols = [c for c in num_cols if 'review' in c.lower() or 'rating' in c.lower() or 'score' in c.lower()]
if len(review_sub_cols) > 1:
    X['sub_mean'] = X[review_sub_cols].mean(axis=1)
    X_test['sub_mean'] = X_test[review_sub_cols].mean(axis=1)
    X['sub_std'] = X[review_sub_cols].std(axis=1).fillna(0)
    X_test['sub_std'] = X_test[review_sub_cols].std(axis=1).fillna(0)

for col in cat_cols:
    X[col] = X[col].astype('category')
    X_test[col] = X_test[col].astype('category')

kf = KFold(n_splits=5, shuffle=True, random_state=42)

lgb_preds = np.zeros(len(X_test))
cat_preds = np.zeros(len(X_test))

for train_idx, val_idx in kf.split(X, y):
    X_tr, y_tr = X.iloc[train_idx], y[train_idx]
    X_va, y_va = X.iloc[val_idx], y[val_idx]

    lgb_model = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=63,
        random_state=42,
        n_jobs=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        verbose=-1
    )
    lgb_model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]
    )
    lgb_preds += lgb_model.predict(X_test) / kf.n_splits

    X_tr_cb = X_tr.copy()
    X_va_cb = X_va.copy()
    X_te_cb = X_test.copy()

    for c in cat_cols:
        X_tr_cb[c] = X_tr_cb[c].astype(str)
        X_va_cb[c] = X_va_cb[c].astype(str)
        X_te_cb[c] = X_te_cb[c].astype(str)

    cat_model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        random_seed=42,
        verbose=0,
        cat_features=cat_cols
    )
    cat_model.fit(
        X_tr_cb, y_tr,
        eval_set=(X_va_cb, y_va),
        early_stopping_rounds=30,
        verbose=False
    )
    cat_preds += cat_model.predict(X_te_cb) / kf.n_splits

final_preds = lgb_preds * 0.5 + cat_preds * 0.5

min_y = y.min()
max_y = y.max()
final_preds = np.clip(final_preds, min_y, max_y)

sub = pd.DataFrame({
    'id': np.arange(len(final_preds)),
    'review_overall': final_preds
})
sub.to_csv('predictions.csv', index=False)