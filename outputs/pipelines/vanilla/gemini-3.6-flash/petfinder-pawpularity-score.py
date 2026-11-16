import os
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge

train_path = "/path/to/user/workspace/ds_failures/datasets/mlebench/petfinder-pawpularity-score/data/train.parquet"
test_path = "/path/to/user/workspace/ds_failures/datasets/mlebench/petfinder-pawpularity-score/data/test.parquet"

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = "Pawpularity"
y = train_df[target_col].values.astype(np.float32)

drop_cols = [target_col, "Id", "id", "path", "image_path", "filename"]
feature_cols = [c for c in train_df.columns if c not in drop_cols]

X = train_df[feature_cols].copy()
X_test = test_df[feature_cols].copy()

binary_cols = [c for c in feature_cols if set(X[c].dropna().unique()).issubset({0, 1, 0.0, 1.0})]
if len(binary_cols) > 0:
    X['meta_sum'] = X[binary_cols].sum(axis=1)
    X['meta_mean'] = X[binary_cols].mean(axis=1)
    X_test['meta_sum'] = X_test[binary_cols].sum(axis=1)
    X_test['meta_mean'] = X_test[binary_cols].mean(axis=1)

categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
numeric_cols = [c for c in X.columns if c not in categorical_cols]

for col in numeric_cols:
    med = X[col].median() if not np.isnan(X[col].median()) else 0
    X[col] = X[col].fillna(med)
    X_test[col] = X_test[col].fillna(med)

for col in categorical_cols:
    X[col] = X[col].astype(str).fillna("missing")
    X_test[col] = X_test[col].astype(str).fillna("missing")
    freq = X[col].value_counts().to_dict()
    X[col + '_freq'] = X[col].map(freq).fillna(0)
    X_test[col + '_freq'] = X_test[col].map(freq).fillna(0)

X_all = pd.concat([X, X_test], axis=0, ignore_index=True)
if categorical_cols:
    X_all = pd.get_dummies(X_all, columns=categorical_cols, drop_first=True)

X_proc = X_all.iloc[:len(train_df)].copy()
X_test_proc = X_all.iloc[len(train_df):].copy()

X_proc = X_proc.apply(pd.to_numeric, errors='coerce').fillna(0)
X_test_proc = X_test_proc.apply(pd.to_numeric, errors='coerce').fillna(0)

kf = KFold(n_splits=10, shuffle=True, random_state=42)

lgb_preds = np.zeros(len(X_test_proc))
xgb_preds = np.zeros(len(X_test_proc))
cat_preds = np.zeros(len(X_test_proc))
ridge_preds = np.zeros(len(X_test_proc))

for fold, (train_idx, val_idx) in enumerate(kf.split(X_proc, y)):
    X_tr, y_tr = X_proc.iloc[train_idx], y[train_idx]
    X_va, y_va = X_proc.iloc[val_idx], y[val_idx]

    model_lgb = lgb.LGBMRegressor(
        n_estimators=300,
        learning_rate=0.015,
        num_leaves=15,
        random_state=42 + fold,
        verbosity=-1
    )
    model_lgb.fit(X_tr, y_tr)
    lgb_preds += model_lgb.predict(X_test_proc) / kf.n_splits

    model_xgb = xgb.XGBRegressor(
        n_estimators=300,
        learning_rate=0.015,
        max_depth=3,
        random_state=42 + fold,
        verbosity=0
    )
    model_xgb.fit(X_tr, y_tr)
    xgb_preds += model_xgb.predict(X_test_proc) / kf.n_splits

    model_cat = CatBoostRegressor(
        iterations=300,
        learning_rate=0.015,
        depth=4,
        random_seed=42 + fold,
        verbose=0
    )
    model_cat.fit(X_tr, y_tr)
    cat_preds += model_cat.predict(X_test_proc) / kf.n_splits

    model_ridge = Ridge(alpha=10.0)
    model_ridge.fit(X_tr, y_tr)
    ridge_preds += model_ridge.predict(X_test_proc) / kf.n_splits

final_preds = 0.35 * lgb_preds + 0.35 * cat_preds + 0.20 * xgb_preds + 0.10 * ridge_preds
final_preds = np.clip(final_preds, 1.0, 100.0)

submission = pd.DataFrame({
    'id': np.arange(len(test_df)),
    'Pawpularity': final_preds
})

submission.to_csv('predictions.csv', index=False)