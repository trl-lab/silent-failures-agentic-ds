import numpy as np
import pandas as pd

from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import KFold
import warnings

warnings.filterwarnings("ignore")

train_path = "/path/to/user/workspace/ds_failures/datasets/strable/college-creditcard-marketing/data/train.parquet"
test_path = "/path/to/user/workspace/ds_failures/datasets/strable/college-creditcard-marketing/data/test.parquet"

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = "PAYMENTS_BY_ISSUER"

y = train_df[target_col].values.astype(np.float64)
X_train = train_df.drop(columns=[target_col])

if target_col in test_df.columns:
    X_test = test_df.drop(columns=[target_col])
else:
    X_test = test_df.copy()

id_cols = [c for c in X_train.columns if c.lower() in ["id", "index", "row_id"]]
if id_cols:
    X_train = X_train.drop(columns=id_cols)
    X_test = X_test.drop(columns=id_cols)

for c in X_train.columns:
    if pd.api.types.is_datetime64_any_dtype(X_train[c]):
        for dt_df in [X_train, X_test]:
            dt_df[f"{c}_year"] = dt_df[c].dt.year
            dt_df[f"{c}_month"] = dt_df[c].dt.month
            dt_df[f"{c}_day"] = dt_df[c].dt.day
            dt_df[f"{c}_dayofweek"] = dt_df[c].dt.dayofweek
        X_train = X_train.drop(columns=[c])
        X_test = X_test.drop(columns=[c])

cat_cols = []
num_cols = []

for c in X_train.columns:
    if X_train[c].dtype == 'object' or X_train[c].dtype.name == 'category' or X_train[c].dtype == 'string':
        cat_cols.append(c)
    else:
        num_cols.append(c)

for c in cat_cols:
    combined = pd.concat([X_train[c], X_test[c]], axis=0).astype(str).fillna("Missing")
    labels, _ = pd.factorize(combined, sort=True)
    X_train[c] = labels[:len(X_train)]
    X_test[c] = labels[len(X_train):]

for c in num_cols:
    X_train[c] = pd.to_numeric(X_train[c], errors="coerce").astype(np.float64)
    X_test[c] = pd.to_numeric(X_test[c], errors="coerce").astype(np.float64)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

lgb_preds = np.zeros(len(X_test))
cat_preds = np.zeros(len(X_test))
xgb_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y)):
    X_tr, y_tr = X_train.iloc[train_idx], y[train_idx]
    X_va, y_va = X_train.iloc[val_idx], y[val_idx]
    
    model_lgb = LGBMRegressor(
        n_estimators=600,
        learning_rate=0.03,
        num_leaves=31,
        random_state=42 + fold,
        n_jobs=-1,
        verbose=-1
    )
    model_lgb.fit(X_tr, y_tr)
    lgb_preds += model_lgb.predict(X_test) / kf.n_splits

    model_cat = CatBoostRegressor(
        iterations=600,
        learning_rate=0.03,
        depth=6,
        random_seed=42 + fold,
        verbose=0
    )
    model_cat.fit(X_tr, y_tr)
    cat_preds += model_cat.predict(X_test) / kf.n_splits

    model_xgb = XGBRegressor(
        n_estimators=600,
        learning_rate=0.03,
        max_depth=6,
        random_state=42 + fold,
        n_jobs=-1,
        verbosity=0
    )
    model_xgb.fit(X_tr, y_tr)
    xgb_preds += model_xgb.predict(X_test) / kf.n_splits

final_preds = (lgb_preds + cat_preds + xgb_preds) / 3.0

submission = pd.DataFrame({
    "id": np.arange(len(final_preds)),
    "PAYMENTS_BY_ISSUER": final_preds
})

submission.to_csv("predictions.csv", index=False)