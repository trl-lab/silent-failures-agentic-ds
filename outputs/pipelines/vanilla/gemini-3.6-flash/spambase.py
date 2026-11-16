import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold

train_path = "/path/to/user/workspace/ds_failures/datasets/openml/spambase/data/train.parquet"
test_path = "/path/to/user/workspace/ds_failures/datasets/openml/spambase/data/test.parquet"

train = pd.read_parquet(train_path)
test = pd.read_parquet(test_path)

target_col = "class"
y_raw = train[target_col]

unique_vals = sorted(y_raw.unique())
mapping = {val: i for i, val in enumerate(unique_vals)}
y = y_raw.map(mapping).values

features = [c for c in train.columns if c != target_col]
X = train[features].copy()
X_test = test[features].copy()

for col in features:
    X[col] = pd.to_numeric(X[col], errors="coerce")
    X_test[col] = pd.to_numeric(X_test[col], errors="coerce")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

lgb_preds = np.zeros(len(X_test))
xgb_preds = np.zeros(len(X_test))
cat_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_tr, y_tr = X.iloc[train_idx], y[train_idx]
    X_va, y_va = X.iloc[val_idx], y[val_idx]

    model_lgb = LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        verbose=-1
    )
    model_lgb.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric='auc',
        callbacks=[early_stopping(100, verbose=False)]
    )
    lgb_preds += model_lgb.predict_proba(X_test)[:, 1] / skf.n_splits

    model_xgb = XGBClassifier(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        eval_metric='auc',
        early_stopping_rounds=100,
        tree_method='hist'
    )
    model_xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    xgb_preds += model_xgb.predict_proba(X_test)[:, 1] / skf.n_splits

    model_cat = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.03,
        depth=6,
        random_seed=42 + fold,
        verbose=0
    )
    model_cat.fit(X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=100)
    cat_preds += model_cat.predict_proba(X_test)[:, 1] / skf.n_splits

final_preds = (lgb_preds + xgb_preds + cat_preds) / 3.0

submission = pd.DataFrame({
    "id": np.arange(len(test)),
    "class": final_preds
})

submission.to_csv("predictions.csv", index=False)