import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")

train_path = "/path/to/user/workspace/ds_failures/datasets/strable/hypertension-control-wide/data/train.parquet"
test_path = "/path/to/user/workspace/ds_failures/datasets/strable/hypertension-control-wide/data/test.parquet"

train = pd.read_parquet(train_path)
test = pd.read_parquet(test_path)

target_col = "Estimated Percentage of Patients with Controlled Blood Pressure"

train = train.dropna(subset=[target_col]).reset_index(drop=True)

y_raw = train[target_col]
if y_raw.dtype == 'object' or str(y_raw.dtype) == 'category' or y_raw.dtype == 'bool':
    y = LabelEncoder().fit_transform(y_raw.astype(str))
else:
    if y_raw.nunique() <= 10:
        y = LabelEncoder().fit_transform(y_raw)
    else:
        y = (y_raw > y_raw.median()).astype(int).values

X_train = train.drop(columns=[target_col])
X_test = test.copy()

cat_cols = []
num_cols = []

for c in X_train.columns:
    if X_train[c].dtype == 'object' or X_train[c].dtype == 'bool' or str(X_train[c].dtype) == 'category':
        cat_cols.append(c)
    else:
        num_cols.append(c)

X_all = pd.concat([X_train, X_test], axis=0, ignore_index=True)

for c in cat_cols:
    X_all[c] = LabelEncoder().fit_transform(X_all[c].astype(str))

for c in num_cols:
    X_all[c] = pd.to_numeric(X_all[c], errors='coerce')

X_train_proc = X_all.iloc[:len(X_train)].copy()
X_test_proc = X_all.iloc[len(X_train):].copy()

n_splits = 5
try:
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    splits = list(cv.split(X_train_proc, y))
except Exception:
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    splits = list(cv.split(X_train_proc, y))

lgb_preds = np.zeros(len(X_test_proc))
xgb_preds = np.zeros(len(X_test_proc))
cat_preds = np.zeros(len(X_test_proc))

for fold, (train_idx, val_idx) in enumerate(splits):
    X_tr, y_tr = X_train_proc.iloc[train_idx], y[train_idx]
    X_va, y_va = X_train_proc.iloc[val_idx], y[val_idx]

    model_lgb = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.03,
        random_state=42 + fold,
        verbose=-1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8
    )
    model_lgb.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    lgb_preds += model_lgb.predict_proba(X_test_proc)[:, 1] / n_splits

    model_xgb = xgb.XGBClassifier(
        n_estimators=1000,
        learning_rate=0.03,
        random_state=42 + fold,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='auc',
        early_stopping_rounds=50,
        tree_method='hist'
    )
    model_xgb.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        verbose=False
    )
    xgb_preds += model_xgb.predict_proba(X_test_proc)[:, 1] / n_splits

    model_cat = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.03,
        depth=6,
        random_seed=42 + fold,
        verbose=0,
        eval_metric='AUC',
        early_stopping_rounds=50
    )
    model_cat.fit(
        X_tr, y_tr,
        eval_set=(X_va, y_va),
        use_best_model=True
    )
    cat_preds += model_cat.predict_proba(X_test_proc)[:, 1] / n_splits

final_preds = (lgb_preds + xgb_preds + cat_preds) / 3.0

submission = pd.DataFrame({
    'id': np.arange(len(test)),
    target_col: final_preds
})

submission.to_csv("predictions.csv", index=False)