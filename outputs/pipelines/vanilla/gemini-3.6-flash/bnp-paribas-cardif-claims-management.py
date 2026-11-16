import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb

train_path = "/path/to/user/workspace/ds_failures/datasets/kaggle/bnp-paribas-cardif-claims-management/data/train.parquet"
test_path = "/path/to/user/workspace/ds_failures/datasets/kaggle/bnp-paribas-cardif-claims-management/data/test.parquet"

train = pd.read_parquet(train_path)
test = pd.read_parquet(test_path)

target_col = "target"
y = train[target_col].values
X_train = train.drop(columns=[target_col])
X_test = test.copy()

for col in ["ID", "id"]:
    if col in X_train.columns:
        X_train = X_train.drop(columns=[col])
    if col in X_test.columns:
        X_test = X_test.drop(columns=[col])

X_train["num_nulls"] = X_train.isnull().sum(axis=1)
X_test["num_nulls"] = X_test.isnull().sum(axis=1)

cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()

for col in cat_cols:
    train_str = X_train[col].astype(str)
    test_str = X_test[col].astype(str)

    freq = pd.concat([train_str, test_str]).value_counts().to_dict()
    X_train[col + "_freq"] = train_str.map(freq).fillna(-1)
    X_test[col + "_freq"] = test_str.map(freq).fillna(-1)

    combined = pd.concat([train_str, test_str], axis=0)
    labels, _ = pd.factorize(combined)
    X_train[col] = labels[: len(X_train)]
    X_test[col] = labels[len(X_train) :]

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

preds_lgb = np.zeros(len(X_test))
preds_xgb = np.zeros(len(X_test))

lgb_params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "learning_rate": 0.03,
    "num_leaves": 31,
    "max_depth": -1,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "n_estimators": 1000,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}

xgb_params = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "learning_rate": 0.03,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "n_estimators": 1000,
    "early_stopping_rounds": 50,
    "random_state": 42,
    "n_jobs": -1,
    "tree_method": "hist",
}

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y)):
    X_tr, y_tr = X_train.iloc[train_idx], y[train_idx]
    X_va, y_va = X_train.iloc[val_idx], y[val_idx]

    model_lgb = lgb.LGBMClassifier(**lgb_params)
    model_lgb.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    preds_lgb += model_lgb.predict_proba(X_test)[:, 1] / skf.n_splits

    model_xgb = xgb.XGBClassifier(**xgb_params)
    model_xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    preds_xgb += model_xgb.predict_proba(X_test)[:, 1] / skf.n_splits

final_preds = 0.5 * preds_lgb + 0.5 * preds_xgb
final_preds = np.clip(final_preds, 1e-15, 1 - 1e-15)

submission = pd.DataFrame({"id": np.arange(len(X_test)), "target": final_preds})
submission.to_csv("predictions.csv", index=False)