import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

TARGET = "serious_ae"
WORK_DIR = "./working"
os.makedirs(WORK_DIR, exist_ok=True)


def parse_binary_target(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return (s.astype(float) > 0).astype(int)

    s_str = s.astype(str).str.strip().str.lower()
    mapping = {
        "1": 1,
        "0": 0,
        "true": 1,
        "false": 0,
        "yes": 1,
        "no": 0,
        "y": 1,
        "n": 0,
        "t": 1,
        "f": 0,
    }
    mapped = s_str.map(mapping)
    if mapped.notna().all():
        return mapped.astype(int)

    return (pd.to_numeric(s, errors="coerce").fillna(0) > 0).astype(int)


train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

train = train.dropna(subset=[TARGET]).reset_index(drop=True)
y = parse_binary_target(train[TARGET])

drop_names = {
    "id",
    "index",
    "row_id",
    "patient_id",
    "event_id",
    "report_id",
    "case_id",
}

feature_cols = [
    c
    for c in train.columns
    if c != TARGET and str(c).lower() not in drop_names and c in test.columns
]

X_train = train[feature_cols].copy()
X_test = test[feature_cols].copy()

cat_cols = []
num_cols = []

for c in feature_cols:
    s = X_train[c]

    if pd.api.types.is_bool_dtype(s) or pd.api.types.is_numeric_dtype(s):
        num_cols.append(c)
        continue

    combined = pd.concat([s, X_test[c]], ignore_index=True)
    non_null = int(combined.notna().sum())

    if non_null > 0:
        converted = pd.to_numeric(combined, errors="coerce")
        if int(converted.notna().sum()) >= 0.8 * non_null:
            X_train[c] = pd.to_numeric(s, errors="coerce")
            X_test[c] = pd.to_numeric(X_test[c], errors="coerce")
            num_cols.append(c)
            continue

    cat_cols.append(c)

for c in cat_cols:
    cats = X_train[c].dropna().unique()
    dtype = pd.CategoricalDtype(categories=cats)
    X_train[c] = X_train[c].astype(dtype)
    X_test[c] = X_test[c].astype(dtype)

keep = []
for c in feature_cols:
    s = X_train[c]
    if int(s.notna().sum()) == 0:
        continue
    if int(s.nunique(dropna=True)) <= 1:
        continue
    keep.append(c)

feature_cols = keep
X_train = X_train[feature_cols]
X_test = X_test[feature_cols]
cat_cols = [c for c in cat_cols if c in feature_cols]
num_cols = [c for c in num_cols if c in feature_cols]

if y.nunique() < 2 or len(feature_cols) == 0 or len(test) == 0:
    cv_auc = 0.5
    test_pred = np.full(len(test), 0.5)
else:
    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 100,
        "subsample": 0.9,
        "subsample_freq": 1,
        "colsample_bytree": 0.9,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "is_unbalance": True,
        "random_state": 42,
        "verbosity": -1,
    }

    n = len(y)
    pos = int(y.sum())
    neg = n - pos
    min_class = min(pos, neg)

    splits = []
    if min_class >= 5:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        splits = list(skf.split(X_train, y))
    elif min_class >= 2:
        skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
        splits = list(skf.split(X_train, y))
    elif min_class == 1 and n >= 2:
        pos_idx = np.where(y.values == 1)[0]
        neg_idx = np.where(y.values == 0)[0]
        if len(pos_idx) > 0 and len(neg_idx) > 0:
            va_idx = np.array([pos_idx[0], neg_idx[0]])
            tr_idx = np.setdiff1d(np.arange(n), va_idx)
            if len(tr_idx) > 0:
                splits = [(tr_idx, va_idx)]

    oof = np.full(n, np.nan)
    fold_aucs = []
    best_iters = []
    categorical_feature = cat_cols if cat_cols else None

    for tr_idx, va_idx in splits:
        X_tr = X_train.iloc[tr_idx]
        X_va = X_train.iloc[va_idx]
        y_tr = y.iloc[tr_idx]
        y_va = y.iloc[va_idx]

        if len(np.unique(y_va.values)) < 2:
            continue

        train_ds = lgb.Dataset(
            X_tr,
            label=y_tr,
            categorical_feature=categorical_feature,
            free_raw_data=False,
        )
        valid_ds = lgb.Dataset(
            X_va,
            label=y_va,
            reference=train_ds,
            categorical_feature=categorical_feature,
            free_raw_data=False,
        )

        model = lgb.train(
            params,
            train_ds,
            num_boost_round=1000,
            valid_sets=[valid_ds],
            callbacks=[
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(-1),
            ],
        )

        best_it = int(model.best_iteration) if model.best_iteration > 0 else 100
        best_iters.append(best_it)

        pred_va = model.predict(X_va, num_iteration=best_it)
        oof[va_idx] = pred_va
        fold_aucs.append(roc_auc_score(y_va, pred_va))

    if len(fold_aucs) > 0 and not np.isnan(oof).any():
        cv_auc = roc_auc_score(y, oof)
    elif len(fold_aucs) > 0:
        cv_auc = float(np.mean(fold_aucs))
    else:
        cv_auc = 0.5

    final_rounds = int(np.clip(np.mean(best_iters), 50, 1000)) if best_iters else 300

    train_ds = lgb.Dataset(
        X_train,
        label=y,
        categorical_feature=categorical_feature,
        free_raw_data=False,
    )
    final_model = lgb.train(
        params,
        train_ds,
        num_boost_round=final_rounds,
        callbacks=[lgb.log_evaluation(-1)],
    )

    test_pred = final_model.predict(X_test)
    test_pred = np.clip(test_pred, 0.0, 1.0)

submission = pd.DataFrame(
    {
        "id": np.arange(len(test)),
        "serious_ae": test_pred,
    }
)
submission.to_csv(os.path.join(WORK_DIR, "submission.csv"), index=False)

print(f"Validation AUC (out-of-fold): {cv_auc:.6f}")
