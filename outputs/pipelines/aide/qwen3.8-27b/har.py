import os
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)

WORK_DIR = "./working"
os.makedirs(WORK_DIR, exist_ok=True)

TARGET = "class"
GROUP_CANDIDATES = ["subject", "Subject", "SUBJECT"]
META_CANDIDATES = ["id", "ID", "Id"] + GROUP_CANDIDATES

train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

if TARGET not in train.columns:
    raise ValueError("Target column 'class' not found in train.parquet")

y_codes, y_uniques = pd.factorize(train[TARGET], sort=True)
num_classes = len(y_uniques)

meta_cols = [c for c in train.columns if c in META_CANDIDATES]
feature_cols = [c for c in train.columns if c not in meta_cols and c != TARGET]
feature_cols = [c for c in feature_cols if c in test.columns]

X = train[feature_cols].copy()
X_test = test[feature_cols].copy()

for df in (X, X_test):
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

drop_cols = [c for c in X.columns if X[c].isna().all()]
if drop_cols:
    X = X.drop(columns=drop_cols)
    X_test = X_test.drop(columns=drop_cols)

group_col = next((c for c in GROUP_CANDIDATES if c in train.columns), None)
groups = train[group_col].fillna(-1).values if group_col is not None else None

if groups is not None and len(np.unique(groups)) >= 5:
    try:
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
        splits = list(cv.split(X, y_codes, groups=groups))
    except Exception:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        splits = list(cv.split(X, y_codes))
else:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    splits = list(cv.split(X, y_codes))

params = {
    "objective": "multiclass",
    "num_class": num_classes,
    "metric": "multi_logloss",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "max_depth": -1,
    "min_child_samples": 20,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "seed": SEED,
    "verbosity": -1,
}

accs = []
best_iters = []

for fold, (train_idx, val_idx) in enumerate(splits, start=1):
    dtrain = lgb.Dataset(X.iloc[train_idx], label=y_codes[train_idx])
    dvalid = lgb.Dataset(X.iloc[val_idx], label=y_codes[val_idx], reference=dtrain)

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=3000,
        valid_sets=[dvalid],
        valid_names=["valid"],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
    )

    pred = model.predict(X.iloc[val_idx], num_iteration=model.best_iteration)
    pred_labels = np.argmax(pred, axis=1)
    acc = accuracy_score(y_codes[val_idx], pred_labels)

    accs.append(acc)
    if model.best_iteration is not None and model.best_iteration > 0:
        best_iters.append(model.best_iteration)

    print(f"Fold {fold} accuracy: {acc:.4f}")

mean_acc = float(np.mean(accs))
std_acc = float(np.std(accs))
print(f"Validation accuracy: {mean_acc:.4f} +/- {std_acc:.4f}")

final_rounds = int(np.clip(np.mean(best_iters), 100, 3000)) if best_iters else 1000

dtrain_full = lgb.Dataset(X, label=y_codes)
final_model = lgb.train(
    params,
    dtrain_full,
    num_boost_round=final_rounds,
    callbacks=[lgb.log_evaluation(-1)],
)

test_pred = final_model.predict(X_test)
test_pred_codes = np.argmax(test_pred, axis=1)
test_pred_labels = y_uniques[test_pred_codes]

submission = pd.DataFrame(
    {
        "id": np.arange(len(test), dtype=int),
        "class": test_pred_labels,
    }
)
submission.to_csv(os.path.join(WORK_DIR, "submission.csv"), index=False)
print(f"Saved submission to {os.path.join(WORK_DIR, 'submission.csv')}")
