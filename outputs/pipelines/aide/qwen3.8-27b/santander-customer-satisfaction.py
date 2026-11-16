import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

SEED = 42
SEEDS = [42, 123, 456, 789, 1024]
os.makedirs("./working", exist_ok=True)

train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

y = train.pop("TARGET")
train = train.drop(columns=["ID"], errors="ignore")
test = test.drop(columns=["ID", "TARGET"], errors="ignore")

mask = y.notna()
y = y[mask].reset_index(drop=True)
train = train[mask].reset_index(drop=True)

train = train.apply(pd.to_numeric, errors="coerce")
test = test.apply(pd.to_numeric, errors="coerce")

features = [c for c in train.columns if c in test.columns]
train = train[features]
test = test[features]

keep = []
for c in features:
    s = train[c]
    if s.notna().sum() > 0 and s.nunique(dropna=True) > 1:
        keep.append(c)

train = train[keep]
test = test[keep]

train = train.replace([np.inf, -np.inf], np.nan).astype(np.float32)
test = test.replace([np.inf, -np.inf], np.nan).astype(np.float32)

X = train.values
y = y.values.astype(np.int8)
X_test = test.values

base_params = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "max_depth": -1,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "min_data_in_leaf": 50,
    "lambda_l1": 1.0,
    "lambda_l2": 1.0,
    "verbosity": -1,
    "seed": SEED,
    "nthread": -1,
}

num_boost_round = 2000
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
folds = list(skf.split(X, y))

oof_avg = np.zeros(len(y), dtype=np.float64)
test_predictions = []

for seed in SEEDS:
    params = base_params.copy()
    params["seed"] = seed

    oof = np.zeros(len(y), dtype=np.float64)
    best_iterations = []

    for fold, (train_idx, valid_idx) in enumerate(folds, start=1):
        X_tr, X_va = X[train_idx], X[valid_idx]
        y_tr, y_va = y[train_idx], y[valid_idx]

        train_set = lgb.Dataset(
            X_tr, label=y_tr, feature_name=keep, free_raw_data=False
        )
        valid_set = lgb.Dataset(
            X_va,
            label=y_va,
            feature_name=keep,
            reference=train_set,
            free_raw_data=False,
        )

        callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)]

        model = lgb.train(
            params,
            train_set,
            num_boost_round=num_boost_round,
            valid_sets=[valid_set],
            valid_names=["valid"],
            callbacks=callbacks,
        )

        best_iter = model.best_iteration
        if best_iter is None or best_iter <= 0:
            best_iter = num_boost_round

        oof[valid_idx] = model.predict(X_va, num_iteration=best_iter)
        best_iterations.append(best_iter)

    seed_auc = roc_auc_score(y, oof)
    print(f"Seed {seed} OOF ROC AUC: {seed_auc:.6f}")
    oof_avg += oof

    final_rounds = max(1, int(np.mean(best_iterations)))
    final_set = lgb.Dataset(X, label=y, feature_name=keep)
    final_model = lgb.train(params, final_set, num_boost_round=final_rounds)
    test_predictions.append(final_model.predict(X_test))

oof_avg /= len(SEEDS)
validation_auc = roc_auc_score(y, oof_avg)
print(f"Validation ROC AUC: {validation_auc:.6f}")

test_pred = np.mean(test_predictions, axis=0)
submission = pd.DataFrame(
    {
        "id": np.arange(len(test_pred), dtype=int),
        "TARGET": test_pred,
    }
)
submission.to_csv("./working/submission.csv", index=False)
