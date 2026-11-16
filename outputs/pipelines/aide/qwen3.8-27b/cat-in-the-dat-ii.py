import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

os.makedirs("./working", exist_ok=True)

train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

y = train.pop("target").astype(int)

if "id" in train.columns:
    train = train.drop(columns=["id"])
if "id" in test.columns:
    test = test.drop(columns=["id"])
if "target" in test.columns:
    test = test.drop(columns=["target"])

test = test[train.columns]

cat_cols = [c for c in train.columns if not pd.api.types.is_numeric_dtype(train[c])]
num_cols = [c for c in train.columns if pd.api.types.is_numeric_dtype(train[c])]

for c in num_cols:
    train[c] = pd.to_numeric(train[c], errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    test[c] = pd.to_numeric(test[c], errors="coerce").replace([np.inf, -np.inf], np.nan)

SMOOTHING = 100.0


def get_categories(X):
    categories = {}
    for c in cat_cols:
        vals = X[c].dropna().unique()
        try:
            vals = np.sort(vals)
        except TypeError:
            vals = sorted(vals)
        categories[c] = pd.Index(vals)
    return categories


def encode(X, categories):
    X = X.copy()
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=categories[c])
    return X


def compute_target_stats(X_tr, y_tr):
    global_mean = float(y_tr.mean())
    stats = {}
    for c in cat_cols:
        grouped = y_tr.groupby(X_tr[c])
        counts = grouped.count()
        means = grouped.mean()
        smoothed = (counts * means + SMOOTHING * global_mean) / (counts + SMOOTHING)
        stats[c] = smoothed.astype("float32")
    return stats, global_mean


def add_target_features(X, stats, global_mean):
    X = X.copy()
    for c in cat_cols:
        mapped = pd.Series(X[c].astype(object)).map(stats[c])
        X[f"te_{c}"] = mapped.fillna(global_mean).astype("float32")
    return X


base_params = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "min_data_in_leaf": 50,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "feature_fraction": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "seed": 42,
    "n_jobs": -1,
    "verbosity": -1,
}

SEEDS = [42, 123, 456, 789, 1024]
num_boost_round = 1000

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof = np.zeros(len(train), dtype=np.float64)
best_iters = []

for fold, (tr_idx, va_idx) in enumerate(skf.split(train, y)):
    X_tr_raw = train.iloc[tr_idx]
    X_va_raw = train.iloc[va_idx]
    y_tr = y.iloc[tr_idx]
    y_va = y.iloc[va_idx]

    categories = get_categories(X_tr_raw)
    X_tr = encode(X_tr_raw, categories)
    X_va = encode(X_va_raw, categories)

    stats, global_mean = compute_target_stats(X_tr_raw, y_tr)
    X_tr = add_target_features(X_tr, stats, global_mean)
    X_va = add_target_features(X_va, stats, global_mean)

    fold_val_preds = np.zeros(len(X_va), dtype=np.float64)

    for seed in SEEDS:
        params = base_params.copy()
        params["seed"] = seed
        params["bagging_seed"] = seed
        params["feature_fraction_seed"] = seed
        params["data_random_seed"] = seed

        dtrain = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat_cols)
        dval = lgb.Dataset(
            X_va, label=y_va, categorical_feature=cat_cols, reference=dtrain
        )

        model = lgb.train(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            valid_sets=[dval],
            valid_names=["valid"],
            callbacks=[
                lgb.early_stopping(50, first_metric_only=True, verbose=False),
                lgb.log_evaluation(-1),
            ],
        )

        best_iter = model.best_iteration
        if best_iter is None or best_iter <= 0:
            best_iter = num_boost_round

        best_iters.append(best_iter)
        fold_val_preds += model.predict(X_va, num_iteration=best_iter)

    oof[va_idx] = fold_val_preds / len(SEEDS)

cv_auc = roc_auc_score(y, oof)
print(f"5-fold CV ROC AUC: {cv_auc:.6f}")

categories_full = get_categories(train)
stats_full, global_mean_full = compute_target_stats(train, y)

X_train_full = encode(train, categories_full)
X_test = encode(test, categories_full)

X_train_full = add_target_features(X_train_full, stats_full, global_mean_full)
X_test = add_target_features(X_test, stats_full, global_mean_full)

final_rounds = max(1, int(np.mean(best_iters)))
test_preds = np.zeros(len(X_test), dtype=np.float64)

for seed in SEEDS:
    params = base_params.copy()
    params["seed"] = seed
    params["bagging_seed"] = seed
    params["feature_fraction_seed"] = seed
    params["data_random_seed"] = seed

    dtrain_full = lgb.Dataset(X_train_full, label=y, categorical_feature=cat_cols)

    model = lgb.train(
        params,
        dtrain_full,
        num_boost_round=final_rounds,
        callbacks=[lgb.log_evaluation(-1)],
    )

    test_preds += model.predict(X_test, num_iteration=final_rounds)

test_preds = np.clip(test_preds / len(SEEDS), 0.0, 1.0)

submission = pd.DataFrame({"id": np.arange(len(test), dtype=int), "target": test_preds})
submission.to_csv("./working/submission.csv", index=False)
