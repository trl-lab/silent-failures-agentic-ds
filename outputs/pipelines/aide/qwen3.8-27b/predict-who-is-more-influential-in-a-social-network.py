import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

os.makedirs("./working", exist_ok=True)

train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

y = train.pop("Choice").astype(int).to_numpy()

a_cols = sorted(
    [c for c in train.columns if str(c).startswith("A_") and c in test.columns]
)
base_cols = []
for c in a_cols:
    base = c[2:]
    b_col = f"B_{base}"
    if b_col in train.columns and b_col in test.columns:
        base_cols.append(base)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = {}
    for base in base_cols:
        a = (
            pd.to_numeric(df[f"A_{base}"], errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype=float)
        )
        b = (
            pd.to_numeric(df[f"B_{base}"], errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype=float)
        )

        a = np.clip(a, 0.0, None)
        b = np.clip(b, 0.0, None)

        log_a = np.log1p(a)
        log_b = np.log1p(b)

        out[f"log_A_{base}"] = log_a
        out[f"log_B_{base}"] = log_b
        out[f"log_diff_{base}"] = log_a - log_b

    return pd.DataFrame(out, index=df.index)


X = build_features(train).astype(float).fillna(0.0)
X_test = build_features(test).astype(float).fillna(0.0)

params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "min_data_in_leaf": 20,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "feature_fraction": 0.8,
    "lambda_l1": 0.1,
    "lambda_l2": 0.1,
    "seed": 42,
    "num_threads": -1,
    "verbosity": -1,
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof = np.zeros(len(y), dtype=float)
best_iterations = []

for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y), start=1):
    train_data = lgb.Dataset(
        X.iloc[train_idx].to_numpy(dtype=float), label=y[train_idx]
    )
    valid_data = lgb.Dataset(
        X.iloc[valid_idx].to_numpy(dtype=float), label=y[valid_idx]
    )

    booster = lgb.train(
        params,
        train_data,
        num_boost_round=5000,
        valid_sets=[valid_data],
        valid_names=["valid"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=-1),
        ],
    )

    best_iter = getattr(booster, "best_iteration", 0)
    if not best_iter or best_iter <= 0:
        best_iter = 5000
    best_iter = int(best_iter)

    oof[valid_idx] = booster.predict(
        X.iloc[valid_idx].to_numpy(dtype=float),
        num_iteration=best_iter,
    )
    best_iterations.append(best_iter)
    print(f"Fold {fold}: best_iteration={best_iter}")

cv_accuracy = accuracy_score(y, (oof >= 0.5).astype(int))
print(f"5-fold CV accuracy: {cv_accuracy:.5f}")

final_rounds = int(max(50, min(5000, np.mean(best_iterations))))
final_data = lgb.Dataset(X.to_numpy(dtype=float), label=y)
final_booster = lgb.train(params, final_data, num_boost_round=final_rounds)

if len(X_test) > 0:
    test_proba = final_booster.predict(X_test.to_numpy(dtype=float))
else:
    test_proba = np.array([], dtype=float)

test_pred = (test_proba >= 0.5).astype(int)
submission = pd.DataFrame({"id": np.arange(len(test)), "Choice": test_pred})
submission.to_csv("./working/submission.csv", index=False)

print("Saved ./working/submission.csv")
