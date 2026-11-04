import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

os.makedirs("./working", exist_ok=True)

TARGET = "ViolentCrimesPerPop"
DROP_COLS = ["state", "county", "community", "communityname", "fold", "id"]


def clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.drop(
        columns=[c for c in DROP_COLS + [TARGET] if c in df.columns],
        errors="ignore",
    )

    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan)
    return df


train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

y = pd.to_numeric(train[TARGET], errors="coerce").astype(float)
X = clean_frame(train)
X_test = clean_frame(test)

valid = y.notna()
X = X.loc[valid]
y = y[valid].values

common_cols = [c for c in X.columns if c in X_test.columns and X[c].notna().any()]
if len(common_cols) == 0:
    common_cols = [c for c in X.columns if X[c].notna().any()]

X = X[common_cols].astype(float)
X_test = X_test.reindex(columns=common_cols, fill_value=np.nan).astype(float)

base_params = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "seed": 42,
    "verbosity": -1,
}

candidate_overrides = [
    {},
    {"learning_rate": 0.03},
    {"learning_rate": 0.03, "num_leaves": 15},
    {
        "learning_rate": 0.03,
        "num_leaves": 63,
        "min_data_in_leaf": 40,
        "lambda_l2": 10.0,
    },
    {"learning_rate": 0.05, "num_leaves": 15, "min_data_in_leaf": 10},
    {
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_data_in_leaf": 40,
        "lambda_l2": 10.0,
    },
    {"learning_rate": 0.05, "feature_fraction": 0.7, "bagging_fraction": 0.7},
    {"learning_rate": 0.05, "feature_fraction": 0.9, "lambda_l2": 0.1},
    {"learning_rate": 0.03, "num_leaves": 31, "min_data_in_leaf": 10, "lambda_l2": 0.1},
    {
        "learning_rate": 0.03,
        "num_leaves": 31,
        "min_data_in_leaf": 40,
        "lambda_l2": 10.0,
        "feature_fraction": 0.9,
    },
]


def evaluate_candidate(params: dict) -> tuple[float, float]:
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.empty(len(y), dtype=float)
    best_iterations = []

    for train_idx, valid_idx in kf.split(X):
        dtrain = lgb.Dataset(X.iloc[train_idx], label=y[train_idx])
        dvalid = lgb.Dataset(X.iloc[valid_idx], label=y[valid_idx], reference=dtrain)

        booster = lgb.train(
            params=params,
            train_set=dtrain,
            num_boost_round=2000,
            valid_sets=[dvalid],
            valid_names=["valid"],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(-1),
            ],
        )

        best_iter = (
            int(booster.best_iteration)
            if getattr(booster, "best_iteration", 0)
            else 2000
        )
        best_iterations.append(best_iter)
        oof[valid_idx] = booster.predict(X.iloc[valid_idx], num_iteration=best_iter)

    rmse = float(np.sqrt(mean_squared_error(y, oof)))
    avg_iter = float(np.mean(best_iterations))
    return rmse, avg_iter


best_rmse = np.inf
best_params = base_params.copy()
best_avg_iter = 2000

for i, overrides in enumerate(candidate_overrides, start=1):
    params = base_params.copy()
    params.update(overrides)

    rmse, avg_iter = evaluate_candidate(params)
    print(
        f"Candidate {i}: RMSE={rmse:.5f}, avg_iter={avg_iter:.1f}, overrides={overrides}"
    )

    if rmse < best_rmse:
        best_rmse = rmse
        best_params = params
        best_avg_iter = avg_iter

print(f"Best CV RMSE: {best_rmse:.5f}")

final_rounds = int(round(best_avg_iter))
if final_rounds <= 0:
    final_rounds = 2000

dtrain_full = lgb.Dataset(X, label=y)
booster_full = lgb.train(
    params=best_params,
    train_set=dtrain_full,
    num_boost_round=final_rounds,
    callbacks=[lgb.log_evaluation(-1)],
)

test_pred = booster_full.predict(X_test)
lower = float(y.min()) if y.min() < 0 else 0.0
test_pred = np.clip(test_pred, lower, None)

submission = pd.DataFrame(
    {
        "id": np.arange(len(test_pred)),
        TARGET: test_pred,
    }
)

submission.to_csv("./working/submission.csv", index=False)
print(f"Saved submission with {len(submission)} rows")
