import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error

# Ensure output directory exists
os.makedirs("./working", exist_ok=True)

# Load data
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

# Normalize column names
train.columns = [str(c).strip() for c in train.columns]
test.columns = [str(c).strip() for c in test.columns]

# Target and columns to exclude
target = "shares"
exclude_cols = {"url", "id", "timedelta", target, "Unnamed: 0"}

# Use only feature columns present in both train and test
feature_cols = [c for c in train.columns if c not in exclude_cols and c in test.columns]

# Prepare features and target
X = train[feature_cols].copy()
y = pd.to_numeric(train[target], errors="coerce")
X_test = test[feature_cols].copy()

# Clean target
valid_target_mask = y.notna() & np.isfinite(y.values)
X = X.loc[valid_target_mask].reset_index(drop=True)
y = y.loc[valid_target_mask].reset_index(drop=True)
y = y.clip(lower=0.0)

# Clean features: coerce to numeric and replace infinities with NaN
for col in feature_cols:
    X[col] = pd.to_numeric(X[col], errors="coerce")
    X_test[col] = pd.to_numeric(X_test[col], errors="coerce")

X.replace([np.inf, -np.inf], np.nan, inplace=True)
X_test.replace([np.inf, -np.inf], np.nan, inplace=True)

# Use float32 for LightGBM
X = X.astype(np.float32)
X_test = X_test.astype(np.float32)
y = y.astype(np.float32)

# Base LightGBM parameters, same as previous solution
base_params = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.01,
    "num_leaves": 63,
    "max_depth": -1,
    "min_child_samples": 20,
    "subsample": 0.9,
    "subsample_freq": 1,
    "colsample_bytree": 0.9,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "num_threads": -1,
    "verbose": -1,
}

# Atomic improvement: average five independently seeded LightGBM models
seeds = [42, 123, 456, 789, 1024]
num_boost_round = 10000
early_stopping_rounds = 200

kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(X), dtype=np.float64)
best_iterations_by_seed = {seed: [] for seed in seeds}

for fold, (train_idx, val_idx) in enumerate(kf.split(X), start=1):
    X_train_fold = X.iloc[train_idx]
    y_train_fold = y.iloc[train_idx]
    X_val_fold = X.iloc[val_idx]
    y_val_fold = y.iloc[val_idx]

    fold_best_iters = []

    for seed in seeds:
        params = base_params.copy()
        params.update(
            {
                "random_seed": seed,
                "bagging_seed": seed,
                "feature_fraction_seed": seed,
                "data_random_seed": seed,
            }
        )

        train_dataset = lgb.Dataset(
            X_train_fold, label=y_train_fold, free_raw_data=False
        )
        val_dataset = lgb.Dataset(X_val_fold, label=y_val_fold, free_raw_data=False)

        model = lgb.train(
            params,
            train_dataset,
            num_boost_round=num_boost_round,
            valid_sets=[val_dataset],
            valid_names=["val"],
            callbacks=[
                lgb.early_stopping(early_stopping_rounds, verbose=False),
                lgb.log_evaluation(-1),
            ],
        )

        best_iter = model.best_iteration
        if best_iter is None or best_iter <= 0:
            best_iter = 1
            fold_pred = np.full(
                len(val_idx), float(y_train_fold.mean()), dtype=np.float64
            )
        else:
            fold_pred = model.predict(X_val_fold, num_iteration=best_iter).astype(
                np.float64
            )

        oof_preds[val_idx] += fold_pred
        best_iterations_by_seed[seed].append(best_iter)
        fold_best_iters.append(best_iter)

    print(f"Fold {fold} average best iteration: {np.mean(fold_best_iters):.1f}")

# Average the five seeded out-of-fold predictions
oof_preds /= len(seeds)

cv_rmse = root_mean_squared_error(y, oof_preds)
print(f"5-fold CV RMSE: {cv_rmse:.4f}")

# Retrain on full training data for each seed using that seed's average CV best iteration
full_dataset = lgb.Dataset(X, label=y, free_raw_data=False)
test_preds = np.zeros(len(X_test), dtype=np.float64)

for seed in seeds:
    params = base_params.copy()
    params.update(
        {
            "random_seed": seed,
            "bagging_seed": seed,
            "feature_fraction_seed": seed,
            "data_random_seed": seed,
        }
    )

    final_num_boost_round = int(
        np.clip(np.mean(best_iterations_by_seed[seed]), 1, num_boost_round)
    )

    final_model = lgb.train(
        params,
        full_dataset,
        num_boost_round=final_num_boost_round,
        callbacks=[lgb.log_evaluation(-1)],
    )

    seed_preds = final_model.predict(
        X_test, num_iteration=final_num_boost_round
    ).astype(np.float64)
    test_preds += seed_preds

# Average final seeded predictions
test_preds /= len(seeds)
test_preds = np.nan_to_num(test_preds, nan=0.0, posinf=0.0, neginf=0.0)
test_preds = np.clip(test_preds, 0.0, None)

# Save submission
submission = pd.DataFrame(
    {
        "id": np.arange(len(test), dtype=int),
        "shares": test_preds,
    }
)
submission.to_csv("./working/submission.csv", index=False)
