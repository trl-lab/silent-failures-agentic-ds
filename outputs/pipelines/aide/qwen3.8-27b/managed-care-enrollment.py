import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

TARGET = "Total Enrollment"
ID_COL = "id"
WORK_DIR = "./working"
os.makedirs(WORK_DIR, exist_ok=True)

train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

train[TARGET] = pd.to_numeric(train[TARGET], errors="coerce").replace(
    [np.inf, -np.inf], np.nan
)
train = train.dropna(subset=[TARGET]).reset_index(drop=True)

feature_cols = [
    c for c in train.columns if c != TARGET and c != ID_COL and c in test.columns
]

X = train[feature_cols].copy()
y = train[TARGET].astype(float)
X_test = test[feature_cols].copy()

cat_cols = [
    c
    for c in feature_cols
    if pd.api.types.is_object_dtype(X[c])
    or pd.api.types.is_string_dtype(X[c])
    or X[c].dtype.name == "category"
]

for c in feature_cols:
    if c not in cat_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
        X_test[c] = pd.to_numeric(X_test[c], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )

for c in cat_cols:
    X[c] = X[c].astype(str).astype("category")
    X_test[c] = pd.Categorical(X_test[c].astype(str), categories=X[c].cat.categories)

mean_target = float(y.mean()) if len(y) > 0 else 0.0
clip_min = 0.0 if len(y) > 0 and y.min() >= 0 else None


def clean_predictions(pred):
    pred = np.asarray(pred, dtype=float)
    pred = np.nan_to_num(pred, nan=mean_target, posinf=mean_target, neginf=mean_target)
    if clip_min is not None:
        pred = np.maximum(pred, clip_min)
    return pred


seeds = [42, 7, 123]
base_params = {
    "objective": "regression_l2",
    "metric": "rmse",
    "learning_rate": 0.03,
    "num_leaves": 31,
    "min_child_samples": 10,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "seed": 42,
    "verbose": -1,
}

max_rounds = 5000
early_stopping_rounds = 100

if len(feature_cols) == 0:
    if len(y) >= 2:
        n_splits = min(5, len(y))
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        rmses = []
        for _, va_idx in kf.split(X):
            pred = np.full(len(va_idx), mean_target)
            rmses.append(np.sqrt(mean_squared_error(y.iloc[va_idx], pred)))
    else:
        rmses = [0.0]

    cv_rmse = float(np.mean(rmses))
    print(f"Validation RMSE: {cv_rmse:.6f}")
    pred_test = np.full(len(X_test), mean_target)

else:
    if len(y) < 2:
        print("Validation RMSE: 0.000000")
        pred_test = np.full(len(X_test), mean_target)
    else:
        n_splits = min(5, len(y))
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        rmses = []
        best_iters_by_seed = {seed: [] for seed in seeds}

        for tr_idx, va_idx in kf.split(X):
            fold_pred = np.zeros(len(va_idx), dtype=float)

            for seed in seeds:
                params = base_params.copy()
                params["seed"] = seed
                params["feature_fraction_seed"] = seed
                params["data_sample_seed"] = seed

                dtrain = lgb.Dataset(
                    X.iloc[tr_idx], y.iloc[tr_idx], categorical_feature=cat_cols
                )
                dvalid = lgb.Dataset(
                    X.iloc[va_idx], y.iloc[va_idx], categorical_feature=cat_cols
                )

                model = lgb.train(
                    params,
                    dtrain,
                    num_boost_round=max_rounds,
                    valid_sets=[dvalid],
                    valid_names=["valid"],
                    callbacks=[
                        lgb.early_stopping(early_stopping_rounds),
                        lgb.log_evaluation(-1),
                    ],
                )

                best_iter = max(1, int(model.best_iteration))
                best_iters_by_seed[seed].append(best_iter)
                fold_pred += model.predict(X.iloc[va_idx], num_iteration=best_iter)

            fold_pred /= len(seeds)
            fold_pred = clean_predictions(fold_pred)
            rmses.append(np.sqrt(mean_squared_error(y.iloc[va_idx], fold_pred)))

        cv_rmse = float(np.mean(rmses))
        print(f"Validation RMSE: {cv_rmse:.6f}")

        if len(X_test) == 0:
            pred_test = np.array([], dtype=float)
        else:
            pred_test = np.zeros(len(X_test), dtype=float)
            dtrain_full = lgb.Dataset(X, y, categorical_feature=cat_cols)

            for seed in seeds:
                params = base_params.copy()
                params["seed"] = seed
                params["feature_fraction_seed"] = seed
                params["data_sample_seed"] = seed

                final_rounds = int(
                    np.clip(np.mean(best_iters_by_seed[seed]), 1, max_rounds)
                )
                final_model = lgb.train(
                    params,
                    dtrain_full,
                    num_boost_round=final_rounds,
                    callbacks=[lgb.log_evaluation(-1)],
                )
                pred_test += final_model.predict(X_test, num_iteration=final_rounds)

            pred_test /= len(seeds)
            pred_test = clean_predictions(pred_test)

submission = pd.DataFrame(
    {
        "id": np.arange(len(X_test), dtype=int),
        "Total Enrollment": pred_test,
    }
)
submission.to_csv(os.path.join(WORK_DIR, "submission.csv"), index=False)
