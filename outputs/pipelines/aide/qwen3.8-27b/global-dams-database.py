import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

TARGET = "Res_capacityM3"
SEED = 42
WORK_DIR = "./working"
os.makedirs(WORK_DIR, exist_ok=True)

train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

# Remove target and id-like columns.
if TARGET in train.columns:
    y = train[TARGET].astype(float)
    train = train.drop(columns=[TARGET])
else:
    y = pd.Series(np.zeros(len(train), dtype=float))

if "id" in train.columns:
    train = train.drop(columns=["id"])
if "id" in test.columns:
    test = test.drop(columns=["id"])
if TARGET in test.columns:
    test = test.drop(columns=[TARGET])

# Keep only rows with a valid target value.
valid_target_mask = y.notna()
y = y[valid_target_mask].reset_index(drop=True)
train = train[valid_target_mask].reset_index(drop=True)

# Keep only columns present in both train and test.
common_cols = [c for c in train.columns if c in test.columns]
train = train[common_cols].copy()
test = test[common_cols].copy()

# Basic cleaning and type normalization.
for col in list(train.columns):
    if train[col].isna().all():
        continue

    if pd.api.types.is_bool_dtype(train[col]) or pd.api.types.is_numeric_dtype(
        train[col]
    ):
        train[col] = pd.to_numeric(train[col], errors="coerce").astype(float)
        test[col] = pd.to_numeric(test[col], errors="coerce").astype(float)
        continue

    if pd.api.types.is_string_dtype(train[col]) or train[col].dtype == object:
        non_null_count = int(train[col].notna().sum())
        if non_null_count == 0:
            continue

        n_unique = int(train[col].nunique(dropna=True))
        if n_unique <= 50:
            categories = list(train[col].dropna().unique())
            train[col] = pd.Categorical(train[col], categories=categories)
            test[col] = pd.Categorical(test[col], categories=categories)
        else:
            numeric_train = pd.to_numeric(train[col], errors="coerce")
            if numeric_train.notna().sum() >= 0.8 * non_null_count:
                train[col] = numeric_train.astype(float)
                test[col] = pd.to_numeric(test[col], errors="coerce").astype(float)
            else:
                train = train.drop(columns=[col])
                test = test.drop(columns=[col])
        continue

    # Fallback: try numeric conversion, otherwise drop unsupported column.
    numeric_train = pd.to_numeric(train[col], errors="coerce")
    non_null_count = int(train[col].notna().sum())
    if non_null_count > 0 and numeric_train.notna().sum() >= 0.8 * non_null_count:
        train[col] = numeric_train.astype(float)
        test[col] = pd.to_numeric(test[col], errors="coerce").astype(float)
    else:
        train = train.drop(columns=[col])
        test = test.drop(columns=[col])

# Drop constant or all-null features based on the training set only.
valid_cols = []
for col in train.columns:
    if train[col].notna().sum() == 0:
        continue
    if train[col].nunique(dropna=True) <= 1:
        continue
    valid_cols.append(col)

# Fallback if no usable features remain.
if len(valid_cols) == 0:
    fallback_value = float(y.mean()) if len(y) > 0 else 0.0
    oof = np.full(len(y), fallback_value, dtype=float)
    cv_rmse = float(np.sqrt(mean_squared_error(y, oof))) if len(y) > 0 else 0.0
    test_pred = np.full(len(test), fallback_value, dtype=float)
    if len(y) > 0 and y.min() >= 0:
        test_pred = np.clip(test_pred, 0.0, None)

    print(f"CV RMSE: {cv_rmse:.6f}")
    pd.DataFrame({"id": np.arange(len(test)), TARGET: test_pred}).to_csv(
        os.path.join(WORK_DIR, "submission.csv"), index=False
    )
    raise SystemExit(0)

X = train[valid_cols].copy()
X_test = test[valid_cols].copy()

params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "learning_rate": 0.05,
    "max_depth": 6,
    "min_child_weight": 3,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "lambda": 1.0,
    "alpha": 0.0,
    "tree_method": "hist",
    "random_state": SEED,
}

num_boost_round = 5000
early_stopping_rounds = 50

kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
oof_predictions = np.zeros(len(y), dtype=float)
best_iterations = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X), start=1):
    X_train_fold = X.iloc[train_idx]
    X_val_fold = X.iloc[val_idx]
    y_train_fold = y.iloc[train_idx]
    y_val_fold = y.iloc[val_idx]

    dtrain = xgb.DMatrix(X_train_fold, label=y_train_fold, enable_categorical=True)
    dval = xgb.DMatrix(X_val_fold, label=y_val_fold, enable_categorical=True)

    evals_result = {}
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        evals=[(dval, "valid")],
        early_stopping_rounds=early_stopping_rounds,
        evals_result=evals_result,
        verbose_eval=False,
    )

    best_iter = getattr(model, "best_iteration", None)
    if best_iter is None:
        best_iter = num_boost_round - 1
    best_iterations.append(int(best_iter))

    fold_pred = model.predict(dval, iteration_range=(0, int(best_iter) + 1))
    oof_predictions[val_idx] = fold_pred

cv_rmse = float(np.sqrt(mean_squared_error(y, oof_predictions)))
print(f"CV RMSE: {cv_rmse:.6f}")

# Refit on the full training set using the average best iteration.
avg_best_iter = max(1, int(np.mean(best_iterations)))
dtrain_full = xgb.DMatrix(X, label=y, enable_categorical=True)
final_model = xgb.train(
    params,
    dtrain_full,
    num_boost_round=avg_best_iter,
    verbose_eval=False,
)

dtest = xgb.DMatrix(X_test, enable_categorical=True)
test_pred = final_model.predict(dtest)

# Capacity values are expected to be non-negative; clip only when supported by train target.
if len(y) > 0 and y.min() >= 0:
    test_pred = np.clip(test_pred, 0.0, None)

submission = pd.DataFrame({"id": np.arange(len(test)), TARGET: test_pred})
submission.to_csv(os.path.join(WORK_DIR, "submission.csv"), index=False)
