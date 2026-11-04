import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error

os.makedirs("./working", exist_ok=True)

TARGET = "NADAC Per Unit"
ID_COL = "id"
SEED = 42

train = pd.read_parquet("./input/train.parquet").reset_index(drop=True)
test = pd.read_parquet("./input/test.parquet").reset_index(drop=True)

# Remove explicit id columns from features; submission id will be the row index.
for df in (train, test):
    if ID_COL in df.columns:
        df.drop(columns=[ID_COL], inplace=True)

if TARGET not in train.columns:
    raise ValueError(f"Target column {TARGET!r} not found in train data.")

# Keep only rows with finite target values.
train[TARGET] = pd.to_numeric(train[TARGET], errors="coerce")
train = train[np.isfinite(train[TARGET])].reset_index(drop=True)

# Use features present in both train and test to avoid leakage or mismatch.
feature_cols = [col for col in train.columns if col != TARGET and col in test.columns]


def preprocess_features(train_df, test_df, cols):
    X_train = train_df[cols].copy()
    X_test = test_df[cols].copy()
    categorical_cols = []

    for col in cols:
        tr = X_train[col]
        te = X_test[col]

        # Boolean features are safe as numeric 0/1.
        if pd.api.types.is_bool_dtype(tr) or pd.api.types.is_bool_dtype(te):
            X_train[col] = tr.astype(int)
            X_test[col] = te.astype(int)
            continue

        # Numeric features: keep as numeric and replace infinities with NaN.
        if pd.api.types.is_numeric_dtype(tr) and pd.api.types.is_numeric_dtype(te):
            X_train[col] = tr.replace([np.inf, -np.inf], np.nan)
            X_test[col] = te.replace([np.inf, -np.inf], np.nan)
            continue

        # Datetime features: convert to days since epoch as numeric.
        if pd.api.types.is_datetime64_any_dtype(
            tr
        ) or pd.api.types.is_datetime64_any_dtype(te):
            X_train[col] = (
                pd.to_numeric(X_train[col].astype("int64"), errors="coerce") // 10**9
            )
            X_test[col] = (
                pd.to_numeric(X_test[col].astype("int64"), errors="coerce") // 10**9
            )
            continue

        # Object/category/string features: try numeric conversion if mostly numeric.
        combined = pd.concat([tr, te], ignore_index=True)
        if pd.api.types.is_categorical_dtype(combined):
            categorical_cols.append(col)
            continue

        numeric_like = pd.to_numeric(combined, errors="coerce")
        non_null = combined.dropna()
        non_empty = non_null.astype(str).str.strip().ne("")
        mostly_numeric = (
            len(non_null) > 0
            and non_empty.mean() > 0.9
            and numeric_like.notna().mean() > 0.9
        )

        if mostly_numeric:
            X_train[col] = pd.to_numeric(tr, errors="coerce")
            X_test[col] = pd.to_numeric(te, errors="coerce")
        else:
            categorical_cols.append(col)

    # Drop columns that are all NaN or constant in the training data.
    keep_cols = []
    for col in cols:
        combined = pd.concat([X_train[col], X_test[col]], ignore_index=True)
        if combined.notna().any() and X_train[col].nunique(dropna=True) > 1:
            keep_cols.append(col)

    # Ensure train and test categorical columns share the same category set.
    for col in categorical_cols:
        if col not in keep_cols:
            continue
        categories = pd.unique(
            pd.concat(
                [X_train[col].astype(object), X_test[col].astype(object)],
                ignore_index=True,
            ).dropna()
        )
        X_train[col] = X_train[col].astype("category").cat.set_categories(categories)
        X_test[col] = X_test[col].astype("category").cat.set_categories(categories)

    X_train = X_train[keep_cols]
    X_test = X_test[keep_cols]
    categorical_cols = [col for col in categorical_cols if col in keep_cols]
    return X_train, X_test, keep_cols, categorical_cols


X_train, X_test, feature_cols, categorical_cols = preprocess_features(
    train, test, feature_cols
)
y = train[TARGET].to_numpy(dtype=float)

# Use log1p for non-negative cost targets to stabilize skewed regression targets.
use_log_transform = bool(np.all(y >= 0))
y_model = np.log1p(y) if use_log_transform else y


def inverse_transform(predictions):
    if use_log_transform:
        predictions = np.expm1(predictions)
        predictions = np.clip(predictions, 0.0, None)
    return np.nan_to_num(predictions, nan=float(np.median(y)))


params = {
    "objective": "regression_l2",
    "metric": "rmse",
    "learning_rate": 0.03,
    "num_leaves": 15,
    "max_depth": -1,
    "min_child_samples": 10,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 1,
    "lambda_l1": 0.1,
    "lambda_l2": 1.0,
    "seed": SEED,
    "verbose": -1,
    "num_threads": 4,
}

callbacks = [lgb.early_stopping(50), lgb.log_evaluation(-1)]

# If there are no usable features, fall back to a constant prediction.
if len(feature_cols) == 0 or len(X_train) == 0:
    constant = float(np.median(y)) if len(y) > 0 else 0.0
    print(
        f"Validation RMSE: {root_mean_squared_error(y, np.full(len(y), constant)):.6f}"
    )
    test_pred = np.full(len(test), constant)
    submission = pd.DataFrame(
        {
            "id": np.arange(len(test), dtype=int),
            TARGET: test_pred,
        }
    )
    submission.to_csv("./working/submission.csv", index=False)
    raise SystemExit(0)

# 5-fold cross-validation gives a hold-out estimate of generalization RMSE.
n_splits = min(5, len(X_train))
if n_splits < 2:
    n_splits = 2

kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
oof_pred = np.zeros(len(X_train), dtype=float)
best_iterations = []

for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train), start=1):
    dtrain = lgb.Dataset(
        X_train.iloc[tr_idx],
        label=y_model[tr_idx],
        categorical_feature=categorical_cols,
    )
    dval = lgb.Dataset(
        X_train.iloc[val_idx],
        label=y_model[val_idx],
        reference=dtrain,
        categorical_feature=categorical_cols,
    )

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=5000,
        valid_sets=[dval],
        valid_names=["val"],
        callbacks=callbacks,
    )

    best_iter = getattr(model, "best_iteration", 0)
    if best_iter is None or best_iter <= 0:
        best_iter = model.current_iteration()
    best_iterations.append(int(best_iter))

    fold_pred = model.predict(X_train.iloc[val_idx], num_iteration=best_iter)
    oof_pred[val_idx] = inverse_transform(fold_pred)

    fold_rmse = root_mean_squared_error(y[val_idx], oof_pred[val_idx])
    print(f"Fold {fold} RMSE: {fold_rmse:.6f}")

cv_rmse = root_mean_squared_error(y, oof_pred)
print(f"Validation RMSE: {cv_rmse:.6f}")

# Train final model on all training data using the median early-stopping iteration.
final_num_boost_round = int(np.median(best_iterations)) if best_iterations else 500
if final_num_boost_round <= 0:
    final_num_boost_round = 500

dtrain_full = lgb.Dataset(
    X_train,
    label=y_model,
    categorical_feature=categorical_cols,
)

final_model = lgb.train(
    params,
    dtrain_full,
    num_boost_round=final_num_boost_round,
    callbacks=[lgb.log_evaluation(-1)],
)

test_pred = final_model.predict(X_test, num_iteration=final_num_boost_round)
test_pred = inverse_transform(test_pred)

submission = pd.DataFrame(
    {
        "id": np.arange(len(test), dtype=int),
        TARGET: test_pred,
    }
)
submission.to_csv("./working/submission.csv", index=False)
