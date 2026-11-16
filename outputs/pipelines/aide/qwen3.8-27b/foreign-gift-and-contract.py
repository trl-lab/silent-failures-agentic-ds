import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

os.makedirs("./working", exist_ok=True)

train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

target = "Amount"

# Extract target and drop only columns that actually exist.
y = train[target].copy()
X = train.drop(columns=[target], errors="ignore").copy()

if "id" in X.columns:
    X = X.drop(columns=["id"])

# Remove rows with missing target and keep train/test aligned by row order.
mask = y.notna()
y = y[mask].reset_index(drop=True)
X = X[mask].reset_index(drop=True)

# Align test features to training features; this also drops test id/target if present.
X_test = test.reindex(columns=X.columns).copy()

categorical_cols = []
numerical_cols = []

# Prepare features with explicit train-only category mapping.
for col in X.columns:
    train_series = X[col]
    test_series = X_test[col]

    is_text_like = (
        pd.api.types.is_object_dtype(train_series)
        or pd.api.types.is_string_dtype(train_series)
        or pd.api.types.is_categorical_dtype(train_series)
    )

    if is_text_like:
        # If the column is mostly numeric, treat it as numeric; otherwise categorical.
        converted = pd.to_numeric(train_series, errors="coerce")
        non_null_original = max(1, int(train_series.notna().sum()))
        non_null_numeric = int(converted.notna().sum())

        if non_null_numeric > 0 and (non_null_numeric / non_null_original) >= 0.9:
            X[col] = converted
            X_test[col] = pd.to_numeric(test_series, errors="coerce")
            numerical_cols.append(col)
        else:
            X[col] = train_series.astype("category")
            train_categories = X[col].cat.categories
            X_test[col] = pd.Categorical(test_series, categories=train_categories)
            categorical_cols.append(col)
    else:
        # Numeric-like or other dtype: coerce to numeric when possible.
        X[col] = pd.to_numeric(train_series, errors="coerce")
        X_test[col] = pd.to_numeric(test_series, errors="coerce")

        if X[col].dtype == object:
            X[col] = train_series.astype(str).astype("category")
            train_categories = X[col].cat.categories
            X_test[col] = pd.Categorical(
                test_series.astype(str), categories=train_categories
            )
            categorical_cols.append(col)
        else:
            numerical_cols.append(col)

# Fallback for any columns that were not classified.
for col in X.columns:
    if col not in categorical_cols and col not in numerical_cols:
        X[col] = X[col].astype(str).astype("category")
        X_test[col] = pd.Categorical(
            X_test[col].astype(str), categories=X[col].cat.categories
        )
        categorical_cols.append(col)

categorical_feature = categorical_cols if categorical_cols else "auto"

params = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "feature_pre_filter": False,
    "verbose": -1,
}

# 5-fold cross-validation to estimate generalization performance.
n_splits = 5 if len(X) >= 5 else max(1, len(X))
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

fold_rmses = []
best_iterations = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X), start=1):
    X_train_fold = X.iloc[train_idx]
    y_train_fold = y.iloc[train_idx]
    X_val_fold = X.iloc[val_idx]
    y_val_fold = y.iloc[val_idx]

    train_data = lgb.Dataset(
        X_train_fold,
        label=y_train_fold,
        categorical_feature=categorical_feature,
    )
    val_data = lgb.Dataset(
        X_val_fold,
        label=y_val_fold,
        categorical_feature=categorical_feature,
        reference=train_data,
    )

    callbacks = [
        lgb.early_stopping(stopping_rounds=50, verbose=False),
        lgb.log_evaluation(-1),
    ]

    model = lgb.train(
        params,
        train_data,
        num_boost_round=2000,
        valid_sets=[val_data],
        valid_names=["val"],
        callbacks=callbacks,
    )

    best_iter = (
        model.best_iteration if model.best_iteration > 0 else model.current_iteration
    )
    best_iterations.append(max(1, int(best_iter)))

    val_pred = model.predict(X_val_fold, num_iteration=model.best_iteration)
    fold_rmse = np.sqrt(mean_squared_error(y_val_fold, val_pred))
    fold_rmses.append(fold_rmse)
    print(f"Fold {fold} RMSE: {fold_rmse:.4f}")

cv_rmse = float(np.mean(fold_rmses))
print(f"Validation RMSE: {cv_rmse:.4f}")

# Retrain on the full training set using the median best iteration from CV.
final_num_boost_round = max(1, int(np.median(best_iterations)))
full_train_data = lgb.Dataset(
    X,
    label=y,
    categorical_feature=categorical_feature,
)
final_model = lgb.train(
    params,
    full_train_data,
    num_boost_round=final_num_boost_round,
    callbacks=[lgb.log_evaluation(-1)],
)

if len(X_test) == 0:
    y_pred_test = np.array([])
else:
    y_pred_test = final_model.predict(X_test)
    y_pred_test = np.nan_to_num(y_pred_test, nan=0.0)

submission = pd.DataFrame(
    {
        "id": np.arange(len(y_pred_test), dtype=int),
        "Amount": y_pred_test,
    }
)
submission.to_csv("./working/submission.csv", index=False)
