import os
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error

warnings.filterwarnings("ignore")

INPUT_DIR = "./input"
WORK_DIR = "./working"
os.makedirs(WORK_DIR, exist_ok=True)

TARGET = "RATIO_ENROLL_EMP"
RANDOM_STATE = 42
N_FOLDS = 5
NUM_BOOST_ROUND = 2000
EARLY_STOPPING_ROUNDS = 50

train = pd.read_parquet(os.path.join(INPUT_DIR, "train.parquet"))
test = pd.read_parquet(os.path.join(INPUT_DIR, "test.parquet"))

# Identify and remove non-feature columns
id_cols = {
    col
    for col in train.columns
    if col.lower()
    in {"id", "index", "unzip", "unitid", "unitid1", "unitid2", "unitid3"}
}
feature_cols = [col for col in train.columns if col != TARGET and col not in id_cols]

# Preserve original test row order for submission ids
test_ids = np.arange(len(test))

# Align test features to training feature names
for col in feature_cols:
    if col not in test.columns:
        test[col] = np.nan


def prepare_features(
    df: pd.DataFrame,
    feature_cols: list,
    train_categories: dict | None = None,
    is_train: bool = False,
) -> tuple[pd.DataFrame, list]:
    """
    Prepare features for LightGBM.

    For training, categorical columns are converted to pandas categorical dtype and
    their categories are stored. For test, the same categories are enforced so unseen
    values become missing instead of creating inconsistent category codes.
    """
    X = df[feature_cols].copy()
    categorical_columns = []

    for col in X.columns:
        if is_train:
            if (
                X[col].dtype == object
                or str(X[col].dtype) == "category"
                or pd.api.types.is_string_dtype(X[col])
            ):
                X[col] = X[col].astype("category")
                if train_categories is not None:
                    train_categories[col] = X[col].cat.categories
                categorical_columns.append(col)
            elif pd.api.types.is_numeric_dtype(X[col]):
                X[col] = pd.to_numeric(X[col], errors="coerce")
                X[col] = X[col].replace([np.inf, -np.inf], np.nan)
            else:
                # Fallback for unusual dtypes: treat as categorical strings.
                X[col] = X[col].astype(str).astype("category")
                if train_categories is not None:
                    train_categories[col] = X[col].cat.categories
                categorical_columns.append(col)
        else:
            if train_categories is not None and col in train_categories:
                X[col] = pd.Categorical(X[col], categories=train_categories[col])
                categorical_columns.append(col)
            elif pd.api.types.is_numeric_dtype(X[col]):
                X[col] = pd.to_numeric(X[col], errors="coerce")
                X[col] = X[col].replace([np.inf, -np.inf], np.nan)
            else:
                # Should not normally happen; coerce to numeric if possible.
                X[col] = pd.to_numeric(X[col], errors="coerce")
                X[col] = X[col].replace([np.inf, -np.inf], np.nan)

    return X, categorical_columns


train_categories = {}
X_train, _ = prepare_features(
    train,
    feature_cols,
    train_categories=train_categories,
    is_train=True,
)
X_test, _ = prepare_features(
    test,
    feature_cols,
    train_categories=train_categories,
    is_train=False,
)

y_train = pd.to_numeric(train[TARGET], errors="coerce")

# Remove rows with missing target for modelling
valid_mask = y_train.notna()
X_train = X_train[valid_mask].reset_index(drop=True)
y_train = y_train[valid_mask].reset_index(drop=True)

categorical_features = [
    col for col in X_train.columns if str(X_train[col].dtype) == "category"
]

params = {
    "objective": "regression_l2",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "min_child_samples": 20,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 1,
    "lambda_l1": 0.1,
    "lambda_l2": 1.0,
    "seed": RANDOM_STATE,
    "verbose": -1,
    "num_threads": -1,
}

models = []
oof_predictions = np.full(len(X_train), np.nan, dtype=float)
overall_rmse = np.nan

if len(X_train) >= 2:
    n_splits = min(N_FOLDS, len(X_train))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train), start=1):
        X_tr = X_train.iloc[train_idx]
        y_tr = y_train.iloc[train_idx]
        X_val = X_train.iloc[val_idx]
        y_val = y_train.iloc[val_idx]

        dtrain = lgb.Dataset(
            X_tr,
            label=y_tr,
            categorical_feature=categorical_features,
            free_raw_data=False,
        )
        dval = lgb.Dataset(
            X_val,
            label=y_val,
            categorical_feature=categorical_features,
            free_raw_data=False,
        )

        model = lgb.train(
            params,
            dtrain,
            num_boost_round=NUM_BOOST_ROUND,
            valid_sets=[dval],
            valid_names=["validation"],
            callbacks=[
                lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                lgb.log_evaluation(-1),
            ],
        )
        models.append(model)

        oof_predictions[val_idx] = model.predict(
            X_val, num_iteration=model.best_iteration
        )

        fold_rmse = root_mean_squared_error(y_val, oof_predictions[val_idx])
        print(f"Fold {fold} validation RMSE: {fold_rmse:.6f}")

    overall_rmse = root_mean_squared_error(y_train, oof_predictions)
    print(f"Hold-out validation RMSE: {overall_rmse:.6f}")
else:
    print("Not enough valid training rows to compute cross-validation RMSE.")

# Train final model on all available training data
if len(X_train) > 0:
    best_iters = [
        m.best_iteration
        for m in models
        if hasattr(m, "best_iteration") and m.best_iteration > 0
    ]
    final_iters = int(np.median(best_iters)) if best_iters else NUM_BOOST_ROUND
    final_iters = max(1, min(final_iters, NUM_BOOST_ROUND))

    dtrain_full = lgb.Dataset(
        X_train,
        label=y_train,
        categorical_feature=categorical_features,
        free_raw_data=False,
    )
    final_model = lgb.train(
        params,
        dtrain_full,
        num_boost_round=final_iters,
        callbacks=[lgb.log_evaluation(-1)],
    )

    test_predictions = final_model.predict(X_test)
    test_predictions = np.clip(test_predictions, y_train.min(), y_train.max())
else:
    test_predictions = np.zeros(len(test), dtype=float)

submission = pd.DataFrame({"id": test_ids, TARGET: test_predictions})
submission_path = os.path.join(WORK_DIR, "submission.csv")
submission.to_csv(submission_path, index=False)

print(f"Saved submission to {submission_path}")
print(submission.head())
