import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

os.makedirs("./working", exist_ok=True)

TARGET = "APR"
SEED = 42

train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

if TARGET not in train.columns:
    raise ValueError(f"Target column {TARGET!r} not found in train data.")

y = pd.to_numeric(train[TARGET], errors="coerce")
X = train.drop(columns=[TARGET], errors="ignore").copy()

valid_target = y.notna()
X = X.loc[valid_target].copy()
y = y.loc[valid_target].reset_index(drop=True)
X = X.reset_index(drop=True)

test = test.drop(columns=[TARGET], errors="ignore").copy()

drop_cols = []
for col in X.columns:
    col_lower = str(col).lower()
    if col_lower in {"id", "index", "unnamed: 0"}:
        drop_cols.append(col)
        continue

    if X[col].isna().all():
        drop_cols.append(col)
        continue

    if X[col].nunique(dropna=True) <= 1:
        drop_cols.append(col)
        continue

    if X[col].isna().mean() > 0.95:
        drop_cols.append(col)
        continue

    if X[col].dtype == object and X[col].nunique(dropna=True) >= max(
        2, int(0.95 * len(X))
    ):
        drop_cols.append(col)

X = X.drop(columns=drop_cols, errors="ignore")

test = test.reindex(columns=X.columns)

categorical_cols = [
    col
    for col in X.columns
    if X[col].dtype == object or pd.api.types.is_categorical_dtype(X[col])
]

for col in categorical_cols:
    X[col] = X[col].astype("category")
    if col in test.columns:
        test[col] = pd.Categorical(test[col], categories=X[col].cat.categories)
    else:
        test[col] = pd.Categorical(
            pd.Series(index=test.index, dtype=object), categories=X[col].cat.categories
        )

for col in X.columns:
    if col in categorical_cols:
        continue
    X[col] = pd.to_numeric(X[col], errors="coerce")
    if col in test.columns:
        test[col] = pd.to_numeric(test[col], errors="coerce")
    else:
        test[col] = np.nan

params = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "min_child_samples": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l1": 0.1,
    "lambda_l2": 0.1,
    "verbosity": -1,
    "seed": SEED,
    "nthread": 4,
}

if len(X) < 10 or X.shape[1] == 0:
    val_rmse = float(
        np.sqrt(mean_squared_error(y, np.full(len(y), y.mean()), squared=False))
    )
    print(f"Validation RMSE: {val_rmse:.6f}")
    test_pred = np.full(len(test), y.mean())
else:
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    oof_pred = np.zeros(len(X))
    best_iterations = []

    for fold, (train_idx, valid_idx) in enumerate(kf.split(X), start=1):
        X_train_fold = X.iloc[train_idx]
        y_train_fold = y.iloc[train_idx]
        X_valid_fold = X.iloc[valid_idx]
        y_valid_fold = y.iloc[valid_idx]

        train_dataset = lgb.Dataset(
            X_train_fold,
            label=y_train_fold,
            categorical_feature=categorical_cols,
            free_raw_data=False,
        )
        valid_dataset = lgb.Dataset(
            X_valid_fold,
            label=y_valid_fold,
            categorical_feature=categorical_cols,
            reference=train_dataset,
            free_raw_data=False,
        )

        model = lgb.train(
            params,
            train_dataset,
            valid_sets=[valid_dataset],
            valid_names=["valid"],
            num_boost_round=10000,
            callbacks=[
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(-1),
            ],
        )

        best_iterations.append(model.best_iteration)
        oof_pred[valid_idx] = model.predict(
            X_valid_fold, num_iteration=model.best_iteration
        )

        print(f"Fold {fold} best iteration: {model.best_iteration}")

    val_rmse = float(np.sqrt(mean_squared_error(y, oof_pred, squared=False)))
    print(f"Validation RMSE: {val_rmse:.6f}")

    final_dataset = lgb.Dataset(
        X,
        label=y,
        categorical_feature=categorical_cols,
        free_raw_data=False,
    )
    final_num_boost_round = max(1, int(np.mean(best_iterations)))
    final_model = lgb.train(
        params,
        final_dataset,
        num_boost_round=final_num_boost_round,
        callbacks=[lgb.log_evaluation(-1)],
    )
    test_pred = final_model.predict(test, num_iteration=final_num_boost_round)

submission = pd.DataFrame(
    {
        "id": np.arange(len(test), dtype=int),
        "APR": np.asarray(test_pred, dtype=float),
    }
)
submission.to_csv("./working/submission.csv", index=False)
