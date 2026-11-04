import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error

TARGET = "ACA FUL"
ID_COLS = {"id", "index", "row_id", "index_id"}


def clean_data(train: pd.DataFrame, test: pd.DataFrame):
    train = train.copy()
    test = test.copy()

    # Remove ID-like columns that should not be used as features.
    train.drop(
        columns=[c for c in train.columns if str(c).lower() in ID_COLS],
        inplace=True,
        errors="ignore",
    )
    test.drop(
        columns=[c for c in test.columns if str(c).lower() in ID_COLS],
        inplace=True,
        errors="ignore",
    )

    # The test set should not contain the target, but drop it defensively.
    if TARGET in test.columns:
        test = test.drop(columns=[TARGET])

    # Ensure target is numeric and remove rows with missing target.
    train[TARGET] = pd.to_numeric(train[TARGET], errors="coerce")
    train = train.dropna(subset=[TARGET])

    # Use only features present in both train and test.
    features = [c for c in train.columns if c != TARGET and c in test.columns]
    train = train[features + [TARGET]]
    test = test[features]

    # Basic cleaning: numeric coercion, infinities to NaN, categoricals aligned to train categories.
    for col in features:
        if train[col].dtype == bool:
            train[col] = train[col].astype(int)
            test[col] = test[col].astype(int)
        elif pd.api.types.is_numeric_dtype(train[col]):
            train[col] = pd.to_numeric(train[col], errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            )
            test[col] = pd.to_numeric(test[col], errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            )
        else:
            train[col] = train[col].astype("category")
            test[col] = test[col].astype("category")
            train_categories = train[col].cat.categories
            test[col] = pd.Categorical(test[col], categories=train_categories)

    # Drop constant or all-missing features.
    keep = [col for col in features if train[col].nunique(dropna=True) > 1]

    train = train[keep + [TARGET]]
    test = test[keep]

    return train, test, keep


def align_categories_for_fold(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    categorical_columns: list,
):
    """
    Align validation categories to the training fold only to avoid leakage
    from categories that appear only in the validation fold.
    """
    X_train = X_train.copy()
    X_val = X_val.copy()

    for col in categorical_columns:
        cats = pd.unique(X_train[col].dropna())
        if len(cats) == 0:
            X_train[col] = pd.Categorical(X_train[col], categories=[])
            X_val[col] = pd.Categorical(X_val[col], categories=[])
        else:
            X_train[col] = pd.Categorical(X_train[col], categories=cats)
            X_val[col] = pd.Categorical(X_val[col], categories=cats)

    return X_train, X_val


def main():
    os.makedirs("./working", exist_ok=True)

    train = pd.read_parquet("./input/train.parquet")
    test = pd.read_parquet("./input/test.parquet")

    train, test, features = clean_data(train, test)

    X = train[features]
    y = train[TARGET].to_numpy(dtype=float)
    X_test = test[features]

    categorical_columns = [col for col in features if X[col].dtype.name == "category"]

    # Align test categories to the full training categories.
    if categorical_columns:
        X_test = X_test.copy()
        for col in categorical_columns:
            X_test[col] = pd.Categorical(X_test[col], categories=X[col].cat.categories)

    # Fallback if no usable features remain.
    if len(features) == 0:
        mean_prediction = float(np.mean(y)) if len(y) > 0 else 0.0
        rmse = (
            float(np.sqrt(np.mean((y - mean_prediction) ** 2))) if len(y) > 0 else 0.0
        )
        print(f"CV RMSE: {rmse:.6f}")

        submission = pd.DataFrame(
            {
                "id": np.arange(len(test)),
                TARGET: np.full(len(test), mean_prediction, dtype=float),
            }
        )
        submission.to_csv("./working/submission.csv", index=False)
        return

    params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 5,
        "subsample": 0.9,
        "subsample_freq": 1,
        "colsample_bytree": 0.9,
        "reg_lambda": 1.0,
        "seed": 42,
        "verbosity": -1,
    }

    n_samples = len(X)
    n_splits = 5 if n_samples >= 5 else (2 if n_samples >= 2 else 1)

    oof_predictions = np.zeros(n_samples, dtype=float)
    best_iterations = []

    if n_splits >= 2:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

        for train_idx, val_idx in kf.split(X):
            X_train_fold, X_val_fold = align_categories_for_fold(
                X.iloc[train_idx],
                X.iloc[val_idx],
                categorical_columns,
            )

            dtrain = lgb.Dataset(
                X_train_fold,
                label=y[train_idx],
                categorical_feature=categorical_columns,
                free_raw_data=False,
            )
            dval = lgb.Dataset(
                X_val_fold,
                label=y[val_idx],
                categorical_feature=categorical_columns,
                reference=dtrain,
                free_raw_data=False,
            )

            model = lgb.train(
                params,
                dtrain,
                num_boost_round=2000,
                valid_sets=[dval],
                callbacks=[
                    lgb.early_stopping(50, verbose=False),
                    lgb.log_evaluation(-1),
                ],
            )

            best_iteration = getattr(model, "best_iteration", None)
            if best_iteration is None or best_iteration <= 0:
                best_iteration = 1

            best_iterations.append(best_iteration)
            oof_predictions[val_idx] = model.predict(
                X_val_fold, num_iteration=best_iteration
            )

        rmse = root_mean_squared_error(y, oof_predictions)
    else:
        # Extremely small-data fallback: no meaningful CV split is possible.
        rmse = 0.0
        best_iterations = [100]

    print(f"CV RMSE: {rmse:.6f}")

    # Train the final model on the full training set using the median CV best iteration.
    final_boost_rounds = (
        int(max(1, np.median(best_iterations))) if best_iterations else 100
    )

    dtrain_full = lgb.Dataset(
        X,
        label=y,
        categorical_feature=categorical_columns,
        free_raw_data=False,
    )

    final_model = lgb.train(
        params,
        dtrain_full,
        num_boost_round=final_boost_rounds,
        callbacks=[lgb.log_evaluation(-1)],
    )

    test_predictions = final_model.predict(X_test, num_iteration=final_boost_rounds)

    submission = pd.DataFrame(
        {
            "id": np.arange(len(test)),
            TARGET: test_predictions,
        }
    )
    submission.to_csv("./working/submission.csv", index=False)


if __name__ == "__main__":
    main()
