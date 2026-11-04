import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error

TARGET = "Adjusted MLR"
RANDOM_STATE = 42
WORK_DIR = "./working"
os.makedirs(WORK_DIR, exist_ok=True)


def clean_dataframe(df: pd.DataFrame, is_train: bool = False) -> pd.DataFrame:
    df = df.copy()

    if is_train and TARGET in df.columns:
        df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
        df = df.dropna(subset=[TARGET])

    if not is_train and TARGET in df.columns:
        df = df.drop(columns=[TARGET])

    n = len(df)

    # Drop likely row identifiers only when they appear high-cardinality.
    id_like = []
    for col in df.columns:
        if col == TARGET:
            continue
        lower = str(col).lower()
        is_id_name = lower in {
            "id",
            "index",
            "row_id",
            "sample_id",
            "member_id",
            "patient_id",
            "claim_id",
        } or lower.endswith("_id")
        if is_id_name and df[col].nunique(dropna=True) > 0.9 * n:
            id_like.append(col)
    df = df.drop(columns=id_like, errors="ignore")

    # Drop constant or all-null feature columns, but never drop the target.
    drop_cols = [
        col for col in df.columns if col != TARGET and df[col].nunique(dropna=True) <= 1
    ]
    df = df.drop(columns=drop_cols, errors="ignore")

    # Convert feature columns to numeric or categorical types.
    for col in list(df.columns):
        if col == TARGET:
            continue

        if pd.api.types.is_datetime64_any_dtype(df[col]):
            int_vals = df[col].astype("int64") // 10**9
            int_vals = int_vals.where(int_vals != -9223372036854775808, np.nan)
            df[col] = int_vals.astype("float64")

        elif df[col].dtype == object:
            converted = pd.to_numeric(df[col], errors="coerce")
            non_null = df[col].notna().sum()
            if (
                non_null > 0
                and converted.notna().sum() >= 0.9 * non_null
                and converted.nunique(dropna=True) <= 1000
            ):
                df[col] = converted
            else:
                df[col] = df[col].astype("category")

        elif not pd.api.types.is_numeric_dtype(df[col]) and not isinstance(
            df[col].dtype, pd.CategoricalDtype
        ):
            converted = pd.to_numeric(df[col], errors="coerce")
            non_null = df[col].notna().sum()
            if non_null > 0 and converted.notna().sum() >= 0.9 * non_null:
                df[col] = converted
            else:
                df[col] = df[col].astype("category")

    return df


def align_features(
    train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: list
) -> tuple[pd.DataFrame, pd.DataFrame]:
    X = train_df[feature_cols].copy()
    X_test = test_df[feature_cols].copy()

    for col in feature_cols:
        train_is_cat = isinstance(X[col].dtype, pd.CategoricalDtype)
        test_is_cat = isinstance(X_test[col].dtype, pd.CategoricalDtype)

        if train_is_cat or test_is_cat:
            if not train_is_cat:
                X[col] = X[col].astype("category")

            train_categories = X[col].cat.categories
            X_test[col] = pd.Categorical(X_test[col], categories=train_categories)
        else:
            X[col] = pd.to_numeric(X[col], errors="coerce")
            X_test[col] = pd.to_numeric(X_test[col], errors="coerce")

    return X, X_test


def main() -> None:
    train = pd.read_parquet("./input/train.parquet")
    test = pd.read_parquet("./input/test.parquet")

    train = clean_dataframe(train, is_train=True)
    test = clean_dataframe(test, is_train=False)

    feature_cols = [
        col for col in train.columns if col != TARGET and col in test.columns
    ]

    if len(feature_cols) == 0:
        mean_target = (
            float(train[TARGET].mean())
            if TARGET in train.columns and len(train) > 0
            else 0.0
        )
        submission = pd.DataFrame({"id": np.arange(len(test)), TARGET: mean_target})
        submission.to_csv(os.path.join(WORK_DIR, "submission.csv"), index=False)
        print("Validation RMSE: nan")
        return

    X, X_test = align_features(train, test, feature_cols)
    y = train[TARGET].astype("float64")

    params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 5,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "lambda_l1": 0.1,
        "lambda_l2": 0.1,
        "seed": RANDOM_STATE,
        "verbosity": -1,
    }

    num_boost_round = 5000
    early_stopping_rounds = 50

    if len(X) >= 5:
        n_splits = 5
    elif len(X) >= 2:
        n_splits = 2
    else:
        n_splits = 1

    if n_splits >= 2:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
        fold_rmses = []
        best_iterations = []

        for fold, (train_idx, valid_idx) in enumerate(kf.split(X), start=1):
            X_tr, X_va = X.iloc[train_idx], X.iloc[valid_idx]
            y_tr, y_va = y.iloc[train_idx], y.iloc[valid_idx]

            train_data = lgb.Dataset(X_tr, label=y_tr, categorical_feature="auto")
            valid_data = lgb.Dataset(
                X_va, label=y_va, reference=train_data, categorical_feature="auto"
            )

            model = lgb.train(
                params,
                train_data,
                num_boost_round=num_boost_round,
                valid_sets=[valid_data],
                valid_names=["valid"],
                callbacks=[
                    lgb.early_stopping(early_stopping_rounds, first_metric_only=True),
                    lgb.log_evaluation(-1),
                ],
            )

            pred_va = model.predict(X_va)
            fold_rmse = root_mean_squared_error(y_va, pred_va)
            fold_rmses.append(fold_rmse)

            best_iter = model.best_iteration
            if best_iter is None or best_iter <= 0:
                best_iter = num_boost_round
            best_iterations.append(int(best_iter))

            print(f"Fold {fold} RMSE: {fold_rmse:.6f}")

        cv_rmse = float(np.mean(fold_rmses))
        final_boost_rounds = int(max(1, np.mean(best_iterations)))
    else:
        cv_rmse = float("nan")
        final_boost_rounds = 100

    # Refit on all training data for the final submission.
    full_train_data = lgb.Dataset(X, label=y, categorical_feature="auto")
    final_model = lgb.train(
        params,
        full_train_data,
        num_boost_round=final_boost_rounds,
        callbacks=[lgb.log_evaluation(-1)],
    )

    test_pred = final_model.predict(X_test)
    test_pred = np.asarray(test_pred, dtype="float64")

    submission = pd.DataFrame(
        {
            "id": np.arange(len(test), dtype=int),
            TARGET: test_pred,
        }
    )
    submission.to_csv(os.path.join(WORK_DIR, "submission.csv"), index=False)

    print(f"Validation RMSE: {cv_rmse:.6f}")


if __name__ == "__main__":
    main()
