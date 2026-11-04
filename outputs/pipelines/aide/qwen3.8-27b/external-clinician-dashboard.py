import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

TARGET = "Score"
WORK_DIR = "./working"
INPUT_DIR = "./input"
RANDOM_STATE = 42
NUM_BOOST_ROUND = 3000
EARLY_STOPPING_ROUNDS = 50

ID_LIKE_COLUMNS = {
    "id",
    "index",
    "row_id",
    "rowid",
    "uid",
    "subject_id",
    "patient_id",
    "sample_id",
    "observation_id",
    "record_id",
    "case_id",
    "claim_id",
    "encounter_id",
    "visit_id",
    "patientid",
    "subjectid",
    "userid",
    "user_id",
}

MISSING_TOKEN = "__MISSING__"
UNKNOWN_TOKEN = "__UNKNOWN__"


def drop_id_columns(df: pd.DataFrame) -> pd.DataFrame:
    id_cols = [c for c in df.columns if str(c).lower() in ID_LIKE_COLUMNS]
    return df.drop(columns=id_cols, errors="ignore")


def align_columns(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    common_cols = [c for c in train.columns if c in test.columns]
    return train[common_cols].copy(), test[common_cols].copy()


def datetime_to_numeric(series: pd.Series) -> pd.Series:
    """Convert datetime-like values to integer nanoseconds, preserving missing values as NaN."""
    if not pd.api.types.is_datetime64_any_dtype(series):
        return series

    dt = pd.to_datetime(series, errors="coerce")
    # astype("int64") maps NaT to iNaT; explicitly restore missingness as NaN.
    numeric = dt.astype("int64").astype("float64")
    numeric[dt.isna()] = np.nan
    return numeric


def basic_numeric_conversion(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = train.copy()
    test = test.copy()

    for col in train.columns:
        if col not in test.columns:
            continue

        train_series = train[col]
        test_series = test[col]

        if pd.api.types.is_datetime64_any_dtype(
            train_series
        ) or pd.api.types.is_datetime64_any_dtype(test_series):
            train[col] = datetime_to_numeric(train_series)
            test[col] = datetime_to_numeric(test_series)

        elif pd.api.types.is_bool_dtype(train_series) or pd.api.types.is_bool_dtype(
            test_series
        ):
            train[col] = pd.to_numeric(train_series, errors="coerce").astype("float64")
            test[col] = pd.to_numeric(test_series, errors="coerce").astype("float64")

    return train, test


def get_categorical_columns(train: pd.DataFrame) -> list[str]:
    cat_cols = []
    for col in train.columns:
        dtype = train[col].dtype
        if (
            pd.api.types.is_object_dtype(dtype)
            or pd.api.types.is_categorical_dtype(dtype)
            or pd.api.types.is_string_dtype(dtype)
        ):
            cat_cols.append(col)
    return cat_cols


def build_categories(train_df: pd.DataFrame, cat_cols: list[str]) -> dict[str, list]:
    categories = {}
    for col in cat_cols:
        series = train_df[col].astype("object")
        series = series.where(series.notna(), MISSING_TOKEN).astype(str)
        cats = sorted(series.unique().tolist())
        if MISSING_TOKEN not in cats:
            cats.append(MISSING_TOKEN)
        if UNKNOWN_TOKEN not in cats:
            cats.append(UNKNOWN_TOKEN)
        categories[col] = sorted(cats)
    return categories


def encode_with_categories(
    df: pd.DataFrame, cat_cols: list[str], categories: dict[str, list]
) -> pd.DataFrame:
    out = df.copy()
    for col in cat_cols:
        cats = categories[col]
        dtype = pd.CategoricalDtype(categories=cats)
        series = out[col].astype("object")
        series = series.where(series.notna(), MISSING_TOKEN).astype(str)
        series = series.where(series.isin(cats), UNKNOWN_TOKEN)
        out[col] = series.astype(dtype)
    return out


def preprocess_pair(
    train_raw: pd.DataFrame,
    other_raw: pd.DataFrame,
    cat_cols: list[str],
    categories: dict[str, list] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, list]]:
    if categories is None:
        categories = build_categories(train_raw, cat_cols)

    train_proc = train_raw.copy()
    other_proc = other_raw.copy()

    num_cols = [c for c in train_proc.columns if c not in cat_cols]
    for col in num_cols:
        train_proc[col] = pd.to_numeric(train_proc[col], errors="coerce")
        other_proc[col] = pd.to_numeric(other_proc[col], errors="coerce")

    train_proc = encode_with_categories(train_proc, cat_cols, categories)
    other_proc = encode_with_categories(other_proc, cat_cols, categories)

    return train_proc, other_proc, categories


def main() -> None:
    os.makedirs(WORK_DIR, exist_ok=True)

    train = pd.read_parquet(os.path.join(INPUT_DIR, "train.parquet"))
    test = pd.read_parquet(os.path.join(INPUT_DIR, "test.parquet"))

    if TARGET not in train.columns:
        raise ValueError(f"Target column {TARGET!r} not found in train.parquet.")

    y = pd.to_numeric(train[TARGET], errors="coerce").astype(float)
    train = train.drop(columns=[TARGET])
    train = train.loc[y.notna()].copy()
    y = y.loc[y.notna()].copy()

    if TARGET in test.columns:
        test = test.drop(columns=[TARGET])

    train, test = drop_id_columns(train), drop_id_columns(test)
    train, test = align_columns(train, test)
    train, test = basic_numeric_conversion(train, test)

    train = train.reset_index(drop=True)
    y = y.reset_index(drop=True)
    test = test.reset_index(drop=True)

    cat_cols = get_categorical_columns(train)
    params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "subsample": 0.9,
        "subsample_freq": 1,
        "colsample_bytree": 0.9,
        "min_child_samples": 20,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "random_seed": RANDOM_STATE,
        "verbosity": -1,
    }

    if train.shape[1] == 0 or len(train) == 0:
        fallback = float(y.mean()) if len(y) > 0 else 0.0
        oof = np.full(len(y), fallback)
        cv_rmse = float(np.sqrt(mean_squared_error(y, oof))) if len(y) > 0 else 0.0
        print(f"5-fold CV RMSE: {cv_rmse:.6f}")

        submission = pd.DataFrame(
            {"id": np.arange(len(test)), "Score": np.full(len(test), fallback)}
        )
        submission.to_csv(os.path.join(WORK_DIR, "submission.csv"), index=False)
        print(f"Saved submission to {os.path.join(WORK_DIR, 'submission.csv')}")
        return

    n_splits = min(5, len(train)) if len(train) >= 2 else 1

    if n_splits < 2:
        fallback = float(y.mean()) if len(y) > 0 else 0.0
        oof = np.full(len(y), fallback)
        cv_rmse = float(np.sqrt(mean_squared_error(y, oof))) if len(y) > 0 else 0.0
        print(f"5-fold CV RMSE: {cv_rmse:.6f}")

        submission = pd.DataFrame(
            {"id": np.arange(len(test)), "Score": np.full(len(test), fallback)}
        )
        submission.to_csv(os.path.join(WORK_DIR, "submission.csv"), index=False)
        print(f"Saved submission to {os.path.join(WORK_DIR, 'submission.csv')}")
        return

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    oof_predictions = np.zeros(len(train), dtype=float)
    best_iterations = []

    for fold, (train_idx, valid_idx) in enumerate(kf.split(train), start=1):
        X_train_raw = train.iloc[train_idx]
        X_valid_raw = train.iloc[valid_idx]
        y_train = y.iloc[train_idx]
        y_valid = y.iloc[valid_idx]

        X_train, X_valid, _ = preprocess_pair(X_train_raw, X_valid_raw, cat_cols)

        dtrain = lgb.Dataset(
            X_train, label=y_train.to_numpy(), categorical_feature=cat_cols
        )
        dvalid = lgb.Dataset(
            X_valid,
            label=y_valid.to_numpy(),
            reference=dtrain,
            categorical_feature=cat_cols,
        )

        model = lgb.train(
            params,
            dtrain,
            num_boost_round=NUM_BOOST_ROUND,
            valid_sets=[dvalid],
            valid_names=["valid"],
            callbacks=[
                lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                lgb.log_evaluation(-1),
            ],
        )

        best_iter = (
            int(model.best_iteration) if model.best_iteration > 0 else NUM_BOOST_ROUND
        )
        best_iterations.append(best_iter)

        oof_predictions[valid_idx] = model.predict(X_valid, num_iteration=best_iter)

    cv_rmse = float(np.sqrt(mean_squared_error(y, oof_predictions)))
    print(f"5-fold CV RMSE: {cv_rmse:.6f}")

    final_rounds = int(np.clip(np.mean(best_iterations) + 10, 10, NUM_BOOST_ROUND))
    X_train_final, X_test_final, _ = preprocess_pair(train, test, cat_cols)

    dtrain_final = lgb.Dataset(
        X_train_final, label=y.to_numpy(), categorical_feature=cat_cols
    )
    final_model = lgb.train(
        params,
        dtrain_final,
        num_boost_round=final_rounds,
        callbacks=[lgb.log_evaluation(-1)],
    )

    test_predictions = final_model.predict(X_test_final)
    submission = pd.DataFrame(
        {
            "id": np.arange(len(test)),
            "Score": test_predictions,
        }
    )
    submission_path = os.path.join(WORK_DIR, "submission.csv")
    submission.to_csv(submission_path, index=False)
    print(f"Saved submission to {submission_path}")


if __name__ == "__main__":
    main()
