import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor

TARGET = "Original Principal Amount (US$)"
TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/strable/ibrd-statement-loans-guarantees/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/strable/ibrd-statement-loans-guarantees/data/test.parquet"

ID_COLS = ["Loan Number", "Project ID", "Project Name"]
DATE_COLS = [
    "First Repayment Date",
    "Last Repayment Date",
    "Agreement Signing Date",
    "Board Approval Date",
    "Effective Date (Most Recent)",
    "Closed Date (Most Recent)",
    "Last Disbursement Date",
]
DATE_DIFFS = [
    ("term_days", "First Repayment Date", "Last Repayment Date"),
    ("disbursement_days", "Effective Date (Most Recent)", "Last Disbursement Date"),
    ("closed_days", "Effective Date (Most Recent)", "Closed Date (Most Recent)"),
    ("signing_to_effective_days", "Agreement Signing Date", "Effective Date (Most Recent)"),
    ("board_to_signing_days", "Board Approval Date", "Agreement Signing Date"),
    ("last_repayment_to_closed_days", "Last Repayment Date", "Closed Date (Most Recent)"),
]
MISSING_TOKENS = ["", "None", "nan", "NaN", "null", "NULL", "missing"]


def get_categorical_columns(df):
    exclude = set([TARGET] + ID_COLS + DATE_COLS)
    cat_cols = []
    for col in df.columns:
        if col in exclude:
            continue
        dtype = df[col].dtype
        if dtype == object or str(dtype) == "category":
            cat_cols.append(col)
    return cat_cols


def preprocess(df, cat_cols):
    df = df.copy()

    if TARGET in df.columns:
        df = df.drop(columns=[TARGET])

    for col in ID_COLS:
        if col in df.columns:
            df = df.drop(columns=[col])

    parsed_dates = {}
    for col in DATE_COLS:
        if col not in df.columns:
            continue
        dt = pd.to_datetime(df[col], errors="coerce")
        parsed_dates[col] = dt
        df[col + "_year"] = dt.dt.year.astype("float32")
        df[col + "_month"] = dt.dt.month.astype("float32")
        df[col + "_day"] = dt.dt.day.astype("float32")
        df[col + "_days"] = (dt - pd.Timestamp("1900-01-01")).dt.days.astype("float32")

    for name, start_col, end_col in DATE_DIFFS:
        if start_col in parsed_dates and end_col in parsed_dates:
            df[name] = (parsed_dates[end_col] - parsed_dates[start_col]).dt.days.astype("float32")

    for col in DATE_COLS:
        if col in df.columns:
            df = df.drop(columns=[col])

    for col in cat_cols:
        if col not in df.columns:
            continue
        df[col] = (
            df[col]
            .astype("object")
            .fillna("missing")
            .astype(str)
            .str.strip()
        )
        df[col] = df[col].replace(MISSING_TOKENS, "missing")

    for col in df.columns:
        if col not in cat_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)

    return df


def main():
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    if TARGET not in train_df.columns:
        raise ValueError(f"Target column {TARGET!r} not found in train data.")

    train_df = train_df.dropna(subset=[TARGET])
    train_df[TARGET] = train_df[TARGET].astype(float)

    cat_cols = get_categorical_columns(train_df)

    X_train_full = preprocess(train_df, cat_cols)
    X_test = preprocess(test_df, cat_cols)
    y_train_full = train_df[TARGET].to_numpy(dtype=float)

    X_test = X_test.reindex(columns=X_train_full.columns)

    cat_feature_indices = [
        X_train_full.columns.get_loc(col)
        for col in cat_cols
        if col in X_train_full.columns
    ]

    if len(X_train_full) < 20:
        best_iterations = 500
    else:
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train_full,
            y_train_full,
            test_size=0.2,
            random_state=42,
            shuffle=True,
        )

        val_model = CatBoostRegressor(
            iterations=3000,
            learning_rate=0.05,
            depth=6,
            l2_leaf_reg=3.0,
            border_count=128,
            loss_function="RMSE",
            eval_metric="RMSE",
            random_seed=42,
            verbose=0,
            allow_writing_files=False,
            cat_features=cat_feature_indices,
        )
        val_model.fit(
            X_tr,
            y_tr,
            eval_set=(X_val, y_val),
            early_stopping_rounds=100,
            verbose=0,
        )

        val_pred = val_model.predict(X_val)
        val_rmse = float(np.sqrt(mean_squared_error(y_val, val_pred)))
        print(f"Validation RMSE: {val_rmse:.6f}")

        best_iterations = max(1, int(getattr(val_model, "best_iteration_", 3000)))

    final_model = CatBoostRegressor(
        iterations=best_iterations,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=3.0,
        border_count=128,
        loss_function="RMSE",
        random_seed=42,
        verbose=0,
        allow_writing_files=False,
        cat_features=cat_feature_indices,
    )
    final_model.fit(X_train_full, y_train_full, verbose=0)

    predictions = final_model.predict(X_test)
    predictions = np.asarray(predictions, dtype=float)
    predictions = np.where(np.isfinite(predictions), predictions, 0.0)
    predictions = np.clip(predictions, 0.0, None)

    output = pd.DataFrame(
        {
            "id": np.arange(len(predictions), dtype=int),
            TARGET: predictions,
        }
    )
    output.to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    main()