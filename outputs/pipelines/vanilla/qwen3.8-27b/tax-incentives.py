import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
import lightgbm as lgb
from lightgbm import LGBMRegressor

warnings.filterwarnings("ignore")

TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/strable/tax-incentives/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/strable/tax-incentives/data/test.parquet"
TARGET = "credit_amount"

ID_COLS = {
    "taxcredit_cmbc_id",
    "taxcreditid",
    "onwardtaxcreditid",
}

DATE_COLS = [
    "createdon",
    "enddate",
    "modifiedon",
    "startdate",
]

NUMERIC_COLS = [
    "msrplimit",
    "phaseout25amount",
    "phaseout50amount",
    "year",
]


def clean_missing_strings(series: pd.Series) -> pd.Series:
    s = series.astype(str)
    s = s.str.strip()
    s = s.replace(
        {
            "nan": "missing",
            "NaN": "missing",
            "None": "missing",
            "none": "missing",
            "": "missing",
            "<NA>": "missing",
        }
    )
    return s


def add_date_features(df: pd.DataFrame, col: str) -> pd.DataFrame:
    dt = pd.to_datetime(df[col], errors="coerce", utc=True)
    base = f"{col}_"
    df[f"{base}year"] = dt.dt.year
    df[f"{base}month"] = dt.dt.month
    df[f"{base}day"] = dt.dt.day
    df[f"{base}dayofweek"] = dt.dt.dayofweek
    df[f"{base}days_since_2000"] = (
        dt - pd.Timestamp("2000-01-01", tz="UTC")
    ).dt.days
    return df


def prepare_data(train: pd.DataFrame, test: pd.DataFrame):
    train = train.dropna(subset=[TARGET]).copy()
    y = train[TARGET].astype(float)
    X = train.drop(columns=[TARGET]).copy()

    if TARGET in test.columns:
        test = test.drop(columns=[TARGET]).copy()

    common_cols = [c for c in X.columns if c in test.columns]
    X = X[common_cols].copy()
    test = test[common_cols].copy()

    drop_cols = [c for c in ID_COLS if c in X.columns]
    X = X.drop(columns=drop_cols, errors="ignore")
    test = test.drop(columns=drop_cols, errors="ignore")

    for col in DATE_COLS:
        if col in X.columns:
            X = add_date_features(X, col)
            test = add_date_features(test, col)
            X = X.drop(columns=[col], errors="ignore")
            test = test.drop(columns=[col], errors="ignore")

    for col in NUMERIC_COLS:
        if col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce")
            test[col] = pd.to_numeric(test[col], errors="coerce")

    categorical_cols = []
    for col in X.columns:
        if (
            X[col].dtype == object
            or X[col].dtype == "category"
            or pd.api.types.is_string_dtype(X[col])
        ):
            categorical_cols.append(col)

    for col in categorical_cols:
        X[col] = clean_missing_strings(X[col])
        test[col] = clean_missing_strings(test[col])

        categories = sorted(X[col].unique().tolist())
        if "missing" not in categories:
            categories.append("missing")

        X[col] = X[col].astype("category").cat.set_categories(categories)
        test[col] = test[col].where(test[col].isin(categories), "missing")
        test[col] = test[col].astype("category").cat.set_categories(categories)

    numeric_cols = [c for c in X.columns if c not in categorical_cols]
    for col in numeric_cols:
        X[col] = pd.to_numeric(X[col], errors="coerce")
        test[col] = pd.to_numeric(test[col], errors="coerce")

    return X, y, test, categorical_cols


def make_model(n_estimators: int, categorical_cols):
    return LGBMRegressor(
        n_estimators=n_estimators,
        learning_rate=0.03,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=20,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.9,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
        categorical_feature=categorical_cols,
    )


def main():
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    X, y, X_test, categorical_cols = prepare_data(train_df, test_df)

    if X.shape[1] == 0 or len(X) == 0:
        pred = np.full(len(X_test), float(y.mean()) if len(y) else 0.0)
    else:
        X_fit, X_val, y_fit, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = make_model(5000, categorical_cols)
        model.fit(
            X_fit,
            y_fit,
            eval_set=[(X_val, y_val)],
            eval_metric="rmse",
            callbacks=[
                lgb.early_stopping(100, verbose=False),
                lgb.log_evaluation(0),
            ],
        )

        val_pred = model.predict(X_val)
        val_rmse = root_mean_squared_error(y_val, val_pred)
        print(f"Validation RMSE: {val_rmse:.6f}")

        best_iter = int(getattr(model, "best_iteration_", 0) or 0)
        best_iter = max(1, min(best_iter, 5000))

        final_model = make_model(best_iter, categorical_cols)
        final_model.fit(X, y)
        pred = final_model.predict(X_test)

    pred = np.asarray(pred, dtype=float)
    pred = np.nan_to_num(pred, nan=float(y.mean()) if len(y) else 0.0)
    pred = np.clip(pred, 0.0, None)

    output = pd.DataFrame(
        {
            "id": np.arange(len(pred), dtype=int),
            "credit_amount": pred,
        }
    )
    output.to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    main()