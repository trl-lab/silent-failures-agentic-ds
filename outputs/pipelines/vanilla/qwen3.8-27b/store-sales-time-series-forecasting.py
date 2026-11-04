import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/kaggle/store-sales-time-series-forecasting/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/kaggle/store-sales-time-series-forecasting/data/test.parquet"
TARGET = "sales"
ID_COL = "id"
DATE_COL = "date"


def rmsle(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_pred = np.clip(y_pred, 0.0, None)
    return float(np.sqrt(np.mean((np.log1p(y_true) - np.log1p(y_pred)) ** 2)))


def add_date_features(df):
    df = df.copy()
    if DATE_COL not in df.columns:
        return df

    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")

    df["year"] = df[DATE_COL].dt.year.astype("float64")
    df["month"] = df[DATE_COL].dt.month.astype("float64")
    df["day"] = df[DATE_COL].dt.day.astype("float64")
    df["dayofweek"] = df[DATE_COL].dt.dayofweek.astype("float64")
    df["dayofyear"] = df[DATE_COL].dt.dayofyear.astype("float64")
    df["is_weekend"] = (df["dayofweek"] >= 5).astype("float64")

    df["month_sin"] = np.sin(2.0 * np.pi * df["month"] / 12.0)
    df["month_cos"] = np.cos(2.0 * np.pi * df["month"] / 12.0)
    df["dayofweek_sin"] = np.sin(2.0 * np.pi * df["dayofweek"] / 7.0)
    df["dayofweek_cos"] = np.cos(2.0 * np.pi * df["dayofweek"] / 7.0)

    return df


def make_store_family(df):
    if "store_nbr" in df.columns and "family" in df.columns:
        df = df.copy()
        df["store_family"] = df["store_nbr"].astype(str) + "_" + df["family"].astype(str)
    return df


def prepare_data(train_df, test_df):
    train_df = train_df.copy()
    test_df = test_df.copy()

    train_df = add_date_features(train_df)
    test_df = add_date_features(test_df)

    train_df = make_store_family(train_df)
    test_df = make_store_family(test_df)

    train_df[TARGET] = pd.to_numeric(train_df[TARGET], errors="coerce")
    train_df[TARGET] = train_df[TARGET].replace([np.inf, -np.inf], np.nan)

    drop_cols = [TARGET]
    if DATE_COL in train_df.columns:
        drop_cols.append(DATE_COL)

    train_df = train_df.dropna(subset=drop_cols).copy()

    preferred_categoricals = ["family", "store_nbr", "store_family"]
    categorical_cols = [col for col in preferred_categoricals if col in train_df.columns]

    for col in train_df.columns:
        if col in categorical_cols:
            continue
        if col in {TARGET, ID_COL, DATE_COL}:
            continue
        if (
            train_df[col].dtype == object
            or isinstance(train_df[col].dtype, pd.CategoricalDtype)
            or pd.api.types.is_datetime64_any_dtype(train_df[col])
        ):
            categorical_cols.append(col)

    for col in categorical_cols:
        if col not in test_df.columns:
            test_df[col] = np.nan

        train_df[col] = train_df[col].astype(str)
        test_df[col] = test_df[col].astype(str)

        train_df[col] = train_df[col].replace({"nan": "missing", "None": "missing", "": "missing"})
        test_df[col] = test_df[col].replace({"nan": "missing", "None": "missing", "": "missing"})

        categories = np.sort(
            np.unique(np.concatenate((train_df[col].unique(), np.array(["missing"]))))
        )

        train_df[col] = pd.Categorical(train_df[col], categories=categories)
        test_df[col] = pd.Categorical(test_df[col], categories=categories)

        train_df[col] = train_df[col].fillna("missing")
        test_df[col] = test_df[col].fillna("missing")

    numeric_cols = [
        col
        for col in train_df.columns
        if col not in categorical_cols and col not in {TARGET, ID_COL, DATE_COL}
    ]

    for col in numeric_cols:
        if col not in test_df.columns:
            test_df[col] = np.nan

        train_df[col] = pd.to_numeric(train_df[col], errors="coerce")
        test_df[col] = pd.to_numeric(test_df[col], errors="coerce")

        train_df[col] = train_df[col].replace([np.inf, -np.inf], np.nan)
        test_df[col] = test_df[col].replace([np.inf, -np.inf], np.nan)

        median_value = train_df[col].median()
        if pd.isna(median_value):
            median_value = 0.0

        train_df[col] = train_df[col].fillna(median_value)
        test_df[col] = test_df[col].fillna(median_value)

    feature_cols = numeric_cols + categorical_cols

    for col in feature_cols:
        if col not in test_df.columns:
            test_df[col] = np.nan

    X_train = train_df[feature_cols]
    X_test = test_df[feature_cols]
    y_train = np.log1p(train_df[TARGET].clip(lower=0.0))
    dates = train_df[DATE_COL] if DATE_COL in train_df.columns else None

    return X_train, y_train, X_test, dates, feature_cols


def time_based_split(X_train, y_train, dates):
    if dates is None or len(dates) == 0:
        return X_train, y_train, X_train.iloc[0:0], y_train.iloc[0:0]

    max_date = dates.max()
    if pd.notna(max_date):
        cutoff = max_date - pd.Timedelta(days=13)
        val_mask = dates >= cutoff
    else:
        val_mask = pd.Series(False, index=dates.index)

    if val_mask.sum() < max(1000, int(0.01 * len(dates))):
        sorted_dates = dates.sort_values()
        val_size = max(1, int(0.10 * len(sorted_dates)))
        val_index = sorted_dates.tail(val_size).index
        val_mask = dates.index.isin(val_index)

    if val_mask.sum() == 0 or (~val_mask).sum() == 0:
        return X_train, y_train, X_train.iloc[0:0], y_train.iloc[0:0]

    train_mask = ~val_mask
    return X_train[train_mask], y_train[train_mask], X_train[val_mask], y_train[val_mask]


def main():
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    X_train, y_train, X_test, dates, feature_cols = prepare_data(train_df, test_df)

    if len(X_train) == 0 or len(feature_cols) == 0:
        pd.DataFrame(
            {
                "id": np.arange(len(test_df), dtype=int),
                "sales": 0.0,
            }
        ).to_csv("predictions.csv", index=False)
        return

    X_fit, y_fit, X_val, y_val = time_based_split(X_train, y_train, dates)

    base_params = {
        "objective": "regression",
        "metric": "l2",
        "n_estimators": 3000,
        "learning_rate": 0.05,
        "num_leaves": 127,
        "max_depth": -1,
        "min_child_samples": 100,
        "subsample": 0.85,
        "subsample_freq": 1,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }

    if len(X_fit) > 0 and len(X_val) > 0:
        val_params = base_params.copy()
        val_params["early_stopping_rounds"] = 100

        model = LGBMRegressor(**val_params)
        model.fit(X_fit, y_fit, eval_set=[(X_val, y_val)])

        best_iteration = getattr(model, "best_iteration_", 0)
        final_n_estimators = int(max(100, best_iteration)) if best_iteration else 1000
    else:
        final_n_estimators = 1000

    final_params = base_params.copy()
    final_params["n_estimators"] = final_n_estimators

    final_model = LGBMRegressor(**final_params)
    final_model.fit(X_train, y_train)

    test_pred_log = final_model.predict(X_test)
    test_pred = np.expm1(np.clip(test_pred_log, 0.0, 20.0))
    test_pred = np.clip(test_pred, 0.0, None)

    submission = pd.DataFrame(
        {
            "id": np.arange(len(test_df), dtype=int),
            "sales": test_pred,
        }
    )
    submission.to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    main()