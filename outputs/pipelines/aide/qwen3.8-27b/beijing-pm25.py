import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error


def clean_data(df):
    df = df.copy()

    numeric_cols = [
        "year",
        "month",
        "day",
        "hour",
        "pm2.5",
        "DEWP",
        "TEMP",
        "PRES",
        "Iws",
        "Is",
        "Ir",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)

    # Remove physically impossible negative values.
    for col in ["Iws", "Is", "Ir", "PRES", "pm2.5"]:
        if col in df.columns:
            df.loc[df[col] < 0, col] = np.nan

    # Very broad sanity bounds for temperature and dew point.
    for col in ["TEMP", "DEWP"]:
        if col in df.columns:
            df.loc[(df[col] < -100) | (df[col] > 100), col] = np.nan

    if "cbwd" in df.columns:
        df["cbwd"] = df["cbwd"].astype(str).str.strip()
        df["cbwd"] = df["cbwd"].replace(
            {"": "NA", "nan": "NA", "None": "NA", "NaN": "NA"}
        )

    return df


def add_features(df):
    df = df.copy()

    date_parts = df[["year", "month", "day", "hour"]].copy()
    for col in date_parts.columns:
        date_parts[col] = pd.to_numeric(date_parts[col], errors="coerce")

    dt = pd.to_datetime(date_parts, errors="coerce")

    df["dayofweek"] = dt.dt.dayofweek.fillna(-1)
    df["dayofyear"] = dt.dt.dayofyear.fillna(-1)
    df["is_weekend"] = np.where(df["dayofweek"] >= 5, 1, 0)

    hour = pd.to_numeric(df["hour"], errors="coerce")
    month = pd.to_numeric(df["month"], errors="coerce")

    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)

    return df


CAT_COLS = ["cbwd"]
BASE_FEATURES = [
    "year",
    "month",
    "day",
    "hour",
    "DEWP",
    "TEMP",
    "PRES",
    "Iws",
    "Is",
    "Ir",
    "cbwd",
]
DERIVED_FEATURES = [
    "dayofweek",
    "dayofyear",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
]
FEATURES = BASE_FEATURES + DERIVED_FEATURES


def fit_category_map(train_part):
    maps = {}
    for col in CAT_COLS:
        if col not in train_part.columns:
            continue
        values = train_part[col].astype(str).unique()
        maps[col] = {value: idx for idx, value in enumerate(sorted(values))}
    return maps


def encode_categories(df, maps):
    df = df.copy()
    for col in CAT_COLS:
        if col not in df.columns:
            continue
        mapping = maps.get(col, {})
        df[col] = df[col].astype(str).map(mapping).fillna(-1).astype(int)
    return df


def main():
    train = pd.read_parquet("./input/train.parquet")
    test = pd.read_parquet("./input/test.parquet")

    train = clean_data(train)
    test = clean_data(test)

    train = train[np.isfinite(train["pm2.5"])].copy()

    train = add_features(train)
    test = add_features(test)

    train = train.sort_values(
        ["year", "month", "day", "hour"], na_position="last"
    ).reset_index(drop=True)

    if len(train) < 2:
        os.makedirs("./working", exist_ok=True)
        submission = pd.DataFrame({"id": np.arange(len(test), dtype=int), "pm2.5": 0.0})
        submission.to_csv("./working/submission.csv", index=False)
        print("Validation RMSE: nan")
        return

    val_size = max(1, int(0.10 * len(train)))
    if len(train) - val_size < 10:
        val_size = max(1, len(train) // 2)
    val_size = min(val_size, len(train) - 1)

    train_part = train.iloc[:-val_size].reset_index(drop=True)
    val_part = train.iloc[-val_size:].reset_index(drop=True)

    features = [f for f in FEATURES if f in train_part.columns and f in test.columns]

    cat_map_val = fit_category_map(train_part)

    X_train = encode_categories(train_part, cat_map_val)[features]
    y_train = train_part["pm2.5"].astype(float)

    X_val = encode_categories(val_part, cat_map_val)[features]
    y_val = val_part["pm2.5"].astype(float)

    params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "max_depth": -1,
        "min_child_samples": 50,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "verbosity": -1,
        "seed": 42,
    }

    categorical_feature = ["cbwd"] if "cbwd" in features else None

    dtrain = lgb.Dataset(
        X_train, label=y_train, categorical_feature=categorical_feature
    )
    dval = lgb.Dataset(
        X_val, label=y_val, categorical_feature=categorical_feature, reference=dtrain
    )

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=3000,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
    )

    val_pred = model.predict(X_val, num_iteration=model.best_iteration)
    val_pred = np.nan_to_num(
        val_pred, nan=float(y_val.mean()), posinf=float(y_val.max()), neginf=0.0
    )
    val_pred = np.clip(val_pred, 0.0, None)

    val_rmse = float(np.sqrt(mean_squared_error(y_val, val_pred)))
    print(f"Validation RMSE: {val_rmse:.6f}")

    # Retrain on the full training set for the final submission.
    cat_map_full = fit_category_map(train)
    X_full = encode_categories(train, cat_map_full)[features]
    y_full = train["pm2.5"].astype(float)

    dfull = lgb.Dataset(X_full, label=y_full, categorical_feature=categorical_feature)

    best_iter = (
        model.best_iteration
        if model.best_iteration and model.best_iteration > 0
        else 3000
    )

    final_model = lgb.train(
        params, dfull, num_boost_round=best_iter, callbacks=[lgb.log_evaluation(-1)]
    )

    X_test = encode_categories(test, cat_map_full)[features]
    test_pred = final_model.predict(X_test)
    test_pred = np.nan_to_num(
        test_pred, nan=float(y_full.mean()), posinf=float(y_full.max()), neginf=0.0
    )
    test_pred = np.clip(test_pred, 0.0, None)

    os.makedirs("./working", exist_ok=True)
    submission = pd.DataFrame(
        {"id": np.arange(len(test), dtype=int), "pm2.5": test_pred.astype(float)}
    )
    submission.to_csv("./working/submission.csv", index=False)


if __name__ == "__main__":
    main()
