import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import lightgbm as lgb

TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/kaggle/optiver-trading-at-the-close/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/kaggle/optiver-trading-at-the-close/data/test.parquet"
TARGET = "target"
DROP_COLS = ["row_id", "date_id", "time_id"]


def safe_divide(numerator, denominator):
    numerator = np.asarray(numerator, dtype=np.float32)
    denominator = np.asarray(denominator, dtype=np.float32)
    out = np.zeros_like(numerator, dtype=np.float32)
    np.divide(
        numerator,
        denominator,
        out=out,
        where=(denominator != 0) & np.isfinite(denominator),
    )
    return out


def encode_categoricals(train_df, test_df):
    cat_cols = []
    for col in train_df.columns:
        dtype = train_df[col].dtype
        if dtype == object or str(dtype) == "category":
            cat_cols.append(col)

    for col in cat_cols:
        train_values = train_df[col].astype(str)
        categories = pd.Index(train_values.unique())
        mapping = {value: idx for idx, value in enumerate(categories)}

        train_df[col] = train_values.map(mapping).astype(np.int32)

        if col in test_df.columns:
            test_values = test_df[col].astype(str)
            test_df[col] = test_values.map(mapping).fillna(-1).astype(np.int32)
        else:
            test_df[col] = -1

    return train_df, test_df, cat_cols


def ensure_numeric(train_df, test_df):
    for col in train_df.columns:
        if not pd.api.types.is_numeric_dtype(train_df[col]):
            train_df[col] = pd.to_numeric(train_df[col], errors="coerce")

        if col in test_df.columns and not pd.api.types.is_numeric_dtype(test_df[col]):
            test_df[col] = pd.to_numeric(test_df[col], errors="coerce")

    for df in (train_df, test_df):
        for col in df.select_dtypes(include=[np.number]).columns:
            df[col] = np.where(np.isfinite(df[col]), df[col], np.nan)

    return train_df, test_df


def add_features(df):
    df = df.copy()

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col] = df[col].astype(np.float32)

    if {"ask_price", "bid_price"}.issubset(df.columns):
        df["spread"] = df["ask_price"] - df["bid_price"]
        df["mid_price"] = (df["ask_price"] + df["bid_price"]) / 2.0
        df["mid_minus_wap"] = df["mid_price"] - df["wap"]

    if {"wap", "reference_price"}.issubset(df.columns):
        df["wap_minus_reference"] = df["wap"] - df["reference_price"]

    if {"bid_price", "reference_price"}.issubset(df.columns):
        df["bid_minus_reference"] = df["bid_price"] - df["reference_price"]

    if {"ask_price", "reference_price"}.issubset(df.columns):
        df["ask_minus_reference"] = df["ask_price"] - df["reference_price"]

    if {"far_price", "near_price"}.issubset(df.columns):
        df["far_minus_near"] = df["far_price"] - df["near_price"]

    if {"far_price", "reference_price"}.issubset(df.columns):
        df["far_minus_reference"] = df["far_price"] - df["reference_price"]

    if {"near_price", "reference_price"}.issubset(df.columns):
        df["near_minus_reference"] = df["near_price"] - df["reference_price"]

    if {"imbalance_size", "imbalance_buy_sell_flag"}.issubset(df.columns):
        df["signed_imbalance"] = df["imbalance_size"] * df["imbalance_buy_sell_flag"]
        df["abs_imbalance"] = np.abs(df["imbalance_size"])

    if {"bid_size", "ask_size"}.issubset(df.columns):
        total_size = df["bid_size"] + df["ask_size"]
        df["bid_size_ratio"] = safe_divide(df["bid_size"], total_size)
        df["ask_size_ratio"] = safe_divide(df["ask_size"], total_size)
        df["size_diff"] = df["bid_size"] - df["ask_size"]
        df["bid_ask_size_ratio"] = safe_divide(df["bid_size"], df["ask_size"])

    for col in ["matched_size", "bid_size", "ask_size", "imbalance_size"]:
        if col in df.columns:
            df[f"log_{col}"] = np.log1p(np.abs(df[col])).astype(np.float32)

    if "seconds_in_bucket" in df.columns:
        seconds = df["seconds_in_bucket"].astype(np.float32)
        df["seconds_sin"] = np.sin(2.0 * np.pi * seconds / 600.0).astype(np.float32)
        df["seconds_cos"] = np.cos(2.0 * np.pi * seconds / 600.0).astype(np.float32)

    return df


def main():
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    if TARGET not in train_df.columns:
        raise ValueError(f"Target column '{TARGET}' not found in train data.")

    train_df = train_df[train_df[TARGET].notna()].copy()
    y = train_df[TARGET].astype(np.float32)
    train_df = train_df.drop(columns=[TARGET])

    date_id_for_split = train_df["date_id"].copy() if "date_id" in train_df.columns else None

    for col in DROP_COLS:
        if col in train_df.columns:
            train_df = train_df.drop(columns=[col])
        if col in test_df.columns:
            test_df = test_df.drop(columns=[col])

    train_df, test_df, encoded_cat_cols = encode_categoricals(train_df, test_df)
    train_df, test_df = ensure_numeric(train_df, test_df)

    train_df = add_features(train_df)
    test_df = add_features(test_df)

    feature_cols = [
        col
        for col in train_df.columns
        if col in test_df.columns
        and pd.api.types.is_numeric_dtype(train_df[col])
        and train_df[col].notna().any()
    ]

    X_all = train_df[feature_cols].astype(np.float32)
    X_test = test_df[feature_cols].astype(np.float32)

    categorical_feature = [col for col in encoded_cat_cols if col in X_all.columns]
    if "stock_id" in X_all.columns:
        categorical_feature.append("stock_id")
    categorical_feature = list(dict.fromkeys(categorical_feature))

    val_mask = None
    if date_id_for_split is not None:
        dates = np.unique(date_id_for_split.to_numpy())
        if len(dates) > 20:
            val_dates = np.sort(dates)[-max(1, int(0.10 * len(dates))):]
            val_mask = date_id_for_split.isin(val_dates).to_numpy()

    if val_mask is None or val_mask.sum() == 0 or val_mask.sum() == len(val_mask):
        val_mask = None
        if len(X_all) > 10:
            _, val_idx = train_test_split(
                np.arange(len(X_all)),
                test_size=0.2,
                random_state=42,
                shuffle=True,
            )
            val_mask = np.zeros(len(X_all), dtype=bool)
            val_mask[val_idx] = True

    params = {
        "objective": "regression_l1",
        "n_estimators": 700,
        "learning_rate": 0.05,
        "num_leaves": 127,
        "max_depth": -1,
        "min_child_samples": 200,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": -1,
    }

    fit_kwargs = {}
    if categorical_feature:
        fit_kwargs["categorical_feature"] = categorical_feature

    if val_mask is not None and val_mask.sum() > 0 and (~val_mask).sum() > 0:
        X_train = X_all[~val_mask]
        y_train = y[~val_mask]
        X_val = X_all[val_mask]
        y_val = y[val_mask]

        val_model = lgb.LGBMRegressor(**params)
        val_model.fit(X_train, y_train, **fit_kwargs)
        val_pred = val_model.predict(X_val)
        val_mae = float(np.mean(np.abs(val_pred - y_val)))

    final_model = lgb.LGBMRegressor(**params)
    final_model.fit(X_all, y, **fit_kwargs)

    predictions = final_model.predict(X_test)
    predictions = np.nan_to_num(predictions, nan=0.0, posinf=0.0, neginf=0.0)

    output = pd.DataFrame(
        {
            "id": np.arange(len(predictions), dtype=int),
            "target": predictions.astype(np.float64),
        }
    )
    output.to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    main()