import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import root_mean_squared_error


def main():
    os.makedirs("./working", exist_ok=True)

    # 1. Load data
    train = pd.read_parquet("./input/train.parquet")
    test = pd.read_parquet("./input/test.parquet")

    # Ensure proper datetime type
    train["date"] = pd.to_datetime(train["date"])
    test["date"] = pd.to_datetime(test["date"])

    # Sort for time-series operations
    train = train.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)
    test = test.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)

    # Target log-transform
    train["log_sales"] = np.log1p(np.clip(train["sales"].values, 0, None))
    test["sales"] = 0.0
    test["log_sales"] = 0.0

    # Combine data to create lag and rolling features across the train-test boundary
    df = pd.concat([train, test], ignore_index=True)
    df = df.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)

    # Calendar features
    df["year"] = df["date"].dt.year.astype("int16")
    df["month"] = df["date"].dt.month.astype("int8")
    df["day"] = df["date"].dt.day.astype("int8")
    df["dayofweek"] = df["date"].dt.dayofweek.astype("int8")
    df["dayofyear"] = df["date"].dt.dayofyear.astype("int16")
    df["is_weekend"] = (df["dayofweek"] >= 5).astype("int8")
    df["weekofyear"] = df["date"].dt.isocalendar().week.astype("int8")

    # Group series identifier
    df["family"] = df["family"].astype("category")
    df["store_nbr"] = df["store_nbr"].astype("category")

    # Lag and Rolling Features based on lag >= 16 days (since test horizon is 16 days)
    grouped = df.groupby(["store_nbr", "family"])["log_sales"]

    lags = [16, 17, 18, 19, 20, 21, 28, 35, 42, 56]
    for lag in lags:
        df[f"sales_lag_{lag}"] = grouped.shift(lag).astype("float32")

    for window in [7, 14, 30, 60]:
        df[f"sales_rolling_mean_16_{window}"] = (
            grouped.shift(16).rolling(window, min_periods=1).mean().astype("float32")
        )
        df[f"sales_rolling_std_16_{window}"] = (
            grouped.shift(16).rolling(window, min_periods=1).std().astype("float32")
        )
        df[f"sales_rolling_mean_28_{window}"] = (
            grouped.shift(28).rolling(window, min_periods=1).mean().astype("float32")
        )

    # Promotion rolling features
    grouped_promo = df.groupby(["store_nbr", "family"])["onpromotion"]
    for window in [7, 14, 30]:
        df[f"promo_rolling_mean_{window}"] = (
            grouped_promo.rolling(window, min_periods=1)
            .mean()
            .astype("float32")
            .reset_index(drop=True)
        )

    # Split back into train and test
    train_df = df[df["id"].isin(train["id"])].copy()
    test_df = df[df["id"].isin(test["id"])].copy()

    # Filter early dates where lags are not fully available
    train_df = train_df[train_df["date"] >= "2014-01-01"].reset_index(drop=True)

    # Validation split: hold out the last 16 days of training data
    max_train_date = train_df["date"].max()
    val_start_date = max_train_date - pd.Timedelta(days=15)

    val_mask = train_df["date"] >= val_start_date
    tr_data = train_df[~val_mask].reset_index(drop=True)
    va_data = train_df[val_mask].reset_index(drop=True)

    drop_cols = ["id", "date", "sales", "log_sales"]
    feature_cols = [c for c in train_df.columns if c not in drop_cols]

    X_tr = tr_data[feature_cols]
    y_tr = tr_data["log_sales"]
    X_va = va_data[feature_cols]
    y_va = va_data["log_sales"]
    X_test = test_df[feature_cols]

    # Train LightGBM model
    params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 127,
        "max_depth": -1,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }

    dtrain = lgb.Dataset(X_tr, label=y_tr)
    dval = lgb.Dataset(X_va, label=y_va, reference=dtrain)

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=1500,
        valid_sets=[dtrain, dval],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )

    # Evaluate on validation set
    val_preds_log = model.predict(X_va, num_iteration=model.best_iteration)
    val_preds = np.expm1(np.clip(val_preds_log, 0, None))
    val_true = np.expm1(y_va.values)

    # RMSLE is RMSE of log1p
    rmsle = np.sqrt(np.mean((np.log1p(val_preds) - np.log1p(val_true)) ** 2))
    print(f"Validation RMSLE: {rmsle:.5f}")

    # Retrain on full train data using optimal iteration count
    dfull = lgb.Dataset(train_df[feature_cols], label=train_df["log_sales"])
    full_model = lgb.train(
        params,
        dfull,
        num_boost_round=int(model.best_iteration * 1.1),
        callbacks=[lgb.log_evaluation(period=0)],
    )

    # Predict on test data
    test_preds_log = full_model.predict(X_test)
    test_preds = np.expm1(np.clip(test_preds_log, 0, None))
    test_df["sales"] = test_preds

    # Format submission
    submission = test_df[["id", "sales"]].sort_values("id").reset_index(drop=True)
    submission["id"] = submission["id"].astype(int)
    submission.to_csv("./working/submission.csv", index=False)
    print(f"Submission saved to ./working/submission.csv with shape {submission.shape}")


if __name__ == "__main__":
    main()
