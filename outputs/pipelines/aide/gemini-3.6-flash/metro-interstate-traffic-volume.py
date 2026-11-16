import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error


def main():
    # 1. Load data
    train_path = "./input/train.parquet"
    test_path = "./input/test.parquet"
    output_dir = "./working"
    os.makedirs(output_dir, exist_ok=True)

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    # Save original test length and indices
    n_train = len(train_df)
    n_test = len(test_df)

    # Combine datasets for consistent preprocessing and feature engineering
    test_df["traffic_volume"] = np.nan
    df = pd.concat([train_df, test_df], axis=0, ignore_index=True)

    # 2. Data Cleaning
    # Handle non-physical zero temperatures
    df["temp"] = df["temp"].replace(0, np.nan)
    df["temp"] = df["temp"].interpolate(method="linear").ffill().bfill()

    # Clip extreme rain_1h outliers (e.g., > 100 mm)
    df["rain_1h"] = df["rain_1h"].clip(upper=100.0)

    # 3. Feature Engineering
    df["date_time"] = pd.to_datetime(df["date_time"])

    # Sort temporally to compute dynamic rolling weather statistics safely
    df["orig_id"] = np.arange(len(df))
    df = df.sort_values("date_time").reset_index(drop=True)

    # Temporal features
    df["hour"] = df["date_time"].dt.hour
    df["dayofweek"] = df["date_time"].dt.dayofweek
    df["month"] = df["date_time"].dt.month
    df["year"] = df["date_time"].dt.year
    df["dayofyear"] = df["date_time"].dt.dayofyear
    df["quarter"] = df["date_time"].dt.quarter
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["is_rush_hour"] = (
        ((df["hour"].isin([7, 8, 9, 16, 17, 18])) & (df["dayofweek"] < 5))
    ).astype(int)

    # Time trend (hours elapsed since start)
    start_time = df["date_time"].min()
    df["hours_elapsed"] = (df["date_time"] - start_time).dt.total_seconds() / 3600.0

    # Cyclical encodings
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7.0)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)

    # Interaction categorical keys
    df["hour_dow"] = df["hour"].astype(str) + "_" + df["dayofweek"].astype(str)
    df["hour_month"] = df["hour"].astype(str) + "_" + df["month"].astype(str)

    # Weather rolling / lag features (features only, no target leakage)
    df["temp_diff_1h"] = df["temp"] - df["temp"].shift(1).bfill()
    df["temp_roll_mean_3h"] = df["temp"].rolling(3, min_periods=1).mean()
    df["temp_roll_mean_24h"] = df["temp"].rolling(24, min_periods=1).mean()
    df["rain_roll_sum_3h"] = df["rain_1h"].rolling(3, min_periods=1).sum()
    df["rain_roll_sum_24h"] = df["rain_1h"].rolling(24, min_periods=1).sum()
    df["clouds_roll_mean_3h"] = df["clouds_all"].rolling(3, min_periods=1).mean()

    # Restore original dataset ordering
    df = df.sort_values("orig_id").reset_index(drop=True)
    df = df.drop(columns=["orig_id"])

    # Categorical columns
    cat_cols = [
        "holiday",
        "weather_main",
        "weather_description",
        "hour_dow",
        "hour_month",
    ]
    for col in cat_cols:
        df[col] = df[col].astype("category")

    # Split back into train and test
    train = df.iloc[:n_train].copy().reset_index(drop=True)
    test = df.iloc[n_train:].copy().reset_index(drop=True)

    features = [c for c in train.columns if c not in ["traffic_volume", "date_time"]]
    target = "traffic_volume"

    # 4. Cross-Validation and LightGBM Modeling
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(train))
    test_preds = np.zeros(len(test))

    lgb_params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "n_estimators": 3000,
        "learning_rate": 0.03,
        "num_leaves": 127,
        "max_depth": -1,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }

    for fold, (train_idx, val_idx) in enumerate(kf.split(train)):
        X_train, y_train = train.loc[train_idx, features], train.loc[train_idx, target]
        X_val, y_val = train.loc[val_idx, features], train.loc[val_idx, target]

        model = lgb.LGBMRegressor(**lgb_params)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
        )

        val_pred = model.predict(X_val)
        oof_preds[val_idx] = val_pred
        test_preds += model.predict(test[features]) / kf.n_splits

    # Clip negative predictions
    oof_preds = np.clip(oof_preds, 0, None)
    test_preds = np.clip(test_preds, 0, None)

    # Compute metric
    cv_rmse = root_mean_squared_error(train[target], oof_preds)
    print(f"Validation Metric (CV RMSE): {cv_rmse:.4f}")

    # 5. Create submission file
    submission = pd.DataFrame(
        {"id": np.arange(len(test_preds)), "traffic_volume": test_preds}
    )
    submission_path = os.path.join(output_dir, "submission.csv")
    submission.to_csv(submission_path, index=False)


if __name__ == "__main__":
    main()
