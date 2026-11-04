import gc
import os
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error


def main():
    # Setup directories
    os.makedirs("./working", exist_ok=True)
    os.makedirs("./submission", exist_ok=True)

    print("Loading train and test parquet files...")
    train_df = pd.read_parquet("./input/train.parquet")
    test_df = pd.read_parquet("./input/test.parquet")

    # Ensure test set has standard 'id' column
    if "id" not in test_df.columns:
        if "row_id" in test_df.columns:
            test_df["id"] = test_df["row_id"].astype(np.int32)
        else:
            test_df["id"] = test_df.index.astype(np.int32)

    # Clean target: ensure non-negative and create log1p target
    if "meter_reading" in train_df.columns:
        train_df["meter_reading"] = np.maximum(0.0, train_df["meter_reading"].values)
        train_df["target"] = np.log1p(train_df["meter_reading"].values)

    # Standardize timestamp parsing
    train_df["timestamp"] = pd.to_datetime(train_df["timestamp"])
    test_df["timestamp"] = pd.to_datetime(test_df["timestamp"])

    # Time-based Train / Validation Split (Hold out last 60 days of training period)
    max_train_date = train_df["timestamp"].max()
    val_cutoff = max_train_date - pd.Timedelta(days=60)

    is_val = train_df["timestamp"] >= val_cutoff
    train_split = train_df[~is_val].copy().reset_index(drop=True)
    val_split = train_df[is_val].copy().reset_index(drop=True)

    print(
        f"Train split range: {train_split['timestamp'].min()} to {train_split['timestamp'].max()} ({len(train_split)} rows)"
    )
    print(
        f"Val split range: {val_split['timestamp'].min()} to {val_split['timestamp'].max()} ({len(val_split)} rows)"
    )

    del train_df
    gc.collect()

    # Compute building-meter and building-meter-hour target aggregations strictly on train_split
    train_split["hour"] = train_split["timestamp"].dt.hour.astype(np.int8)

    bm_stats = (
        train_split.groupby(["building_id", "meter"])["target"]
        .agg(
            bm_target_mean="mean",
            bm_target_std="std",
            bm_target_median="median",
            bm_target_min="min",
            bm_target_max="max",
        )
        .reset_index()
    )

    bmh_stats = (
        train_split.groupby(["building_id", "meter", "hour"])["target"]
        .agg(
            bmh_target_mean="mean",
            bmh_target_median="median",
        )
        .reset_index()
    )

    global_mean = float(train_split["target"].mean())
    global_std = float(train_split["target"].std())
    global_median = float(train_split["target"].median())

    # Helper function to engineer features
    def extract_features(df):
        df["hour"] = df["timestamp"].dt.hour.astype(np.int8)
        df["day"] = df["timestamp"].dt.day.astype(np.int8)
        df["dayofweek"] = df["timestamp"].dt.dayofweek.astype(np.int8)
        df["month"] = df["timestamp"].dt.month.astype(np.int8)
        df["is_weekend"] = (df["dayofweek"] >= 5).astype(np.int8)

        # Cyclical temporal encodings
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0).astype(np.float32)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0).astype(np.float32)
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0).astype(np.float32)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0).astype(np.float32)
        df["dayofweek_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7.0).astype(
            np.float32
        )
        df["dayofweek_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7.0).astype(
            np.float32
        )

        # Categorical IDs
        df["building_id"] = df["building_id"].astype(np.int16)
        df["meter"] = df["meter"].astype(np.int8)

        # Domain features
        if "square_feet" in df.columns:
            df["log_square_feet"] = np.log1p(df["square_feet"].values).astype(
                np.float32
            )
        if "air_temperature" in df.columns and "dew_temperature" in df.columns:
            df["temp_dew_diff"] = (
                df["air_temperature"] - df["dew_temperature"]
            ).astype(np.float32)

        # Merge target statistics
        df = df.merge(bm_stats, on=["building_id", "meter"], how="left")
        df["bm_target_mean"] = (
            df["bm_target_mean"].fillna(global_mean).astype(np.float32)
        )
        df["bm_target_std"] = df["bm_target_std"].fillna(global_std).astype(np.float32)
        df["bm_target_median"] = (
            df["bm_target_median"].fillna(global_median).astype(np.float32)
        )
        df["bm_target_min"] = df["bm_target_min"].fillna(0.0).astype(np.float32)
        df["bm_target_max"] = (
            df["bm_target_max"].fillna(global_mean + 2 * global_std).astype(np.float32)
        )

        df = df.merge(bmh_stats, on=["building_id", "meter", "hour"], how="left")
        df["bmh_target_mean"] = (
            df["bmh_target_mean"].fillna(df["bm_target_mean"]).astype(np.float32)
        )
        df["bmh_target_median"] = (
            df["bmh_target_median"].fillna(df["bm_target_median"]).astype(np.float32)
        )

        # Downcast float64 to float32
        for col in df.select_dtypes(include=["float64"]).columns:
            df[col] = df[col].astype(np.float32)

        return df

    print("Extracting features for train, val, and test split...")
    train_split = extract_features(train_split)
    val_split = extract_features(val_split)
    test_df = extract_features(test_df)

    # Determine feature column names
    exclude_cols = {"timestamp", "meter_reading", "target", "id", "row_id"}
    feature_cols = [c for c in train_split.columns if c not in exclude_cols]

    categorical_cols = [
        "building_id",
        "meter",
        "hour",
        "day",
        "dayofweek",
        "month",
        "is_weekend",
    ]
    cat_features = [c for c in categorical_cols if c in feature_cols]

    print(f"Features: {len(feature_cols)}, Categorical: {len(cat_features)}")

    # Cast categoricals to category dtype
    for col in cat_features:
        train_split[col] = train_split[col].astype("category")
        val_split[col] = val_split[col].astype("category")
        test_df[col] = test_df[col].astype("category")

    unique_meters = [0, 1, 2, 3]
    val_preds_meter = np.zeros(len(val_split), dtype=np.float32)
    test_preds_meter = np.zeros(len(test_df), dtype=np.float32)
    test_ids = test_df["id"].values

    # Model Hyperparameters
    model_params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "n_estimators": 1500,
        "learning_rate": 0.05,
        "num_leaves": 127,
        "max_depth": -1,
        "min_child_samples": 50,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }

    callbacks = [
        lgb.early_stopping(stopping_rounds=50, verbose=False),
        lgb.log_evaluation(period=200),
    ]

    for m in unique_meters:
        print(f"\n--- Training LightGBM Model for Meter {m} ---")
        train_mask = (train_split["meter"].astype(int) == m)
        val_mask = (val_split["meter"].astype(int) == m)
        test_mask = (test_df["meter"].astype(int) == m)

        X_tr = train_split.loc[train_mask, feature_cols]
        y_tr = train_split.loc[train_mask, "target"]
        X_va = val_split.loc[val_mask, feature_cols]
        y_va = val_split.loc[val_mask, "target"]

        if len(X_tr) == 0:
            continue

        model = lgb.LGBMRegressor(**model_params)
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=callbacks,
        )

        val_preds_log = model.predict(X_va)
        val_preds_meter[val_mask.values] = np.maximum(0.0, np.expm1(val_preds_log))

        if test_mask.any():
            X_te = test_df.loc[test_mask, feature_cols]
            test_preds_log = model.predict(X_te)
            test_preds_meter[test_mask.values] = np.maximum(0.0, np.expm1(test_preds_log))

        del model, X_tr, y_tr, X_va, y_va
        gc.collect()

    del train_split
    gc.collect()

    # Validation Evaluation
    print("\nEvaluating overall validation split...")
    val_true_meter = np.maximum(0.0, val_split["meter_reading"].values)

    val_rmsle = float(
        root_mean_squared_error(np.log1p(val_true_meter), np.log1p(val_preds_meter))
    )

    del val_split, test_df
    gc.collect()

    # Save Submission
    submission_path = "./submission/submission.csv"
    sub_df = pd.DataFrame(
        {
            "id": test_ids.astype(np.int32),
            "meter_reading": test_preds_meter.astype(np.float32),
        }
    )
    sub_df.to_csv(submission_path, index=False)
    print(f"Saved submission to {submission_path} with shape {sub_df.shape}")

    # Output final metric score as required
    print(f"Final Validation Score: {val_rmsle}")


if __name__ == "__main__":
    main()
