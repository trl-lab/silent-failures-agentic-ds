import gc
import json
import os
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder


def reduce_mem_usage(df, verbose=False):
    """Iterate through all columns of a dataframe and modify data types to reduce memory usage."""
    numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    return df


def calculate_relative_humidity(air_temp, dew_temp):
    """Calculate relative humidity (%) using Magnus-Tetens approximation."""
    a = 17.625
    b = 243.04
    alpha_dew = (a * dew_temp) / (b + dew_temp)
    alpha_air = (a * air_temp) / (b + air_temp)
    rh = 100.0 * np.exp(alpha_dew - alpha_air)
    return np.clip(rh, 0.0, 100.0)


def calculate_apparent_temperature(air_temp, dew_temp, wind_speed):
    """Calculate apparent temperature accounting for humidity and wind cooling."""
    rh = calculate_relative_humidity(air_temp, dew_temp)
    e = (rh / 100.0) * 6.105 * np.exp((17.27 * air_temp) / (237.7 + air_temp))
    wind = np.where(pd.isna(wind_speed), 0.0, wind_speed)
    apparent_temp = air_temp + 0.33 * e - 0.70 * wind - 4.00
    return apparent_temp


def load_and_merge_metadata(df, is_train=True):
    """Merge auxiliary building and weather metadata if available and not already present."""
    input_dir = "./input"
    orig_dir = "./input/original"

    if "square_feet" not in df.columns:
        bldg_path = os.path.join(input_dir, "building_metadata.parquet")
        if not os.path.exists(bldg_path):
            bldg_path = os.path.join(input_dir, "building_metadata.csv")
            if not os.path.exists(bldg_path):
                bldg_path = os.path.join(orig_dir, "building_metadata.csv")

        if os.path.exists(bldg_path):
            bldg_df = (
                pd.read_parquet(bldg_path)
                if bldg_path.endswith(".parquet")
                else pd.read_csv(bldg_path)
            )
            bldg_df = reduce_mem_usage(bldg_df)
            df = df.merge(bldg_df, on="building_id", how="left")
            del bldg_df
            gc.collect()

    if "air_temperature" not in df.columns:
        weather_file = f"weather_{'train' if is_train else 'test'}"
        w_path = os.path.join(input_dir, f"{weather_file}.parquet")
        if not os.path.exists(w_path):
            w_path = os.path.join(input_dir, f"{weather_file}.csv")
            if not os.path.exists(w_path):
                w_path = os.path.join(orig_dir, f"{weather_file}.csv")

        if os.path.exists(w_path):
            w_df = (
                pd.read_parquet(w_path)
                if w_path.endswith(".parquet")
                else pd.read_csv(w_path)
            )
            w_df = reduce_mem_usage(w_df)
            if "site_id" in df.columns and "timestamp" in df.columns:
                df["timestamp"] = df["timestamp"].astype(str)
                w_df["timestamp"] = w_df["timestamp"].astype(str)
                df = df.merge(w_df, on=["site_id", "timestamp"], how="left")
            del w_df
            gc.collect()

    return df


SITE_TIMEZONE_OFFSETS = {
    0: -5,
    1: 0,
    2: -7,
    3: -5,
    4: -8,
    5: 0,
    6: -5,
    7: -5,
    8: -5,
    9: -6,
    10: -7,
    11: -5,
    12: 0,
    13: -6,
    14: -5,
    15: -5,
}


def engineer_features(df):
    """Apply feature engineering: temporal, local solar timezone alignment, cyclical, domain physical, and structural interactions."""
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"] = df["timestamp"].dt.hour.astype(np.int8)
    df["day"] = df["timestamp"].dt.day.astype(np.int8)
    df["dayofweek"] = df["timestamp"].dt.dayofweek.astype(np.int8)
    df["month"] = df["timestamp"].dt.month.astype(np.int8)
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(np.int8)

    # Local solar timezone alignment
    if "site_id" in df.columns:
        offsets = df["site_id"].map(SITE_TIMEZONE_OFFSETS).fillna(0).astype(int)
        local_ts = df["timestamp"] + pd.to_timedelta(offsets, unit="h")
        df["local_hour"] = local_ts.dt.hour.astype(np.int8)
        df["local_dayofweek"] = local_ts.dt.dayofweek.astype(np.int8)
        df["local_is_weekend"] = (df["local_dayofweek"] >= 5).astype(np.int8)
        df["local_hour_sin"] = np.sin(2 * np.pi * df["local_hour"] / 24.0).astype(
            np.float32
        )
        df["local_hour_cos"] = np.cos(2 * np.pi * df["local_hour"] / 24.0).astype(
            np.float32
        )
        del offsets, local_ts

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0).astype(np.float32)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0).astype(np.float32)
    df["month_sin"] = np.sin(2 * np.pi * (df["month"] - 1) / 12.0).astype(np.float32)
    df["month_cos"] = np.cos(2 * np.pi * (df["month"] - 1) / 12.0).astype(np.float32)

    if "square_feet" in df.columns:
        df["log_square_feet"] = np.log1p(df["square_feet"]).astype(np.float32)

    if "year_built" in df.columns:
        df["building_age"] = (2019 - df["year_built"]).astype(np.float32)
        df["building_age"] = df["building_age"].fillna(-1).astype(np.float32)

    if "floor_count" in df.columns and "square_feet" in df.columns:
        df["square_feet_per_floor"] = (
            df["square_feet"] / df["floor_count"].replace(0, np.nan)
        ).astype(np.float32)
        df["square_feet_per_floor"] = (
            df["square_feet_per_floor"].fillna(-1).astype(np.float32)
        )

    if "air_temperature" in df.columns and "dew_temperature" in df.columns:
        df["temp_spread"] = (df["air_temperature"] - df["dew_temperature"]).astype(
            np.float32
        )
        df["relative_humidity"] = calculate_relative_humidity(
            df["air_temperature"].values, df["dew_temperature"].values
        ).astype(np.float32)

        wind_col = (
            df["wind_speed"].values if "wind_speed" in df.columns else np.zeros(len(df))
        )
        df["apparent_temperature"] = calculate_apparent_temperature(
            df["air_temperature"].values, df["dew_temperature"].values, wind_col
        ).astype(np.float32)

        df["hdd"] = np.maximum(0.0, 18.0 - df["air_temperature"]).astype(np.float32)
        df["cdd"] = np.maximum(0.0, df["air_temperature"] - 18.0).astype(np.float32)
        if "log_square_feet" in df.columns:
            df["hdd_x_log_sqft"] = (df["hdd"] * df["log_square_feet"]).astype(
                np.float32
            )
            df["cdd_x_log_sqft"] = (df["cdd"] * df["log_square_feet"]).astype(
                np.float32
            )

    return df


def create_weather_rolling_features(df):
    """Compute multi-scale rolling thermal inertia and weather lag features per site_id chronologically."""
    if "site_id" in df.columns and "air_temperature" in df.columns:
        site_weather = df[["site_id", "timestamp", "air_temperature"]].drop_duplicates()
        site_weather = site_weather.sort_values(["site_id", "timestamp"]).reset_index(
            drop=True
        )

        site_weather["air_temp_roll_mean_3h"] = (
            site_weather.groupby("site_id")["air_temperature"]
            .transform(lambda x: x.rolling(window=3, min_periods=1).mean())
            .astype(np.float32)
        )

        site_weather["air_temp_roll_mean_12h"] = (
            site_weather.groupby("site_id")["air_temperature"]
            .transform(lambda x: x.rolling(window=12, min_periods=1).mean())
            .astype(np.float32)
        )

        site_weather["air_temp_roll_mean_24h"] = (
            site_weather.groupby("site_id")["air_temperature"]
            .transform(lambda x: x.rolling(window=24, min_periods=1).mean())
            .astype(np.float32)
        )

        site_weather["air_temp_roll_mean_48h"] = (
            site_weather.groupby("site_id")["air_temperature"]
            .transform(lambda x: x.rolling(window=48, min_periods=1).mean())
            .astype(np.float32)
        )

        site_weather["air_temp_roll_std_24h"] = (
            site_weather.groupby("site_id")["air_temperature"]
            .transform(lambda x: x.rolling(window=24, min_periods=1).std())
            .fillna(0)
            .astype(np.float32)
        )

        site_weather["air_temp_roll_std_48h"] = (
            site_weather.groupby("site_id")["air_temperature"]
            .transform(lambda x: x.rolling(window=48, min_periods=1).std())
            .fillna(0)
            .astype(np.float32)
        )

        site_weather["air_temp_diff_3h"] = (
            site_weather.groupby("site_id")["air_temperature"]
            .transform(lambda x: x.diff(3))
            .fillna(0)
            .astype(np.float32)
        )

        site_weather["air_temp_diff_12h"] = (
            site_weather.groupby("site_id")["air_temperature"]
            .transform(lambda x: x.diff(12))
            .fillna(0)
            .astype(np.float32)
        )

        df = df.merge(
            site_weather[
                [
                    "site_id",
                    "timestamp",
                    "air_temp_roll_mean_3h",
                    "air_temp_roll_mean_12h",
                    "air_temp_roll_mean_24h",
                    "air_temp_roll_mean_48h",
                    "air_temp_roll_std_24h",
                    "air_temp_roll_std_48h",
                    "air_temp_diff_3h",
                    "air_temp_diff_12h",
                ]
            ],
            on=["site_id", "timestamp"],
            how="left",
        )
        del site_weather
        gc.collect()
    return df


def compute_rmsle(y_true, y_pred):
    """Compute Root Mean Squared Logarithmic Error (RMSLE)."""
    y_t = np.maximum(0.0, np.array(y_true, dtype=np.float64))
    y_p = np.maximum(0.0, np.array(y_pred, dtype=np.float64))
    return float(np.sqrt(np.mean((np.log1p(y_p) - np.log1p(y_t)) ** 2)))


def get_meter_model_params(meter_type: int = 0) -> dict:
    """Return tuned LightGBM hyperparameters for each utility meter modality."""
    base_params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 128,
        "max_depth": -1,
        "min_child_samples": 50,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42 + meter_type,
        "n_jobs": -1,
        "verbose": -1,
    }

    if meter_type == 0:  # Electricity
        base_params.update(
            {
                "num_leaves": 256,
                "learning_rate": 0.05,
                "min_child_samples": 64,
                "colsample_bytree": 0.75,
            }
        )
    elif meter_type == 1:  # Chilled Water
        base_params.update(
            {
                "num_leaves": 128,
                "learning_rate": 0.04,
                "min_child_samples": 40,
                "reg_lambda": 2.0,
            }
        )
    elif meter_type == 2:  # Steam
        base_params.update(
            {
                "num_leaves": 96,
                "learning_rate": 0.03,
                "min_child_samples": 30,
                "reg_alpha": 0.5,
                "reg_lambda": 5.0,
            }
        )
    elif meter_type == 3:  # Hot Water
        base_params.update(
            {
                "num_leaves": 96,
                "learning_rate": 0.04,
                "min_child_samples": 30,
                "reg_lambda": 2.0,
            }
        )

    return base_params


def main():
    os.makedirs("./working", exist_ok=True)
    os.makedirs("./submission", exist_ok=True)

    print("Loading raw training and test datasets...")
    train_df = pd.read_parquet("./input/train.parquet")
    test_df = pd.read_parquet("./input/test.parquet")

    train_df = reduce_mem_usage(train_df)
    test_df = reduce_mem_usage(test_df)

    # Merge auxiliary metadata if available
    train_df = load_and_merge_metadata(train_df, is_train=True)
    test_df = load_and_merge_metadata(test_df, is_train=False)

    # Anomaly cleaning and target transformation
    if "site_id" in train_df.columns and "meter" in train_df.columns:
        site0_filter = (
            (train_df["site_id"] == 0)
            & (train_df["meter"] == 0)
            & (pd.to_datetime(train_df["timestamp"]) < "2016-05-21")
        )
        train_df = train_df[~site0_filter].reset_index(drop=True)

    train_df["target_log"] = np.log1p(
        np.maximum(0, train_df["meter_reading"].values)
    ).astype(np.float32)

    # Feature Engineering
    train_df = engineer_features(train_df)
    test_df = engineer_features(test_df)

    train_df = create_weather_rolling_features(train_df)
    test_df = create_weather_rolling_features(test_df)

    # Categorical encoding fitted strictly on training data
    if "primary_use" in train_df.columns:
        le = LabelEncoder()
        train_df["primary_use"] = train_df["primary_use"].astype(str)
        test_df["primary_use"] = test_df["primary_use"].astype(str)
        le.fit(train_df["primary_use"])

        train_df["primary_use"] = le.transform(train_df["primary_use"]).astype(np.int8)
        test_df["primary_use"] = (
            test_df["primary_use"]
            .map(lambda s: le.transform([s])[0] if s in le.classes_ else -1)
            .astype(np.int8)
        )

    # Preserve test ID
    if "row_id" in test_df.columns and "id" not in test_df.columns:
        test_df["id"] = test_df["row_id"].astype(np.int32)
    elif "id" not in test_df.columns:
        test_df["id"] = np.arange(len(test_df), dtype=np.int32)

    drop_cols = ["meter_reading", "target_log", "timestamp", "row_id", "id"]
    feature_cols = [c for c in train_df.columns if c not in drop_cols]

    CATEGORICAL_FEATURES = [
        "building_id",
        "site_id",
        "primary_use",
        "meter",
        "hour",
        "dayofweek",
        "month",
        "is_weekend",
        "local_hour",
        "local_dayofweek",
        "local_is_weekend",
    ]
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in feature_cols]

    train_df = reduce_mem_usage(train_df)
    test_df = reduce_mem_usage(test_df)

    n_splits = 3
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    oof_predictions = np.zeros(len(train_df), dtype=np.float32)
    test_predictions = np.zeros(len(test_df), dtype=np.float32)
    meter_types = sorted(train_df["meter"].unique())

    # Decoupled Meter-Specific Training Loop with K-Fold Ensemble
    for m in meter_types:
        m_train_mask = train_df["meter"] == m
        m_train_indices = np.where(m_train_mask)[0]
        df_m = train_df.iloc[m_train_indices].reset_index(drop=True)

        test_m_mask = test_df["meter"] == m
        test_m_indices = np.where(test_m_mask)[0]
        has_test = len(test_m_indices) > 0
        if has_test:
            df_test_m = test_df.iloc[test_m_indices].reset_index(drop=True)
            test_m_log_preds = np.zeros(len(df_test_m), dtype=np.float32)

        params = get_meter_model_params(int(m))
        print(f"--- Training Meter {m} ({len(df_m)} samples, {n_splits} folds) ---")

        for fold, (trn_idx, val_idx) in enumerate(kf.split(df_m)):
            trn_fold = df_m.iloc[trn_idx]
            val_fold = df_m.iloc[val_idx]

            # Compute building baseline statistics strictly on fold training data
            bldg_stats = (
                trn_fold.groupby("building_id")["target_log"]
                .agg(["mean", "median", "std"])
                .reset_index()
                .rename(
                    columns={
                        "mean": "bldg_mean",
                        "median": "bldg_median",
                        "std": "bldg_std",
                    }
                )
            )
            bldg_stats["bldg_std"] = bldg_stats["bldg_std"].fillna(0).astype(np.float32)
            bldg_stats["bldg_mean"] = bldg_stats["bldg_mean"].astype(np.float32)
            bldg_stats["bldg_median"] = bldg_stats["bldg_median"].astype(np.float32)

            global_mean = float(trn_fold["target_log"].mean())
            global_median = float(trn_fold["target_log"].median())
            global_std = (
                float(trn_fold["target_log"].std())
                if not np.isnan(trn_fold["target_log"].std())
                else 0.0
            )

            # Attach stats to train and validation sets
            X_tr = trn_fold[feature_cols].merge(
                bldg_stats, on="building_id", how="left"
            )
            X_tr["bldg_mean"] = (
                X_tr["bldg_mean"].fillna(global_mean).astype(np.float32)
            )
            X_tr["bldg_median"] = (
                X_tr["bldg_median"].fillna(global_median).astype(np.float32)
            )
            X_tr["bldg_std"] = (
                X_tr["bldg_std"].fillna(global_std).astype(np.float32)
            )
            y_tr = trn_fold["target_log"].values

            X_va = val_fold[feature_cols].merge(
                bldg_stats, on="building_id", how="left"
            )
            X_va["bldg_mean"] = (
                X_va["bldg_mean"].fillna(global_mean).astype(np.float32)
            )
            X_va["bldg_median"] = (
                X_va["bldg_median"].fillna(global_median).astype(np.float32)
            )
            X_va["bldg_std"] = (
                X_va["bldg_std"].fillna(global_std).astype(np.float32)
            )
            y_va = val_fold["target_log"].values
            y_va_raw = val_fold["meter_reading"].values

            curr_cat_cols = [c for c in cat_cols if c in X_tr.columns]

            trn_data = lgb.Dataset(
                X_tr,
                label=y_tr,
                categorical_feature=curr_cat_cols,
                free_raw_data=False,
            )
            val_data = lgb.Dataset(
                X_va,
                label=y_va,
                reference=trn_data,
                categorical_feature=curr_cat_cols,
                free_raw_data=False,
            )

            callbacks = [
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=0),
            ]

            booster = lgb.train(
                params,
                trn_data,
                num_boost_round=2000,
                valid_sets=[trn_data, val_data],
                valid_names=["train", "valid"],
                callbacks=callbacks,
            )

            preds_val_log = booster.predict(
                X_va, num_iteration=booster.best_iteration
            )
            preds_val_real = np.maximum(0.0, np.expm1(np.maximum(0.0, preds_val_log)))
            oof_predictions[m_train_indices[val_idx]] = preds_val_real

            fold_rmsle = compute_rmsle(y_va_raw, preds_val_real)
            print(
                f"Meter {m} | Fold {fold + 1}/{n_splits} | Best Iter: {booster.best_iteration} | Val RMSLE: {fold_rmsle:.5f}"
            )

            if has_test:
                X_te = df_test_m[feature_cols].merge(
                    bldg_stats, on="building_id", how="left"
                )
                X_te["bldg_mean"] = (
                    X_te["bldg_mean"].fillna(global_mean).astype(np.float32)
                )
                X_te["bldg_median"] = (
                    X_te["bldg_median"].fillna(global_median).astype(np.float32)
                )
                X_te["bldg_std"] = (
                    X_te["bldg_std"].fillna(global_std).astype(np.float32)
                )

                preds_test_fold_log = booster.predict(
                    X_te, num_iteration=booster.best_iteration
                )
                test_m_log_preds += (preds_test_fold_log / n_splits).astype(
                    np.float32
                )
                del X_te

            del trn_data, val_data, X_tr, y_tr, X_va, y_va, bldg_stats
            gc.collect()

        if has_test:
            preds_test_real = np.maximum(
                0.0, np.expm1(np.maximum(0.0, test_m_log_preds))
            )
            test_predictions[test_m_indices] = preds_test_real.astype(np.float32)
            del df_test_m

        meter_oof_score = compute_rmsle(
            df_m["meter_reading"].values, oof_predictions[m_train_indices]
        )
        print(f"--> Meter {m} Full OOF RMSLE: {meter_oof_score:.5f}")

        del df_m
        gc.collect()

    # Overall out-of-fold validation score across all meter types
    final_val_score = compute_rmsle(train_df["meter_reading"].values, oof_predictions)

    # Save submission file
    sub_df = pd.DataFrame(
        {
            "id": test_df["id"].values.astype(np.int32),
            "meter_reading": np.round(test_predictions, 4),
        }
    )
    sub_df.to_csv("./submission/submission.csv", index=False)

    print(f"Final Validation Score: {final_val_score}")


if __name__ == "__main__":
    main()
