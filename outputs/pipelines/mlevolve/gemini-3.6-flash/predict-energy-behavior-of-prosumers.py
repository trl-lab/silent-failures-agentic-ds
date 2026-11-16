import json
import os
import warnings
import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings("ignore")


def load_raw_data(data_dir="./input"):
    """Load all relational tables from input directory with fallback format support."""

    def read_table(name):
        p_parquet = os.path.join(data_dir, f"{name}.parquet")
        p_csv = os.path.join(data_dir, f"{name}.csv")
        p_sub_csv = os.path.join(data_dir, name, f"{name}.csv")

        if os.path.exists(p_parquet):
            return pd.read_parquet(p_parquet)
        elif os.path.exists(p_csv):
            return pd.read_csv(p_csv)
        elif os.path.exists(p_sub_csv):
            return pd.read_csv(p_sub_csv)
        else:
            raise FileNotFoundError(f"Data file for '{name}' not found in {data_dir}")

    train = read_table("train")
    test = read_table("test")
    client = read_table("client")
    electricity = read_table("electricity_prices")
    forecast_w = read_table("forecast_weather")
    gas = read_table("gas_prices")
    station_map = read_table("weather_station_to_county_mapping")
    hist_w = read_table("historical_weather")

    return train, test, client, electricity, forecast_w, gas, station_map, hist_w


def process_weather_features(forecast_w, station_map):
    """Map weather station coordinates to counties and aggregate weather forecast features."""
    station_map_clean = (
        station_map.copy()
        .dropna(subset=["county"])
        .drop_duplicates(subset=["latitude", "longitude"])
    )
    station_map_clean["latitude"] = station_map_clean["latitude"].round(1)
    station_map_clean["longitude"] = station_map_clean["longitude"].round(1)

    fw = forecast_w.copy()
    fw["latitude"] = fw["latitude"].round(1)
    fw["longitude"] = fw["longitude"].round(1)
    fw["forecast_datetime"] = pd.to_datetime(fw["forecast_datetime"])

    fw = fw.merge(
        station_map_clean[["latitude", "longitude", "county"]],
        on=["latitude", "longitude"],
        how="left",
    )

    if (
        "10_metre_u_wind_component" in fw.columns
        and "10_metre_v_wind_component" in fw.columns
    ):
        fw["wind_speed"] = np.sqrt(
            fw["10_metre_u_wind_component"] ** 2 + fw["10_metre_v_wind_component"] ** 2
        )

    county_agg = {}
    for col in [
        "temperature",
        "dewpoint",
        "cloudcover_total",
        "direct_solar_radiation",
        "surface_solar_radiation",
        "wind_speed",
        "total_precipitation",
    ]:
        if col in fw.columns:
            county_agg[col] = "mean"

    fw_county = (
        fw.groupby(["county", "forecast_datetime"]).agg(county_agg).reset_index()
    )

    col_rename = {"forecast_datetime": "datetime"}
    for col in county_agg.keys():
        if col == "temperature":
            col_rename[col] = "fw_temp_mean"
        elif col == "dewpoint":
            col_rename[col] = "fw_dewpoint_mean"
        elif col == "cloudcover_total":
            col_rename[col] = "fw_cloud_total_mean"
        elif col == "direct_solar_radiation":
            col_rename[col] = "fw_solar_direct_mean"
        elif col == "surface_solar_radiation":
            col_rename[col] = "fw_solar_surface_mean"
        elif col == "wind_speed":
            col_rename[col] = "fw_wind_speed_mean"
        elif col == "total_precipitation":
            col_rename[col] = "fw_precip_mean"

    fw_county.rename(columns=col_rename, inplace=True)

    nat_agg = {}
    for col in ["temperature", "direct_solar_radiation", "cloudcover_total"]:
        if col in fw.columns:
            nat_agg[col] = "mean"

    fw_national = fw.groupby("forecast_datetime").agg(nat_agg).reset_index()

    nat_rename = {"forecast_datetime": "datetime"}
    if "temperature" in nat_agg:
        nat_rename["temperature"] = "fw_national_temp"
    if "direct_solar_radiation" in nat_agg:
        nat_rename["direct_solar_radiation"] = "fw_national_solar_direct"
    if "cloudcover_total" in nat_agg:
        nat_rename["cloudcover_total"] = "fw_national_cloud_total"

    fw_national.rename(columns=nat_rename, inplace=True)

    return fw_county, fw_national


def process_historical_weather_features(hist_w, station_map):
    """Map weather station coordinates to counties and aggregate historical weather features."""
    station_map_clean = (
        station_map.copy()
        .dropna(subset=["county"])
        .drop_duplicates(subset=["latitude", "longitude"])
    )
    station_map_clean["latitude"] = station_map_clean["latitude"].round(1)
    station_map_clean["longitude"] = station_map_clean["longitude"].round(1)

    hw = hist_w.copy()
    hw["latitude"] = hw["latitude"].round(1)
    hw["longitude"] = hw["longitude"].round(1)
    hw["datetime"] = pd.to_datetime(hw["datetime"])

    hw = hw.merge(
        station_map_clean[["latitude", "longitude", "county"]],
        on=["latitude", "longitude"],
        how="left",
    )

    agg_dict = {}
    if "temperature" in hw.columns:
        agg_dict["temperature"] = "mean"
    if "direct_solar_radiation" in hw.columns:
        agg_dict["direct_solar_radiation"] = "mean"
    elif "shortwave_radiation" in hw.columns:
        agg_dict["shortwave_radiation"] = "mean"
    if "windspeed_10m" in hw.columns:
        agg_dict["windspeed_10m"] = "mean"

    hw_county = hw.groupby(["county", "datetime"]).agg(agg_dict).reset_index()

    rename_map = {}
    if "temperature" in hw_county.columns:
        rename_map["temperature"] = "hist_temp_lag48"
    if "direct_solar_radiation" in hw_county.columns:
        rename_map["direct_solar_radiation"] = "hist_solar_lag48"
    elif "shortwave_radiation" in hw_county.columns:
        rename_map["shortwave_radiation"] = "hist_solar_lag48"
    if "windspeed_10m" in hw_county.columns:
        rename_map["windspeed_10m"] = "hist_wind_lag48"

    hw_county.rename(columns=rename_map, inplace=True)
    return hw_county


def compute_solar_elevation(datetime_series):
    """Calculate physical solar elevation angle proxy for solar energy generation potential."""
    lat_rad = np.radians(58.6)
    doy = datetime_series.dt.dayofyear
    hour = datetime_series.dt.hour

    declination = np.radians(23.45 * np.sin(np.radians(360.0 / 365.0 * (doy - 81))))
    hour_angle = np.radians(15.0 * (hour - 12.0))

    sin_elevation = np.sin(lat_rad) * np.sin(declination) + np.cos(lat_rad) * np.cos(
        declination
    ) * np.cos(hour_angle)
    solar_elevation = np.maximum(0.0, sin_elevation)
    return solar_elevation


def build_feature_pipeline():
    """Full data loading, feature extraction, temporal splitting, and artifact export."""
    os.makedirs("./working", exist_ok=True)
    train, test, client, electricity, forecast_w, gas, station_map, hist_w = load_raw_data()

    train["is_test"] = 0
    test["is_test"] = 1
    if "target" not in test.columns:
        test["target"] = np.nan

    if "id" not in test.columns:
        test["id"] = (
            test["row_id"] if "row_id" in test.columns else test.index.astype(int)
        )

    if "id" not in train.columns:
        train["id"] = -1

    full_df = pd.concat([train, test], ignore_index=True)
    full_df["datetime"] = pd.to_datetime(full_df["datetime"])

    if "prediction_unit_id" not in full_df.columns:
        full_df["prediction_unit_id"] = (
            full_df["county"].astype(str)
            + "_"
            + full_df["is_business"].astype(str)
            + "_"
            + full_df["product_type"].astype(str)
        )

    full_df["hour"] = full_df["datetime"].dt.hour.astype(np.int8)
    full_df["dayofweek"] = full_df["datetime"].dt.dayofweek.astype(np.int8)
    full_df["dayofyear"] = full_df["datetime"].dt.dayofyear.astype(np.int16)
    full_df["month"] = full_df["datetime"].dt.month.astype(np.int8)
    full_df["is_weekend"] = (full_df["dayofweek"] >= 5).astype(np.int8)

    full_df["sin_hour"] = np.sin(2 * np.pi * full_df["hour"] / 24.0).astype(np.float32)
    full_df["cos_hour"] = np.cos(2 * np.pi * full_df["hour"] / 24.0).astype(np.float32)
    full_df["sin_dayofyear"] = np.sin(2 * np.pi * full_df["dayofyear"] / 365.25).astype(
        np.float32
    )
    full_df["cos_dayofyear"] = np.cos(2 * np.pi * full_df["dayofyear"] / 365.25).astype(
        np.float32
    )
    full_df["solar_elevation"] = compute_solar_elevation(full_df["datetime"]).astype(
        np.float32
    )
    full_df["is_daylight"] = (full_df["solar_elevation"] > 0.01).astype(np.int8)

    client_clean = client.copy()
    client_clean["client_data_block_id"] = client_clean["data_block_id"] + 2
    client_cols = [
        "county",
        "is_business",
        "product_type",
        "client_data_block_id",
        "installed_capacity",
        "eic_count",
    ]
    client_clean = client_clean[client_cols].drop_duplicates(
        subset=["county", "is_business", "product_type", "client_data_block_id"]
    )

    full_df = full_df.merge(
        client_clean,
        left_on=[
            "county",
            "is_business",
            "product_type",
            "data_block_id",
        ],
        right_on=[
            "county",
            "is_business",
            "product_type",
            "client_data_block_id",
        ],
        how="left",
    ).drop(columns=["client_data_block_id"])

    full_df["installed_capacity"] = full_df["installed_capacity"].fillna(0.0)
    full_df["eic_count"] = full_df["eic_count"].fillna(0.0)

    elec = electricity.copy()
    elec["datetime"] = pd.to_datetime(elec["forecast_date"])
    elec_clean = elec[["datetime", "euros_per_mwh"]].drop_duplicates(
        subset=["datetime"]
    )
    elec_clean.rename(columns={"euros_per_mwh": "electricity_price"}, inplace=True)
    full_df = full_df.merge(elec_clean, on="datetime", how="left")
    full_df["electricity_price"] = full_df["electricity_price"].ffill().bfill()

    gas_clean = gas.copy()
    gas_clean["gas_data_block_id"] = gas_clean["data_block_id"] + 1
    gas_clean["gas_price_mean"] = (
        gas_clean["lowest_price_per_mwh"] + gas_clean["highest_price_per_mwh"]
    ) / 2.0
    gas_clean = gas_clean[["gas_data_block_id", "gas_price_mean"]].drop_duplicates(
        subset=["gas_data_block_id"]
    )

    full_df = full_df.merge(
        gas_clean,
        left_on="data_block_id",
        right_on="gas_data_block_id",
        how="left",
    ).drop(columns=["gas_data_block_id"])
    full_df["gas_price_mean"] = full_df["gas_price_mean"].ffill().bfill()

    fw_county, fw_national = process_weather_features(forecast_w, station_map)
    full_df = full_df.merge(fw_county, on=["county", "datetime"], how="left").merge(
        fw_national, on="datetime", how="left"
    )

    hw_county = process_historical_weather_features(hist_w, station_map)
    full_df["datetime_lag_48"] = full_df["datetime"] - pd.Timedelta(hours=48)
    full_df = full_df.merge(
        hw_county,
        left_on=["county", "datetime_lag_48"],
        right_on=["county", "datetime"],
        how="left",
        suffixes=("", "_hw"),
    ).drop(columns=["datetime_lag_48"])

    if "datetime_hw" in full_df.columns:
        full_df.drop(columns=["datetime_hw"], inplace=True)

    weather_cols_to_fill = [
        c
        for c in fw_county.columns.tolist() + fw_national.columns.tolist() + ["hist_temp_lag48", "hist_solar_lag48", "hist_wind_lag48"]
        if c not in ["county", "datetime"]
    ]
    for col in weather_cols_to_fill:
        if col in full_df.columns:
            full_df[col] = full_df.groupby("datetime")[col].transform(
                lambda x: x.ffill().bfill()
            )
            full_df[col] = full_df[col].ffill().bfill().fillna(0.0)

    full_df = full_df.sort_values(
        ["prediction_unit_id", "is_consumption", "datetime"]
    ).reset_index(drop=True)

    group_key = ["prediction_unit_id", "is_consumption"]

    full_df["target_lag_48"] = full_df.groupby(group_key)["target"].shift(48)
    full_df["target_lag_72"] = full_df.groupby(group_key)["target"].shift(72)
    full_df["target_lag_168"] = full_df.groupby(group_key)["target"].shift(168)
    full_df["target_lag_336"] = full_df.groupby(group_key)["target"].shift(336)

    full_df["target_lag_48_168_diff"] = full_df["target_lag_48"] - full_df["target_lag_168"]
    train_mask = full_df["is_test"] == 0
    county_hour_means = full_df[train_mask].groupby(["county", "is_consumption", "hour"])["target_lag_48"].mean().reset_index()
    county_hour_means.rename(columns={"target_lag_48": "target_lag_48_county_hour_mean"}, inplace=True)
    full_df = full_df.merge(county_hour_means, on=["county", "is_consumption", "hour"], how="left")

    full_df["target_roll_mean_120"] = full_df.groupby(group_key)[
        "target_lag_48"
    ].transform(lambda x: x.rolling(120, min_periods=12).mean())
    full_df["target_roll_std_120"] = full_df.groupby(group_key)[
        "target_lag_48"
    ].transform(lambda x: x.rolling(120, min_periods=12).std())
    full_df["target_roll_mean_168"] = full_df.groupby(group_key)[
        "target_lag_168"
    ].transform(lambda x: x.rolling(168, min_periods=24).mean())

    full_df["target_lag_48_cap_ratio"] = full_df["target_lag_48"] / (
        full_df["installed_capacity"] + 1e-3
    )

    float_cols = full_df.select_dtypes(include=["float64"]).columns
    full_df[float_cols] = full_df[float_cols].astype(np.float32)

    feature_cols = [
        "county",
        "is_business",
        "product_type",
        "is_consumption",
        "hour",
        "dayofweek",
        "dayofyear",
        "month",
        "is_weekend",
        "sin_hour",
        "cos_hour",
        "sin_dayofyear",
        "cos_dayofyear",
        "solar_elevation",
        "is_daylight",
        "installed_capacity",
        "eic_count",
        "electricity_price",
        "gas_price_mean",
        "fw_temp_mean",
        "fw_dewpoint_mean",
        "fw_cloud_total_mean",
        "fw_solar_direct_mean",
        "fw_solar_surface_mean",
        "fw_wind_speed_mean",
        "fw_precip_mean",
        "fw_national_temp",
        "fw_national_solar_direct",
        "fw_national_cloud_total",
        "hist_temp_lag48",
        "hist_solar_lag48",
        "hist_wind_lag48",
        "target_lag_48",
        "target_lag_72",
        "target_lag_168",
        "target_lag_336",
        "target_lag_48_168_diff",
        "target_lag_48_county_hour_mean",
        "target_roll_mean_120",
        "target_roll_std_120",
        "target_roll_mean_168",
        "target_lag_48_cap_ratio",
    ]
    feature_cols = [c for c in feature_cols if c in full_df.columns]

    train_data = full_df[(full_df["is_test"] == 0) & (full_df["target"].notna())].copy()
    test_data = full_df[full_df["is_test"] == 1].copy()

    max_train_block = train_data["data_block_id"].max()
    val_cutoff_block = max_train_block - 30

    tr_split = train_data[train_data["data_block_id"] <= val_cutoff_block].copy()
    val_split = train_data[train_data["data_block_id"] > val_cutoff_block].copy()

    tr_split.to_parquet("./working/train_processed.parquet", index=False)
    val_split.to_parquet("./working/val_processed.parquet", index=False)
    test_data.to_parquet("./working/test_processed.parquet", index=False)

    meta = {
        "feature_cols": feature_cols,
        "categorical_cols": [
            "county",
            "is_business",
            "product_type",
            "is_consumption",
        ],
        "target_col": "target",
        "id_col": "id",
        "val_cutoff_block": int(val_cutoff_block),
    }

    with open("./working/feature_meta.json", "w") as f:
        json.dump(meta, f, indent=4)

    return tr_split, val_split, test_data, meta


def get_lgb_model(is_consumption: int, seed: int = 42) -> lgb.LGBMRegressor:
    """Factory function for LightGBM models aligned with stream characteristics."""
    if is_consumption == 0:
        params = {
            "objective": "regression_l1",
            "metric": "mae",
            "boosting_type": "gbdt",
            "n_estimators": 2500,
            "learning_rate": 0.04,
            "num_leaves": 63,
            "max_depth": 8,
            "subsample": 0.8,
            "colsample_bytree": 0.7,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "min_child_samples": 20,
            "random_state": seed,
            "n_jobs": -1,
            "verbose": -1,
        }
    else:
        params = {
            "objective": "regression_l1",
            "metric": "mae",
            "boosting_type": "gbdt",
            "n_estimators": 3000,
            "learning_rate": 0.03,
            "num_leaves": 127,
            "max_depth": 10,
            "subsample": 0.85,
            "colsample_bytree": 0.75,
            "reg_alpha": 0.5,
            "reg_lambda": 2.0,
            "min_child_samples": 30,
            "random_state": seed,
            "n_jobs": -1,
            "verbose": -1,
        }
    return lgb.LGBMRegressor(**params)


def get_cat_model(is_consumption: int, seed: int = 42) -> cb.CatBoostRegressor:
    """Factory function for CatBoost models aligned with L1 objective."""
    if is_consumption == 0:
        params = {
            "loss_function": "MAE",
            "eval_metric": "MAE",
            "iterations": 1800,
            "learning_rate": 0.05,
            "depth": 7,
            "l2_leaf_reg": 3.0,
            "random_seed": seed,
            "verbose": 0,
            "thread_count": -1,
        }
    else:
        params = {
            "loss_function": "MAE",
            "eval_metric": "MAE",
            "iterations": 2000,
            "learning_rate": 0.04,
            "depth": 8,
            "l2_leaf_reg": 5.0,
            "random_seed": seed,
            "verbose": 0,
            "thread_count": -1,
        }
    return cb.CatBoostRegressor(**params)


def get_xgb_model(is_consumption: int, seed: int = 42) -> xgb.XGBRegressor:
    """Factory function for XGBoost models with MAE objective."""
    if is_consumption == 0:
        params = {
            "objective": "reg:absoluteerror",
            "eval_metric": "mae",
            "n_estimators": 2000,
            "learning_rate": 0.04,
            "max_depth": 7,
            "subsample": 0.8,
            "colsample_bytree": 0.7,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "random_state": seed,
            "n_jobs": -1,
            "tree_method": "hist",
            "enable_categorical": True,
            "early_stopping_rounds": 50,
        }
    else:
        params = {
            "objective": "reg:absoluteerror",
            "eval_metric": "mae",
            "n_estimators": 2500,
            "learning_rate": 0.03,
            "max_depth": 8,
            "subsample": 0.85,
            "colsample_bytree": 0.75,
            "reg_alpha": 0.5,
            "reg_lambda": 2.0,
            "random_state": seed,
            "n_jobs": -1,
            "tree_method": "hist",
            "enable_categorical": True,
            "early_stopping_rounds": 50,
        }
    return xgb.XGBRegressor(**params)


def post_process_predictions(preds: np.ndarray) -> np.ndarray:
    """Apply non-negative physical domain constraint and nan-filling."""
    preds = np.nan_to_num(preds, nan=0.0)
    return np.clip(preds, a_min=0.0, a_max=None)


def train_and_evaluate():
    """Execute dual-stream multi-seed ensemble training, validation, and test inference."""
    train_df, val_df, test_df, meta = build_feature_pipeline()

    feature_cols = meta["feature_cols"]
    cat_cols = meta["categorical_cols"]
    target_col = meta["target_col"]

    for c in cat_cols:
        if c in feature_cols:
            train_df[c] = train_df[c].astype("category")
            val_df[c] = val_df[c].astype("category")
            test_df[c] = test_df[c].astype("category")

    val_preds = np.zeros(len(val_df), dtype=np.float32)
    test_preds = np.zeros(len(test_df), dtype=np.float32)

    seeds = [42, 123, 2024]

    for is_cons in [0, 1]:
        stream_name = "Consumption" if is_cons == 1 else "Production"

        tr_mask = train_df["is_consumption"] == is_cons
        va_mask = val_df["is_consumption"] == is_cons
        te_mask = test_df["is_consumption"] == is_cons

        X_tr, y_tr = (
            train_df.loc[tr_mask, feature_cols],
            train_df.loc[tr_mask, target_col],
        )
        X_va, y_va = val_df.loc[va_mask, feature_cols], val_df.loc[va_mask, target_col]
        X_te = test_df.loc[te_mask, feature_cols]

        va_stream_preds = np.zeros(len(X_va), dtype=np.float32)
        te_stream_preds = np.zeros(len(X_te), dtype=np.float32)
        model_count = 0

        for s in seeds:
            lgb_m = get_lgb_model(is_consumption=is_cons, seed=s)
            lgb_m.fit(
                X_tr,
                y_tr,
                eval_set=[(X_va, y_va)],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=50, verbose=False),
                    lgb.log_evaluation(period=0),
                ],
            )

            p_va = lgb_m.predict(X_va)
            p_te = lgb_m.predict(X_te)

            va_stream_preds += post_process_predictions(p_va)
            te_stream_preds += post_process_predictions(p_te)
            model_count += 1

            xgb_m = get_xgb_model(is_consumption=is_cons, seed=s)
            xgb_m.fit(
                X_tr,
                y_tr,
                eval_set=[(X_va, y_va)],
                verbose=False,
            )

            p_va_xgb = xgb_m.predict(X_va)
            p_te_xgb = xgb_m.predict(X_te)

            va_stream_preds += post_process_predictions(p_va_xgb)
            te_stream_preds += post_process_predictions(p_te_xgb)
            model_count += 1

        cat_m = get_cat_model(is_consumption=is_cons, seed=42)
        X_tr_cb = X_tr.copy()
        X_va_cb = X_va.copy()
        X_te_cb = X_te.copy()

        for col in cat_cols:
            if col in X_tr_cb.columns:
                X_tr_cb[col] = X_tr_cb[col].astype(int)
                X_va_cb[col] = X_va_cb[col].astype(int)
                X_te_cb[col] = X_te_cb[col].astype(int)

        cat_m.fit(
            X_tr_cb,
            y_tr,
            eval_set=(X_va_cb, y_va),
            early_stopping_rounds=50,
            verbose=False,
        )

        p_va_cb = cat_m.predict(X_va_cb)
        p_te_cb = cat_m.predict(X_te_cb)

        va_stream_preds += post_process_predictions(p_va_cb)
        te_stream_preds += post_process_predictions(p_te_cb)
        model_count += 1

        va_stream_preds /= model_count
        te_stream_preds /= model_count

        stream_mae = mean_absolute_error(y_va, va_stream_preds)
        print(f"Validation MAE for {stream_name} Stream: {stream_mae:.4f}")

        val_preds[va_mask] = va_stream_preds
        test_preds[te_mask] = te_stream_preds

    final_val_mae = mean_absolute_error(val_df[target_col], val_preds)

    os.makedirs("./submission", exist_ok=True)
    submission_path = "./submission/submission.csv"

    sub_df = pd.DataFrame(
        {
            "id": test_df["id"].astype(int),
            "target": post_process_predictions(test_preds),
        }
    )

    sub_df.sort_values("id", inplace=True)
    sub_df.to_csv(submission_path, index=False)

    print(f"Final Validation Score: {final_val_mae}")


if __name__ == "__main__":
    train_and_evaluate()