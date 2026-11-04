import gc
import os
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings("ignore")

# 1. Load Data
data_dir = "./input"

if os.path.exists(os.path.join(data_dir, "train.parquet")):
    train = pd.read_parquet(os.path.join(data_dir, "train.parquet"))
else:
    train = pd.read_csv(os.path.join(data_dir, "train/train.csv"))

if os.path.exists(os.path.join(data_dir, "test.parquet")):
    test = pd.read_parquet(os.path.join(data_dir, "test.parquet"))
else:
    test = pd.read_csv(os.path.join(data_dir, "test/test.csv"))

client = pd.read_csv(os.path.join(data_dir, "client.csv"))
electricity_prices = pd.read_csv(os.path.join(data_dir, "electricity_prices.csv"))
gas_prices = pd.read_csv(os.path.join(data_dir, "gas_prices.csv"))
forecast_weather = pd.read_csv(os.path.join(data_dir, "forecast_weather.csv"))
weather_mapping = pd.read_csv(
    os.path.join(data_dir, "weather_station_to_county_mapping.csv")
)

train["is_train"] = 1
test["is_train"] = 0
if "target" not in test.columns:
    test["target"] = np.nan

df = pd.concat([train, test], ignore_index=True)
df["datetime"] = pd.to_datetime(df["datetime"])
df = df.sort_values(["prediction_unit_id", "is_consumption", "datetime"]).reset_index(
    drop=True
)

# 2. Feature Engineering: Target Lags and Differences
df["target_lag_48"] = df.groupby(["prediction_unit_id", "is_consumption"])[
    "target"
].shift(48)
df["target_lag_72"] = df.groupby(["prediction_unit_id", "is_consumption"])[
    "target"
].shift(72)
df["target_lag_168"] = df.groupby(["prediction_unit_id", "is_consumption"])[
    "target"
].shift(168)
df["target_lag_336"] = df.groupby(["prediction_unit_id", "is_consumption"])[
    "target"
].shift(336)

df["target_diff_48_72"] = df["target_lag_48"] - df["target_lag_72"]
df["target_diff_48_168"] = df["target_lag_48"] - df["target_lag_168"]

df["target_roll_mean_24"] = df.groupby(["prediction_unit_id", "is_consumption"])[
    "target_lag_48"
].transform(lambda x: x.rolling(24, min_periods=1).mean())
df["target_roll_mean_168"] = df.groupby(["prediction_unit_id", "is_consumption"])[
    "target_lag_48"
].transform(lambda x: x.rolling(168, min_periods=1).mean())

# 3. Datetime Features
df["hour"] = df["datetime"].dt.hour
df["day"] = df["datetime"].dt.day
df["dayofweek"] = df["datetime"].dt.dayofweek
df["month"] = df["datetime"].dt.month
df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
df["sin_hour"] = np.sin(2 * np.pi * df["hour"] / 24)
df["cos_hour"] = np.cos(2 * np.pi * df["hour"] / 24)

# 4. Auxiliary Data Merges
client_cols = [
    "county",
    "is_business",
    "product_type",
    "data_block_id",
    "eic_count",
    "installed_capacity",
]
client_sub = client[client_cols].drop_duplicates()
df = df.merge(
    client_sub,
    on=["county", "is_business", "product_type", "data_block_id"],
    how="left",
)

electricity_prices["forecast_date"] = pd.to_datetime(
    electricity_prices["forecast_date"]
)
electricity_prices["hour"] = electricity_prices["forecast_date"].dt.hour
elec_agg = (
    electricity_prices.groupby(["data_block_id", "hour"])["euros_per_mwh"]
    .mean()
    .reset_index()
)
df = df.merge(elec_agg, on=["data_block_id", "hour"], how="left")

gas_prices["mean_gas_price"] = (
    gas_prices["lowest_price_per_mwh"] + gas_prices["highest_price_per_mwh"]
) / 2
gas_agg = gas_prices.groupby("data_block_id")["mean_gas_price"].mean().reset_index()
df = df.merge(gas_agg, on="data_block_id", how="left")

# Spatial Weather Aggregation by County
forecast_weather["forecast_datetime"] = pd.to_datetime(
    forecast_weather["forecast_datetime"]
)
forecast_weather["wind_speed"] = np.sqrt(
    forecast_weather["10_metre_u_wind_component"] ** 2
    + forecast_weather["10_metre_v_wind_component"] ** 2
)
forecast_weather["latitude"] = forecast_weather["latitude"].round(2)
forecast_weather["longitude"] = forecast_weather["longitude"].round(2)

weather_mapping_clean = weather_mapping.dropna(subset=["county"]).copy()
weather_mapping_clean["county"] = weather_mapping_clean["county"].astype(int)
weather_mapping_clean["latitude"] = weather_mapping_clean["latitude"].round(2)
weather_mapping_clean["longitude"] = weather_mapping_clean["longitude"].round(2)
weather_mapping_sub = weather_mapping_clean[
    ["latitude", "longitude", "county"]
].drop_duplicates()

fw_mapped = forecast_weather.merge(
    weather_mapping_sub, on=["latitude", "longitude"], how="inner"
)

fw_county_agg = (
    fw_mapped.groupby(["forecast_datetime", "county"])[
        [
            "temperature",
            "dewpoint",
            "cloudcover_total",
            "direct_solar_radiation",
            "wind_speed",
        ]
    ]
    .mean()
    .reset_index()
)

fw_national_agg = (
    forecast_weather.groupby("forecast_datetime")[
        [
            "temperature",
            "dewpoint",
            "cloudcover_total",
            "direct_solar_radiation",
            "wind_speed",
        ]
    ]
    .mean()
    .reset_index()
)

df = df.merge(
    fw_county_agg,
    left_on=["datetime", "county"],
    right_on=["forecast_datetime", "county"],
    how="left",
)
if "forecast_datetime" in df.columns:
    df.drop(columns=["forecast_datetime"], inplace=True)

df = df.merge(
    fw_national_agg,
    left_on="datetime",
    right_on="forecast_datetime",
    how="left",
    suffixes=("", "_nat"),
)
if "forecast_datetime" in df.columns:
    df.drop(columns=["forecast_datetime"], inplace=True)

weather_cols = [
    "temperature",
    "dewpoint",
    "cloudcover_total",
    "direct_solar_radiation",
    "wind_speed",
]
for col in weather_cols:
    if f"{col}_nat" in df.columns:
        df[col] = df[col].fillna(df[f"{col}_nat"])
        df.drop(columns=[f"{col}_nat"], inplace=True)

# Domain Interactions
df["solar_capacity"] = df["installed_capacity"] * df["direct_solar_radiation"]
df["temp_capacity"] = df["installed_capacity"] * df["temperature"]

# 5. Split back to Train and Test Sets
train_df = df[df["is_train"] == 1].sort_values("datetime").reset_index(drop=True)
test_df = df[df["is_train"] == 0].sort_values("datetime").reset_index(drop=True)

features = [
    "county",
    "is_business",
    "product_type",
    "prediction_unit_id",
    "target_lag_48",
    "target_lag_72",
    "target_lag_168",
    "target_lag_336",
    "target_diff_48_72",
    "target_diff_48_168",
    "target_roll_mean_24",
    "target_roll_mean_168",
    "hour",
    "day",
    "dayofweek",
    "month",
    "is_weekend",
    "sin_hour",
    "cos_hour",
    "eic_count",
    "installed_capacity",
    "euros_per_mwh",
    "mean_gas_price",
    "temperature",
    "dewpoint",
    "cloudcover_total",
    "direct_solar_radiation",
    "wind_speed",
    "solar_capacity",
    "temp_capacity",
]

X_train = train_df[features]
y_train = train_df["target"]
X_test = test_df[features]

# 6. Group Block Cross-Validation & Decoupled Model Training
unique_blocks = train_df["data_block_id"].unique()
np.random.seed(42)
block_folds = np.array_split(unique_blocks, 5)

oof_preds = np.zeros(len(train_df))
test_preds = np.zeros(len(test_df))

for cons_type in [0, 1]:
    tr_mask_c = train_df["is_consumption"] == cons_type
    te_mask_c = test_df["is_consumption"] == cons_type

    for fold, val_blocks in enumerate(block_folds):
        val_idx = tr_mask_c & train_df["data_block_id"].isin(val_blocks)
        tr_idx = tr_mask_c & (~train_df["data_block_id"].isin(val_blocks))

        X_tr, y_tr = X_train[tr_idx], y_train[tr_idx]
        X_va, y_va = X_train[val_idx], y_train[val_idx]

        model = xgb.XGBRegressor(
            objective="reg:absoluteerror",
            n_estimators=500,
            learning_rate=0.08,
            max_depth=7,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            random_state=42 + fold + cons_type * 100,
            n_jobs=-1,
        )

        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

        oof_preds[val_idx] = model.predict(X_va)
        if te_mask_c.sum() > 0:
            test_preds[te_mask_c] += model.predict(X_test[te_mask_c]) / 5

overall_mae = mean_absolute_error(y_train, oof_preds)
print(f"Overall CV MAE: {overall_mae:.4f}")

# 7. Generate Submission
os.makedirs("./working", exist_ok=True)
sub_ids = test_df["id"] if "id" in test_df.columns else np.arange(len(test_df))
submission = pd.DataFrame({"id": sub_ids, "target": np.clip(test_preds, 0, None)})
submission.to_csv("./working/submission.csv", index=False)
