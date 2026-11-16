import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error

# Load data
train_df = pd.read_parquet("./input/train/train.csv")
test_df = pd.read_parquet("./input/test/test.csv")
client_df = pd.read_csv("./input/client.csv")
weather_df = pd.read_parquet("./input/forecast_weather.parquet")
historical_weather = pd.read_parquet("./input/historical_weather.parquet")
weather_station_mapping = pd.read_csv("./input/weather_station_to_county_mapping.csv")
electricity_prices = pd.read_csv("./input/electricity_prices.csv")
gas_prices = pd.read_csv("./input/gas_prices.csv")

# Create a simple feature engineering approach
# Extract datetime features
train_df["datetime"] = pd.to_datetime(train_df["datetime"])
test_df["datetime"] = pd.to_datetime(test_df["datetime"])

# Join with reference data
# Join with client data
train_df = train_df.merge(
    client_df[["county", "product_type", "is_business", "date"]],
    on=["county", "product_type", "is_business", "date"],
    how="left",
)

# Create validation split
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Train and evaluate
best_model = None
best_val_score = float("inf")

for fold, (train_idx, val_idx) in enumerate(kf.split(train_df)):
    fold_train = train_df.iloc[train_idx]
    fold_val = train_df.iloc[val_idx]

    # Feature engineering
    # Create features from weather data (simplified)
    fold_train["hour"] = fold_train["datetime"].dt.hour
    fold_train["day_of_week"] = fold_train["datetime"].dt.dayofweek
    fold_train["month"] = fold_train["datetime"].dt.month

    # Create LightGBM model
    train_data = lgb.Dataset(fold_train.drop("target", axis=1), fold_train["target"])
    val_data = lgb.Dataset(
        fold_val.drop("target", axis=1), fold_val["target"], reference=train_data
    )

    # Train model
    params = {
        "objective": "regression",
        "metric": "mae",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "seed": 42,
    }

    model = lgb.train(
        params,
        train_data,
        num_boost_round=100,
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
    )

    # Evaluate
    val_pred = model.predict(
        fold_val.drop("target", axis=1), num_iteration=model.best_iteration
    )
    val_score = mean_absolute_error(fold_val["target"], val_pred)

    print(f"Fold {fold+1}: MAE = {val_score:.4f}")

    if val_score < best_val_score:
        best_val_score = val_score
        best_model = model

# Train final model on all data
final_train = train_df.dropna(subset=["target"])
final_train["hour"] = final_train["datetime"].dt.hour
final_train["day_of_week"] = final_train["datetime"].dt.dayofweek
final_train["month"] = final_train["datetime"].dt.month

final_data = lgb.Dataset(final_train.drop("target", axis=1), final_train["target"])
final_model = lgb.train(
    params,
    final_data,
    num_boost_round=500,
    callbacks=[lgb.early_stopping(100), lgb.log_evaluation(-1)],
)

# Predict on test data
test_df["datetime"] = pd.to_datetime(test_df["datetime"])
test_df["hour"] = test_df["datetime"].dt.hour
test_df["day_of_week"] = test_df["datetime"].dt.dayofweek
test_df["month"] = test_df["datetime"].dt.month

test_data = lgb.Dataset(test_df.drop("target", axis=1), reference=final_data)
test_pred = final_model.predict(test_data, num_iteration=final_model.best_iteration)

# Create submission
submission = pd.DataFrame({"id": test_df["id"], "target": test_pred})
submission.to_csv("./working/submission.csv", index=False)


import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error

# Load data
train_df = pd.read_parquet("./input/train/train.csv")
test_df = pd.read_parquet("./input/test/test.csv")
client_df = pd.read_csv("./input/client.csv")
weather_df = pd.read_parquet("./input/forecast_weather.parquet")
historical_weather = pd.read_parquet("./input/historical_weather.parquet")
weather_station_mapping = pd.read_csv("./input/weather_station_to_county_mapping.csv")
electricity_prices = pd.read_csv("./input/electricity_prices.csv")
gas_prices = pd.read_csv("./input/gas_prices.csv")

# Create datetime features
train_df["datetime"] = pd.to_datetime(train_df["datetime"])
test_df["datetime"] = pd.to_datetime(test_df["datetime"])
train_df["hour"] = train_df["datetime"].dt.hour
train_df["day_of_week"] = train_df["datetime"].dt.dayofweek
train_df["month"] = train_df["datetime"].dt.month
test_df["hour"] = test_df["datetime"].dt.hour
test_df["day_of_week"] = test_df["datetime"].dt.dayofweek
test_df["month"] = test_df["datetime"].dt.month

# Feature engineering from weather data
# Create features from historical weather
historical_weather["datetime"] = pd.to_datetime(historical_weather["datetime"])
historical_weather["hour"] = historical_weather["datetime"].dt.hour
historical_weather["day_of_week"] = historical_weather["datetime"].dt.dayofweek
historical_weather["month"] = historical_weather["datetime"].dt.month

# Create weather features
weather_features = pd.DataFrame()
weather_features["hour"] = historical_weather["hour"]
weather_features["day_of_week"] = historical_weather["day_of_week"]
weather_features["month"] = historical_weather["month"]
weather_features["temperature"] = historical_weather["temperature"]
weather_features["dewpoint"] = historical_weather["dewpoint"]
weather_features["cloudcover_total"] = historical_weather["cloudcover_total"]
weather_features["cloudcover_low"] = historical_weather["cloudcover_low"]
weather_features["cloudcover_mid"] = historical_weather["cloudcover_mid"]
weather_features["cloudcover_high"] = historical_weather["cloudcover_high"]
weather_features["windspeed_10m"] = historical_weather["windspeed_10m"]
weather_features["winddirection_10m"] = historical_weather["winddirection_10m"]
weather_features["shortwave_radiation"] = historical_weather["shortwave_radiation"]
weather_features["direct_solar_radiation"] = historical_weather[
    "direct_solar_radiation"
]
weather_features["diffuse_radiation"] = historical_weather["diffuse_radiation"]

# Merge weather features with train data
train_weather = train_df.merge(
    weather_features, on=["hour", "day_of_week", "month"], how="left"
)
test_weather = test_df.merge(
    weather_features, on=["hour", "day_of_week", "month"], how="left"
)

# Feature engineering from electricity prices
train_prices = train_df.merge(
    electricity_prices[["forecast_date", "euros_per_mwh"]],
    left_on=["datetime"],
    right_on=["forecast_date"],
    how="left",
)
test_prices = test_df.merge(
    electricity_prices[["forecast_date", "euros_per_mwh"]],
    left_on=["datetime"],
    right_on=["forecast_date"],
    how="left",
)

# Feature engineering from client data
train_client = train_df.merge(
    client_df[
        ["product_type", "county", "is_business", "eic_count", "installed_capacity"]
    ],
    on=["product_type", "county", "is_business"],
    how="left",
)
test_client = test_df.merge(
    client_df[
        ["product_type", "county", "is_business", "eic_count", "installed_capacity"]
    ],
    on=["product_type", "county", "is_business"],
    how="left",
)

# Combine all features
train_df["train_weather"] = train_weather.drop("target", axis=1).merge(
    train_weather.drop("target", axis=1), how="left"
)
test_df["test_weather"] = test_weather.drop("target", axis=1).merge(
    test_weather.drop("target", axis=1), how="left"
)

# Prepare features for LightGBM
X_train = train_df.drop("target", axis=1)
y_train = train_df["target"]
X_test = test_df

# Create LightGBM model
model = lgb.LGBMRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    num_leaves=31,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
    verbose=-1,
)

# Train model
model.fit(X_train, y_train)

# Predict on test data
test_pred = model.predict(X_test)

# Create submission
submission = pd.DataFrame({"id": test_df["id"], "target": test_pred})
submission.to_csv("./working/submission.csv", index=False)

# Calculate MAE on validation set
val_score = mean_absolute_error(y_train, model.predict(X_train))
print(f"Validation MAE: {val_score:.4f}")


import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error

# Load data
train_df = pd.read_parquet("./input/train/train.csv")
test_df = pd.read_parquet("./input/test/test.csv")

# Create datetime features
train_df["datetime"] = pd.to_datetime(train_df["datetime"])
test_df["datetime"] = pd.to_datetime(test_df["datetime"])
train_df["hour"] = train_df["datetime"].dt.hour
train_df["day_of_week"] = train_df["datetime"].dt.dayofweek
train_df["month"] = train_df["datetime"].dt.month
test_df["hour"] = test_df["datetime"].dt.hour
test_df["day_of_week"] = test_df["datetime"].dt.dayofweek
test_df["month"] = test_df["datetime"].dt.month

# Load reference data
client_df = pd.read_csv("./input/client.csv")
weather_df = pd.read_parquet("./input/forecast_weather.parquet")
historical_weather = pd.read_parquet("./input/historical_weather.parquet")
weather_station_mapping = pd.read_csv("./input/weather_station_to_county_mapping.csv")
electricity_prices = pd.read_csv("./input/electricity_prices.csv")
gas_prices = pd.read_csv("./input/gas_prices.csv")

# Create validation split
kf = KFold(n_splits=5, shuffle=True, random_state=42)


# Feature engineering
def create_features(df):
    df["hour"] = df["datetime"].dt.hour
    df["day_of_week"] = df["datetime"].dt.dayofweek
    df["month"] = df["datetime"].dt.month
    df["is_business"] = df["is_business"].astype(int)
    df["is_consumption"] = df["is_consumption"].astype(int)
    return df


# Apply features
train_df = create_features(train_df)
test_df = create_features(test_df)

# Train and evaluate
best_model = None
best_val_score = float("inf")

for fold, (train_idx, val_idx) in enumerate(kf.split(train_df)):
    fold_train = train_df.iloc[train_idx]
    fold_val = train_df.iloc[val_idx]

    # Prepare data
    X_train = fold_train.drop("target", axis=1)
    y_train = fold_train["target"]
    X_val = fold_val.drop("target", axis=1)
    y_val = fold_val["target"]

    # Create LightGBM model
    train_data = lgb.Dataset(X_train, y_train)
    val_data = lgb.Dataset(X_val, y_val, reference=train_data)

    # Train model
    params = {
        "objective": "regression",
        "metric": "mae",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "seed": 42,
    }

    model = lgb.train(
        params,
        train_data,
        num_boost_round=100,
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
    )

    # Evaluate
    val_pred = model.predict(X_val, num_iteration=model.best_iteration)
    val_score = mean_absolute_error(y_val, val_pred)

    print(f"Fold {fold+1}: MAE = {val_score:.4f}")

    if val_score < best_val_score:
        best_val_score = val_score
        best_model = model

# Train final model on all data
final_train = train_df.dropna(subset=["target"])
X_final = final_train.drop("target", axis=1)
y_final = final_train["target"]

final_data = lgb.Dataset(X_final, y_final)
final_model = lgb.train(
    params,
    final_data,
    num_boost_round=500,
    callbacks=[lgb.early_stopping(100), lgb.log_evaluation(-1)],
)

# Predict on test data
test_data = lgb.Dataset(test_df.drop("target", axis=1), reference=final_data)
test_pred = final_model.predict(test_data, num_iteration=final_model.best_iteration)

# Create submission
submission = pd.DataFrame({"id": test_df["id"], "target": test_pred})
submission.to_csv("./working/submission.csv", index=False)

# Calculate MAE on validation set
val_score = mean_absolute_error(y_final, final_model.predict(X_final))
print(f"Final Validation MAE: {val_score:.4f}")
