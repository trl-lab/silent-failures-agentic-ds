import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error
from xgboost import XGBRegressor

# Load dataset
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")


# Preprocessing & Feature Engineering
def feature_engineering(train, test):
    train_cp = train.copy()
    test_cp = test.copy()

    # Store target columns
    casual_target = np.log1p(train_cp["casual"].values)
    registered_target = np.log1p(train_cp["registered"].values)
    actual_count = train_cp["count"].values

    combined = pd.concat(
        [
            train_cp.drop(columns=["casual", "registered", "count"], errors="ignore"),
            test_cp,
        ],
        axis=0,
        ignore_index=True,
    )
    combined["datetime"] = pd.to_datetime(combined["datetime"])

    # Temporal features
    combined["year"] = combined["datetime"].dt.year
    combined["month"] = combined["datetime"].dt.month
    combined["day"] = combined["datetime"].dt.day
    combined["hour"] = combined["datetime"].dt.hour
    combined["dayofweek"] = combined["datetime"].dt.dayofweek
    combined["quarter"] = combined["datetime"].dt.quarter
    combined["is_weekend"] = (combined["dayofweek"] >= 5).astype(int)
    combined["dayofyear"] = combined["datetime"].dt.dayofyear

    # Cyclical representations
    combined["sin_hour"] = np.sin(2 * np.pi * combined["hour"] / 24.0)
    combined["cos_hour"] = np.cos(2 * np.pi * combined["hour"] / 24.0)
    combined["sin_month"] = np.sin(2 * np.pi * combined["month"] / 12.0)
    combined["cos_month"] = np.cos(2 * np.pi * combined["month"] / 12.0)
    combined["sin_dow"] = np.sin(2 * np.pi * combined["dayofweek"] / 7.0)
    combined["cos_dow"] = np.cos(2 * np.pi * combined["dayofweek"] / 7.0)

    # Domain & Weather features
    combined["temp_atemp_diff"] = combined["atemp"] - combined["temp"]
    combined["windspeed_is_zero"] = (combined["windspeed"] == 0).astype(int)
    combined["ideal_temp"] = np.exp(-((combined["temp"] - 25.0) ** 2) / 100.0)
    combined["discomfort_index"] = combined["temp"] - 0.55 * (
        1.0 - combined["humidity"] / 100.0
    ) * (combined["temp"] - 14.5)

    # Commuting & Period Indicators
    combined["rush_hour_morning"] = (
        (combined["workingday"] == 1) & (combined["hour"].between(7, 9))
    ).astype(int)
    combined["rush_hour_evening"] = (
        (combined["workingday"] == 1) & (combined["hour"].between(17, 19))
    ).astype(int)
    combined["weekend_peak"] = (
        (combined["workingday"] == 0) & (combined["hour"].between(11, 16))
    ).astype(int)
    combined["night_time"] = ((combined["hour"] >= 0) & (combined["hour"] <= 5)).astype(
        int
    )
    combined["hour_workingday"] = combined["hour"] * 2 + combined["workingday"]
    combined["hour_season"] = combined["hour"] * 4 + (combined["season"] - 1)

    # Relative weather deviations by month and hour
    group_weather = combined.groupby(["month", "hour"])[["temp", "humidity"]].transform(
        "mean"
    )
    combined["temp_diff_hour_mean"] = combined["temp"] - group_weather["temp"]
    combined["humidity_diff_hour_mean"] = (
        combined["humidity"] - group_weather["humidity"]
    )

    # Drop datetime column
    combined = combined.drop(columns=["datetime"])

    n_train = len(train_cp)
    X_train = combined.iloc[:n_train].copy()
    X_test = combined.iloc[n_train:].copy()

    return X_train, X_test, casual_target, registered_target, actual_count


X_train, X_test, y_casual, y_registered, y_count = feature_engineering(
    train_df, test_df
)

# Modeling with XGBoost using 5-Fold Cross Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

oof_casual = np.zeros(len(X_train))
oof_registered = np.zeros(len(X_train))
test_casual = np.zeros(len(X_test))
test_registered = np.zeros(len(X_test))

xgb_params = {
    "n_estimators": 1500,
    "learning_rate": 0.025,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "n_jobs": -1,
    "tree_method": "hist",
}

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]

    # Casual submodel
    model_casual = XGBRegressor(**xgb_params)
    model_casual.fit(X_tr, y_casual[train_idx])
    oof_casual[val_idx] = model_casual.predict(X_val)
    test_casual += model_casual.predict(X_test) / 5.0

    # Registered submodel
    model_registered = XGBRegressor(**xgb_params)
    model_registered.fit(X_tr, y_registered[train_idx])
    oof_registered[val_idx] = model_registered.predict(X_val)
    test_registered += model_registered.predict(X_test) / 5.0

# Reconstruct predicted demand counts
oof_pred_count = np.maximum(0, np.expm1(oof_casual)) + np.maximum(
    0, np.expm1(oof_registered)
)
test_pred_count = np.maximum(0, np.expm1(test_casual)) + np.maximum(
    0, np.expm1(test_registered)
)

# Evaluate RMSLE
rmsle = root_mean_squared_error(np.log1p(y_count), np.log1p(oof_pred_count))
print(f"Validation RMSLE: {rmsle:.5f}")

# Save submission
submission = pd.DataFrame({"id": np.arange(len(X_test)), "count": test_pred_count})
submission.to_csv("./working/submission.csv", index=False)
