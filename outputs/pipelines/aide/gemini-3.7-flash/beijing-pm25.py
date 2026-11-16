import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error

# 1. Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

os.makedirs("./working", exist_ok=True)


# 2. Physics-based and Temporal Feature Engineering
def preprocess_features(df):
    df = df.copy()

    # Datetime and continuous progression
    dt = pd.to_datetime(df[["year", "month", "day", "hour"]])
    df["dayofweek"] = dt.dt.dayofweek
    df["dayofyear"] = dt.dt.dayofyear
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["time_elapsed"] = (dt - pd.Timestamp("2010-01-01")).dt.total_seconds() / 3600.0

    # Cyclical representations
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)

    # Meteorological physics features
    df["dew_point_diff"] = df["TEMP"] - df["DEWP"]

    # Magnus-Tetens formulation for vapor pressures (hPa) & Relative Humidity (%)
    es = 6.112 * np.exp((17.67 * df["TEMP"]) / (df["TEMP"] + 243.5))
    ea = 6.112 * np.exp((17.67 * df["DEWP"]) / (df["DEWP"] + 243.5))
    df["relative_humidity"] = np.clip(100.0 * (ea / (es + 1e-6)), 0.0, 100.0)
    df["vapor_pressure_deficit"] = np.maximum(0.0, es - ea)

    # Atmospheric stability & density proxies
    temp_kelvin = df["TEMP"] + 273.15
    df["potential_temp"] = temp_kelvin * (1000.0 / df["PRES"]) ** 0.286
    df["air_density_proxy"] = df["PRES"] / temp_kelvin

    # Wind decomposition into directional velocity components (u: East-West, v: North-South)
    cbwd_angles = {
        "NE": np.pi / 4.0,
        "SE": 3.0 * np.pi / 4.0,
        "SW": 5.0 * np.pi / 4.0,
        "NW": 7.0 * np.pi / 4.0,
    }
    angles = df["cbwd"].map(cbwd_angles).fillna(0.0)
    is_calm = (df["cbwd"] == "cv").astype(float)
    df["wind_u"] = np.where(is_calm == 1, 0.0, -df["Iws"] * np.sin(angles))
    df["wind_v"] = np.where(is_calm == 1, 0.0, -df["Iws"] * np.cos(angles))

    # Precipitation indicators and interactions
    df["is_rain"] = (df["Ir"] > 0).astype(int)
    df["is_snow"] = (df["Is"] > 0).astype(int)
    df["wind_rain_interaction"] = df["Iws"] * df["Ir"]
    df["wind_snow_interaction"] = df["Iws"] * df["Is"]

    # Categorical feature handling
    if "cbwd" in df.columns:
        df["cbwd"] = df["cbwd"].astype("category")

    return df


train_fe = preprocess_features(train_df)
test_fe = preprocess_features(test_df)

ignore_cols = {"No", "pm2.5"}
features = [c for c in train_fe.columns if c not in ignore_cols]
target = "pm2.5"

X = train_fe[features]
y = train_fe[target]
X_test = test_fe[features]

# 3. Model Training & 5-Fold Cross Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

oof_preds = np.zeros(len(train_df))
test_preds = np.zeros(len(test_df))

lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "n_estimators": 3000,
    "learning_rate": 0.03,
    "num_leaves": 63,
    "max_depth": -1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

    model = lgb.LGBMRegressor(**lgb_params)

    callbacks = [
        lgb.early_stopping(stopping_rounds=100, verbose=False),
        lgb.log_evaluation(period=-1),
    ]

    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=callbacks)

    val_pred = model.predict(X_val)
    oof_preds[val_idx] = np.clip(val_pred, 0, None)

    test_fold_pred = model.predict(X_test)
    test_preds += np.clip(test_fold_pred, 0, None) / kf.n_splits

cv_rmse = root_mean_squared_error(y, oof_preds)
print(f"5-Fold CV Out-of-Fold RMSE: {cv_rmse:.4f}")

# 4. Save Submission
submission = pd.DataFrame({"id": np.arange(len(test_preds)), "pm2.5": test_preds})
submission.to_csv("./working/submission.csv", index=False)
print("Submission successfully saved to ./working/submission.csv")
