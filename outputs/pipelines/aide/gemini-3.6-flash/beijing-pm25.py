import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# 1. Load Data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Mark dataset splits and preserve original indices
train_df["is_test"] = 0
test_df["is_test"] = 1
test_df["pm2.5"] = np.nan

train_df = train_df.dropna(subset=["pm2.5"]).reset_index(drop=True)
combined_df = pd.concat([train_df, test_df], ignore_index=True)
combined_df["orig_id"] = combined_df.index

# Create datetime ordering
combined_df["datetime"] = pd.to_datetime(combined_df[["year", "month", "day", "hour"]])
combined_df = combined_df.sort_values("datetime").reset_index(drop=True)

# 2. Domain Feature Engineering
# Relative Humidity (Magnus formula)
temp = combined_df["TEMP"].astype(float)
dewp = combined_df["DEWP"].astype(float)
alpha_t = (17.625 * temp) / (243.04 + temp)
alpha_d = (17.625 * dewp) / (243.04 + dewp)
combined_df["RH"] = 100.0 * np.exp(alpha_d - alpha_t)

# Dew point depression & Air density proxy
combined_df["dew_point_dep"] = temp - dewp
combined_df["air_density_proxy"] = combined_df["PRES"].astype(float) / (
    287.058 * (temp + 273.15)
)

# Wind components (U/V vectors)
wind_dir_map_u = {"NW": -0.7071, "NE": 0.7071, "SE": 0.7071, "cv": 0.0, "SW": -0.7071}
wind_dir_map_v = {"NW": -0.7071, "NE": -0.7071, "SE": 0.7071, "cv": 0.0, "SW": 0.7071}

iws = combined_df["Iws"].astype(float)
combined_df["wind_u"] = combined_df["cbwd"].map(wind_dir_map_u).fillna(0.0) * iws
combined_df["wind_v"] = combined_df["cbwd"].map(wind_dir_map_v).fillna(0.0) * iws

# Precipitation interaction
combined_df["precip_sum"] = combined_df["Is"] + combined_df["Ir"]
combined_df["is_snowing"] = (combined_df["Is"] > 0).astype(int)
combined_df["is_raining"] = (combined_df["Ir"] > 0).astype(int)

# Cyclical temporal encoding
combined_df["dayofweek"] = combined_df["datetime"].dt.dayofweek
combined_df["dayofyear"] = combined_df["datetime"].dt.dayofyear

combined_df["hour_sin"] = np.sin(2 * np.pi * combined_df["hour"] / 24.0)
combined_df["hour_cos"] = np.cos(2 * np.pi * combined_df["hour"] / 24.0)
combined_df["month_sin"] = np.sin(2 * np.pi * combined_df["month"] / 12.0)
combined_df["month_cos"] = np.cos(2 * np.pi * combined_df["month"] / 12.0)
combined_df["dow_sin"] = np.sin(2 * np.pi * combined_df["dayofweek"] / 7.0)
combined_df["dow_cos"] = np.cos(2 * np.pi * combined_df["dayofweek"] / 7.0)

# Categorical One-Hot Encoding for cbwd
combined_df = pd.get_dummies(combined_df, columns=["cbwd"], drop_first=False)

# Weather variables for temporal lag, lead, and rolling features
weather_cols = [
    "DEWP",
    "TEMP",
    "PRES",
    "Iws",
    "RH",
    "dew_point_dep",
    "wind_u",
    "wind_v",
]

for col in weather_cols:
    # Lags (past observations) and Leads (future observations)
    for s in [1, 2, 3, 6, 12, 24]:
        combined_df[f"{col}_lag_{s}"] = combined_df[col].shift(s)
        combined_df[f"{col}_lead_{s}"] = combined_df[col].shift(-s)

    # Rate of change deltas
    combined_df[f"{col}_diff_1"] = combined_df[col] - combined_df[f"{col}_lag_1"]
    combined_df[f"{col}_lead_diff_1"] = combined_df[f"{col}_lead_1"] - combined_df[col]

    # Backward and Forward Rolling Statistics
    for window in [3, 6, 12, 24]:
        combined_df[f"{col}_roll_mean_{window}"] = (
            combined_df[col].rolling(window, min_periods=1).mean()
        )
        combined_df[f"{col}_roll_std_{window}"] = (
            combined_df[col].rolling(window, min_periods=1).std().fillna(0)
        )
        combined_df[f"{col}_lead_roll_mean_{window}"] = (
            combined_df[col].iloc[::-1].rolling(window, min_periods=1).mean().iloc[::-1]
        )

# Restore original row order
combined_df = combined_df.sort_values("orig_id").reset_index(drop=True)

# 3. Separate Train and Test
train_data = combined_df[combined_df["is_test"] == 0].reset_index(drop=True)
test_data = combined_df[combined_df["is_test"] == 1].reset_index(drop=True)

drop_cols = ["No", "datetime", "is_test", "orig_id", "pm2.5"]
feature_cols = [c for c in train_data.columns if c not in drop_cols]

X = train_data[feature_cols]
y = train_data["pm2.5"].values
X_test = test_data[feature_cols]

# 4. Cross-Validation & Model Training
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(train_data))
test_preds = np.zeros(len(test_data))

xgb_params = {
    "n_estimators": 3000,
    "learning_rate": 0.03,
    "max_depth": 7,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "tree_method": "hist",
    "random_state": 42,
    "early_stopping_rounds": 100,
}

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_train, y_train = X.iloc[train_idx], y[train_idx]
    X_val, y_val = X.iloc[val_idx], y[val_idx]

    model = xgb.XGBRegressor(**xgb_params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    val_pred = model.predict(X_val)
    oof_preds[val_idx] = val_pred
    test_preds += model.predict(X_test) / kf.n_splits

# Post-processing: Bound target values to non-negative domain
oof_preds = np.clip(oof_preds, 0, None)
test_preds = np.clip(test_preds, 0, None)

cv_rmse = np.sqrt(mean_squared_error(y, oof_preds))
print(f"5-Fold CV RMSE: {cv_rmse:.4f}")

# 5. Save Submission
os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame({"id": np.arange(len(test_preds)), "pm2.5": test_preds})
submission.to_csv("./working/submission.csv", index=False)
