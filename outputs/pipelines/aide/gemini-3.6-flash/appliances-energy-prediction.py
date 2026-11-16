import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error

# Load data
train_path = "./input/train.parquet"
test_path = "./input/test.parquet"

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

# Track original order
train_df["orig_id"] = np.arange(len(train_df))
test_df["orig_id"] = np.arange(len(test_df))
train_df["is_train"] = 1
test_df["is_train"] = 0
test_df["Appliances"] = np.nan

# Combine for contiguous time-series feature engineering
combined = pd.concat([train_df, test_df], ignore_index=True)

# Robust date parsing
date_str = combined["date"].astype(str).str.replace(" ", "")
combined["date"] = pd.to_datetime(date_str, format="%Y-%m-%d%H:%M:%S")

# Sort chronologically to construct correct temporal features
combined = combined.sort_values("date").reset_index(drop=True)

# Time components
combined["hour"] = combined["date"].dt.hour
combined["minute"] = combined["date"].dt.minute
combined["dayofweek"] = combined["date"].dt.dayofweek
combined["day"] = combined["date"].dt.day
combined["month"] = combined["date"].dt.month
combined["dayofyear"] = combined["date"].dt.dayofyear
combined["is_weekend"] = (combined["dayofweek"] >= 5).astype(int)

time_min = combined["hour"] * 60 + combined["minute"]
combined["sin_time"] = np.sin(2 * np.pi * time_min / 1440.0)
combined["cos_time"] = np.cos(2 * np.pi * time_min / 1440.0)

combined["sin_dow"] = np.sin(2 * np.pi * combined["dayofweek"] / 7.0)
combined["cos_dow"] = np.cos(2 * np.pi * combined["dayofweek"] / 7.0)

# Temperature aggregated features
t_cols = [f"T{i}" for i in range(1, 10)]
existing_t = [c for c in t_cols if c in combined.columns]
if existing_t:
    combined["T_mean"] = combined[existing_t].mean(axis=1)
    combined["T_std"] = combined[existing_t].std(axis=1)
    combined["T_min"] = combined[existing_t].min(axis=1)
    combined["T_max"] = combined[existing_t].max(axis=1)
    combined["T_range"] = combined["T_max"] - combined["T_min"]
    if "T_out" in combined.columns:
        combined["T_diff_out"] = combined["T_mean"] - combined["T_out"]
        combined["T1_out_diff"] = combined["T1"] - combined["T_out"]
        combined["T2_out_diff"] = combined["T2"] - combined["T_out"]

# Relative Humidity aggregated features
rh_cols = [f"RH_{i}" for i in range(1, 10)]
existing_rh = [c for c in rh_cols if c in combined.columns]
if existing_rh:
    combined["RH_mean"] = combined[existing_rh].mean(axis=1)
    combined["RH_std"] = combined[existing_rh].std(axis=1)
    combined["RH_min"] = combined[existing_rh].min(axis=1)
    combined["RH_max"] = combined[existing_rh].max(axis=1)
    combined["RH_range"] = combined["RH_max"] - combined["RH_min"]
    if "RH_out" in combined.columns:
        combined["RH_diff_out"] = combined["RH_mean"] - combined["RH_out"]

if "T_out" in combined.columns and "Tdewpoint" in combined.columns:
    combined["Dew_diff"] = combined["T_out"] - combined["Tdewpoint"]

# Temporal lag, lead, step differences, and rolling statistics
sensor_cols = [
    c
    for c in [
        "lights",
        "T1",
        "RH_1",
        "T2",
        "RH_2",
        "T3",
        "RH_3",
        "T_out",
        "Press_mm_hg",
        "RH_out",
        "Windspeed",
        "T_mean",
        "RH_mean",
    ]
    if c in combined.columns
]

for col in sensor_cols:
    # Extended lags and leads
    for shift in [1, 2, 3, 6, -1, -2, -3, -6]:
        combined[f"{col}_lag_{shift}"] = combined[col].shift(shift)

    # Multi-step temporal differences
    for diff_step in [1, 3, 6]:
        combined[f"{col}_diff_{diff_step}"] = (
            combined[col] - combined[f"{col}_lag_{diff_step}"]
        )

    # Rolling statistics across 30-min (3 steps) and 60-min (6 steps) windows
    for win in [3, 6]:
        combined[f"{col}_roll_mean_{win}"] = (
            combined[col].rolling(win, min_periods=1).mean()
        )
        combined[f"{col}_roll_std_{win}"] = (
            combined[col].rolling(win, min_periods=1).std().fillna(0)
        )

combined = combined.drop(columns=["date"])

# Re-separate train and test preserving original order
train_proc = (
    combined[combined["is_train"] == 1]
    .sort_values("orig_id")
    .drop(columns=["is_train", "orig_id"])
)
test_proc = (
    combined[combined["is_train"] == 0]
    .sort_values("orig_id")
    .drop(columns=["is_train", "orig_id", "Appliances"])
)

X = train_proc.copy()
y = X.pop("Appliances")
X_test = test_proc[X.columns]

# Model setup
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(train_df))
test_preds = np.zeros(len(test_df))

params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.03,
    "num_leaves": 63,
    "max_depth": 8,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_samples": 20,
    "random_state": 42,
    "verbose": -1,
    "n_jobs": -1,
}

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=2500,
        valid_sets=[dtrain, dval],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )

    val_pred = model.predict(X_val, num_iteration=model.best_iteration)
    oof_preds[val_idx] = val_pred
    test_preds += (
        model.predict(X_test, num_iteration=model.best_iteration) / kf.n_splits
    )

oof_preds = np.clip(oof_preds, 10, None)
cv_rmse = root_mean_squared_error(y, oof_preds)
print(f"5-Fold CV RMSE: {cv_rmse:.4f}")

# Save final predictions
test_preds = np.clip(test_preds, 10, None)

os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame({"id": np.arange(len(test_preds)), "Appliances": test_preds})
submission.to_csv("./working/submission.csv", index=False)
