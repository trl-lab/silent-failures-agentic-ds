import os
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error
import lightgbm as lgb

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")


def parse_dates(date_series):
    # Normalize date strings that might lack whitespace between date and time
    cleaned = date_series.astype(str).str.replace(
        r"(\d{4}-\d{2}-\d{2})\s*(\d{2}:\d{2}:\d{2})", r"\1 \2", regex=True
    )
    return pd.to_datetime(cleaned, format="%Y-%m-%d %H:%M:%S", errors="coerce")


# Feature engineering function
def preprocess(df):
    df = df.copy()
    if "date" in df.columns:
        dt = parse_dates(df["date"])
        df["hour"] = dt.dt.hour
        df["dayofweek"] = dt.dt.dayofweek
        df["day"] = dt.dt.day
        df["month"] = dt.dt.month
        df["minute"] = dt.dt.minute
        df["time_of_day_min"] = df["hour"] * 60 + df["minute"]

        # Cyclical features for periodic time components
        df["sin_time"] = np.sin(2 * np.pi * df["time_of_day_min"] / 1440.0)
        df["cos_time"] = np.cos(2 * np.pi * df["time_of_day_min"] / 1440.0)
        df["sin_dow"] = np.sin(2 * np.pi * df["dayofweek"] / 7.0)
        df["cos_dow"] = np.cos(2 * np.pi * df["dayofweek"] / 7.0)

        df.drop(columns=["date"], inplace=True)

    # Aggregated room sensor features
    temp_cols = [f"T{i}" for i in range(1, 10)]
    rh_cols = [f"RH_{i}" for i in range(1, 10)]

    available_temps = [c for c in temp_cols if c in df.columns]
    available_rhs = [c for c in rh_cols if c in df.columns]

    if available_temps:
        df["T_mean"] = df[available_temps].mean(axis=1)
        df["T_std"] = df[available_temps].std(axis=1)
        df["T_max"] = df[available_temps].max(axis=1)
        df["T_min"] = df[available_temps].min(axis=1)
        if "T_out" in df.columns:
            df["T_diff_out"] = df["T_mean"] - df["T_out"]

    if available_rhs:
        df["RH_mean"] = df[available_rhs].mean(axis=1)
        df["RH_std"] = df[available_rhs].std(axis=1)
        df["RH_max"] = df[available_rhs].max(axis=1)
        df["RH_min"] = df[available_rhs].min(axis=1)
        if "RH_out" in df.columns:
            df["RH_diff_out"] = df["RH_mean"] - df["RH_out"]

    return df


target_col = "Appliances"
y = train_df[target_col].values
X = preprocess(train_df.drop(columns=[target_col]))
X_test = preprocess(test_df)

# Align columns
features = [col for col in X.columns if col in X_test.columns]
X = X[features]
X_test = X_test[features]

# 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(train_df))
test_preds = np.zeros(len(test_df))

lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "n_estimators": 2500,
    "learning_rate": 0.03,
    "num_leaves": 63,
    "max_depth": -1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "n_jobs": -1,
}

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_train_f, y_train_f = X.iloc[train_idx], y[train_idx]
    X_val_f, y_val_f = X.iloc[val_idx], y[val_idx]

    model = lgb.LGBMRegressor(**lgb_params)
    model.fit(
        X_train_f,
        y_train_f,
        eval_set=[(X_val_f, y_val_f)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )

    val_pred = model.predict(X_val_f)
    oof_preds[val_idx] = val_pred
    test_preds += model.predict(X_test) / kf.n_splits

    fold_rmse = root_mean_squared_error(y_val_f, val_pred)
    print(f"Fold {fold + 1} RMSE: {fold_rmse:.4f}")

cv_rmse = root_mean_squared_error(y, oof_preds)
print(f"Overall OOF RMSE: {cv_rmse:.4f}")

# Energy consumption is strictly non-negative
test_preds = np.clip(test_preds, 0, None)

# Save submission
os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame({"id": np.arange(len(test_preds)), "Appliances": test_preds})
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved successfully.")
