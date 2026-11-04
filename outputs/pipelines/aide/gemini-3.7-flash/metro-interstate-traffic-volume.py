import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error

# 1. Load Data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")


# 2. Feature Engineering & Preprocessing
def preprocess_base_features(df):
    df = df.copy()

    # Ensure date_time is datetime
    df["date_time"] = pd.to_datetime(df["date_time"])

    # Temporal features
    df["hour"] = df["date_time"].dt.hour
    df["dayofweek"] = df["date_time"].dt.dayofweek
    df["month"] = df["date_time"].dt.month
    df["year"] = df["date_time"].dt.year
    df["day"] = df["date_time"].dt.day
    df["dayofyear"] = df["date_time"].dt.dayofyear
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)
    df["hour_of_week"] = df["dayofweek"] * 24 + df["hour"]

    # Continuous time trend (days from origin)
    min_date = pd.to_datetime("2012-10-01")
    df["time_elapsed_days"] = (df["date_time"] - min_date).dt.total_seconds() / (
        24 * 3600
    )

    # Rush hour features (typical commuter spikes on weekdays)
    df["is_morning_rush"] = (
        (df["is_weekend"] == 0) & (df["hour"].between(6, 9))
    ).astype(int)
    df["is_evening_rush"] = (
        (df["is_weekend"] == 0) & (df["hour"].between(15, 18))
    ).astype(int)
    df["is_daytime"] = df["hour"].between(6, 20).astype(int)

    # Cyclical representations
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["month_sin"] = np.sin(2 * np.pi * (df["month"] - 1) / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * (df["month"] - 1) / 12.0)
    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7.0)

    # Holiday indicator
    df["is_holiday"] = (~df["holiday"].isin(["None", "none", None])).astype(int)

    # Data cleaning for sensor anomalies
    df.loc[df["temp"] < 100, "temp"] = np.nan
    df.loc[df["rain_1h"] > 500, "rain_1h"] = np.nan

    # Weather domain interactions
    df["temp_celsius"] = df["temp"] - 273.15
    df["is_freezing"] = (df["temp_celsius"] < 0).astype(int)
    df["has_rain"] = (df["rain_1h"] > 0).astype(int)
    df["has_snow"] = (df["snow_1h"] > 0).astype(int)
    df["has_precip"] = ((df["rain_1h"] > 0) | (df["snow_1h"] > 0)).astype(int)
    df["precip_total"] = df["rain_1h"].fillna(0) + df["snow_1h"].fillna(0)

    # Convert categorical columns to category dtype
    cat_cols = ["holiday", "weather_main", "weather_description"]
    for col in cat_cols:
        df[col] = df[col].astype("category")

    drop_cols = ["date_time"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    return df


target_col = "traffic_volume"
y = train_df[target_col].values

X = preprocess_base_features(train_df.drop(columns=[target_col]))
X_test = preprocess_base_features(test_df)

categorical_features = ["holiday", "weather_main", "weather_description"]

# 3. Model Training with Out-of-Fold Target Statistics & Cross-Validation
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_preds = np.zeros(len(train_df))
test_preds = np.zeros(len(test_df))

# Compute overall full training stats for test set target encoding
full_how_mean = train_df.groupby(X["hour_of_week"])[target_col].mean().to_dict()
full_how_std = train_df.groupby(X["hour_of_week"])[target_col].std().to_dict()
full_weekend_hour_mean = (
    train_df.groupby([X["is_weekend"], X["hour"]])[target_col].mean().to_dict()
)

X_test_feat = X_test.copy()
X_test_feat["target_enc_how_mean"] = (
    X_test_feat["hour_of_week"].map(full_how_mean).fillna(y.mean())
)
X_test_feat["target_enc_how_std"] = (
    X_test_feat["hour_of_week"].map(full_how_std).fillna(y.std())
)
weekend_hour_test_tuples = list(
    zip(X_test_feat["is_weekend"], X_test_feat["hour"], strict=False)
)
X_test_feat["target_enc_wh_mean"] = [
    full_weekend_hour_mean.get(t, y.mean()) for t in weekend_hour_test_tuples
]

lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.04,
    "num_leaves": 127,
    "max_depth": -1,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.85,
    "bagging_freq": 1,
    "min_child_samples": 20,
    "verbosity": -1,
    "n_jobs": -1,
    "random_state": 42,
}

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_tr = X.iloc[train_idx].copy()
    y_tr = y[train_idx]
    X_val = X.iloc[val_idx].copy()
    y_val = y[val_idx]

    # Out-of-fold target encodings computed strictly on train_idx
    how_mean_map = pd.Series(y_tr).groupby(X_tr["hour_of_week"].values).mean().to_dict()
    how_std_map = pd.Series(y_tr).groupby(X_tr["hour_of_week"].values).std().to_dict()
    wh_mean_map = (
        pd.Series(y_tr)
        .groupby([X_tr["is_weekend"].values, X_tr["hour"].values])
        .mean()
        .to_dict()
    )

    y_tr_mean = y_tr.mean()
    y_tr_std = y_tr.std()

    # Apply encodings to training split
    X_tr["target_enc_how_mean"] = (
        X_tr["hour_of_week"].map(how_mean_map).fillna(y_tr_mean)
    )
    X_tr["target_enc_how_std"] = X_tr["hour_of_week"].map(how_std_map).fillna(y_tr_std)
    tr_tuples = list(zip(X_tr["is_weekend"], X_tr["hour"], strict=False))
    X_tr["target_enc_wh_mean"] = [wh_mean_map.get(t, y_tr_mean) for t in tr_tuples]

    # Apply encodings to validation split
    X_val["target_enc_how_mean"] = (
        X_val["hour_of_week"].map(how_mean_map).fillna(y_tr_mean)
    )
    X_val["target_enc_how_std"] = (
        X_val["hour_of_week"].map(how_std_map).fillna(y_tr_std)
    )
    val_tuples = list(zip(X_val["is_weekend"], X_val["hour"], strict=False))
    X_val["target_enc_wh_mean"] = [wh_mean_map.get(t, y_tr_mean) for t in val_tuples]

    train_data = lgb.Dataset(
        X_tr, label=y_tr, categorical_feature=categorical_features, free_raw_data=False
    )
    val_data = lgb.Dataset(
        X_val,
        label=y_val,
        categorical_feature=categorical_features,
        reference=train_data,
        free_raw_data=False,
    )

    model = lgb.train(
        params=lgb_params,
        train_set=train_data,
        num_boost_round=3000,
        valid_sets=[val_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )

    oof_preds[val_idx] = model.predict(X_val, num_iteration=model.best_iteration)
    test_preds += (
        model.predict(X_test_feat, num_iteration=model.best_iteration) / n_splits
    )

# 4. Evaluation
cv_rmse = root_mean_squared_error(y, oof_preds)
print(f"5-Fold CV Out-Of-Fold RMSE: {cv_rmse:.4f}")

# 5. Save Submission
os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame(
    {"id": np.arange(len(test_df)), "traffic_volume": np.clip(test_preds, 0, None)}
)
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved to ./working/submission.csv successfully.")
