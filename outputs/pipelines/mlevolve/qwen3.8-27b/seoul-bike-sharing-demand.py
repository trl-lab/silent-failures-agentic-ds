import pandas as pd
import numpy as np
import os
import lightgbm as lgb
from sklearn.metrics import root_mean_squared_error

# ============ SETUP ============
os.makedirs("./submission", exist_ok=True)
os.makedirs("./working", exist_ok=True)

# ============ LOAD DATA ============
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")

# Parse dates and sort
train["Date"] = pd.to_datetime(train["Date"], dayfirst=True)
test["Date"] = pd.to_datetime(test["Date"], dayfirst=True)
train = train.sort_values("Date").reset_index(drop=True)
test = test.sort_values("Date").reset_index(drop=True)

print(f"Train date range: {train['Date'].min()} to {train['Date'].max()}")
print(f"Test date range: {test['Date'].min()} to {test['Date'].max()}")

# ============ DATA CLEANING ============
# Holiday column may contain non-numeric strings (e.g., empty strings)
train["Holiday"] = pd.to_numeric(train["Holiday"], errors="coerce").fillna(0).astype(int)
test["Holiday"] = pd.to_numeric(test["Holiday"], errors="coerce").fillna(0).astype(int)

# Functioning Day may also have issues
if "Functioning Day" in train.columns:
    train["Functioning Day"] = pd.to_numeric(train["Functioning Day"], errors="coerce").fillna(0).astype(int)
if "Functioning Day" in test.columns:
    test["Functioning Day"] = pd.to_numeric(test["Functioning Day"], errors="coerce").fillna(0).astype(int)

# ============ TIME-BASED SPLIT ============
split_idx = int(len(train) * 0.85)
train_split = train.iloc[:split_idx].reset_index(drop=True)
val_split = train.iloc[split_idx:].reset_index(drop=True)

print(f"Train split: {len(train_split)} samples")
print(f"Val split: {len(val_split)} samples")

# ============ FEATURE ENGINEERING ============


def engineer_base_features(df):
    """Engineer non-lag features (safe to apply to any dataframe)."""
    df = df.copy()

    # Date features
    df["year"] = df["Date"].dt.year
    df["month"] = df["Date"].dt.month
    df["day"] = df["Date"].dt.day
    df["day_of_week"] = df["Date"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["week_of_year"] = df["Date"].dt.isocalendar().week.astype(int)
    df["day_of_year"] = df["Date"].dt.dayofyear

    # Cyclical encoding for Hour
    df["hour_sin"] = np.sin(2 * np.pi * df["Hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["Hour"] / 24)

    # Cyclical encoding for day_of_week
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    # Cyclical encoding for month
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # Time-of-day segments
    df["is_morning_rush"] = ((df["Hour"] >= 7) & (df["Hour"] <= 9)).astype(int)
    df["is_evening_rush"] = ((df["Hour"] >= 17) & (df["Hour"] <= 19)).astype(int)
    df["is_night"] = ((df["Hour"] >= 22) | (df["Hour"] <= 5)).astype(int)
    df["is_daytime"] = ((df["Hour"] >= 8) & (df["Hour"] <= 18)).astype(int)

    # Weather interactions
    df["temp_comfort"] = (1 - np.abs(df["Temperature"] - 20) / 20).clip(0, 1)
    df["temp_squared"] = df["Temperature"] ** 2
    df["temp_cubed"] = df["Temperature"] ** 3
    df["humidity_comfort"] = (1 - np.abs(df["Humidity"] - 50) / 50).clip(0, 1)
    df["is_raining"] = (df["Rainfall"] > 0).astype(int)
    df["is_snowing"] = (df["Snowfall"] > 0).astype(int)
    df["precipitation_total"] = df["Rainfall"] + df["Snowfall"]
    df["solar_effective"] = df["Solar Radiation"] * df["is_daytime"]
    df["wind_chill"] = (
        df["Wind speed"] * np.abs(df["Temperature"]) / (df["Temperature"] + 273.15)
    )
    df["dew_spread"] = df["Temperature"] - df["Dew point temperature"]
    df["low_visibility"] = (df["Visibility"] < 10000).astype(int)

    # Season encoding
    season_map = {"Spring": 0, "Summer": 1, "Autumn": 2, "Winter": 3}
    if "Seasons" in df.columns:
        df["season_num"] = df["Seasons"].map(season_map).fillna(0).astype(int)

    # Holiday × Weekend
    df["holiday_weekend"] = (df["Holiday"] * df["is_weekend"]).astype(int)

    return df


def add_lag_features(df, target_col="Rented Bike Count"):
    """Add lag features computed from the dataframe's own target (for train)."""
    df = df.copy()
    dates_sorted = sorted(df["Date"].unique())
    date_to_idx = {d: i for i, d in enumerate(dates_sorted)}
    df["date_idx"] = df["Date"].map(date_to_idx)

    daily = df.groupby("Date")[target_col].agg(["mean"])
    daily.columns = ["daily_mean"]

    # Lag features for daily mean
    for lag in [1, 2, 3, 7]:
        df[f"daily_mean_lag{lag}"] = 0.0
        for i, date in enumerate(dates_sorted):
            if i >= lag:
                prev_date = dates_sorted[i - lag]
                df.loc[df["Date"] == date, f"daily_mean_lag{lag}"] = daily.loc[
                    prev_date, "daily_mean"
                ]

    # Same hour previous day
    df["same_hour_prev_day"] = 0.0
    for i, date in enumerate(dates_sorted):
        if i >= 1:
            prev_date = dates_sorted[i - 1]
            prev_data = df[df["Date"] == prev_date].sort_values("Hour")
            curr_data = df[df["Date"] == date].sort_values("Hour")
            for h in range(min(len(curr_data), len(prev_data))):
                mask = (df["Date"] == date) & (df["Hour"] == curr_data["Hour"].iloc[h])
                df.loc[mask, "same_hour_prev_day"] = prev_data[target_col].iloc[h]

    # Same hour previous week
    df["same_hour_prev_week"] = 0.0
    for i, date in enumerate(dates_sorted):
        if i >= 7:
            prev_date = dates_sorted[i - 7]
            prev_data = df[df["Date"] == prev_date].sort_values("Hour")
            curr_data = df[df["Date"] == date].sort_values("Hour")
            for h in range(min(len(curr_data), len(prev_data))):
                mask = (df["Date"] == date) & (df["Hour"] == curr_data["Hour"].iloc[h])
                df.loc[mask, "same_hour_prev_week"] = prev_data[target_col].iloc[h]

    df.drop(["date_idx"], axis=1, inplace=True)
    return df


def add_test_lag_features(df, train_df, target_col="Rented Bike Count"):
    """Add lag features for test data using only train data (no leakage)."""
    df = df.copy()
    all_dates = sorted(set(train_df["Date"].unique()) | set(df["Date"].unique()))
    date_to_idx_all = {d: i for i, d in enumerate(all_dates)}

    daily_train = train_df.groupby("Date")[target_col].agg(["mean"])
    daily_train.columns = ["daily_mean"]
    dates_train = sorted(train_df["Date"].unique())

    # Daily mean lags
    for lag in [1, 2, 3, 7]:
        df[f"daily_mean_lag{lag}"] = 0.0
        for date in df["Date"].unique():
            idx = date_to_idx_all.get(date, -1)
            if idx >= lag:
                prev_date = all_dates[idx - lag]
                if prev_date in daily_train.index:
                    df.loc[df["Date"] == date, f"daily_mean_lag{lag}"] = (
                        daily_train.loc[prev_date, "daily_mean"]
                    )
                else:
                    df.loc[df["Date"] == date, f"daily_mean_lag{lag}"] = daily_train[
                        "daily_mean"
                    ].mean()
            else:
                df.loc[df["Date"] == date, f"daily_mean_lag{lag}"] = daily_train[
                    "daily_mean"
                ].mean()

    # Same hour prev day
    df["same_hour_prev_day"] = 0.0
    train_dates_set = set(train_df["Date"].unique())
    for date in df["Date"].unique():
        idx = date_to_idx_all.get(date, -1)
        if idx >= 1:
            prev_date = all_dates[idx - 1]
            if prev_date in train_dates_set:
                prev_data = train_df[train_df["Date"] == prev_date].sort_values("Hour")
                prev_hours = set(prev_data["Hour"].values)
                for h in range(24):
                    if h in prev_hours:
                        val = prev_data[prev_data["Hour"] == h][target_col].values[0]
                        df.loc[
                            (df["Date"] == date) & (df["Hour"] == h),
                            "same_hour_prev_day",
                        ] = val

    # Same hour prev week
    df["same_hour_prev_week"] = 0.0
    for date in df["Date"].unique():
        idx = date_to_idx_all.get(date, -1)
        if idx >= 7:
            prev_date = all_dates[idx - 7]
            if prev_date in train_dates_set:
                prev_data = train_df[train_df["Date"] == prev_date].sort_values("Hour")
                prev_hours = set(prev_data["Hour"].values)
                for h in range(24):
                    if h in prev_hours:
                        val = prev_data[prev_data["Hour"] == h][target_col].values[0]
                        df.loc[
                            (df["Date"] == date) & (df["Hour"] == h),
                            "same_hour_prev_week",
                        ] = val

    return df


# Apply feature engineering
# For train: compute lags on full train (valid since lags only use past info), then split
train_full = engineer_base_features(train)
train_full = add_lag_features(train_full, "Rented Bike Count")

train_feat = train_full.iloc[:split_idx].reset_index(drop=True)
val_feat = train_full.iloc[split_idx:].reset_index(drop=True)

# For test: compute lags using only train data
test_feat = engineer_base_features(test)
test_feat = add_test_lag_features(test_feat, train, "Rented Bike Count")

# Define feature columns
exclude_cols = ["Date", "Rented Bike Count", "Seasons"]
feature_cols = [c for c in train_feat.columns if c not in exclude_cols]

# Ensure test has all feature columns
for col in feature_cols:
    if col not in test_feat.columns:
        test_feat[col] = 0
test_feat = test_feat[feature_cols].fillna(0)

train_feat = train_feat[feature_cols].fillna(0)
val_feat = val_feat[feature_cols].fillna(0)

train_target = train.iloc[:split_idx]["Rented Bike Count"].reset_index(drop=True)
val_target = train.iloc[split_idx:]["Rented Bike Count"].reset_index(drop=True)

print(f"\nFeatures: {len(feature_cols)}")
print(f"Train: {train_feat.shape}, Val: {val_feat.shape}, Test: {test_feat.shape}")

# ============ BASELINE ============
baseline_pred = np.full(len(val_target), train_target.mean())
baseline_rmse = root_mean_squared_error(val_target, baseline_pred)
print(f"Baseline RMSE (mean): {baseline_rmse:.4f}")

# ============ MULTI-SEED ENSEMBLE TRAINING ============
seeds = [42, 123, 7]
models = []
val_preds = np.zeros(len(val_feat))
test_preds = np.zeros(len(test_feat))

for seed_idx, seed in enumerate(seeds):
    params = {
        "objective": "huber",
        "alpha": 0.5,
        "metric": "l2",
        "boosting_type": "gbdt",
        "learning_rate": 0.01,
        "num_leaves": 15,
        "max_depth": 4,
        "min_child_samples": 50,
        "feature_fraction": 0.6,
        "bagging_fraction": 0.7,
        "bagging_freq": 3,
        "reg_alpha": 1.0,
        "reg_lambda": 5.0,
        "verbose": -1,
        "n_jobs": -1,
        "seed": seed,
    }

    train_data = lgb.Dataset(train_feat.values, label=train_target.values)
    val_data = lgb.Dataset(
        val_feat.values, label=val_target.values, reference=train_data
    )

    model = lgb.train(
        params,
        train_data,
        num_boost_round=10000,
        valid_sets=[train_data, val_data],
        valid_names=["train", "valid"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=300, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )

    best_iter = model.best_iteration
    best_val_l2 = model.best_score["valid"]["l2"]
    best_train_l2 = model.best_score["train"]["l2"]
    print(
        f"  Seed {seed}: best_iter={best_iter}, train_l2={best_train_l2:.4f}, val_l2={best_val_l2:.4f}"
    )

    models.append(model)

    vp = model.predict(val_feat.values, num_iteration=best_iter)
    vp = np.maximum(vp, 0)
    val_preds += vp

    tp = model.predict(test_feat.values, num_iteration=best_iter)
    tp = np.maximum(tp, 0)
    test_preds += tp

# Average ensemble
val_preds /= len(seeds)
test_preds /= len(seeds)

# ============ VALIDATION METRIC (Official: RMSE) ============
val_rmse = root_mean_squared_error(val_target, val_preds)
print(f"\nEnsemble Validation RMSE: {val_rmse:.4f}")
print(
    f"Improvement over baseline: {baseline_rmse - val_rmse:.4f} ({(baseline_rmse - val_rmse)/baseline_rmse*100:.1f}%)"
)

# ============ FEATURE IMPORTANCE ============
importance = models[0].feature_importance(importance_type="gain")
feat_imp = pd.Series(importance, index=train_feat.columns).sort_values(ascending=False)
print(f"\nTop 10 features by gain:")
for i, (name, val) in enumerate(feat_imp.head(10).items()):
    print(f"  {i+1}. {name}: {val:.1f}")

# ============ SUBMISSION ============
submission = pd.DataFrame(
    {"id": range(len(test_preds)), "Rented Bike Count": test_preds}
)
submission.to_csv("./submission/submission.csv", index=False)
print(f"\nSubmission saved: {submission.shape}")
print(
    f"Prediction stats: mean={test_preds.mean():.2f}, std={test_preds.std():.2f}, min={test_preds.min():.2f}, max={test_preds.max():.2f}"
)

# ============ FINAL SCORE ============
score = val_rmse
print(f"Final Validation Score: {score}")