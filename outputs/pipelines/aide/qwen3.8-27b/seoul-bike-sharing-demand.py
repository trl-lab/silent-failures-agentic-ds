import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import root_mean_squared_error

# -----------------------------
# Setup
# -----------------------------
os.makedirs("./working", exist_ok=True)

TARGET = "Rented Bike Count"
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# -----------------------------
# Load data
# -----------------------------
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

# If test accidentally contains target, remove it
test = test.drop(columns=[TARGET], errors="ignore")

# Keep only rows with target for training
train = train.dropna(subset=[TARGET]).copy()


# -----------------------------
# Feature engineering / cleaning
# -----------------------------
def parse_dates(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    return pd.to_datetime(series, errors="coerce")


def add_features(
    df: pd.DataFrame, train_medians: pd.Series | None = None
) -> pd.DataFrame:
    df = df.copy()

    # Parse date
    df["Date"] = parse_dates(df["Date"])
    if df["Date"].isna().any():
        fallback = pd.Timestamp("2000-01-01")
        df["Date"] = df["Date"].fillna(fallback)

    # Numeric conversion
    numeric_cols = [
        "Hour",
        "Temperature",
        "Humidity",
        "Wind speed",
        "Visibility",
        "Dew point temperature",
        "Solar Radiation",
        "Rainfall",
        "Snowfall",
        "Holiday",
        "Functioning Day",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Calendar features
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["Dayofweek"] = df["Date"].dt.dayofweek
    df["IsWeekend"] = (df["Dayofweek"] >= 5).astype(int)

    # Cyclical features
    hour = df["Hour"].fillna(12)
    df["HourSin"] = np.sin(2 * np.pi * hour / 24)
    df["HourCos"] = np.cos(2 * np.pi * hour / 24)

    dow = df["Dayofweek"].fillna(0)
    df["DowSin"] = np.sin(2 * np.pi * dow / 7)
    df["DowCos"] = np.cos(2 * np.pi * dow / 7)

    # Season encoding
    season_map = {
        "spring": 0,
        "summer": 1,
        "autumn": 2,
        "winter": 3,
    }
    df["SeasonCode"] = df["Seasons"].astype(str).str.strip().str.lower().map(season_map)

    # Fill missing values
    fill_cols = [
        "Hour",
        "Temperature",
        "Humidity",
        "Wind speed",
        "Visibility",
        "Dew point temperature",
        "Solar Radiation",
        "Rainfall",
        "Snowfall",
        "Holiday",
        "Functioning Day",
        "Year",
        "Month",
        "Day",
        "Dayofweek",
        "IsWeekend",
        "HourSin",
        "HourCos",
        "DowSin",
        "DowCos",
        "SeasonCode",
    ]

    for col in fill_cols:
        if col not in df.columns:
            continue

        if col in ["Rainfall", "Snowfall"]:
            df[col] = df[col].fillna(0)
        elif (
            train_medians is not None
            and col in train_medians.index
            and not np.isnan(train_medians[col])
        ):
            df[col] = df[col].fillna(train_medians[col])
        else:
            df[col] = df[col].fillna(df[col].median())

    return df


# Prepare train first to get medians
train = add_features(train, train_medians=None)
train_medians = train.median(numeric_only=True)

# Prepare test using training medians
test = add_features(test, train_medians=train_medians)

# Feature list
base_features = [
    "Hour",
    "Temperature",
    "Humidity",
    "Wind speed",
    "Visibility",
    "Dew point temperature",
    "Solar Radiation",
    "Rainfall",
    "Snowfall",
    "Holiday",
    "Functioning Day",
]
engineered_features = [
    "Year",
    "Month",
    "Day",
    "Dayofweek",
    "IsWeekend",
    "HourSin",
    "HourCos",
    "DowSin",
    "DowCos",
    "SeasonCode",
]
features = [f for f in base_features + engineered_features if f in train.columns]

# Sort training data chronologically
train = train.sort_values(["Date", "Hour"], na_position="last").reset_index(drop=True)


# -----------------------------
# Chronological 5-fold validation
# -----------------------------
def make_time_series_folds(df: pd.DataFrame, n_splits: int = 5):
    """
    Create chronological folds grouped by (Date, Hour) to avoid splitting
    the same timestamp across train/validation.
    """
    year = df["Date"].dt.year.fillna(0).astype(int)
    month = df["Date"].dt.month.fillna(0).astype(int)
    day = df["Date"].dt.day.fillna(0).astype(int)
    hour = df["Hour"].fillna(0).astype(int)

    timestamp_key = year * 10_000_000 + month * 100_000 + day * 1_000 + hour
    unique_keys, inverse = np.unique(timestamp_key, return_inverse=True)
    n_unique = len(unique_keys)

    # If too few unique timestamps, fall back to row-based TimeSeriesSplit
    if n_unique < n_splits + 1:
        from sklearn.model_selection import TimeSeriesSplit

        tscv = TimeSeriesSplit(n_splits=min(n_splits, max(1, len(df) - 1)))
        return list(tscv.split(df))

    # Split unique timestamps into n_splits + 1 consecutive chunks.
    # Validation folds are chunks 1..n_splits; training uses all earlier chunks.
    chunks = np.array_split(np.arange(n_unique), n_splits + 1)
    unique_fold = np.repeat(np.arange(n_splits + 1), [len(c) for c in chunks])
    row_fold = unique_fold[inverse]

    folds = []
    for i in range(1, n_splits + 1):
        train_idx = np.where(row_fold < i)[0]
        val_idx = np.where(row_fold == i)[0]
        if len(train_idx) > 0 and len(val_idx) > 0:
            folds.append((train_idx, val_idx))

    if len(folds) < n_splits:
        from sklearn.model_selection import TimeSeriesSplit

        tscv = TimeSeriesSplit(n_splits=min(n_splits, max(1, len(df) - 1)))
        return list(tscv.split(df))

    return folds


folds = make_time_series_folds(train, n_splits=5)

# -----------------------------
# LightGBM parameters
# -----------------------------
params = {
    "objective": "regression_l2",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "min_child_samples": 20,
    "subsample": 0.9,
    "subsample_freq": 1,
    "colsample_bytree": 0.9,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "random_seed": RANDOM_STATE,
    "n_jobs": -1,
    "verbosity": -1,
}

num_boost_round = 1000
early_stopping_rounds = 50

# -----------------------------
# Cross-validation
# -----------------------------
rmse_scores = []
best_iterations = []

for fold_idx, (train_idx, val_idx) in enumerate(folds, start=1):
    X_train = train.loc[train_idx, features]
    y_train = train.loc[train_idx, TARGET]
    X_val = train.loc[val_idx, features]
    y_val = train.loc[val_idx, TARGET]

    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        valid_sets=[dval],
        valid_names=["val"],
        callbacks=[
            lgb.early_stopping(early_stopping_rounds),
            lgb.log_evaluation(-1),
        ],
    )

    best_iter = getattr(model, "best_iteration", num_boost_round)
    if best_iter is None or best_iter <= 0:
        best_iter = num_boost_round

    best_iterations.append(best_iter)

    val_pred = model.predict(X_val, num_iteration=best_iter)
    val_pred = np.clip(val_pred, 0, None)

    fold_rmse = root_mean_squared_error(y_val, val_pred)
    rmse_scores.append(fold_rmse)

    print(f"Fold {fold_idx}: RMSE = {fold_rmse:.4f}, best_iteration = {best_iter}")

mean_rmse = float(np.mean(rmse_scores))
std_rmse = float(np.std(rmse_scores))
print(f"Mean CV RMSE: {mean_rmse:.4f} (+/- {std_rmse:.4f})")

# -----------------------------
# Final model on full training data
# -----------------------------
final_rounds = int(np.median(best_iterations)) if best_iterations else 500
final_rounds = max(100, min(final_rounds, num_boost_round))

X_all = train[features]
y_all = train[TARGET]

dtrain_full = lgb.Dataset(X_all, label=y_all)
final_model = lgb.train(
    params,
    dtrain_full,
    num_boost_round=final_rounds,
    callbacks=[lgb.log_evaluation(-1)],
)

# -----------------------------
# Predict test and save submission
# -----------------------------
X_test = test[features]
test_pred = final_model.predict(X_test, num_iteration=final_rounds)
test_pred = np.clip(test_pred, 0, None)

submission = pd.DataFrame(
    {
        "id": np.arange(len(test), dtype=int),
        "Rented Bike Count": test_pred.astype(float),
    }
)

submission.to_csv("./working/submission.csv", index=False)
print("Saved ./working/submission.csv")
