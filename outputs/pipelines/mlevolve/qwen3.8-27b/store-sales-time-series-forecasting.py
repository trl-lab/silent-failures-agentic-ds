import pandas as pd
import numpy as np
import lightgbm as lgb
import json
import os

# Create directories
os.makedirs("./submission", exist_ok=True)
os.makedirs("./working", exist_ok=True)

# ============ LOAD DATA ============
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")

# Parse dates
train["date"] = pd.to_datetime(train["date"])
test["date"] = pd.to_datetime(test["date"])

# Sort by date, store, family for consistency
train = train.sort_values(["date", "store_nbr", "family"]).reset_index(drop=True)
test = test.sort_values(["date", "store_nbr", "family"]).reset_index(drop=True)

# Time-based split: use last 90 days of train as validation
max_train_date = train["date"].max()
split_date = max_train_date - pd.Timedelta(days=90)

train_split = train[train["date"] <= split_date].reset_index(drop=True)
val_split = train[train["date"] > split_date].reset_index(drop=True)

print(
    f"Train split: {len(train_split)} rows, {train_split['date'].min()} to {train_split['date'].max()}"
)
print(
    f"Val split: {len(val_split)} rows, {val_split['date'].min()} to {val_split['date'].max()}"
)

# ============ FEATURE ENGINEERING ============


def engineer_features(df, is_train=True, train_df=None):
    """Engineer features for the dataframe. Fit statistics only on train_df."""
    df = df.copy()

    # Time features
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["day_of_week"] = df["date"].dt.dayofweek
    df["day_of_year"] = df["date"].dt.dayofyear
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["quarter"] = df["date"].dt.quarter
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # Cyclical encodings
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["dayofyear_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["dayofyear_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365)

    # Promotion features
    df["log_onpromotion"] = np.log1p(df["onpromotion"])
    df["has_promotion"] = (df["onpromotion"] > 0).astype(int)
    df["promotion_intensity"] = df["onpromotion"] / 742.0

    # Family and store encoding (label encode based on train)
    if is_train:
        family_map = {f: i for i, f in enumerate(sorted(df["family"].unique()))}
        store_map = {s: i for i, s in enumerate(sorted(df["store_nbr"].unique()))}
        df["family_encoded"] = df["family"].map(family_map)
        df["store_encoded"] = df["store_nbr"].map(store_map)
    else:
        family_map = {f: i for i, f in enumerate(sorted(train_df["family"].unique()))}
        store_map = {s: i for i, s in enumerate(sorted(train_df["store_nbr"].unique()))}
        df["family_encoded"] = df["family"].map(family_map)
        df["store_encoded"] = df["store_nbr"].map(store_map)

    # Lag features and rolling stats
    if is_train:
        for lag in [7, 14, 28, 35, 365]:
            df[f"lag_{lag}"] = df.groupby(["store_nbr", "family"])["sales"].shift(lag)

        for window in [7, 14, 28, 90]:
            rolled = df.groupby(["store_nbr", "family"])["sales"].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).mean()
            )
            df[f"rolling_mean_{window}"] = rolled

            rolled_std = df.groupby(["store_nbr", "family"])["sales"].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).std()
            )
            df[f"rolling_std_{window}"] = rolled_std
    else:
        # For val/test, compute lags using full train data + current data
        if "sales" in df.columns:
            combined = pd.concat(
                [
                    train_df[["date", "store_nbr", "family", "sales"]],
                    df[["date", "store_nbr", "family", "sales"]],
                ],
                ignore_index=True,
            )
        else:
            combined = pd.concat(
                [
                    train_df[["date", "store_nbr", "family", "sales"]],
                    df[["date", "store_nbr", "family"]],
                ],
                ignore_index=True,
            )
            combined["sales"] = np.nan

        combined = combined.sort_values(["store_nbr", "family", "date"]).reset_index(
            drop=True
        )

        for lag in [7, 14, 28, 35, 365]:
            combined[f"lag_{lag}"] = combined.groupby(["store_nbr", "family"])[
                "sales"
            ].shift(lag)

        for window in [7, 14, 28, 90]:
            combined[f"rolling_mean_{window}"] = combined.groupby(
                ["store_nbr", "family"]
            )["sales"].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).mean()
            )
            combined[f"rolling_std_{window}"] = combined.groupby(
                ["store_nbr", "family"]
            )["sales"].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).std()
            )

        # Extract features for the current df (last len(df) rows in combined)
        start_idx = len(train_df)
        for lag in [7, 14, 28, 35, 365]:
            df[f"lag_{lag}"] = combined.iloc[start_idx:][f"lag_{lag}"].values
        for window in [7, 14, 28, 90]:
            df[f"rolling_mean_{window}"] = combined.iloc[start_idx:][
                f"rolling_mean_{window}"
            ].values
            df[f"rolling_std_{window}"] = combined.iloc[start_idx:][
                f"rolling_std_{window}"
            ].values

    # Interaction features
    df["promo_x_dow"] = df["onpromotion"] * df["day_of_week"]
    df["promo_x_month"] = df["onpromotion"] * df["month"]
    df["store_family"] = df["store_encoded"] * 100 + df["family_encoded"]

    # Log transform of target
    if "sales" in df.columns:
        df["log_sales"] = np.log1p(df["sales"])

    return df


# Apply feature engineering
train_full = train.copy()

train_feat = engineer_features(train_split, is_train=True, train_df=None)
val_feat = engineer_features(val_split, is_train=False, train_df=train_split)
test_feat = engineer_features(test, is_train=False, train_df=train_full)

# Define feature columns
exclude_cols = ["id", "date", "sales", "log_sales", "family", "store_nbr"]
feature_cols = [c for c in train_feat.columns if c not in exclude_cols]

# Fill NaN in lag/rolling features with 0 (instead of dropping rows,
# which would eliminate all test samples when lag_365 is unavailable)
lag_cols = [f"lag_{lag}" for lag in [7, 14, 28, 35, 365]]
rolling_cols = [f"rolling_mean_{w}" for w in [7, 14, 28, 90]] + [
    f"rolling_std_{w}" for w in [7, 14, 28, 90]
]
all_nan_cols = lag_cols + rolling_cols

for col in all_nan_cols:
    if col in train_feat.columns:
        train_feat[col] = train_feat[col].fillna(0)
    if col in val_feat.columns:
        val_feat[col] = val_feat[col].fillna(0)
    if col in test_feat.columns:
        test_feat[col] = test_feat[col].fillna(0)

# Fill any remaining NaNs
for col in feature_cols:
    if train_feat[col].dtype in ["float64", "float32"]:
        train_feat[col] = train_feat[col].fillna(0)
        val_feat[col] = val_feat[col].fillna(0)
        test_feat[col] = test_feat[col].fillna(0)

print(f"Train: {train_feat.shape[0]} samples, {len(feature_cols)} features")
print(f"Val: {val_feat.shape[0]} samples")
print(f"Test: {test_feat.shape[0]} samples")

# Save test IDs
test_ids = (
    test_feat["id"].values if "id" in test_feat.columns else np.arange(len(test_feat))
)

# ============ MODEL TRAINING ============
X_train = train_feat[feature_cols].values.astype(np.float32)
y_train = train_feat["log_sales"].values
X_val = val_feat[feature_cols].values.astype(np.float32)
y_val = val_feat["log_sales"].values
X_test = test_feat[feature_cols].values.astype(np.float32)

params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.03,
    "num_leaves": 255,
    "max_depth": -1,
    "min_child_samples": 30,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.7,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "n_estimators": 10000,
    "verbose": -1,
    "random_state": 42,
    "n_jobs": -1,
    "force_row_wise": True,
}

train_dataset = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols)
val_dataset = lgb.Dataset(
    X_val, label=y_val, feature_name=feature_cols, reference=train_dataset
)

model = lgb.train(
    params,
    train_dataset,
    valid_sets=[train_dataset, val_dataset],
    valid_names=["train", "validation"],
    callbacks=[
        lgb.early_stopping(stopping_rounds=150, verbose=False),
        lgb.log_evaluation(period=500),
    ],
)

print(f"Best iteration: {model.best_iteration}")

# ============ VALIDATION (Official RMSLE) ============
val_pred_log = model.predict(X_val, num_iteration=model.best_iteration)
val_pred_log = np.maximum(val_pred_log, 0)

# RMSLE = sqrt(mean((log1p(y_true) - log1p(y_pred))^2))
# y_val = log1p(y_true), val_pred_log = log1p(y_pred)
rmsle_val = np.sqrt(np.mean((y_val - val_pred_log) ** 2))
print(f"Validation RMSLE: {rmsle_val:.6f}")

# ============ TEST INFERENCE ============
test_pred_log = model.predict(X_test, num_iteration=model.best_iteration)
test_pred_log = np.maximum(test_pred_log, 0)

# Convert from log-space to original scale
test_pred = np.expm1(test_pred_log)
test_pred = np.maximum(test_pred, 0)

# ============ SUBMISSION ============
submission = pd.DataFrame({"id": test_ids.astype(int), "sales": test_pred})
submission.to_csv("./submission/submission.csv", index=False)
print(f"Submission saved: {submission.shape}")
print(
    f"Submission stats: mean={submission['sales'].mean():.2f}, min={submission['sales'].min():.2f}, max={submission['sales'].max():.2f}"
)

# Feature importance
importance = model.feature_importance(importance_type="gain")
feat_imp = pd.DataFrame(
    {"feature": feature_cols, "importance": importance}
).sort_values("importance", ascending=False)
print("\nTop 15 features by gain:")
print(feat_imp.head(15).to_string(index=False))

print(f"Final Validation Score: {rmsle_val:.6f}")
