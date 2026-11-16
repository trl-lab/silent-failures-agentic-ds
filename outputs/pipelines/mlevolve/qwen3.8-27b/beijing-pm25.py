import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import os
import warnings

warnings.filterwarnings("ignore")
os.makedirs("./submission", exist_ok=True)
os.makedirs("./working", exist_ok=True)

# ============================================================
# 1. LOAD DATA
# ============================================================
print("Loading data...")
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")


# ============================================================
# 2. CREATE DATETIME INDEX
# ============================================================
def create_datetime(df):
    df = df.copy()
    df["datetime"] = pd.to_datetime(
        df[["year", "month", "day", "hour"]].rename(
            columns={"year": "year", "month": "month", "day": "day", "hour": "hour"}
        ),
        format="%Y-%m-%d %H:00:00",
        errors="coerce",
    )
    mask = df["datetime"].isna()
    if mask.any():
        df.loc[mask, "datetime"] = pd.to_datetime(
            df.loc[mask, "year"].astype(str)
            + "-"
            + df.loc[mask, "month"].astype(str).str.zfill(2)
            + "-"
            + df.loc[mask, "day"].astype(str).str.zfill(2)
            + " "
            + df.loc[mask, "hour"].astype(str).str.zfill(2)
            + ":00:00",
            errors="coerce",
        )
    return df


train_df = create_datetime(train_df)
test_df = create_datetime(test_df)
train_df = train_df.sort_values("datetime").reset_index(drop=True)
test_df = test_df.sort_values("datetime").reset_index(drop=True)

print(f"Train date range: {train_df['datetime'].min()} to {train_df['datetime'].max()}")
print(f"Test date range: {test_df['datetime'].min()} to {test_df['datetime'].max()}")

# ============================================================
# 3. HANDLE MISSING VALUES
# ============================================================
feature_cols = ["DEWP", "TEMP", "PRES", "cbwd", "Iws", "Is", "Ir"]
target_col = "pm2.5"

for col in ["DEWP", "TEMP", "PRES", "Iws", "Is", "Ir"]:
    train_df[col] = train_df[col].where(train_df[col] >= 0, np.nan)
    test_df[col] = test_df[col].where(test_df[col] >= 0, np.nan)


# ============================================================
# 4. FEATURE ENGINEERING
# ============================================================
def engineer_features(df):
    df = df.copy()

    # Temporal Features
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["day_sin"] = np.sin(2 * np.pi * df["day"] / 31)
    df["day_cos"] = np.cos(2 * np.pi * df["day"] / 31)

    df["dow"] = df["datetime"].dt.dayofweek
    df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7)
    df["doy"] = df["datetime"].dt.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * df["doy"] / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * df["doy"] / 365.25)

    df["week"] = df["datetime"].dt.isocalendar().week.astype(int)
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["quarter"] = df["month"].map(lambda m: (m - 1) // 3 + 1)

    # Meteorological Interaction Features
    df["temp_dewp_spread"] = df["TEMP"] - df["DEWP"]
    df["TEMP_sq"] = df["TEMP"] ** 2
    df["DEWP_sq"] = df["DEWP"] ** 2
    df["PRES_TEMP"] = df["PRES"] * df["TEMP"]
    df["PRES_DEWP"] = df["PRES"] * df["DEWP"]
    df["Iws_inv"] = 1.0 / (df["Iws"] + 1e-6)
    df["Iws_sq"] = df["Iws"] ** 2
    df["Ir_TEMP"] = df["Ir"] * df["TEMP"]
    df["Is_Ir"] = df["Is"] * df["Ir"]

    # Wind direction encoding (cyclical)
    wind_map = {"NA": 4, "NE": 1, "SE": 3, "SW": 5, "N": 7, "E": 2, "S": 6, "W": 0}
    df["cbwd_num"] = df["cbwd"].map(wind_map).fillna(4)
    df["wind_sin"] = np.sin(2 * np.pi * df["cbwd_num"] / 8)
    df["wind_cos"] = np.cos(2 * np.pi * df["cbwd_num"] / 8)

    # Drop the original string column - it's fully encoded above
    if "cbwd" in df.columns:
        df.drop(columns=["cbwd"], inplace=True)

    return df


print("\nEngineering features...")
train_df = engineer_features(train_df)
test_df = engineer_features(test_df)


# ============================================================
# 5. LAG AND ROLLING FEATURES
# ============================================================
def add_lag_rolling_features(df, is_train=False, train_df_for_stats=None):
    df = df.copy()

    if is_train:
        for lag in [1, 2, 3, 6, 12, 24, 48, 168]:
            df[f"pm25_lag_{lag}"] = df["pm2.5"].shift(lag)

        for window in [6, 12, 24, 168]:
            df[f"pm25_rollmean_{window}"] = (
                df["pm2.5"].rolling(window, min_periods=1).mean()
            )
            df[f"pm25_rollstd_{window}"] = (
                df["pm2.5"].rolling(window, min_periods=1).std()
            )

        df["pm25_diff_1"] = df["pm2.5"].diff(1)
        df["pm25_diff_24"] = df["pm2.5"].diff(24)

    else:
        if train_df_for_stats is not None:
            train_tail = train_df_for_stats["pm2.5"].values
            max_lag = 168

            for lag in [1, 2, 3, 6, 12, 24, 48, 168]:
                prefix = (
                    train_tail[-lag:]
                    if len(train_tail) >= lag
                    else np.full(lag, np.nan)
                )
                if target_col in df.columns and not df[target_col].isna().all():
                    combined = np.concatenate([prefix, df[target_col].values])
                    shifted = pd.Series(combined).shift(lag).values[lag:]
                    if len(shifted) < len(df):
                        shifted = np.pad(
                            shifted, (len(df) - len(shifted), 0), constant_values=np.nan
                        )
                    df[f"pm25_lag_{lag}"] = shifted[: len(df)]
                else:
                    df[f"pm25_lag_{lag}"] = np.nan

            for window in [6, 12, 24, 168]:
                if target_col in df.columns and not df[target_col].isna().all():
                    prefix_len = max(window, 168)
                    prefix = (
                        train_tail[-prefix_len:]
                        if len(train_tail) >= prefix_len
                        else np.full(prefix_len, np.nan)
                    )
                    combined = np.concatenate([prefix, df[target_col].values])
                    roll_mean = (
                        pd.Series(combined)
                        .rolling(window, min_periods=1)
                        .mean()
                        .values[prefix_len:]
                    )
                    roll_std = (
                        pd.Series(combined)
                        .rolling(window, min_periods=1)
                        .std()
                        .values[prefix_len:]
                    )

                    if len(roll_mean) < len(df):
                        roll_mean = np.pad(
                            roll_mean,
                            (len(df) - len(roll_mean), 0),
                            constant_values=np.nan,
                        )
                        roll_std = np.pad(
                            roll_std,
                            (len(df) - len(roll_std), 0),
                            constant_values=np.nan,
                        )

                    df[f"pm25_rollmean_{window}"] = roll_mean[: len(df)]
                    df[f"pm25_rollstd_{window}"] = roll_std[: len(df)]
                else:
                    df[f"pm25_rollmean_{window}"] = np.nan
                    df[f"pm25_rollstd_{window}"] = np.nan

            if target_col in df.columns and not df[target_col].isna().all():
                prefix = train_tail[-1:] if len(train_tail) >= 1 else np.array([np.nan])
                combined = np.concatenate([prefix, df[target_col].values])
                df["pm25_diff_1"] = pd.Series(combined).diff(1).values[1:][: len(df)]

                prefix24 = (
                    train_tail[-24:] if len(train_tail) >= 24 else np.full(24, np.nan)
                )
                combined24 = np.concatenate([prefix24, df[target_col].values])
                df["pm25_diff_24"] = (
                    pd.Series(combined24).diff(24).values[24:][: len(df)]
                )
            else:
                df["pm25_diff_1"] = np.nan
                df["pm25_diff_24"] = np.nan

    # Lag features for meteorological variables
    for col in ["TEMP", "DEWP", "PRES", "Iws"]:
        for lag in [1, 3, 6, 12, 24]:
            df[f"{col}_lag_{lag}"] = df[col].shift(lag)

    for col in ["TEMP", "DEWP", "Iws"]:
        for window in [6, 24]:
            df[f"{col}_rollmean_{window}"] = (
                df[col].rolling(window, min_periods=1).mean()
            )

    return df


# ============================================================
# 6. CHRONOLOGICAL SPLIT
# ============================================================
n_train = len(train_df)
val_size = int(n_train * 0.15)
train_split = train_df.iloc[:-val_size].copy()
val_split = train_df.iloc[-val_size:].copy()

print(f"\nChronological split: train={len(train_split)}, val={len(val_split)}")
print(
    f"Train date range: {train_split['datetime'].min()} to {train_split['datetime'].max()}"
)
print(f"Val date range: {val_split['datetime'].min()} to {val_split['datetime'].max()}")

# ============================================================
# 7. APPLY LAG/ROLLING FEATURES
# ============================================================
print("\nAdding lag and rolling features...")
train_split = add_lag_rolling_features(train_split, is_train=True)
val_split = add_lag_rolling_features(
    val_split, is_train=False, train_df_for_stats=train_split
)
test_df = add_lag_rolling_features(test_df, is_train=False, train_df_for_stats=train_df)

# ============================================================
# 8. HANDLE REMAINING MISSING VALUES
# ============================================================
id_cols = ["No", "datetime", "year", "month", "day", "hour"]
all_cols = train_split.columns.tolist()
feature_cols_final = [c for c in all_cols if c not in id_cols + [target_col]]

# Safety: only keep numeric columns for modelling
feature_cols_final = [
    c for c in feature_cols_final if pd.api.types.is_numeric_dtype(train_split[c])
]

print(f"\nTotal feature columns: {len(feature_cols_final)}")

train_medians = train_split[feature_cols_final].median()

for col in feature_cols_final:
    if train_split[col].isna().any():
        train_split[col] = train_split[col].fillna(train_medians[col])
    if val_split[col].isna().any():
        val_split[col] = val_split[col].fillna(train_medians[col])
    if test_df[col].isna().any():
        test_df[col] = test_df[col].fillna(train_medians[col])

train_split = train_split.dropna(subset=[target_col]).reset_index(drop=True)
val_split = val_split.dropna(subset=[target_col]).reset_index(drop=True)

print(
    f"\nAfter cleaning: train={len(train_split)}, val={len(val_split)}, test={len(test_df)}"
)

# ============================================================
# 9. LOG TRANSFORM TARGET
# ============================================================
train_split["pm2.5_log"] = np.log1p(train_split["pm2.5"])
val_split["pm2.5_log"] = np.log1p(val_split["pm2.5"])

# ============================================================
# 10. PREPARE DATA FOR TREE MODELS
# ============================================================
X_train = train_split[feature_cols_final].values.astype(np.float64)
X_val = val_split[feature_cols_final].values.astype(np.float64)
X_test = test_df[feature_cols_final].values.astype(np.float64)

y_train = train_split["pm2.5_log"].values.astype(np.float64)
y_val = val_split["pm2.5_log"].values.astype(np.float64)
y_val_orig = val_split["pm2.5"].values.astype(np.float64)

print(f"\nData shapes: X_train={X_train.shape}, X_val={X_val.shape}, X_test={X_test.shape}")
print(f"Target (log scale): train mean={y_train.mean():.4f}, val mean={y_val.mean():.4f}")

# ============================================================
# 11. DEFINE AND TRAIN LIGHTGBM
# ============================================================
print("\n" + "=" * 60)
print("TRAINING LIGHTGBM")
print("=" * 60)

lgbm_params = {
    "objective": "regression",
    "metric": "rmse",
    "n_estimators": 3000,
    "learning_rate": 0.02,
    "max_depth": 8,
    "num_leaves": 128,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "verbose": -1,
    "n_jobs": -1,
    "random_state": 42,
}

lgbm_model = lgb.LGBMRegressor(**lgbm_params)
lgbm_model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    eval_metric="rmse",
    callbacks=[lgb.early_stopping(stopping_rounds=200, verbose=True)],
)

lgbm_val_pred_log = lgbm_model.predict(X_val)
lgbm_val_pred = np.clip(np.expm1(lgbm_val_pred_log), 0, None)
lgbm_val_rmse = np.sqrt(mean_squared_error(y_val_orig, lgbm_val_pred))
print(f"LightGBM Validation RMSE: {lgbm_val_rmse:.4f}")
print(f"LightGBM best iteration: {lgbm_model.best_iteration_}")

# ============================================================
# 12. DEFINE AND TRAIN XGBOOST
# ============================================================
print("\n" + "=" * 60)
print("TRAINING XGBOOST")
print("=" * 60)

xgb_model = xgb.XGBRegressor(
    n_estimators=3000,
    learning_rate=0.02,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    objective="reg:squarederror",
    early_stopping_rounds=200,
    n_jobs=-1,
    random_state=42,
    verbosity=1,
)

xgb_model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    verbose=50,
)

xgb_val_pred_log = xgb_model.predict(X_val)
xgb_val_pred = np.clip(np.expm1(xgb_val_pred_log), 0, None)
xgb_val_rmse = np.sqrt(mean_squared_error(y_val_orig, xgb_val_pred))
print(f"XGBoost Validation RMSE: {xgb_val_rmse:.4f}")

# ============================================================
# 13. ENSEMBLE PREDICTIONS
# ============================================================
print("\n" + "=" * 60)
print("ENSEMBLE PREDICTIONS")
print("=" * 60)

# Ensemble on validation set
val_ensemble_log = 0.5 * lgbm_val_pred_log + 0.5 * xgb_val_pred_log
val_ensemble_pred = np.clip(np.expm1(val_ensemble_log), 0, None)
final_val_rmse = mean_squared_error(y_val_orig, val_ensemble_pred, squared=False)
print(f"Ensemble Validation RMSE: {final_val_rmse:.4f}")

# Test predictions
lgbm_test_pred_log = lgbm_model.predict(X_test)
xgb_test_pred_log = xgb_model.predict(X_test)
test_ensemble_log = 0.5 * lgbm_test_pred_log + 0.5 * xgb_test_pred_log
test_preds_orig = np.clip(np.expm1(test_ensemble_log), 0, None)

print(
    f"Test predictions: mean={test_preds_orig.mean():.2f}, std={test_preds_orig.std():.2f}, "
    f"min={test_preds_orig.min():.2f}, max={test_preds_orig.max():.2f}"
)

# ============================================================
# 14. SAVE SUBMISSION
# ============================================================
submission = pd.DataFrame({"id": range(len(test_preds_orig)), "pm2.5": test_preds_orig})
submission.to_csv("./submission/submission.csv", index=False)
print(f"\nSubmission saved: {submission.shape}")
print(f"Submission head:\n{submission.head()}")

# ============================================================
# 15. FINAL SCORE
# ============================================================
score = final_val_rmse
print(f"Final Validation Score: {score}")