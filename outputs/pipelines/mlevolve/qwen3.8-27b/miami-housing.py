import numpy as np
import pandas as pd
import os
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# ============================================================
# 1. LOAD DATA
# ============================================================
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")

# ============================================================
# 2. CLEANING
# ============================================================
# Handle duplicates: keep the last occurrence (most recent sale)
train = train[~train["PARCELNO"].duplicated(keep="last")].reset_index(drop=True)
print(f"Train after dedup: {train.shape}")

# Handle missing values
for col in train.select_dtypes(include=[np.number]).columns:
    if train[col].isnull().sum() > 0:
        train[col] = train[col].fillna(train[col].median())
        test[col] = test[col].fillna(train[col].median())

for col in train.select_dtypes(include=["object", "category"]).columns:
    if train[col].isnull().sum() > 0:
        train[col] = train[col].fillna(train[col].mode()[0])
        test[col] = test[col].fillna(train[col].mode()[0])

# Ensure no negative values in features that should be positive
for col in [
    "LND_SQFOOT",
    "TOT_LVG_AREA",
    "SPEC_FEAT_VAL",
    "RAIL_DIST",
    "OCEAN_DIST",
    "WATER_DIST",
    "CNTR_DIST",
    "SUBCNTR_DI",
    "HWY_DIST",
]:
    if col in train.columns:
        train[col] = train[col].clip(lower=1)
        test[col] = test[col].clip(lower=1)


# ============================================================
# 3. FEATURE ENGINEERING
# ============================================================
def engineer_features(df):
    """Apply feature engineering to a dataframe."""
    f = df.copy()

    # Log transforms for right-skewed features
    f["log_LND_SQFOOT"] = np.log1p(f["LND_SQFOOT"])
    f["log_TOT_LVG_AREA"] = np.log1p(f["TOT_LVG_AREA"])
    f["log_SPEC_FEAT_VAL"] = np.log1p(f["SPEC_FEAT_VAL"])
    f["log_RAIL_DIST"] = np.log1p(f["RAIL_DIST"])
    f["log_OCEAN_DIST"] = np.log1p(f["OCEAN_DIST"])
    f["log_WATER_DIST"] = np.log1p(f["WATER_DIST"])
    f["log_CNTR_DIST"] = np.log1p(f["CNTR_DIST"])
    f["log_SUBCNTR_DI"] = np.log1p(f["SUBCNTR_DI"])
    f["log_HWY_DIST"] = np.log1p(f["HWY_DIST"])

    # Ratio features
    f["floor_area_ratio"] = f["TOT_LVG_AREA"] / (f["LND_SQFOOT"] + 1)
    f["spec_per_sqft"] = f["SPEC_FEAT_VAL"] / (f["TOT_LVG_AREA"] + 1)
    f["land_per_sqft"] = f["LND_SQFOOT"] / (f["TOT_LVG_AREA"] + 1)

    # Interaction features
    f["quality_x_log_area"] = f["structure_quality"] * f["log_TOT_LVG_AREA"]
    f["quality_x_log_land"] = f["structure_quality"] * f["log_LND_SQFOOT"]
    f["quality_x_age"] = f["structure_quality"] * f["age"]
    f["quality_x_spec"] = f["structure_quality"] * f["log_SPEC_FEAT_VAL"]
    f["age_x_log_area"] = f["age"] * f["log_TOT_LVG_AREA"]

    # Spatial proximity features
    f["ocean_proximity"] = 1.0 / (f["log_OCEAN_DIST"] + 1)
    f["water_proximity"] = 1.0 / (f["log_WATER_DIST"] + 1)
    f["cbd_proximity"] = 1.0 / (f["log_CNTR_DIST"] + 1)
    f["subcenter_proximity"] = 1.0 / (f["log_SUBCNTR_DI"] + 1)

    # Noise penalty
    f["noise_penalty"] = 1.0 / (f["log_RAIL_DIST"] + 1) + 1.0 / (f["log_HWY_DIST"] + 1)

    # Combined spatial score
    f["spatial_score"] = (
        f["ocean_proximity"]
        + f["water_proximity"]
        + f["cbd_proximity"]
        + f["subcenter_proximity"]
        - f["noise_penalty"]
    )

    # Distance ratios
    f["ocean_to_cbd_ratio"] = f["log_OCEAN_DIST"] / (f["log_CNTR_DIST"] + 1)
    f["water_to_cbd_ratio"] = f["log_WATER_DIST"] / (f["log_CNTR_DIST"] + 1)
    f["rail_to_ocean_ratio"] = f["log_RAIL_DIST"] / (f["log_OCEAN_DIST"] + 1)

    # Age features
    f["age_squared"] = f["age"] ** 2
    f["age_cubed"] = f["age"] ** 3
    f["log_age"] = np.log1p(f["age"])
    f["is_new"] = (f["age"] <= 5).astype(int)
    f["is_old"] = (f["age"] >= 50).astype(int)

    # Cyclical encoding for month
    f["month_sin"] = np.sin(2 * np.pi * f["month_sold"] / 12)
    f["month_cos"] = np.cos(2 * np.pi * f["month_sold"] / 12)

    # Spatial coordinates: polynomial features
    f["lat_sq"] = f["LATITUDE"] ** 2
    f["lon_sq"] = f["LONGITUDE"] ** 2
    f["lat_lon_interaction"] = f["LATITUDE"] * f["LONGITUDE"]

    # Aviation noise interaction
    f["avnoise_x_quality"] = f["avno60plus"] * f["structure_quality"]
    f["avnoise_x_log_area"] = f["avno60plus"] * f["log_TOT_LVG_AREA"]

    return f


train_eng = engineer_features(train)
test_eng = engineer_features(test)

# ============================================================
# 4. CREATE TRAIN/VALIDATION SPLIT
# ============================================================
target_col = "SALE_PRC"
train_eng["log_SALE_PRC"] = np.log1p(train_eng[target_col])

train_df, val_df = train_test_split(
    train_eng,
    test_size=0.10,
    random_state=42,
    stratify=train_eng["structure_quality"].astype(int),
)

print(f"Train split: {train_df.shape}")
print(f"Validation split: {val_df.shape}")
print(f"Test: {test_eng.shape}")

# ============================================================
# 5. DEFINE FEATURE COLUMNS
# ============================================================
exclude_cols = ["PARCELNO", "SALE_PRC", "log_SALE_PRC"]
feature_cols = [c for c in train_eng.columns if c not in exclude_cols]
print(f"Number of features: {len(feature_cols)}")

# Prepare matrices
X_train = train_df[feature_cols].values
X_val = val_df[feature_cols].values
X_test = test_eng[feature_cols].values

y_train_log = train_df["log_SALE_PRC"].values
y_val_log = val_df["log_SALE_PRC"].values
y_val_orig = val_df["SALE_PRC"].values

# ============================================================
# 6. TRAIN LIGHTGBM (Huber loss for robustness)
# ============================================================
print("\n=== Training LightGBM ===")
lgb_params = {
    "objective": "huber",
    "alpha": 100000,
    "metric": "rmse",
    "boosting_type": "gbdt",
    "num_leaves": 127,
    "max_depth": -1,
    "learning_rate": 0.03,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 20,
    "min_child_weight": 1e-3,
    "lambda_l1": 10.0,
    "lambda_l2": 100.0,
    "max_bin": 255,
    "verbosity": -1,
    "n_estimators": 5000,
    "random_state": 42,
    "n_jobs": -1,
}

lgb_model = lgb.LGBMRegressor(**lgb_params)
lgb_model.fit(
    X_train,
    y_train_log,
    eval_set=[(X_val, y_val_log)],
    callbacks=[
        lgb.early_stopping(100, verbose=False),
        lgb.log_evaluation(-1),
    ],
)

val_pred_log_lgb = lgb_model.predict(X_val, num_iteration=lgb_model.best_iteration_)
val_pred_lgb = np.expm1(val_pred_log_lgb)
rmse_lgb = np.sqrt(mean_squared_error(y_val_orig, val_pred_lgb))
print(f"LightGBM - Best iter: {lgb_model.best_iteration_}, Val RMSE: {rmse_lgb:.2f}")

# ============================================================
# 7. TRAIN XGBOOST (squared error)
# ============================================================
print("\n=== Training XGBoost ===")
xgb_params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "max_depth": 8,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "min_child_weight": 5,
    "reg_alpha": 5.0,
    "reg_lambda": 50.0,
    "n_estimators": 5000,
    "random_state": 42,
    "tree_method": "hist",
    "verbosity": 0,
    "early_stopping_rounds": 100,
}

xgb_model = xgb.XGBRegressor(**xgb_params)
xgb_model.fit(
    X_train,
    y_train_log,
    eval_set=[(X_val, y_val_log)],
    verbose=False,
)

val_pred_log_xgb = xgb_model.predict(
    X_val, iteration_range=(0, xgb_model.best_iteration)
)
val_pred_xgb = np.expm1(val_pred_log_xgb)
rmse_xgb = np.sqrt(mean_squared_error(y_val_orig, val_pred_xgb))
print(f"XGBoost - Best iter: {xgb_model.best_iteration}, Val RMSE: {rmse_xgb:.2f}")

# ============================================================
# 8. BASE ENSEMBLE (inverse-RMSE weighted)
# ============================================================
w_lgb = (1.0 / rmse_lgb) / (1.0 / rmse_lgb + 1.0 / rmse_xgb)
w_xgb = (1.0 / rmse_xgb) / (1.0 / rmse_lgb + 1.0 / rmse_xgb)

val_pred_base = w_lgb * val_pred_lgb + w_xgb * val_pred_xgb
rmse_base = np.sqrt(mean_squared_error(y_val_orig, val_pred_base))
print(f"\nBase Ensemble (LGB={w_lgb:.3f}, XGB={w_xgb:.3f}) - Val RMSE: {rmse_base:.2f}")

# ============================================================
# 9. RESIDUAL CORRECTION MODEL
# ============================================================
print("\n=== Training Residual Correction Model ===")

# Compute base predictions on train
train_pred_log_lgb = lgb_model.predict(X_train, num_iteration=lgb_model.best_iteration_)
train_pred_log_xgb = xgb_model.predict(
    X_train, iteration_range=(0, xgb_model.best_iteration)
)
base_pred_train = w_lgb * np.expm1(train_pred_log_lgb) + w_xgb * np.expm1(
    train_pred_log_xgb
)

# Residual target
y_train_resid = train_df["SALE_PRC"].values - base_pred_train
y_val_resid = y_val_orig - val_pred_base

# Build residual model features
X_train_resid = np.column_stack([X_train, base_pred_train])
X_val_resid = np.column_stack([X_val, val_pred_base])

resid_params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "num_leaves": 63,
    "max_depth": -1,
    "learning_rate": 0.05,
    "feature_fraction": 0.6,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 30,
    "min_child_weight": 1e-3,
    "lambda_l1": 50.0,
    "lambda_l2": 200.0,
    "max_bin": 255,
    "verbosity": -1,
    "n_estimators": 3000,
    "random_state": 42,
    "n_jobs": -1,
}

resid_model = lgb.LGBMRegressor(**resid_params)
resid_model.fit(
    X_train_resid,
    y_train_resid,
    eval_set=[(X_val_resid, y_val_resid)],
    callbacks=[
        lgb.early_stopping(80, verbose=False),
        lgb.log_evaluation(-1),
    ],
)

val_resid_pred = resid_model.predict(X_val_resid)
val_pred_corrected = val_pred_base + val_resid_pred
rmse_corrected = np.sqrt(mean_squared_error(y_val_orig, val_pred_corrected))
print(
    f"Residual Correction - Best iter: {resid_model.best_iteration_}, Val RMSE: {rmse_corrected:.2f}"
)

# ============================================================
# 10. SELECT BEST APPROACH
# ============================================================
candidates = {
    "lgb": rmse_lgb,
    "xgb": rmse_xgb,
    "base_ensemble": rmse_base,
    "corrected_ensemble": rmse_corrected,
}

best_name = min(candidates, key=lambda k: candidates[k])
best_rmse = candidates[best_name]
print(f"\nBest approach: {best_name} (RMSE: {best_rmse:.2f})")

# ============================================================
# 11. GENERATE TEST PREDICTIONS
# ============================================================
test_pred_log_lgb = lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration_)
test_pred_log_xgb = xgb_model.predict(
    X_test, iteration_range=(0, xgb_model.best_iteration)
)
test_pred_lgb = np.expm1(test_pred_log_lgb)
test_pred_xgb = np.expm1(test_pred_log_xgb)
test_pred_base = w_lgb * test_pred_lgb + w_xgb * test_pred_xgb

if best_name == "lgb":
    test_pred_final = test_pred_lgb
elif best_name == "xgb":
    test_pred_final = test_pred_xgb
elif best_name == "base_ensemble":
    test_pred_final = test_pred_base
else:
    X_test_resid = np.column_stack([X_test, test_pred_base])
    test_resid_pred = resid_model.predict(X_test_resid)
    test_pred_final = test_pred_base + test_resid_pred

# Clip to positive values
test_pred_final = np.clip(test_pred_final, 1, None)

# ============================================================
# 12. SAVE SUBMISSION
# ============================================================
os.makedirs("./submission", exist_ok=True)
submission = pd.DataFrame(
    {"id": range(len(test_pred_final)), "SALE_PRC": test_pred_final}
)
submission.to_csv("./submission/submission.csv", index=False)
print(f"\nSubmission saved: {submission.shape}")
print(submission.head())

# ============================================================
# FINAL VALIDATION SCORE
# ============================================================
score = best_rmse
print(f"Final Validation Score: {score}")
