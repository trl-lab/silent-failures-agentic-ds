import os
import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold
import xgboost as xgb

# 1. Load Data
os.makedirs("./working", exist_ok=True)
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Drop unused ID and sub-target columns if present
drop_cols = ["instant", "casual", "registered"]
for col in drop_cols:
    if col in train_df.columns:
        train_df = train_df.drop(columns=[col])
    if col in test_df.columns:
        test_df = test_df.drop(columns=[col])

target_col = "cnt"
y = train_df[target_col].values.astype(np.float64)
X_train_raw = train_df.drop(columns=[target_col])
X_test_raw = test_df.copy()


# 2. Feature Engineering Pipeline
def engineer_features(df):
    df = df.copy()

    # Parse date and longitudinal time progression
    df["dteday"] = pd.to_datetime(df["dteday"])
    min_date = pd.to_datetime("2011-01-01")
    df["time_index"] = (df["dteday"] - min_date).dt.days * 24 + df["hr"]
    df["day_of_year"] = df["dteday"].dt.dayofyear
    df["day"] = df["dteday"].dt.day
    df["week"] = df["dteday"].dt.isocalendar().week.astype(int)
    df["quarter"] = df["dteday"].dt.quarter
    df["is_month_end"] = df["dteday"].dt.is_month_end.astype(int)

    # Cyclical encodings
    df["hr_sin"] = np.sin(2 * np.pi * df["hr"] / 24.0)
    df["hr_cos"] = np.cos(2 * np.pi * df["hr"] / 24.0)
    df["mnth_sin"] = np.sin(2 * np.pi * df["mnth"] / 12.0)
    df["mnth_cos"] = np.cos(2 * np.pi * df["mnth"] / 12.0)
    df["weekday_sin"] = np.sin(2 * np.pi * df["weekday"] / 7.0)
    df["weekday_cos"] = np.cos(2 * np.pi * df["weekday"] / 7.0)
    df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25)

    # Physical atmospheric features
    t_c = df["temp"] * 47.0 - 8.0
    at_c = df["atemp"] * 66.0 - 16.0
    hum_pct = df["hum"] * 100.0
    wind_kmh = df["windspeed"] * 67.0

    df["temp_c"] = t_c
    df["atemp_c"] = at_c
    df["hum_pct"] = hum_pct
    df["wind_kmh"] = wind_kmh
    df["temp_diff"] = at_c - t_c
    df["temp_hum_ratio"] = df["temp"] / (df["hum"] + 0.01)
    df["temp_wind_prod"] = df["temp"] * df["windspeed"]
    df["temp_hum_prod"] = df["temp"] * df["hum"]
    df["weather_severity"] = df["weathersit"] * (df["hum"] + df["windspeed"])

    # Commute, work, and diurnal indicators
    df["is_peak_commute"] = (
        (df["workingday"] == 1) & (df["hr"].isin([7, 8, 9, 17, 18, 19]))
    ).astype(int)
    df["is_work_midday"] = (
        (df["workingday"] == 1) & (df["hr"].isin([11, 12, 13, 14, 15, 16]))
    ).astype(int)
    df["is_weekend_day"] = (
        (df["workingday"] == 0) & (df["hr"].between(10, 18))
    ).astype(int)
    df["is_night"] = (df["hr"].between(0, 5)).astype(int)

    # Cross interaction categories
    df["workingday_hr"] = df["workingday"] * 24 + df["hr"]
    df["season_hr"] = df["season"] * 24 + df["hr"]
    df["weathersit_workingday"] = df["weathersit"] * 2 + df["workingday"]

    # Drop raw timestamp
    df = df.drop(columns=["dteday"])
    return df


X_train = engineer_features(X_train_raw)
X_test = engineer_features(X_test_raw)

# 3. Model Definitions & 5-Fold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

oof_xgb = np.zeros(len(X_train))
oof_lgb = np.zeros(len(X_train))
oof_cat = np.zeros(len(X_train))

test_xgb = np.zeros(len(X_test))
test_lgb = np.zeros(len(X_test))
test_cat = np.zeros(len(X_test))

xgb_params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "learning_rate": 0.035,
    "max_depth": 8,
    "min_child_weight": 4,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "n_estimators": 1200,
    "random_state": 42,
    "tree_method": "hist",
    "n_jobs": -1,
}

lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.035,
    "num_leaves": 95,
    "min_child_samples": 15,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "n_estimators": 1200,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}

cat_params = {
    "loss_function": "RMSE",
    "eval_metric": "RMSE",
    "learning_rate": 0.045,
    "depth": 8,
    "l2_leaf_reg": 3.0,
    "iterations": 1200,
    "random_seed": 42,
    "verbose": 0,
    "thread_count": -1,
}

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y)):
    X_tr, y_tr = X_train.iloc[train_idx], y[train_idx]
    X_va, y_val = X_train.iloc[val_idx], y[val_idx]

    # Model 1: XGBoost
    model_xgb = xgb.XGBRegressor(**xgb_params)
    model_xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_val)], verbose=False)
    oof_xgb[val_idx] = model_xgb.predict(X_va)
    test_xgb += model_xgb.predict(X_test) / kf.n_splits

    # Model 2: LightGBM
    model_lgb = lgb.LGBMRegressor(**lgb_params)
    model_lgb.fit(
        X_tr, y_tr, eval_set=[(X_va, y_val)], callbacks=[lgb.log_evaluation(-1)]
    )
    oof_lgb[val_idx] = model_lgb.predict(X_va)
    test_lgb += model_lgb.predict(X_test) / kf.n_splits

    # Model 3: CatBoost
    model_cat = cb.CatBoostRegressor(**cat_params)
    model_cat.fit(X_tr, y_tr, eval_set=(X_va, y_val), verbose=False)
    oof_cat[val_idx] = model_cat.predict(X_va)
    test_cat += model_cat.predict(X_test) / kf.n_splits

    fold_rmse_xgb = root_mean_squared_error(y_val, np.clip(oof_xgb[val_idx], 0, None))
    fold_rmse_lgb = root_mean_squared_error(y_val, np.clip(oof_lgb[val_idx], 0, None))
    fold_rmse_cat = root_mean_squared_error(y_val, np.clip(oof_cat[val_idx], 0, None))
    print(
        f"Fold {fold + 1} Individual RMSEs - XGB: {fold_rmse_xgb:.4f}, "
        f"LGB: {fold_rmse_lgb:.4f}, CAT: {fold_rmse_cat:.4f}"
    )

# 4. Out-of-Fold Ensemble Optimization
oof_matrix = np.column_stack([oof_xgb, oof_lgb, oof_cat])
test_matrix = np.column_stack([test_xgb, test_lgb, test_cat])


def ensemble_loss(weights):
    w = weights / np.sum(weights)
    pred = np.clip(np.dot(oof_matrix, w), 0, None)
    return root_mean_squared_error(y, pred)


init_weights = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
bounds = [(0, 1), (0, 1), (0, 1)]
opt_res = minimize(
    ensemble_loss,
    init_weights,
    method="SLSQP",
    bounds=bounds,
    constraints={"type": "eq", "fun": lambda w: 1 - sum(w)},
)
weights = opt_res.x / np.sum(opt_res.x)

oof_ensemble = np.clip(np.dot(oof_matrix, weights), 0, None)
cv_rmse = root_mean_squared_error(y, oof_ensemble)

print(f"\nOptimal Ensemble Weights (XGB, LGB, CAT): {weights.round(4)}")
print(f"Overall Cross-Validation RMSE: {cv_rmse:.4f}")

# 5. Generate Submission
test_preds = np.clip(np.dot(test_matrix, weights), 0, None)
submission = pd.DataFrame({"id": np.arange(len(test_preds)), "cnt": test_preds})
submission.to_csv("./working/submission.csv", index=False)
print("Submission successfully saved to ./working/submission.csv")
