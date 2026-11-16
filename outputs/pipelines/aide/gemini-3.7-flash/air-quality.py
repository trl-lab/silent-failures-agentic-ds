import os
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from scipy.optimize import minimize
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold

# 1. Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "CO(GT)"


# 2. Preprocessing & Feature Engineering Function
def preprocess_features(df):
    data = df.copy()

    # Replace UCI Air Quality missing sentinel (-200) with NaN
    numeric_cols = [c for c in data.columns if c not in ["Date", "Time"]]
    for c in numeric_cols:
        data[c] = data[c].replace(-200, np.nan)
        data[c] = pd.to_numeric(data[c], errors="coerce")

    # Process Date and Time
    if "Date" in data.columns and "Time" in data.columns:
        time_str = data["Time"].astype(str).str.replace(".", ":", regex=False)
        datetime_str = data["Date"].astype(str) + " " + time_str
        dt = pd.to_datetime(datetime_str, errors="coerce")

        data["hour"] = dt.dt.hour
        data["dayofweek"] = dt.dt.dayofweek
        data["month"] = dt.dt.month
        data["day"] = dt.dt.day
        data["is_weekend"] = data["dayofweek"].isin([5, 6]).astype(int)

        # Cyclical temporal features
        data["hour_sin"] = np.sin(2 * np.pi * data["hour"] / 24.0)
        data["hour_cos"] = np.cos(2 * np.pi * data["hour"] / 24.0)
        data["month_sin"] = np.sin(2 * np.pi * data["month"] / 12.0)
        data["month_cos"] = np.cos(2 * np.pi * data["month"] / 12.0)
        data["dow_sin"] = np.sin(2 * np.pi * data["dayofweek"] / 7.0)
        data["dow_cos"] = np.cos(2 * np.pi * data["dayofweek"] / 7.0)

    # Sensor ratios and cross-features
    if "PT08.S1(CO)" in data.columns and "PT08.S2(NMHC)" in data.columns:
        data["ratio_s1_s2"] = data["PT08.S1(CO)"] / (data["PT08.S2(NMHC)"] + 1e-5)
    if "PT08.S1(CO)" in data.columns and "PT08.S3(NOx)" in data.columns:
        data["ratio_s1_s3"] = data["PT08.S1(CO)"] / (data["PT08.S3(NOx)"] + 1e-5)
    if "PT08.S2(NMHC)" in data.columns and "PT08.S3(NOx)" in data.columns:
        data["ratio_s2_s3"] = data["PT08.S2(NMHC)"] / (data["PT08.S3(NOx)"] + 1e-5)
    if "C6H6(GT)" in data.columns and "PT08.S1(CO)" in data.columns:
        data["c6h6_s1_prod"] = data["C6H6(GT)"] * data["PT08.S1(CO)"]
    if "NOx(GT)" in data.columns and "NO2(GT)" in data.columns:
        data["nox_no2_ratio"] = data["NOx(GT)"] / (data["NO2(GT)"] + 1e-5)
        data["nox_no2_diff"] = data["NOx(GT)"] - data["NO2(GT)"]
    if "T" in data.columns and "RH" in data.columns:
        data["temp_rh_interaction"] = data["T"] * data["RH"]

    # Drop raw date/time columns
    drop_cols = ["Date", "Time"]
    data = data.drop(columns=[c for c in drop_cols if c in data.columns])

    return data


# Preprocess datasets
train_processed = preprocess_features(train_df)
test_processed = preprocess_features(test_df)

# Filter out rows with missing target in the training set
train_clean = train_processed.dropna(subset=[target_col]).reset_index(drop=True)

feature_cols = [c for c in train_clean.columns if c != target_col]
X = train_clean[feature_cols]
y = train_clean[target_col]
X_test = test_processed[feature_cols]

# 3. K-Fold Cross Validation Setup
kf = KFold(n_splits=5, shuffle=True, random_state=42)

oof_cat = np.zeros(len(train_clean))
oof_lgb = np.zeros(len(train_clean))
oof_xgb = np.zeros(len(train_clean))

test_cat = np.zeros(len(test_df))
test_lgb = np.zeros(len(test_df))
test_xgb = np.zeros(len(test_df))

cat_params = {
    "iterations": 1500,
    "learning_rate": 0.03,
    "depth": 6,
    "loss_function": "RMSE",
    "eval_metric": "RMSE",
    "random_seed": 42,
    "verbose": False,
}

lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.03,
    "num_leaves": 31,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "random_state": 42,
    "verbose": -1,
    "n_estimators": 1500,
}

xgb_params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "learning_rate": 0.03,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "n_estimators": 1500,
}

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

    # CatBoost
    cat_model = CatBoostRegressor(**cat_params)
    cat_model.fit(
        X_train,
        y_train,
        eval_set=(X_val, y_val),
        early_stopping_rounds=50,
        verbose=False,
    )
    oof_cat[val_idx] = cat_model.predict(X_val)
    test_cat += cat_model.predict(X_test) / kf.n_splits

    # LightGBM
    lgb_model = lgb.LGBMRegressor(**lgb_params)
    lgb_model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )
    oof_lgb[val_idx] = lgb_model.predict(X_val)
    test_lgb += lgb_model.predict(X_test) / kf.n_splits

    # XGBoost
    xgb_model = xgb.XGBRegressor(**xgb_params)
    xgb_model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    oof_xgb[val_idx] = xgb_model.predict(X_val)
    test_xgb += xgb_model.predict(X_test) / kf.n_splits

# 4. Find optimal blending weights using constrained optimization
oof_matrix = np.column_stack([oof_cat, oof_lgb, oof_xgb])


def loss_func(weights):
    weights = np.array(weights)
    weights = weights / np.sum(weights)
    blend_pred = oof_matrix @ weights
    return root_mean_squared_error(y, blend_pred)


init_weights = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
bounds = [(0, 1), (0, 1), (0, 1)]
opt_res = minimize(
    loss_func,
    init_weights,
    method="SLSQP",
    bounds=bounds,
    constraints={"type": "eq", "fun": lambda w: sum(w) - 1.0},
)

best_weights = opt_res.x / np.sum(opt_res.x)
oof_blend = oof_matrix @ best_weights
test_preds = (
    best_weights[0] * test_cat + best_weights[1] * test_lgb + best_weights[2] * test_xgb
)

# 5. Evaluation metric
overall_rmse = root_mean_squared_error(y, oof_blend)
print(f"Validation Metric (5-Fold CV Blended RMSE): {overall_rmse:.5f}")

# 6. Save submission
os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame({"id": np.arange(len(test_preds)), "CO(GT)": test_preds})
submission.to_csv("./working/submission.csv", index=False)
