import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
import lightgbm as lgb
from catboost import CatBoostRegressor
from scipy.optimize import minimize

# Load datasets
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

# Filter train set for missing/invalid target values
valid_target_mask = (
    train["CO(GT)"].notnull() & (train["CO(GT)"] != -200) & (train["CO(GT)"] > 0)
)
train_valid = train[valid_target_mask].copy().reset_index(drop=True)

# Assign split indicators and original order
train_valid["is_train"] = 1
train_valid["orig_idx"] = np.arange(len(train_valid))
test["is_train"] = 0
test["orig_idx"] = np.arange(len(test))

# Combine datasets for consistent time-series feature engineering
df_full = pd.concat([train_valid, test], ignore_index=True)


# Parse Date and Time columns into a single datetime
def parse_datetime(df):
    date_s = df["Date"].astype(str)
    time_s = df["Time"].astype(str).str.replace(".", ":", regex=False)
    return pd.to_datetime(date_s + " " + time_s, format="mixed", dayfirst=True)


df_full["datetime"] = parse_datetime(df_full)

# Replace missing value code -200 with NaN across all continuous features
num_cols = [
    c
    for c in df_full.columns
    if c not in ["CO(GT)", "Date", "Time", "datetime", "is_train", "orig_idx", "id"]
]
for c in num_cols:
    df_full[c] = pd.to_numeric(df_full[c], errors="coerce")
    df_full[c] = df_full[c].replace([-200, -200.0], np.nan)

# Sort chronologically for continuous temporal feature calculations
df_full = df_full.sort_values("datetime").reset_index(drop=True)

# 1. Granular Temporal Features
df_full["hour"] = df_full["datetime"].dt.hour
df_full["dayofweek"] = df_full["datetime"].dt.dayofweek
df_full["month"] = df_full["datetime"].dt.month
df_full["day"] = df_full["datetime"].dt.day
df_full["is_weekend"] = df_full["dayofweek"].isin([5, 6]).astype(int)

df_full["sin_hour"] = np.sin(2 * np.pi * df_full["hour"] / 24.0)
df_full["cos_hour"] = np.cos(2 * np.pi * df_full["hour"] / 24.0)
df_full["sin_month"] = np.sin(2 * np.pi * df_full["month"] / 12.0)
df_full["cos_month"] = np.cos(2 * np.pi * df_full["month"] / 12.0)
df_full["sin_dow"] = np.sin(2 * np.pi * df_full["dayofweek"] / 7.0)
df_full["cos_dow"] = np.cos(2 * np.pi * df_full["dayofweek"] / 7.0)

# 2. Atmospheric & Thermodynamic Interactions
T = df_full["T"]
RH = df_full["RH"]
AH = df_full["AH"]

df_full["dew_point"] = T - ((100.0 - RH) / 5.0)
df_full["temp_rh_product"] = T * RH
df_full["ah_rh_product"] = AH * RH
df_full["ah_temp_product"] = AH * T
df_full["temp_rh_ratio"] = T / (RH + 1e-3)
df_full["vp_est"] = 6.112 * np.exp((17.67 * T) / (T + 243.5)) * (RH / 100.0)

# 3. Multi-Sensor Cross-Ratios and Aggregations
s_cols = ["PT08.S1(CO)", "PT08.S2(NMHC)", "PT08.S3(NOx)", "PT08.S4(NO2)", "PT08.S5(O3)"]
for i in range(len(s_cols)):
    for j in range(i + 1, len(s_cols)):
        col1, col2 = s_cols[i], s_cols[j]
        tag1 = col1.split("(")[0]
        tag2 = col2.split("(")[0]
        df_full[f"{tag1}_{tag2}_ratio"] = df_full[col1] / (df_full[col2] + 1e-3)
        df_full[f"{tag1}_{tag2}_mult"] = df_full[col1] * df_full[col2]

df_full["sensor_mean"] = df_full[s_cols].mean(axis=1)
df_full["sensor_std"] = df_full[s_cols].std(axis=1)
df_full["sensor_max"] = df_full[s_cols].max(axis=1)
df_full["sensor_min"] = df_full[s_cols].min(axis=1)

# Chemical interactions
if "C6H6(GT)" in df_full.columns and "PT08.S2(NMHC)" in df_full.columns:
    df_full["c6h6_s2_ratio"] = df_full["C6H6(GT)"] / (df_full["PT08.S2(NMHC)"] + 1e-3)
    df_full["log_c6h6"] = np.log1p(np.maximum(0, df_full["C6H6(GT)"]))
if "NOx(GT)" in df_full.columns and "PT08.S3(NOx)" in df_full.columns:
    df_full["nox_s3_ratio"] = df_full["NOx(GT)"] / (df_full["PT08.S3(NOx)"] + 1e-3)
    df_full["log_nox"] = np.log1p(np.maximum(0, df_full["NOx(GT)"]))
if "NO2(GT)" in df_full.columns and "PT08.S4(NO2)" in df_full.columns:
    df_full["no2_s4_ratio"] = df_full["NO2(GT)"] / (df_full["PT08.S4(NO2)"] + 1e-3)
    df_full["log_no2"] = np.log1p(np.maximum(0, df_full["NO2(GT)"]))

# 4. Lag, Lead, and Rolling Window Statistics
base_time_cols = [
    c
    for c in [
        "PT08.S1(CO)",
        "PT08.S2(NMHC)",
        "PT08.S3(NOx)",
        "PT08.S4(NO2)",
        "PT08.S5(O3)",
        "C6H6(GT)",
        "NOx(GT)",
        "NO2(GT)",
        "T",
        "RH",
        "AH",
    ]
    if c in df_full.columns
]

lags = [-3, -2, -1, 1, 2, 3]
for col in base_time_cols:
    for shift in lags:
        df_full[f"{col}_shift_{shift}"] = df_full[col].shift(shift)
    df_full[f"{col}_diff_lag1"] = df_full[col] - df_full[col].shift(1)
    df_full[f"{col}_diff_lead1"] = df_full[col].shift(-1) - df_full[col]
    for w in [3, 6, 12]:
        rolling_obj = df_full[col].rolling(window=w, min_periods=1)
        df_full[f"{col}_roll_mean_{w}"] = rolling_obj.mean()
        df_full[f"{col}_roll_std_{w}"] = rolling_obj.std()

# 5. Group Aggregations
for g_col in ["hour", "dayofweek"]:
    for feat in ["PT08.S1(CO)", "PT08.S2(NMHC)", "C6H6(GT)"]:
        if feat in df_full.columns:
            df_full[f"{feat}_gmean_{g_col}"] = df_full.groupby(g_col)[feat].transform(
                "mean"
            )

# Clean infinite values
df_full = df_full.replace([np.inf, -np.inf], np.nan)

# Separate train and test sets and preserve original row index order
train_df = (
    df_full[df_full["is_train"] == 1].sort_values("orig_idx").reset_index(drop=True)
)
test_df = (
    df_full[df_full["is_train"] == 0].sort_values("orig_idx").reset_index(drop=True)
)

drop_cols = ["CO(GT)", "Date", "Time", "datetime", "is_train", "orig_idx", "id"]
features = [c for c in train_df.columns if c not in drop_cols]

X = train_df[features]
y = train_df["CO(GT)"].values
X_test = test_df[features]

# Prepare imputed datasets for ExtraTrees
imputer = SimpleImputer(strategy="median")
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=features)
X_test_imputed = pd.DataFrame(imputer.transform(X_test), columns=features)

# 5-Fold Cross-Validation Framework
kf = KFold(n_splits=5, shuffle=True, random_state=42)

oof_cat = np.zeros(len(X))
oof_lgb = np.zeros(len(X))
oof_et = np.zeros(len(X))

test_cat_preds = np.zeros(len(X_test))
test_lgb_preds = np.zeros(len(X_test))
test_et_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_train, y_train = X.iloc[train_idx], y[train_idx]
    X_val, y_val = X.iloc[val_idx], y[val_idx]

    # Model 1: CatBoost Regressor
    cat = CatBoostRegressor(
        iterations=1500,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=3,
        eval_metric="RMSE",
        random_seed=42 + fold,
        verbose=0,
    )
    cat.fit(
        X_train,
        y_train,
        eval_set=(X_val, y_val),
        early_stopping_rounds=100,
        verbose=False,
    )
    oof_cat[val_idx] = cat.predict(X_val)
    test_cat_preds += cat.predict(X_test) / 5.0

    # Model 2: LightGBM Regressor
    lgbm = lgb.LGBMRegressor(
        n_estimators=1500,
        learning_rate=0.03,
        max_depth=6,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        verbosity=-1,
    )
    lgbm.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )
    oof_lgb[val_idx] = lgbm.predict(X_val)
    test_lgb_preds += lgbm.predict(X_test) / 5.0

    # Model 3: ExtraTrees Regressor
    X_tr_imp, X_v_imp = X_imputed.iloc[train_idx], X_imputed.iloc[val_idx]
    et = ExtraTreesRegressor(
        n_estimators=250, max_depth=16, random_state=42 + fold, n_jobs=-1
    )
    et.fit(X_tr_imp, y_train)
    oof_et[val_idx] = et.predict(X_v_imp)
    test_et_preds += et.predict(X_test_imputed) / 5.0


# Optimize ensemble blending weights
def loss_func(weights):
    w1, w2, w3 = weights
    pred = w1 * oof_cat + w2 * oof_lgb + w3 * oof_et
    return np.sqrt(np.mean((y - pred) ** 2))


res = minimize(
    loss_func,
    [0.45, 0.45, 0.10],
    bounds=[(0, 1), (0, 1), (0, 1)],
    constraints={"type": "eq", "fun": lambda w: sum(w) - 1.0},
)
w1, w2, w3 = res.x

oof_blend = w1 * oof_cat + w2 * oof_lgb + w3 * oof_et
final_rmse = np.sqrt(np.mean((y - oof_blend) ** 2))

test_preds = w1 * test_cat_preds + w2 * test_lgb_preds + w3 * test_et_preds

# Save predictions
os.makedirs("./working", exist_ok=True)
sub = pd.DataFrame({"id": np.arange(len(test_preds)), "CO(GT)": test_preds})
sub.to_csv("./working/submission.csv", index=False)

print(f"Validation RMSE: {final_rmse:.4f}")
