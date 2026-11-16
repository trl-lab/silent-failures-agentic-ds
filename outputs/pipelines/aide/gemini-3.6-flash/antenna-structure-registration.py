import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error

# 1. Load Data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "Strucht"

# Extract target and transform to log1p scale
y_train_raw = train_df[target_col].values
y_train_log = np.log1p(np.maximum(0, y_train_raw))

# Concatenate for consistent feature engineering
df_all = pd.concat(
    [train_df.drop(columns=[target_col], errors="ignore"), test_df],
    ignore_index=True,
)

# 2. Feature Engineering

# Identify column types
date_cols = []
spatial_cols = {}

for col in df_all.columns:
    if pd.api.types.is_datetime64_any_dtype(df_all[col]) or "date" in col.lower():
        date_cols.append(col)
    col_lower = col.lower()
    if "lat" in col_lower:
        spatial_cols["lat"] = col
    elif "lon" in col_lower or "long" in col_lower:
        spatial_cols["lon"] = col

# Parse Date Features
for dcol in date_cols:
    df_all[dcol] = pd.to_datetime(df_all[dcol], errors="coerce")
    df_all[f"{dcol}_year"] = df_all[dcol].dt.year
    df_all[f"{dcol}_month"] = df_all[dcol].dt.month
    df_all[f"{dcol}_day"] = df_all[dcol].dt.day
    df_all[f"{dcol}_dayofweek"] = df_all[dcol].dt.dayofweek
    df_all[f"{dcol}_dayofyear"] = df_all[dcol].dt.dayofyear
    df_all[f"{dcol}_ts"] = df_all[dcol].astype("int64") // 10**9
    df_all.drop(columns=[dcol], inplace=True)

# Spatial Transformations
if "lat" in spatial_cols and "lon" in spatial_cols:
    lat_col = spatial_cols["lat"]
    lon_col = spatial_cols["lon"]

    lat_val = df_all[lat_col].fillna(df_all[lat_col].median())
    lon_val = df_all[lon_col].fillna(df_all[lon_col].median())

    for angle in [30, 45, 60]:
        rad = np.radians(angle)
        df_all[f"rot_{angle}_x"] = lat_val * np.cos(rad) - lon_val * np.sin(rad)
        df_all[f"rot_{angle}_y"] = lat_val * np.sin(rad) + lon_val * np.cos(rad)

    lat_mid, lon_mid = lat_val.median(), lon_val.median()
    df_all["dist_centroid"] = np.sqrt(
        (lat_val - lat_mid) ** 2 + (lon_val - lon_mid) ** 2
    )

# Categorical and Numeric Handling
cat_cols = []
num_cols = []

for col in df_all.columns:
    if col in ["id"]:
        continue
    if df_all[col].dtype == "object" or df_all[col].dtype.name == "category":
        cat_cols.append(col)
        df_all[col] = df_all[col].fillna("Missing").astype(str)
        freq = df_all[col].value_counts()
        df_all[f"{col}_freq"] = df_all[col].map(freq)
        df_all[col] = df_all[col].astype("category")
    else:
        num_cols.append(col)

# Fill missing numerical values
for col in num_cols:
    if df_all[col].isnull().sum() > 0:
        df_all[col] = df_all[col].fillna(df_all[col].median())

# Split back to Train / Test
n_train = len(train_df)
X_train = df_all.iloc[:n_train].drop(columns=["id"], errors="ignore").copy()
X_test = df_all.iloc[n_train:].drop(columns=["id"], errors="ignore").copy()

cat_features_indices = [X_train.columns.get_loc(c) for c in cat_cols]

# 3. Model Training & Validation (5-Fold CV Ensemble)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

oof_preds_cb = np.zeros(n_train)
test_preds_cb = np.zeros(len(test_df))

oof_preds_lgb = np.zeros(n_train)
test_preds_lgb = np.zeros(len(test_df))

cb_params = {
    "iterations": 1500,
    "learning_rate": 0.05,
    "depth": 7,
    "loss_function": "RMSE",
    "eval_metric": "RMSE",
    "random_seed": 42,
    "verbose": 0,
}

lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "max_depth": 8,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "random_state": 42,
    "verbose": -1,
    "n_jobs": -1,
}

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train_log)):
    X_tr, y_tr = X_train.iloc[train_idx], y_train_log[train_idx]
    X_val, y_val = X_train.iloc[val_idx], y_train_log[val_idx]

    # --- CatBoost Regressor ---
    train_pool = Pool(X_tr, y_tr, cat_features=cat_features_indices)
    val_pool = Pool(X_val, y_val, cat_features=cat_features_indices)
    test_pool = Pool(X_test, cat_features=cat_features_indices)

    cb_model = CatBoostRegressor(**cb_params)
    cb_model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=100)

    val_preds_cb_log = cb_model.predict(val_pool)
    oof_preds_cb[val_idx] = np.expm1(val_preds_cb_log)
    oof_preds_cb[val_idx] = np.clip(oof_preds_cb[val_idx], 0, None)

    fold_test_preds_cb = cb_model.predict(test_pool)
    test_preds_cb += np.expm1(fold_test_preds_cb) / kf.n_splits

    # --- LightGBM Regressor ---
    train_data = lgb.Dataset(X_tr, label=y_tr)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    lgb_model = lgb.train(
        lgb_params,
        train_data,
        num_boost_round=1500,
        valid_sets=[val_data],
        callbacks=[
            lgb.early_stopping(100, verbose=False),
            lgb.log_evaluation(0),
        ],
    )

    val_preds_lgb_log = lgb_model.predict(X_val)
    oof_preds_lgb[val_idx] = np.expm1(val_preds_lgb_log)
    oof_preds_lgb[val_idx] = np.clip(oof_preds_lgb[val_idx], 0, None)

    fold_test_preds_lgb = lgb_model.predict(X_test)
    test_preds_lgb += np.expm1(fold_test_preds_lgb) / kf.n_splits

# 4. Find Optimal Blending Weight
best_w = 0.5
best_rmse = float("inf")

for w in np.linspace(0, 1, 101):
    blended_oof = w * oof_preds_cb + (1 - w) * oof_preds_lgb
    rmse = root_mean_squared_error(y_train_raw, blended_oof)
    if rmse < best_rmse:
        best_rmse = rmse
        best_w = w

oof_rmse = best_rmse
print(f"Optimal CatBoost Weight: {best_w:.2f}")
print(f"Validation OOF RMSE: {oof_rmse:.5f}")

# 5. Generate Test Predictions & Save Submission File
final_test_preds = best_w * test_preds_cb + (1 - best_w) * test_preds_lgb

os.makedirs("./working", exist_ok=True)
sub = pd.DataFrame(
    {"id": np.arange(len(test_df)), "Strucht": np.clip(final_test_preds, 0, None)}
)
sub.to_csv("./working/submission.csv", index=False)
