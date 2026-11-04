import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error
import xgboost as xgb

# Ensure output directory exists
os.makedirs("./working", exist_ok=True)

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "Res_capacityM3"
y = train_df[target_col].values
X_train = train_df.drop(columns=[target_col]).copy()
X_test = test_df.copy()

# Combine for uniform preprocessing
n_train = len(X_train)
df_all = pd.concat([X_train, X_test], ignore_index=True)

# Identify feature types
cat_cols = df_all.select_dtypes(include=["object", "category"]).columns.tolist()
num_cols = df_all.select_dtypes(include=[np.number]).columns.tolist()

# Spatial feature engineering if latitude and longitude exist
lat_col = next((c for c in num_cols if "lat" in c.lower()), None)
lon_col = next((c for c in num_cols if "lon" in c.lower()), None)

if lat_col and lon_col:
    lat_rad = np.radians(df_all[lat_col].fillna(0))
    lon_rad = np.radians(df_all[lon_col].fillna(0))
    df_all["coord_x"] = np.cos(lat_rad) * np.cos(lon_rad)
    df_all["coord_y"] = np.cos(lat_rad) * np.sin(lon_rad)
    df_all["coord_z"] = np.sin(lat_rad)

# Volumetric & geometric interaction features
area_col = next(
    (c for c in num_cols if "area" in c.lower() or "sqkm" in c.lower()), None
)
height_col = next(
    (c for c in num_cols if "height" in c.lower() or "hgt" in c.lower()), None
)
catchment_col = next(
    (c for c in num_cols if "catch" in c.lower() or "basin" in c.lower()), None
)

if area_col and height_col:
    df_all["est_geom_vol"] = df_all[area_col].fillna(0) * df_all[height_col].fillna(0)
    df_all["height_area_ratio"] = df_all[height_col] / (df_all[area_col].abs() + 1e-5)

if catchment_col and area_col:
    df_all["catchment_area_ratio"] = df_all[catchment_col] / (
        df_all[area_col].abs() + 1e-5
    )

# Frequency encoding for categoricals
for col in cat_cols:
    freq = df_all[col].value_counts(dropna=False, normalize=True)
    df_all[f"{col}_freq"] = df_all[col].map(freq).astype(float)
    df_all[col] = df_all[col].astype("category").cat.codes

# Separate back to train and test
X_train_proc = df_all.iloc[:n_train].copy()
X_test_proc = df_all.iloc[n_train:].copy()

# Setup 5-Fold Cross Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(n_train)
test_preds = np.zeros(len(X_test_proc))

xgb_params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "n_estimators": 1500,
    "early_stopping_rounds": 50,
}

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_proc, y)):
    X_tr, y_tr = X_train_proc.iloc[train_idx], y[train_idx]
    X_va, y_va = X_train_proc.iloc[val_idx], y[val_idx]

    model = xgb.XGBRegressor(**xgb_params)
    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        verbose=False,
    )

    val_pred = model.predict(X_va)
    oof_preds[val_idx] = val_pred
    test_preds += model.predict(X_test_proc) / kf.n_splits

# Ensure non-negative predictions if capacity cannot be negative
oof_preds_clipped = np.clip(oof_preds, 0, None)
test_preds_clipped = np.clip(test_preds, 0, None)

# Evaluation Metric
cv_rmse = root_mean_squared_error(y, oof_preds_clipped)
print(f"5-Fold CV Out-of-Fold RMSE: {cv_rmse:.5f}")

# Generate Submission File
submission = pd.DataFrame(
    {"id": np.arange(len(test_preds_clipped)), "Res_capacityM3": test_preds_clipped}
)
submission.to_csv("./working/submission.csv", index=False)
print(
    "Submission saved successfully to ./working/submission.csv with shape:",
    submission.shape,
)
