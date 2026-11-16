import os
import re
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error
import lightgbm as lgb
import xgboost as xgb

warnings.filterwarnings("ignore")

# 1. Load Data
train_path = "./input/train.parquet"
test_path = "./input/test.parquet"

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = "Res_capacityM3"
if target_col not in train_df.columns:
    for c in train_df.columns:
        if "capacity" in c.lower() or "res_" in c.lower():
            target_col = c
            break

y = pd.to_numeric(train_df[target_col], errors="coerce").values
X_train_raw = train_df.drop(columns=[target_col], errors="ignore").copy()
X_test_raw = test_df.drop(columns=[target_col], errors="ignore").copy()


# 2. Feature Engineering Function
def create_features(df_tr, df_te):
    tr = df_tr.copy()
    te = df_te.copy()

    # Coerce columns to numeric where possible
    for col in tr.columns:
        if tr[col].dtype == object or str(tr[col].dtype).startswith("string"):
            try_tr = pd.to_numeric(tr[col], errors="coerce")
            try_te = pd.to_numeric(te[col], errors="coerce")
            # If at least some non-null values converted, accept conversion
            if try_tr.notna().sum() > 0:
                tr[col] = try_tr
                te[col] = try_te

    # Drop non-feature ID/URL columns
    drop_cols = [
        c
        for c in tr.columns
        if c.lower() in ["id", "url", "index", "dam_id", "grand_id"]
    ]
    tr = tr.drop(columns=drop_cols, errors="ignore")
    te = te.drop(columns=drop_cols, errors="ignore")

    # Dynamic detection of lat/lon
    lat_cols = [c for c in tr.columns if "lat" in c.lower()]
    lon_cols = [c for c in tr.columns if "lon" in c.lower()]

    if lat_cols and lon_cols:
        lat_col, lon_col = lat_cols[0], lon_cols[0]
        for df in [tr, te]:
            lat_vals = pd.to_numeric(df[lat_col], errors="coerce").fillna(0)
            lon_vals = pd.to_numeric(df[lon_col], errors="coerce").fillna(0)
            rad_lat = np.radians(lat_vals)
            rad_lon = np.radians(lon_vals)
            df["coord_x"] = np.cos(rad_lat) * np.cos(rad_lon)
            df["coord_y"] = np.cos(rad_lat) * np.sin(rad_lon)
            df["coord_z"] = np.sin(rad_lat)

    # Physical metric ratios and interactions
    num_cols = tr.select_dtypes(include=[np.number]).columns.tolist()

    area_cols = [c for c in num_cols if "area" in c.lower()]
    height_cols = [c for c in num_cols if "hgt" in c.lower() or "height" in c.lower()]
    catch_cols = [c for c in num_cols if "catch" in c.lower()]
    len_cols = [c for c in num_cols if "len" in c.lower()]

    for df in [tr, te]:
        if area_cols and height_cols:
            df["area_x_height"] = pd.to_numeric(
                df[area_cols[0]], errors="coerce"
            ).fillna(0) * pd.to_numeric(df[height_cols[0]], errors="coerce").fillna(0)
        if catch_cols and area_cols:
            df["catch_to_area"] = pd.to_numeric(
                df[catch_cols[0]], errors="coerce"
            ).fillna(0) / (
                pd.to_numeric(df[area_cols[0]], errors="coerce").fillna(0) + 1e-5
            )
        if height_cols and len_cols:
            df["height_to_len"] = pd.to_numeric(
                df[height_cols[0]], errors="coerce"
            ).fillna(0) / (
                pd.to_numeric(df[len_cols[0]], errors="coerce").fillna(0) + 1e-5
            )

        # Log1p transforms for positive continuous physical columns
        for c in num_cols:
            val = pd.to_numeric(df[c], errors="coerce").fillna(0)
            if (val.min() >= 0) and ("year" not in c.lower()) and ("coord" not in c):
                df[f"{c}_log1p"] = np.log1p(val.clip(lower=0))

    # Categorical encodings
    cat_cols = tr.select_dtypes(include=["object", "category"]).columns.tolist()
    for col in cat_cols:
        freq = pd.concat([tr[col], te[col]]).value_counts().to_dict()
        tr[f"{col}_freq"] = tr[col].map(freq).fillna(0)
        te[f"{col}_freq"] = te[col].map(freq).fillna(0)

        tr[col] = tr[col].astype(str).astype("category").cat.codes
        te[col] = te[col].astype(str).astype("category").cat.codes

    # Sanitize feature column names
    clean_cols = [re.sub(r"[^\w\s]", "_", str(c)) for c in tr.columns]
    tr.columns = clean_cols
    te.columns = clean_cols

    return tr, te


X_tr, X_te = create_features(X_train_raw, X_test_raw)

# Align columns
common_cols = [c for c in X_tr.columns if c in X_te.columns]
X_tr = X_tr[common_cols]
X_te = X_te[common_cols]

# 3. Cross-Validation and Model Training
kf = KFold(n_splits=5, shuffle=True, random_state=42)

oof_lgb = np.zeros(len(y))
oof_xgb = np.zeros(len(y))
test_lgb = np.zeros(len(X_te))
test_xgb = np.zeros(len(X_te))

for fold, (train_idx, val_idx) in enumerate(kf.split(X_tr, y)):
    X_train_f, y_train_f = X_tr.iloc[train_idx], y[train_idx]
    X_val_f, y_val_f = X_tr.iloc[val_idx], y[val_idx]

    # LightGBM Model
    model_lgb = lgb.LGBMRegressor(
        objective="regression",
        metric="rmse",
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        verbosity=-1,
    )

    model_lgb.fit(
        X_train_f,
        y_train_f,
        eval_set=[(X_val_f, y_val_f)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
    )

    oof_lgb[val_idx] = model_lgb.predict(X_val_f)
    test_lgb += model_lgb.predict(X_te) / kf.n_splits

    # XGBoost Model
    model_xgb = xgb.XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        early_stopping_rounds=50,
        tree_method="hist",
    )

    model_xgb.fit(X_train_f, y_train_f, eval_set=[(X_val_f, y_val_f)], verbose=False)

    oof_xgb[val_idx] = model_xgb.predict(X_val_f)
    test_xgb += model_xgb.predict(X_te) / kf.n_splits

# Ensure non-negative predictions
oof_lgb = np.clip(oof_lgb, a_min=0, a_max=None)
oof_xgb = np.clip(oof_xgb, a_min=0, a_max=None)
test_lgb = np.clip(test_lgb, a_min=0, a_max=None)
test_xgb = np.clip(test_xgb, a_min=0, a_max=None)

# Blend Predictions
oof_blend = 0.5 * oof_lgb + 0.5 * oof_xgb
test_blend = 0.5 * test_lgb + 0.5 * test_xgb

if np.isnan(test_blend).any():
    test_blend = pd.Series(test_blend).fillna(np.nanmedian(y)).values

# 4. Evaluation Score
cv_rmse = root_mean_squared_error(y, oof_blend)
print(f"5-Fold CV RMSE: {cv_rmse:.4f}")

# 5. Create Submission
os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame(
    {
        "id": (
            test_df["id"].values if "id" in test_df.columns else np.arange(len(test_df))
        ),
        "Res_capacityM3": test_blend,
    }
)
submission.to_csv("./working/submission.csv", index=False)
