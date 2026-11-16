import os
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import xgboost as xgb

warnings.filterwarnings("ignore")

# Create directories
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# ============================================================
# 1. LOAD DATA
# ============================================================
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")
print(f"Train columns: {list(train.columns)}")
print(f"Target stats:\n{train['gisacres'].describe()}")
print(f"Target log stats:\n{np.log1p(train['gisacres']).describe()}")

# ============================================================
# 2. CREATE TRAIN/VAL SPLIT (BEFORE any feature engineering)
# ============================================================
train["log_gisacres"] = np.log1p(train["gisacres"])

train_data, val_data = train_test_split(
    train, test_size=0.2, random_state=42, stratify=None
)

print(f"Train split shape: {train_data.shape}")
print(f"Val split shape: {val_data.shape}")

# ============================================================
# 3. FEATURE ENGINEERING
# ============================================================


def parse_geometry_features(df, col_name):
    """Extract features from a geometry/WKT column."""
    features = {}
    if col_name not in df.columns:
        return features

    wkt_col = df[col_name]

    try:
        from shapely import wkt as shapely_wkt

        geometries = wkt_col.apply(
            lambda x: shapely_wkt.loads(x) if pd.notna(x) else None
        )

        features["geom_centroid_x"] = geometries.apply(
            lambda g: g.centroid.x if g is not None else np.nan
        )
        features["geom_centroid_y"] = geometries.apply(
            lambda g: g.centroid.y if g is not None else np.nan
        )
        features["geom_area"] = geometries.apply(
            lambda g: g.area if g is not None else np.nan
        )
        features["geom_perimeter"] = geometries.apply(
            lambda g: g.length if g is not None else np.nan
        )
        features["geom_bbox_minx"] = geometries.apply(
            lambda g: g.bounds[0] if g is not None else np.nan
        )
        features["geom_bbox_miny"] = geometries.apply(
            lambda g: g.bounds[1] if g is not None else np.nan
        )
        features["geom_bbox_maxx"] = geometries.apply(
            lambda g: g.bounds[2] if g is not None else np.nan
        )
        features["geom_bbox_maxy"] = geometries.apply(
            lambda g: g.bounds[3] if g is not None else np.nan
        )
        features["geom_bbox_width"] = (
            features["geom_bbox_maxx"] - features["geom_bbox_minx"]
        )
        features["geom_bbox_height"] = (
            features["geom_bbox_maxy"] - features["geom_bbox_miny"]
        )
        features["geom_aspect_ratio"] = features["geom_bbox_width"] / (
            features["geom_bbox_height"] + 1e-10
        )
        features["geom_compactness"] = (
            4
            * np.pi
            * features["geom_area"]
            / (features["geom_perimeter"] ** 2 + 1e-10)
        )
        features["geom_num_points"] = geometries.apply(
            lambda g: (
                len(g.exterior.coords)
                if g is not None and g.geom_type == "Polygon"
                else np.nan
            )
        )

    except Exception as e:
        print(f"Geometry parsing failed for {col_name}: {e}")

    return features


def create_temporal_features(df, date_col):
    """Extract temporal features from a date column."""
    features = {}
    if date_col not in df.columns:
        return features

    try:
        dates = pd.to_datetime(df[date_col], errors="coerce")
        features[f"{date_col}_year"] = dates.dt.year.astype('float64')
        features[f"{date_col}_month"] = dates.dt.month.astype('float64')
        features[f"{date_col}_day"] = dates.dt.day.astype('float64')
        features[f"{date_col}_dayofweek"] = dates.dt.dayofweek.astype('float64')
        features[f"{date_col}_dayofyear"] = dates.dt.dayofyear.astype('float64')
        features[f"{date_col}_quarter"] = dates.dt.quarter.astype('float64')
        features[f"{date_col}_weekofyear"] = dates.dt.isocalendar().week.astype('float64')

        features[f"{date_col}_month_sin"] = np.sin(2 * np.pi * dates.dt.month / 12)
        features[f"{date_col}_month_cos"] = np.cos(2 * np.pi * dates.dt.month / 12)
        features[f"{date_col}_dayofyear_sin"] = np.sin(
            2 * np.pi * dates.dt.dayofyear / 365.25
        )
        features[f"{date_col}_dayofyear_cos"] = np.cos(
            2 * np.pi * dates.dt.dayofyear / 365.25
        )

        season_map = {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3}
        features[f"{date_col}_season"] = dates.dt.month.map(season_map).astype('float64')

    except Exception as e:
        print(f"Temporal feature creation failed for {date_col}: {e}")

    return features


# Identify column types
target_col = "gisacres"

# High-cardinality identifier columns to exclude entirely
id_cols = [
    "OBJECTID",
    "uniquefireidentifier",
    "incidentname",
    "irwinid",
    "inciwebid",
    "localincidentidentifier",
    "complexparentirwinid",
]
# Also exclude 'id' if present
if "id" in train.columns:
    id_cols.append("id")

geom_cols = [
    c
    for c in train.columns
    if any(
        kw in c.lower()
        for kw in ["geom", "wkt", "geometry", "shape", "boundary"]
    )
    and "datetime" not in c.lower()
    and "date" not in c.lower()
]
print(f"Potential geometry columns: {geom_cols}")

date_cols = [
    c
    for c in train.columns
    if any(
        kw in c.lower()
        for kw in ["date", "time", "start", "end", "day", "month", "year"]
    )
]
date_cols_filtered = []
for c in date_cols:
    if train[c].dtype == object or str(train[c].dtype) == "datetime64[ns]":
        try:
            pd.to_datetime(train[c].iloc[:10], errors="coerce")
            date_cols_filtered.append(c)
        except:
            pass
print(f"Potential date columns: {date_cols_filtered}")

# Explicitly define genuinely informative categorical columns
cat_cols = [
    "agency",
    "mapmethod",
    "firecode",
    "state",
    "incomplex",
    "complexfirecode",
    "pooownerunit",
    "pooresponsibleunit",
    "complexname",
]
# Filter to only those that actually exist in the data
cat_cols = [c for c in cat_cols if c in train.columns]
print(f"Categorical columns: {cat_cols}")

# Numeric columns: everything that's not target, id, geom, date, cat, or log target
numeric_cols = [
    c
    for c in train.columns
    if c
    not in [target_col]
    + id_cols
    + geom_cols
    + date_cols_filtered
    + cat_cols
    + ["log_gisacres"]
]
print(f"Numeric columns: {numeric_cols}")

# Columns for frequency encoding (medium-cardinality categoricals)
freq_encode_cols = [c for c in cat_cols if c in ["state", "firecode", "agency"]]
print(f"Frequency encoding columns: {freq_encode_cols}")


# ============================================================
# 4. APPLY FEATURE ENGINEERING (fit on train_data only)
# ============================================================


def engineer_features(
    df, fit_data=None, cat_cols=None, geom_cols=None, date_cols=None, numeric_cols=None,
    freq_encode_cols=None
):
    """Apply feature engineering. If fit_data is provided, use its statistics."""
    result = df.copy()
    is_fitting = fit_data is None

    # --- Drop high-cardinality ID columns ---
    id_cols_to_drop = [
        "OBJECTID", "uniquefireidentifier", "incidentname", "irwinid",
        "inciwebid", "localincidentidentifier", "complexparentirwinid", "id"
    ]
    cols_to_drop = [c for c in id_cols_to_drop if c in result.columns]
    if cols_to_drop:
        result = result.drop(columns=cols_to_drop)

    # --- Geometry features ---
    for gcol in geom_cols:
        if gcol in df.columns:
            geom_feats = parse_geometry_features(df, gcol)
            for k, v in geom_feats.items():
                result[k] = v
            if gcol in result.columns:
                result = result.drop(columns=[gcol])

    # --- Temporal features ---
    for dcol in date_cols:
        if dcol in df.columns:
            temp_feats = create_temporal_features(df, dcol)
            for k, v in temp_feats.items():
                result[k] = v
            if dcol in result.columns:
                result = result.drop(columns=[dcol])

    # --- Categorical encoding (label encoding for tree models) ---
    if cat_cols:
        valid_cat_cols = [c for c in cat_cols if c in result.columns]
        if valid_cat_cols:
            if is_fitting:
                cat_mappings = {}
                for c in valid_cat_cols:
                    cats = sorted(result[c].dropna().unique().tolist())
                    cat_mappings[c] = {v: i for i, v in enumerate(cats)}
                    result[c] = result[c].map(cat_mappings[c])
                result.attrs["cat_mappings"] = cat_mappings
            else:
                cat_mappings = fit_data.attrs.get("cat_mappings", {})
                for c in valid_cat_cols:
                    if c in cat_mappings:
                        result[c] = result[c].map(cat_mappings[c])
                    else:
                        result[c] = result[c].map(
                            {
                                v: i
                                for i, v in enumerate(
                                    sorted(result[c].dropna().unique().tolist())
                                )
                            }
                        )

    # --- Frequency encoding for medium-cardinality categoricals ---
    if freq_encode_cols:
        valid_freq_cols = [c for c in freq_encode_cols if c in result.columns]
        for c in valid_freq_cols:
            freq_col_name = f"{c}_freq"
            if is_fitting:
                # Compute frequency from training data only
                freq_map = result[c].value_counts(normalize=True).to_dict()
                result[freq_col_name] = result[c].map(freq_map).fillna(0.0)
                result.attrs[f"freq_{c}"] = freq_map
            else:
                # Use frequency map from training data
                freq_map = fit_data.attrs.get(f"freq_{c}", {})
                if len(freq_map) > 0:
                    result[freq_col_name] = result[c].map(freq_map).fillna(0.0)
                else:
                    result[freq_col_name] = 0.0

    # --- Numeric log transforms ---
    if is_fitting:
        for c in numeric_cols:
            if c in result.columns and result[c].dtype in [
                "float64",
                "int64",
                "float32",
            ]:
                pos_vals = result[c][result[c] > 0]
                if len(pos_vals) > 0 and pos_vals.std() > pos_vals.mean() * 2:
                    result[f"{c}_log"] = np.log1p(result[c].clip(lower=0))
    else:
        for c in numeric_cols:
            if c in result.columns and result[c].dtype in [
                "float64",
                "int64",
                "float32",
            ]:
                if f"{c}_log" in fit_data.columns:
                    result[f"{c}_log"] = np.log1p(result[c].clip(lower=0))

    # --- Interaction features ---
    centroid_x = "geom_centroid_x"
    centroid_y = "geom_centroid_y"
    if centroid_x in result.columns and centroid_y in result.columns:
        result["geom_distance_from_origin"] = np.sqrt(
            result[centroid_x] ** 2 + result[centroid_y] ** 2
        )

    if "geom_perimeter" in result.columns and "geom_area" in result.columns:
        result["geom_perim_area_ratio"] = result["geom_perimeter"] / (
            np.sqrt(result["geom_area"].abs()) + 1e-10
        )

    # --- Handle remaining NaN values ---
    for c in result.columns:
        if result[c].dtype in ["float64", "float32"] and result[c].isnull().any():
            if is_fitting:
                fill_val = result[c].median()
                result[c] = result[c].fillna(fill_val)
            else:
                if c in fit_data.columns:
                    fill_val = fit_data[c].median()
                else:
                    fill_val = 0
                result[c] = result[c].fillna(fill_val)

    return result


print("\nEngineering features...")
train_processed = engineer_features(
    train_data,
    fit_data=None,
    cat_cols=cat_cols,
    geom_cols=geom_cols,
    date_cols=date_cols_filtered,
    numeric_cols=numeric_cols,
    freq_encode_cols=freq_encode_cols,
)

val_processed = engineer_features(
    val_data,
    fit_data=train_processed,
    cat_cols=cat_cols,
    geom_cols=geom_cols,
    date_cols=date_cols_filtered,
    numeric_cols=numeric_cols,
    freq_encode_cols=freq_encode_cols,
)

test_processed = engineer_features(
    test,
    fit_data=train_processed,
    cat_cols=cat_cols,
    geom_cols=geom_cols,
    date_cols=date_cols_filtered,
    numeric_cols=numeric_cols,
    freq_encode_cols=freq_encode_cols,
)

print(f"Processed train shape: {train_processed.shape}")
print(f"Processed val shape: {val_processed.shape}")
print(f"Processed test shape: {test_processed.shape}")

# ============================================================
# 5. PREPARE FINAL DATA FRAMES FOR MODELING
# ============================================================
exclude_cols = [target_col, "log_gisacres"] + id_cols
feature_cols = [c for c in train_processed.columns if c not in exclude_cols]

for c in feature_cols:
    if train_processed[c].dtype == object:
        try:
            train_processed[c] = pd.to_numeric(train_processed[c], errors="coerce")
            val_processed[c] = pd.to_numeric(val_processed[c], errors="coerce")
            test_processed[c] = pd.to_numeric(test_processed[c], errors="coerce")
        except:
            feature_cols.remove(c)

for c in feature_cols:
    if train_processed[c].isnull().any():
        fill_val = (
            train_processed[c].median() if train_processed[c].notna().any() else 0
        )
        train_processed[c] = train_processed[c].fillna(fill_val)
        val_processed[c] = val_processed[c].fillna(fill_val)
        test_processed[c] = test_processed[c].fillna(fill_val)

print(f"Final feature count: {len(feature_cols)}")

# ============================================================
# 6. SAVE PROCESSED DATA
# ============================================================
train_processed.attrs = {}
val_processed.attrs = {}
test_processed.attrs = {}
train_processed.to_parquet("./working/train_processed.parquet")
val_processed.to_parquet("./working/val_processed.parquet")
test_processed.to_parquet("./working/test_processed.parquet")

metadata = {
    "feature_cols": feature_cols,
    "target_col": target_col,
    "id_cols": id_cols,
    "geom_cols": geom_cols,
    "date_cols": date_cols_filtered,
    "cat_cols": cat_cols,
    "numeric_cols": numeric_cols,
    "freq_encode_cols": freq_encode_cols,
}
with open("./working/metadata.json", "w") as f:
    json.dump(metadata, f, indent=2, default=str)

# ============================================================
# 7. TRAIN MODELS
# ============================================================
y_train_log = train_processed["log_gisacres"].values
y_val_log = val_processed["log_gisacres"].values
y_val_orig = val_processed[target_col].values

X_train = train_processed[feature_cols].values
X_val = val_processed[feature_cols].values
X_test = test_processed[feature_cols].values

# --- Model 1: LightGBM with Huber loss ---
lgb_params = {
    "objective": "huber",
    "huber_delta": 1.0,
    "metric": "rmse",
    "boosting_type": "gbdt",
    "num_leaves": 63,
    "learning_rate": 0.03,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 20,
    "min_child_weight": 1e-3,
    "lambda_l1": 0.1,
    "lambda_l2": 1.0,
    "max_depth": -1,
    "n_estimators": 5000,
    "verbose": -1,
    "n_jobs": -1,
    "random_state": 42,
}

print("\n--- Training LightGBM (Huber loss) ---")
lgb_model = lgb.LGBMRegressor(**lgb_params)
lgb_model.fit(
    X_train,
    y_train_log,
    eval_set=[(X_val, y_val_log)],
    callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(500)],
)
print(f"  LGBM best iteration: {lgb_model.best_iteration_}")

# --- Model 2: XGBoost with Pseudo-Huber loss ---
xgb_params = {
    "objective": "reg:pseudohubererror",
    "huber_slope": 1.0,
    "eval_metric": "rmse",
    "max_depth": 6,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "min_child_weight": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "n_estimators": 5000,
    "tree_method": "hist",
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": 0,
    "early_stopping_rounds": 100,
}

print("\n--- Training XGBoost (Pseudo-Huber loss) ---")
xgb_model = xgb.XGBRegressor(**xgb_params)
xgb_model.fit(
    X_train,
    y_train_log,
    eval_set=[(X_val, y_val_log)],
    verbose=False,
)
print(f"  XGB best iteration: {xgb_model.best_iteration}")

# ============================================================
# 8. VALIDATION PREDICTIONS & METRIC (official: RMSE on original scale)
# ============================================================
val_pred_lgb_log = lgb_model.predict(X_val)
val_pred_xgb_log = xgb_model.predict(X_val)

val_pred_lgb = np.expm1(val_pred_lgb_log)
val_pred_xgb = np.expm1(val_pred_xgb_log)

val_pred_lgb = np.maximum(val_pred_lgb, 0)
val_pred_xgb = np.maximum(val_pred_xgb, 0)

rmse_lgb = np.sqrt(mean_squared_error(y_val_orig, val_pred_lgb))
rmse_xgb = np.sqrt(mean_squared_error(y_val_orig, val_pred_xgb))

print(f"\n  LGBM RMSE (original scale): {rmse_lgb:.4f}")
print(f"  XGB  RMSE (original scale): {rmse_xgb:.4f}")

# ============================================================
# 9. ENSEMBLE (inverse-RMSE weighted on log scale for stability)
# ============================================================
w_lgb = (1.0 / rmse_lgb) / (1.0 / rmse_lgb + 1.0 / rmse_xgb)
w_xgb = (1.0 / rmse_xgb) / (1.0 / rmse_lgb + 1.0 / rmse_xgb)
print(f"  Ensemble weights: LGBM={w_lgb:.4f}, XGB={w_xgb:.4f}")

val_pred_ens_log = w_lgb * val_pred_lgb_log + w_xgb * val_pred_xgb_log
val_pred_ens = np.expm1(val_pred_ens_log)
val_pred_ens = np.maximum(val_pred_ens, 0)

# ============================================================
# 10. FINAL VALIDATION SCORE (official metric: RMSE on original scale)
# ============================================================
final_rmse = np.sqrt(mean_squared_error(y_val_orig, val_pred_ens))
print(f"\n  Ensemble RMSE (original scale): {final_rmse:.4f}")

# ============================================================
# 11. TEST PREDICTIONS
# ============================================================
test_pred_lgb_log = lgb_model.predict(X_test)
test_pred_xgb_log = xgb_model.predict(X_test)

test_pred_ens_log = w_lgb * test_pred_lgb_log + w_xgb * test_pred_xgb_log
test_pred_ens = np.expm1(test_pred_ens_log)
test_pred_ens = np.maximum(test_pred_ens, 0)

# ============================================================
# 12. SAVE SUBMISSION
# ============================================================
if "id" in test_processed.columns:
    test_ids = test_processed["id"].values.astype(int)
else:
    test_ids = np.arange(len(test_processed))

submission = pd.DataFrame({"id": test_ids, "gisacres": test_pred_ens})
submission.to_csv("./submission/submission.csv", index=False)
print(f"\nSubmission saved: {submission.shape[0]} rows")
print(submission.head())

# ============================================================
# 13. SAVE MODELS
# ============================================================
os.makedirs("./working/models", exist_ok=True)
lgb_model.booster_.save_model("./working/models/lgb_model.txt")
xgb_model.save_model("./working/models/xgb_model.json")
print("Models saved to ./working/models/")

# ============================================================
# 14. FINAL OUTPUT
# ============================================================
score = final_rmse
print(f"Final Validation Score: {score}")
