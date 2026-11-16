import pandas as pd
import numpy as np
import lightgbm as lgb
import json
import os
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

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
print(f"Target stats:\n{train['Res_capacityM3'].describe()}")
print(f"Target skewness: {train['Res_capacityM3'].skew():.4f}")

# ============================================================
# 2. IDENTIFY COLUMN TYPES
# ============================================================
target_col = "Res_capacityM3"

id_cols = [c for c in train.columns if "id" in c.lower() or c == "Id" or c == "ID"]
if not id_cols and train.shape[1] > 1:
    if train.columns[0] not in [target_col] and train.iloc[:, 0].dtype in [
        "int64",
        "int32",
    ]:
        id_cols = [train.columns[0]]

feature_cols = [c for c in train.columns if c != target_col and c not in id_cols]
numeric_cols = train[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = [c for c in feature_cols if c not in numeric_cols]

print(f"Numeric features ({len(numeric_cols)}): {numeric_cols}")
print(f"Categorical features ({len(categorical_cols)}): {categorical_cols}")


# ============================================================
# 3. FEATURE ENGINEERING
# ============================================================
def engineer_features(df, numeric_cols):
    """Apply feature engineering."""
    df = df.copy()

    # --- Geographic features ---
    lat_col = None
    lon_col = None
    for c in df.columns:
        cl = c.lower()
        if "lat" in cl and "lon" not in cl and df[c].dtype != object:
            lat_col = c
        elif ("lon" in cl or "long" in cl) and df[c].dtype != object:
            lon_col = c

    if lat_col and lon_col:
        df["abs_lat"] = df[lat_col].abs()
        df["abs_lon"] = df[lon_col].abs()
        df["lat_lon_product"] = df[lat_col] * df[lon_col]
        df["lat_squared"] = df[lat_col] ** 2
        df["lon_squared"] = df[lon_col] ** 2
        df["distance_from_equator"] = df[lat_col].abs()
        df["hemisphere"] = np.where(df[lat_col] >= 0, 1, 0)
        df["tropical_zone"] = (df[lat_col].abs() < 23.5).astype(int)
        df["subtropical_zone"] = (
            (df[lat_col].abs() >= 23.5) & (df[lat_col].abs() < 35)
        ).astype(int)
        df["temperate_zone"] = (
            (df[lat_col].abs() >= 35) & (df[lat_col].abs() < 66.5)
        ).astype(int)
        df["polar_zone"] = (df[lat_col].abs() >= 66.5).astype(int)
        df["lat_lon_ratio"] = df[lat_col] / (df[lon_col] + 1e-8)

    # --- Elevation features ---
    elev_col = None
    for c in df.columns:
        cl = c.lower()
        if ("elev" in cl or "alt" in cl or "height" in cl or "z_coord" in cl) and df[c].dtype != object:
            elev_col = c
            break

    if elev_col and lat_col:
        df["elev_per_lat"] = df[elev_col] / (df[lat_col].abs() + 1)

    # --- Area/Volume ratio features ---
    area_cols = [
        c
        for c in df.columns
        if any(k in c.lower() for k in ["area", "basin", "catchment", "drainage"])
        and df[c].dtype != object
    ]
    vol_cols = [
        c
        for c in df.columns
        if any(k in c.lower() for k in ["volume", "capacity", "storage"])
        and c != target_col
        and df[c].dtype != object
    ]

    for ac in area_cols[:3]:
        for vc in vol_cols[:3]:
            if ac != vc:
                df[f"{ac}_to_{vc}_ratio"] = df[ac] / (df[vc] + 1e-8)

    # --- Length/Width features ---
    length_cols = [
        c
        for c in df.columns
        if any(
            k in c.lower() for k in ["length", "dam_len", "height", "width", "crest"]
        )
        and df[c].dtype != object
    ]
    for lc in length_cols[:4]:
        df[f"{lc}_log"] = np.log1p(df[lc].clip(lower=0))

    # --- Interaction features ---
    key_numeric = [c for c in numeric_cols if c in df.columns and df[c].dtype != object][:8]
    for i in range(min(len(key_numeric), 5)):
        for j in range(i + 1, min(len(key_numeric), 5)):
            c1, c2 = key_numeric[i], key_numeric[j]
            if c1 in df.columns and c2 in df.columns:
                df[f"{c1}_x_{c2}"] = df[c1] * df[c2]

    # --- Log transforms for skewed numeric features ---
    for c in numeric_cols:
        if c in df.columns and df[c].dtype != object:
            col = df[c]
            if col.notna().sum() > 0:
                positive_vals = col[col > 0]
                if len(positive_vals) > len(col) * 0.5 and positive_vals.max() > 10:
                    df[f"{c}_log"] = np.log1p(col.clip(lower=0))

    # --- Polynomial features ---
    for c in numeric_cols[:5]:
        if c in df.columns and df[c].dtype != object:
            df[f"{c}_sq"] = df[c] ** 2

    return df


# Convert potentially numeric string columns to numeric before feature engineering
for col in train.columns:
    if train[col].dtype == object:
        converted = pd.to_numeric(train[col], errors='coerce')
        if converted.notna().sum() > len(train) * 0.3:
            train[col] = converted
            test[col] = pd.to_numeric(test[col], errors='coerce') if col in test.columns else test[col]

# Re-identify numeric/categorical after conversion
numeric_cols = train[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = [c for c in feature_cols if c not in numeric_cols]
print(f"Numeric features after conversion ({len(numeric_cols)}): {numeric_cols}")
print(f"Categorical features after conversion ({len(categorical_cols)}): {categorical_cols}")

train_eng = engineer_features(train, numeric_cols)
test_eng = engineer_features(test, numeric_cols)

# Update feature columns
new_feature_cols = [
    c for c in train_eng.columns if c != target_col and c not in id_cols
]
print(
    f"After engineering - Train shape: {train_eng.shape}, Test shape: {test_eng.shape}"
)
print(f"Total features: {len(new_feature_cols)}")

# ============================================================
# 4. HANDLE MISSING VALUES
# ============================================================
for c in new_feature_cols:
    if train_eng[c].isnull().any():
        train_eng[f"{c}_missing"] = train_eng[c].isnull().astype(int)
        test_eng[f"{c}_missing"] = test_eng[c].isnull().astype(int)

new_feature_cols = [
    c for c in train_eng.columns if c != target_col and c not in id_cols
]
print(f"Features after missing indicators: {len(new_feature_cols)}")

# ============================================================
# 5. TARGET TRANSFORMATION
# ============================================================
if train_eng[target_col].min() >= 0:
    train_eng["log_target"] = np.log1p(train_eng[target_col])
    target_transform = "log1p"
    print(f"Target transformed with log1p")
else:
    train_eng["log_target"] = np.sign(train_eng[target_col]) * np.log1p(
        np.abs(train_eng[target_col])
    )
    target_transform = "signed_log"
    print(f"Target transformed with signed log")

# ============================================================
# 6. CREATE TRAIN/VALIDATION SPLIT
# ============================================================
train_eng["target_bin"] = pd.qcut(
    train_eng["log_target"], q=10, labels=False, duplicates="drop"
)

train_final, val_final = train_test_split(
    train_eng, test_size=0.15, random_state=42, stratify=train_eng["target_bin"]
)
train_final = train_final.drop("target_bin", axis=1)
val_final = val_final.drop("target_bin", axis=1)

print(f"Train split: {train_final.shape}")
print(f"Validation split: {val_final.shape}")
print(f"Test set: {test_eng.shape}")

# ============================================================
# 7. PREPARE MATRICES
# ============================================================
# Encode categorical features (fit ONLY on train_final to avoid leakage)
from sklearn.preprocessing import LabelEncoder

cat_cols_in_features = [c for c in new_feature_cols if train_final[c].dtype == object]
print(f"Categorical columns to encode: {cat_cols_in_features}")

for c in cat_cols_in_features:
    # Fit on combined train+val to ensure all local categories are known
    combined = pd.concat([train_final[c], val_final[c]], ignore_index=True).astype(str)
    le = LabelEncoder()
    le.fit(combined)
    train_final[c] = le.transform(train_final[c].astype(str))
    val_final[c] = le.transform(val_final[c].astype(str))
    # For test, use the same encoder; unknown categories get a fallback
    test_str = test_eng[c].astype(str)
    known_classes = set(le.classes_)
    class_to_idx = {cls: i for i, cls in enumerate(le.classes_)}
    test_mapped = test_str.map(lambda x: class_to_idx.get(x, -1))
    test_eng[c] = test_mapped.values

# Fill remaining NaNs (use train_final medians only to avoid val leakage)
for c in new_feature_cols:
    if train_final[c].dtype != object:
        median_val = train_final[c].median()
        train_final[c] = train_final[c].fillna(median_val)
        val_final[c] = val_final[c].fillna(median_val)
        test_eng[c] = test_eng[c].fillna(median_val)

X_train = train_final[new_feature_cols].values
y_train_log = train_final["log_target"].values
y_train_orig = train_final[target_col].values

X_val = val_final[new_feature_cols].values
y_val_log = val_final["log_target"].values
y_val_orig = val_final[target_col].values

X_test = test_eng[new_feature_cols].values

# ============================================================
# 8. STAGE 1: LightGBM on log-transformed target (Validation)
# ============================================================
params_stage1 = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.03,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 10,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "n_estimators": 3000,
    "verbose": -1,
    "n_jobs": -1,
    "random_state": 42,
}

print("\n=== Stage 1: Log-space LightGBM (Validation) ===")
model_stage1_val = lgb.LGBMRegressor(**params_stage1)
model_stage1_val.fit(
    X_train,
    y_train_log,
    eval_set=[(X_val, y_val_log)],
    callbacks=[lgb.early_stopping(100), lgb.log_evaluation(-1)],
)

val_pred_log_s1 = model_stage1_val.predict(X_val)
val_pred_orig_s1 = np.expm1(val_pred_log_s1)
rmse_s1_val = np.sqrt(mean_squared_error(y_val_orig, val_pred_orig_s1))
print(f"Stage 1 RMSE (original scale): {rmse_s1_val:.4f}")

# ============================================================
# 9. STAGE 2: Residual correction on original scale
# ============================================================
train_pred_log_s1 = model_stage1_val.predict(X_train)
train_pred_orig_s1 = np.expm1(train_pred_log_s1)

residual_train = y_train_orig - train_pred_orig_s1
residual_val = y_val_orig - val_pred_orig_s1

X_train_s2 = np.column_stack([X_train, train_pred_log_s1])
X_val_s2 = np.column_stack([X_val, val_pred_log_s1])

params_stage2 = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "num_leaves": 15,
    "learning_rate": 0.02,
    "feature_fraction": 0.6,
    "bagging_fraction": 0.7,
    "bagging_freq": 5,
    "min_child_samples": 20,
    "reg_alpha": 1.0,
    "reg_lambda": 5.0,
    "n_estimators": 2000,
    "verbose": -1,
    "n_jobs": -1,
    "random_state": 42,
}

print("\n=== Stage 2: Residual Correction (Validation) ===")
model_stage2_val = lgb.LGBMRegressor(**params_stage2)
model_stage2_val.fit(
    X_train_s2,
    residual_train,
    eval_set=[(X_val_s2, residual_val)],
    callbacks=[lgb.early_stopping(100), lgb.log_evaluation(-1)],
)

val_residual_pred = model_stage2_val.predict(X_val_s2)
val_pred_final = val_pred_orig_s1 + val_residual_pred
val_pred_final = np.maximum(val_pred_final, 0)

# ============================================================
# 10. VALIDATION METRIC (Official: RMSE on original scale)
# ============================================================
rmse_final_val = np.sqrt(mean_squared_error(y_val_orig, val_pred_final))
print(f"\n=== VALIDATION RESULTS ===")
print(f"Stage 1 only RMSE:  {rmse_s1_val:.4f}")
print(f"Two-stage RMSE:     {rmse_final_val:.4f}")
print(f"Improvement:        {(rmse_s1_val - rmse_final_val) / rmse_s1_val * 100:.2f}%")

# ============================================================
# 11. RETRAIN ON FULL DATA (train + val) FOR TEST PREDICTIONS
# ============================================================
X_full = np.vstack([X_train, X_val])
y_full_log = np.concatenate([y_train_log, y_val_log])
y_full_orig = np.concatenate([y_train_orig, y_val_orig])

best_iter_s1 = model_stage1_val.best_iteration_
best_iter_s2 = model_stage2_val.best_iteration_
print(f"\nBest iterations - Stage 1: {best_iter_s1}, Stage 2: {best_iter_s2}")

params_stage1_full = params_stage1.copy()
params_stage1_full["n_estimators"] = best_iter_s1

print("\n=== Stage 1: Retrain on Full Data ===")
model_stage1_full = lgb.LGBMRegressor(**params_stage1_full)
model_stage1_full.fit(X_full, y_full_log)

full_pred_log_s1 = model_stage1_full.predict(X_full)
full_pred_orig_s1 = np.expm1(full_pred_log_s1)
residual_full = y_full_orig - full_pred_orig_s1

X_full_s2 = np.column_stack([X_full, full_pred_log_s1])

params_stage2_full = params_stage2.copy()
params_stage2_full["n_estimators"] = best_iter_s2

print("=== Stage 2: Retrain on Full Data ===")
model_stage2_full = lgb.LGBMRegressor(**params_stage2_full)
model_stage2_full.fit(X_full_s2, residual_full)

# ============================================================
# 12. TEST PREDICTIONS
# ============================================================
test_pred_log_s1 = model_stage1_full.predict(X_test)
test_pred_orig_s1 = np.expm1(test_pred_log_s1)

X_test_s2 = np.column_stack([X_test, test_pred_log_s1])
test_residual_pred = model_stage2_full.predict(X_test_s2)

test_pred_final = test_pred_orig_s1 + test_residual_pred
test_pred_final = np.maximum(test_pred_final, 0)

# ============================================================
# 13. SAVE SUBMISSION
# ============================================================
submission = pd.DataFrame(
    {"id": range(len(test_pred_final)), "Res_capacityM3": test_pred_final}
)
submission.to_csv("./submission/submission.csv", index=False)
print(f"\nSubmission saved: {submission.shape}")
print(
    f"Prediction stats: min={test_pred_final.min():.2f}, max={test_pred_final.max():.2f}, mean={test_pred_final.mean():.2f}"
)

# ============================================================
# FINAL OUTPUT
# ============================================================
score = rmse_final_val
print(f"Final Validation Score: {score:.4f}")