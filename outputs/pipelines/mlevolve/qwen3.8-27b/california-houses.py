import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold, train_test_split
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from scipy.optimize import nnls
import os
import warnings

warnings.filterwarnings("ignore")
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# ============================================================
# 1. LOAD DATA
# ============================================================
train_orig = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

print(f"Train shape: {train_orig.shape}")
print(f"Test shape: {test_df.shape}")

TARGET = "Total interior livable area"
y_series = train_orig[TARGET]
train_orig = train_orig.drop(columns=[TARGET])

# Identify column types
numeric_cols = train_orig.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = train_orig.select_dtypes(
    include=["object", "category"]
).columns.tolist()

print(f"Numeric columns ({len(numeric_cols)}): {numeric_cols[:20]}...")
print(f"Categorical columns ({len(categorical_cols)}): {categorical_cols}")

# ============================================================
# 2. SPLIT TRAIN INTO TRAIN/VAL (BEFORE any feature engineering)
# ============================================================
train_df, val_df = train_test_split(
    train_orig, test_size=0.2, random_state=42, shuffle=True
)
y_train = y_series.loc[train_df.index].values
y_val = y_series.loc[val_df.index].values

print(f"Train split: {train_df.shape}, Val split: {val_df.shape}")
print(f"y_train mean: {y_train.mean():.2f}, y_val mean: {y_val.mean():.2f}")

# ============================================================
# 3. FEATURE ENGINEERING
# ============================================================


def engineer_features(train_df, val_df, test_df, numeric_cols):
    """Engineer features with strict train-only fitting for stateful transforms."""

    # --- Convert 'Bedrooms' from categorical string to numeric ---
    # 'Bedrooms' values look like '2', '3', '4+', '5' etc.
    # Extract the leading integer: '3' -> 3, '4+' -> 4
    if "Bedrooms" in train_df.columns:
        def _extract_bed_count(val):
            if pd.isna(val):
                return np.nan
            s = str(val).strip()
            # Extract leading digits
            import re
            m = re.match(r"(\d+)", s)
            if m:
                return int(m.group(1))
            return np.nan

        train_df["Bedrooms"] = train_df["Bedrooms"].apply(_extract_bed_count)
        val_df["Bedrooms"] = val_df["Bedrooms"].apply(_extract_bed_count)
        test_df["Bedrooms"] = test_df["Bedrooms"].apply(_extract_bed_count)
        # Move Bedrooms from categorical to numeric
        if "Bedrooms" in categorical_cols:
            categorical_cols.remove("Bedrooms")
        if "Bedrooms" not in numeric_cols:
            numeric_cols.append("Bedrooms")
        print(f"Converted 'Bedrooms' to numeric. Train values: {sorted(train_df['Bedrooms'].dropna().unique())}")

    # --- Handle missing values ---
    miss_cols = [c for c in train_df.columns if train_df[c].isnull().any()]
    print(f"Columns with missing values: {miss_cols}")

    for col in miss_cols:
        if col in numeric_cols:
            fill_val = train_df[col].median()
            train_df[col] = train_df[col].fillna(fill_val)
            val_df[col] = val_df[col].fillna(fill_val)
            test_df[col] = test_df[col].fillna(fill_val)
        elif col in categorical_cols:
            fill_val = train_df[col].mode()
            fill_val = fill_val[0] if len(fill_val) > 0 else "UNKNOWN"
            train_df[col] = train_df[col].fillna(fill_val)
            val_df[col] = val_df[col].fillna(fill_val)
            test_df[col] = test_df[col].fillna(fill_val)

    # --- Domain-specific interaction features ---
    lot_cols = [
        c
        for c in numeric_cols
        if "lot" in c.lower() or "acre" in c.lower() or "size" in c.lower()
    ]
    bed_cols = [c for c in numeric_cols if "bed" in c.lower() or "bedroom" in c.lower()]
    bath_cols = [c for c in numeric_cols if "bath" in c.lower()]
    room_cols = [c for c in numeric_cols if "room" in c.lower() or "space" in c.lower()]
    year_cols = [c for c in numeric_cols if "year" in c.lower() or "built" in c.lower()]
    floor_cols = [
        c
        for c in numeric_cols
        if "floor" in c.lower() or "story" in c.lower() or "level" in c.lower()
    ]
    area_cols = [
        c
        for c in numeric_cols
        if "area" in c.lower() or "sqft" in c.lower() or "square" in c.lower()
    ]
    pop_cols = [
        c
        for c in numeric_cols
        if "pop" in c.lower() or "occup" in c.lower() or "household" in c.lower()
    ]
    income_cols = [
        c
        for c in numeric_cols
        if "income" in c.lower()
        or "inc" in c.lower()
        or "price" in c.lower()
        or "value" in c.lower()
    ]
    geo_cols = [
        c
        for c in numeric_cols
        if "lat" in c.lower() or "lon" in c.lower() or "zip" in c.lower()
    ]
    tax_cols = [c for c in numeric_cols if "tax" in c.lower()]

    print(f"Lot cols: {lot_cols}")
    print(f"Bed cols: {bed_cols}")
    print(f"Bath cols: {bath_cols}")
    print(f"Room cols: {room_cols}")
    print(f"Year cols: {year_cols}")
    print(f"Floor cols: {floor_cols}")
    print(f"Area cols: {area_cols}")
    print(f"Income cols: {income_cols}")
    print(f"Geo cols: {geo_cols}")

    dfs = {"train": train_df, "val": val_df, "test": test_df}

    # Pre-compute which numeric columns should get log transform (based on train only)
    log_cols = []
    for col in numeric_cols:
        if col in train_df.columns:
            col_vals = train_df[col].dropna()
            if len(col_vals) > 0 and col_vals.max() > 0 and col_vals.min() >= 0:
                if col_vals.skew() > 1.0:
                    log_cols.append(col)
    print(f"Log-transform columns: {log_cols}")

    for name, df in dfs.items():
        # --- Bedroom-based interaction features (now that Bedrooms is numeric) ---
        if lot_cols and bed_cols:
            df["lot_per_bed"] = df[lot_cols[0]] / (df[bed_cols[0]] + 1e-8)
        if lot_cols and len(lot_cols) > 1:
            df["lot_ratio_12"] = df[lot_cols[0]] / (df[lot_cols[1]] + 1e-8)
        if room_cols and bed_cols:
            df["room_per_bed"] = df[room_cols[0]] / (df[bed_cols[0]] + 1e-8)
        if bath_cols and bed_cols:
            df["bath_per_bed"] = df[bath_cols[0]] / (df[bed_cols[0]] + 1e-8)
        if area_cols and bed_cols:
            df["area_per_bed"] = df[area_cols[0]] / (df[bed_cols[0]] + 1e-8)
        if area_cols and room_cols:
            df["area_per_room"] = df[area_cols[0]] / (df[room_cols[0]] + 1e-8)
        if lot_cols and area_cols:
            df["lot_to_area"] = df[lot_cols[0]] / (df[area_cols[0]] + 1e-8)

        # --- New: bedrooms * total_spaces, bedrooms / bathrooms, total_spaces / bedrooms ---
        if bed_cols and room_cols:
            df["bed_times_rooms"] = df[bed_cols[0]] * df[room_cols[0]]
            df["total_spaces_per_bed"] = df[room_cols[0]] / (df[bed_cols[0]] + 1e-8)
        if bed_cols and bath_cols:
            df["bed_per_bath"] = df[bed_cols[0]] / (df[bath_cols[0]] + 1e-8)

        if year_cols:
            df["age"] = 2024 - df[year_cols[0]]
            df["age_sq"] = df["age"] ** 2
            df["decade"] = (df[year_cols[0]] // 10) * 10

        if floor_cols:
            df["floor_sq"] = df[floor_cols[0]] ** 2

        if income_cols:
            df["log_income"] = np.log1p(df[income_cols[0]].clip(lower=0))
            if len(income_cols) > 1:
                df["income_ratio"] = df[income_cols[0]] / (df[income_cols[1]] + 1e-8)

        if pop_cols:
            df["log_pop"] = np.log1p(df[pop_cols[0]].clip(lower=0))

        if tax_cols:
            df["log_tax"] = np.log1p(df[tax_cols[0]].clip(lower=0))
            if income_cols:
                df["tax_to_income"] = df[tax_cols[0]] / (df[income_cols[0]] + 1e-8)

        if "Latitude" in df.columns and "Longitude" in df.columns:
            df["dist_sf"] = np.sqrt(
                (df["Latitude"] - 37.7749) ** 2 + (df["Longitude"] - -122.4194) ** 2
            )
            df["dist_la"] = np.sqrt(
                (df["Latitude"] - 34.0522) ** 2 + (df["Longitude"] - -118.2437) ** 2
            )
            df["dist_sd"] = np.sqrt(
                (df["Latitude"] - 32.7157) ** 2 + (df["Longitude"] - -117.1611) ** 2
            )
            df["lat_lon_product"] = df["Latitude"] * df["Longitude"]

        # --- Log1p transforms for skewed numeric columns ---
        for col in log_cols:
            if col in df.columns:
                df[f"log_{col}"] = np.log1p(df[col].clip(lower=0))

        # --- Polynomial (squared) features for top 5 numeric features ---
        top5 = numeric_cols[:5]
        for col in top5:
            if col in df.columns:
                df[f"{col}_sq"] = df[col] ** 2

    return train_df, val_df, test_df


train_df, val_df, test_df = engineer_features(train_df, val_df, test_df, numeric_cols)

# ============================================================
# 4. ENCODE CATEGORICALS (Target Encoding with CV for train)
# ============================================================


def target_encode_cv(train_df, val_df, test_df, y_train, cat_cols, n_splits=5):
    """Target encoding with cross-validation to prevent leakage."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    global_mean = y_train.mean()

    for col in cat_cols:
        if col not in train_df.columns:
            continue
        train_df[f"{col}_te"] = 0.0
        val_df[f"{col}_te"] = 0.0
        test_df[f"{col}_te"] = 0.0

        for tr_idx, va_idx in kf.split(train_df):
            tr_target = y_train[tr_idx]
            tr_cat = train_df.iloc[tr_idx][col].values
            va_cat = train_df.iloc[va_idx][col].values

            enc_map = pd.Series(tr_target, index=tr_cat).groupby(level=0).mean()
            enc_map = enc_map.fillna(global_mean)

            train_df.iloc[va_idx, train_df.columns.get_loc(f"{col}_te")] = (
                pd.Series(va_cat).map(enc_map).fillna(global_mean).values
            )

        full_enc = (
            pd.Series(y_train, index=train_df[col].values).groupby(level=0).mean()
        )
        full_enc = full_enc.fillna(global_mean)

        val_df[f"{col}_te"] = val_df[col].map(full_enc).fillna(global_mean).values
        test_df[f"{col}_te"] = test_df[col].map(full_enc).fillna(global_mean).values

    for col in cat_cols:
        if col not in train_df.columns:
            continue
        count_map = train_df[col].value_counts()
        train_df[f"{col}_count"] = train_df[col].map(count_map).fillna(0)
        val_df[f"{col}_count"] = val_df[col].map(count_map).fillna(0)
        test_df[f"{col}_count"] = test_df[col].map(count_map).fillna(0)

    return train_df, val_df, test_df


cat_cols_final = [
    c
    for c in train_df.columns
    if train_df[c].dtype == "object" or train_df[c].dtype.name == "category"
]
print(f"Categorical columns for encoding: {cat_cols_final}")

if cat_cols_final:
    train_df, val_df, test_df = target_encode_cv(
        train_df, val_df, test_df, y_train, cat_cols_final
    )

# ============================================================
# 5. PREPARE FINAL FEATURE MATRICES
# ============================================================
drop_cols = [c for c in cat_cols_final if c in train_df.columns]
train_features = train_df.drop(columns=drop_cols)
val_features = val_df.drop(columns=drop_cols)
test_features = test_df.drop(columns=drop_cols)

train_features = train_features.apply(pd.to_numeric, errors="coerce").fillna(0)
val_features = val_features.apply(pd.to_numeric, errors="coerce").fillna(0)
test_features = test_features.apply(pd.to_numeric, errors="coerce").fillna(0)

train_features = train_features.replace([np.inf, -np.inf], 0)
val_features = val_features.replace([np.inf, -np.inf], 0)
test_features = test_features.replace([np.inf, -np.inf], 0)

scaler = StandardScaler()
X_train = scaler.fit_transform(train_features.values)
X_val = scaler.transform(val_features.values)
X_test = scaler.transform(test_features.values)

y_train_log = np.log1p(y_train)
y_val_log = np.log1p(y_val)
test_ids = np.arange(len(test_features))

print(f"\nFinal feature matrix shapes:")
print(f"  X_train: {X_train.shape}")
print(f"  X_val: {X_val.shape}")
print(f"  X_test: {X_test.shape}")

# ============================================================
# 6. MODEL 1: LightGBM in original space with CV
# ============================================================
print("\n" + "=" * 60)
print("Training LightGBM (original space) with 5-fold CV...")
print("=" * 60)

# LGBM_strong_reg: stable, heavily regularized
lgb_strong_reg_params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "num_leaves": 63,
    "learning_rate": 0.01,
    "feature_fraction": 0.5,
    "bagging_fraction": 0.7,
    "bagging_freq": 5,
    "min_child_samples": 50,
    "reg_alpha": 5.0,
    "reg_lambda": 10.0,
    "verbose": -1,
    "random_state": 42,
}

# LGBM_moderate: more capacity, lower learning rate
lgb_moderate_params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "num_leaves": 255,
    "learning_rate": 0.005,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.7,
    "bagging_freq": 5,
    "min_child_samples": 20,
    "reg_alpha": 1.0,
    "reg_lambda": 5.0,
    "verbose": -1,
    "random_state": 42,
}

kf = KFold(n_splits=5, shuffle=True, random_state=42)

# --- Train LGBM_strong_reg with CV (OOF + fold models for test) ---
lgb_strong_reg_cv_scores = []
lgb_strong_reg_best_iters = []
lgb_strong_reg_oof = np.zeros(len(X_train))
lgb_strong_reg_test_preds = np.zeros(len(X_test))

for fold, (tr_idx, va_idx) in enumerate(kf.split(X_train)):
    X_tr, X_va = X_train[tr_idx], X_train[va_idx]
    y_tr, y_va = y_train[tr_idx], y_train[va_idx]

    model = lgb.LGBMRegressor(n_estimators=10000, **lgb_strong_reg_params)
    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(-1)],
    )
    pred = model.predict(X_va)
    score = root_mean_squared_error(y_va, pred)
    lgb_strong_reg_cv_scores.append(score)
    lgb_strong_reg_best_iters.append(model.best_iteration_)
    lgb_strong_reg_oof[va_idx] = pred
    lgb_strong_reg_test_preds += model.predict(X_test) / 5
    print(f"  Strong_Reg Fold {fold+1}: RMSE={score:.4f}, best_iter={model.best_iteration_}")

lgb_strong_reg_cv_mean = np.mean(lgb_strong_reg_cv_scores)
lgb_strong_reg_cv_std = np.std(lgb_strong_reg_cv_scores)
print(f"  LGBM_strong_reg CV: {lgb_strong_reg_cv_mean:.4f} ± {lgb_strong_reg_cv_std:.4f}")

# --- Train LGBM_moderate with CV (OOF + fold models for test) ---
lgb_moderate_cv_scores = []
lgb_moderate_best_iters = []
lgb_moderate_oof = np.zeros(len(X_train))
lgb_moderate_test_preds = np.zeros(len(X_test))

for fold, (tr_idx, va_idx) in enumerate(kf.split(X_train)):
    X_tr, X_va = X_train[tr_idx], X_train[va_idx]
    y_tr, y_va = y_train[tr_idx], y_train[va_idx]

    model = lgb.LGBMRegressor(n_estimators=10000, **lgb_moderate_params)
    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(-1)],
    )
    pred = model.predict(X_va)
    score = root_mean_squared_error(y_va, pred)
    lgb_moderate_cv_scores.append(score)
    lgb_moderate_best_iters.append(model.best_iteration_)
    lgb_moderate_oof[va_idx] = pred
    lgb_moderate_test_preds += model.predict(X_test) / 5
    print(f"  Moderate Fold {fold+1}: RMSE={score:.4f}, best_iter={model.best_iteration_}")

lgb_moderate_cv_mean = np.mean(lgb_moderate_cv_scores)
lgb_moderate_cv_std = np.std(lgb_moderate_cv_scores)
print(f"  LGBM_moderate CV: {lgb_moderate_cv_mean:.4f} ± {lgb_moderate_cv_std:.4f}")

# ============================================================
# 7. MODEL 2: XGBoost in log-space with CV (OOF + fold models for test)
# ============================================================
print("\n" + "=" * 60)
print("Training XGBoost (log space) with 5-fold CV...")
print("=" * 60)

xgb_params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "max_depth": 6,
    "learning_rate": 0.02,
    "subsample": 0.7,
    "colsample_bytree": 0.6,
    "min_child_weight": 30,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "tree_method": "hist",
    "random_state": 42,
    "verbosity": 0,
}

xgb_cv_scores = []
xgb_best_iters = []
xgb_oof = np.zeros(len(X_train))
xgb_test_preds = np.zeros(len(X_test))

for fold, (tr_idx, va_idx) in enumerate(kf.split(X_train)):
    X_tr, X_va = X_train[tr_idx], X_train[va_idx]
    y_tr, y_va = y_train_log[tr_idx], y_train_log[va_idx]

    model = xgb.XGBRegressor(n_estimators=5000, early_stopping_rounds=200, **xgb_params)
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    pred_log = model.predict(X_va)
    pred = np.expm1(pred_log)
    score = root_mean_squared_error(y_train[va_idx], pred)
    xgb_cv_scores.append(score)
    xgb_best_iters.append(model.best_iteration)
    xgb_oof[va_idx] = pred
    xgb_test_preds += np.expm1(model.predict(X_test)) / 5
    print(f"  XGB Fold {fold+1}: RMSE={score:.4f}, best_iter={model.best_iteration}")

xgb_cv_mean = np.mean(xgb_cv_scores)
xgb_cv_std = np.std(xgb_cv_scores)
print(f"  XGB CV: {xgb_cv_mean:.4f} ± {xgb_cv_std:.4f}")

# ============================================================
# 8. OOF-BASED STACKING ENSEMBLE (Ridge meta-learner)
# ============================================================
print("\n" + "=" * 60)
print("OOF-based stacking ensemble with Ridge meta-learner...")
print("=" * 60)

# Stack OOF predictions from the 3 models
oof_matrix = np.column_stack([lgb_strong_reg_oof, lgb_moderate_oof, xgb_oof])
test_oof_matrix = np.column_stack([
    lgb_strong_reg_test_preds,
    lgb_moderate_test_preds,
    xgb_test_preds,
])

# Generate val predictions using the same fold models (val was never seen by any fold)
# Since val is a separate 20% split, we use the fold models' test-style predictions
# on val: each fold model was trained on 4/5 of X_train, so we predict val directly
lgb_strong_reg_val_preds = np.zeros(len(X_val))
lgb_moderate_val_preds = np.zeros(len(X_val))
xgb_val_preds = np.zeros(len(X_val))

for fold, (tr_idx, va_idx) in enumerate(kf.split(X_train)):
    X_tr = X_train[tr_idx]
    y_tr = y_train[tr_idx]

    m1 = lgb.LGBMRegressor(n_estimators=10000, **lgb_strong_reg_params)
    m1.fit(
        X_tr, y_tr,
        eval_set=[(X_train[va_idx], y_train[va_idx])],
        callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(-1)],
    )
    lgb_strong_reg_val_preds += m1.predict(X_val) / 5

    m2 = lgb.LGBMRegressor(n_estimators=10000, **lgb_moderate_params)
    m2.fit(
        X_tr, y_tr,
        eval_set=[(X_train[va_idx], y_train[va_idx])],
        callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(-1)],
    )
    lgb_moderate_val_preds += m2.predict(X_val) / 5

    m3 = xgb.XGBRegressor(n_estimators=5000, early_stopping_rounds=200, **xgb_params)
    m3.fit(X_tr, y_train_log[tr_idx], eval_set=[(X_train[va_idx], y_train_log[va_idx])], verbose=False)
    xgb_val_preds += np.expm1(m3.predict(X_val)) / 5

val_oof_matrix = np.column_stack([lgb_strong_reg_val_preds, lgb_moderate_val_preds, xgb_val_preds])

# Train Ridge meta-learner on OOF predictions
meta_ridge = Ridge(alpha=1.0)
meta_ridge.fit(oof_matrix, y_train)
meta_weights = meta_ridge.coef_
meta_intercept = meta_ridge.intercept_
print(f"  Ridge meta-learner weights: {meta_weights}")
print(f"  Ridge meta-learner intercept: {meta_intercept:.4f}")

# Meta predictions
ensemble_val_pred = meta_ridge.predict(val_oof_matrix)
ensemble_test_pred = meta_ridge.predict(test_oof_matrix)

# Also compute simple average for comparison
simple_avg_val_pred = val_oof_matrix.mean(axis=1)
simple_avg_test_pred = test_oof_matrix.mean(axis=1)

ensemble_val_rmse = root_mean_squared_error(y_val, ensemble_val_pred)
simple_avg_val_rmse = root_mean_squared_error(y_val, simple_avg_val_pred)
print(f"  Ridge ensemble Val RMSE: {ensemble_val_rmse:.4f}")
print(f"  Simple average Val RMSE: {simple_avg_val_rmse:.4f}")

# ============================================================
# 9. INDIVIDUAL MODEL VAL PREDICTIONS (from fold models)
# ============================================================
# Individual model val predictions (from averaged fold models)
lgb_strong_reg_val_rmse = root_mean_squared_error(y_val, lgb_strong_reg_val_preds)
lgb_moderate_val_rmse = root_mean_squared_error(y_val, lgb_moderate_val_preds)
xgb_val_rmse = root_mean_squared_error(y_val, xgb_val_preds)

print(f"  LGBM_strong_reg val RMSE: {lgb_strong_reg_val_rmse:.4f}")
print(f"  LGBM_moderate val RMSE: {lgb_moderate_val_rmse:.4f}")
print(f"  XGB val RMSE: {xgb_val_rmse:.4f}")

# ============================================================
# 10. SELECT BEST APPROACH
# ============================================================
print("\n" + "=" * 60)
print("Model Selection Summary")
print("=" * 60)

candidates = {
    "LGBM_strong_reg": (lgb_strong_reg_val_preds, lgb_strong_reg_test_preds, lgb_strong_reg_val_rmse),
    "LGBM_moderate": (lgb_moderate_val_preds, lgb_moderate_test_preds, lgb_moderate_val_rmse),
    "XGB": (xgb_val_preds, xgb_test_preds, xgb_val_rmse),
    "Ridge_Ensemble": (ensemble_val_pred, ensemble_test_pred, ensemble_val_rmse),
    "Simple_Avg": (simple_avg_val_pred, simple_avg_test_pred, simple_avg_val_rmse),
}

for name, (val_p, test_p, rmse) in candidates.items():
    print(f"  {name:18s}: val RMSE = {rmse:.4f}")

best_name = min(candidates, key=lambda k: candidates[k][2])
best_val_pred, best_test_pred, best_rmse = candidates[best_name]
print(f"\n  BEST: {best_name} (RMSE={best_rmse:.4f})")

# ============================================================
# 11. POST-PROCESSING: Clip to reasonable range (applied identically to val & test)
# ============================================================
clip_lower = 0
clip_upper = y_train.max() * 1.5

best_val_pred_clipped = np.clip(best_val_pred, clip_lower, clip_upper)
best_test_pred_clipped = np.clip(best_test_pred, clip_lower, clip_upper)

final_val_rmse = root_mean_squared_error(y_val, best_val_pred_clipped)
print(f"\nAfter clipping [{clip_lower}, {clip_upper:.0f}]:")
print(f"  Val RMSE: {final_val_rmse:.4f}")

# ============================================================
# 12. GENERATE SUBMISSION
# ============================================================
submission = pd.DataFrame(
    {"id": test_ids.astype(int), "Total interior livable area": best_test_pred_clipped}
)
submission.to_csv("./submission/submission.csv", index=False)
print(f"\nSubmission saved: {submission.shape}")
print(submission.head())
print(
    f"Prediction stats: mean={best_test_pred_clipped.mean():.0f}, "
    f"std={best_test_pred_clipped.std():.0f}, "
    f"min={best_test_pred_clipped.min():.0f}, max={best_test_pred_clipped.max():.0f}"
)

# ============================================================
# 13. FINAL VALIDATION SCORE
# ============================================================
score = final_val_rmse
print(
    f"\nValidation setup: 80/20 random split, RMSE metric, target='Total interior livable area'"
)
print(f"Best model: {best_name}")
print(f"Final Validation Score: {score}")