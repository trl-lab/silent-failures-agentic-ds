import pandas as pd
import numpy as np
import os
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

# ============================================================
# 1. LOAD DATA
# ============================================================
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")

# ============================================================
# 2. IDENTIFY COLUMNS
# ============================================================
TARGET = "TOTAL_MWH"
ID_COL = "id"

if TARGET in train.columns:
    y = train[TARGET].copy()
    train_features = train.drop(columns=[TARGET])
else:
    y = None
    train_features = train.copy()

if ID_COL in train_features.columns:
    train_ids = train_features[ID_COL].copy()
    train_features = train_features.drop(columns=[ID_COL])
else:
    train_ids = pd.Series(range(len(train_features)))

if ID_COL in test.columns:
    test_ids = test[ID_COL].copy()
    test_features = test.drop(columns=[ID_COL])
else:
    test_ids = pd.Series(range(len(test)))
    test_features = test.copy()

print(
    f"Feature columns ({len(train_features.columns)}): {list(train_features.columns)}"
)

# ============================================================
# 3. IDENTIFY COLUMN TYPES
# ============================================================
cat_cols = train_features.select_dtypes(include=["object", "category"]).columns.tolist()
num_cols = train_features.select_dtypes(include=[np.number]).columns.tolist()
date_cols = [
    c for c in train_features.columns if train_features[c].dtype == "datetime64[ns]"
]

print(f"Categorical columns ({len(cat_cols)}): {cat_cols}")
print(f"Numeric columns ({len(num_cols)}): {num_cols}")
print(f"Date columns ({len(date_cols)}): {date_cols}")


# ============================================================
# 4. FEATURE ENGINEERING
# ============================================================
def engineer_features(df, cat_cols, num_cols, date_cols, fit_params=None):
    """Apply feature engineering. If fit_params is None, compute from this df (training)."""
    df = df.copy()

    # --- Temporal features from date columns ---
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
            df[f"{col}_year"] = df[col].dt.year
            df[f"{col}_month"] = df[col].dt.month
            df[f"{col}_day"] = df[col].dt.day
            df[f"{col}_dayofweek"] = df[col].dt.dayofweek
            df[f"{col}_dayofyear"] = df[col].dt.dayofyear
            df[f"{col}_quarter"] = df[col].dt.quarter
            df[f"{col}_weekofyear"] = df[col].dt.isocalendar().week.astype(int)
            # Cyclical encoding
            df[f"{col}_month_sin"] = np.sin(2 * np.pi * df[col].dt.month / 12)
            df[f"{col}_month_cos"] = np.cos(2 * np.pi * df[col].dt.month / 12)
            df[f"{col}_dayofweek_sin"] = np.sin(2 * np.pi * df[col].dt.dayofweek / 7)
            df[f"{col}_dayofweek_cos"] = np.cos(2 * np.pi * df[col].dt.dayofweek / 7)
            df[f"{col}_dayofyear_sin"] = np.sin(2 * np.pi * df[col].dt.dayofyear / 365)
            df[f"{col}_dayofyear_cos"] = np.cos(2 * np.pi * df[col].dt.dayofyear / 365)
            df = df.drop(columns=[col])

    # Update column lists after date expansion
    new_num_cols = [
        c
        for c in df.columns
        if c not in cat_cols and pd.api.types.is_numeric_dtype(df[c])
    ]

    # --- Categorical encoding (label encoding for tree models) ---
    for col in cat_cols:
        if col in df.columns:
            if fit_params is not None and col in fit_params.get("cat_mappings", {}):
                mapping = fit_params["cat_mappings"][col]
                df[col] = df[col].map(mapping).fillna(-1).astype(int)
            else:
                categories = sorted(df[col].dropna().unique().tolist())
                mapping = {c: i for i, c in enumerate(categories)}
                if fit_params is None:
                    fit_params = {"cat_mappings": {}}
                if "cat_mappings" not in fit_params:
                    fit_params["cat_mappings"] = {}
                fit_params["cat_mappings"][col] = mapping
                df[col] = df[col].map(mapping).fillna(-1).astype(int)

    # --- Numeric feature engineering (only log for positive columns) ---
    for col in new_num_cols:
        if col not in df.columns:
            continue
        if df[col].min() > 0:
            df[f"{col}_log"] = np.log1p(df[col])

    # --- Interaction features: limit to top 3 numeric columns only ---
    top_num = new_num_cols[:3]
    for i in range(len(top_num)):
        for j in range(i + 1, len(top_num)):
            c1, c2 = top_num[i], top_num[j]
            if c1 in df.columns and c2 in df.columns:
                df[f"{c1}_x_{c2}"] = df[c1] * df[c2]

    # --- Categorical x Numeric interactions ---
    for ccol in cat_cols:
        if ccol not in df.columns:
            continue
        for ncol in new_num_cols[:3]:
            if ncol in df.columns:
                df[f"{ccol}_x_{ncol}"] = df[ccol] * df[ncol]

    return df, fit_params


# Compute fit parameters from training data only
fit_params = {"cat_mappings": {}}
train_features, fit_params = engineer_features(
    train_features, cat_cols, num_cols, date_cols, fit_params
)
test_features, _ = engineer_features(
    test_features, cat_cols, num_cols, date_cols, fit_params
)

# ============================================================
# 4b. CREATE TRAIN/VALIDATION SPLIT (before target encoding)
# ============================================================
train_part, val_part, y_train, y_val = train_test_split(
    train_features, y, test_size=0.2, random_state=42
)
train_part = train_part.reset_index(drop=True)
val_part = val_part.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
y_val = y_val.reset_index(drop=True)

print(
    f"Train part: {train_part.shape}, Val part: {val_part.shape}, Test: {test_features.shape}"
)
print(f"Target stats - Train: mean={y_train.mean():.4f}, std={y_train.std():.4f}")
print(f"Target stats - Val: mean={y_val.mean():.4f}, std={y_val.std():.4f}")

# ============================================================
# 4c. TARGET ENCODING (fit on train_part only, apply to val/test)
# ============================================================
target_enc_cols = ['ID', 'NAME', 'ADDRESS', 'CITY', 'ZIP', 'TELEPHONE', 'CNTRL_AREA', 'PLAN_AREA', 'HOLDING_CO']
target_enc_cols = [c for c in target_enc_cols if c in train_part.columns]

SMOOTHING_M = 10
global_mean = y_train.mean()

# Fit target encoding on train_part, apply OOF-style to train_part and directly to val/test
n_splits_te = 5
kf = np.array_split(np.arange(len(train_part)), n_splits_te)

for te_col in target_enc_cols:
    if te_col not in train_part.columns:
        continue
    # OOF encoding for train_part
    te_series = pd.Series(np.zeros(len(train_part)), index=train_part.index)
    for fold_idx in range(n_splits_te):
        val_idx = kf[fold_idx]
        tr_idx = np.concatenate([kf[i] for i in range(n_splits_te) if i != fold_idx])

        fold_data = train_part.iloc[tr_idx]
        fold_target = y_train.iloc[tr_idx]

        group_stats = fold_target.groupby(fold_data[te_col]).agg(['mean', 'count'])
        smoothed = (group_stats['mean'] * group_stats['count'] + global_mean * SMOOTHING_M) / (group_stats['count'] + SMOOTHING_M)

        val_fold_data = train_part.iloc[val_idx]
        te_series.iloc[val_idx] = val_fold_data[te_col].map(smoothed).fillna(global_mean).values

    train_part[f'{te_col}_target_enc'] = te_series.values

    # Full train_part stats for val and test
    full_stats = y_train.groupby(train_part[te_col]).agg(['mean', 'count'])
    smoothed_full = (full_stats['mean'] * full_stats['count'] + global_mean * SMOOTHING_M) / (full_stats['count'] + SMOOTHING_M)
    val_part[f'{te_col}_target_enc'] = val_part[te_col].map(smoothed_full).fillna(global_mean).values
    test_features[f'{te_col}_target_enc'] = test_features[te_col].map(smoothed_full).fillna(global_mean).values

print(f"Target encoding applied to {len(target_enc_cols)} columns")
print(f"After feature engineering: Train {train_part.shape}, Test {test_features.shape}")

# ============================================================
# 5. HANDLE MISSING VALUES (fit on train_part only)
# ============================================================
num_cols_after = [
    c
    for c in train_part.columns
    if pd.api.types.is_numeric_dtype(train_part[c])
]
cat_cols_after = [
    c
    for c in train_part.columns
    if not pd.api.types.is_numeric_dtype(train_part[c])
]

if num_cols_after:
    imputer = SimpleImputer(strategy="median")
    train_part[num_cols_after] = imputer.fit_transform(
        train_part[num_cols_after]
    )
    val_part[num_cols_after] = imputer.transform(val_part[num_cols_after])
    test_features[num_cols_after] = imputer.transform(test_features[num_cols_after])

if cat_cols_after:
    cat_imputer = SimpleImputer(strategy="most_frequent")
    train_part[cat_cols_after] = cat_imputer.fit_transform(
        train_part[cat_cols_after]
    )
    val_part[cat_cols_after] = cat_imputer.transform(val_part[cat_cols_after])
    test_features[cat_cols_after] = cat_imputer.transform(test_features[cat_cols_after])

print(
    f"Missing after imputation - Train: {train_part.isnull().sum().sum()}, Val: {val_part.isnull().sum().sum()}, Test: {test_features.isnull().sum().sum()}"
)

feature_columns = train_part.columns.tolist()

# ============================================================
# 7. TRAIN LIGHTGBM WITH EARLY STOPPING
# ============================================================
print("\n--- Training LightGBM ---")

lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.01,
    "num_leaves": 31,
    "max_depth": 5,
    "min_child_samples": 20,
    "min_child_weight": 5,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "n_estimators": 5000,
    "verbose": -1,
    "random_state": 42,
    "n_jobs": -1,
}

model_lgb = lgb.LGBMRegressor(**lgb_params)
model_lgb.fit(
    train_part[feature_columns],
    y_train,
    eval_set=[(val_part[feature_columns], y_val)],
    eval_metric="rmse",
    callbacks=[lgb.early_stopping(stopping_rounds=200, verbose=False)],
)
best_iter_lgb = model_lgb.best_iteration_
print(f"LightGBM best iteration: {best_iter_lgb}")

# ============================================================
# 8. TRAIN XGBOOST WITH EARLY STOPPING
# ============================================================
print("\n--- Training XGBoost ---")

model_xgb = xgb.XGBRegressor(
    max_depth=4,
    learning_rate=0.01,
    subsample=0.8,
    colsample_bytree=0.6,
    reg_alpha=0.5,
    reg_lambda=2.0,
    n_estimators=5000,
    early_stopping_rounds=200,
    random_state=42,
    n_jobs=-1,
    verbosity=0,
)
model_xgb.fit(
    train_part[feature_columns],
    y_train,
    eval_set=[(val_part[feature_columns], y_val)],
    verbose=False,
)
best_iter_xgb = model_xgb.best_iteration
print(f"XGBoost best iteration: {best_iter_xgb}")

# ============================================================
# 9. GENERATE PREDICTIONS FROM EACH MODEL
# ============================================================
val_pred_lgb = model_lgb.predict(val_part[feature_columns])
val_pred_xgb = model_xgb.predict(val_part[feature_columns])

test_pred_lgb = model_lgb.predict(test_features[feature_columns])
test_pred_xgb = model_xgb.predict(test_features[feature_columns])

# ============================================================
# 10. EVALUATE INDIVIDUAL MODELS ON VALIDATION
# ============================================================
val_rmse_lgb = np.sqrt(mean_squared_error(y_val, val_pred_lgb))
val_rmse_xgb = np.sqrt(mean_squared_error(y_val, val_pred_xgb))
val_rmse_mean = np.sqrt(mean_squared_error(y_val, np.full(len(y_val), y_train.mean())))

print(f"\n--- Individual Model RMSE ---")
print(f"LightGBM:  {val_rmse_lgb:.6f}")
print(f"XGBoost:   {val_rmse_xgb:.6f}")
print(f"Mean base: {val_rmse_mean:.6f}")

# ============================================================
# 11. OPTIMAL BLEND WEIGHT SEARCH ON VALIDATION
# ============================================================
best_w = 0.5
best_blend_rmse = float("inf")

for w in np.linspace(0.0, 1.0, 101):
    blend_pred = w * val_pred_lgb + (1 - w) * val_pred_xgb
    blend_rmse = np.sqrt(mean_squared_error(y_val, blend_pred))
    if blend_rmse < best_blend_rmse:
        best_blend_rmse = blend_rmse
        best_w = w

print(f"\n--- Blend Optimization ---")
print(f"Best weight (LGB): {best_w:.3f}")
print(f"Blend RMSE: {best_blend_rmse:.6f}")

# ============================================================
# 12. SELECT BEST STRATEGY
# ============================================================
candidates = {
    "lgb": (val_rmse_lgb, test_pred_lgb),
    "xgb": (val_rmse_xgb, test_pred_xgb),
    "blend": (best_blend_rmse, best_w * test_pred_lgb + (1 - best_w) * test_pred_xgb),
}

best_strategy = min(candidates, key=lambda k: candidates[k][0])
best_val_rmse = candidates[best_strategy][0]
test_pred = candidates[best_strategy][1]

print(f"\nBest strategy: {best_strategy} (RMSE: {best_val_rmse:.6f})")

# ============================================================
# 13. POST-PROCESSING (consistent for val and test)
# ============================================================
test_pred = np.maximum(test_pred, 0)

if best_strategy == "blend":
    val_pred_final = np.maximum(
        best_w * val_pred_lgb + (1 - best_w) * val_pred_xgb, 0
    )
elif best_strategy == "lgb":
    val_pred_final = np.maximum(val_pred_lgb, 0)
else:
    val_pred_final = np.maximum(val_pred_xgb, 0)

final_val_rmse = np.sqrt(mean_squared_error(y_val, val_pred_final))

# ============================================================
# 14. SAVE SUBMISSION
# ============================================================
submission = pd.DataFrame({"id": test_ids, "TOTAL_MWH": test_pred})
submission.to_csv("./submission/submission.csv", index=False)
print(f"\nSubmission saved: {submission.shape}")
print(
    f"Prediction stats: mean={test_pred.mean():.4f}, std={test_pred.std():.4f}, min={test_pred.min():.4f}, max={test_pred.max():.4f}"
)

# ============================================================
# 15. FINAL VALIDATION SCORE
# ============================================================
print(f"Final Validation Score: {final_val_rmse}")