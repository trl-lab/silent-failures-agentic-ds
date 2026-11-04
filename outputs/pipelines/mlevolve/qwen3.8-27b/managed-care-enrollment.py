import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import os
import json
import warnings

warnings.filterwarnings("ignore")

# Create directories
os.makedirs("./submission", exist_ok=True)
os.makedirs("./working", exist_ok=True)

# ============================================================
# 1. LOAD DATA
# ============================================================
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")
print(f"\nTrain columns: {list(train.columns)}")
print(f"\nTrain dtypes:\n{train.dtypes}")
print(f"\nTrain head:\n{train.head()}")
print(f"\nTrain describe:\n{train.describe()}")
print(f"\nMissing values in train:\n{train.isnull().sum()}")
print(f"\nMissing values in test:\n{test.isnull().sum()}")

# ============================================================
# 2. IDENTIFY COLUMNS
# ============================================================
TARGET = "Total Enrollment"

if TARGET not in train.columns:
    possible_targets = [
        c for c in train.columns if "total" in c.lower() or "enroll" in c.lower()
    ]
    print(f"Possible targets: {possible_targets}")
    TARGET = [c for c in train.columns if c == "Total Enrollment"][0]

id_col = None
for col in train.columns:
    if col.lower() in ["id", "index", "row_id", "sample_id"]:
        id_col = col
        break

if id_col is None:
    if "id" in train.columns:
        id_col = "id"
    else:
        train["id"] = range(len(train))
        test["id"] = range(len(test))
        id_col = "id"

print(f"\nID column: {id_col}")
print(f"Target column: {TARGET}")

# Separate target
y_full = train[TARGET].copy()
train = train.drop(columns=[TARGET])

feature_cols = [c for c in train.columns if c != id_col]
print(f"\nFeature columns ({len(feature_cols)}): {feature_cols}")

# ============================================================
# 3. IDENTIFY FEATURE TYPES
# ============================================================
cat_cols = []
num_cols = []

for col in feature_cols:
    if (
        train[col].dtype == object
        or train[col].dtype == "category"
        or str(train[col].dtype) == "category"
    ):
        cat_cols.append(col)
    else:
        num_cols.append(col)

print(f"\nCategorical columns ({len(cat_cols)}): {cat_cols}")
print(f"Numerical columns ({len(num_cols)}): {num_cols}")

# ============================================================
# 4. CREATE TRAIN/VALIDATION SPLIT (BEFORE feature engineering)
# ============================================================
val_size = 0.15
train_idx, val_idx = train_test_split(
    range(len(train)), test_size=val_size, random_state=42, shuffle=True
)

X_train_raw = train.iloc[train_idx].reset_index(drop=True)
y_train = y_full.iloc[train_idx].reset_index(drop=True)

X_val_raw = train.iloc[val_idx].reset_index(drop=True)
y_val = y_full.iloc[val_idx].reset_index(drop=True)

X_test_raw = test.reset_index(drop=True)
test_ids = test[id_col].values

print(f"\nSplit sizes: Train={len(X_train_raw)}, Val={len(X_val_raw)}, Test={len(X_test_raw)}")
print(f"Target stats - Train: mean={y_train.mean():.2f}, std={y_train.std():.2f}")
print(f"Target stats - Val:   mean={y_val.mean():.2f}, std={y_val.std():.2f}")

# ============================================================
# 5. FEATURE ENGINEERING (fit on train split only)
# ============================================================


def engineer_features(
    df, train_ref, cat_cols, num_cols, fit=False, cat_maps=None, freq_maps=None
):
    """Apply feature engineering. If fit=True, compute stats from train_ref."""
    df = df.copy()

    # --- Log transform for skewed numerical features ---
    # Always use train_ref for skew decision to ensure consistent features
    for col in num_cols:
        if col in df.columns and col in train_ref.columns:
            vals = train_ref[col].dropna()
            if len(vals) > 0 and vals.abs().mean() > 0:
                skew = vals.skew()
                if abs(skew) > 1.0:
                    df[f"{col}_log"] = np.log1p(df[col].clip(lower=0))

    # --- Ratio features between enrollment-type columns ---
    # Use train_ref for max check to ensure consistency
    enroll_cols = [c for c in num_cols if c in df.columns and c in train_ref.columns and train_ref[c].max() > 100]

    if len(enroll_cols) >= 2:
        for i, c1 in enumerate(enroll_cols):
            for c2 in enroll_cols[i + 1 :]:
                ratio_name = f"{c1}_ratio_{c2}"
                denom = df[c2].replace(0, np.nan)
                df[ratio_name] = df[c1] / denom
        df["enroll_sum"] = df[enroll_cols].sum(axis=1)
        df["enroll_mean"] = df[enroll_cols].mean(axis=1)
        df["enroll_std"] = df[enroll_cols].std(axis=1)
        df["enroll_max"] = df[enroll_cols].max(axis=1)
        df["enroll_min"] = df[enroll_cols].min(axis=1)
        df["enroll_range"] = df["enroll_max"] - df["enroll_min"]

    # --- Interaction features between key numerical columns ---
    # Use train_ref for consistency
    key_num = [c for c in num_cols if c in df.columns and c in train_ref.columns][:10]
    if len(key_num) >= 2:
        for i, c1 in enumerate(key_num[:5]):
            for c2 in key_num[i + 1 : i + 3]:
                df[f"{c1}_x_{c2}"] = df[c1] * df[c2]

    # --- Polynomial features for important numericals ---
    for col in num_cols[:5]:
        if col in df.columns and col in train_ref.columns:
            df[f"{col}_sq"] = df[col] ** 2

    # --- Categorical encoding ---
    for col in cat_cols:
        if col in df.columns:
            if fit:
                freq = train_ref[col].value_counts(normalize=True)
                df[f"{col}_freq"] = df[col].map(freq).fillna(0)
                categories = train_ref[col].unique()
                cat_map = {c: i for i, c in enumerate(sorted(categories))}
                df[f"{col}_label"] = df[col].map(cat_map).fillna(-1).astype(int)
                # Store maps for later use
                if cat_maps is not None:
                    cat_maps[col] = cat_map
                if freq_maps is not None:
                    freq_maps[col] = freq
            else:
                if cat_maps is not None and col in cat_maps:
                    df[f"{col}_label"] = (
                        df[col].map(cat_maps[col]).fillna(-1).astype(int)
                    )
                if freq_maps is not None and col in freq_maps:
                    df[f"{col}_freq"] = df[col].map(freq_maps[col]).fillna(0)

    return df


# Impute missing values (fit on train split only)
train_clean = X_train_raw.copy()
val_clean = X_val_raw.copy()
test_clean = X_test_raw.copy()

for col in num_cols:
    if col in train_clean.columns:
        if train_clean[col].isnull().sum() > 0:
            median_val = train_clean[col].median()
            train_clean[col] = train_clean[col].fillna(median_val)
            if col in val_clean.columns:
                val_clean[col] = val_clean[col].fillna(median_val)
            if col in test_clean.columns:
                test_clean[col] = test_clean[col].fillna(median_val)

for col in cat_cols:
    if col in train_clean.columns:
        if train_clean[col].isnull().sum() > 0:
            mode_val = (
                train_clean[col].mode()[0]
                if len(train_clean[col].mode()) > 0
                else "UNKNOWN"
            )
            train_clean[col] = train_clean[col].fillna(mode_val)
            if col in val_clean.columns:
                val_clean[col] = val_clean[col].fillna(mode_val)
            if col in test_clean.columns:
                test_clean[col] = test_clean[col].fillna(mode_val)

# Engineer features (fit on train split only)
cat_maps = {}
freq_maps = {}
train_feat = engineer_features(
    train_clean,
    train_clean,
    cat_cols,
    num_cols,
    fit=True,
    cat_maps=cat_maps,
    freq_maps=freq_maps,
)
val_feat = engineer_features(
    val_clean,
    train_clean,
    cat_cols,
    num_cols,
    fit=False,
    cat_maps=cat_maps,
    freq_maps=freq_maps,
)
test_feat = engineer_features(
    test_clean,
    train_clean,
    cat_cols,
    num_cols,
    fit=False,
    cat_maps=cat_maps,
    freq_maps=freq_maps,
)

# Remove original categorical columns
for col in cat_cols:
    if col in train_feat.columns:
        train_feat = train_feat.drop(columns=[col])
    if col in val_feat.columns:
        val_feat = val_feat.drop(columns=[col])
    if col in test_feat.columns:
        test_feat = test_feat.drop(columns=[col])

# Handle remaining NaN and inf
train_feat = train_feat.fillna(0)
val_feat = val_feat.fillna(0)
test_feat = test_feat.fillna(0)
train_feat = train_feat.replace([np.inf, -np.inf], 0)
val_feat = val_feat.replace([np.inf, -np.inf], 0)
test_feat = test_feat.replace([np.inf, -np.inf], 0)

# Ensure val and test features have the same columns in the same order as train
val_feat = val_feat.reindex(columns=train_feat.columns, fill_value=0)
test_feat = test_feat.reindex(columns=train_feat.columns, fill_value=0)

print(f"\nAfter feature engineering:")
print(f"  Train features: {train_feat.shape[1]}")
print(f"  Val features: {val_feat.shape[1]}")
print(f"  Test features: {test_feat.shape[1]}")

X_train = train_feat
X_val = val_feat
X_test = test_feat

# ============================================================
# 6. TRAINING: Seed-Averaged LightGBM + Ridge Ensemble
# ============================================================

# --- Model 1: Ridge Regression (captures additive structure) ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

ridge_model = Ridge(alpha=10.0, fit_intercept=True)
ridge_model.fit(X_train_scaled, y_train)
ridge_val_pred = np.clip(ridge_model.predict(X_val_scaled), 0, None)
ridge_test_pred = np.clip(ridge_model.predict(X_test_scaled), 0, None)
ridge_rmse = np.sqrt(mean_squared_error(y_val, ridge_val_pred))
print(f"\nRidge RMSE: {ridge_rmse:.4f}")

# --- Model 2: Seed-Averaged LightGBM (reduces variance via multi-seed) ---
SEEDS = [42, 123, 456, 789, 1024]
lgb_val_preds = []
lgb_test_preds = []
lgb_best_iters = []

for seed in SEEDS:
    params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "learning_rate": 0.03,
        "num_leaves": 7,
        "max_depth": 3,
        "min_child_samples": 5,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.7,
        "reg_alpha": 1.0,
        "reg_lambda": 5.0,
        "n_estimators": 3000,
        "verbose": -1,
        "random_state": seed,
    }

    model = lgb.LGBMRegressor(**params)
    early_stop = lgb.early_stopping(stopping_rounds=100, verbose=False)

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[early_stop],
    )

    best_iter = model.best_iteration_
    lgb_best_iters.append(best_iter)

    val_pred = np.clip(model.predict(X_val), 0, None)
    test_pred = np.clip(model.predict(X_test), 0, None)

    lgb_val_preds.append(val_pred)
    lgb_test_preds.append(test_pred)

    single_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    print(f"  LGBM seed={seed}: best_iter={best_iter}, RMSE={single_rmse:.4f}")

# Average across seeds
lgb_val_pred_avg = np.mean(lgb_val_preds, axis=0)
lgb_test_pred_avg = np.mean(lgb_test_preds, axis=0)
lgb_rmse = np.sqrt(mean_squared_error(y_val, lgb_val_pred_avg))
print(f"LightGBM (seed-avg) RMSE: {lgb_rmse:.4f}")

# ============================================================
# 7. ENSEMBLE: Fixed equal-weight blend (no grid search on val)
# ============================================================
best_w = 0.5
ensemble_val_pred = np.clip(best_w * ridge_val_pred + (1 - best_w) * lgb_val_pred_avg, 0, None)
best_ensemble_rmse = np.sqrt(mean_squared_error(y_val, ensemble_val_pred))

print(f"\nEnsemble weight (Ridge fraction): {best_w:.2f}")
print(f"Ensemble validation RMSE: {best_ensemble_rmse:.4f}")

# ============================================================
# 8. SELECT BEST APPROACH
# ============================================================
candidates = {
    "Ridge": (ridge_rmse, ridge_test_pred),
    "LGBM_seed_avg": (lgb_rmse, lgb_test_pred_avg),
    "Ensemble": (
        best_ensemble_rmse,
        np.clip(best_w * ridge_test_pred + (1 - best_w) * lgb_test_pred_avg, 0, None),
    ),
}

best_name = min(candidates, key=lambda k: candidates[k][0])
best_rmse = candidates[best_name][0]
test_pred = candidates[best_name][1]

print(f"\nSelected model: {best_name} (RMSE={best_rmse:.4f})")

# ============================================================
# 9. GENERATE SUBMISSION
# ============================================================
test_pred = np.clip(test_pred, 0, None)
submission = pd.DataFrame({"id": test_ids.astype(int), "Total Enrollment": test_pred})
submission.to_csv("./submission/submission.csv", index=False)
print(f"\nSubmission saved: {submission.shape}")
print(submission.head())

# ============================================================
# 10. FINAL VALIDATION SCORE (official metric: RMSE)
# ============================================================
score = best_rmse
print(f"\nModel comparison:")
print(f"  Ridge: {ridge_rmse:.4f}")
print(f"  LightGBM (seed-avg): {lgb_rmse:.4f}")
print(f"  Ensemble (w={best_w:.2f}): {best_ensemble_rmse:.4f}")
print(f"  Selected: {best_name}")
print(f"Final Validation Score: {score}")