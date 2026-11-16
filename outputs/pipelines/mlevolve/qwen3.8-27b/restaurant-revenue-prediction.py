import pandas as pd
import numpy as np
import os
import json
import copy
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error

# ============================================================
# 1. LOAD DATA
# ============================================================
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")

# ============================================================
# 2. IDENTIFY COLUMNS
# ============================================================
target_col = "revenue"
id_col = "Id" if "Id" in train_df.columns else None

p_cols = [c for c in train_df.columns if c.startswith("P") and c[1:].isdigit()]
p_cols.sort(key=lambda x: int(x[1:]))
print(f"P features: {len(p_cols)}")

cat_cols = ["City", "City Group", "Type"]
cat_cols = [c for c in cat_cols if c in train_df.columns]

date_col = "Open Date" if "Open Date" in train_df.columns else None

# ============================================================
# 3. SPLIT TRAIN INTO TRAIN/VAL
# ============================================================
train_split, val_split = train_test_split(train_df, test_size=0.2, random_state=42)

print(f"Train split: {train_split.shape}, Val split: {val_split.shape}")


# ============================================================
# 4. FEATURE ENGINEERING
# ============================================================
def engineer_features(df, train_ref, is_train=True):
    df = df.copy()

    # --- Date Features ---
    if date_col is not None and date_col in df.columns:
        df["Open Date"] = pd.to_datetime(
            df[date_col], format="%m/%d/%Y", errors="coerce"
        )
        df["open_year"] = df["Open Date"].dt.year
        df["open_month"] = df["Open Date"].dt.month
        df["open_day_of_week"] = df["Open Date"].dt.dayofweek
        df["years_since_open"] = 2024 - df["open_year"]
        df["years_since_open"] = df["years_since_open"].clip(lower=0)

    # --- Type Encoding ---
    if "Type" in df.columns:
        type_map = {"IL": 0, "DT": 1, "FC": 2}
        df["Type_ordinal"] = df["Type"].map(type_map).fillna(0).astype(int)

    # --- City Group encoding ---
    if "City Group" in df.columns:
        df["City Group_enc"] = (df["City Group"] == "Big Cities").astype(int)

    # --- P feature aggregations (keep only key ones) ---
    available_p = [c for c in p_cols if c in df.columns]

    df["P_mean"] = df[available_p].mean(axis=1)
    df["P_std"] = df[available_p].std(axis=1).fillna(0)
    df["P_range"] = df[available_p].max(axis=1) - df[available_p].min(axis=1)

    # --- Log-transform target ---
    if target_col in df.columns:
        df["log_revenue"] = np.log1p(df[target_col])

    return df


train_feat = engineer_features(train_split, train_split, is_train=True)
val_feat = engineer_features(val_split, train_split, is_train=False)
test_feat = engineer_features(test_df, train_split, is_train=False)

# ============================================================
# 5. PREPARE FEATURE MATRICES
# ============================================================
exclude_cols = [
    target_col,
    "log_revenue",
    "Id",
    "City",
    "City Group",
    "Type",
    "Open Date",
]
feature_cols = [c for c in train_feat.columns if c not in exclude_cols]
print(f"Number of features: {len(feature_cols)}")

# Handle NaN
for col in feature_cols:
    if train_feat[col].isna().any():
        fill_val = train_feat[col].median()
        train_feat[col] = train_feat[col].fillna(fill_val)
        val_feat[col] = val_feat[col].fillna(fill_val)
        test_feat[col] = test_feat[col].fillna(fill_val)

X_train = train_feat[feature_cols].values.astype(np.float32)
X_val = val_feat[feature_cols].values.astype(np.float32)
X_test = test_feat[feature_cols].values.astype(np.float32)

y_train = np.log1p(train_feat[target_col].values).astype(np.float32)
y_val_orig = val_feat[target_col].values.astype(np.float32)

# Also prepare full training data for final model
full_feat = engineer_features(train_df, train_df, is_train=True)
for col in feature_cols:
    if full_feat[col].isna().any():
        fill_val = train_feat[col].median()
        full_feat[col] = full_feat[col].fillna(fill_val)
X_full = full_feat[feature_cols].values.astype(np.float32)
y_full = np.log1p(full_feat[target_col].values).astype(np.float32)


# ============================================================
# 7. MODEL DEFINITIONS (LightGBM + XGBoost)
# ============================================================
lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "min_child_samples": 5,
    "num_leaves": 15,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "verbose": -1,
    "n_estimators": 2000,
    "random_state": 42,
}

xgb_params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "min_child_weight": 5,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "n_estimators": 2000,
    "random_state": 42,
    "verbosity": 0,
}

# ============================================================
# 8. 5-FOLD CROSS-VALIDATION
# ============================================================
print("\n=== 5-Fold Cross-Validation ===")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_full)):
    X_tr, X_vl = X_full[train_idx], X_full[val_idx]
    y_tr, y_vl = y_full[train_idx], y_full[val_idx]

    # LightGBM
    lgb_model = lgb.LGBMRegressor(**lgb_params)
    lgb_model.fit(
        X_tr, y_tr,
        eval_set=[(X_vl, y_vl)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    lgb_pred = lgb_model.predict(X_vl)

    # XGBoost
    xgb_model = xgb.XGBRegressor(**xgb_params)
    xgb_model.fit(
        X_tr, y_tr,
        verbose=False,
    )
    xgb_pred = xgb_model.predict(X_vl)

    # Ensemble prediction in log space
    ensemble_pred_log = (lgb_pred + xgb_pred) / 2.0
    ensemble_pred = np.expm1(ensemble_pred_log)
    y_vl_orig = np.expm1(y_vl)

    fold_rmse = np.sqrt(mean_squared_error(y_vl_orig, np.maximum(ensemble_pred, 0)))
    cv_scores.append(fold_rmse)
    print(f"  Fold {fold+1}: RMSE = {fold_rmse:.0f}")

print(f"  CV Mean RMSE: {np.mean(cv_scores):.0f} (+/- {np.std(cv_scores):.0f})")

# ============================================================
# 9. TRAIN FINAL MODELS ON ALL DATA
# ============================================================
print("\n=== Training Final Models on All Data ===")
final_lgb = lgb.LGBMRegressor(**lgb_params)
final_lgb.fit(X_full, y_full)

final_xgb = xgb.XGBRegressor(**xgb_params)
final_xgb.fit(X_full, y_full)

# ============================================================
# 10. FINAL VALIDATION SCORE (same 80/20 split)
# ============================================================
val_pred_lgb = final_lgb.predict(X_val)
val_pred_xgb = final_xgb.predict(X_val)
val_pred_log = (val_pred_lgb + val_pred_xgb) / 2.0
val_pred = np.expm1(val_pred_log)
val_pred = np.maximum(val_pred, 0)
final_rmse = np.sqrt(mean_squared_error(y_val_orig, val_pred))

print(f"\nFinal Validation RMSE (80/20 split): {final_rmse:.2f}")

# ============================================================
# 11. GENERATE TEST PREDICTIONS
# ============================================================
test_pred_lgb = final_lgb.predict(X_test)
test_pred_xgb = final_xgb.predict(X_test)
test_pred_log = (test_pred_lgb + test_pred_xgb) / 2.0
test_pred = np.expm1(test_pred_log)
test_pred = np.maximum(test_pred, 0)
print(
    f"\nTest predictions: shape={test_pred.shape}, range=[{test_pred.min():.0f}, {test_pred.max():.0f}]"
)
print(f"Test predictions mean: {test_pred.mean():.0f}")

# ============================================================
# 12. SAVE SUBMISSION
# ============================================================
os.makedirs("./submission", exist_ok=True)
submission = pd.DataFrame({"Id": range(len(test_pred)), "revenue": test_pred})
submission.to_csv("./submission/submission.csv", index=False)
print(f"Submission saved: {submission.shape}")
print(f"Submission head:\n{submission.head()}")

# ============================================================
# 13. FINAL OUTPUT
# ============================================================
print(f"\n=== FINAL RESULTS ===")
print(f"Validation split: {len(y_val_orig)} samples")
print(f"Official metric: RMSE (lower is better)")
print(f"Final Validation Score: {final_rmse:.4f}")