import numpy as np
import pandas as pd
import lightgbm as lgb
import os
import json
import warnings
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import root_mean_squared_error

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

# ============================================================
# 2. IDENTIFY FEATURE TYPES
# ============================================================
target_col = "funding"
id_col = "id" if "id" in train.columns else None

numeric_cols = train.select_dtypes(include=[np.number]).columns.tolist()
if target_col in numeric_cols:
    numeric_cols.remove(target_col)
if id_col and id_col in numeric_cols:
    numeric_cols.remove(id_col)

categorical_cols = train.select_dtypes(include=["object", "category"]).columns.tolist()
if id_col and id_col in categorical_cols:
    categorical_cols.remove(id_col)

# Check for low-cardinality numeric columns that might be categorical
for col in numeric_cols[:]:
    if train[col].nunique() < 20 and train[col].nunique() < len(train) * 0.3:
        categorical_cols.append(col)
        numeric_cols.remove(col)

print(f"Numeric features ({len(numeric_cols)}): {numeric_cols}")
print(f"Categorical features ({len(categorical_cols)}): {categorical_cols}")

# ============================================================
# 3. SPLIT DATA FIRST (before any stateful transformation)
# ============================================================
y = train[target_col].values
X_train_full = train.drop(columns=[target_col] + ([id_col] if id_col else []))
X_test = test.drop(columns=[id_col] if id_col else [])

X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_full, y, test_size=0.15, random_state=42
)
X_tr = X_tr.reset_index(drop=True)
X_val = X_val.reset_index(drop=True)

print(f"Train split: {X_tr.shape}, Val split: {X_val.shape}, Test: {X_test.shape}")

# ============================================================
# 4. FEATURE ENGINEERING (fit on train only)
# ============================================================

# --- 4a: Handle missing values ---
num_imputer = SimpleImputer(strategy="median")
cat_imputer = SimpleImputer(strategy="most_frequent")

# Add missing indicators
for col in X_tr.columns:
    if X_tr[col].isnull().any():
        X_tr[f"{col}_missing"] = X_tr[col].isnull().astype(int)
        X_val[f"{col}_missing"] = X_val[col].isnull().astype(int)
        X_test[f"{col}_missing"] = X_test[col].isnull().astype(int)

# Impute
num_cols_in_tr = [
    c
    for c in X_tr.columns
    if X_tr[c].dtype in [np.float64, np.int64, np.float32, np.int32]
]
cat_cols_in_tr = [
    c
    for c in X_tr.columns
    if X_tr[c].dtype == object or str(X_tr[c].dtype) == "category"
]

if num_cols_in_tr:
    X_tr[num_cols_in_tr] = num_imputer.fit_transform(X_tr[num_cols_in_tr])
    X_val[num_cols_in_tr] = num_imputer.transform(X_val[num_cols_in_tr])
    X_test[num_cols_in_tr] = num_imputer.transform(X_test[num_cols_in_tr])

if cat_cols_in_tr:
    X_tr[cat_cols_in_tr] = cat_imputer.fit_transform(X_tr[cat_cols_in_tr])
    X_val[cat_cols_in_tr] = cat_imputer.transform(X_val[cat_cols_in_tr])
    X_test[cat_cols_in_tr] = cat_imputer.transform(X_test[cat_cols_in_tr])

# --- 4b: Encode categoricals ---
for col in categorical_cols:
    if col not in X_tr.columns:
        continue
    nunique = X_tr[col].nunique()
    if nunique <= 15:
        # One-hot encode
        dummies = pd.get_dummies(X_tr[col], prefix=col, drop_first=True, dtype=float)
        X_tr = pd.concat([X_tr, dummies], axis=1)
        dummies_val = pd.get_dummies(
            X_val[col], prefix=col, drop_first=True, dtype=float
        )
        dummies_test = pd.get_dummies(
            X_test[col], prefix=col, drop_first=True, dtype=float
        )
        for c in dummies.columns:
            if c not in dummies_val.columns:
                dummies_val[c] = 0
            if c not in dummies_test.columns:
                dummies_test[c] = 0
        X_val = pd.concat([X_val, dummies_val[dummies.columns]], axis=1)
        X_test = pd.concat([X_test, dummies_test[dummies.columns]], axis=1)
        X_tr = X_tr.drop(columns=[col])
        X_val = X_val.drop(columns=[col])
        X_test = X_test.drop(columns=[col])
    else:
        # Target encoding (fit on train only)
        global_mean = y_tr.mean()
        temp_df = pd.DataFrame({"col": X_tr[col].values, "target": y_tr})
        target_means = temp_df.groupby("col")["target"].agg(["mean", "count"])
        min_samples = 10
        X_tr[f"{col}_target_enc"] = X_tr[col].map(
            target_means["mean"].where(
                target_means["count"] >= min_samples, global_mean
            )
        )
        X_val[f"{col}_target_enc"] = (
            X_val[col]
            .map(
                target_means["mean"].where(
                    target_means["count"] >= min_samples, global_mean
                )
            )
            .fillna(global_mean)
        )
        X_test[f"{col}_target_enc"] = (
            X_test[col]
            .map(
                target_means["mean"].where(
                    target_means["count"] >= min_samples, global_mean
                )
            )
            .fillna(global_mean)
        )
        X_tr = X_tr.drop(columns=[col])
        X_val = X_val.drop(columns=[col])
        X_test = X_test.drop(columns=[col])

# --- 4c: Numeric feature engineering ---
for col in numeric_cols:
    if col not in X_tr.columns:
        continue
    skew = X_tr[col].skew()
    if abs(skew) > 1.0:
        X_tr[col] = np.sign(X_tr[col]) * np.log1p(np.abs(X_tr[col]))
        X_val[col] = np.sign(X_val[col]) * np.log1p(np.abs(X_val[col]))
        X_test[col] = np.sign(X_test[col]) * np.log1p(np.abs(X_test[col]))

# Interaction features
num_features_in_tr = [
    c
    for c in X_tr.columns
    if X_tr[c].dtype in [np.float64, np.float32, np.int64, np.int32]
]
if len(num_features_in_tr) >= 2:
    top_num = num_features_in_tr[:5]
    for i in range(len(top_num)):
        for j in range(i + 1, len(top_num)):
            a, b = top_num[i], top_num[j]
            X_tr[f"{a}_x_{b}"] = X_tr[a] * X_tr[b]
            X_val[f"{a}_x_{b}"] = X_val[a] * X_val[b]
            X_test[f"{a}_x_{b}"] = X_test[a] * X_test[b]
            X_tr[f"{a}_div_{b}"] = X_tr[a] / X_tr[b].replace(0, np.nan)
            X_val[f"{a}_div_{b}"] = X_val[a] / X_val[b].replace(0, np.nan)
            X_test[f"{a}_div_{b}"] = X_test[a] / X_test[b].replace(0, np.nan)

# --- 4d: Count encoding for categoricals ---
for col in categorical_cols:
    if col not in X_tr.columns:
        continue
    counts = X_tr[col].value_counts()
    X_tr[f"{col}_count_enc"] = X_tr[col].map(counts).fillna(0)
    X_val[f"{col}_count_enc"] = X_val[col].map(counts).fillna(0)
    X_test[f"{col}_count_enc"] = X_test[col].map(counts).fillna(0)

# --- 4e: Scale numeric features ---
final_num_cols = [
    c
    for c in X_tr.columns
    if X_tr[c].dtype in [np.float64, np.float32, np.int64, np.int32]
]
scaler = StandardScaler()
X_tr[final_num_cols] = scaler.fit_transform(X_tr[final_num_cols])
X_val[final_num_cols] = scaler.transform(X_val[final_num_cols])
X_test[final_num_cols] = scaler.transform(X_test[final_num_cols])

# --- 4f: Handle remaining NaNs ---
X_tr = X_tr.fillna(0)
X_val = X_val.fillna(0)
X_test = X_test.fillna(0)

# Convert to numpy
X_train_np = X_tr.values.astype(np.float32)
X_val_np = X_val.values.astype(np.float32)
X_test_np = X_test.values.astype(np.float32)

# Target: log1p for modeling
y_train_log = np.log1p(np.maximum(y_tr, 0))
y_val_log = np.log1p(np.maximum(y_val, 0))

print(f"\nFinal feature matrix shapes:")
print(f"  Train: {X_train_np.shape}")
print(f"  Val:   {X_val_np.shape}")
print(f"  Test:  {X_test_np.shape}")

# ============================================================
# 5. LIGHTGBM PARAMETERS
# ============================================================
# Use standard regression objective in log-space (numerically stable).
# Predictions are transformed back via expm1 for RMSE evaluation on original scale.
lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "max_depth": 6,
    "learning_rate": 0.03,
    "min_child_samples": 10,
    "min_child_weight": 1.0,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.7,
    "reg_alpha": 1.0,
    "reg_lambda": 5.0,
    "min_split_gain": 0.0,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}

# ============================================================
# 7. K-FOLD CROSS-VALIDATION
# ============================================================
print("\n--- 5-Fold Cross-Validation ---")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []
best_iterations = []

for fold, (tr_idx, va_idx) in enumerate(kf.split(X_train_np)):
    X_tr_cv = X_train_np[tr_idx]
    X_va_cv = X_train_np[va_idx]
    y_tr_cv = y_train_log[tr_idx]
    y_va_cv = y_train_log[va_idx]

    train_data = lgb.Dataset(X_tr_cv, label=y_tr_cv, free_raw_data=False)
    val_data = lgb.Dataset(
        X_va_cv, label=y_va_cv, reference=train_data, free_raw_data=False
    )

    callbacks = [
        lgb.early_stopping(stopping_rounds=100, verbose=False),
        lgb.log_evaluation(period=1000),
    ]

    model = lgb.train(
        lgb_params,
        train_data,
        num_boost_round=5000,
        valid_sets=[val_data],
        valid_names=["valid"],
        callbacks=callbacks,
    )

    best_iter = model.best_iteration
    best_score = model.best_score["valid"]["rmse"]
    cv_scores.append(best_score)
    best_iterations.append(best_iter)
    print(f"  Fold {fold+1}: RMSE={best_score:.4f}, best_iter={best_iter}")

print(f"\nCV RMSE: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
print(f"Mean best iteration: {np.mean(best_iterations):.0f}")

# ============================================================
# 8. TRAIN FINAL MODEL ON ALL TRAINING DATA
# ============================================================
print("\n--- Training Final Model (all train data) ---")
avg_best_iter = int(np.mean(best_iterations))
final_num_rounds = min(avg_best_iter + 50, 5000)

train_data = lgb.Dataset(X_train_np, label=y_train_log, free_raw_data=False)
val_data = lgb.Dataset(
    X_val_np, label=y_val_log, reference=train_data, free_raw_data=False
)

callbacks = [
    lgb.early_stopping(stopping_rounds=150, verbose=False),
    lgb.log_evaluation(period=1000),
]

final_model = lgb.train(
    lgb_params,
    train_data,
    num_boost_round=final_num_rounds,
    valid_sets=[train_data, val_data],
    valid_names=["train", "valid"],
    callbacks=callbacks,
)

print(f"Final model best iteration: {final_model.best_iteration}")

# ============================================================
# 9. PREDICTIONS THROUGH MODEL INFERENCE
# ============================================================
# Validation predictions
val_preds_log = final_model.predict(X_val_np, num_iteration=final_model.best_iteration)
val_preds_orig = np.expm1(val_preds_log)
val_preds_orig = np.maximum(val_preds_orig, 0)

# Test predictions (IDENTICAL processing logic)
test_preds_log = final_model.predict(
    X_test_np, num_iteration=final_model.best_iteration
)
test_preds_orig = np.expm1(test_preds_log)
test_preds_orig = np.maximum(test_preds_orig, 0)

# ============================================================
# 10. EVALUATE ON VALIDATION SET (RMSE on original scale = official metric)
# ============================================================
val_rmse = root_mean_squared_error(y_val, val_preds_orig)
print(f"\nValidation RMSE (original scale): {val_rmse:.4f}")
print(f"CV RMSE (5-fold): {np.mean(cv_scores):.4f}")

# Feature importance
importance = final_model.feature_importance(importance_type="gain")
feature_names = list(X_tr.columns)
top_features = sorted(zip(feature_names, importance), key=lambda x: x[1], reverse=True)[
    :10
]
print("\nTop 10 features by gain:")
for name, imp in top_features:
    print(f"  {name}: {imp:.1f}")

# ============================================================
# 11. SAVE MODEL
# ============================================================
final_model.save_model("./working/final_lgb_model.txt")
print("\nModel saved to ./working/final_lgb_model.txt")

# ============================================================
# 12. GENERATE SUBMISSION
# ============================================================
test_ids = test[id_col].values if id_col else np.arange(len(test))
submission = pd.DataFrame({"id": test_ids.astype(int), "funding": test_preds_orig})
submission.to_csv("./submission/submission.csv", index=False)
print(f"\nSubmission saved: {submission.shape}")
print(submission.head())

# ============================================================
# 13. FINAL SCORE
# ============================================================
score = val_rmse
print(f"Final Validation Score: {score}")