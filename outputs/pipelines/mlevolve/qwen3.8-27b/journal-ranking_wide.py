import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import os
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# 1. LOAD DATA
# ============================================================
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")

target_col = "H index"
id_col = "id" if "id" in train.columns else None

# Separate target
y_full = train[target_col].copy()
train = train.drop(columns=[target_col])

# Identify column types
numeric_cols = train.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = train.select_dtypes(include=["object", "category"]).columns.tolist()
bool_cols = [c for c in train.columns if train[c].dtype == bool]

if id_col and id_col in numeric_cols:
    numeric_cols.remove(id_col)
if id_col and id_col in categorical_cols:
    categorical_cols.remove(id_col)

print(f"Numeric features ({len(numeric_cols)}): {numeric_cols[:10]}...")
print(f"Categorical features ({len(categorical_cols)}): {categorical_cols[:10]}...")

# ============================================================
# 2. HANDLE MISSING VALUES (fit on train only)
# ============================================================
for col in numeric_cols:
    if train[col].isnull().any():
        train[col] = train[col].fillna(train[col].median())
    if test[col].isnull().any():
        test[col] = test[col].fillna(train[col].median())

for col in categorical_cols:
    if train[col].isnull().any():
        mode_val = train[col].mode()[0] if not train[col].mode().empty else "UNKNOWN"
        train[col] = train[col].fillna(mode_val)
    if test[col].isnull().any():
        mode_val = train[col].mode()[0] if not train[col].mode().empty else "UNKNOWN"
        test[col] = test[col].fillna(mode_val)

for col in bool_cols:
    train[col] = train[col].astype(int)
    test[col] = test[col].astype(int)
    if col not in numeric_cols:
        numeric_cols.append(col)

# ============================================================
# 3. FEATURE ENGINEERING (Target Encoding + Domain Features)
# ============================================================


def target_encode(train_df, test_df, col, target, n_splits=5, smoothing=10):
    """K-fold target encoding fitted ONLY on training data."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    global_mean = target.mean()
    train_enc = pd.Series(index=train_df.index, dtype=float)

    for tr_idx, va_idx in kf.split(train_df):
        tr_target = target.iloc[tr_idx]
        tr_col = train_df[col].iloc[tr_idx]

        stats = tr_target.groupby(tr_col).agg(["mean", "count"])
        smooth_mean = (stats["count"] * stats["mean"] + smoothing * global_mean) / (
            stats["count"] + smoothing
        )

        va_col = train_df[col].iloc[va_idx]
        train_enc.iloc[va_idx] = va_col.map(smooth_mean).fillna(global_mean).values

    # For test: use full training set statistics
    stats_full = target.groupby(train_df[col]).agg(["mean", "count"])
    smooth_mean_full = (
        stats_full["count"] * stats_full["mean"] + smoothing * global_mean
    ) / (stats_full["count"] + smoothing)
    test_enc = test_df[col].map(smooth_mean_full).fillna(global_mean)

    return train_enc, test_enc


print("\nEngineering features (target encoding + domain features)...")

train_fe = train.copy()
test_fe = test.copy()

# Log transforms for numeric features
for col in numeric_cols:
    if col not in train_fe.columns:
        continue
    if train_fe[col].min() >= 0:
        train_fe[f"{col}_log"] = np.log1p(train_fe[col].clip(lower=0))
        test_fe[f"{col}_log"] = np.log1p(test_fe[col].clip(lower=0))
    elif train_fe[col].min() > -abs(train_fe[col].max()) * 0.1:
        shift = abs(train_fe[col].min()) + 1
        train_fe[f"{col}_logshift"] = np.log1p(train_fe[col] + shift)
        test_fe[f"{col}_logshift"] = np.log1p(test_fe[col] + shift)

# Domain-specific features
if "Total Refs." in train_fe.columns and "Coverage" in train_fe.columns:
    # Coverage is categorical, convert to numeric label for this feature
    cov_map = {v: i for i, v in enumerate(sorted(train_fe["Coverage"].unique()))}
    train_fe["Total Refs. / (Coverage+1)"] = train_fe["Total Refs."] / (
        train_fe["Coverage"].map(cov_map).fillna(0) + 1
    )
    test_fe["Total Refs. / (Coverage+1)"] = test_fe["Total Refs."] / (
        test_fe["Coverage"].map(cov_map).fillna(0) + 1
    )

if "Total Refs." in train_fe.columns and "Sourceid" in train_fe.columns:
    train_fe["Total Refs * (1/Sourceid)"] = train_fe["Total Refs."] / (
        train_fe["Sourceid"].abs() + 1e-8
    )
    test_fe["Total Refs * (1/Sourceid)"] = test_fe["Total Refs."] / (
        test_fe["Sourceid"].abs() + 1e-8
    )

# Target encoding for categorical variables
target_te_cols = [
    "Type",
    "Coverage",
    "Categories",
    "Areas",
    "Country",
    "Publisher",
    "Region",
    "Issn",
]

for col in target_te_cols:
    if col in train_fe.columns and col in test_fe.columns:
        te_train, te_test = target_encode(
            train_fe, test_fe, col, y_full, n_splits=5, smoothing=10
        )
        train_fe[f"{col}_target"] = te_train.values
        test_fe[f"{col}_target"] = te_test.values

# Keep label encoding for categoricals (useful for tree models)
for col in categorical_cols:
    if col not in train_fe.columns:
        continue
    cat_map = {v: i for i, v in enumerate(sorted(train_fe[col].unique()))}
    train_fe[f"{col}_label"] = train_fe[col].map(cat_map).fillna(-1).astype(int)
    test_fe[f"{col}_label"] = test_fe[col].map(cat_map).fillna(-1).astype(int)

# Drop original categoricals
drop_cols = [c for c in categorical_cols if c in train_fe.columns]
train_fe = train_fe.drop(columns=drop_cols)
test_fe = test_fe.drop(columns=drop_cols)

print(f"After FE - Train: {train_fe.shape}, Test: {test_fe.shape}")

# ============================================================
# 4. FEATURE SCALING (StandardScaler - preserves ordinal relationships)
# ============================================================
final_numeric_cols = train_fe.select_dtypes(include=[np.number]).columns.tolist()
if id_col and id_col in final_numeric_cols:
    final_numeric_cols.remove(id_col)

print(f"Final numeric features: {len(final_numeric_cols)}")

scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_fe[final_numeric_cols])
test_scaled = scaler.transform(test_fe[final_numeric_cols])

train_final = pd.DataFrame(
    train_scaled, columns=final_numeric_cols, index=train_fe.index
)
test_final = pd.DataFrame(test_scaled, columns=final_numeric_cols, index=test_fe.index)

# ============================================================
# 5. 5-FOLD CROSS-VALIDATION SETUP
# ============================================================
y_full_aligned = y_full.reset_index(drop=True)
X_all = train_final.values
y_all = y_full_aligned.values
X_test = test_final.values

# Test IDs
if id_col and id_col in test.columns:
    test_ids = test[id_col].values
else:
    test_ids = np.arange(len(test))

print(f"Total train samples: {len(X_all)}, Test samples: {len(X_test)}")
print(f"Target: mean={y_all.mean():.4f}, std={y_all.std():.4f}")

# ============================================================
# 6. MODEL PARAMETERS (Two MSE LightGBM models for ensemble diversity)
# ============================================================
model_a_params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "num_leaves": 127,
    "max_depth": -1,
    "learning_rate": 0.03,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.85,
    "bagging_freq": 3,
    "min_child_samples": 10,
    "lambda_l2": 2.0,
    "verbosity": -1,
    "n_jobs": -1,
}

model_b_params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "max_depth": -1,
    "learning_rate": 0.05,
    "feature_fraction": 0.5,
    "bagging_fraction": 0.8,
    "bagging_freq": 3,
    "min_child_samples": 20,
    "lambda_l2": 1.0,
    "verbosity": -1,
    "n_jobs": -1,
}

# ============================================================
# 7. 5-FOLD CV + MULTI-SEED TRAINING
# ============================================================
n_folds = 5
seeds = [42, 123, 7, 2023, 999]
kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

cv_val_preds = np.zeros(len(y_all))
fold_best_iters_A = []
fold_best_iters_B = []

print("\n--- 5-Fold CV to determine best iterations ---")
for fold_idx, (tr_idx, va_idx) in enumerate(kf.split(X_all)):
    X_tr, y_tr = X_all[tr_idx], y_all[tr_idx]
    X_va, y_va = X_all[va_idx], y_all[va_idx]

    fold_val_preds = np.zeros(len(y_va))

    for seed in seeds:
        # Model A (complex)
        params_a = model_a_params.copy()
        params_a["seed"] = seed
        dtrain = lgb.Dataset(X_tr, label=y_tr, feature_name=list(final_numeric_cols))
        dval = lgb.Dataset(X_va, label=y_va, reference=dtrain)
        model_a = lgb.train(
            params_a,
            dtrain,
            num_boost_round=5000,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(stopping_rounds=150, verbose=False)],
        )
        pred_a = model_a.predict(X_va, num_iteration=model_a.best_iteration)
        fold_val_preds += pred_a

        # Model B (simple)
        params_b = model_b_params.copy()
        params_b["seed"] = seed
        dtrain_b = lgb.Dataset(X_tr, label=y_tr, feature_name=list(final_numeric_cols))
        dval_b = lgb.Dataset(X_va, label=y_va, reference=dtrain_b)
        model_b = lgb.train(
            params_b,
            dtrain_b,
            num_boost_round=5000,
            valid_sets=[dval_b],
            callbacks=[lgb.early_stopping(stopping_rounds=150, verbose=False)],
        )
        pred_b = model_b.predict(X_va, num_iteration=model_b.best_iteration)
        fold_val_preds += pred_b

    fold_val_preds /= (2 * len(seeds))
    cv_val_preds[va_idx] = fold_val_preds
    fold_rmse = np.sqrt(mean_squared_error(y_va, fold_val_preds))
    print(f"  Fold {fold_idx+1}: RMSE={fold_rmse:.4f}, best_iter_A={model_a.best_iteration}, best_iter_B={model_b.best_iteration}")
    fold_best_iters_A.append(model_a.best_iteration)
    fold_best_iters_B.append(model_b.best_iteration)

cv_rmse = np.sqrt(mean_squared_error(y_all, cv_val_preds))
print(f"\n5-Fold CV RMSE: {cv_rmse:.4f}")

avg_iter_A = int(np.mean(fold_best_iters_A))
avg_iter_B = int(np.mean(fold_best_iters_B))
print(f"Average best iterations: Model A={avg_iter_A}, Model B={avg_iter_B}")

# ============================================================
# 8. RETRAIN ON FULL DATA FOR TEST PREDICTIONS
# ============================================================
print("\n--- Retraining on full data for test predictions ---")
test_preds_list = []

for seed in seeds:
    # Model A (complex) - full data
    params_a = model_a_params.copy()
    params_a["seed"] = seed
    dtrain_full = lgb.Dataset(X_all, label=y_all, feature_name=list(final_numeric_cols))
    model_a_full = lgb.train(
        params_a,
        dtrain_full,
        num_boost_round=avg_iter_A,
    )
    pred_a_test = model_a_full.predict(X_test, num_iteration=model_a_full.best_iteration)
    test_preds_list.append(pred_a_test)

    # Model B (simple) - full data
    params_b = model_b_params.copy()
    params_b["seed"] = seed
    dtrain_full_b = lgb.Dataset(X_all, label=y_all, feature_name=list(final_numeric_cols))
    model_b_full = lgb.train(
        params_b,
        dtrain_full_b,
        num_boost_round=avg_iter_B,
    )
    pred_b_test = model_b_full.predict(X_test, num_iteration=model_b_full.best_iteration)
    test_preds_list.append(pred_b_test)

    print(f"  Seed {seed}: Model A iter={model_a_full.best_iteration}, Model B iter={model_b_full.best_iteration}")

final_test = np.mean(test_preds_list, axis=0)
print(f"Total test predictions averaged: {len(test_preds_list)} models")

# ============================================================
# 9. GENERATE SUBMISSION
# ============================================================
os.makedirs("./submission", exist_ok=True)
submission = pd.DataFrame({"id": test_ids.astype(int), "H index": final_test})
submission.to_csv("./submission/submission.csv", index=False)
print(f"\nSubmission saved: {submission.shape[0]} rows")
print(f"Test predictions: mean={final_test.mean():.4f}, std={final_test.std():.4f}")

# ============================================================
# 10. FINAL VALIDATION SCORE
# ============================================================
score = cv_rmse
print(f"Final Validation Score: {score:.4f}")
