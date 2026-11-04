import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error
import os
import json
import warnings

warnings.filterwarnings("ignore")

# Create output directories
os.makedirs("./submission", exist_ok=True)
os.makedirs("./working", exist_ok=True)

# ============================================================
# 1. LOAD DATA
# ============================================================
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")

# Identify target and id columns
TARGET = "Total Contribution (USD)"
ID_COL = "id"

if ID_COL not in train.columns:
    train[ID_COL] = range(len(train))
if ID_COL not in test.columns:
    test[ID_COL] = range(len(test))

# ============================================================
# 2. SEPARATE TARGET AND FEATURES
# ============================================================
y_train = train[TARGET].copy()
X_train = train.drop(columns=[TARGET])

X_test = test.copy()
test_ids = X_test[ID_COL].copy()
X_test = X_test.drop(columns=[ID_COL])
X_train = X_train.drop(columns=[ID_COL])

# ============================================================
# 3. CREATE TRAIN/VALIDATION SPLIT
# ============================================================
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.15, random_state=42
)

print(f"Train split: {X_tr.shape}, Val split: {X_val.shape}")
print(f"Target stats - Train: mean={y_tr.mean():.2f}, std={y_tr.std():.2f}")
print(f"Target stats - Val: mean={y_val.mean():.2f}, std={y_val.std():.2f}")

# ============================================================
# 4. FEATURE ENGINEERING
# ============================================================
all_X = pd.concat([X_tr, X_val, X_test], axis=0, ignore_index=True)

numeric_cols = all_X.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = all_X.select_dtypes(include=["object", "category"]).columns.tolist()
bool_cols = all_X.select_dtypes(include=["bool"]).columns.tolist()

print(f"Numeric columns ({len(numeric_cols)}): {numeric_cols}")
print(f"Categorical columns ({len(categorical_cols)}): {categorical_cols}")
print(f"Boolean columns ({len(bool_cols)}): {bool_cols}")

for col in bool_cols:
    all_X[col] = all_X[col].astype(int)
    if col not in numeric_cols:
        numeric_cols.append(col)

for col in categorical_cols:
    all_X[col] = all_X[col].astype("category")

# Split back
X_tr = all_X.iloc[: len(X_tr)].copy()
X_val = all_X.iloc[len(X_tr) : len(X_tr) + len(X_val)].copy()
X_test = all_X.iloc[len(X_tr) + len(X_val) :].copy()


# Domain-specific feature engineering
def create_features(df, numeric_cols, categorical_cols, fit=False, ref_stats=None):
    df = df.copy()

    if fit:
        ref_stats = {}
        for col in numeric_cols:
            if col in df.columns:
                ref_stats[col] = {
                    "mean": df[col].mean(),
                    "std": df[col].std() + 1e-8,
                    "min": df[col].min(),
                    "max": df[col].max(),
                    "median": df[col].median(),
                }

    for col in numeric_cols:
        if col in df.columns and ref_stats is not None:
            if col in ref_stats:
                df[f"{col}_z"] = (df[col] - ref_stats[col]["mean"]) / ref_stats[col][
                    "std"
                ]

    for col in numeric_cols:
        if col in df.columns:
            positive_mask = df[col] > 0
            if positive_mask.sum() > len(df) * 0.5:
                df[f"{col}_log"] = np.log1p(df[col].clip(lower=0))

    if len(numeric_cols) >= 2:
        key_numeric = [c for c in numeric_cols if c in df.columns][:10]
        for i, c1 in enumerate(key_numeric):
            for c2 in key_numeric[i + 1 :]:
                if c2 in df.columns and df[c2].abs().sum() > 0:
                    ratio_name = f"{c1}_div_{c2}"
                    df[ratio_name] = df[c1] / (df[c2].abs() + 1e-8)

    for cat_col in categorical_cols[:5]:
        if cat_col in df.columns:
            for num_col in numeric_cols[:5]:
                if num_col in df.columns:
                    interaction_name = f"{cat_col}_x_{num_col}"
                    df[interaction_name] = df[cat_col].cat.codes * df[num_col]

    return df, ref_stats


X_tr, ref_stats = create_features(X_tr, numeric_cols, categorical_cols, fit=True)
X_val, _ = create_features(
    X_val, numeric_cols, categorical_cols, fit=False, ref_stats=ref_stats
)
X_test, _ = create_features(
    X_test, numeric_cols, categorical_cols, fit=False, ref_stats=ref_stats
)

print(
    f"After feature engineering: Train={X_tr.shape}, Val={X_val.shape}, Test={X_test.shape}"
)

# ============================================================
# 5. HANDLE MISSING VALUES
# ============================================================
for col in X_tr.columns:
    if X_tr[col].isnull().any():
        if (
            X_tr[col].dtype in ["object", "category"]
            or str(X_tr[col].dtype) == "category"
        ):
            # Add "missing" as a category before filling
            for df in [X_tr, X_val, X_test]:
                if str(df[col].dtype) == "category" and "missing" not in df[col].cat.categories:
                    df[col] = df[col].cat.add_categories("missing")
            X_tr[col] = X_tr[col].fillna("missing")
            X_val[col] = X_val[col].fillna("missing")
            X_test[col] = X_test[col].fillna("missing")
        else:
            median_val = X_tr[col].median()
            X_tr[col] = X_tr[col].fillna(median_val)
            X_val[col] = X_val[col].fillna(median_val)
            X_test[col] = X_test[col].fillna(median_val)

print(
    f"Missing after imputation: Train={X_tr.isnull().sum().sum()}, Val={X_val.isnull().sum().sum()}, Test={X_test.isnull().sum().sum()}"
)

# ============================================================
# 6. ENCODE CATEGORICAL FEATURES
# ============================================================
final_cat_cols = []
for col in X_tr.columns:
    if X_tr[col].dtype in ["object", "category"] or str(X_tr[col].dtype) == "category":
        final_cat_cols.append(col)

cat_encoders = {}
for col in final_cat_cols:
    categories = X_tr[col].unique().tolist()
    if "missing" not in categories:
        categories.append("missing")
    cat_encoders[col] = {cat: idx for idx, cat in enumerate(categories)}
    X_tr[col] = X_tr[col].map(lambda x: cat_encoders[col].get(x, len(categories)))
    X_val[col] = X_val[col].map(lambda x: cat_encoders[col].get(x, len(categories)))
    X_test[col] = X_test[col].map(lambda x: cat_encoders[col].get(x, len(categories)))
    X_tr[col] = X_tr[col].astype(int)
    X_val[col] = X_val[col].astype(int)
    X_test[col] = X_test[col].astype(int)

all_feature_cols = list(X_tr.columns)
print(f"Total features: {len(all_feature_cols)}")

# ============================================================
# 7. TARGET TRANSFORMATION (LOG1P)
# ============================================================
print(
    f"Target distribution: Min={y_tr.min():.2f}, Max={y_tr.max():.2f}, Skewness={y_tr.skew():.2f}"
)
print(
    f"  % zeros: {(y_tr == 0).mean()*100:.1f}%, % negative: {(y_tr < 0).mean()*100:.1f}%"
)

use_log_target = y_tr.abs().max() > 100

if use_log_target:
    y_tr_log = np.log1p(y_tr.clip(lower=0))
    y_val_log = np.log1p(y_val.clip(lower=0))
    print("Using log1p target transformation")
else:
    y_tr_log = y_tr.copy()
    y_val_log = y_val.copy()
    print("Using raw target")

# Define original-scale validation target for final scoring
y_val_orig = y_val.copy()

# ============================================================
# 8. DEFINE MODEL CONFIGURATIONS
# ============================================================
lgb_configs = [
    {
        "objective": "huber",
        "alpha": 0.9,
        "metric": "rmse",
        "boosting_type": "gbdt",
        "num_leaves": 7,
        "max_depth": 3,
        "learning_rate": 0.03,
        "n_estimators": 3000,
        "min_child_samples": 10,
        "subsample": 0.7,
        "subsample_freq": 1,
        "colsample_bytree": 0.6,
        "reg_alpha": 1.0,
        "reg_lambda": 2.0,
        "random_state": 42,
        "verbose": -1,
    },
    {
        "objective": "huber",
        "alpha": 1.5,
        "metric": "rmse",
        "boosting_type": "gbdt",
        "num_leaves": 15,
        "max_depth": 5,
        "learning_rate": 0.05,
        "n_estimators": 2000,
        "min_child_samples": 5,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.5,
        "reg_lambda": 1.0,
        "random_state": 42,
        "verbose": -1,
    },
    {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "num_leaves": 11,
        "max_depth": 4,
        "learning_rate": 0.04,
        "n_estimators": 2500,
        "min_child_samples": 8,
        "subsample": 0.75,
        "subsample_freq": 1,
        "colsample_bytree": 0.7,
        "reg_alpha": 0.8,
        "reg_lambda": 1.5,
        "random_state": 42,
        "verbose": -1,
    },
    {
        "objective": "huber",
        "alpha": 0.5,
        "metric": "rmse",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "max_depth": 6,
        "learning_rate": 0.02,
        "n_estimators": 4000,
        "min_child_samples": 3,
        "subsample": 0.85,
        "subsample_freq": 1,
        "colsample_bytree": 0.9,
        "reg_alpha": 0.3,
        "reg_lambda": 0.8,
        "random_state": 42,
        "verbose": -1,
    },
    {
        "objective": "regression_l1",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "num_leaves": 9,
        "max_depth": 4,
        "learning_rate": 0.03,
        "n_estimators": 3000,
        "min_child_samples": 7,
        "subsample": 0.7,
        "subsample_freq": 1,
        "colsample_bytree": 0.65,
        "reg_alpha": 1.5,
        "reg_lambda": 2.5,
        "random_state": 42,
        "verbose": -1,
    },
]

xgb_configs = [
    {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "max_depth": 3,
        "learning_rate": 0.03,
        "n_estimators": 3000,
        "min_child_weight": 10,
        "subsample": 0.7,
        "colsample_bytree": 0.6,
        "reg_alpha": 1.0,
        "reg_lambda": 2.0,
        "random_state": 42,
        "tree_method": "hist",
        "verbosity": 0,
    },
    {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "max_depth": 5,
        "learning_rate": 0.05,
        "n_estimators": 2000,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.5,
        "reg_lambda": 1.0,
        "random_state": 42,
        "tree_method": "hist",
        "verbosity": 0,
    },
]

# ============================================================
# 9. K-FOLD CROSS-VALIDATION FOR MODEL SELECTION
# ============================================================
print("\n" + "=" * 60)
print("K-FOLD CROSS-VALIDATION FOR MODEL SELECTION")
print("=" * 60)

kf = KFold(n_splits=5, shuffle=True, random_state=42)


def inverse_transform(pred_log):
    if use_log_target:
        return np.expm1(pred_log)
    return pred_log


# Evaluate LGB configs
lgb_cv_scores = []
for i, params in enumerate(lgb_configs):
    fold_scores = []
    for fold_idx, (tr_idx, va_idx) in enumerate(kf.split(X_tr)):
        X_fold_tr = X_tr.iloc[tr_idx]
        X_fold_va = X_tr.iloc[va_idx]
        y_fold_tr = y_tr_log.iloc[tr_idx]
        y_fold_va = y_tr_log.iloc[va_idx]

        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_fold_tr,
            y_fold_tr,
            eval_set=[(X_fold_va, y_fold_va)],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
        )

        pred_log = model.predict(X_fold_va)
        pred_orig = inverse_transform(pred_log)
        y_fold_va_orig = np.expm1(y_fold_va) if use_log_target else y_fold_va
        rmse = np.sqrt(mean_squared_error(y_fold_va_orig, pred_orig))
        fold_scores.append(rmse)

    mean_score = np.mean(fold_scores)
    std_score = np.std(fold_scores)
    lgb_cv_scores.append(mean_score)
    print(f"  LGB config {i}: CV RMSE = {mean_score:.4f} ± {std_score:.4f}")

# Evaluate XGB configs
xgb_cv_scores = []
for i, params in enumerate(xgb_configs):
    fold_scores = []
    for fold_idx, (tr_idx, va_idx) in enumerate(kf.split(X_tr)):
        X_fold_tr = X_tr.iloc[tr_idx]
        X_fold_va = X_tr.iloc[va_idx]
        y_fold_tr = y_tr_log.iloc[tr_idx]
        y_fold_va = y_tr_log.iloc[va_idx]

        model = xgb.XGBRegressor(**params, early_stopping_rounds=50)
        model.fit(
            X_fold_tr,
            y_fold_tr,
            eval_set=[(X_fold_va, y_fold_va)],
            verbose=False,
        )

        pred_log = model.predict(X_fold_va)
        pred_orig = inverse_transform(pred_log)
        y_fold_va_orig = np.expm1(y_fold_va) if use_log_target else y_fold_va
        rmse = np.sqrt(mean_squared_error(y_fold_va_orig, pred_orig))
        fold_scores.append(rmse)

    mean_score = np.mean(fold_scores)
    std_score = np.std(fold_scores)
    xgb_cv_scores.append(mean_score)
    print(f"  XGB config {i}: CV RMSE = {mean_score:.4f} ± {std_score:.4f}")

# ============================================================
# 10. RETRAIN BEST MODELS ON FULL TRAINING DATA
# ============================================================
best_lgb_idx = int(np.argmin(lgb_cv_scores))
best_xgb_idx = int(np.argmin(xgb_cv_scores))

print(f"\nBest LGB config: {best_lgb_idx} (Val RMSE: {lgb_cv_scores[best_lgb_idx]:.4f})")
print(f"Best XGB config: {best_xgb_idx} (Val RMSE: {xgb_cv_scores[best_xgb_idx]:.4f})")

# Use best_iteration from selection phase as n_estimators for final training
# We need to get the best iteration count from the selection phase
# Retrain on full X_train (original) using the best config
# For simplicity, use the same number of estimators found during selection
# We'll retrain on all available training data (X_tr) with early stopping on X_val

# Get best iterations from CV phase by re-running with tracking
print("\nDetermining best iterations from CV...")
lgb_best_iters = []
for fold_idx, (tr_idx, va_idx) in enumerate(kf.split(X_tr)):
    model = lgb.LGBMRegressor(**lgb_configs[best_lgb_idx])
    model.fit(
        X_tr.iloc[tr_idx], y_tr_log.iloc[tr_idx],
        eval_set=[(X_tr.iloc[va_idx], y_tr_log.iloc[va_idx])],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
    )
    lgb_best_iters.append(model.best_iteration_)
lgb_n_est = int(np.median(lgb_best_iters)) + 50
print(f"  LGB median best iteration: {lgb_n_est}")

xgb_best_iters = []
for fold_idx, (tr_idx, va_idx) in enumerate(kf.split(X_tr)):
    model = xgb.XGBRegressor(**xgb_configs[best_xgb_idx], early_stopping_rounds=50)
    model.fit(
        X_tr.iloc[tr_idx], y_tr_log.iloc[tr_idx],
        eval_set=[(X_tr.iloc[va_idx], y_tr_log.iloc[va_idx])],
        verbose=False,
    )
    xgb_best_iters.append(model.best_iteration)
xgb_n_est = int(np.median(xgb_best_iters)) + 50
print(f"  XGB median best iteration: {xgb_n_est}")

# Train final models on X_tr only (excluding X_val) to maintain a clean validation set
print("\nTraining best LightGBM on training split...")
lgb_final_params = dict(lgb_configs[best_lgb_idx])
lgb_final_params["n_estimators"] = lgb_n_est
best_lgb_model = lgb.LGBMRegressor(**lgb_final_params)
best_lgb_model.fit(X_tr, y_tr_log)
print(f"  Trained with n_estimators={lgb_n_est}")

print("Training best XGBoost on training split...")
xgb_final_params = dict(xgb_configs[best_xgb_idx])
xgb_final_params["n_estimators"] = xgb_n_est
best_xgb_model = xgb.XGBRegressor(**xgb_final_params)
best_xgb_model.fit(X_tr, y_tr_log)
print(f"  Trained with n_estimators={xgb_n_est}")

# ============================================================
# 11. BLEND AND EVALUATE ON VALIDATION SET
# ============================================================
print("\n" + "=" * 60)
print("BLENDING AND VALIDATION")
print("=" * 60)

lgb_val_pred_log = best_lgb_model.predict(X_val)
xgb_val_pred_log = best_xgb_model.predict(X_val)
lgb_test_pred_log = best_lgb_model.predict(X_test)
xgb_test_pred_log = best_xgb_model.predict(X_test)

# Use equal-weight blend (avoid tuning on validation set to prevent overfitting the score)
best_w = 0.5
blend_log_val = best_w * lgb_val_pred_log + (1 - best_w) * xgb_val_pred_log
blend_orig_val = inverse_transform(blend_log_val)
if use_log_target:
    blend_orig_val = np.clip(blend_orig_val, 0, None)
best_rmse = np.sqrt(mean_squared_error(y_val_orig, blend_orig_val))

print(f"Blend weight (LGB): {best_w:.2f} (fixed)")
print(f"Blend RMSE: {best_rmse:.4f}")

lgb_val_orig = (
    np.clip(inverse_transform(lgb_val_pred_log), 0, None)
    if use_log_target
    else lgb_val_pred_log
)
xgb_val_orig = (
    np.clip(inverse_transform(xgb_val_pred_log), 0, None)
    if use_log_target
    else xgb_val_pred_log
)
print(f"  LGB RMSE: {np.sqrt(mean_squared_error(y_val_orig, lgb_val_orig)):.4f}")
print(f"  XGB RMSE: {np.sqrt(mean_squared_error(y_val_orig, xgb_val_orig)):.4f}")

# Apply best blend
val_pred_log = best_w * lgb_val_pred_log + (1 - best_w) * xgb_val_pred_log
test_pred_log = best_w * lgb_test_pred_log + (1 - best_w) * xgb_test_pred_log

# Compute validation predictions for final score
val_pred_final = inverse_transform(val_pred_log)
if use_log_target:
    val_pred_final = np.clip(val_pred_final, 0, None)
score = np.sqrt(mean_squared_error(y_val_orig, val_pred_final))

# ============================================================
# 12. FINAL PREDICTIONS AND SCORE
# ============================================================
test_pred_log = best_w * lgb_test_pred_log + (1 - best_w) * xgb_test_pred_log
val_pred_log = best_w * lgb_val_pred_log + (1 - best_w) * xgb_val_pred_log

if use_log_target:
    test_pred = np.clip(np.expm1(test_pred_log), 0, None)
    val_pred_final = np.clip(np.expm1(val_pred_log), 0, None)
else:
    test_pred = test_pred_log
    val_pred_final = val_pred_log

score = np.sqrt(mean_squared_error(y_val_orig, val_pred_final))
print(f"Validation RMSE: {score:.4f}")

# ============================================================
# 13. GENERATE SUBMISSION
# ============================================================
submission = pd.DataFrame(
    {"id": test_ids.astype(int), "Total Contribution (USD)": test_pred}
)

submission.to_csv("./submission/submission.csv", index=False)
print(f"\nSubmission saved: {submission.shape}")
print(
    f"  Test pred stats: mean={test_pred.mean():.2f}, std={test_pred.std():.2f}, min={test_pred.min():.2f}, max={test_pred.max():.2f}"
)

# ============================================================
# 14. FINAL VALIDATION SCORE
# ============================================================
print(f"\n{'='*60}")
print(f"FINAL RESULTS")
print(f"{'='*60}")
print(f"  Best LGB config: {best_lgb_idx}")
print(f"  Best XGB config: {best_xgb_idx}")
print(f"  Blend weight: {best_w:.2f}")
print(f"  Validation RMSE: {score:.4f}")
print(f"{'='*60}")

print(f"Final Validation Score: {score}")