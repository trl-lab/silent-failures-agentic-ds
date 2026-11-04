import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import xgboost as xgb
import os
import joblib

# Create directories
os.makedirs("./submission", exist_ok=True)
os.makedirs("./working", exist_ok=True)

# ============ LOAD DATA ============
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")

# Identify feature types
target_col = "State Share"
id_col = "id" if "id" in train.columns else None

# Separate target and features
if id_col:
    train_ids = train[id_col].copy()
    test_ids = test[id_col].copy()
    train = train.drop(columns=[id_col])
    test = test.drop(columns=[id_col])
else:
    test_ids = pd.Series(range(len(test)))

y = train[target_col].copy()
X = train.drop(columns=[target_col])

# Directly predict State Share (no residual transformation)
# Federal Share columns contain dollar amounts, not percentages
print(f"Target - mean: {y.mean():.4f}, std: {y.std():.4f}")
print(f"Target range: [{y.min():.4f}, {y.max():.4f}]")

# Identify column types
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
bool_cols = [c for c in X.columns if X[c].dtype == bool]

print(f"Numerical columns ({len(num_cols)}): {num_cols}")
print(f"Categorical columns ({len(cat_cols)}): {cat_cols}")

# ============ TRAIN/VAL SPLIT (BEFORE ANY STATEFUL TRANSFORMATION) ============
X_train_full, X_val, y_train_full, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Use target directly
y_train_target = y_train_full
y_val_target = y_val

print(f"Full train: {X_train_full.shape}, Val: {X_val.shape}, Test: {test.shape}")
print(f"Target train - mean: {y_train_target.mean():.4f}, std: {y_train_target.std():.4f}")
print(f"Target val - mean: {y_val_target.mean():.4f}, std: {y_val_target.std():.4f}")

# ============ FEATURE ENGINEERING ============

# Step 1: Handle missing values
num_medians = X_train_full[num_cols].median()
for c in num_cols:
    X_train_full[c] = X_train_full[c].fillna(X_train_full[c].median())
    X_val[c] = X_val[c].fillna(num_medians[c])
    test[c] = test[c].fillna(num_medians[c])

for c in cat_cols:
    X_train_full[c] = X_train_full[c].fillna("missing")
    X_val[c] = X_val[c].fillna("missing")
    test[c] = test[c].fillna("missing")

# Step 2: Log transform highly skewed numerical features
skewed_cols = []
for c in num_cols:
    skew = X_train_full[c].skew()
    if abs(skew) > 1.0:
        skewed_cols.append(c)
        if X_train_full[c].min() >= 0:
            X_train_full[c + "_log"] = np.log1p(X_train_full[c])
            X_val[c + "_log"] = np.log1p(X_val[c])
            test[c + "_log"] = np.log1p(test[c])
        else:
            X_train_full[c + "_log"] = np.sign(X_train_full[c]) * np.log1p(
                np.abs(X_train_full[c])
            )
            X_val[c + "_log"] = np.sign(X_val[c]) * np.log1p(np.abs(X_val[c]))
            test[c + "_log"] = np.sign(test[c]) * np.log1p(np.abs(test[c]))

print(f"Log-transformed columns: {skewed_cols}")

# Step 2b: Remove log features that correlate >0.95 with their original
cols_to_drop = []
for c in skewed_cols:
    log_col = c + "_log"
    if log_col in X_train_full.columns and c in X_train_full.columns:
        corr_val = abs(X_train_full[c].corr(X_train_full[log_col]))
        if corr_val > 0.95:
            cols_to_drop.append(log_col)
            print(f"  Dropping {log_col} (corr with {c} = {corr_val:.4f})")

if cols_to_drop:
    X_train_full = X_train_full.drop(columns=cols_to_drop)
    X_val = X_val.drop(columns=cols_to_drop)
    test = test.drop(columns=cols_to_drop)

# Step 2c: Add interaction features between federal share components (ratios)
federal_cols = [c for c in num_cols if "Federal Share" in c]
for i in range(len(federal_cols)):
    for j in range(i + 1, len(federal_cols)):
        ratio_name = f"{federal_cols[i]}_div_{federal_cols[j]}"
        X_train_full[ratio_name] = X_train_full[federal_cols[i]] / (X_train_full[federal_cols[j]].abs() + 1e-8)
        X_val[ratio_name] = X_val[federal_cols[i]] / (X_val[federal_cols[j]].abs() + 1e-8)
        test[ratio_name] = test[federal_cols[i]] / (test[federal_cols[j]].abs() + 1e-8)

print(f"Added ratio features between federal share components: {federal_cols}")

# Step 3: Create interaction features
new_num_cols = [
    c
    for c in X_train_full.columns
    if X_train_full[c].dtype in [np.float64, np.float32, np.int64, np.int32]
]

if len(new_num_cols) >= 2:
    corr_matrix = X_train_full[new_num_cols].corr()
    pairs = []
    for i in range(len(new_num_cols)):
        for j in range(i + 1, len(new_num_cols)):
            pairs.append(
                (new_num_cols[i], new_num_cols[j], abs(corr_matrix.iloc[i, j]))
            )
    pairs.sort(key=lambda x: x[2], reverse=True)

    top_pairs = pairs[: min(20, len(pairs))]
    for c1, c2, corr_val in top_pairs:
        if corr_val > 0.3:
            ratio_name = f"{c1}_div_{c2}"
            X_train_full[ratio_name] = X_train_full[c1] / (X_train_full[c2] + 1e-8)
            X_val[ratio_name] = X_val[c1] / (X_val[c2] + 1e-8)
            test[ratio_name] = test[c1] / (test[c2] + 1e-8)

            prod_name = f"{c1}_mul_{c2}"
            X_train_full[prod_name] = X_train_full[c1] * X_train_full[c2]
            X_val[prod_name] = X_val[c1] * X_val[c2]
            test[prod_name] = test[c1] * test[c2]

    print(f"Created ratio and product features from {len(top_pairs)} pairs")

# Step 4: Aggregate features for low-cardinality categoricals
for c in cat_cols:
    if X_train_full[c].nunique() < 50:
        # Use only the training portion (exclude val rows) to avoid leakage
        # X_train_full here is already the train split (post train_test_split), so this is safe
        group_mean = X_train_full.groupby(c)[num_cols].mean()
        group_std = X_train_full.groupby(c)[num_cols].std()

        for nc in num_cols[:5]:
            mean_col = f"{c}_{nc}_mean"
            std_col = f"{c}_{nc}_std"

            X_train_full[mean_col] = X_train_full[c].map(group_mean[nc])
            X_val[mean_col] = X_val[c].map(group_mean[nc])
            test[mean_col] = test[c].map(group_mean[nc])

            X_train_full[std_col] = X_train_full[c].map(group_std[nc])
            X_val[std_col] = X_val[c].map(group_std[nc])
            test[std_col] = test[c].map(group_std[nc])

print(
    f"After feature engineering - Train: {X_train_full.shape}, Val: {X_val.shape}, Test: {test.shape}"
)

# Step 5: Encode categorical variables
final_cat_cols = [
    c for c in X_train_full.columns if X_train_full[c].dtype in ["object", "category"]
]
final_num_cols = [
    c
    for c in X_train_full.columns
    if c not in final_cat_cols
    and X_train_full[c].dtype in [np.float64, np.float32, np.int64, np.int32, np.bool_]
]

print(f"Final categorical cols: {final_cat_cols}")
print(f"Final numerical cols ({len(final_num_cols)})")

oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
X_train_full[final_cat_cols] = oe.fit_transform(X_train_full[final_cat_cols])
X_val[final_cat_cols] = oe.transform(X_val[final_cat_cols])
test[final_cat_cols] = oe.transform(test[final_cat_cols])

# Convert all to float
X_train_full = X_train_full.astype(np.float64)
X_val = X_val.astype(np.float64)
test = test.astype(np.float64)

# Step 6: No scaling - tree-based models handle raw features natively
print(
    f"Final shapes - Train: {X_train_full.shape}, Val: {X_val.shape}, Test: {test.shape}"
)
print(f"Target mean: {y_train_full.mean():.4f}, std: {y_train_full.std():.4f}")
print(f"Target range: [{y_train_full.min():.4f}, {y_train_full.max():.4f}]")

# ============ MODEL DESIGN & TRAINING (Direct regression on State Share) ============

# Tuned parameters for direct regression
lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.01,
    "num_leaves": 20,
    "max_depth": 5,
    "min_child_samples": 5,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.7,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "verbose": -1,
    "n_jobs": -1,
}

n_seeds = 5
seeds = [42, 123, 456, 789, 1024]

# 5 LightGBM models with different seeds
model_configs = [
    (lgb_params, seeds[0]),
    (lgb_params, seeds[1]),
    (lgb_params, seeds[2]),
    (lgb_params, seeds[3]),
    (lgb_params, seeds[4]),
]

models = []
val_preds_list = []
test_preds_list = []
val_scores = []

for i, (base_cfg, seed) in enumerate(model_configs):
    params = base_cfg.copy()
    params["random_state"] = seed

    # Train on direct target
    train_data = lgb.Dataset(X_train_full, label=y_train_target)
    val_data = lgb.Dataset(X_val, label=y_val_target, reference=train_data)

    callbacks = [
        lgb.early_stopping(stopping_rounds=300, verbose=False),
        lgb.log_evaluation(period=500),
    ]

    model = lgb.train(
        params,
        train_data,
        num_boost_round=15000,
        valid_sets=[val_data],
        callbacks=callbacks,
    )

    best_iter = model.best_iteration
    best_val_rmse = model.best_score["valid_0"]["rmse"]

    # Predict directly
    val_pred = model.predict(X_val, num_iteration=best_iter)
    test_pred = model.predict(test, num_iteration=best_iter)

    # Compute RMSE (official metric)
    val_rmse_reconstructed = np.sqrt(mean_squared_error(y_val, val_pred))

    val_preds_list.append(val_pred)
    test_preds_list.append(test_pred)
    val_scores.append(val_rmse_reconstructed)
    models.append(model)

    print(f"  Seed {seed}: best_iter={best_iter}, val_rmse={best_val_rmse:.4f}")

# ============ ADD XGBOOST MODELS (2 models, direct target) ============
xgb_params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "max_depth": 5,
    "learning_rate": 0.01,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "n_estimators": 15000,
    "tree_method": "hist",
    "n_jobs": -1,
    "verbosity": 0,
}

xgb_seeds = [42, 123]

for seed in xgb_seeds:
    params = xgb_params.copy()
    params["random_state"] = seed

    # Train on direct target
    model_xgb = xgb.train(
        params,
        xgb.DMatrix(X_train_full, label=y_train_target),
        num_boost_round=15000,
        evals=[(xgb.DMatrix(X_val, label=y_val_target), "val")],
        early_stopping_rounds=300,
        verbose_eval=500,
    )

    best_iter = model_xgb.best_iteration
    best_val_rmse = model_xgb.best_score

    # Predict directly
    val_pred = model_xgb.predict(
        xgb.DMatrix(X_val), iteration_range=(0, best_iter + 1)
    )
    test_pred = model_xgb.predict(
        xgb.DMatrix(test), iteration_range=(0, best_iter + 1)
    )

    # Compute RMSE (official metric)
    val_rmse_reconstructed = np.sqrt(mean_squared_error(y_val, val_pred))

    val_preds_list.append(val_pred)
    test_preds_list.append(test_pred)
    val_scores.append(val_rmse_reconstructed)
    models.append(model_xgb)

    print(f"  XGBoost Seed {seed}: best_iter={best_iter}, val_rmse={best_val_rmse:.4f}")

# ============ ENSEMBLE PREDICTIONS (Inverse-Variance Weighted) ============
# Weight each model by 1/(val_rmse^2), normalized to sum to 1
variances = np.array([s**2 for s in val_scores])
weights = (1.0 / variances)
weights = weights / weights.sum()

print(f"\nModel weights: {weights}")

val_pred_ensemble = np.average(val_preds_list, axis=0, weights=weights)
test_pred_ensemble = np.average(test_preds_list, axis=0, weights=weights)

# ============ VALIDATION METRIC (Official: RMSE) ============
val_rmse = np.sqrt(mean_squared_error(y_val, val_pred_ensemble))

print(f"\n=== ENSEMBLE RESULTS ===")
print(f"Individual val RMSEs: {[f'{s:.4f}' for s in val_scores]}")
print(f"Ensemble Val RMSE: {val_rmse:.4f}")
print(f"Target std: {y_val.std():.4f}")
print(f"Improvement over mean baseline: {(1 - val_rmse / y_val.std()) * 100:.1f}%")
print(f"Mean baseline RMSE: {np.sqrt(mean_squared_error(y_val, np.full(len(y_val), y_train_target.mean())))}")

# ============ SAVE MODELS ============
joblib.dump(models, "./working/models_ensemble.joblib")

# ============ SUBMISSION ============
submission = pd.DataFrame(
    {"id": test_ids.values.astype(int), "State Share": test_pred_ensemble}
)
submission.to_csv("./submission/submission.csv", index=False)
print(f"\nSubmission saved: {submission.shape}")
print(
    f"Prediction stats: mean={test_pred_ensemble.mean():.4f}, std={test_pred_ensemble.std():.4f}"
)
print(
    f"Prediction range: [{test_pred_ensemble.min():.4f}, {test_pred_ensemble.max():.4f}]"
)

# ============ FEATURE IMPORTANCE ============
importance = models[0].feature_importance(importance_type="gain")
feat_imp = pd.Series(importance, index=X_train_full.columns).sort_values(
    ascending=False
)
print(f"\nTop 10 features by gain:")
for i, (feat, imp) in enumerate(feat_imp.head(10).items()):
    print(f"  {i+1}. {feat}: {imp:.1f}")

# Final validation score using official metric (RMSE)
score = val_rmse
print(f"Final Validation Score: {score}")