import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import xgboost as xgb
import os
import pickle
import warnings
from sklearn.model_selection import KFold

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
print(f"Train columns: {list(train.columns)}")

# Identify target and ID columns
target_col = "Overall"
id_col = "id" if "id" in train.columns else None

# Separate target
y = train[target_col].values
train = train.drop(columns=[target_col])

if target_col in test.columns:
    test = test.drop(columns=[target_col])

# ============================================================
# 2. CREATE TRAIN/VALIDATION SPLIT
# ============================================================
indices = np.arange(len(train))
train_idx, val_idx = train_test_split(
    indices, test_size=0.15, random_state=42, shuffle=True
)

X_train = train.iloc[train_idx].copy()
X_val = train.iloc[val_idx].copy()
y_train = y[train_idx]
y_val = y[val_idx]
X_test = test.copy()

print(f"Split sizes: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
print(f"Target stats - Train: mean={y_train.mean():.4f}, std={y_train.std():.4f}")
print(f"Target stats - Val: mean={y_val.mean():.4f}, std={y_val.std():.4f}")

# Remove ID column from features if present
if id_col and id_col in X_train.columns:
    X_train = X_train.drop(columns=[id_col])
    X_val = X_val.drop(columns=[id_col])
    X_test = X_test.drop(columns=[id_col])

# Save test IDs
if id_col and id_col in test.columns:
    test_ids = test[id_col].values
else:
    test_ids = np.arange(len(X_test))

# ============================================================
# 3. FEATURE ENGINEERING - Target Encoding with 5-fold CV
# ============================================================

categorical_cols = ['Media', 'Domain', 'Country', 'Region', 'Language', 'Typology']
SMOOTHING_K = 10

# Ensure all categorical columns are string type
for col in categorical_cols:
    X_train[col] = X_train[col].astype(str)
    X_val[col] = X_val[col].astype(str)
    X_test[col] = X_test[col].astype(str)

# --- Target Encoding function with 5-fold CV (for training data) ---
def target_encode_cv(train_df, val_df, test_df, col, y_train, k=SMOOTHING_K):
    """Smoothed target encoding with 5-fold CV to prevent leakage."""
    global_mean = y_train.mean()

    # For training data: use 5-fold CV
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    te_train = np.zeros(len(train_df))

    for fold_idx, (tr_idx, va_idx) in enumerate(kf.split(train_df)):
        # Compute encoding stats from training fold only
        fold_data = pd.DataFrame({col: train_df[col].values[tr_idx], 'target': y_train[tr_idx]})
        cat_stats = fold_data.groupby(col)['target'].agg(['sum', 'count'])

        # Apply smoothed encoding to validation fold
        va_cats = train_df[col].values[va_idx]
        te_vals = np.array([
            (global_mean * k + cat_stats.loc[cat, 'sum']) / (k + cat_stats.loc[cat, 'count'])
            if cat in cat_stats.index else global_mean
            for cat in va_cats
        ])
        te_train[va_idx] = te_vals

    # For validation and test: fit on ALL training data
    full_stats = pd.DataFrame({col: train_df[col].values, 'target': y_train})
    full_stats = full_stats.groupby(col)['target'].agg(['sum', 'count'])

    te_val = np.array([
        (global_mean * k + full_stats.loc[cat, 'sum']) / (k + full_stats.loc[cat, 'count'])
        if cat in full_stats.index else global_mean
        for cat in val_df[col].values
    ])

    te_test = np.array([
        (global_mean * k + full_stats.loc[cat, 'sum']) / (k + full_stats.loc[cat, 'count'])
        if cat in full_stats.index else global_mean
        for cat in test_df[col].values
    ])

    return te_train, te_val, te_test

# --- Frequency/Count Encoding ---
def frequency_encode(train_df, val_df, test_df, col):
    """Frequency encoding fit on training data."""
    freq = train_df[col].value_counts(normalize=True)
    return (
        train_df[col].map(freq).fillna(0).values,
        val_df[col].map(freq).fillna(0).values,
        test_df[col].map(freq).fillna(0).values
    )

# --- Interaction Target Encoding ---
def interaction_target_encode_cv(train_df, val_df, test_df, col1, col2, y_train, k=SMOOTHING_K):
    """Target encoding for pairwise interaction with 5-fold CV."""
    global_mean = y_train.mean()

    # Create interaction key
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    train_df['_inter'] = train_df[col1] + '||' + train_df[col2]
    val_df['_inter'] = val_df[col1] + '||' + val_df[col2]
    test_df['_inter'] = test_df[col1] + '||' + test_df[col2]

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    te_train = np.zeros(len(train_df))

    for fold_idx, (tr_idx, va_idx) in enumerate(kf.split(train_df)):
        fold_data = pd.DataFrame({'_inter': train_df['_inter'].values[tr_idx], 'target': y_train[tr_idx]})
        cat_stats = fold_data.groupby('_inter')['target'].agg(['sum', 'count'])

        va_cats = train_df['_inter'].values[va_idx]
        te_vals = np.array([
            (global_mean * k + cat_stats.loc[cat, 'sum']) / (k + cat_stats.loc[cat, 'count'])
            if cat in cat_stats.index else global_mean
            for cat in va_cats
        ])
        te_train[va_idx] = te_vals

    # For val/test: fit on all training data
    full_data = pd.DataFrame({'_inter': train_df['_inter'].values, 'target': y_train})
    full_stats = full_data.groupby('_inter')['target'].agg(['sum', 'count'])

    te_val = np.array([
        (global_mean * k + full_stats.loc[cat, 'sum']) / (k + full_stats.loc[cat, 'count'])
        if cat in full_stats.index else global_mean
        for cat in val_df['_inter'].values
    ])

    te_test = np.array([
        (global_mean * k + full_stats.loc[cat, 'sum']) / (k + full_stats.loc[cat, 'count'])
        if cat in full_stats.index else global_mean
        for cat in test_df['_inter'].values
    ])

    return te_train, te_val, te_test

# --- Build Feature Matrix ---
print("Building target-encoded features...")

feature_train = {}
feature_val = {}
feature_test = {}

# 6 target encodings
for col in categorical_cols:
    te_tr, te_val, te_test = target_encode_cv(X_train, X_val, X_test, col, y_train)
    feature_train[f'{col}_te'] = te_tr
    feature_val[f'{col}_te'] = te_val
    feature_test[f'{col}_te'] = te_test

# 6 frequency features
for col in categorical_cols:
    freq_tr, freq_val, freq_test = frequency_encode(X_train, X_val, X_test, col)
    feature_train[f'{col}_freq'] = freq_tr
    feature_val[f'{col}_freq'] = freq_val
    feature_test[f'{col}_freq'] = freq_test

# 15 pairwise interaction target encodings
import itertools
pair_cols = list(itertools.combinations(categorical_cols, 2))
for col1, col2 in pair_cols:
    inter_tr, inter_val, inter_test = interaction_target_encode_cv(
        X_train, X_val, X_test, col1, col2, y_train
    )
    feature_train[f'{col1}_x_{col2}_te'] = inter_tr
    feature_val[f'{col1}_x_{col2}_te'] = inter_val
    feature_test[f'{col1}_x_{col2}_te'] = inter_test

# Build final feature matrices
X_train_feat = pd.DataFrame(feature_train)
X_val_feat = pd.DataFrame(feature_val)
X_test_feat = pd.DataFrame(feature_test)

print(f"Final feature count: {X_train_feat.shape[1]}")
print(f"Train: {X_train_feat.shape}, Val: {X_val_feat.shape}, Test: {X_test_feat.shape}")

# ============================================================
# 4. MODEL DESIGN & TRAINING: 3-Model Ensemble
# ============================================================
# Rationale:
# - 3 diverse models reduce variance on small dataset
# - Model 1: Conservative LightGBM (smaller trees, lower lr)
# - Model 2: XGBoost (different algorithm, similar regularization)
# - Model 3: LightGBM with different structure (more leaves, feature subsampling)
# - Equal-weight blending for final prediction

print("\n--- Training 3-Model Ensemble ---")

# Model 1: Conservative LightGBM
lgb_params1 = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.03,
    "num_leaves": 15,
    "max_depth": -1,
    "min_child_samples": 20,
    "min_child_weight": 1e-3,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "n_estimators": 5000,
    "random_state": 42,
    "verbose": -1,
    "n_jobs": -1,
}

model1 = lgb.LGBMRegressor(**lgb_params1)
model1.fit(
    X_train_feat, y_train,
    eval_set=[(X_val_feat, y_val)],
    eval_metric="rmse",
    callbacks=[lgb.early_stopping(stopping_rounds=150, verbose=False)],
)
best_iter1 = model1.best_iteration_
print(f"Model 1 (LGB conservative) - Best iter: {best_iter1}")

# Model 2: XGBoost
xgb_params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "max_depth": 4,
    "min_child_weight": 20,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "n_estimators": 5000,
    "random_state": 42,
    "tree_method": "hist",
    "n_jobs": -1,
    "verbosity": 0,
}

model2 = xgb.XGBRegressor(early_stopping_rounds=150, **xgb_params)
model2.fit(
    X_train_feat, y_train,
    eval_set=[(X_val_feat, y_val)],
    verbose=False,
)
best_iter2 = model2.best_iteration
print(f"Model 2 (XGBoost) - Best iter: {best_iter2}")

# Safety: if best_iteration is 0 (edge case), use n_estimators
if best_iter2 == 0:
    best_iter2 = xgb_params["n_estimators"]

# Model 3: LightGBM with different structure
lgb_params3 = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "min_child_samples": 10,
    "min_child_weight": 1e-3,
    "feature_fraction": 0.6,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "n_estimators": 5000,
    "random_state": 42,
    "verbose": -1,
    "n_jobs": -1,
}

model3 = lgb.LGBMRegressor(**lgb_params3)
model3.fit(
    X_train_feat, y_train,
    eval_set=[(X_val_feat, y_val)],
    eval_metric="rmse",
    callbacks=[lgb.early_stopping(stopping_rounds=150, verbose=False)],
)
best_iter3 = model3.best_iteration_
print(f"Model 3 (LGB diverse) - Best iter: {best_iter3}")

# ============================================================
# 5. VALIDATION & TEST PREDICTIONS (Ensemble)
# ============================================================

# Individual model validation predictions
val_pred1 = model1.predict(X_val_feat, num_iteration=best_iter1)
val_pred2 = model2.predict(X_val_feat, iteration_range=(1, best_iter2 + 1)) if best_iter2 > 0 else model2.predict(X_val_feat)
val_pred3 = model3.predict(X_val_feat, num_iteration=best_iter3)

# Individual model validation RMSE
rmse1 = np.sqrt(mean_squared_error(y_val, val_pred1))
rmse2 = np.sqrt(mean_squared_error(y_val, val_pred2))
rmse3 = np.sqrt(mean_squared_error(y_val, val_pred3))
print(f"\nModel 1 Validation RMSE: {rmse1:.6f}")
print(f"Model 2 Validation RMSE: {rmse2:.6f}")
print(f"Model 3 Validation RMSE: {rmse3:.6f}")

# Ensemble validation prediction (equal weights)
val_pred = (val_pred1 + val_pred2 + val_pred3) / 3.0
val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))

print(f"\nEnsemble Validation RMSE (computed via sklearn): {val_rmse:.6f}")

# Baseline comparison
baseline_rmse = np.sqrt(mean_squared_error(y_val, np.full_like(y_val, y_train.mean())))
print(f"Baseline RMSE (predict train mean): {baseline_rmse:.6f}")
print(f"Improvement over baseline: {(1 - val_rmse / baseline_rmse) * 100:.2f}%")

# Test predictions (ensemble)
test_pred1 = model1.predict(X_test_feat, num_iteration=best_iter1)
test_pred2 = model2.predict(X_test_feat, iteration_range=(1, best_iter2 + 1)) if best_iter2 > 0 else model2.predict(X_test_feat)
test_pred3 = model3.predict(X_test_feat, num_iteration=best_iter3)

test_pred = (test_pred1 + test_pred2 + test_pred3) / 3.0

print(f"\nTest predictions: mean={test_pred.mean():.4f}, std={test_pred.std():.4f}")
print(f"Test prediction range: [{test_pred.min():.4f}, {test_pred.max():.4f}]")

# Feature importance from Model 1
importance = model1.feature_importances_
feat_imp = pd.DataFrame(
    {"feature": X_train_feat.columns, "importance": importance}
).sort_values("importance", ascending=False)

print(f"\nTop 10 features by importance (Model 1):")
for i in range(min(10, len(feat_imp))):
    print(f"  {feat_imp.iloc[i]['feature']}: {feat_imp.iloc[i]['importance']:.0f}")

# ============================================================
# 6. SAVE SUBMISSION
# ============================================================
submission = pd.DataFrame({"id": test_ids.astype(int), "Overall": test_pred})
submission.to_csv("./submission/submission.csv", index=False)
print(f"\nSubmission saved: {submission.shape}")
print(f"Submission head:\n{submission.head()}")

# Save models
with open("./working/model_ensemble_trained.pkl", "wb") as f:
    pickle.dump({"model1": model1, "model2": model2, "model3": model3,
                 "best_iter1": best_iter1, "best_iter2": best_iter2, "best_iter3": best_iter3}, f)

# ============================================================
# 7. FINAL VALIDATION SCORE (official metric: RMSE)
# ============================================================
score = val_rmse
print(f"\n=== SUMMARY ===")
print(f"Model: 3-Model Ensemble (LGB cons + XGB + LGB diverse)")
print(f"Best iterations: {best_iter1}, {best_iter2}, {best_iter3}")
print(f"Validation set size: {len(y_val)}")
print(f"Metric: RMSE (official)")
print(f"Final Validation Score: {score}")
