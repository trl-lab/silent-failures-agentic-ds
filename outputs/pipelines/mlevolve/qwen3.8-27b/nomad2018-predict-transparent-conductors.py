import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
import lightgbm as lgb
import xgboost as xgb
from sklearn.svm import SVR
import os
import json
import re
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. LOAD DATA
# ============================================================
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")
print(f"Train columns: {list(train_df.columns)}")
print(f"\nTarget stats:\n{train_df['formation_energy_ev_natom'].describe()}")

# ============================================================
# 2. OFFICIAL METRIC: RMSLE
# ============================================================
def compute_rmsle(y_true, y_pred):
    """Exact RMSLE metric: sqrt(mean((log1p(|y_true|) - log1p(|y_pred|))^2))"""
    return np.sqrt(np.mean((np.log1p(np.abs(y_true)) - np.log1p(np.abs(y_pred))) ** 2))

# ============================================================
# 2. FEATURE ENGINEERING (Domain-specific for Al-Ga-In ternary system)
# ============================================================
def engineer_features(df):
    """Engineer domain-specific features for Al-Ga-In ternary alloy system."""
    features = pd.DataFrame(index=df.index)

    # (1) Lattice volume: V = a*b*c*sqrt(1 - cos²α - cos²β - cos²γ + 2*cosα*cosβ*cosγ)
    a = df['lattice_vector_1_ang'].values
    b = df['lattice_vector_2_ang'].values
    c = df['lattice_vector_3_ang'].values
    alpha = np.radians(df['lattice_angle_alpha_degree'].values)
    beta = np.radians(df['lattice_angle_beta_degree'].values)
    gamma = np.radians(df['lattice_angle_gamma_degree'].values)

    cos_a, cos_b, cos_g = np.cos(alpha), np.cos(beta), np.cos(gamma)
    vol_factor = 1 - cos_a**2 - cos_b**2 - cos_g**2 + 2 * cos_a * cos_b * cos_g
    vol_factor = np.maximum(vol_factor, 1e-10)  # numerical safety
    lattice_volume = a * b * c * np.sqrt(vol_factor)
    features['lattice_volume'] = lattice_volume

    # (2) Volume per atom
    features['volume_per_atom'] = lattice_volume / (df['number_of_total_atoms'].values + 1e-10)

    # (3) Composition interaction terms (non-ideal mixing in ternary alloys)
    pct_al = df['percent_atom_al'].values
    pct_ga = df['percent_atom_ga'].values
    pct_in = df['percent_atom_in'].values
    features['pct_al_x_pct_ga'] = pct_al * pct_ga
    features['pct_ga_x_pct_in'] = pct_ga * pct_in
    features['pct_al_x_pct_in'] = pct_al * pct_in

    # (4) Lattice anisotropy
    features['v2_over_v1'] = b / (a + 1e-10)
    features['v3_over_v1'] = c / (a + 1e-10)
    features['v_mean'] = (a + b + c) / 3.0
    features['v_max_min_ratio'] = np.maximum(a, np.maximum(b, c)) / (np.minimum(a, np.minimum(b, c)) + 1e-10)

    # (5) Angle deviations from 90 degrees (distortion measures)
    features['alpha_dev'] = df['lattice_angle_alpha_degree'].values - 90.0
    features['beta_dev'] = df['lattice_angle_beta_degree'].values - 90.0
    features['gamma_dev'] = df['lattice_angle_gamma_degree'].values - 90.0
    features['angle_dev_sum'] = np.abs(features['alpha_dev']) + np.abs(features['beta_dev']) + np.abs(features['gamma_dev'])

    # (6) Composition-weighted volume (effective volume proxy)
    comp_weight = (pct_al + 2 * pct_ga + 3 * pct_in) / 6.0
    features['comp_weighted_vol'] = lattice_volume * comp_weight

    # (7) Spacegroup as label-encoded categorical
    features['spacegroup'] = pd.factorize(df['spacegroup'])[0].astype(float)

    # Keep all raw features
    features['number_of_total_atoms'] = df['number_of_total_atoms'].values
    features['percent_atom_al'] = pct_al
    features['percent_atom_ga'] = pct_ga
    features['percent_atom_in'] = pct_in
    features['lattice_vector_1_ang'] = a
    features['lattice_vector_2_ang'] = b
    features['lattice_vector_3_ang'] = c
    features['lattice_angle_alpha_degree'] = df['lattice_angle_alpha_degree'].values
    features['lattice_angle_beta_degree'] = df['lattice_angle_beta_degree'].values
    features['lattice_angle_gamma_degree'] = df['lattice_angle_gamma_degree'].values

    # Additional physics-informed features
    # (8) Volume per atom squared and cubed (non-linear volume effects)
    features['volume_per_atom_sq'] = features['volume_per_atom'] ** 2
    features['volume_per_atom_cu'] = features['volume_per_atom'] ** 3

    # (9) Composition entropy (Shannon entropy of composition)
    comp = np.column_stack([pct_al, pct_ga, pct_in])
    comp = comp / (comp.sum(axis=1, keepdims=True) + 1e-10)
    comp = np.clip(comp, 1e-10, 1.0)
    features['comp_entropy'] = -np.sum(comp * np.log(comp), axis=1)

    # (10) Composition variance (measure of how spread out the composition is)
    features['comp_var'] = np.var(comp, axis=1)

    # (11) Lattice volume per atom interaction with composition
    features['vol_per_atom_x_entropy'] = features['volume_per_atom'] * features['comp_entropy']

    # (12) Angle product (capturing combined angular distortion)
    features['angle_product'] = features['alpha_dev'] * features['beta_dev'] * features['gamma_dev']

    # (13) Volume / (number of atoms) interaction with composition
    features['vol_atom_x_al'] = features['volume_per_atom'] * pct_al
    features['vol_atom_x_ga'] = features['volume_per_atom'] * pct_ga
    features['vol_atom_x_in'] = features['volume_per_atom'] * pct_in

    # (14) Lattice vector products (area-like quantities)
    features['a_times_b'] = a * b
    features['b_times_c'] = b * c
    features['a_times_c'] = a * c

    # (15) Sine of angles (related to cell shape)
    features['sin_alpha'] = np.sin(alpha)
    features['sin_beta'] = np.sin(beta)
    features['sin_gamma'] = np.sin(gamma)

    # (16) Deviation from cubic (all vectors equal, all angles 90)
    features['cubic_dev_vol'] = np.abs(lattice_volume - (a * b * c))
    features['cubic_dev_angle'] = np.abs(1.0 - (np.sin(alpha) * np.sin(beta) * np.sin(gamma)))

    return features

print("Engineering features...")
X_train_full = engineer_features(train_df)
X_test = engineer_features(test_df)

print(f"Feature matrix shape (train): {X_train_full.shape}")
print(f"Feature matrix shape (test): {X_test.shape}")
print(f"Features: {list(X_train_full.columns)}")
print(f"Number of features: {X_train_full.shape[1]}")

# ============================================================
# 4. SPLIT TRAIN INTO TRAIN/VAL (80/20)
# ============================================================
target = train_df["formation_energy_ev_natom"].values

train_idx, val_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.20, random_state=42, shuffle=True
)

X_train = X_train_full.iloc[train_idx].values
X_val = X_train_full.iloc[val_idx].values
y_train = target[train_idx]
y_val = target[val_idx]
X_test_arr = X_test.values
X_full = X_train_full.values

print(f"Split: train={len(X_train)}, val={len(X_val)}, test={len(X_test_arr)}")
print(f"Target range: [{y_train.min():.4f}, {y_train.max():.4f}]")
print(f"Target min: {target.min():.6f}, max: {target.max():.6f}")
print(f"Target has negatives: {(target < 0).sum()} / {len(target)}")

# ============================================================
# 5. TARGET TRANSFORMATION (log1p since target is all non-negative)
# ============================================================
y_train_log = np.log1p(y_train)
y_val_log = np.log1p(y_val)
y_full_log = np.log1p(target)


# ============================================================
# 7. MODEL 1: XGBoost (squarederror on log1p target)
# ============================================================
print("\n--- Training Model 1: XGBoost ---")
model_xgb = xgb.XGBRegressor(
    objective="reg:squarederror",
    eval_metric="rmse",
    max_depth=4,
    learning_rate=0.02,
    n_estimators=3000,
    early_stopping_rounds=200,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    reg_alpha=0.3,
    reg_lambda=1.5,
    random_state=42,
    n_jobs=-1,
    verbosity=0,
)
model_xgb.fit(
    X_train,
    y_train_log,
    eval_set=[(X_val, y_val_log)],
    verbose=False,
)

val_pred_xgb_log = model_xgb.predict(X_val)
val_pred_xgb = np.expm1(val_pred_xgb_log)
rmsle_1 = compute_rmsle(y_val, val_pred_xgb)
best_iter_xgb = model_xgb.best_iteration
print(f"  Best iteration: {best_iter_xgb}")
print(f"  Val RMSLE: {rmsle_1:.6f}")

# ============================================================
# 8. MODEL 2: LightGBM with Huber loss (alpha=0.9)
# ============================================================
print("\n--- Training Model 2: LGBM with Huber loss ---")
model_huber = lgb.LGBMRegressor(
    objective="huber",
    alpha=0.9,
    num_leaves=15,
    learning_rate=0.02,
    n_estimators=3000,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=5,
    min_child_samples=20,
    reg_alpha=0.3,
    reg_lambda=1.5,
    verbose=-1,
    n_jobs=-1,
    random_state=42,
)
model_huber.fit(
    X_train,
    y_train_log,
    eval_set=[(X_val, y_val_log)],
    callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(period=0)],
)

val_pred_huber_log = model_huber.predict(X_val)
val_pred_huber = np.expm1(val_pred_huber_log)
rmsle_2 = compute_rmsle(y_val, val_pred_huber)
best_iter_huber = model_huber.best_iteration_
print(f"  Best iteration: {best_iter_huber}")
print(f"  Val RMSLE: {rmsle_2:.6f}")

# ============================================================
# 9. MODEL 3: LightGBM with different config
# ============================================================
print("\n--- Training Model 3: LGBM (deeper, slower) ---")
model_lgb2 = lgb.LGBMRegressor(
    objective="huber",
    alpha=0.9,
    num_leaves=25,
    learning_rate=0.01,
    n_estimators=5000,
    feature_fraction=0.7,
    bagging_fraction=0.8,
    bagging_freq=5,
    min_child_samples=10,
    reg_alpha=1.0,
    reg_lambda=2.0,
    verbose=-1,
    n_jobs=-1,
    random_state=42,
)
model_lgb2.fit(
    X_train,
    y_train_log,
    eval_set=[(X_val, y_val_log)],
    callbacks=[lgb.early_stopping(300, verbose=False), lgb.log_evaluation(period=0)],
)

val_pred_lgb2_log = model_lgb2.predict(X_val)
val_pred_lgb2 = np.expm1(val_pred_lgb2_log)
rmsle_3 = compute_rmsle(y_val, val_pred_lgb2)
best_iter_lgb2 = model_lgb2.best_iteration_
print(f"  Best iteration: {best_iter_lgb2}")
print(f"  Val RMSLE: {rmsle_3:.6f}")

# ============================================================
# 10. MODEL 4: SVR with RBF kernel (provides ensemble diversity)
# ============================================================
print("\n--- Training Model 4: SVR (RBF kernel) ---")
model_svr = SVR(
    kernel="rbf",
    C=10,
    gamma="scale",
    epsilon=0.01,
)
model_svr.fit(X_train, y_train_log)

val_pred_svr_log = model_svr.predict(X_val)
val_pred_svr = np.expm1(val_pred_svr_log)
rmsle_4 = compute_rmsle(y_val, val_pred_svr)
print(f"  Val RMSLE: {rmsle_4:.6f}")

# ============================================================
# 11. ENSEMBLE: Inverse-error weighting (4 models)
# ============================================================
scores = np.array([rmsle_1, rmsle_2, rmsle_3, rmsle_4])
weights = 1.0 / (scores + 1e-8)
weights = weights / weights.sum()
print(f"\nIndividual RMSLE: [{rmsle_1:.6f}, {rmsle_2:.6f}, {rmsle_3:.6f}, {rmsle_4:.6f}]")
print(f"Ensemble weights: [{weights[0]:.4f}, {weights[1]:.4f}, {weights[2]:.4f}, {weights[3]:.4f}]")

# Validation ensemble
val_pred_ensemble = (
    weights[0] * val_pred_xgb
    + weights[1] * val_pred_huber
    + weights[2] * val_pred_lgb2
    + weights[3] * val_pred_svr
)
score = compute_rmsle(y_val, val_pred_ensemble)
print(f"Ensemble Val RMSLE: {score:.6f}")

# ============================================================
# 11. RETRAIN ON ALL DATA FOR FINAL SUBMISSION
# ============================================================
print("\n--- Retraining on ALL data for final submission ---")

# Model 1: XGBoost retrained on all data
model_xgb_final = xgb.XGBRegressor(
    objective="reg:squarederror",
    eval_metric="rmse",
    max_depth=4,
    learning_rate=0.02,
    n_estimators=best_iter_xgb,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    reg_alpha=0.3,
    reg_lambda=1.5,
    random_state=42,
    n_jobs=-1,
    verbosity=0,
)
model_xgb_final.fit(X_full, y_full_log)

# Model 2: LGBM Huber retrained on all data
model_huber_final = lgb.LGBMRegressor(
    objective="huber",
    alpha=0.9,
    num_leaves=15,
    learning_rate=0.02,
    n_estimators=best_iter_huber,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=5,
    min_child_samples=20,
    reg_alpha=0.3,
    reg_lambda=1.5,
    verbose=-1,
    n_jobs=-1,
    random_state=42,
)
model_huber_final.fit(X_full, y_full_log)

# Model 3: LGBM deeper retrained on all data
model_lgb2_final = lgb.LGBMRegressor(
    objective="huber",
    alpha=0.9,
    num_leaves=25,
    learning_rate=0.01,
    n_estimators=best_iter_lgb2,
    feature_fraction=0.7,
    bagging_fraction=0.8,
    bagging_freq=5,
    min_child_samples=10,
    reg_alpha=1.0,
    reg_lambda=2.0,
    verbose=-1,
    n_jobs=-1,
    random_state=42,
)
model_lgb2_final.fit(X_full, y_full_log)

# Model 4: SVR retrained on all data
model_svr_final = SVR(
    kernel="rbf",
    C=10,
    gamma="scale",
    epsilon=0.01,
)
model_svr_final.fit(X_full, y_full_log)

# ============================================================
# 12. TEST INFERENCE using retrained models
# ============================================================
test_pred_xgb = np.expm1(model_xgb_final.predict(X_test_arr))
test_pred_huber = np.expm1(model_huber_final.predict(X_test_arr))
test_pred_lgb2 = np.expm1(model_lgb2_final.predict(X_test_arr))
test_pred_svr = np.expm1(model_svr_final.predict(X_test_arr))

test_pred_ensemble = (
    weights[0] * test_pred_xgb
    + weights[1] * test_pred_huber
    + weights[2] * test_pred_lgb2
    + weights[3] * test_pred_svr
)

# Post-processing: clip predictions to physically valid range
test_pred_ensemble = np.clip(test_pred_ensemble, 0.0, 0.66)

print(
    f"\nTest prediction range: [{test_pred_ensemble.min():.4f}, {test_pred_ensemble.max():.4f}]"
)
print(
    f"Test prediction mean: {test_pred_ensemble.mean():.4f}, std: {test_pred_ensemble.std():.4f}"
)

# ============================================================
# 13. SAVE SUBMISSION
# ============================================================
os.makedirs("./submission", exist_ok=True)
n_test = len(test_df)
submission = pd.DataFrame(
    {"id": range(n_test), "formation_energy_ev_natom": test_pred_ensemble[:n_test]}
)
submission.to_csv("./submission/submission.csv", index=False)
print(f"Submission saved: {len(submission)} rows")
print(f"Submission head:\n{submission.head()}")

# ============================================================
# 14. FINAL OUTPUT
# ============================================================
print(f"\n=== SUMMARY ===")
print(
    f"Models: XGBoost (iter={best_iter_xgb}), "
    f"LGBM-Huber (iter={best_iter_huber}), "
    f"LGBM-deep (iter={best_iter_lgb2}), "
    f"SVR-RBF"
)
print(f"Weights: [{weights[0]:.4f}, {weights[1]:.4f}, {weights[2]:.4f}, {weights[3]:.4f}]")
print(f"Metric: RMSLE = sqrt(mean((log1p(|y_true|) - log1p(|y_pred|))^2))")
print(f"Validation split: 80% train / 20% val, random_state=42")
print(f"Final Validation Score: {score}")