import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import torch
import os

# ============================================================
# SETUP
# ============================================================
os.makedirs("./submission", exist_ok=True)
os.makedirs("./working", exist_ok=True)

# Load data
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")
print(
    f"Target stats: mean={train['critical_temp'].mean():.2f}, std={train['critical_temp'].std():.2f}"
)

target_col = "critical_temp"
id_col = "id" if "id" in train.columns else None

# 5-fold CV setup for ensemble weight selection
kf = KFold(n_splits=5, shuffle=True, random_state=42)
print(f"Using 5-fold CV for ensemble weight selection")

# ============================================================
# FEATURE ENGINEERING
# ============================================================


def engineer_features(df):
    """Apply domain-informed feature engineering for superconductivity prediction."""
    df = df.copy()

    properties = [
        "atomic_mass",
        "fie",
        "atomic_radius",
        "Density",
        "ElectronAffinity",
        "FusionHeat",
        "ThermalConductivity",
        "Valence",
    ]

    new_features = {}

    # === 1. Dispersion features (composition heterogeneity) ===
    for prop in properties:
        mean_col = f"mean_{prop}"
        std_col = f"std_{prop}"
        range_col = f"range_{prop}"
        gmean_col = f"gmean_{prop}"
        wtd_mean_col = f"wtd_mean_{prop}"
        wtd_std_col = f"wtd_std_{prop}"
        wtd_range_col = f"wtd_range_{prop}"
        entropy_col = f"entropy_{prop}"
        wtd_entropy_col = f"wtd_entropy_{prop}"

        if mean_col not in df.columns:
            continue

        safe_mean = df[mean_col].abs() + 1e-8

        if std_col in df.columns:
            new_features[f"cv_{prop}"] = df[std_col] / safe_mean
            new_features[f"cv_wtd_{prop}"] = (
                df[wtd_std_col] / safe_mean if wtd_std_col in df.columns else 0
            )

        if range_col in df.columns:
            new_features[f"range_cv_{prop}"] = df[range_col] / safe_mean
            new_features[f"range_cv_wtd_{prop}"] = (
                df[wtd_range_col] / safe_mean if wtd_range_col in df.columns else 0
            )

        if gmean_col in df.columns:
            new_features[f"geo_arith_{prop}"] = df[gmean_col] / safe_mean

        if wtd_mean_col in df.columns:
            new_features[f"wtd_shift_{prop}"] = (
                df[wtd_mean_col] - df[mean_col]
            ) / safe_mean

        if entropy_col in df.columns:
            new_features[f"ent_ratio_{prop}"] = df[entropy_col] / safe_mean
            if wtd_entropy_col in df.columns:
                new_features[f"wtd_ent_ratio_{prop}"] = df[wtd_entropy_col] / safe_mean
                new_features[f"ent_diff_{prop}"] = (
                    df[wtd_entropy_col] - df[entropy_col]
                ) / safe_mean

    # === 2. Cross-property interactions (physics-motivated) ===
    if "mean_fie" in df.columns and "mean_Valence" in df.columns:
        new_features["fie_x_valence"] = df["mean_fie"] * df["mean_Valence"]
        new_features["fie_per_valence"] = df["mean_fie"] / (
            df["mean_Valence"].abs() + 1e-8
        )

    if "mean_ElectronAffinity" in df.columns and "mean_Valence" in df.columns:
        new_features["ea_x_valence"] = df["mean_ElectronAffinity"] * df["mean_Valence"]
        new_features["ea_per_valence"] = df["mean_ElectronAffinity"] / (
            df["mean_Valence"].abs() + 1e-8
        )

    if "mean_fie" in df.columns and "mean_ElectronAffinity" in df.columns:
        new_features["fie_x_ea"] = df["mean_fie"] * df["mean_ElectronAffinity"]
        new_features["fie_ea_ratio"] = df["mean_fie"] / (
            df["mean_ElectronAffinity"].abs() + 1e-8
        )

    if "mean_atomic_mass" in df.columns and "mean_Density" in df.columns:
        new_features["mass_x_density"] = df["mean_atomic_mass"] * df["mean_Density"]
        new_features["mass_per_density"] = df["mean_atomic_mass"] / (
            df["mean_Density"].abs() + 1e-8
        )
        new_features["density_per_mass"] = df["mean_Density"] / (
            df["mean_atomic_mass"].abs() + 1e-8
        )

    if "mean_atomic_radius" in df.columns and "mean_atomic_mass" in df.columns:
        new_features["radius_x_mass"] = (
            df["mean_atomic_radius"] * df["mean_atomic_mass"]
        )
        new_features["radius_per_mass"] = df["mean_atomic_radius"] / (
            df["mean_atomic_mass"].abs() + 1e-8
        )

    if "mean_ThermalConductivity" in df.columns and "mean_atomic_mass" in df.columns:
        new_features["thermal_per_mass"] = df["mean_ThermalConductivity"] / (
            df["mean_atomic_mass"].abs() + 1e-8
        )

    if "mean_FusionHeat" in df.columns and "mean_atomic_mass" in df.columns:
        new_features["fusion_per_mass"] = df["mean_FusionHeat"] / (
            df["mean_atomic_mass"].abs() + 1e-8
        )

    if "mean_fie" in df.columns and "mean_atomic_mass" in df.columns:
        new_features["fie_per_mass"] = df["mean_fie"] / (
            df["mean_atomic_mass"].abs() + 1e-8
        )
        new_features["fie_x_mass"] = df["mean_fie"] * df["mean_atomic_mass"]

    if "mean_Valence" in df.columns and "mean_atomic_mass" in df.columns:
        new_features["valence_per_mass"] = df["mean_Valence"] / (
            df["mean_atomic_mass"].abs() + 1e-8
        )

    # === 3. Per-element normalized features ===
    if "number_of_elements" in df.columns:
        n_elem = df["number_of_elements"].astype(float) + 1e-8
        for prop in properties:
            mean_col = f"mean_{prop}"
            if mean_col in df.columns:
                new_features[f"per_elem_{prop}"] = df[mean_col] / n_elem

    # === 4. Higher-order interactions ===
    if "mean_Valence" in df.columns and "mean_atomic_radius" in df.columns:
        radius_safe = df["mean_atomic_radius"].abs() + 1e-8
        new_features["electron_density"] = df["mean_Valence"] / (radius_safe**3)

    if (
        "mean_ThermalConductivity" in df.columns
        and "mean_atomic_mass" in df.columns
        and "mean_Density" in df.columns
    ):
        mass_safe = df["mean_atomic_mass"].abs() + 1e-8
        density_safe = df["mean_Density"].abs() + 1e-8
        new_features["debye_proxy"] = np.sqrt(
            df["mean_ThermalConductivity"] * density_safe / mass_safe
        )

    if (
        "mean_Valence" in df.columns
        and "mean_atomic_mass" in df.columns
        and "mean_atomic_radius" in df.columns
    ):
        mass_safe = df["mean_atomic_mass"].abs() + 1e-8
        radius_safe = df["mean_atomic_radius"].abs() + 1e-8
        new_features["n_ef_proxy"] = df["mean_Valence"] / (mass_safe * radius_safe)

    # === 5. Ratio features between weighted and unweighted ===
    for prop in properties:
        wtd_mean_col = f"wtd_mean_{prop}"
        mean_col = f"mean_{prop}"
        wtd_gmean_col = f"wtd_gmean_{prop}"
        gmean_col = f"gmean_{prop}"

        if wtd_mean_col in df.columns and mean_col in df.columns:
            safe = df[mean_col].abs() + 1e-8
            new_features[f"wtd_vs_mean_{prop}"] = df[wtd_mean_col] / safe

        if wtd_gmean_col in df.columns and gmean_col in df.columns:
            safe = df[gmean_col].abs() + 1e-8
            new_features[f"wtd_vs_gmean_{prop}"] = df[wtd_gmean_col] / safe

    # === 6. Product of dispersion measures ===
    cv_features = [f"cv_{prop}" for prop in properties if f"cv_{prop}" in new_features]
    if len(cv_features) > 1:
        cv_vals = np.ones(len(df))
        for cf in cv_features:
            cv_vals = cv_vals * (new_features[cf].abs() + 1e-8)
        new_features["total_heterogeneity"] = np.log(cv_vals + 1e-8)

    # === 7. Physics-motivated superconductivity features (BCS/McMillan theory) ===
    if (
        "mean_Valence" in df.columns
        and "mean_Density" in df.columns
        and "mean_fie" in df.columns
        and "mean_atomic_mass" in df.columns
    ):
        # (1) Electron-phonon coupling proxy: lambda_proxy
        lambda_proxy = (
            df["mean_Valence"] * df["mean_Density"]
            / (df["mean_fie"].abs() * df["mean_atomic_mass"].abs() + 1e-8)
        )
        new_features["lambda_proxy"] = lambda_proxy

        # (2) Phonon frequency proxy: omega_log_proxy
        omega_log_proxy = np.sqrt(
            df["mean_Density"] / (df["mean_atomic_mass"].abs() + 1e-8)
        ) * 100
        new_features["omega_log_proxy"] = omega_log_proxy

        # (3) McMillan equation input: ratio omega_log_proxy / lambda_proxy
        mcmillan_ratio = omega_log_proxy / (lambda_proxy.abs() + 1e-8)
        new_features["mcmillan_ratio"] = mcmillan_ratio

        # (4) BCS gap proxy: mean_fie * exp(-1/lambda_proxy)
        bcs_gap_proxy = df["mean_fie"] * np.exp(-1.0 / (lambda_proxy.abs() + 1e-8))
        new_features["bcs_gap_proxy"] = bcs_gap_proxy

    # (5) Composition complexity: number_of_elements * std_Valence
    if "number_of_elements" in df.columns and "std_Valence" in df.columns:
        new_features["composition_complexity"] = (
            df["number_of_elements"] * df["std_Valence"]
        )

    # Add all new features
    for name, values in new_features.items():
        df[name] = values

    return df


# Apply feature engineering
train_full = engineer_features(train)
test = engineer_features(test)

# Get feature columns
if id_col:
    feature_cols = [c for c in train_full.columns if c != target_col and c != id_col]
else:
    feature_cols = [c for c in train_full.columns if c != target_col]

print(f"Total features after engineering: {len(feature_cols)}")

# Handle inf/NaN
for df in [train_full, test]:
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    df[feature_cols] = df[feature_cols].fillna(0)

print(f"Train NaNs: {train_full[feature_cols].isna().sum().sum()}")
print(f"Test NaNs: {test[feature_cols].isna().sum().sum()}")

# Prepare matrices
X_train_full = train_full[feature_cols].values.astype(np.float32)
y_train_full = train_full[target_col].values.astype(np.float32)
X_test = test[feature_cols].values.astype(np.float32)

print(
    f"Model input shapes - Train: {X_train_full.shape}, Test: {X_test.shape}"
)

# ============================================================
# 5-FOLD CV: Generate OOF predictions for ensemble weight selection
# ============================================================
print("\n=== 5-Fold CV for Ensemble Weight Selection ===")

lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.02,
    "num_leaves": 127,
    "max_depth": 7,
    "min_child_samples": 15,
    "feature_fraction": 0.6,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "verbosity": -1,
    "n_jobs": -1,
    "seed": 42,
}

catboost_params = {
    "iterations": 5000,
    "learning_rate": 0.03,
    "depth": 8,
    "l2_leaf_reg": 3,
    "random_seed": 42,
    "loss_function": "RMSE",
    "verbose": 0,
}

xgb_params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "max_depth": 8,
    "colsample_bytree": 0.6,
    "subsample": 0.8,
    "reg_alpha": 0.5,
    "reg_lambda": 2.0,
    "learning_rate": 0.03,
    "n_estimators": 5000,
    "tree_method": "hist",
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": 0,
    "early_stopping_rounds": 100,
}

# OOF prediction arrays
oof_lgb = np.zeros(len(X_train_full))
oof_xgb = np.zeros(len(X_train_full))
oof_cb = np.zeros(len(X_train_full))
xgb_best_iters = []
lgb_best_iters = []
cb_best_iters = []

for fold, (tr_idx, va_idx) in enumerate(kf.split(X_train_full)):
    print(f"\n--- Fold {fold+1}/5 ---")
    X_tr, X_va = X_train_full[tr_idx], X_train_full[va_idx]
    y_tr, y_va = y_train_full[tr_idx], y_train_full[va_idx]

    # LightGBM fold
    lgb_train = lgb.Dataset(X_tr, label=y_tr, feature_name=feature_cols)
    lgb_val = lgb.Dataset(X_va, label=y_va, feature_name=feature_cols, reference=lgb_train)
    callbacks = [
        lgb.early_stopping(stopping_rounds=100, verbose=False),
        lgb.log_evaluation(period=0),
    ]
    lgb_model = lgb.train(
        lgb_params, lgb_train, num_boost_round=5000,
        valid_sets=[lgb_val], callbacks=callbacks,
    )
    oof_lgb[va_idx] = lgb_model.predict(X_va, num_iteration=lgb_model.best_iteration)
    lgb_best_iters.append(lgb_model.best_iteration)
    print(f"  LGBM best_iter: {lgb_model.best_iteration}")

    # XGBoost fold
    xgb_model = xgb.XGBRegressor(**xgb_params)
    xgb_model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        verbose=False,
    )
    best_iter = xgb_model.best_iteration
    xgb_best_iters.append(best_iter)
    oof_xgb[va_idx] = xgb_model.predict(X_va, iteration_range=(0, best_iter + 1))
    print(f"  XGB best_iter: {best_iter}")

    # CatBoost fold
    cb_model = CatBoostRegressor(**catboost_params)
    cb_model.fit(
        X_tr, y_tr,
        eval_set=(X_va, y_va),
        early_stopping_rounds=100,
        verbose=0,
    )
    cb_best_iter = cb_model.get_best_iteration()
    cb_best_iters.append(cb_best_iter)
    oof_cb[va_idx] = cb_model.predict(X_va)
    print(f"  CatBoost best_iter: {cb_best_iter}")

# Compute OOF RMSEs
lgb_oof_rmse = np.sqrt(mean_squared_error(y_train_full, oof_lgb))
xgb_oof_rmse = np.sqrt(mean_squared_error(y_train_full, oof_xgb))
cb_oof_rmse = np.sqrt(mean_squared_error(y_train_full, oof_cb))
print(f"\nLightGBM OOF RMSE: {lgb_oof_rmse:.4f}")
print(f"XGBoost OOF RMSE: {xgb_oof_rmse:.4f}")
print(f"CatBoost OOF RMSE: {cb_oof_rmse:.4f}")

# ============================================================
# ENSEMBLE: Optimal weight search on OOF predictions
# ============================================================
print("\n=== Ensemble Optimization (OOF) ===")

best_w1, best_w2, best_w3 = 1/3, 1/3, 1/3
best_ens_rmse = float("inf")
step = 0.05
for w1 in np.arange(0.0, 1.01, step):
    for w2 in np.arange(0.0, 1.01 - w1, step):
        w3 = 1.0 - w1 - w2
        if w3 < -1e-9:
            continue
        w3 = max(w3, 0.0)
        ens_pred = w1 * oof_lgb + w2 * oof_xgb + w3 * oof_cb
        ens_rmse = np.sqrt(mean_squared_error(y_train_full, ens_pred))
        if ens_rmse < best_ens_rmse:
            best_ens_rmse = ens_rmse
            best_w1, best_w2, best_w3 = w1, w2, w3

print(f"Optimal weights - LGBM: {best_w1:.3f}, XGB: {best_w2:.3f}, CatBoost: {best_w3:.3f}")
print(f"Ensemble OOF RMSE: {best_ens_rmse:.4f}")

# ============================================================
# RETRAIN ON FULL DATA FOR FINAL PREDICTIONS
# ============================================================
print("\n=== Retraining on Full Training Data ===")

# LightGBM full model
lgb_train_full = lgb.Dataset(X_train_full, label=y_train_full, feature_name=feature_cols)

# Use average best_iteration from CV folds as num_boost_round
avg_lgb_best_iter = int(np.mean(lgb_best_iters))
lgb_model_full = lgb.train(
    lgb_params, lgb_train_full, num_boost_round=avg_lgb_best_iter,
    callbacks=[lgb.log_evaluation(period=0)],
)
test_pred_lgb = lgb_model_full.predict(X_test)
print(f"LightGBM full model trained ({avg_lgb_best_iter} rounds)")

# XGBoost full model
avg_xgb_best_iter = int(np.mean(xgb_best_iters))
xgb_params_full = {k: v for k, v in xgb_params.items() if k != "early_stopping_rounds"}
xgb_params_full["n_estimators"] = avg_xgb_best_iter
xgb_model_full = xgb.XGBRegressor(**xgb_params_full)
xgb_model_full.fit(X_train_full, y_train_full, verbose=False)
test_pred_xgb = xgb_model_full.predict(X_test)
print(f"XGBoost full model trained ({avg_xgb_best_iter} rounds)")

# CatBoost full model
avg_cb_best_iter = int(np.mean(cb_best_iters))
cb_params_full = {k: v for k, v in catboost_params.items() if k != "iterations"}
cb_params_full["iterations"] = avg_cb_best_iter
cb_model_full = CatBoostRegressor(**cb_params_full)
cb_model_full.fit(X_train_full, y_train_full, verbose=0)
test_pred_cb = cb_model_full.predict(X_test)
print(f"CatBoost full model trained ({avg_cb_best_iter} rounds)")

# Final ensemble prediction
test_pred_final = (
    best_w1 * test_pred_lgb
    + best_w2 * test_pred_xgb
    + best_w3 * test_pred_cb
)

# ============================================================
# GENERATE SUBMISSION
# ============================================================
submission = pd.DataFrame(
    {"id": np.arange(len(test_pred_final)), "critical_temp": test_pred_final}
)
submission.to_csv("./submission/submission.csv", index=False)
print(f"\nSubmission saved: {submission.shape}")
print(
    f"Prediction stats - mean: {test_pred_final.mean():.2f}, std: {test_pred_final.std():.2f}"
)
print(f"Prediction range: [{test_pred_final.min():.2f}, {test_pred_final.max():.2f}]")

# ============================================================
# FINAL VALIDATION SCORE (Official Metric: RMSE)
# ============================================================
print(f"\n=== Final Summary ===")
print(f"LightGBM OOF RMSE: {lgb_oof_rmse:.4f}")
print(f"XGBoost OOF RMSE: {xgb_oof_rmse:.4f}")
print(f"CatBoost OOF RMSE: {cb_oof_rmse:.4f}")
print(f"Ensemble weights: LGBM={best_w1:.3f}, XGB={best_w2:.3f}, CatBoost={best_w3:.3f}")
print(f"Ensemble OOF RMSE: {best_ens_rmse:.4f}")
print(f"Final Validation Score: {best_ens_rmse:.4f}")