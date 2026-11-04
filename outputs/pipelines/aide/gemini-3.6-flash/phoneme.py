import os
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
import lightgbm as lgb
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
import xgboost as xgb

# Set up output directory
os.makedirs("./working", exist_ok=True)

# Read input datasets
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

target_col = "class"
feature_cols = [c for c in train.columns if c != target_col and c != "id"]

# Reliable target encoding
le = LabelEncoder()
y_train = le.fit_transform(train[target_col])


# Domain-specific Acoustic & Harmonic Feature Engineering
def create_features(df, cols):
    X = df[cols].copy()
    for col in cols:
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0.0)

    vals = X[cols].values.astype(np.float32)
    abs_vals = np.abs(vals)
    n_samples, n_harmonics = vals.shape

    # Harmonic weights for spectral statistics (1-indexed for 5 harmonics)
    weights = np.arange(1, n_harmonics + 1, dtype=np.float32)

    # basic absolute, squared, and sign representations
    for i, c in enumerate(cols):
        X[f"{c}_abs"] = abs_vals[:, i]
        X[f"{c}_sign"] = np.sign(vals[:, i])
        X[f"{c}_sq"] = vals[:, i] ** 2
        X[f"{c}_log_abs"] = np.log1p(abs_vals[:, i])

    # Spectral Centroid: weighted average harmonic index
    sum_abs = abs_vals.sum(axis=1, keepdims=True) + 1e-8
    centroid = np.sum(abs_vals * weights, axis=1, keepdims=True) / sum_abs
    X["spectral_centroid"] = centroid.ravel()

    # Spectral Spread (Variance around centroid)
    spread = (
        np.sum(abs_vals * ((weights - centroid) ** 2), axis=1, keepdims=True) / sum_abs
    )
    X["spectral_spread"] = spread.ravel()

    # Spectral Tilt (Linear slope across harmonic amplitudes)
    weights_centered = weights - np.mean(weights)
    var_w = np.sum(weights_centered**2)
    tilt = (np.sum(vals * weights_centered, axis=1, keepdims=True) / var_w).ravel()
    tilt_abs = (
        np.sum(abs_vals * weights_centered, axis=1, keepdims=True) / var_w
    ).ravel()
    X["spectral_tilt"] = tilt
    X["spectral_tilt_abs"] = tilt_abs

    # Spectral Flatness (Geometric Mean / Arithmetic Mean)
    geom_mean = np.exp(np.mean(np.log(abs_vals + 1e-6), axis=1))
    arith_mean = np.mean(abs_vals, axis=1) + 1e-6
    X["spectral_flatness"] = geom_mean / arith_mean

    # Harmonic Energy Sub-band Ratios
    low_freq_energy = np.sum(abs_vals[:, :2], axis=1) + 1e-8
    high_freq_energy = np.sum(abs_vals[:, 2:], axis=1) + 1e-8
    X["low_freq_energy"] = low_freq_energy
    X["high_freq_energy"] = high_freq_energy
    X["low_to_high_ratio"] = low_freq_energy / high_freq_energy
    X["positive_harmonics_count"] = (vals > 0).sum(axis=1)

    # Statistical aggregates across harmonics
    X["sum"] = vals.sum(axis=1)
    X["abs_sum"] = abs_vals.sum(axis=1)
    X["mean"] = vals.mean(axis=1)
    X["abs_mean"] = abs_vals.mean(axis=1)
    X["std"] = vals.std(axis=1)
    X["max"] = vals.max(axis=1)
    X["min"] = vals.min(axis=1)
    X["abs_max"] = abs_vals.max(axis=1)
    X["abs_min"] = abs_vals.min(axis=1)

    # Energy normalizations
    total_energy = np.sum(vals**2, axis=1, keepdims=True) + 1e-8
    for i, c in enumerate(cols):
        X[f"{c}_energy_ratio"] = (vals[:, i : i + 1] ** 2) / total_energy

    # Pairwise feature interactions
    for i in range(n_harmonics):
        for j in range(i + 1, n_harmonics):
            f1, f2 = cols[i], cols[j]
            v1, v2 = vals[:, i], vals[:, j]
            X[f"{f1}_minus_{f2}"] = v1 - v2
            X[f"{f1}_plus_{f2}"] = v1 + v2
            X[f"{f1}_prod_{f2}"] = v1 * v2
            X[f"{f1}_ratio_{f2}"] = v1 / (v2 + np.sign(v2) * 1e-6 + 1e-6)
            X[f"{f1}_abs_ratio_{f2}"] = np.abs(v1) / (np.abs(v2) + 1e-6)

    return X


# Generate engineered features
X_train = create_features(train, feature_cols)
X_test = create_features(test, feature_cols)

# Define models to ensemble
model_names = ["lgb", "cb", "xgb", "et", "hgb", "svc", "knn"]

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_model_preds = {m: np.zeros(len(X_train)) for m in model_names}
test_model_preds = {m: np.zeros(len(X_test)) for m in model_names}

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    X_tr, y_tr = X_train.iloc[train_idx], y_train[train_idx]
    X_va, y_va = X_train.iloc[val_idx], y_train[val_idx]

    # Standard scale for distance and kernel-based models
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_va_scaled = scaler.transform(X_va)
    X_test_scaled = scaler.transform(X_test)

    # Initialize models
    lgb_m = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=31,
        random_state=42 + fold,
        verbose=-1,
    )
    cb_m = CatBoostClassifier(
        iterations=1000, learning_rate=0.03, depth=6, random_seed=42 + fold, verbose=0
    )
    xgb_m = xgb.XGBClassifier(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=6,
        random_state=42 + fold,
        eval_metric="logloss",
        early_stopping_rounds=50,
    )
    et_m = ExtraTreesClassifier(n_estimators=500, random_state=42 + fold, n_jobs=-1)
    hgb_m = HistGradientBoostingClassifier(
        max_iter=500, learning_rate=0.03, random_state=42 + fold
    )
    svc_m = SVC(C=2.0, kernel="rbf", probability=True, random_state=42 + fold)
    knn_m = KNeighborsClassifier(n_neighbors=15, weights="distance", n_jobs=-1)

    # Train tree models
    lgb_m.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    cb_m.fit(X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=50)
    xgb_m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    et_m.fit(X_tr, y_tr)
    hgb_m.fit(X_tr, y_tr)

    # Train distance/kernel models on scaled data
    svc_m.fit(X_tr_scaled, y_tr)
    knn_m.fit(X_tr_scaled, y_tr)

    # Predict validation probabilities
    oof_model_preds["lgb"][val_idx] = lgb_m.predict_proba(X_va)[:, 1]
    oof_model_preds["cb"][val_idx] = cb_m.predict_proba(X_va)[:, 1]
    oof_model_preds["xgb"][val_idx] = xgb_m.predict_proba(X_va)[:, 1]
    oof_model_preds["et"][val_idx] = et_m.predict_proba(X_va)[:, 1]
    oof_model_preds["hgb"][val_idx] = hgb_m.predict_proba(X_va)[:, 1]
    oof_model_preds["svc"][val_idx] = svc_m.predict_proba(X_va_scaled)[:, 1]
    oof_model_preds["knn"][val_idx] = knn_m.predict_proba(X_va_scaled)[:, 1]

    # Predict test probabilities
    test_model_preds["lgb"] += lgb_m.predict_proba(X_test)[:, 1] / skf.n_splits
    test_model_preds["cb"] += cb_m.predict_proba(X_test)[:, 1] / skf.n_splits
    test_model_preds["xgb"] += xgb_m.predict_proba(X_test)[:, 1] / skf.n_splits
    test_model_preds["et"] += et_m.predict_proba(X_test)[:, 1] / skf.n_splits
    test_model_preds["hgb"] += hgb_m.predict_proba(X_test)[:, 1] / skf.n_splits
    test_model_preds["svc"] += svc_m.predict_proba(X_test_scaled)[:, 1] / skf.n_splits
    test_model_preds["knn"] += knn_m.predict_proba(X_test_scaled)[:, 1] / skf.n_splits

# Prepare out-of-fold feature matrix for stacking
OOF_meta = np.column_stack([oof_model_preds[m] for m in model_names])
Test_meta = np.column_stack([test_model_preds[m] for m in model_names])

# Stacking Meta-Learner (Logistic Regression)
meta_oof_preds = np.zeros(len(X_train))
meta_test_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(skf.split(OOF_meta, y_train)):
    meta_model = LogisticRegression(C=1.0, max_iter=1000, random_state=42 + fold)
    meta_model.fit(OOF_meta[train_idx], y_train[train_idx])
    meta_oof_preds[val_idx] = meta_model.predict_proba(OOF_meta[val_idx])[:, 1]
    meta_test_preds += meta_model.predict_proba(Test_meta)[:, 1] / skf.n_splits

# Optimize decision threshold on stacked out-of-fold predictions
best_thresh = 0.5
best_acc = 0.0
for thresh in np.linspace(0.3, 0.7, 101):
    acc = accuracy_score(y_train, (meta_oof_preds >= thresh).astype(int))
    if acc > best_acc:
        best_acc = acc
        best_thresh = thresh

print(f"5-Fold CV Accuracy: {best_acc:.5f} (threshold: {best_thresh:.3f})")

# Generate final test set predictions
final_test_binary = (meta_test_preds >= best_thresh).astype(int)
final_test_preds = le.inverse_transform(final_test_binary)

# Prepare submission file
if "id" in test.columns:
    ids = test["id"]
else:
    ids = np.arange(len(test))

submission = pd.DataFrame({"id": ids, "class": final_test_preds})
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved successfully.")
