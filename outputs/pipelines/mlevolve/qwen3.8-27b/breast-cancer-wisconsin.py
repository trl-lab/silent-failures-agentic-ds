import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import xgboost as xgb
import os

# Create directories
os.makedirs("./submission", exist_ok=True)
os.makedirs("./working", exist_ok=True)

# ============================================================
# STEP 1: Data Processing & Feature Engineering
# ============================================================

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")
print(f"Target distribution:\n{train_df['Diagnosis'].value_counts()}")

# Identify feature groups
base_features = [
    "radius",
    "texture",
    "perimeter",
    "area",
    "smoothness",
    "compactness",
    "concavity",
    "concave_points",
    "symmetry",
    "fractal_dimension",
]

group_cols = {}
for bf in base_features:
    group_cols[bf] = [f"{bf}{i}" for i in [1, 2, 3]]


def engineer_features(df):
    """Apply feature engineering to a dataframe."""
    feats = pd.DataFrame(index=df.index)

    # 1. Group aggregate features (mean, std, min, max, range, cv)
    for bf in base_features:
        cols = group_cols[bf]
        group = df[cols].values
        feats[f"{bf}_mean"] = group.mean(axis=1)
        feats[f"{bf}_std"] = group.std(axis=1)
        feats[f"{bf}_min"] = group.min(axis=1)
        feats[f"{bf}_max"] = group.max(axis=1)
        feats[f"{bf}_range"] = group.max(axis=1) - group.min(axis=1)
        feats[f"{bf}_cv"] = group.std(axis=1) / (group.mean(axis=1) + 1e-10)

    # 2. Keep original features
    for bf in base_features:
        for i in [1, 2, 3]:
            feats[f"{bf}{i}"] = df[f"{bf}{i}"]

    # 3. Domain-informed interaction features
    for i in [1, 2, 3]:
        feats[f"circularity{i}"] = (
            df[f"perimeter{i}"] / (2 * np.pi * df[f"radius{i}"] + 1e-10) - 1
        )
        feats[f"shape_compactness{i}"] = df[f"area{i}"] / (
            df[f"perimeter{i}"] ** 2 + 1e-10
        )
        feats[f"smooth_compact{i}"] = df[f"smoothness{i}"] * df[f"compactness{i}"]
        feats[f"concavity_points{i}"] = df[f"concavity{i}"] * df[f"concave_points{i}"]
        feats[f"fractal_symmetry{i}"] = df[f"fractal_dimension{i}"] * df[f"symmetry{i}"]
        feats[f"area_radius_ratio{i}"] = df[f"area{i}"] / (
            df[f"radius{i}"] ** 2 + 1e-10
        )
        feats[f"texture_radius_ratio{i}"] = df[f"texture{i}"] / (
            df[f"radius{i}"] + 1e-10
        )

    # 4. Log transforms for skewed features
    for bf in ["radius", "perimeter", "area"]:
        for i in [1, 2, 3]:
            feats[f"{bf}{i}_log"] = np.log1p(df[f"{bf}{i}"])

    # 5. Cross-measurement heterogeneity features
    for bf in base_features:
        cols = group_cols[bf]
        group = df[cols].values
        mean = group.mean(axis=1, keepdims=True)
        max_dev = np.abs(group - mean).max(axis=1)
        feats[f"{bf}_max_dev"] = max_dev

    return feats


# Apply feature engineering
train_feats = engineer_features(train_df)
test_feats = engineer_features(test_df)

# Create stratified train/val split
y_train_full = train_df["Diagnosis"].values
X_train_full = train_feats

X_train, X_val, y_train_split, y_val_split = train_test_split(
    X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
)

print(f"Feature count: {X_train.shape[1]}")
print(
    f"Train split: {X_train.shape[0]}, Val split: {X_val.shape[0]}, Test: {test_feats.shape[0]}"
)

# Fit scaler on train only, transform all (no data leakage)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(test_feats)

# Encode target
y_train_encoded = (y_train_split == "M").astype(int)
y_val_encoded = (y_val_split == "M").astype(int)

print(
    f"Train class distribution: M={y_train_encoded.sum()}, B={len(y_train_encoded)-y_train_encoded.sum()}"
)
print(
    f"Val class distribution: M={y_val_encoded.sum()}, B={len(y_val_encoded)-y_val_encoded.sum()}"
)

# Test IDs
test_ids = np.arange(len(test_df))

# ============================================================
# STEP 2: Model Design (Random Forest)
# ============================================================

# Initialize Random Forest classifier
rf_model = RandomForestClassifier(
    n_estimators=500,
    max_depth=6,
    min_samples_leaf=3,
    class_weight='balanced',
    random_state=42,
)

# ============================================================
# STEP 3: Training & Evaluation (Random Forest)
# ============================================================

rf_model.fit(X_train_scaled, y_train_encoded)

rf_val_preds = rf_model.predict_proba(X_val_scaled)[:, 1]
rf_test_preds = rf_model.predict_proba(X_test_scaled)[:, 1]

rf_val_auc = roc_auc_score(y_val_encoded, rf_val_preds)
print(f"\nRandom Forest Val ROC-AUC: {rf_val_auc:.4f}")

# ============================================================
# STEP 4: LightGBM Model
# ============================================================

lgb_model = lgb.LGBMClassifier(
    num_leaves=31,
    learning_rate=0.05,
    n_estimators=500,
    random_state=42,
    verbose=-1,
)

lgb_model.fit(
    X_train_scaled,
    y_train_encoded,
    eval_set=[(X_val_scaled, y_val_encoded)],
    eval_metric="auc",
    callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
)

lgb_val_preds = lgb_model.predict_proba(X_val_scaled)[:, 1]
lgb_test_preds = lgb_model.predict_proba(X_test_scaled)[:, 1]

lgb_val_auc = roc_auc_score(y_val_encoded, lgb_val_preds)
print(f"LightGBM Val ROC-AUC: {lgb_val_auc:.4f}")

# ============================================================
# STEP 5: XGBoost Model
# ============================================================

xgb_model = xgb.XGBClassifier(
    max_depth=4,
    learning_rate=0.05,
    n_estimators=500,
    random_state=42,
    eval_metric="auc",
    use_label_encoder=False,
)

xgb_model.fit(
    X_train_scaled,
    y_train_encoded,
    eval_set=[(X_val_scaled, y_val_encoded)],
    verbose=False,
)

xgb_val_preds = xgb_model.predict_proba(X_val_scaled)[:, 1]
xgb_test_preds = xgb_model.predict_proba(X_test_scaled)[:, 1]

xgb_val_auc = roc_auc_score(y_val_encoded, xgb_val_preds)
print(f"XGBoost Val ROC-AUC: {xgb_val_auc:.4f}")

# ============================================================
# STEP 6: Ensemble Prediction with Optimized Weights
# ============================================================

# Grid search over weight combinations (w1=RF, w2=LGBM, w3=XGB) where w1+w2+w3=1
best_weight_auc = 0.0
best_weights = (1/3, 1/3, 1/3)

step = 0.05
for w1 in np.arange(0, 1.0 + step, step):
    for w2 in np.arange(0, 1.0 - w1 + step, step):
        w3 = 1.0 - w1 - w2
        if w3 < -1e-10:
            continue
        w3 = max(w3, 0.0)
        weighted_val = w1 * rf_val_preds + w2 * lgb_val_preds + w3 * xgb_val_preds
        auc = roc_auc_score(y_val_encoded, weighted_val)
        if auc > best_weight_auc:
            best_weight_auc = auc
            best_weights = (w1, w2, w3)

print(f"Optimized weights (RF, LGBM, XGB): {best_weights}")
print(f"Optimized ensemble Val ROC-AUC: {best_weight_auc:.4f}")

# 5-fold cross-validation ensemble for more robust test predictions
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_test_preds_list = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_scaled, y_train_encoded)):
    X_tr = X_train_scaled[train_idx]
    X_vl = X_train_scaled[val_idx]
    y_tr = y_train_encoded[train_idx]

    # RF
    rf_cv = RandomForestClassifier(
        n_estimators=500, max_depth=6, min_samples_leaf=3,
        class_weight='balanced', random_state=42
    )
    rf_cv.fit(X_tr, y_tr)
    rf_cv_test = rf_cv.predict_proba(X_test_scaled)[:, 1]

    # LGBM
    lgb_cv = lgb.LGBMClassifier(
        num_leaves=31, learning_rate=0.05, n_estimators=500,
        random_state=42, verbose=-1
    )
    lgb_cv.fit(X_tr, y_tr)
    lgb_cv_test = lgb_cv.predict_proba(X_test_scaled)[:, 1]

    # XGB
    xgb_cv = xgb.XGBClassifier(
        max_depth=4, learning_rate=0.05, n_estimators=500,
        random_state=42, eval_metric="auc", use_label_encoder=False
    )
    xgb_cv.fit(X_tr, y_tr, verbose=False)
    xgb_cv_test = xgb_cv.predict_proba(X_test_scaled)[:, 1]

    # Apply optimized weights
    w1, w2, w3 = best_weights
    cv_fold_test = w1 * rf_cv_test + w2 * lgb_cv_test + w3 * xgb_cv_test
    cv_test_preds_list.append(cv_fold_test)

cv_ensemble_test_preds = np.mean(cv_test_preds_list, axis=0)

# Final ensemble: average of weighted single-split and 5-fold CV ensemble
ensemble_val_preds = best_weights[0] * rf_val_preds + best_weights[1] * lgb_val_preds + best_weights[2] * xgb_val_preds
ensemble_test_preds = (
    best_weights[0] * rf_test_preds + best_weights[1] * lgb_test_preds + best_weights[2] * xgb_test_preds + cv_ensemble_test_preds
) / 2.0

final_val_auc = roc_auc_score(y_val_encoded, ensemble_val_preds)
print(f"\nEnsemble Val ROC-AUC: {final_val_auc:.4f}")

# Create submission
submission = pd.DataFrame({"id": test_ids.astype(int), "Diagnosis": ensemble_test_preds})
submission.to_csv("./submission/submission.csv", index=False)
print(f"Submission saved: {submission.shape}")
print(f"Prediction range: [{ensemble_test_preds.min():.4f}, {ensemble_test_preds.max():.4f}]")
print(f"Prediction mean: {ensemble_test_preds.mean():.4f}")

score = final_val_auc
print(f"Final Validation Score: {score}")
