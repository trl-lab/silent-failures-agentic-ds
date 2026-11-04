import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

# ============================================================
# 1. LOAD DATA
# ============================================================
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")
print(f"Target distribution:\n{train_df['class'].value_counts()}")

target_col = "class"
feature_cols = [c for c in train_df.columns if c != target_col]

# Rename feature columns to expected canonical names (actual names may differ)
expected_feature_names = [
    "short-line-density-5",
    "short-line-density-2",
    "vedge-mean",
    "vedge-sd",
    "hedge-mean",
    "hedge-sd",
    "intensity-mean",
    "rawred-mean",
    "rawblue-mean",
    "rawgreen-mean",
    "exred-mean",
    "exblue-mean",
    "exgreen-mean",
    "value-mean",
    "saturation-mean",
    "hue-mean",
]
print(f"Actual train feature columns: {feature_cols}")
train_df = train_df.rename(columns=dict(zip(feature_cols, expected_feature_names)))
test_df = test_df.rename(columns=dict(zip(list(test_df.columns), expected_feature_names)))
feature_cols = expected_feature_names


# ============================================================
# 2. FEATURE ENGINEERING
# ============================================================
def engineer_features(df):
    df = df.copy()

    # --- A. COLOR SPACE FEATURES ---
    r = df["rawred-mean"]
    g = df["rawgreen-mean"]
    b = df["rawblue-mean"]

    r_norm = r / 255.0
    g_norm = g / 255.0
    b_norm = b / 255.0

    max_c = np.maximum(np.maximum(r_norm, g_norm), b_norm)
    min_c = np.minimum(np.minimum(r_norm, g_norm), b_norm)
    delta = max_c - min_c

    df["chroma_r"] = np.where(max_c > 0, r_norm / max_c, 0)
    df["chroma_g"] = np.where(max_c > 0, g_norm / max_c, 0)
    df["chroma_b"] = np.where(max_c > 0, b_norm / max_c, 0)

    df["color_temperature"] = r_norm - b_norm
    df["rg_balance"] = np.abs(r_norm - g_norm)
    df["gb_balance"] = np.abs(g_norm - b_norm)
    df["rb_balance"] = np.abs(r_norm - b_norm)

    # --- B. EDGE/TEXTURE FEATURES ---
    df["vedge_sharpness"] = np.where(
        df["vedge-sd"] > 1e-8, df["vedge-mean"] / (df["vedge-sd"] + 1e-8), 0
    )
    df["hedge_sharpness"] = np.where(
        df["hedge-sd"] > 1e-8, df["hedge-mean"] / (df["hedge-sd"] + 1e-8), 0
    )

    df["edge_energy"] = df["vedge-mean"] + df["hedge-mean"]
    df["edge_energy_sq"] = df["vedge-mean"] ** 2 + df["hedge-mean"] ** 2
    df["edge_anisotropy"] = np.abs(df["vedge-mean"] - df["hedge-mean"])
    df["edge_ratio"] = np.where(
        df["hedge-mean"] > 1e-8, df["vedge-mean"] / (df["hedge-mean"] + 1e-8), 0
    )

    df["texture_complexity"] = df["vedge-sd"] + df["hedge-sd"]
    df["texture_anisotropy"] = np.abs(df["vedge-sd"] - df["hedge-sd"])

    df["line_density_total"] = df["short-line-density-5"] + df["short-line-density-2"]
    df["line_density_ratio"] = np.where(
        df["short-line-density-2"] > 1e-8,
        df["short-line-density-5"] / (df["short-line-density-2"] + 1e-8),
        0,
    )

    # --- C. COLOR-EDGE INTERACTION FEATURES ---
    df["color_edge_interaction_v"] = df["intensity-mean"] * df["vedge-mean"]
    df["color_edge_interaction_h"] = df["intensity-mean"] * df["hedge-mean"]

    df["sat_weighted_vedge"] = df["saturation-mean"] * df["vedge-mean"]
    df["sat_weighted_hedge"] = df["saturation-mean"] * df["hedge-mean"]

    df["hue_weighted_vedge"] = df["hue-mean"] * df["vedge-mean"]
    df["hue_weighted_hedge"] = df["hue-mean"] * df["hedge-mean"]

    # --- D. PERCEPTUAL COLOR FEATURES ---
    df["value_sat_product"] = df["value-mean"] * df["saturation-mean"]
    df["value_sat_ratio"] = np.where(
        df["saturation-mean"] > 1e-8,
        df["value-mean"] / (df["saturation-mean"] + 1e-8),
        0,
    )

    df["colorfulness"] = df["hue-mean"] * df["saturation-mean"]

    df["exred_exblue"] = df["exred-mean"] * df["exblue-mean"]
    df["exred_exgreen"] = df["exred-mean"] * df["exgreen-mean"]
    df["exblue_exgreen"] = df["exblue-mean"] * df["exgreen-mean"]

    df["excess_magnitude"] = np.sqrt(
        df["exred-mean"] ** 2 + df["exblue-mean"] ** 2 + df["exgreen-mean"] ** 2
    )

    # --- E. POLYNOMIAL FEATURES ---
    df["intensity_sq"] = df["intensity-mean"] ** 2
    df["intensity_cubed"] = df["intensity-mean"] ** 3
    df["saturation_sq"] = df["saturation-mean"] ** 2
    df["hue_sq"] = df["hue-mean"] ** 2

    # --- F. COMPOSITE REGION TYPE FEATURES ---
    df["grass_score"] = df["exgreen-mean"] * df["saturation-mean"]
    df["sky_score"] = df["exblue-mean"] * (1 / (df["edge_energy"] + 1))
    df["gray_region_score"] = (1 / (df["saturation-mean"] + 1)) * df["intensity-mean"]
    df["window_score"] = df["edge_energy"] * (1 / (df["saturation-mean"] + 1))
    df["foliage_score"] = df["exgreen-mean"] * df["texture_complexity"]
    df["brick_score"] = df["edge_energy"] * df["color_temperature"]

    return df


# Apply feature engineering
train_fe = engineer_features(train_df)
test_fe = engineer_features(test_df)

new_feature_cols = [
    c for c in train_fe.columns if c not in feature_cols and c != target_col
]
all_feature_cols = feature_cols + new_feature_cols

print(f"Total features after engineering: {len(all_feature_cols)}")

# ============================================================
# 3. TRAIN/VALIDATION SPLIT
# ============================================================
train_data = train_fe.drop(columns=[target_col])
train_target = train_fe[target_col]

X_train, X_val, y_train, y_val = train_test_split(
    train_data[all_feature_cols],
    train_target,
    test_size=0.2,
    random_state=42,
    stratify=train_target,
)

print(f"Split: Train={X_train.shape[0]}, Val={X_val.shape[0]}, Test={test_fe.shape[0]}")

# ============================================================
# 4. FEATURE SCALING (fit on train only)
# ============================================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(test_fe[all_feature_cols])

# ============================================================
# 5. TARGET ENCODING
# ============================================================
class_labels = sorted(train_target.unique())
class_to_idx = {c: i for i, c in enumerate(class_labels)}
n_classes = len(class_labels)

y_train_enc = y_train.map(class_to_idx).astype(int).values
y_val_enc = y_val.map(class_to_idx).astype(int).values

print(f"Classes: {n_classes} -> {class_labels}")

# ============================================================
# 6. IDENTIFY FEATURE GROUPS FOR DOMAIN-SPECIFIC EXPERTS
# ============================================================
color_features = [
    "rawred-mean",
    "rawgreen-mean",
    "rawblue-mean",
    "intensity-mean",
    "exred-mean",
    "exblue-mean",
    "exgreen-mean",
    "value-mean",
    "saturation-mean",
    "hue-mean",
    "chroma_r",
    "chroma_g",
    "chroma_b",
    "color_temperature",
    "rg_balance",
    "gb_balance",
    "rb_balance",
    "value_sat_product",
    "value_sat_ratio",
    "colorfulness",
    "exred_exblue",
    "exred_exgreen",
    "exblue_exgreen",
    "excess_magnitude",
    "intensity_sq",
    "intensity_cubed",
    "saturation_sq",
    "hue_sq",
    "grass_score",
    "sky_score",
    "gray_region_score",
    "window_score",
    "foliage_score",
    "brick_score",
]

edge_features = [
    "vedge-mean",
    "vedge-sd",
    "hedge-mean",
    "hedge-sd",
    "vedge_sharpness",
    "hedge_sharpness",
    "edge_energy",
    "edge_energy_sq",
    "edge_anisotropy",
    "edge_ratio",
    "texture_complexity",
    "texture_anisotropy",
    "color_edge_interaction_v",
    "color_edge_interaction_h",
    "sat_weighted_vedge",
    "sat_weighted_hedge",
    "hue_weighted_vedge",
    "hue_weighted_hedge",
]

line_features = [
    "short-line-density-5",
    "short-line-density-2",
    "line_density_total",
    "line_density_ratio",
]


def get_feature_indices(names, feature_names):
    indices = []
    for name in names:
        if name in feature_names:
            indices.append(feature_names.index(name))
    return indices


color_idx = get_feature_indices(color_features, all_feature_cols)
edge_idx = get_feature_indices(edge_features, all_feature_cols)
line_idx = get_feature_indices(line_features, all_feature_cols)

all_assigned = set(color_idx + edge_idx + line_idx)
general_idx = [i for i in range(len(all_feature_cols)) if i not in all_assigned]

print(
    f"Feature groups: Color={len(color_idx)}, Edge={len(edge_idx)}, Line={len(line_idx)}, General={len(general_idx)}"
)

# ============================================================
# 7. REORDER FEATURES TO MATCH MODEL INPUT ORDER
# ============================================================
combined_indices = []
for idx in [color_idx, edge_idx, line_idx, general_idx]:
    if len(idx) > 0:
        combined_indices.extend(idx)

feature_dims = [
    len(idx) for idx in [color_idx, edge_idx, line_idx, general_idx] if len(idx) > 0
]

X_train_arr = X_train_scaled[:, combined_indices]
X_val_arr = X_val_scaled[:, combined_indices]
X_test_arr = X_test_scaled[:, combined_indices]


# ============================================================
# 8. SETUP ENSEMBLE MODELS
# ============================================================
# Model 1: LightGBM (Config A - original)
lgb_model = lgb.LGBMClassifier(
    num_leaves=31,
    learning_rate=0.05,
    n_estimators=1000,
    random_state=42,
    verbose=-1,
)

# Model 2: LightGBM (Config B - higher leaves, lower LR, different feature fraction)
lgb_model2 = lgb.LGBMClassifier(
    num_leaves=63,
    learning_rate=0.02,
    n_estimators=2000,
    feature_fraction=0.7,
    bagging_fraction=0.8,
    bagging_freq=3,
    random_state=123,
    verbose=-1,
)

# Model 3: XGBoost (with early stopping)
xgb_model = xgb.XGBClassifier(
    max_depth=6,
    learning_rate=0.05,
    n_estimators=1000,
    random_state=42,
    eval_metric="mlogloss",
    early_stopping_rounds=50,
)

# Model 4: CatBoost (replaces NN - better for small tabular data)
cat_model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    random_seed=42,
    verbose=0,
    early_stopping_rounds=50,
)

print("Ensemble models: LightGBM-A, LightGBM-B, XGBoost, CatBoost")

# ============================================================
# 10. PREPARE ARRAYS FOR GRADIENT BOOSTING MODELS
# ============================================================
# Arrays already prepared as numpy: X_train_arr, X_val_arr, X_test_arr, y_train_enc

# ============================================================
# 11. TRAIN ENSEMBLE MODELS
# ============================================================
# --- Train LightGBM Config A ---
print("\n--- Training LightGBM-A ---")
lgb_model.fit(
    X_train_arr,
    y_train_enc,
    eval_set=[(X_val_arr, y_val_enc)],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
)
lgb_val_proba = lgb_model.predict_proba(X_val_arr)
lgb_test_proba = lgb_model.predict_proba(X_test_arr)
lgb_val_pred = lgb_model.predict(X_val_arr)
lgb_val_acc = accuracy_score(y_val_enc, lgb_val_pred)
print(f"LightGBM-A Val Accuracy: {lgb_val_acc:.4f} (best_iteration={lgb_model.best_iteration_})")

# --- Train LightGBM Config B ---
print("\n--- Training LightGBM-B ---")
lgb_model2.fit(
    X_train_arr,
    y_train_enc,
    eval_set=[(X_val_arr, y_val_enc)],
    callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)],
)
lgb2_val_proba = lgb_model2.predict_proba(X_val_arr)
lgb2_test_proba = lgb_model2.predict_proba(X_test_arr)
lgb2_val_pred = lgb_model2.predict(X_val_arr)
lgb2_val_acc = accuracy_score(y_val_enc, lgb2_val_pred)
print(f"LightGBM-B Val Accuracy: {lgb2_val_acc:.4f} (best_iteration={lgb_model2.best_iteration_})")

# --- Train XGBoost (with early stopping) ---
print("\n--- Training XGBoost ---")
xgb_model.fit(
    X_train_arr,
    y_train_enc,
    eval_set=[(X_val_arr, y_val_enc)],
    verbose=False,
)
xgb_val_proba = xgb_model.predict_proba(X_val_arr)
xgb_test_proba = xgb_model.predict_proba(X_test_arr)
xgb_val_pred = xgb_model.predict(X_val_arr)
xgb_val_acc = accuracy_score(y_val_enc, xgb_val_pred)
print(f"XGBoost Val Accuracy: {xgb_val_acc:.4f}")

# --- Train CatBoost ---
print("\n--- Training CatBoost ---")
cat_model.fit(
    X_train_arr,
    y_train_enc,
    eval_set=(X_val_arr, y_val_enc),
    verbose=50,
)
cat_val_proba = cat_model.predict_proba(X_val_arr)
cat_test_proba = cat_model.predict_proba(X_test_arr)
cat_val_pred = cat_model.predict(X_val_arr)
cat_val_acc = accuracy_score(y_val_enc, cat_val_pred)
print(f"CatBoost Val Accuracy: {cat_val_acc:.4f}")

# ============================================================
# 12. ENSEMBLE PREDICTIONS (Weighted + Stacking)
# ============================================================
# Collect all model probabilities
val_probas = [lgb_val_proba, lgb2_val_proba, xgb_val_proba, cat_val_proba]
test_probas = [lgb_test_proba, lgb2_test_proba, xgb_test_proba, cat_test_proba]
model_accs = [lgb_val_acc, lgb2_val_acc, xgb_val_acc, cat_val_acc]
model_names = ["LightGBM-A", "LightGBM-B", "XGBoost", "CatBoost"]

print(f"\nIndividual Model Accuracies:")
for name, acc in zip(model_names, model_accs):
    print(f"  {name}: {acc:.4f}")

# --- Method 1: Weighted Average (weights proportional to validation accuracy) ---
weights = np.array(model_accs)
weights = weights / weights.sum()
print(f"\nWeighted ensemble weights: {dict(zip(model_names, [f'{w:.4f}' for w in weights]))}")

val_proba_weighted = np.zeros_like(val_probas[0])
test_proba_weighted = np.zeros_like(test_probas[0])
for w, vp, tp in zip(weights, val_probas, test_probas):
    val_proba_weighted += w * vp
    test_proba_weighted += w * tp

val_pred_weighted_idx = val_proba_weighted.argmax(axis=1)
val_pred_weighted_labels = np.array(class_labels)[val_pred_weighted_idx]
val_acc_weighted = accuracy_score(y_val, val_pred_weighted_labels)
print(f"Weighted Ensemble Val Accuracy: {val_acc_weighted:.4f}")

# --- Method 2: Stacking with Logistic Regression meta-learner ---
# Build meta-features from out-of-fold (validation) predictions
val_meta_features = np.hstack([vp for vp in val_probas])  # (n_val, n_models * n_classes)
test_meta_features = np.hstack([tp for tp in test_probas])  # (n_test, n_models * n_classes)

# Train logistic regression meta-learner
meta_learner = LogisticRegression(
    C=1.0,
    max_iter=1000,
    multi_class='multinomial',
    random_state=42,
)
meta_learner.fit(val_meta_features, y_val_enc)

# Meta-learner predictions
val_pred_meta_idx = meta_learner.predict(val_meta_features)
val_pred_meta_labels = np.array(class_labels)[val_pred_meta_idx]
val_acc_meta = accuracy_score(y_val, val_pred_meta_labels)
print(f"Stacking (LogReg) Val Accuracy: {val_acc_meta:.4f}")

# Meta-learner test predictions
test_pred_meta_idx = meta_learner.predict(test_meta_features)
test_pred_meta_labels = np.array(class_labels)[test_pred_meta_idx]

# --- Choose best method ---
if val_acc_meta >= val_acc_weighted:
    print(f"\nUsing Stacking method (accuracy: {val_acc_meta:.4f})")
    val_accuracy = val_acc_meta
    val_pred_labels = val_pred_meta_labels
    test_pred_labels = test_pred_meta_labels
else:
    print(f"\nUsing Weighted Average method (accuracy: {val_acc_weighted:.4f})")
    val_accuracy = val_acc_weighted
    val_pred_labels = val_pred_weighted_labels
    test_pred_labels = np.array(class_labels)[test_proba_weighted.argmax(axis=1)]

print(f"\nFinal Ensemble Validation Accuracy: {val_accuracy:.4f}")
print(f"Class distribution in validation predictions:")
for c in class_labels:
    n_true = (y_val.values == c).sum()
    n_pred = (val_pred_labels == c).sum()
    print(f"  {c}: true={n_true}, pred={n_pred}")

# ============================================================
# 13. SAVE SUBMISSION
# ============================================================
os.makedirs("./submission", exist_ok=True)
test_ids = np.arange(len(test_pred_labels))
submission = pd.DataFrame({"id": test_ids, "class": test_pred_labels})
submission.to_csv("./submission/submission.csv", index=False)
print(f"\nSubmission saved: {submission.shape}")
print(submission.head(10))
print(f"\nPrediction class distribution:")
print(submission["class"].value_counts().to_string())

# ============================================================
# 14. FINAL VALIDATION SCORE
# ============================================================
score = val_accuracy
print(f"Final Validation Score: {score}")