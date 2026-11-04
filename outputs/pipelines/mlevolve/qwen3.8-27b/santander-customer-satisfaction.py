import numpy as np
import pandas as pd
import os
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import warnings

warnings.filterwarnings("ignore")

# Create directories
os.makedirs("./submission", exist_ok=True)
os.makedirs("./working", exist_ok=True)

# ============================================================
# 1. LOAD DATA
# ============================================================
print("Loading data...")
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")
print(f"Target distribution:\n{train_df['TARGET'].value_counts()}")

# Separate target
y = train_df["TARGET"].values
test_ids = test_df["ID"].values if "ID" in test_df.columns else np.arange(len(test_df))

# Remove ID and TARGET from feature frames
if "ID" in train_df.columns:
    train_ids = train_df["ID"].values
    train_df = train_df.drop(columns=["ID"])
else:
    train_ids = np.arange(len(train_df))

if "ID" in test_df.columns:
    test_df = test_df.drop(columns=["ID"])

if "TARGET" in train_df.columns:
    train_df = train_df.drop(columns=["TARGET"])

# ============================================================
# 2. IDENTIFY FEATURE GROUPS
# ============================================================
all_cols = list(train_df.columns)

var_cols = [c for c in all_cols if c.startswith("var") and "_" not in c]
imp_ent_cols = [c for c in all_cols if c.startswith("imp_ent_")]
imp_op_comer_ult1 = [c for c in all_cols if "comer" in c and c.endswith("_ult1")]
imp_op_comer_ult3 = [c for c in all_cols if "comer" in c and c.endswith("_ult3")]
imp_op_efect_ult1 = [c for c in all_cols if "efect" in c and c.endswith("_ult1")]
imp_op_efect_ult3 = [c for c in all_cols if "efect" in c and c.endswith("_ult3")]
imp_op_ult1 = [
    c
    for c in all_cols
    if c.startswith("imp_op_")
    and c.endswith("_ult1")
    and "comer" not in c
    and "efect" not in c
]
imp_op_ult3 = [
    c
    for c in all_cols
    if c.startswith("imp_op_")
    and c.endswith("_ult3")
    and "comer" not in c
    and "efect" not in c
]

print(
    f"\nFeature groups: var={len(var_cols)}, imp_ent={len(imp_ent_cols)}, "
    f"comer_ult1={len(imp_op_comer_ult1)}, comer_ult3={len(imp_op_comer_ult3)}, "
    f"efect_ult1={len(imp_op_efect_ult1)}, efect_ult3={len(imp_op_efect_ult3)}, "
    f"imp_op_ult1={len(imp_op_ult1)}, imp_op_ult3={len(imp_op_ult3)}"
)

# ============================================================
# 3. FEATURE ENGINEERING
# ============================================================
print("\nEngineering features...")


def engineer_features(df, is_train=False, train_stats=None):
    """Apply feature engineering to a dataframe."""
    df = df.copy()

    # --- 3a. Temporal trend features (ult1 vs ult3 pairs) ---
    comer_base = [
        c.replace("_ult1", "")
        for c in imp_op_comer_ult1
        if c.replace("_ult1", "") + "_ult3" in df.columns
    ]
    for base in comer_base:
        c1 = base + "_ult1"
        c3 = base + "_ult3"
        if c1 in df.columns and c3 in df.columns:
            df[f"{base}_diff"] = df[c1] - df[c3]
            df[f"{base}_ratio"] = df[c1] / (df[c3].abs() + 1e-6)
            df[f"{base}_sum"] = df[c1] + df[c3]

    efect_base = [
        c.replace("_ult1", "")
        for c in imp_op_efect_ult1
        if c.replace("_ult1", "") + "_ult3" in df.columns
    ]
    for base in efect_base:
        c1 = base + "_ult1"
        c3 = base + "_ult3"
        if c1 in df.columns and c3 in df.columns:
            df[f"{base}_diff"] = df[c1] - df[c3]
            df[f"{base}_ratio"] = df[c1] / (df[c3].abs() + 1e-6)
            df[f"{base}_sum"] = df[c1] + df[c3]

    imp_op_base = [
        c.replace("_ult1", "")
        for c in imp_op_ult1
        if c.replace("_ult1", "") + "_ult3" in df.columns
    ]
    for base in imp_op_base:
        c1 = base + "_ult1"
        c3 = base + "_ult3"
        if c1 in df.columns and c3 in df.columns:
            df[f"{base}_diff"] = df[c1] - df[c3]
            df[f"{base}_ratio"] = df[c1] / (df[c3].abs() + 1e-6)
            df[f"{base}_sum"] = df[c1] + df[c3]

    # --- 3a2. Segment-conditional temporal interaction features ---
    if "var3" in df.columns and "var15" in df.columns:
        # Interactions of temporal diff features with var3 and var15
        for base in comer_base:
            diff_col = f"{base}_diff"
            if diff_col in df.columns:
                df[f"{diff_col}_x_var3"] = df[diff_col] * df["var3"]
                df[f"{diff_col}_div_var3"] = df[diff_col] / (df["var3"].abs() + 1e-6)
                df[f"{diff_col}_x_var15"] = df[diff_col] * df["var15"]
                df[f"{diff_col}_div_var15"] = df[diff_col] / (df["var15"].abs() + 1e-6)
                df[f"{diff_col}_sq"] = df[diff_col] ** 2

        for base in efect_base:
            diff_col = f"{base}_diff"
            if diff_col in df.columns:
                df[f"{diff_col}_x_var3"] = df[diff_col] * df["var3"]
                df[f"{diff_col}_div_var3"] = df[diff_col] / (df["var3"].abs() + 1e-6)
                df[f"{diff_col}_x_var15"] = df[diff_col] * df["var15"]
                df[f"{diff_col}_div_var15"] = df[diff_col] / (df["var15"].abs() + 1e-6)
                df[f"{diff_col}_sq"] = df[diff_col] ** 2

        for base in imp_op_base:
            diff_col = f"{base}_diff"
            if diff_col in df.columns:
                df[f"{diff_col}_x_var3"] = df[diff_col] * df["var3"]
                df[f"{diff_col}_div_var3"] = df[diff_col] / (df["var3"].abs() + 1e-6)
                df[f"{diff_col}_x_var15"] = df[diff_col] * df["var15"]
                df[f"{diff_col}_div_var15"] = df[diff_col] / (df["var15"].abs() + 1e-6)
                df[f"{diff_col}_sq"] = df[diff_col] ** 2

    # --- 3b. Group-level aggregations ---
    if len(imp_ent_cols) > 0:
        ent_vals = df[imp_ent_cols].values
        df["imp_ent_mean"] = np.nanmean(ent_vals, axis=1)
        df["imp_ent_std"] = np.nanstd(ent_vals, axis=1)
        df["imp_ent_max"] = np.nanmax(ent_vals, axis=1)
        df["imp_ent_min"] = np.nanmin(ent_vals, axis=1)
        df["imp_ent_nonzero"] = np.nansum(np.abs(ent_vals) > 1e-6, axis=1)
        df["imp_ent_sum"] = np.nansum(ent_vals, axis=1)

    if len(var_cols) > 0:
        var_vals = df[var_cols].values
        df["var_mean"] = np.nanmean(var_vals, axis=1)
        df["var_std"] = np.nanstd(var_vals, axis=1)
        df["var_max"] = np.nanmax(var_vals, axis=1)
        df["var_min"] = np.nanmin(var_vals, axis=1)
        df["var_nonzero"] = np.nansum(np.abs(var_vals) > 1e-6, axis=1)
        df["var_sum"] = np.nansum(var_vals, axis=1)

    comer_all = [c for c in df.columns if "comer" in c]
    if len(comer_all) > 0:
        comer_vals = df[comer_all].values
        df["comer_mean"] = np.nanmean(comer_vals, axis=1)
        df["comer_std"] = np.nanstd(comer_vals, axis=1)
        df["comer_max"] = np.nanmax(comer_vals, axis=1)
        df["comer_nonzero"] = np.nansum(np.abs(comer_vals) > 1e-6, axis=1)

    efect_all = [c for c in df.columns if "efect" in c]
    if len(efect_all) > 0:
        efect_vals = df[efect_all].values
        df["efect_mean"] = np.nanmean(efect_vals, axis=1)
        df["efect_std"] = np.nanstd(efect_vals, axis=1)
        df["efect_max"] = np.nanmax(efect_vals, axis=1)
        df["efect_nonzero"] = np.nansum(np.abs(efect_vals) > 1e-6, axis=1)

    # --- 3c. Key interactions ---
    if "var3" in df.columns and "var15" in df.columns:
        df["var3_x_var15"] = df["var3"] * df["var15"]
        df["var3_div_var15"] = df["var3"] / (df["var15"].abs() + 1e-6)
        df["var15_div_var3"] = df["var15"] / (df["var3"].abs() + 1e-6)
        df["var3_plus_var15"] = df["var3"] + df["var15"]
        df["var3_minus_var15"] = df["var3"] - df["var15"]

    if "var3" in df.columns and "var27" in df.columns:
        df["var3_x_var27"] = df["var3"] * df["var27"]
        df["var3_div_var27"] = df["var3"] / (df["var27"].abs() + 1e-6)

    if "var15" in df.columns and "var27" in df.columns:
        df["var15_x_var27"] = df["var15"] * df["var27"]

    if "var3" in df.columns and "var32" in df.columns:
        df["var3_x_var32"] = df["var3"] * df["var32"]

    if "var3" in df.columns and "var53" in df.columns:
        df["var3_x_var53"] = df["var3"] * df["var53"]

    if "var15" in df.columns and "var53" in df.columns:
        df["var15_x_var53"] = df["var15"] * df["var53"]

    # --- 3d. Missing value features ---
    df["n_missing"] = df.isna().sum(axis=1)

    if len(imp_ent_cols) > 0:
        df["n_missing_ent"] = df[imp_ent_cols].isna().sum(axis=1)
    if len(var_cols) > 0:
        df["n_missing_var"] = df[var_cols].isna().sum(axis=1)

    # --- 3e. Log transforms for skewed features ---
    for col in [
        "var3",
        "var15",
        "var27",
        "var32",
        "var53",
        "var58",
        "var61",
        "var68",
        "var75",
    ]:
        if col in df.columns:
            df[f"{col}_log"] = np.log1p(df[col].clip(lower=0))

    # --- 3f. Cross-group ratio features ---
    if "var_mean" in df.columns and "imp_ent_mean" in df.columns:
        df["var_to_ent_ratio"] = df["var_mean"] / (df["imp_ent_mean"].abs() + 1e-6)

    if "comer_mean" in df.columns and "efect_mean" in df.columns:
        df["comer_to_efect_ratio"] = df["comer_mean"] / (df["efect_mean"].abs() + 1e-6)

    # --- 3g. Activity intensity features ---
    if len(imp_ent_cols) > 0:
        ent_ult1 = [c for c in imp_ent_cols if c.endswith("_ult1")]
        if len(ent_ult1) > 0:
            df["ent_ult1_mean"] = df[ent_ult1].mean(axis=1)

    # --- 3h. Coefficient of variation ---
    if "var_std" in df.columns and "var_mean" in df.columns:
        df["var_cv"] = df["var_std"] / (df["var_mean"].abs() + 1e-6)
    if "imp_ent_std" in df.columns and "imp_ent_mean" in df.columns:
        df["ent_cv"] = df["imp_ent_std"] / (df["imp_ent_mean"].abs() + 1e-6)

    return df


# Apply feature engineering
train_feat = engineer_features(train_df, is_train=True)
test_feat = engineer_features(test_df, is_train=False)

print(f"Features after engineering: {train_feat.shape[1]}")

# ============================================================
# 4. CREATE TRAIN/VALIDATION SPLIT
# ============================================================
print("\nCreating train/val split...")
X_train, X_val, y_train, y_val = train_test_split(
    train_feat, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {test_feat.shape}")
print(f"Val target distribution: {pd.Series(y_val).value_counts().to_dict()}")

# ============================================================
# 5. DEFINE FEATURE SUBSETS FOR DUAL-MODEL ARCHITECTURE
# ============================================================
all_features = X_train.columns.tolist()

temporal_features = [
    c
    for c in all_features
    if any(c.endswith(suffix) for suffix in ["_diff", "_ratio", "_ult1", "_log", "_sq"])
    or c
    in ["var3", "var15", "var27", "var32", "var53", "var58", "var61", "var68", "var75"]
    or "var3_x_" in c
    or "var15_x_" in c
    or "var3_div_" in c
    or "var15_div_" in c
    or "var3_plus_" in c
    or "var3_minus_" in c
    or "_diff_x_var3" in c
    or "_diff_div_var3" in c
    or "_diff_x_var15" in c
    or "_diff_div_var15" in c
]

structural_features = [
    c
    for c in all_features
    if any(
        c.endswith(suffix)
        for suffix in [
            "_ult3",
            "_mean",
            "_std",
            "_max",
            "_min",
            "_sum",
            "_nonzero",
            "_cv",
        ]
    )
    or c.startswith("n_missing")
    or "var_to_ent_ratio" in c
    or "comer_to_efect_ratio" in c
    or "ent_ult1_mean" in c
]

# Remove overlap
overlap = set(temporal_features) & set(structural_features)
structural_features = [c for c in structural_features if c not in overlap]

# Remaining features go to structural
remaining = set(all_features) - set(temporal_features) - set(structural_features)
structural_features = structural_features + list(remaining)

print(f"Temporal features: {len(temporal_features)}")
print(f"Structural features: {len(structural_features)}")
print(
    f"Total: {len(temporal_features) + len(structural_features)} (should be {len(all_features)})"
)

# ============================================================
# 6. DEFINE MODELS
# ============================================================
# Model 1: Temporal Dynamics - deeper trees, lower regularization
model_temporal = lgb.LGBMClassifier(
    n_estimators=2000,
    learning_rate=0.03,
    max_depth=8,
    num_leaves=63,
    min_child_samples=15,
    subsample=0.75,
    subsample_freq=1,
    colsample_bytree=0.6,
    reg_alpha=0.5,
    reg_lambda=1.0,
    min_split_gain=0.01,
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)

# Model 2: Structural Pattern - shallower trees, stronger regularization
model_structural = lgb.LGBMClassifier(
    n_estimators=2000,
    learning_rate=0.02,
    max_depth=5,
    num_leaves=23,
    min_child_samples=30,
    subsample=0.85,
    subsample_freq=1,
    colsample_bytree=0.7,
    reg_alpha=1.0,
    reg_lambda=2.0,
    min_split_gain=0.02,
    random_state=123,
    n_jobs=-1,
    verbose=-1,
)

# Model 3: XGBoost - architecture diversity, trained on ALL features
model_xgb = xgb.XGBClassifier(
    n_estimators=3000,
    max_depth=4,
    min_child_weight=50,
    subsample=0.8,
    colsample_bytree=0.5,
    reg_alpha=2.0,
    reg_lambda=5.0,
    learning_rate=0.01,
    random_state=42,
    n_jobs=-1,
    use_label_encoder=False,
    eval_metric="auc",
    early_stopping_rounds=100,
)

# ============================================================
# 7. TRAINING WITH EARLY STOPPING
# ============================================================
print("\n" + "=" * 60)
print("TRAINING TEMPORAL DYNAMICS MODEL")
print("=" * 60)

model_temporal.fit(
    X_train[temporal_features],
    y_train,
    eval_set=[(X_val[temporal_features], y_val)],
    eval_metric="auc",
    callbacks=[lgb.early_stopping(100, verbose=False)],
)
best_iter_temporal = model_temporal.best_iteration_
print(f"  Best iteration: {best_iter_temporal}")

print("\n" + "=" * 60)
print("TRAINING STRUCTURAL PATTERN MODEL")
print("=" * 60)

model_structural.fit(
    X_train[structural_features],
    y_train,
    eval_set=[(X_val[structural_features], y_val)],
    eval_metric="auc",
    callbacks=[lgb.early_stopping(100, verbose=False)],
)
best_iter_structural = model_structural.best_iteration_
print(f"  Best iteration: {best_iter_structural}")

print("\n" + "=" * 60)
print("TRAINING XGBOOST MODEL (ALL FEATURES)")
print("=" * 60)

model_xgb.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    verbose=False,
)
best_iter_xgb = model_xgb.best_iteration
print(f"  Best iteration: {best_iter_xgb}")

# ============================================================
# 8. VALIDATION PREDICTIONS
# ============================================================
val_pred_temporal = model_temporal.predict_proba(X_val[temporal_features])[:, 1]
val_pred_structural = model_structural.predict_proba(X_val[structural_features])[:, 1]
val_pred_xgb = model_xgb.predict_proba(X_val)[:, 1]

auc_temporal = roc_auc_score(y_val, val_pred_temporal)
auc_structural = roc_auc_score(y_val, val_pred_structural)
auc_xgb = roc_auc_score(y_val, val_pred_xgb)
print(f"\nIndividual Validation AUCs:")
print(f"  Temporal:   {auc_temporal:.6f}")
print(f"  Structural: {auc_structural:.6f}")
print(f"  XGBoost:    {auc_xgb:.6f}")

# ============================================================
# 9. BLEND WEIGHT OPTIMIZATION (3-way coarse-to-fine)
# ============================================================
print("\nOptimizing 3-way blend weights (w1+w2+w3=1)...")


def blend_3way(w1, w2, w3, p1, p2, p3):
    return w1 * p1 + w2 * p2 + w3 * p3


# Coarse search over simplex
best_w1, best_w2, best_w3 = 0.5, 0.3, 0.2
best_auc = 0.0
step = 0.1
for w1 in np.arange(0.0, 1.0 + step, step):
    for w2 in np.arange(0.0, 1.0 - w1 + step, step):
        w3 = 1.0 - w1 - w2
        if w3 < -1e-9:
            continue
        w3 = max(0.0, w3)
        blend = blend_3way(w1, w2, w3, val_pred_temporal, val_pred_structural, val_pred_xgb)
        auc = roc_auc_score(y_val, blend)
        if auc > best_auc:
            best_auc = auc
            best_w1, best_w2, best_w3 = w1, w2, w3

# Fine search around best
fine_step = 0.02
for w1 in np.arange(max(0.0, best_w1 - 0.1), min(1.0, best_w1 + 0.1) + fine_step, fine_step):
    for w2 in np.arange(max(0.0, best_w2 - 0.1), min(1.0 - w1, best_w2 + 0.1) + fine_step, fine_step):
        w3 = 1.0 - w1 - w2
        if w3 < -1e-9:
            continue
        w3 = max(0.0, w3)
        blend = blend_3way(w1, w2, w3, val_pred_temporal, val_pred_structural, val_pred_xgb)
        auc = roc_auc_score(y_val, blend)
        if auc > best_auc:
            best_auc = auc
            best_w1, best_w2, best_w3 = w1, w2, w3

print(f"  Optimal weights: temporal={best_w1:.3f}, structural={best_w2:.3f}, xgb={best_w3:.3f}")
print(f"  Blended Validation AUC:    {best_auc:.6f}")

# ============================================================
# 10. TEST PREDICTIONS
# ============================================================
test_pred_temporal = model_temporal.predict_proba(test_feat[temporal_features])[:, 1]
test_pred_structural = model_structural.predict_proba(test_feat[structural_features])[:, 1]
test_pred_xgb = model_xgb.predict_proba(test_feat)[:, 1]
test_pred = blend_3way(best_w1, best_w2, best_w3, test_pred_temporal, test_pred_structural, test_pred_xgb)

# ============================================================
# 11. SAVE SUBMISSION
# ============================================================
submission = pd.DataFrame({"id": range(len(test_pred)), "TARGET": test_pred})
submission.to_csv("./submission/submission.csv", index=False)
print(f"\nSubmission saved: {submission.shape}")
print(f"  Prediction range: [{test_pred.min():.6f}, {test_pred.max():.6f}]")
print(f"  Prediction mean:  {test_pred.mean():.6f}")

# Save models
model_temporal.booster_.save_model("./working/model_temporal.txt")
model_structural.booster_.save_model("./working/model_structural.txt")
model_xgb.save_model("./working/model_xgb.json")
print("Models saved to ./working/")

# ============================================================
# 12. FINAL SUMMARY
# ============================================================
print(f"\n--- SUMMARY ---")
print(f"Split: 80/20 stratified (random_state=42)")
print(f"Metric: ROC-AUC (official)")
print(f"Temporal AUC:   {auc_temporal:.6f}")
print(f"Structural AUC: {auc_structural:.6f}")
print(f"XGBoost AUC:    {auc_xgb:.6f}")
print(f"Blend weights:  temporal={best_w1:.3f}, structural={best_w2:.3f}, xgb={best_w3:.3f}")
print(f"Blended AUC:    {best_auc:.6f}")

score = best_auc
print(f"Final Validation Score: {score}")