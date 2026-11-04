import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import os

# Create directories
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# ============================================================
# 1. LOAD DATA
# ============================================================
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")
print(f"Target distribution:\n{train_df['Class'].value_counts()}")

# ============================================================
# 2. CLEANING & COLUMN MAPPING
# ============================================================
target_col = "Class"
# Convert Class (1, 2) to binary (0, 1): Class 1 = non-patient (0), Class 2 = patient (1)
y_train_full = (train_df[target_col].astype(int) - 1).clip(0, 1).astype(int)
train_features = train_df.drop(columns=[target_col]).copy()
test_features = test_df.copy()

# Map actual column names to canonical names
# The UCI dataset may use V1-V10 or other naming conventions
canonical_names = [
    "Age", "Gender", "Total Bilirubin", "Direct Bilirubin", "Alkphos",
    "SGPT", "SGOT", "Total Proteins", "Albumin", "A/G Ratio"
]

def map_columns(df):
    """Map columns to canonical names based on position or existing names."""
    df = df.copy()
    current_cols = list(df.columns)

    # Check if columns already match canonical names (case-insensitive)
    col_lower = [c.lower().replace(" ", "") for c in current_cols]
    canonical_lower = [c.lower().replace(" ", "") for c in canonical_names]

    if all(c in col_lower for c in canonical_lower):
        # Already canonical - just rename to exact canonical
        mapping = {}
        for i, c in enumerate(current_cols):
            cl = c.lower().replace(" ", "")
            for j, cn in enumerate(canonical_names):
                if cl == cn.lower().replace(" ", ""):
                    mapping[c] = canonical_names[j]
                    break
        if len(mapping) == len(current_cols):
            df.rename(columns=mapping, inplace=True)
            return df

    # Fallback: assume positional mapping (V1=Age, V2=Gender, ...)
    # If there are exactly 10 feature columns, map by position
    if len(current_cols) == 10:
        df.columns = canonical_names
    else:
        # Try to match by checking if any column names contain keywords
        print(f"WARNING: Unexpected columns: {current_cols}")
        # Map first 10 columns positionally
        df.columns = canonical_names[:len(current_cols)]

    return df

train_features = map_columns(train_features)
test_features = map_columns(test_features)

print(f"Feature columns: {list(train_features.columns)}")
print(f"Test columns: {list(test_features.columns)}")

# ============================================================
# 3. FEATURE ENGINEERING
# ============================================================
def engineer_features(df):
    """Apply domain-informed feature engineering to a dataframe."""
    df = df.copy()

    # --- Age features ---
    df["age_is_capped"] = (df["Age"] == 90).astype(int)
    df["age_squared"] = df["Age"] ** 2
    df["age_log"] = np.log1p(df["Age"])

    # --- Gender encoding ---
    if df["Gender"].dtype == object or df["Gender"].dtype.name == "category":
        # Could be "M"/"F" or "Male"/"Female"
        male_mask = df["Gender"].astype(str).str.lower().str.startswith("m")
        df["gender_encoded"] = male_mask.astype(int)
    else:
        # Numeric: assume 1=male, 0=female or similar
        df["gender_encoded"] = (df["Gender"] == df["Gender"].max()).astype(int)

    # --- Bilirubin features ---
    df["bilirubin_ratio"] = df["Direct Bilirubin"] / (df["Total Bilirubin"] + 1e-8)
    df["indirect_bilirubin"] = df["Total Bilirubin"] - df["Direct Bilirubin"]
    df["bilirubin_diff"] = df["Total Bilirubin"] - df["Direct Bilirubin"]
    df["bilirubin_product"] = df["Total Bilirubin"] * df["Direct Bilirubin"]

    # --- Protein features ---
    df["globulin"] = df["Total Proteins"] - df["Albumin"]
    df["ag_ratio_derived"] = df["Albumin"] / (df["globulin"] + 1e-8)
    df["ag_ratio_diff"] = df["A/G Ratio"] - df["ag_ratio_derived"]
    df["albumin_ratio"] = df["Albumin"] / (df["Total Proteins"] + 1e-8)

    # --- Enzyme features ---
    df["sgpt_sgot_ratio"] = df["SGPT"] / (df["SGOT"] + 1e-8)
    df["sgpt_sgot_diff"] = df["SGPT"] - df["SGOT"]
    df["sgpt_sgot_product"] = df["SGPT"] * df["SGOT"]
    df["sgpt_log"] = np.log1p(df["SGPT"])
    df["sgot_log"] = np.log1p(df["SGOT"])
    df["sgpt_squared"] = df["SGPT"] ** 2
    df["sgot_squared"] = df["SGOT"] ** 2

    # --- Alkaline Phosphatase interactions ---
    df["alkphos_log"] = np.log1p(df["Alkphos"])
    df["alkphos_bilirubin"] = df["Alkphos"] * df["Total Bilirubin"]
    df["alkphos_sgot"] = df["Alkphos"] * df["SGOT"]

    # --- Cross-marker interactions ---
    df["alkphos_bilirubin_ratio"] = df["Alkphos"] / (df["Total Bilirubin"] + 1e-8)
    df["bilirubin_sgot"] = df["Total Bilirubin"] * df["SGOT"]
    df["bilirubin_sgpt"] = df["Total Bilirubin"] * df["SGPT"]

    # --- Composite liver damage score ---
    df["bilirubin_rank"] = df["Total Bilirubin"].rank(pct=True)
    df["sgpt_rank"] = df["SGPT"].rank(pct=True)
    df["sgot_rank"] = df["SGOT"].rank(pct=True)
    df["alkphos_rank"] = df["Alkphos"].rank(pct=True)
    df["liver_damage_score"] = (
        df["bilirubin_rank"] + df["sgpt_rank"] + df["sgot_rank"] + df["alkphos_rank"]
    ) / 4.0

    # --- Protein-enzyme interactions ---
    df["albumin_sgpt"] = df["Albumin"] * df["SGPT"]
    df["total_prot_sgpt"] = df["Total Proteins"] * df["SGPT"]
    df["total_prot_bilirubin"] = df["Total Proteins"] * df["Total Bilirubin"]

    # --- Age interactions with key markers ---
    df["age_bilirubin"] = df["Age"] * df["Total Bilirubin"]
    df["age_sgpt"] = df["Age"] * df["SGPT"]
    df["age_alkphos"] = df["Age"] * df["Alkphos"]

    # --- Gender interactions ---
    df["gender_bilirubin"] = df["gender_encoded"] * df["Total Bilirubin"]
    df["gender_sgpt"] = df["gender_encoded"] * df["SGPT"]
    df["gender_alkphos"] = df["gender_encoded"] * df["Alkphos"]

    # --- Log transforms for skewed features ---
    df["total_bilirubin_log"] = np.log1p(df["Total Bilirubin"])
    df["direct_bilirubin_log"] = np.log1p(df["Direct Bilirubin"])
    df["total_proteins_log"] = np.log1p(df["Total Proteins"])
    df["albumin_log"] = np.log1p(df["Albumin"])

    # --- Remove original Gender (replaced by encoded) ---
    if "Gender" in df.columns:
        df = df.drop(columns=["Gender"])

    return df


# ============================================================
# 4. TRAIN/VALIDATION SPLIT (Stratified) - BEFORE feature engineering
# ============================================================
X_train_raw, X_val_raw, y_train, y_val = train_test_split(
    train_features, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
)

print(f"Train split: {X_train_raw.shape}, Val split: {X_val_raw.shape}")
print(f"Val class distribution:\n{y_val.value_counts()}")

# Fill missing values with median (fit on train split only)
missing_cols = X_train_raw.columns[X_train_raw.isnull().any()].tolist()
if missing_cols:
    medians = X_train_raw[missing_cols].median()
    X_train_raw[missing_cols] = X_train_raw[missing_cols].fillna(medians)
    X_val_raw[missing_cols] = X_val_raw[missing_cols].fillna(medians)
    test_features[missing_cols] = test_features[missing_cols].fillna(medians)
    print(f"Filled missing values in: {missing_cols}")

# Apply feature engineering AFTER split to avoid rank/encoding leakage
X_train = engineer_features(X_train_raw)
X_val = engineer_features(X_val_raw)
test_features = engineer_features(test_features)

print(f"After feature engineering - Train shape: {X_train.shape}")
print(f"After feature engineering - Val shape: {X_val.shape}")
print(f"After feature engineering - Test shape: {test_features.shape}")

# ============================================================
# 5. MODEL TRAINING: LightGBM Binary Classifier
# ============================================================
# Class imbalance handling
n_pos = int((y_train == 1).sum())
n_neg = int((y_train == 0).sum())
scale_pos_weight = n_neg / n_pos
print(
    f"Class balance: pos={n_pos}, neg={n_neg}, scale_pos_weight={scale_pos_weight:.3f}"
)

# LightGBM parameters - tuned for small dataset with strong regularization
lgb_params = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 15,
    "max_depth": 4,
    "min_data_in_leaf": 15,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 3,
    "scale_pos_weight": scale_pos_weight,
    "lambda_l1": 1.0,
    "lambda_l2": 5.0,
    "min_split_gain": 0.1,
    "verbose": -1,
    "seed": 42,
    "n_jobs": -1,
}

# Create LightGBM datasets
train_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

# Train with early stopping
callbacks = [
    lgb.early_stopping(stopping_rounds=50, verbose=False),
    lgb.log_evaluation(period=-1),
]

model = lgb.train(
    lgb_params,
    train_data,
    num_boost_round=1000,
    valid_sets=[train_data, val_data],
    valid_names=["train", "valid"],
    callbacks=callbacks,
)

best_iter = model.best_iteration
best_val_auc = model.best_score["valid"]["auc"]
print(f"Best iteration: {best_iter}")
print(f"Best validation AUC (from training): {best_val_auc:.4f}")

# ============================================================
# 6. VALIDATION: Compute ROC-AUC on held-out val set
# ============================================================
val_probs = model.predict(X_val, num_iteration=best_iter)
val_auc = roc_auc_score(y_val, val_probs)
print(f"Validation ROC-AUC (computed): {val_auc:.4f}")

# ============================================================
# 7. TEST INFERENCE
# ============================================================
test_probs = model.predict(test_features, num_iteration=best_iter)

# ============================================================
# 8. SAVE SUBMISSION
# ============================================================
# The target is binary classification (Class: 1 or 2)
# For roc_auc metric, we output probabilities of the positive class (Class=2)
submission = pd.DataFrame({"id": range(len(test_probs)), "Class": test_probs})
submission.to_csv("./submission/submission.csv", index=False)
print(f"Submission saved: {submission.shape}")
print(
    f"Test prediction stats: mean={test_probs.mean():.4f}, min={test_probs.min():.4f}, max={test_probs.max():.4f}"
)

# ============================================================
# 9. FEATURE IMPORTANCE (audit)
# ============================================================
importance = model.feature_importance(importance_type="gain")
feat_imp = pd.Series(importance, index=X_train.columns).sort_values(ascending=False)
print(f"\nTop 10 features by gain:")
print(feat_imp.head(10).to_string())

# ============================================================
# FINAL SCORE
# ============================================================
score = val_auc
print(f"Final Validation Score: {score}")