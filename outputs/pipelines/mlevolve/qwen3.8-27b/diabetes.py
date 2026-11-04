import pandas as pd
import numpy as np
import lightgbm as lgb
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# Create output directories
os.makedirs("./submission", exist_ok=True)
os.makedirs("./working", exist_ok=True)

# ============================================================
# 1. LOAD DATA
# ============================================================
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")
print(f"Train columns: {list(train_df.columns)}")
print(f"Target distribution:\n{train_df['class'].value_counts()}")

# ============================================================
# 2. IDENTIFY FEATURES AND TARGET
# ============================================================
target_col = "class"
feature_cols = [c for c in train_df.columns if c != target_col]
print(f"Feature columns: {feature_cols}")

# ============================================================
# 3. DATA CLEANING - Replace physiologically impossible zeros
# ============================================================
zero_as_missing_cols = [
    "plas",
    "pres",
    "skin",
    "insu",
    "mass",
]

# Fit imputation statistics on TRAINING data only (prevent leakage)
impute_stats = {}
for col in zero_as_missing_cols:
    if col in train_df.columns:
        non_zero = train_df.loc[train_df[col] != 0, col]
        impute_stats[col] = (
            non_zero.median() if len(non_zero) > 0 else train_df[col].median()
        )
        print(
            f"Imputing zeros in '{col[:40]}...' with median={impute_stats[col]:.2f} "
            f"(n_zeros={int((train_df[col] == 0).sum())})"
        )


def clean_zeros(df, stats):
    """Replace zeros with imputed values using training statistics."""
    df = df.copy()
    for col, val in stats.items():
        if col in df.columns:
            df[col] = df[col].replace(0, val)
    return df


train_clean = clean_zeros(train_df, impute_stats)
test_clean = clean_zeros(test_df, impute_stats)


# ============================================================
# 4. FEATURE ENGINEERING
# ============================================================
def engineer_features(df):
    """Create domain-informed features from raw features."""
    df = df.copy()

    glucose = df["plas"]
    bp = df["pres"]
    skin = df["skin"]
    insulin = df["insu"]
    bmi = df["mass"]
    pedigree = df["pedi"]
    age = df["age"]
    pregnancies = df["preg"]

    # Physiological interaction features
    df["glucose_bmi"] = glucose * bmi
    df["insulin_glucose_ratio"] = insulin / (glucose + 1)
    df["glucose_insulin"] = glucose * insulin
    df["bmi_age"] = bmi * age
    df["pedigree_age"] = pedigree * age
    df["pregnancies_age"] = pregnancies * age
    df["glucose_per_bmi"] = glucose / (bmi + 1)
    df["insulin_bmi"] = insulin * bmi
    df["bp_bmi"] = bp * bmi
    df["pedigree_glucose"] = pedigree * glucose
    df["skin_bmi"] = skin * bmi

    # Non-linear transforms
    df["log_glucose"] = np.log1p(glucose)
    df["log_insulin"] = np.log1p(insulin)
    df["log_bmi"] = np.log1p(bmi)
    df["log_age"] = np.log1p(age)

    # Squared terms
    df["glucose_sq"] = glucose**2
    df["bmi_sq"] = bmi**2
    df["age_sq"] = age**2

    # Ratio features
    df["glucose_bp_ratio"] = glucose / (bp + 1)
    df["insulin_skin_ratio"] = insulin / (skin + 1)
    df["age_adjusted_glucose"] = glucose / (age + 1)

    # Composite risk score
    df["risk_score"] = (
        glucose * 0.3
        + bmi * 0.2
        + pedigree * 0.2
        + age * 0.15
        + insulin * 0.1
        + pregnancies * 0.05
    )

    return df


train_feat = engineer_features(train_clean)
test_feat = engineer_features(test_clean)

final_feature_cols = [c for c in train_feat.columns if c != target_col]
print(f"\nTotal features after engineering: {len(final_feature_cols)}")

# ============================================================
# 5. CREATE TRAIN/VALIDATION SPLIT (Stratified)
# ============================================================
# Encode target: 'tested_negative' -> 0, 'tested_positive' -> 1
target_encoding = {"tested_negative": 0, "tested_positive": 1}
y_train_full = train_feat[target_col].map(target_encoding).astype(int).values
X_train_full = train_feat[final_feature_cols].values

X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
)

X_test = test_feat[final_feature_cols].values

print(f"\nFinal splits:")
print(f"  Training: {X_tr.shape}")
print(f"  Validation: {X_val.shape}")
print(f"  Test: {X_test.shape}")
print(f"  Train class balance: {np.bincount(y_tr.astype(int))}")
print(f"  Val class balance: {np.bincount(y_val.astype(int))}")

# ============================================================
# 6. MODEL DESIGN: LightGBM with conservative regularization
# ============================================================
n_pos = y_tr.sum()
n_neg = len(y_tr) - n_pos
scale_pos_weight = n_neg / n_pos
print(
    f"Class balance: {n_pos} pos, {n_neg} neg, scale_pos_weight={scale_pos_weight:.3f}"
)

lgb_params = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "num_leaves": 15,
    "max_depth": 4,
    "min_child_samples": 20,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.7,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "learning_rate": 0.05,
    "n_estimators": 500,
    "scale_pos_weight": scale_pos_weight,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}

# ============================================================
# 7. TRAINING WITH EARLY STOPPING
# ============================================================
train_data = lgb.Dataset(X_tr, label=y_tr, feature_name=final_feature_cols)
val_data = lgb.Dataset(
    X_val, label=y_val, feature_name=final_feature_cols, reference=train_data
)

callbacks = [
    lgb.early_stopping(stopping_rounds=50, verbose=False),
    lgb.log_evaluation(period=-1),
]

model = lgb.train(
    lgb_params,
    train_data,
    valid_sets=[train_data, val_data],
    valid_names=["train", "valid"],
    callbacks=callbacks,
)

best_iteration = model.best_iteration
best_val_auc = model.best_score["valid"]["auc"]
best_train_auc = model.best_score["train"]["auc"]

print(f"Best iteration: {best_iteration}")
print(f"Train AUC at best: {best_train_auc:.4f}")
print(f"Val AUC at best: {best_val_auc:.4f}")
print(f"Overfitting gap: {best_train_auc - best_val_auc:.4f}")

# Save model
model.save_model("./working/best_lgb_model.txt")

# ============================================================
# 8. PREDICTIONS AND EVALUATION
# ============================================================
val_pred_proba = model.predict(X_val, num_iteration=best_iteration)
test_pred_proba = model.predict(X_test, num_iteration=best_iteration)

val_auc = roc_auc_score(y_val, val_pred_proba)
print(f"Validation ROC AUC (official metric): {val_auc:.4f}")

# Feature importance
importance = model.feature_importance(importance_type="gain")
feat_imp = sorted(zip(final_feature_cols, importance), key=lambda x: x[1], reverse=True)
print("\nTop 10 features by gain:")
for name, imp in feat_imp[:10]:
    print(f"  {name}: {imp:.1f}")

# ============================================================
# 9. GENERATE SUBMISSION
# ============================================================
submission = pd.DataFrame(
    {"id": np.arange(len(test_pred_proba)), "class": test_pred_proba}
)
submission.to_csv("./submission/submission.csv", index=False)
print(f"\nSubmission saved: {submission.shape}")
print(f"Prediction range: [{test_pred_proba.min():.4f}, {test_pred_proba.max():.4f}]")
print(f"Prediction mean: {test_pred_proba.mean():.4f}")

# ============================================================
# FINAL SCORE
# ============================================================
score = val_auc
print(f"Final Validation Score: {score}")
