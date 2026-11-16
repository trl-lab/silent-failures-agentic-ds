import pandas as pd
import numpy as np
import os
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ============================================================
# 1. LOAD DATA
# ============================================================
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")
print(f"Train columns: {list(train_df.columns)}")
print(f"Target distribution:\n{train_df['car'].value_counts()}")

# ============================================================
# 2. DEFINE ORDINAL MAPPINGS (domain-informed)
# ============================================================
ordinal_mappings = {
    "buying": {"vhigh": 3, "high": 2, "med": 1, "low": 0},
    "maint": {"vhigh": 3, "high": 2, "med": 1, "low": 0},
    "doors": {"2": 0, "3": 1, "4": 2, "5more": 3},
    "persons": {"2": 0, "4": 1, "more": 2},
    "lug_boot": {"small": 0, "med": 1, "big": 2},
    "safety": {"low": 0, "med": 1, "high": 2},
}

target_mapping = {
    "unacc": 0,
    "acc": 1,
    "good": 2,
    "vgood": 3,
}

inverse_target_mapping = {v: k for k, v in target_mapping.items()}

feature_cols = ["buying", "maint", "doors", "persons", "lug_boot", "safety"]

# ============================================================
# 3. CREATE TRAIN/VAL SPLIT (before any feature engineering)
# ============================================================
X_all = train_df[feature_cols].copy()
y_all = train_df["car"].copy()

X_train_raw, X_val_raw, y_train_raw, y_val_raw = train_test_split(
    X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
)


# ============================================================
# 4. FEATURE ENGINEERING FUNCTION
# ============================================================
def engineer_features(df_raw, ordinal_mappings, feature_cols):
    """Apply ordinal encoding and create interaction features."""
    df = df_raw.copy()

    # Ordinal encoding
    for col in feature_cols:
        df[col] = df[col].map(ordinal_mappings[col]).astype(np.int8)

    # Pairwise interaction features (products capture decision-tree logic)
    n = len(feature_cols)
    for i in range(n):
        for j in range(i + 1, n):
            name = f"{feature_cols[i]}_x_{feature_cols[j]}"
            df[name] = (df[feature_cols[i]] * df[feature_cols[j]]).astype(np.int8)

    # Sum features (captures overall "quality" signal)
    df["sum_all"] = df[feature_cols].sum(axis=1).astype(np.int8)

    # Difference features (captures relative ordering)
    df["buying_minus_maint"] = (df["buying"] - df["maint"]).astype(np.int8)
    df["safety_plus_lug"] = (df["safety"] + df["lug_boot"]).astype(np.int8)
    df["persons_minus_doors"] = (df["persons"] - df["doors"]).astype(np.int8)

    return df


# Apply feature engineering consistently to all splits
X_train = engineer_features(X_train_raw, ordinal_mappings, feature_cols)
X_val = engineer_features(X_val_raw, ordinal_mappings, feature_cols)
X_test = engineer_features(test_df[feature_cols], ordinal_mappings, feature_cols)

# Encode target
y_train = y_train_raw.map(target_mapping).astype(np.int8)
y_val = y_val_raw.map(target_mapping).astype(np.int8)

# Store test IDs for submission
test_ids = np.arange(len(test_df))

print(f"\nEngineered features: {X_train.shape[1]}")
print(
    f"X_train shape: {X_train.shape}, X_val shape: {X_val.shape}, X_test shape: {X_test.shape}"
)
print(f"y_train distribution:\n{y_train.value_counts().sort_index()}")

# ============================================================
# 5. MODEL DESIGN: LightGBM Gradient Boosting
# ============================================================
# LightGBM with histogram-based splitting is ideal for this small categorical
# dataset because: (1) it handles ordinal-encoded features natively,
# (2) boosting corrects the underfitting of a single shallow tree by
# iteratively fitting residuals, (3) with 1382 samples and simple decision
# boundaries, a well-tuned LGBM should achieve near-perfect accuracy.
model = lgb.LGBMClassifier(
    num_leaves=31,
    learning_rate=0.1,
    n_estimators=200,
    min_child_samples=5,
    random_state=42,
    verbose=-1,
)

# ============================================================
# 6. TRAIN THE MODEL WITH EARLY STOPPING
# ============================================================
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(50, verbose=False)],
)
best_iter = model.best_iteration_
print(f"\nModel trained. Best iteration: {best_iter}")
print(f"Training accuracy: {model.score(X_train, y_train):.6f}")

# ============================================================
# 7. VALIDATION
# ============================================================
val_pred_int = model.predict(X_val)
val_accuracy = accuracy_score(y_val, val_pred_int)

print(f"Validation Accuracy: {val_accuracy:.6f}")
print(
    f"Validation predictions distribution: {pd.Series(val_pred_int).map(inverse_target_mapping).value_counts().to_dict()}"
)
print(
    f"True validation distribution: {pd.Series(y_val).map(inverse_target_mapping).value_counts().to_dict()}"
)

# ============================================================
# 8. RETRAIN ON FULL DATA AND TEST INFERENCE
# ============================================================
# Retrain on the full training data (train+val) with the optimal number
# of iterations found by early stopping for maximum data utilization.
X_full = pd.concat([X_train, X_val], axis=0).reset_index(drop=True)
y_full = pd.concat([y_train, y_val], axis=0).reset_index(drop=True)

final_model = lgb.LGBMClassifier(
    num_leaves=31,
    learning_rate=0.1,
    n_estimators=best_iter,
    min_child_samples=5,
    random_state=42,
    verbose=-1,
)
final_model.fit(X_full, y_full)
print(f"Final model retrained on full data with {best_iter} iterations.")

test_pred_int = final_model.predict(X_test)
test_pred_str = np.array([inverse_target_mapping[p] for p in test_pred_int])

print(
    f"\nTest predictions distribution: {pd.Series(test_pred_str).value_counts().to_dict()}"
)

# ============================================================
# 9. SAVE SUBMISSION
# ============================================================
os.makedirs("./submission", exist_ok=True)

submission = pd.DataFrame({"id": test_ids, "car": test_pred_str})
submission.to_csv("./submission/submission.csv", index=False)
print(f"\nSubmission saved: {submission.shape[0]} rows")
print(submission.head())

# ============================================================
# 10. FINAL SCORE
# ============================================================
score = val_accuracy
print(f"Final Validation Score: {score}")
