import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os
import warnings

warnings.filterwarnings("ignore")

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
print(f"Target distribution:\n{train_df['NObeyesdad'].value_counts()}")

# ============================================================
# 2. CLEANING
# ============================================================
# Remove extreme outliers (biologically implausible values)
train_df = train_df[
    (train_df["Height"] > 0.5)
    & (train_df["Height"] < 2.5)
    & (train_df["Weight"] > 20)
    & (train_df["Weight"] < 300)
    & (train_df["Age"] >= 18)
    & (train_df["Age"] <= 100)
].reset_index(drop=True)

# Do NOT filter test set - we must predict for ALL test rows
# Clip test values to valid ranges to avoid extreme outliers affecting predictions
for col, lo, hi in [("Height", 0.5, 2.5), ("Weight", 20, 300), ("Age", 18, 100)]:
    test_df[col] = test_df[col].clip(lo, hi)

print(f"After cleaning - Train: {train_df.shape}, Test: {test_df.shape}")


# ============================================================
# 3. FEATURE ENGINEERING
# ============================================================
def engineer_features(df):
    """Create domain-informed features for obesity prediction."""
    df = df.copy()

    # --- Core BMI features (most critical) ---
    df["BMI"] = df["Weight"] / (df["Height"] ** 2)
    df["BMI_squared"] = df["BMI"] ** 2
    df["BMI_cubed"] = df["BMI"] ** 3
    df["BMI_log"] = np.log1p(df["BMI"])
    df["BMI_inv"] = 1.0 / df["BMI"]

    # BMI deviation from healthy range midpoint (21.7)
    df["BMI_dev_from_healthy"] = df["BMI"] - 21.7
    df["BMI_abs_dev"] = np.abs(df["BMI_dev_from_healthy"])

    # BMI relative to obesity thresholds
    df["BMI_over_25"] = df["BMI"] - 25.0
    df["BMI_over_30"] = df["BMI"] - 30.0
    df["BMI_over_35"] = df["BMI"] - 35.0
    df["BMI_over_40"] = df["BMI"] - 40.0

    # --- Body composition features ---
    df["Height_sq"] = df["Height"] ** 2
    df["Weight_per_height"] = df["Weight"] / df["Height"]
    df["Body_mass_index_alt"] = df["Weight"] / (df["Height"] ** 1.5)

    # --- Caloric balance proxy ---
    df["Caloric_intake_proxy"] = (
        df["FAVC"] * 2
        + (3 - df["FCVC"]) * 1
        + df["NCP"] * 0.5
        + (df["CAEC"].astype(str).str.lower() == "yes").astype(int) * 1.5
    )

    # --- Activity vs sedentary balance ---
    df["Activity_vs_sedentary"] = df["FAF"] - df["TUE"]
    df["Activity_ratio"] = df["FAF"] / (df["TUE"] + 1)
    df["Sedentary_score"] = df["TUE"] * 2 + (df["CAEC"].astype(str).str.lower() == "yes").astype(int)

    # --- Metabolic health indicators ---
    df["Water_per_kg"] = df["CH2O"] / df["Weight"]
    df["Water_activity_ratio"] = df["CH2O"] / (df["FAF"] + 1)
    df["Smoking_metabolic_effect"] = df["SMOKE"] * df["BMI"]

    # --- Lifestyle composite scores ---
    df["Unhealthy_lifestyle"] = (
        df["FAVC"] * 2
        + (3 - df["FCVC"])
        + (df["CAEC"].astype(str).str.lower() == "yes").astype(int) * 1.5
        + df["SMOKE"] * 1
        + (df["CALC"].astype(str).isin(["Sometimes", "Always"]).astype(int)) * 1
        + df["TUE"] * 0.3
    )

    df["Healthy_lifestyle"] = (
        df["FCVC"] * 1
        + df["FAF"] * 1
        + df["CH2O"] * 0.5
        + df["SCC"] * 1
        + (df["CALC"].astype(str).str.lower() == "no").astype(int) * 0.5
    )

    # Lifestyle-BMI interaction
    df["BMI_x_unhealthy"] = df["BMI"] * df["Unhealthy_lifestyle"]
    df["BMI_x_healthy"] = df["BMI"] * df["Healthy_lifestyle"]
    df["BMI_x_activity"] = df["BMI"] * df["FAF"]
    df["BMI_x_water"] = df["BMI"] * df["CH2O"]
    df["BMI_x_smoke"] = df["BMI"] * df["SMOKE"]
    df["BMI_x_veg"] = df["BMI"] * df["FCVC"]
    df["BMI_x_caloric"] = df["BMI"] * df["Caloric_intake_proxy"]

    # --- Age interactions ---
    df["Age_BMI"] = df["Age"] * df["BMI"]
    df["Age_squared"] = df["Age"] ** 2
    df["Age_activity"] = df["Age"] * df["FAF"]
    df["Age_sedentary"] = df["Age"] * df["TUE"]

    # --- Transportation & activity ---
    df["Transport_activity"] = (df["MTRANS"].astype(str) == "Walking").astype(int) + df["FAF"]
    df["Transport_sedentary"] = (
        df["MTRANS"].astype(str).isin(["Car", "Public transportation", "Motorbike"])
    ).astype(int)

    # --- Alcohol effects ---
    calc_map = {"no": 0, "sometimes": 1, "always": 2}
    calc_numeric = df["CALC"].astype(str).str.lower().map(calc_map).fillna(0)
    df["Alcohol_BMI"] = calc_numeric * df["BMI"]
    df["Alcohol_activity"] = calc_numeric * df["FAF"]

    # --- Family history interaction ---
    df["Family_BMI"] = df["family_history_with_overweight"] * df["BMI"]
    df["Family_unhealthy"] = (
        df["family_history_with_overweight"] * df["Unhealthy_lifestyle"]
    )

    # --- Meals & snacking ---
    df["Meals_snacks"] = df["NCP"] + (df["CAEC"].astype(str).str.lower() == "yes").astype(int) * 2
    df["Meals_BMI"] = df["NCP"] * df["BMI"]

    # --- Technology & BMI ---
    df["Tech_BMI"] = df["TUE"] * df["BMI"]
    df["Tech_activity_gap"] = df["TUE"] * df["FAF"]

    # --- Refined BMI bins (ordinal encoding) ---
    df["BMI_bin"] = pd.cut(
        df["BMI"], bins=[0, 18.5, 25, 30, 35, 40, 100], labels=[0, 1, 2, 3, 4, 5]
    ).astype(int)

    return df


# ============================================================
# 3b. CONVERT BINARY COLUMNS TO NUMERIC (before feature engineering)
# ============================================================
binary_cols = ["family_history_with_overweight", "FAVC", "SMOKE", "SCC"]

for col in binary_cols:
    # Handle string representations: "yes"/"no", "True"/"False", "1"/"0", etc.
    if train_df[col].dtype == object:
        train_df[col] = train_df[col].map(
            lambda x: 1 if str(x).lower() in ("yes", "true", "1", "1.0") else 0
        )
    else:
        train_df[col] = train_df[col].astype(int)

    if test_df[col].dtype == object:
        test_df[col] = test_df[col].map(
            lambda x: 1 if str(x).lower() in ("yes", "true", "1", "1.0") else 0
        )
    else:
        test_df[col] = test_df[col].astype(int)

train_df = engineer_features(train_df)
test_df = engineer_features(test_df)

# ============================================================
# 4. ENCODING CATEGORICAL FEATURES
# ============================================================
categorical_cols = ["Gender", "CAEC", "CALC", "MTRANS"]

# One-hot encode categorical columns (fit on train only)
train_encoded = pd.get_dummies(train_df, columns=categorical_cols, drop_first=True)
test_encoded = pd.get_dummies(test_df, columns=categorical_cols, drop_first=True)

# Align columns
train_cols = [c for c in train_encoded.columns if c not in ["NObeyesdad"]]
test_encoded = test_encoded.reindex(columns=train_cols, fill_value=0)

# Ensure all feature columns are numeric (fixes object dtype issues)
for col in train_cols:
    train_encoded[col] = pd.to_numeric(train_encoded[col], errors="coerce")
    test_encoded[col] = pd.to_numeric(test_encoded[col], errors="coerce")

# Fill any NaN introduced by coercion with 0
train_encoded[train_cols] = train_encoded[train_cols].fillna(0)
test_encoded[train_cols] = test_encoded[train_cols].fillna(0)

# ============================================================
# 5. TARGET ENCODING
# ============================================================
le = LabelEncoder()
train_df["target_encoded"] = le.fit_transform(train_df["NObeyesdad"])
print(f"Target classes: {le.classes_}")

# ============================================================
# 6. TRAIN/VALIDATION SPLIT (Stratified)
# ============================================================
X_train_full = train_encoded[train_cols].astype(np.float64).values
y_train_full = train_df["target_encoded"].values
y_train_full_original = train_df["NObeyesdad"].values

X_train, X_val, y_train, y_val, y_train_orig, y_val_orig = train_test_split(
    X_train_full,
    y_train_full,
    y_train_full_original,
    test_size=0.2,
    random_state=42,
    stratify=y_train_full,
)

X_test = test_encoded[train_cols].astype(np.float64).values

print(f"Final shapes:")
print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"  X_val: {X_val.shape}, y_val: {y_val.shape}")
print(f"  X_test: {X_test.shape}")

# Verify no NaN
assert not np.isnan(X_train).any(), "NaN in X_train!"
assert not np.isnan(X_val).any(), "NaN in X_val!"
assert not np.isnan(X_test).any(), "NaN in X_test!"

# ============================================================
# 7. MODEL DESIGN & TRAINING
# ============================================================
n_classes = len(le.classes_)
class_counts = np.bincount(y_train, minlength=n_classes)
class_freq = class_counts / class_counts.sum()
print(f"Class frequencies: {class_freq}")

# LightGBM parameters
params = {
    "objective": "multiclass",
    "num_class": n_classes,
    "metric": "multi_logloss",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "max_depth": 6,
    "learning_rate": 0.05,
    "n_estimators": 2000,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "min_split_gain": 0.01,
    "class_weight": "balanced",
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}

print(
    f"\nTraining LightGBM: num_leaves={params['num_leaves']}, lr={params['learning_rate']}, max_depth={params['max_depth']}"
)

# ============================================================
# 8. TRAIN WITH EARLY STOPPING
# ============================================================
model = lgb.LGBMClassifier(**params)

model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    eval_metric="multi_logloss",
    callbacks=[
        lgb.early_stopping(stopping_rounds=100, verbose=False),
        lgb.log_evaluation(period=500),
    ],
)

best_iter = model.best_iteration_
print(f"\nEarly stopping triggered at iteration: {best_iter}")
print(f"Best logloss: {model.best_score_['valid_0']['multi_logloss']:.6f}")

# ============================================================
# 9. EVALUATION ON VALIDATION SET (Official Metric: Accuracy)
# ============================================================
val_pred_proba = model.predict_proba(X_val)
val_pred = val_pred_proba.argmax(axis=1)
val_accuracy = accuracy_score(y_val, val_pred)

print(f"\n{'='*60}")
print(f"VALIDATION RESULTS")
print(f"{'='*60}")
print(f"Best iteration: {best_iter}")
print(f"Validation Accuracy (OFFICIAL METRIC): {val_accuracy:.6f}")
print(f"\nClassification Report:")
print(classification_report(y_val, val_pred, target_names=le.classes_, digits=4))

cm = confusion_matrix(y_val, val_pred)
print(f"Confusion Matrix:")
print(cm)

# Train set metrics for overfitting diagnosis
train_pred_proba = model.predict_proba(X_train)
train_pred = train_pred_proba.argmax(axis=1)
train_accuracy = accuracy_score(y_train, train_pred)
print(f"\nTrain Accuracy: {train_accuracy:.6f}")
print(f"Gap (Train - Val): {train_accuracy - val_accuracy:.6f}")
if train_accuracy - val_accuracy > 0.05:
    print("⚠️  Potential overfitting detected (gap > 5%)")
else:
    print("✓ No significant overfitting detected")

# ============================================================
# 10. FEATURE IMPORTANCE
# ============================================================
importances = model.feature_importances_
feat_imp = pd.Series(importances, index=train_cols).sort_values(ascending=False)
print(f"\nTop 10 Features:")
for i, (name, imp) in enumerate(feat_imp.head(10).items()):
    print(f"  {i+1}. {name}: {imp}")

# ============================================================
# 11. TEST INFERENCE & SUBMISSION
# ============================================================
test_pred_proba = model.predict_proba(X_test)
test_pred = test_pred_proba.argmax(axis=1)
test_pred_labels = le.inverse_transform(test_pred)

test_ids = np.arange(len(test_df))
submission = pd.DataFrame({"id": test_ids.astype(int), "NObeyesdad": test_pred_labels})
submission.to_csv("./submission/submission.csv", index=False)

print(f"\nSubmission saved: {submission.shape}")
print(f"Submission class distribution:")
print(submission["NObeyesdad"].value_counts())
print(f"\nSubmission head:\n{submission.head()}")

# Verify submission format
assert list(submission.columns) == ["id", "NObeyesdad"], "Wrong column names!"
assert submission["id"].dtype == int, "id must be integer!"
assert len(submission) == len(X_test), "Row count mismatch!"

# ============================================================
# 12. FINAL SUMMARY
# ============================================================
score = val_accuracy
print(f"\n{'='*60}")
print(f"FINAL SUMMARY")
print(f"{'='*60}")
print(f"Model: LightGBM (best_iter={best_iter})")
print(f"Validation Accuracy: {val_accuracy:.6f}")
print(f"Train Accuracy:      {train_accuracy:.6f}")
print(f"Submission rows:     {len(submission)}")
print(f"{'='*60}")

print(f"Final Validation Score: {score}")