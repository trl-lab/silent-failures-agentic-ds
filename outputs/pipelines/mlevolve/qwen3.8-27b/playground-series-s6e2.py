import pandas as pd
import numpy as np
import os
import json
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# ============================================================
# 1. LOAD DATA
# ============================================================
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")

# Encode target - robustly handle various possible value formats
print(f"Raw target values: {train['Heart Disease'].unique()}")
print(f"Raw target dtype: {train['Heart Disease'].dtype}")

if train["Heart Disease"].dtype == object:
    # Strip whitespace and try case-insensitive mapping
    train["Heart Disease"] = train["Heart Disease"].str.strip()
    train["Heart Disease"] = train["Heart Disease"].map(
        lambda x: 1 if x.lower() in ("presence", "1", "yes", "true") else 0
    )
else:
    # Already numeric - ensure it's 0/1
    train["Heart Disease"] = train["Heart Disease"].astype(int)

assert train["Heart Disease"].notna().all(), "Target encoding failed - NaN values remain"
print(f"Target distribution: {train['Heart Disease'].value_counts().to_dict()}")
print(f"Target mean: {train['Heart Disease'].mean():.4f}")


# ============================================================
# 2. FEATURE ENGINEERING (domain-informed for cardiovascular)
# ============================================================
def engineer_features(df):
    df = df.copy()

    # --- Cardiovascular ratio features ---
    df["BP_to_MaxHR"] = df["BP"] / (df["Max HR"] + 1)
    df["Chol_to_Age"] = df["Cholesterol"] / (df["Age"] + 1)
    df["MaxHR_to_Age"] = df["Max HR"] / (df["Age"] + 1)
    df["STdep_to_MaxHR"] = df["ST depression"] / (df["Max HR"] + 1)
    df["BP_to_Chol"] = df["BP"] / (df["Cholesterol"] + 1)
    df["MaxHR_to_BP"] = df["Max HR"] / (df["BP"] + 1)

    # --- Key interaction features (medically motivated) ---
    df["Age_x_Sex"] = df["Age"] * df["Sex"]
    df["Age_x_BP"] = df["Age"] * df["BP"]
    df["Age_x_Chol"] = df["Age"] * df["Cholesterol"]
    df["Age_x_STdep"] = df["Age"] * df["ST depression"]
    df["BP_x_STdep"] = df["BP"] * df["ST depression"]
    df["MaxHR_x_STdep"] = df["Max HR"] * df["ST depression"]
    df["MaxHR_x_Sex"] = df["Max HR"] * df["Sex"]
    df["Chol_x_Sex"] = df["Cholesterol"] * df["Sex"]
    df["Vessels_x_Thallium"] = df["Number of vessels fluro"] * df["Thallium"]
    df["Vessels_x_STdep"] = df["Number of vessels fluro"] * df["ST depression"]
    df["ChestPain_x_MaxHR"] = df["Chest pain type"] * df["Max HR"]
    df["ChestPain_x_Age"] = df["Chest pain type"] * df["Age"]
    df["Slope_x_STdep"] = df["Slope of ST"] * df["ST depression"]
    df["Thallium_x_MaxHR"] = df["Thallium"] * df["Max HR"]
    df["Thallium_x_Age"] = df["Thallium"] * df["Age"]
    df["EKG_x_MaxHR"] = df["EKG results"] * df["Max HR"]
    df["EKG_x_Age"] = df["EKG results"] * df["Age"]
    df["FBS_x_Chol"] = df["FBS over 120"] * df["Cholesterol"]
    df["ExAng_x_MaxHR"] = df["Exercise angina"] * df["Max HR"]
    df["ExAng_x_Age"] = df["Exercise angina"] * df["Age"]

    # --- Polynomial features for key continuous variables ---
    df["Age_sq"] = df["Age"] ** 2
    df["BP_sq"] = df["BP"] ** 2
    df["Chol_sq"] = df["Cholesterol"] ** 2
    df["STdep_sq"] = df["ST depression"] ** 2
    df["MaxHR_sq"] = df["Max HR"] ** 2

    # --- Domain-informed composite features ---
    df["Cardiac_Workload"] = df["BP"] * df["Max HR"]
    df["Exercise_Index"] = df["Max HR"] * (1 - df["Exercise angina"])
    df["Risk_Factor_Count"] = (
        (df["Sex"] == 1).astype(int)
        + (df["FBS over 120"] == 1).astype(int)
        + (df["Exercise angina"] == 1).astype(int)
        + (df["BP"] > 140).astype(int)
        + (df["Cholesterol"] > 240).astype(int)
        + (df["ST depression"] > 1.0).astype(int)
        + (df["Number of vessels fluro"] > 0).astype(int)
        + (df["Thallium"] > 3).astype(int)
        + (df["Slope of ST"] == 3).astype(int)
    )
    df["ST_Severity"] = df["ST depression"] * df["Slope of ST"]
    df["Ischemia_Score"] = (
        df["ST depression"]
        + df["Number of vessels fluro"] * 1.5
        + (df["Thallium"] - 3) * 0.5
        + df["Slope of ST"] * 0.3
    )
    df["Age_logBP"] = df["Age"] * np.log1p(df["BP"])
    df["Chol_Age_Exposure"] = df["Cholesterol"] * df["Age"]

    # --- Log transforms ---
    df["log_Cholesterol"] = np.log1p(df["Cholesterol"])
    df["log_BP"] = np.log1p(df["BP"])
    df["log_MaxHR"] = np.log1p(df["Max HR"])
    df["log_STdep"] = np.log1p(df["ST depression"])

    # --- Difference features ---
    df["BP_MaxHR_diff"] = df["BP"] - df["Max HR"]
    df["Chol_MaxHR_diff"] = df["Cholesterol"] - df["Max HR"]
    df["Age_MaxHR_diff"] = df["Age"] - df["Max HR"]

    # --- Triple interactions ---
    df["Age_BP_MaxHR"] = df["Age"] * df["BP"] / (df["Max HR"] + 1)
    df["STdep_Slope_MaxHR"] = (
        df["ST depression"] * df["Slope of ST"] / (df["Max HR"] + 1)
    )
    df["Vessels_Thallium_MaxHR"] = (
        df["Number of vessels fluro"] * df["Thallium"] / (df["Max HR"] + 1)
    )

    return df


train = engineer_features(train)
test = engineer_features(test)

# ============================================================
# 3. DEFINE FEATURE COLUMNS
# ============================================================
target_col = "Heart Disease"
id_col = "id"
feature_cols = [c for c in train.columns if c not in [target_col, id_col]]
print(f"Number of engineered features: {len(feature_cols)}")

# ============================================================
# 4. CREATE TRAIN/VALIDATION SPLIT (stratified)
# ============================================================
train_data, val_data = train_test_split(
    train, test_size=0.2, random_state=42, stratify=train[target_col].astype(int)
)

print(f"Train split: {train_data.shape}")
print(f"Validation split: {val_data.shape}")
print(f"Test: {test.shape}")
print(f"Train target rate: {train_data[target_col].mean():.4f}")
print(f"Val target rate: {val_data[target_col].mean():.4f}")


# ============================================================
# 5. MODEL DEFINITION (LightGBM)
# ============================================================
lgb_params = {
    "objective": "binary",
    "metric": "auc",
    "num_leaves": 255,
    "max_depth": -1,
    "min_child_samples": 20,
    "learning_rate": 0.01,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.7,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "n_jobs": -1,
    "verbosity": -1,
}

print(f"LightGBM configured with {len(feature_cols)} features")

# ============================================================
# 6. PREPARE DATA FOR TRAINING
# ============================================================
X_train = train_data[feature_cols].values.astype(np.float32)
y_train = train_data[target_col].values.astype(np.float32)
X_val = val_data[feature_cols].values.astype(np.float32)
y_val = val_data[target_col].values.astype(np.float32)
X_test = test[feature_cols].values.astype(np.float32)

print(f"Training samples: {len(X_train)}")
print(f"Validation samples: {len(X_val)}")
print(f"Test samples: {len(X_test)}")

# ============================================================
# 7. TRAIN WITH LIGHTGBM (early stopping on validation)
# ============================================================
os.makedirs("./working", exist_ok=True)

train_ds = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols)
val_ds = lgb.Dataset(X_val, label=y_val, feature_name=feature_cols, reference=train_ds)

callbacks = [lgb.early_stopping(stopping_rounds=100), lgb.log_evaluation(period=-1)]

model = lgb.train(
    lgb_params,
    train_ds,
    num_boost_round=5000,
    valid_sets=[train_ds, val_ds],
    valid_names=["train", "valid"],
    callbacks=callbacks,
)

best_iteration = model.best_iteration
best_val_auc = model.best_score["valid"]["auc"]
print(f"\nBest validation AUC: {best_val_auc:.6f} at iteration {best_iteration}")

# ============================================================
# 8. RETRAIN ON FULL DATA (train + val) USING BEST ITERATION
# ============================================================
X_full = np.vstack([X_train, X_val])
y_full = np.concatenate([y_train, y_val])

full_ds = lgb.Dataset(X_full, label=y_full, feature_name=feature_cols)

model_full = lgb.train(
    lgb_params,
    full_ds,
    num_boost_round=best_iteration,
    callbacks=[lgb.log_evaluation(period=-1)],
)

print(f"Full model trained with {best_iteration} iterations")

# ============================================================
# 9. GENERATE PREDICTIONS
# ============================================================
# Validation predictions from first model (for final score)
val_preds = model.predict(X_val, num_iteration=best_iteration)
final_val_auc = roc_auc_score(y_val, val_preds)

# Test predictions from retrained full model
test_preds = model_full.predict(X_test)

# ============================================================
# 10. SAVE SUBMISSION
# ============================================================
os.makedirs("./submission", exist_ok=True)

submission = pd.DataFrame(
    {"id": test["id"].values.astype(int), "Heart Disease": test_preds}
)
submission.to_csv("./submission/submission.csv", index=False)

print(f"Submission saved: {submission.shape[0]} rows")
print(f"Prediction range: [{test_preds.min():.6f}, {test_preds.max():.6f}]")
print(f"Prediction mean: {test_preds.mean():.6f}")

# ============================================================
# FINAL SCORE
# ============================================================
score = final_val_auc
print(f"Final Validation Score: {score}")
