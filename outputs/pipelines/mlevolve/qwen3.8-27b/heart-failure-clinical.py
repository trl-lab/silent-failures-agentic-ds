import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import os

# Create directories
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# ============================================================
# 1. LOAD DATA
# ============================================================
train_full = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

print(f"Train shape: {train_full.shape}")
print(f"Test shape: {test.shape}")
print(f"Target distribution:\n{train_full['DEATH_EVENT'].value_counts()}")
print(f"Target rate: {train_full['DEATH_EVENT'].mean():.4f}")

# ============================================================
# 2. DATA CLEANING
# ============================================================
print(f"\nMissing values in train:\n{train_full.isnull().sum()}")
print(f"Missing values in test:\n{test.isnull().sum()}")

print(f"\nDuplicate rows in train: {train_full.duplicated().sum()}")
train_full = train_full.drop_duplicates().reset_index(drop=True)

target_col = "DEATH_EVENT"
id_cols = [c for c in train_full.columns if c.lower() in ["id", "patient_id", "index"]]
feature_cols = [c for c in train_full.columns if c != target_col and c not in id_cols]
print(f"\nFeature columns: {feature_cols}")


# ============================================================
# 3. FEATURE ENGINEERING
# ============================================================
def engineer_features(df):
    """Domain-informed feature engineering for heart failure data."""
    df = df.copy()

    # Log transforms for skewed continuous features
    df["log_platelets"] = np.log1p(df["platelets"])
    df["log_creatinine"] = np.log1p(df["serum_creatinine"])
    df["log_cpk"] = np.log1p(df["creatinine_phosphokinase"])

    # Clinical interaction features
    df["ef_creatinine"] = df["ejection_fraction"] * df["serum_creatinine"]
    df["ef_log_creatinine"] = df["ejection_fraction"] * df["log_creatinine"]
    df["time_ef"] = df["time"] * df["ejection_fraction"]
    df["time_log_creatinine"] = df["time"] * df["log_creatinine"]

    # Binary risk stacking
    risk_factors = ["anaemia", "diabetes", "high_blood_pressure", "smoking"]
    df["risk_factor_count"] = df[risk_factors].sum(axis=1)

    # Severity ratios
    df["ef_deficit"] = (70 - df["ejection_fraction"]).clip(lower=0)
    df["sodium_creatinine"] = df["serum_sodium"] * df["serum_creatinine"]

    # Time-based features
    df["time_log"] = np.log1p(df["time"])
    df["time_squared"] = df["time"] ** 2

    # Platelet-creatinine ratio
    df["platelet_creatinine_ratio"] = df["platelets"] / (df["serum_creatinine"] + 0.1)
    df["log_platelet_creatinine_ratio"] = np.log1p(df["platelet_creatinine_ratio"])

    # Age interactions
    df["age_ef"] = df["age"] * df["ejection_fraction"]
    df["age_diabetes"] = df["age"] * df["diabetes"]

    # CPK-creatinine
    df["cpk_creatinine"] = df["creatinine_phosphokinase"] * df["serum_creatinine"]

    return df


train_full_eng = engineer_features(train_full)
test_eng = engineer_features(test)

final_feature_cols = [
    c for c in train_full_eng.columns if c != target_col and c not in id_cols
]
print(f"\nFinal feature count: {len(final_feature_cols)}")

# ============================================================
# 4. TRAIN/VALIDATION SPLIT (Stratified)
# ============================================================
X_train_full = train_full_eng[final_feature_cols]
y_train_full = train_full_eng[target_col]

X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
)

X_test = test_eng[final_feature_cols].values

print(f"\nTrain split: {X_train.shape}")
print(f"Validation split: {X_val.shape}")
print(f"Train target rate: {y_train.mean():.4f}")
print(f"Val target rate: {y_val.mean():.4f}")

# Convert to numpy arrays
X_train = X_train.values
y_train = y_train.values
X_val = X_val.values
y_val = y_val.values

# ============================================================
# 5. MULTI-SEED ENSEMBLE TRAINING
# ============================================================
base_params = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "learning_rate": 0.01,
    "num_leaves": 4,
    "max_depth": 3,
    "min_child_samples": 15,
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    "reg_alpha": 0.5,
    "reg_lambda": 5.0,
    "min_split_gain": 0.1,
    "n_estimators": 2000,
    "verbose": -1,
}

seeds = [42, 123, 456, 789, 1024]
val_preds_list = []
test_preds_list = []
seed_aucs = []

for i, seed in enumerate(seeds):
    params = base_params.copy()
    params["random_state"] = seed
    params["subsample_seed"] = seed
    params["colsample_seed"] = seed

    train_data = lgb.Dataset(X_train, label=y_train, feature_name=final_feature_cols)
    val_data = lgb.Dataset(
        X_val, label=y_val, feature_name=final_feature_cols, reference=train_data
    )

    callbacks = [
        lgb.early_stopping(stopping_rounds=50, verbose=False),
        lgb.log_evaluation(period=-1),
    ]

    model = lgb.train(
        params,
        train_data,
        valid_sets=[val_data],
        valid_names=["valid"],
        callbacks=callbacks,
    )

    best_iter = model.best_iteration
    val_pred = model.predict(X_val, num_iteration=best_iter)
    test_pred = model.predict(X_test, num_iteration=best_iter)

    val_auc = roc_auc_score(y_val, val_pred)
    seed_aucs.append(val_auc)

    val_preds_list.append(val_pred)
    test_preds_list.append(test_pred)

    print(f"  Seed {seed}: best_iter={best_iter}, val_AUC={val_auc:.4f}")

# ============================================================
# 6. ENSEMBLE PREDICTIONS
# ============================================================
val_pred_ensemble = np.mean(val_preds_list, axis=0)
test_pred_ensemble = np.mean(test_preds_list, axis=0)

# ============================================================
# 7. EVALUATE ON VALIDATION (ROC-AUC - official metric)
# ============================================================
val_auc_ensemble = roc_auc_score(y_val, val_pred_ensemble)

print(f"\n--- Ensemble Results ---")
print(f"Per-seed AUCs: {[f'{a:.4f}' for a in seed_aucs]}")
print(f"Mean AUC: {np.mean(seed_aucs):.4f} ± {np.std(seed_aucs):.4f}")
print(f"Ensemble Validation ROC-AUC: {val_auc_ensemble:.4f}")

# ============================================================
# 8. FEATURE IMPORTANCE (from last seed model)
# ============================================================
importance = model.feature_importance(importance_type="gain")
feat_imp = pd.DataFrame(
    {"feature": final_feature_cols, "importance": importance}
).sort_values("importance", ascending=False)

print(f"\nTop 10 features by gain importance:")
print(feat_imp.head(10).to_string(index=False))

# ============================================================
# 9. SAVE SUBMISSION
# ============================================================
submission = pd.DataFrame(
    {"id": range(len(test_pred_ensemble)), "DEATH_EVENT": test_pred_ensemble}
)
submission.to_csv("./submission/submission.csv", index=False)
print(f"\nSubmission saved: {submission.shape}")
print(
    f"Prediction range: [{test_pred_ensemble.min():.4f}, {test_pred_ensemble.max():.4f}]"
)
print(f"Prediction mean: {test_pred_ensemble.mean():.4f}")

# ============================================================
# 10. FINAL SCORE
# ============================================================
score = val_auc_ensemble
print(f"Final Validation Score: {score}")
