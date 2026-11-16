import pandas as pd
import numpy as np
import os
import lightgbm as lgb
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error

# ============================================================
# 1. LOAD DATA
# ============================================================
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")

target_col = "G3"
feature_cols_raw = [c for c in train_df.columns if c != target_col]

# ============================================================
# 2. SPLIT TRAIN INTO TRAIN/VAL (BEFORE any transformation)
# ============================================================
train_data, val_data = train_test_split(train_df, test_size=0.2, random_state=42)
print(f"After split - Train: {train_data.shape}, Val: {val_data.shape}")

# ============================================================
# 3. FEATURE ENGINEERING
# ============================================================
binary_cols = [
    "schoolsup",
    "famsup",
    "paid",
    "activities",
    "nursery",
    "higher",
    "internet",
    "romantic",
]
categorical_cols = [
    "school",
    "sex",
    "address",
    "famsize",
    "Pstatus",
    "Mjob",
    "Fjob",
    "reason",
    "guardian",
]
ordinal_numeric_cols = [
    "age",
    "Medu",
    "Fedu",
    "traveltime",
    "studytime",
    "failures",
    "famrel",
    "freetime",
    "goout",
    "Dalc",
    "Walc",
    "health",
    "absences",
]

binary_cols = [c for c in binary_cols if c in train_data.columns]
categorical_cols = [c for c in categorical_cols if c in train_data.columns]
ordinal_numeric_cols = [c for c in ordinal_numeric_cols if c in train_data.columns]
grade_cols = [c for c in ["G1", "G2"] if c in train_data.columns]


def engineer_features(df, is_training=False, label_maps=None):
    """Apply feature engineering. Fit encoders on training, apply to val/test."""
    df = df.copy()

    # Binary encoding
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].map({"yes": 1, "no": 0})

    # Categorical label encoding
    if is_training:
        label_maps = {}
        for col in categorical_cols:
            if col in df.columns:
                categories = sorted(df[col].unique())
                mapping = {cat: idx for idx, cat in enumerate(categories)}
                label_maps[col] = mapping
                df[col] = df[col].map(mapping)
    else:
        for col in categorical_cols:
            if col in df.columns and col in label_maps:
                df[col] = df[col].map(label_maps[col]).fillna(-1).astype(int)

    # Feature interactions
    if "Medu" in df.columns and "Fedu" in df.columns:
        df["parent_edu_sum"] = df["Medu"] + df["Fedu"]
        df["parent_edu_min"] = np.minimum(df["Medu"], df["Fedu"])
        df["parent_edu_max"] = np.maximum(df["Medu"], df["Fedu"])
        df["parent_edu_diff"] = df["Medu"] - df["Fedu"]

    if "Dalc" in df.columns and "Walc" in df.columns:
        df["alcohol_total"] = df["Dalc"] + df["Walc"]
        df["alcohol_ratio"] = df["Walc"] / (df["Dalc"] + 1e-5)

    if "freetime" in df.columns and "goout" in df.columns:
        df["social_balance"] = df["freetime"] - df["goout"]

    if "studytime" in df.columns and "absences" in df.columns:
        df["study_absence_ratio"] = df["studytime"] / (df["absences"] + 1)

    support_cols = [
        c
        for c in ["schoolsup", "famsup", "paid", "activities", "nursery", "higher"]
        if c in df.columns
    ]
    if len(support_cols) > 0:
        df["total_support"] = df[support_cols].sum(axis=1)

    if "G1" in df.columns and "G2" in df.columns:
        df["G1_G2_mean"] = (df["G1"] + df["G2"]) / 2
        df["G1_G2_diff"] = df["G2"] - df["G1"]
        df["G1_G2_min"] = np.minimum(df["G1"], df["G2"])
        df["G1_G2_max"] = np.maximum(df["G1"], df["G2"])
        df["grade_trend"] = df["G2"] - df["G1"]
        # Domain-informed features: weighted prior grade (strongest predictor of G3)
        df["weighted_prior_grade"] = 0.4 * df["G1"] + 0.6 * df["G2"]
        df["G1_G2_weighted_avg"] = (df["G1"] + 2 * df["G2"]) / 3
        df["G2_minus_G1_trend"] = df["G2"] - df["G1"]

    if "G1" in df.columns:
        df["G1_squared"] = df["G1"] ** 2
    if "G2" in df.columns:
        df["G2_squared"] = df["G2"] ** 2

    if "age" in df.columns and "studytime" in df.columns:
        df["age_studytime"] = df["age"] * df["studytime"]

    if "traveltime" in df.columns and "studytime" in df.columns:
        df["travel_study_ratio"] = df["traveltime"] / (df["studytime"] + 1e-5)

    if "health" in df.columns and "absences" in df.columns:
        df["health_absence"] = df["health"] * df["absences"]

    if "failures" in df.columns and "G1" in df.columns:
        df["failures_G1"] = df["failures"] * df["G1"]

    return df, label_maps


# Apply feature engineering
train_processed, label_maps = engineer_features(train_data, is_training=True)
val_processed, _ = engineer_features(val_data, is_training=False, label_maps=label_maps)
test_processed, _ = engineer_features(test_df, is_training=False, label_maps=label_maps)

# Define final feature set
final_feature_cols = [c for c in train_processed.columns if c != target_col]
print(f"Final feature set: {len(final_feature_cols)} features")

# ============================================================
# 4. PREPARE NUMERIC ARRAYS
# ============================================================
X_train = train_processed[final_feature_cols].values.astype(np.float64)
y_train = train_processed[target_col].values.astype(np.float64)
X_val = val_processed[final_feature_cols].values.astype(np.float64)
y_val = val_processed[target_col].values.astype(np.float64)
X_test = test_processed[final_feature_cols].values.astype(np.float64)

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
print(f"Target range: [{y_train.min():.1f}, {y_train.max():.1f}]")

# ============================================================
# 5. MODEL DEFINITIONS - Multi-configuration LightGBM Ensemble
# ============================================================

# 5 different LightGBM hyperparameter configurations for diversity
lgb_configs = [
    {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.03,
        "num_leaves": 15,
        "max_depth": 4,
        "min_child_samples": 8,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.7,
        "bagging_freq": 3,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "verbose": -1,
        "seed": 42,
    },
    {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.05,
        "num_leaves": 10,
        "max_depth": 3,
        "min_child_samples": 12,
        "feature_fraction": 0.6,
        "bagging_fraction": 0.8,
        "bagging_freq": 2,
        "reg_alpha": 0.5,
        "reg_lambda": 2.0,
        "verbose": -1,
        "seed": 123,
    },
    {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.02,
        "num_leaves": 20,
        "max_depth": 5,
        "min_child_samples": 6,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.7,
        "bagging_freq": 3,
        "reg_alpha": 0.05,
        "reg_lambda": 0.5,
        "verbose": -1,
        "seed": 7,
    },
    {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.04,
        "num_leaves": 12,
        "max_depth": 4,
        "min_child_samples": 10,
        "feature_fraction": 0.65,
        "bagging_fraction": 0.75,
        "bagging_freq": 4,
        "reg_alpha": 0.2,
        "reg_lambda": 1.5,
        "verbose": -1,
        "seed": 2024,
    },
    {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.03,
        "num_leaves": 8,
        "max_depth": 3,
        "min_child_samples": 15,
        "feature_fraction": 0.5,
        "bagging_fraction": 0.85,
        "bagging_freq": 2,
        "reg_alpha": 1.0,
        "reg_lambda": 3.0,
        "verbose": -1,
        "seed": 999,
    },
]

n_models = len(lgb_configs)


# ============================================================
# 6. K-FOLD CV TRAINING (OOF predictions for weighted averaging)
# ============================================================
n_folds = 5
kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

# OOF predictions for each of the 5 LGB configurations
oof_preds = np.zeros((n_models, len(X_train)))

print("\n--- K-Fold CV Training (5 LGB configurations) ---")

for model_idx, params in enumerate(lgb_configs):
    for fold, (fold_train_idx, fold_val_idx) in enumerate(kf.split(X_train)):
        X_fold_tr = X_train[fold_train_idx]
        y_fold_tr = y_train[fold_train_idx]
        X_fold_vl = X_train[fold_val_idx]
        y_fold_vl = y_train[fold_val_idx]

        dtrain = lgb.Dataset(X_fold_tr, label=y_fold_tr)
        dval = lgb.Dataset(X_fold_vl, label=y_fold_vl, reference=dtrain)
        lgb_model = lgb.train(
            params,
            dtrain,
            num_boost_round=2000,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(80), lgb.log_evaluation(-1)],
        )
        oof_preds[model_idx, fold_val_idx] = lgb_model.predict(X_fold_vl)

    oof_rmse_model = np.sqrt(mean_squared_error(y_train, oof_preds[model_idx]))
    print(f"  Model {model_idx+1} (seed={params['seed']}, lr={params['learning_rate']}): OOF RMSE = {oof_rmse_model:.4f}")

# ============================================================
# 7. WEIGHTED AVERAGE (inverse OOF RMSE weights)
# ============================================================
oof_rmses = np.sqrt(np.mean((y_train - oof_preds) ** 2, axis=1))
# Inverse RMSE weighting: better models get more weight
inverse_rmses = 1.0 / oof_rmses
weights = inverse_rmses / inverse_rmses.sum()

oof_ensemble = np.average(oof_preds, axis=0, weights=weights)
oof_rmse = np.sqrt(mean_squared_error(y_train, oof_ensemble))
print(f"\nOOF Ensemble RMSE (5-Fold CV, weighted avg): {oof_rmse:.4f}")
print(f"Weights: {weights}")
print(f"Individual OOF RMSEs: {oof_rmses}")

# ============================================================
# 8. TRAIN FINAL MODELS ON FULL TRAINING DATA
# ============================================================
print("\n--- Training Final Models on Full Training Data ---")

val_preds = np.zeros((n_models, len(X_val)))
test_preds = np.zeros((n_models, len(X_test)))

for model_idx, params in enumerate(lgb_configs):
    dtrain_full = lgb.Dataset(X_train, label=y_train)
    dval_full = lgb.Dataset(X_val, label=y_val, reference=dtrain_full)
    lgb_final = lgb.train(
        params,
        dtrain_full,
        num_boost_round=2000,
        valid_sets=[dval_full],
        callbacks=[lgb.early_stopping(80), lgb.log_evaluation(-1)],
    )
    val_preds[model_idx] = lgb_final.predict(X_val)
    test_preds[model_idx] = lgb_final.predict(X_test)
    val_rmse_model = np.sqrt(mean_squared_error(y_val, val_preds[model_idx]))
    print(f"  Model {model_idx+1} val RMSE: {val_rmse_model:.4f}")

# ============================================================
# 9. ENSEMBLE PREDICTIONS (weighted average)
# ============================================================
val_ensemble = np.average(val_preds, axis=0, weights=weights)
test_ensemble = np.average(test_preds, axis=0, weights=weights)

# Clip to valid grade range [0, 20] - applied identically to both val and test
val_ensemble = np.clip(val_ensemble, 0, 20)
test_ensemble = np.clip(test_ensemble, 0, 20)

# ============================================================
# 10. EVALUATE
# ============================================================
val_rmse = np.sqrt(mean_squared_error(y_val, val_ensemble))
print(f"\n--- Final Evaluation ---")
for i in range(n_models):
    print(f"  Model {i+1} val RMSE: {np.sqrt(mean_squared_error(y_val, val_preds[i])):.4f}")
print(f"  Ensemble val RMSE: {val_rmse:.4f}")
print(f"  OOF CV RMSE:       {oof_rmse:.4f}")

# ============================================================
# 11. SAVE SUBMISSION
# ============================================================
os.makedirs("./submission", exist_ok=True)
submission = pd.DataFrame({"id": range(len(test_ensemble)), "G3": test_ensemble})
submission.to_csv("./submission/submission.csv", index=False)
print(f"\nSubmission saved: {submission.shape}")
print(submission.head())

print(f"Final Validation Score: {val_rmse:.4f}")
