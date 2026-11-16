import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
import xgboost as xgb
import json
import os
import pickle

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
print(f"Target distribution:\n{train_df['income'].value_counts(normalize=True)}")
print(
    f"Missing values in train:\n{train_df.isnull().sum()[train_df.isnull().sum() > 0]}"
)
print(f"Missing values in test:\n{test_df.isnull().sum()[test_df.isnull().sum() > 0]}")


def normalize_income(val):
    """Map income labels to binary: >50K variants -> 1, <=50K variants -> 0."""
    if val in (">50K", ">50K."):
        return 1
    return 0


# ============================================================
# 2. CREATE TRAIN/VAL SPLIT (Stratified)
# ============================================================
TARGET = "income"

train_df, val_df = train_test_split(
    train_df, test_size=0.2, random_state=42, stratify=train_df[TARGET]
)

print(f"Train split shape: {train_df.shape}")
print(f"Val split shape: {val_df.shape}")

# ============================================================
# 3. FEATURE ENGINEERING
# ============================================================


def engineer_features(df):
    """Apply deterministic feature engineering to a dataframe."""
    df = df.copy()

    # --- Missing value indicators (informative missingness) ---
    for col in ["occupation", "workclass", "native-country"]:
        df[f"{col}_missing"] = df[col].isnull().astype(int)
        df[col] = df[col].fillna("Missing")

    # --- Binary indicators for skewed continuous features ---
    df["capital_gain_positive"] = (df["capital-gain"] > 0).astype(int)
    df["capital_loss_positive"] = (df["capital-loss"] > 0).astype(int)
    df["has_capital_gain"] = (df["capital-gain"] > 0).astype(int)
    df["has_capital_loss"] = (df["capital-loss"] > 0).astype(int)

    # Log transform for skewed features
    df["capital_gain_log"] = np.log1p(df["capital-gain"])
    df["capital_loss_log"] = np.log1p(df["capital-loss"])
    df["fnlwgt_log"] = np.log1p(df["fnlwgt"])

    # fnlwgt binned features (log bins)
    fnlwgt_log = np.log1p(df["fnlwgt"])
    df["fnlwgt_bin_1"] = (fnlwgt_log < 10.0).astype(int)
    df["fnlwgt_bin_2"] = ((fnlwgt_log >= 10.0) & (fnlwgt_log < 10.5)).astype(int)
    df["fnlwgt_bin_3"] = ((fnlwgt_log >= 10.5) & (fnlwgt_log < 11.0)).astype(int)
    df["fnlwgt_bin_4"] = ((fnlwgt_log >= 11.0) & (fnlwgt_log < 11.5)).astype(int)
    df["fnlwgt_bin_5"] = (fnlwgt_log >= 11.5).astype(int)

    # --- Domain features ---
    married_statuses = [
        "Married-civ-spouse",
        "Married-spouse-absent",
        "Married-AF-spouse",
    ]
    df["is_married"] = df["marital-status"].isin(married_statuses).astype(int)

    df["is_high_education"] = (df["education-num"] >= 13).astype(int)
    df["is_educated"] = (df["education-num"] >= 12).astype(int)

    df["hours_low"] = (df["hours-per-week"] <= 20).astype(int)
    df["hours_high"] = (df["hours-per-week"] >= 50).astype(int)
    df["hours_40"] = (df["hours-per-week"] == 40).astype(int)

    df["age_young"] = (df["age"] < 25).astype(int)
    df["age_middle"] = ((df["age"] >= 35) & (df["age"] <= 55)).astype(int)
    df["age_senior"] = (df["age"] > 60).astype(int)

    df["is_us"] = (df["native-country"] == "United-States").astype(int)

    # --- Interaction features ---
    df["marital_relationship"] = df["marital-status"] + "_" + df["relationship"]
    df["sex_occupation"] = df["sex"] + "_" + df["occupation"]
    df["sex_marital"] = df["sex"] + "_" + df["marital-status"]
    df["education_marital"] = df["education"] + "_" + df["marital-status"]
    df["workclass_occupation"] = df["workclass"] + "_" + df["occupation"]

    df["age_hours"] = df["age"] * df["hours-per-week"]
    df["age_education_num"] = df["age"] * df["education-num"]
    df["capital_gain_x_hours"] = df["capital_gain_positive"] * df["hours-per-week"]

    # --- Ratio features ---
    df["age_per_education"] = df["age"] / (df["education-num"] + 1)
    df["hours_per_age"] = df["hours-per-week"] / (df["age"] + 1)

    # Drop target if present
    if TARGET in df.columns:
        df = df.drop(columns=[TARGET])

    return df


# Apply feature engineering
train_processed = engineer_features(train_df)
val_processed = engineer_features(val_df)
test_processed = engineer_features(test_df)

# ============================================================
# 3b. TARGET ENCODING (5-fold CV to prevent leakage)
# ============================================================
target_encode_cols = ["marital-status", "occupation", "workclass", "relationship", "education"]

# We need the original (pre-engineering) categorical columns for target encoding
# Since engineer_features creates interaction features, we do target encoding on the
# original columns before they get ordinal-encoded. We'll add target-encoded features.

# Get y for target encoding (from the full train before split - use train_df and val_df)
y_train_full = train_df[TARGET].apply(normalize_income).values
y_val_full = val_df[TARGET].apply(normalize_income).values

# We need to work with the original categorical columns from train_df/val_df/test_df
# Create target encoding using 5-fold CV on train only
n_folds_te = 5
kf_te = KFold(n_splits=n_folds_te, shuffle=True, random_state=42)

# For train: 5-fold target encoding
train_te = pd.DataFrame(index=train_processed.index)
for col in target_encode_cols:
    train_te[f"{col}_te"] = 0.0
    global_mean = y_train_full.mean()
    for fold_idx, (tr_idx, va_idx) in enumerate(kf_te.split(train_processed)):
        # Compute mean target for each category using training fold
        temp_df = train_df.iloc[tr_idx][[col]].copy()
        temp_df["target"] = y_train_full[tr_idx]
        mean_map = temp_df.groupby(col)["target"].mean()
        # Assign to validation fold
        va_cats = train_df.iloc[va_idx][col].values
        train_te.iloc[va_idx, train_te.columns.get_loc(f"{col}_te")] = (
            pd.Series(va_cats).map(mean_map).fillna(global_mean).values
        )

# For val and test: use full train to compute target means
val_te = pd.DataFrame(index=val_processed.index)
test_te = pd.DataFrame(index=test_processed.index)
for col in target_encode_cols:
    global_mean = y_train_full.mean()
    temp_df = train_df[[col]].copy()
    temp_df["target"] = y_train_full
    mean_map = temp_df.groupby(col)["target"].mean()

    val_cats = val_df[col].values
    val_te[f"{col}_te"] = pd.Series(val_cats).map(mean_map).fillna(global_mean).values

    test_cats = test_df[col].values
    test_te[f"{col}_te"] = pd.Series(test_cats).map(mean_map).fillna(global_mean).values

# Add target encoded features to processed dataframes
train_processed = pd.concat([train_processed, train_te], axis=1)
val_processed = pd.concat([val_processed, val_te], axis=1)
test_processed = pd.concat([test_processed, test_te], axis=1)

# Identify feature columns
feature_cols = train_processed.columns.tolist()
print(f"Number of features: {len(feature_cols)}")

# ============================================================
# 4. ENCODE CATEGORICALS (fit on train only)
# ============================================================
categorical_cols = train_processed.select_dtypes(include=["object"]).columns.tolist()
print(f"Categorical columns to encode: {categorical_cols}")

oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
train_processed[categorical_cols] = oe.fit_transform(train_processed[categorical_cols])
val_processed[categorical_cols] = oe.transform(val_processed[categorical_cols])
test_processed[categorical_cols] = oe.transform(test_processed[categorical_cols])

# Prepare arrays
y_train = train_df[TARGET].apply(normalize_income).values
y_val = val_df[TARGET].apply(normalize_income).values

X_train = train_processed[feature_cols].values
X_val = val_processed[feature_cols].values
X_test = test_processed[feature_cols].values

print(f"Training samples: {X_train.shape[0]}")
print(f"Validation samples: {X_val.shape[0]}")
print(f"Test samples: {X_test.shape[0]}")
print(f"Class balance (train): {np.mean(y_train):.3f}")
print(f"Class balance (val): {np.mean(y_val):.3f}")

# ============================================================
# 5. TRAINING LOOP WITH ACCURACY-BASED EARLY STOPPING
# ============================================================


# Define 5 diverse LightGBM configurations
model_configs = [
    {
        "objective": "binary",
        "n_estimators": 5000,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "max_depth": 8,
        "min_child_samples": 30,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    },
    {
        "objective": "binary",
        "n_estimators": 5000,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": 6,
        "min_child_samples": 50,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 123,
        "n_jobs": -1,
        "verbose": -1,
    },
    {
        "objective": "binary",
        "n_estimators": 5000,
        "learning_rate": 0.05,
        "num_leaves": 127,
        "max_depth": 10,
        "min_child_samples": 20,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 456,
        "n_jobs": -1,
        "verbose": -1,
    },
    {
        "objective": "binary",
        "n_estimators": 5000,
        "learning_rate": 0.02,
        "num_leaves": 63,
        "max_depth": 7,
        "min_child_samples": 40,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 789,
        "n_jobs": -1,
        "verbose": -1,
    },
    {
        "objective": "binary",
        "n_estimators": 5000,
        "learning_rate": 0.05,
        "num_leaves": 15,
        "max_depth": 5,
        "min_child_samples": 60,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 1024,
        "n_jobs": -1,
        "verbose": -1,
    },
]

# Train all 5 LightGBM models
models = []
best_iters = []

for i, config in enumerate(model_configs):
    print(f"\n--- Training LGBM Model {i+1}/5 (seed={config['random_state']}, lr={config['learning_rate']}, leaves={config['num_leaves']}) ---")
    model = lgb.LGBMClassifier(**config)

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="binary_logloss",
        callbacks=[
            lgb.early_stopping(stopping_rounds=200, verbose=False),
            lgb.log_evaluation(period=1000),
        ],
    )

    best_iter = model.best_iteration_
    val_pred_lgbm = model.predict_proba(X_val, num_iteration=best_iter)[:, 1]
    val_acc_lgbm = accuracy_score(y_val, (val_pred_lgbm >= 0.5).astype(int))
    print(f"LGBM Model {i+1}: best_iter={best_iter}, acc={val_acc_lgbm:.4f}")

    models.append(model)
    best_iters.append(best_iter)

print(f"\nAll 5 LightGBM models trained. Best iterations: {best_iters}")

# ============================================================
# 5b. TRAIN 3 XGBOOST MODELS
# ============================================================
xgb_configs = [
    {
        "n_estimators": 5000,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": -1,
        "use_label_encoder": False,
        "objective": "binary:logistic",
        "early_stopping_rounds": 200,
    },
    {
        "n_estimators": 5000,
        "max_depth": 8,
        "learning_rate": 0.03,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "reg_alpha": 0.5,
        "reg_lambda": 2.0,
        "random_state": 123,
        "n_jobs": -1,
        "use_label_encoder": False,
        "objective": "binary:logistic",
        "early_stopping_rounds": 200,
    },
    {
        "n_estimators": 5000,
        "max_depth": 10,
        "learning_rate": 0.05,
        "subsample": 0.85,
        "colsample_bytree": 0.75,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 456,
        "n_jobs": -1,
        "use_label_encoder": False,
        "objective": "binary:logistic",
        "early_stopping_rounds": 200,
    },
]

xgb_models = []
xgb_best_iters = []

for i, config in enumerate(xgb_configs):
    print(f"\n--- Training XGB Model {i+1}/3 (seed={config['random_state']}, lr={config['learning_rate']}, depth={config['max_depth']}) ---")
    model = xgb.XGBClassifier(**config)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    best_iter = model.best_iteration
    val_pred_xgb = model.predict_proba(X_val, iteration_range=(0, best_iter + 1))[:, 1]
    val_acc_xgb = accuracy_score(y_val, (val_pred_xgb >= 0.5).astype(int))
    print(f"XGB Model {i+1}: best_iter={best_iter}, acc={val_acc_xgb:.4f}")

    xgb_models.append(model)
    xgb_best_iters.append(best_iter)

print(f"\nAll 3 XGBoost models trained. Best iterations: {xgb_best_iters}")

# ============================================================
# 6. STACKING META-LEARNER + THRESHOLD OPTIMIZATION ON VALIDATION SET
# ============================================================

# Get base model predictions on validation set
val_lgbm_preds = np.column_stack([
    m.predict_proba(X_val, num_iteration=bi)[:, 1] for m, bi in zip(models, best_iters)
])
val_xgb_preds = np.column_stack([
    m.predict_proba(X_val, iteration_range=(0, bi + 1))[:, 1] for m, bi in zip(xgb_models, xgb_best_iters)
])
val_base_preds = np.hstack([val_lgbm_preds, val_xgb_preds])
print(f"Base model prediction matrix shape: {val_base_preds.shape}")

# Use simple average of base model probabilities as stacked prediction
# (avoids fitting a meta-learner on the same data it's evaluated on)
val_pred_proba = val_base_preds.mean(axis=1)

# Use standard 0.5 threshold for final validation score
val_pred_final = (val_pred_proba >= 0.5).astype(int)
final_val_accuracy = accuracy_score(y_val, val_pred_final)
print(f"\nFinal Validation Accuracy (threshold=0.5): {final_val_accuracy:.4f}")

# ============================================================
# 7. TEST PREDICTIONS (stacking, identical pipeline to validation)
# ============================================================
test_lgbm_preds = np.column_stack([
    m.predict_proba(X_test, num_iteration=bi)[:, 1] for m, bi in zip(models, best_iters)
])
test_xgb_preds = np.column_stack([
    m.predict_proba(X_test, iteration_range=(0, bi + 1))[:, 1] for m, bi in zip(xgb_models, xgb_best_iters)
])
test_base_preds = np.hstack([test_lgbm_preds, test_xgb_preds])

# Use simple average for stacked test predictions (same strategy as validation)
test_pred_proba = test_base_preds.mean(axis=1)
test_pred = (test_pred_proba >= 0.5).astype(int)

# Map back to original labels
label_map = {1: ">50K", 0: "<=50K"}
test_pred_labels = [label_map[p] for p in test_pred]
# Verify no NaN in predictions
assert not pd.isna(test_pred_labels).any(), "NaN in test predictions!"

# ============================================================
# 8. GENERATE SUBMISSION
# ============================================================
submission = pd.DataFrame(
    {"id": range(len(test_pred_labels)), "income": test_pred_labels}
)
submission.to_csv("./submission/submission.csv", index=False)
print(f"\nSubmission saved. Shape: {submission.shape}")
print(f"Submission preview:\n{submission.head(10)}")
print(f"Prediction distribution:\n{submission['income'].value_counts()}")

# ============================================================
# 9. FEATURE IMPORTANCE (averaged across ensemble)
# ============================================================
avg_importance = np.zeros(len(feature_cols))
for m in models:
    avg_importance += m.feature_importances_
avg_importance /= len(models)

fi = pd.DataFrame(
    {"feature": feature_cols, "importance": avg_importance}
).sort_values("importance", ascending=False)
print(f"\nTop 15 features by importance (ensemble avg):")
print(fi.head(15).to_string(index=False))

# ============================================================
# FINAL SCORE
# ============================================================
score = final_val_accuracy
print(f"Final Validation Score: {score}")