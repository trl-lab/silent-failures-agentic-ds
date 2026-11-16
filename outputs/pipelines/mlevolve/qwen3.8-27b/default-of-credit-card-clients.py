import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import os

# ============================================================
# 1. LOAD DATA
# ============================================================
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")

id_col = "ID"
target_col = "default payment next month"
feature_cols = [c for c in train_df.columns if c not in [id_col, target_col]]
print(f"Number of features: {len(feature_cols)}")

# ============================================================
# 2. SPLIT TRAIN INTO TRAIN/VAL (Stratified)
# ============================================================
X_all = train_df[feature_cols].values
y_all = train_df[target_col].values

X_train, X_val, y_train, y_val = train_test_split(
    X_all, y_all, test_size=0.15, random_state=42, stratify=y_all
)

print(f"Train split: {X_train.shape}, Val split: {X_val.shape}")
print(f"Target distribution - Train: {y_train.mean():.4f}, Val: {y_val.mean():.4f}")

# ============================================================
# 3. FEATURE ENGINEERING
# ============================================================


def engineer_features(df):
    """
    Domain-informed feature engineering for credit default prediction.
    Captures temporal dynamics, payment behavior patterns, and financial ratios.
    """
    feat = pd.DataFrame(index=df.index)

    # --- Raw features (keep all) ---
    for col in df.columns:
        feat[col] = df[col].values

    # --- PAYMENT STATUS FEATURES ---
    pay_status_cols = ["X6", "X7", "X8", "X9", "X10", "X11"]
    pay_values = df[pay_status_cols].values.astype(float)

    # --- Temporal trend features ---
    months_arr = np.arange(len(pay_status_cols))
    months_centered = (months_arr - months_arr.mean()).reshape(1, -1)
    feat["pay_trend"] = pay_values @ months_centered.T / (len(months_arr) - 1)

    # Recent payment status
    feat["pay_recent"] = df["X11"].values
    feat["pay_prev"] = df["X10"].values

    # --- Payment status derived features ---
    feat["n_late_payments"] = (pay_values >= 2).sum(axis=1)
    feat["n_no_debt"] = (pay_values == -1).sum(axis=1)
    feat["n_settled"] = (pay_values == -2).sum(axis=1)
    feat["n_paid_full"] = (pay_values == 0).sum(axis=1)

    pay_clean = pay_values.copy()
    pay_clean[pay_clean < 0] = 0
    feat["max_pay_status"] = pay_clean.max(axis=1)
    feat["mean_pay_status"] = (
        pay_valid.mean(axis=1) if False else pay_clean.mean(axis=1)
    )
    feat["pay_std"] = pay_values.std(axis=1)
    feat["pay_range"] = pay_values.max(axis=1) - pay_values.min(axis=1)

    # --- Sequential pattern features ---
    def max_consecutive_late(row):
        max_streak = 0
        current = 0
        for v in row:
            if v >= 2:
                current += 1
                max_streak = max(max_streak, current)
            else:
                current = 0
        return max_streak

    feat["max_consecutive_late"] = np.array(
        [max_consecutive_late(row) for row in pay_values]
    )
    feat["pay_deterioration"] = pay_values[:, -1] - pay_values[:, 0]
    feat["pay_improvement"] = pay_values[:, 0] - pay_values[:, -1]

    # --- BILL AMOUNT FEATURES ---
    bill_cols = ["X12", "X13", "X14", "X15", "X16", "X17"]
    bill_values = df[bill_cols].values.astype(float)

    bill_clean = bill_values.copy()
    bill_clean[bill_clean < 0] = 0

    feat["bill_total"] = bill_clean.sum(axis=1)
    feat["bill_mean"] = bill_clean.mean(axis=1)
    feat["bill_max"] = bill_clean.max(axis=1)
    feat["bill_std"] = bill_clean.std(axis=1)
    feat["bill_min"] = bill_clean.min(axis=1)
    feat["bill_trend"] = bill_clean @ months_centered.T / (len(months_arr) - 1)
    feat["bill_volatility"] = np.abs(np.diff(bill_clean, axis=1)).mean(axis=1)
    feat["bill_recent_vs_earlier"] = (bill_clean[:, -1] + 1) / (bill_clean[:, 0] + 1)
    feat["n_bill_zero"] = (bill_values == 0).sum(axis=1)
    feat["n_bill_no_card"] = (bill_values == -1).sum(axis=1)
    feat["n_bill_paid_off"] = (bill_values == -2).sum(axis=1)

    # --- PAYMENT AMOUNT FEATURES ---
    payamt_cols = ["X18", "X19", "X20", "X21", "X22", "X23"]
    payamt_values = df[payamt_cols].values.astype(float)

    payamt_clean = payamt_values.copy()
    payamt_clean[payamt_clean < 0] = 0

    feat["payamt_total"] = payamt_clean.sum(axis=1)
    feat["payamt_mean"] = payamt_clean.mean(axis=1)
    feat["payamt_max"] = payamt_clean.max(axis=1)
    feat["payamt_std"] = payamt_clean.std(axis=1)
    feat["payamt_trend"] = payamt_clean @ months_centered.T / (len(months_arr) - 1)

    # --- INTERACTION FEATURES: Bill vs Payment Amount ---
    feat["pay_to_bill_ratio"] = (payamt_clean.sum(axis=1) + 1) / (
        bill_clean.sum(axis=1) + 1
    )
    feat["unpaid_total"] = bill_clean.sum(axis=1) - payamt_clean.sum(axis=1)
    feat["unpaid_ratio"] = feat["unpaid_total"] / (bill_clean.sum(axis=1) + 1)

    # --- LIMIT_BAL INTERACTIONS ---
    limit_bal = df["X1"].values.astype(float)
    feat["bill_to_limit"] = bill_clean.sum(axis=1) / (limit_bal + 1)
    feat["bill_max_to_limit"] = bill_clean.max(axis=1) / (limit_bal + 1)
    feat["bill_mean_to_limit"] = bill_clean.mean(axis=1) / (limit_bal + 1)
    feat["payamt_to_limit"] = payamt_clean.sum(axis=1) / (limit_bal + 1)

    # --- LOG TRANSFORMS ---
    feat["log_limit_bal"] = np.log1p(limit_bal)
    feat["log_bill_total"] = np.log1p(bill_clean.sum(axis=1))
    feat["log_payamt_total"] = np.log1p(payamt_clean.sum(axis=1))
    feat["log_bill_max"] = np.log1p(bill_clean.max(axis=1))

    # --- AGE & DEMOGRAPHIC INTERACTIONS ---
    age = df["X5"].values.astype(float)
    feat["age_x_bill_to_limit"] = age * feat["bill_to_limit"]
    feat["age_x_pay_status"] = age * feat["mean_pay_status"]

    education = df["X3"].values.astype(float)
    marriage = df["X4"].values.astype(float)
    feat["edu_x_bill_to_limit"] = education * feat["bill_to_limit"]
    feat["marriage_x_pay_status"] = marriage * feat["mean_pay_status"]

    # --- PAYMENT STATUS × BILL INTERACTIONS ---
    feat["late_x_bill"] = feat["n_late_payments"] * feat["bill_to_limit"]
    feat["max_pay_x_bill"] = feat["max_pay_status"] * feat["bill_to_limit"]

    # --- SECOND-ORDER FEATURES ---
    feat["pay_status_sq"] = feat["mean_pay_status"] ** 2
    feat["bill_to_limit_sq"] = feat["bill_to_limit"] ** 2

    # --- RECENT MONTH FOCUS ---
    feat["pay_recent_3_mean"] = pay_values[:, -3:].mean(axis=1)
    feat["pay_recent_3_max"] = pay_clean[:, -3:].max(axis=1)
    feat["bill_recent_3_mean"] = bill_clean[:, -3:].mean(axis=1)

    # --- DIFFERENCES (month-over-month changes) ---
    feat["pay_diff_1"] = pay_values[:, -1] - pay_values[:, -2]
    feat["pay_diff_2"] = pay_values[:, -2] - pay_values[:, -3]
    feat["bill_diff_1"] = bill_clean[:, -1] - bill_clean[:, -2]
    feat["payamt_diff_1"] = payamt_clean[:, -1] - payamt_clean[:, -2]

    # --- COMPOSITE RISK SCORE ---
    feat["risk_score"] = (
        0.3 * feat["max_pay_status"]
        + 0.2 * feat["n_late_payments"]
        + 0.2 * feat["bill_to_limit"]
        + 0.15 * feat["unpaid_ratio"]
        + 0.15 * feat["pay_deterioration"]
    )

    return feat


print("Engineering features for train...")
train_feat = engineer_features(pd.DataFrame(X_train, columns=feature_cols))
print("Engineering features for val...")
val_feat = engineer_features(pd.DataFrame(X_val, columns=feature_cols))
print("Engineering features for test...")
test_feat = engineer_features(pd.DataFrame(test_df[feature_cols], columns=feature_cols))

print(f"Train features: {train_feat.shape}")
print(f"Val features: {val_feat.shape}")
print(f"Test features: {test_feat.shape}")


# ============================================================
# 4. CLEAN FEATURES
# ============================================================
def clean_features(df):
    df = df.replace([np.inf, -np.inf], 0)
    df = df.fillna(0)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return df


train_feat = clean_features(train_feat)
val_feat = clean_features(val_feat)
test_feat = clean_features(test_feat)

# Convert to numpy for model input
X_train_np = train_feat.values.astype(np.float32)
X_val_np = val_feat.values.astype(np.float32)
X_test_np = test_feat.values.astype(np.float32)

feature_names = list(train_feat.columns)
print(f"Total features after engineering: {len(feature_names)}")

# ============================================================
# 5. MODEL DESIGN: LightGBM Ensemble Configs
# ============================================================

# Compute scale_pos_weight safely
n_pos = int(np.sum(y_train == 1))
n_neg = int(np.sum(y_train == 0))
print(f"Positive samples: {n_pos}, Negative samples: {n_neg}")
scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
scale_pos_weight = max(scale_pos_weight, 0.01)  # safety guard
print(f"scale_pos_weight: {scale_pos_weight:.4f}")


def get_model_configs():
    """Return list of LightGBM parameter dicts for ensemble."""
    configs = [
        {
            "objective": "binary",
            "metric": "auc",
            "boosting_type": "gbdt",
            "num_leaves": 63,
            "max_depth": -1,
            "learning_rate": 0.03,
            "n_estimators": 2000,
            "subsample": 0.8,
            "subsample_freq": 1,
            "colsample_bytree": 0.7,
            "min_child_samples": 30,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "scale_pos_weight": scale_pos_weight,
            "random_state": 42,
            "verbose": -1,
            "n_jobs": -1,
        },
        {
            "objective": "binary",
            "metric": "auc",
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "max_depth": 6,
            "learning_rate": 0.05,
            "n_estimators": 2000,
            "subsample": 0.85,
            "subsample_freq": 1,
            "colsample_bytree": 0.8,
            "min_child_samples": 50,
            "reg_alpha": 0.5,
            "reg_lambda": 2.0,
            "scale_pos_weight": scale_pos_weight,
            "random_state": 123,
            "verbose": -1,
            "n_jobs": -1,
        },
        {
            "objective": "binary",
            "metric": "auc",
            "boosting_type": "gbdt",
            "num_leaves": 47,
            "max_depth": -1,
            "learning_rate": 0.04,
            "n_estimators": 2000,
            "subsample": 0.75,
            "subsample_freq": 1,
            "colsample_bytree": 0.65,
            "min_child_samples": 40,
            "reg_alpha": 0.2,
            "reg_lambda": 1.5,
            "scale_pos_weight": scale_pos_weight,
            "random_state": 7,
            "verbose": -1,
            "n_jobs": -1,
        },
    ]
    return configs


# ============================================================
# 6. TRAINING LOOP: LightGBM Ensemble with Early Stopping
# ============================================================
configs = get_model_configs()

models = []
val_preds_ensemble = np.zeros(len(X_val_np))
test_preds_ensemble = np.zeros(len(X_test_np))
model_val_aucs = []

for i, params in enumerate(configs):
    print(
        f"\n[Model {i+1}/{len(configs)}] num_leaves={params['num_leaves']}, lr={params['learning_rate']}, max_depth={params['max_depth']}"
    )

    model = lgb.LGBMClassifier(**params)

    model.fit(
        X_train_np,
        y_train,
        eval_set=[(X_val_np, y_val)],
        eval_metric="auc",
        callbacks=[
            lgb.early_stopping(stopping_rounds=100, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )

    best_iter = model.best_iteration_
    print(f"  Best iteration: {best_iter}")

    val_pred = model.predict_proba(X_val_np)[:, 1]
    val_auc = roc_auc_score(y_val, val_pred)
    model_val_aucs.append(val_auc)
    print(f"  Val AUC: {val_auc:.6f}")

    test_pred = model.predict_proba(X_test_np)[:, 1]

    val_preds_ensemble += val_pred
    test_preds_ensemble += test_pred
    models.append(model)

# Average ensemble predictions
n_models = len(models)
val_preds_ensemble /= n_models
test_preds_ensemble /= n_models

# ============================================================
# 7. EVALUATE ENSEMBLE ON VALIDATION SET
# ============================================================
ensemble_val_auc = roc_auc_score(y_val, val_preds_ensemble)
print(f"\n--- Ensemble Validation Results ---")
print(f"Individual model AUCs: {[f'{a:.6f}' for a in model_val_aucs]}")
print(f"Ensemble Validation AUC: {ensemble_val_auc:.6f}")

# ============================================================
# 8. FEATURE IMPORTANCE (top 15 from first model)
# ============================================================
print("\n--- Top 15 Feature Importances (Model 1) ---")
importances = models[0].feature_importances_
top_idx = np.argsort(importances)[::-1][:15]
for idx in top_idx:
    print(f"  {feature_names[idx]:30s} : {importances[idx]}")

# ============================================================
# 9. GENERATE SUBMISSION
# ============================================================
os.makedirs("./submission", exist_ok=True)

submission = pd.DataFrame(
    {
        "id": np.arange(len(test_preds_ensemble)),
        "default payment next month": test_preds_ensemble,
    }
)
submission.to_csv("./submission/submission.csv", index=False)
print(f"\nSubmission saved: {submission.shape}")
print(submission.head())

# ============================================================
# 10. FINAL SCORE
# ============================================================
score = ensemble_val_auc
print(f"Final Validation Score: {score}")
