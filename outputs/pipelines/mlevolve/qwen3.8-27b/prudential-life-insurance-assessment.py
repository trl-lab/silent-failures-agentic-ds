import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss
import lightgbm as lgb
import xgboost as xgb
import os


# ============================================================
# 1. QWK METRIC (Official competition metric)
# ============================================================
def quadratic_weighted_kappa(y_true, y_pred, num_classes=8):
    """Official QWK metric implementation.
    y_true is expected to be 1-indexed (1..num_classes).
    y_pred is expected to be 0-indexed (0..num_classes-1) from np.digitize.
    """
    y_true = np.array(y_true).astype(int) - 1  # convert 1..8 to 0..7
    y_pred = np.array(y_pred).astype(int)
    y_true = np.clip(y_true, 0, num_classes - 1)
    y_pred = np.clip(y_pred, 0, num_classes - 1)

    C = np.zeros((num_classes, num_classes))
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            C[t, p] += 1

    W = np.zeros((num_classes, num_classes))
    for i in range(num_classes):
        for j in range(num_classes):
            W[i, j] = (i - j) ** 2 / ((num_classes - 1) ** 2)

    row_sums = C.sum(axis=1)
    col_sums = C.sum(axis=0)
    total = C.sum()
    E = np.outer(row_sums, col_sums) / total

    denom = (W * E).sum()
    if denom == 0:
        return 0.0
    kappa = 1.0 - (W * C).sum() / denom
    return kappa


def find_optimal_bins(scores, y_true, num_classes=8, n_iterations=80):
    """Find QWK-optimal bin boundaries via iterative local search."""
    percentiles = np.linspace(0, 100, num_classes + 1)
    boundaries = np.percentile(scores, percentiles[1:-1]).copy()

    for i in range(1, len(boundaries)):
        if boundaries[i] <= boundaries[i - 1]:
            boundaries[i] = boundaries[i - 1] + 1e-6

    best_qwk = quadratic_weighted_kappa(y_true, np.digitize(scores, boundaries))
    best_boundaries = boundaries.copy()

    score_range = np.max(scores) - np.min(scores)
    if score_range == 0:
        return best_boundaries, best_qwk

    step_size = 0.01 * score_range

    for iteration in range(n_iterations):
        improved = False
        for i in range(len(best_boundaries)):
            for delta in [-step_size, step_size]:
                new_bounds = best_boundaries.copy()
                new_bounds[i] += delta
                if np.all(np.diff(new_bounds) > 1e-8):
                    y_pred_new = np.digitize(scores, new_bounds)
                    qwk_new = quadratic_weighted_kappa(y_true, y_pred_new)
                    if qwk_new > best_qwk:
                        best_qwk = qwk_new
                        best_boundaries = new_bounds
                        improved = True
        if not improved:
            step_size *= 0.5
            if step_size < 1e-6 * score_range:
                break

    return best_boundaries, best_qwk


# ============================================================
# 2. LOAD DATA
# ============================================================
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")
print(f"Target distribution:\n{train['Response'].value_counts().sort_index()}")

# ============================================================
# 3. SPLIT TRAIN INTO TRAIN/VAL (before any feature engineering)
# ============================================================
train_data, val_data = train_test_split(
    train, test_size=0.15, random_state=42, stratify=train["Response"]
)

# Drop Id
if "Id" in train_data.columns:
    train_data = train_data.drop("Id", axis=1)
if "Id" in val_data.columns:
    val_data = val_data.drop("Id", axis=1)
if "Id" in test.columns:
    test = test.drop("Id", axis=1)

target_col = "Response"
y_train_full = train_data[target_col].values
y_val = val_data[target_col].values

train_data = train_data.drop(target_col, axis=1)
val_data = val_data.drop(target_col, axis=1)


# ============================================================
# 4. FEATURE ENGINEERING (compact, domain-aware)
# ============================================================
def engineer_features(df):
    df = df.copy()

    if "BMI" in df.columns and "Ins_Age" in df.columns:
        df["BMI_Age"] = df["BMI"] * df["Ins_Age"]
        df["BMI_sq"] = df["BMI"] ** 2
        df["Age_sq"] = df["Ins_Age"] ** 2
        df["Age_log"] = np.log1p(df["Ins_Age"].clip(lower=0))

    if "Ht" in df.columns and "Wt" in df.columns:
        df["Ht_Wt_ratio"] = df["Wt"] / (df["Ht"] + 1e-6)
        df["Ht_sq"] = df["Ht"] ** 2
        df["Wt_sq"] = df["Wt"] ** 2

    if "BMI" in df.columns:
        df["BMI_bin"] = pd.cut(df["BMI"], bins=[0, 18.5, 25, 30, 35, 50], labels=False)
        df["BMI_over_30"] = (df["BMI"] > 30).astype(int)

    if "Ins_Age" in df.columns:
        df["Age_bin"] = pd.cut(
            df["Ins_Age"], bins=[0, 30, 40, 50, 60, 70, 100], labels=False
        )

    top_numeric = [c for c in ["BMI", "Ins_Age", "Ht", "Wt"] if c in df.columns]
    for i in range(len(top_numeric)):
        for j in range(i + 1, len(top_numeric)):
            df[f"{top_numeric[i]}_x_{top_numeric[j]}"] = (
                df[top_numeric[i]] * df[top_numeric[j]]
            )

    # One-hot for low-cardinality categoricals
    for col in df.columns:
        if df[col].dtype == object:
            nunique = df[col].nunique()
            if nunique <= 15:
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                df = pd.concat([df, dummies], axis=1)

    return df


train_data = engineer_features(train_data)
val_data = engineer_features(val_data)
test = engineer_features(test)

# Align columns
all_cols = train_data.columns.tolist()
val_data = val_data.reindex(columns=all_cols, fill_value=0)
test = test.reindex(columns=all_cols, fill_value=0)

# ============================================================
# 5. HANDLE MISSING VALUES & ENCODE (fit on train only)
# ============================================================
for col in train_data.columns:
    if train_data[col].isna().any():
        fill_val = (
            train_data[col].median()
            if pd.api.types.is_numeric_dtype(train_data[col])
            else 0
        )
        train_data[col] = train_data[col].fillna(fill_val)
        val_data[col] = val_data[col].fillna(fill_val)
        test[col] = test[col].fillna(fill_val)

# Convert all to numeric
for col in train_data.columns:
    if train_data[col].dtype == object:
        train_data[col] = pd.to_numeric(train_data[col], errors="coerce").fillna(0)
        val_data[col] = pd.to_numeric(val_data[col], errors="coerce").fillna(0)
        test[col] = pd.to_numeric(test[col], errors="coerce").fillna(0)
    elif train_data[col].dtype == "category":
        train_data[col] = train_data[col].cat.codes.astype(float)
        val_data[col] = val_data[col].cat.codes.astype(float)
        test[col] = test[col].cat.codes.astype(float)

# ============================================================
# 6. SCALE (fit on train only)
# ============================================================
scaler = StandardScaler()
X_train = scaler.fit_transform(train_data.values.astype(np.float32))
X_val = scaler.transform(val_data.values.astype(np.float32))
X_test = scaler.transform(test.values.astype(np.float32))

X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)
X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

print(f"Features: {X_train.shape[1]}")


# ============================================================
# 7. MODEL DESIGN - LightGBM + XGBoost Ensemble (8-class)
# ============================================================
# Convert targets to 0-indexed for classifiers
y_train_0 = y_train_full.astype(int) - 1  # 0..7
y_val_0 = y_val.astype(int) - 1  # 0..7

# LightGBM parameters
lgb_params = {
    'objective': 'multiclass',
    'num_class': 8,
    'metric': 'multi_logloss',
    'num_leaves': 63,
    'learning_rate': 0.05,
    'n_estimators': 2000,
    'class_weight': 'balanced',
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_samples': 20,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1,
}

# XGBoost parameters
xgb_params = {
    'objective': 'multi:softprob',
    'num_class': 8,
    'num_leaves': 63,
    'learning_rate': 0.05,
    'n_estimators': 2000,
    'class_weight': 'balanced',
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 20,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'random_state': 42,
    'n_jobs': -1,
    'tree_method': 'hist',
    'eval_metric': 'mlogloss',
}

print("--- Training LightGBM ---")
lgb_model = lgb.LGBMClassifier(**lgb_params)
lgb_model.fit(
    X_train, y_train_0,
    eval_set=[(X_val, y_val_0)],
    callbacks=[lgb.early_stopping(100, verbose=True), lgb.log_evaluation(200)]
)
lgb_best_iter = lgb_model.best_iteration_
print(f"LightGBM best iteration: {lgb_best_iter}")

print("\n--- Training XGBoost ---")
xgb_model = xgb.XGBClassifier(early_stopping_rounds=100, **xgb_params)
xgb_model.fit(
    X_train, y_train_0,
    eval_set=[(X_val, y_val_0)],
    verbose=200,
)
xgb_best_iter = xgb_model.best_iteration
print(f"XGBoost best iteration: {xgb_best_iter}")

# ============================================================
# 8. ENSEMBLE PREDICTIONS ON VALIDATION
# ============================================================
# Get probability predictions from both models
lgb_val_probs = lgb_model.predict_proba(X_val)  # shape (n_val, 8)
xgb_val_probs = xgb_model.predict_proba(X_val)  # shape (n_val, 8)

# Probability-weighted ensemble: average predicted probabilities
ensemble_val_probs = 0.5 * lgb_val_probs + 0.5 * xgb_val_probs

# Use expected class score as the continuous score for bin optimization
# Score = sum(p_i * i) for i in 0..7
class_scores = np.arange(8)
ensemble_val_score = ensemble_val_probs @ class_scores

val_boundaries, val_qwk = find_optimal_bins(ensemble_val_score, y_val)
print(f"Ensemble Val QWK: {val_qwk:.4f}")

best_val_qwk = val_qwk

# ============================================================
# 10. FINAL EVALUATION
# ============================================================
val_preds_final = np.digitize(ensemble_val_score, val_boundaries)
val_qwk_check = quadratic_weighted_kappa(y_val, val_preds_final)
print(f"Validation QWK (final): {val_qwk_check:.4f}")

# ============================================================
# 11. TEST PREDICTIONS
# ============================================================
# Get probability predictions from both models on test set
lgb_test_probs = lgb_model.predict_proba(X_test)  # shape (n_test, 8)
xgb_test_probs = xgb_model.predict_proba(X_test)  # shape (n_test, 8)

# Probability-weighted ensemble
ensemble_test_probs = 0.5 * lgb_test_probs + 0.5 * xgb_test_probs

# Use expected class score as the continuous score
ensemble_test_score = ensemble_test_probs @ class_scores

test_preds = np.digitize(ensemble_test_score, val_boundaries) + 1  # convert 0..7 to 1..8
test_preds = np.clip(test_preds, 1, 8).astype(int)

print(f"Test prediction distribution: {np.bincount(test_preds, minlength=8)}")

# ============================================================
# 12. SAVE SUBMISSION
# ============================================================
os.makedirs("./submission", exist_ok=True)
submission = pd.DataFrame({"id": range(len(test_preds)), "Response": test_preds})
submission.to_csv("./submission/submission.csv", index=False)
print(f"Submission saved: {submission.shape}")

score = val_qwk_check
print(f"Final Validation Score: {score}")