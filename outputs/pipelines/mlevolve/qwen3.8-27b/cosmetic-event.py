import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OrdinalEncoder
from sklearn.metrics import accuracy_score
import lightgbm as lgb
import os
import pickle

# Create working directory
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# ============================================================
# 1. LOAD DATA
# ============================================================
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")
print(f"Target distribution:\n{train['outcomes'].value_counts()}")
print(f"Target classes: {train['outcomes'].nunique()}")

# ============================================================
# 2. DATA CLEANING & EXPLORATION
# ============================================================
target = train["outcomes"].copy()
train = train.drop("outcomes", axis=1)

numeric_cols = train.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = train.select_dtypes(include=["object", "category"]).columns.tolist()
bool_cols = train.select_dtypes(include=["bool"]).columns.tolist()

print(f"Numeric features ({len(numeric_cols)}): {numeric_cols}")
print(f"Categorical features ({len(categorical_cols)}): {categorical_cols}")
print(f"Boolean features ({len(bool_cols)}): {bool_cols}")

# ============================================================
# 3. TRAIN/VALIDATION SPLIT (stratified) - DONE FIRST
# ============================================================
X_full, X_val_raw, y_full, y_val = train_test_split(
    train, target, test_size=0.15, random_state=42, stratify=target
)

print(f"Full train shape: {X_full.shape}")
print(f"Val raw shape: {X_val_raw.shape}")
print(f"Test shape: {test.shape}")

# Re-identify column types on the split
numeric_cols = X_full.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = X_full.select_dtypes(include=["object", "category"]).columns.tolist()
bool_cols = X_full.select_dtypes(include=["bool"]).columns.tolist()

# ============================================================
# 4. FEATURE ENGINEERING (fit on X_full only, apply to X_val_raw and test)
# ============================================================
def engineer_features(X_fit, X_apply_list):
    """Fit encoders/stats on X_fit, transform X_fit and each X in X_apply_list."""
    X_fit = X_fit.copy()
    results = []

    # Handle boolean columns
    for col in bool_cols:
        if col in X_fit.columns:
            X_fit[col] = X_fit[col].astype(int)
        for i, X_a in enumerate(X_apply_list):
            X_apply_list[i][col] = X_apply_list[i][col].astype(int)

    # Handle categorical columns
    high_card_threshold = 15
    for col in categorical_cols:
        if col not in X_fit.columns:
            continue
        n_unique_fit = X_fit[col].nunique()
        max_unique = n_unique_fit  # use fit set cardinality as threshold basis

        if max_unique <= high_card_threshold:
            oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            X_fit[col] = oe.fit_transform(X_fit[[col]])
            for i, X_a in enumerate(X_apply_list):
                X_apply_list[i][col] = oe.transform(X_a[[col]])
        else:
            freq_map = X_fit[col].value_counts(normalize=True).to_dict()
            X_fit[col] = X_fit[col].map(freq_map).fillna(0)
            for i, X_a in enumerate(X_apply_list):
                X_apply_list[i][col] = X_a[col].map(freq_map).fillna(0)

    # Convert any remaining object columns
    for col in X_fit.columns:
        if X_fit[col].dtype == "object" or X_fit[col].dtype == "category":
            le = LabelEncoder()
            X_fit[col] = le.fit_transform(X_fit[col].astype(str))
            for i, X_a in enumerate(X_apply_list):
                classes_set = set(le.classes_)
                X_apply_list[i][col] = (
                    X_a[col].astype(str).map(lambda x: le.transform([x])[0] if x in classes_set else -1)
                )

    # Update numeric_cols list
    current_numeric = X_fit.select_dtypes(include=[np.number]).columns.tolist()

    # Numerical interaction features
    if len(current_numeric) >= 2:
        numeric_subset = [c for c in current_numeric if X_fit[c].std() > 0]
        if len(numeric_subset) > 1:
            corr_matrix = X_fit[numeric_subset].corr().abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            top_pairs = upper.stack().nlargest(min(5, len(upper.stack()))).index.tolist()

            for col1, col2 in top_pairs:
                if col1 in X_fit.columns and col2 in X_fit.columns:
                    new_col = f"{col1}_x_{col2}"
                    X_fit[new_col] = X_fit[col1] * X_fit[col2]
                    for i, X_a in enumerate(X_apply_list):
                        X_apply_list[i][new_col] = X_a[col1] * X_a[col2]
                    current_numeric.append(new_col)

    # Aggregate features
    if len(current_numeric) > 3:
        X_fit["row_mean"] = X_fit[current_numeric].mean(axis=1)
        X_fit["row_std"] = X_fit[current_numeric].std(axis=1).fillna(0)
        X_fit["row_max"] = X_fit[current_numeric].max(axis=1)
        X_fit["row_min"] = X_fit[current_numeric].min(axis=1)
        X_fit["row_range"] = X_fit["row_max"] - X_fit["row_min"]
        for i, X_a in enumerate(X_apply_list):
            X_a["row_mean"] = X_a[current_numeric].mean(axis=1)
            X_a["row_std"] = X_a[current_numeric].std(axis=1).fillna(0)
            X_a["row_max"] = X_a[current_numeric].max(axis=1)
            X_a["row_min"] = X_a[current_numeric].min(axis=1)
            X_a["row_range"] = X_a["row_max"] - X_a["row_min"]
        current_numeric.extend(["row_mean", "row_std", "row_max", "row_min", "row_range"])

    # Handle remaining missing values (fit stats on X_fit only)
    for col in X_fit.columns:
        if X_fit[col].isnull().any():
            if X_fit[col].dtype in ["float64", "int64", "int32", "float32"]:
                median_val = X_fit[col].median()
                X_fit[col] = X_fit[col].fillna(median_val)
                for i, X_a in enumerate(X_apply_list):
                    X_apply_list[i][col] = X_a[col].fillna(median_val)
            else:
                mode_val = X_fit[col].mode()[0] if not X_fit[col].mode().empty else 0
                X_fit[col] = X_fit[col].fillna(mode_val)
                for i, X_a in enumerate(X_apply_list):
                    X_apply_list[i][col] = X_a[col].fillna(mode_val)

    return X_fit, current_numeric


X_val_raw = X_val_raw.copy()
test = test.copy()
X_full, numeric_cols = engineer_features(X_full, [X_val_raw, test])

# ============================================================
# 5. FEATURE SCALING (fit on X_full only)
# ============================================================
final_features = X_full.columns.tolist()
print(f"Total features after engineering: {len(final_features)}")

scaler = StandardScaler()
X_train = pd.DataFrame(
    scaler.fit_transform(X_full[final_features]),
    columns=final_features,
    index=X_full.index,
)
X_val = pd.DataFrame(
    scaler.transform(X_val_raw[final_features]), columns=final_features, index=X_val_raw.index
)
test_scaled = pd.DataFrame(
    scaler.transform(test[final_features]), columns=final_features, index=test.index
)

y_train = y_full

print(f"Train split shape: {X_train.shape}")
print(f"Validation split shape: {X_val.shape}")
print(f"Test shape: {test_scaled.shape}")

# ============================================================
# 6. MODEL DESIGN - LightGBM
# ============================================================
n_classes = y_train.nunique()

lgb_params = {
    "objective": "multiclass",
    "num_class": n_classes,
    "metric": "multi_logloss",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "max_depth": -1,
    "min_child_samples": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "is_unbalance": True,
    "n_jobs": -1,
    "verbose": -1,
    "random_state": 42,
}

model = lgb.LGBMClassifier(**lgb_params)

# ============================================================
# 7. TRAINING WITH EARLY STOPPING & METRIC TRACKING
# ============================================================
training_history = {"val_logloss": [], "val_accuracy": []}


def record_evaluation(env):
    if env.evaluation_result_list:
        for result in env.evaluation_result_list:
            # result is a tuple: (name, value, iter_no, is_higher_better, extra_info)
            if "multi_logloss" in result[0]:
                training_history["val_logloss"].append(result[1])


callbacks = [
    lgb.early_stopping(stopping_rounds=100, verbose=False),
    lgb.log_evaluation(period=50),
    record_evaluation,
]

model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    eval_metric="multi_logloss",
    callbacks=callbacks,
)

best_iteration = model.best_iteration_
print(f"Best iteration: {best_iteration}")

# ============================================================
# 8. VALIDATION EVALUATION (using best iteration)
# ============================================================
val_pred_proba = model.predict_proba(X_val, num_iteration=best_iteration)
val_pred = np.argmax(val_pred_proba, axis=1)
val_pred_labels = model.classes_[val_pred]

val_accuracy = accuracy_score(y_val, val_pred_labels)
print(f"Validation Accuracy (official metric): {val_accuracy:.4f}")

if training_history["val_logloss"]:
    best_logloss = min(training_history["val_logloss"])
    best_logloss_iter = training_history["val_logloss"].index(best_logloss) + 1
    print(f"Best val logloss: {best_logloss:.4f} at iteration {best_logloss_iter}")

# ============================================================
# 9. TEST INFERENCE & SUBMISSION (using best iteration)
# ============================================================
test_pred_proba = model.predict_proba(test_scaled, num_iteration=best_iteration)
test_pred = np.argmax(test_pred_proba, axis=1)
test_pred_labels = model.classes_[test_pred]

submission = pd.DataFrame(
    {"id": range(len(test_pred_labels)), "outcomes": test_pred_labels}
)
submission.to_csv("./submission/submission.csv", index=False)
print(f"Submission saved: {submission.shape}")
print(f"Prediction distribution:\n{submission['outcomes'].value_counts().to_string()}")

# ============================================================
# 10. SAVE MODEL
# ============================================================
with open("./working/model_lgbm_best.pkl", "wb") as f:
    pickle.dump(model, f)

print(f"\nModel: LightGBM (best_iter={best_iteration})")
print(f"Val Accuracy: {val_accuracy:.4f}")

print(f"Final Validation Score: {val_accuracy:.4f}")