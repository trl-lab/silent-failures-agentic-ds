import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# 1. LOAD DATA
# ============================================================
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")
print(f"Target distribution:\n{train['cannabis'].value_counts()}")

# ============================================================
# 2. IDENTIFY COLUMNS
# ============================================================
target_col = "cannabis"
id_col = "id"

categorical_cols = ["gender", "education", "country", "ethnicity"]
continuous_cols = [
    "age",
    "nscore",
    "escore",
    "oscore",
    "ascore",
    "cscore",
    "impuslive",
    "ss",
]

for col in categorical_cols:
    print(f"{col}: {train[col].nunique()} unique values, dtype={train[col].dtype}")

# ============================================================
# 3. SPLIT FIRST (to prevent leakage)
# ============================================================
train_data, val_data = train_test_split(
    train, test_size=0.2, random_state=42, stratify=train[target_col]
)
print(f"\nTrain split: {train_data.shape}, Val split: {val_data.shape}")
print(f"Val target distribution:\n{val_data[target_col].value_counts()}")


# ============================================================
# 4. FEATURE ENGINEERING
# ============================================================
def engineer_features(df, freq_maps=None):
    """Apply feature engineering to dataframe."""
    df = df.copy()

    # Frequency encoding for categorical features
    if freq_maps is not None:
        for col in categorical_cols:
            if col in freq_maps:
                df[f"{col}_freq"] = df[col].map(freq_maps[col])
                df[f"{col}_freq"] = df[f"{col}_freq"].fillna(0)
    else:
        for col in categorical_cols:
            df[f"{col}_freq"] = 0

    # One-hot encode for low-cardinality categoricals
    for col in categorical_cols:
        if df[col].nunique() <= 10:
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
            df = pd.concat([df, dummies], axis=1)

    # Interaction features between Big Five personality traits
    df["n_x_imp"] = df["nscore"] * df["impuslive"]
    df["e_x_imp"] = df["escore"] * df["impuslive"]
    df["a_x_imp"] = df["ascore"] * df["impuslive"]
    df["c_x_imp"] = df["cscore"] * df["impuslive"]
    df["o_x_imp"] = df["oscore"] * df["impuslive"]

    # Big5 pairwise interactions
    df["n_x_e"] = df["nscore"] * df["escore"]
    df["n_x_o"] = df["nscore"] * df["oscore"]
    df["n_x_a"] = df["nscore"] * df["ascore"]
    df["n_x_c"] = df["nscore"] * df["cscore"]
    df["e_x_o"] = df["escore"] * df["oscore"]
    df["e_x_a"] = df["escore"] * df["ascore"]
    df["e_x_c"] = df["escore"] * df["cscore"]
    df["o_x_a"] = df["oscore"] * df["ascore"]
    df["o_x_c"] = df["oscore"] * df["cscore"]
    df["a_x_c"] = df["ascore"] * df["cscore"]

    # Social desirability interactions
    df["ss_x_n"] = df["ss"] * df["nscore"]
    df["ss_x_e"] = df["ss"] * df["escore"]
    df["ss_x_imp"] = df["ss"] * df["impuslive"]

    # Ratio features
    df["imp_over_ss"] = df["impuslive"] / (df["ss"] + 1e-8)
    df["n_over_c"] = df["nscore"] / (df["cscore"] + 1e-8)
    df["e_over_a"] = df["escore"] / (df["ascore"] + 1e-8)
    df["o_over_c"] = df["oscore"] / (df["cscore"] + 1e-8)

    # Squared terms
    df["n_sq"] = df["nscore"] ** 2
    df["e_sq"] = df["escore"] ** 2
    df["imp_sq"] = df["impuslive"] ** 2
    df["ss_sq"] = df["ss"] ** 2

    # Age interactions
    df["age_x_n"] = df["age"] * df["nscore"]
    df["age_x_imp"] = df["age"] * df["impuslive"]
    df["age_x_e"] = df["age"] * df["escore"]

    # Composite scores
    df["externalizing"] = df["nscore"] + df["impuslive"] - df["cscore"]
    df["internalizing"] = df["nscore"] + df["ascore"]
    df["conscientiousness_gap"] = df["cscore"] - df["impuslive"]

    return df


# Fit frequency maps on train_data only
freq_maps = {}
for col in categorical_cols:
    freq = train_data[col].value_counts(normalize=True).to_dict()
    freq_maps[col] = freq

# Apply feature engineering
train_data = engineer_features(train_data, freq_maps=freq_maps)
val_data = engineer_features(val_data, freq_maps=freq_maps)
test_data = engineer_features(test, freq_maps=freq_maps)

# Align columns: ensure val and test have the same columns as train
# (one-hot encoding may produce different dummy columns per split)
all_cols = train_data.columns.tolist()
val_data = val_data.reindex(columns=all_cols, fill_value=0)
test_data = test_data.reindex(columns=all_cols, fill_value=0)

# Identify feature columns
exclude_cols = [id_col, target_col] + categorical_cols
feature_cols = [c for c in train_data.columns if c not in exclude_cols]
print(f"\nNumber of features: {len(feature_cols)}")

# Standardize (fit on train only)
scaler = StandardScaler()
X_train = scaler.fit_transform(train_data[feature_cols].values)
X_val = scaler.transform(val_data[feature_cols].values)
X_test = scaler.transform(test_data[feature_cols].values)

# Prepare targets
y_train = train_data[target_col].values
y_val = val_data[target_col].values

# Test IDs
test_ids = (
    test_data[id_col].values
    if id_col in test_data.columns
    else np.arange(len(test_data))
)

print(f"X_train shape: {X_train.shape}")
print(f"X_val shape: {X_val.shape}")
print(f"X_test shape: {X_test.shape}")


# ============================================================
# 5. MODEL DEFINITION
# ============================================================
# Ensemble of LightGBM and XGBoost classifiers
# LightGBM: n_estimators=2000, learning_rate=0.01, num_leaves=15,
#           min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
#           reg_alpha=0.1, reg_lambda=1.0, early stopping on val accuracy
# XGBoost:  n_estimators=2000, learning_rate=0.01, max_depth=4,
#           min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
#           reg_alpha=0.1, reg_lambda=1.0, early stopping
# Ensemble: soft voting (average of predicted probabilities)


# ============================================================
# 6. TRAINING SETUP
# ============================================================
# Encode targets as integers for tree-based models
classes = np.unique(y_train)
class_to_idx = {c: i for i, c in enumerate(classes)}
idx_to_class = {i: c for c, i in class_to_idx.items()}
num_classes = len(classes)

y_train_enc = np.array([class_to_idx[c] for c in y_train])
y_val_enc = np.array([class_to_idx[c] for c in y_val])

print(f"\nNumber of classes: {num_classes}")
print(f"Class distribution (train): {np.bincount(y_train_enc, minlength=num_classes)}")
print(f"Class distribution (val): {np.bincount(y_val_enc, minlength=num_classes)}")

print(f"Input dim: {X_train.shape[1]}, Classes: {num_classes}")

# ============================================================
# 7. TRAINING LIGHTGBM AND XGBOOST WITH EARLY STOPPING
# ============================================================
# --- LightGBM ---
lgb_model = lgb.LGBMClassifier(
    n_estimators=2000,
    learning_rate=0.01,
    num_leaves=15,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)

lgb_model.fit(
    X_train,
    y_train_enc,
    eval_set=[(X_val, y_val_enc)],
    eval_metric="multi_logloss",
    callbacks=[lgb.early_stopping(50, verbose=False)],
)
print(f"LightGBM best iteration: {lgb_model.best_iteration_}")

# --- XGBoost ---
xgb_model = xgb.XGBClassifier(
    n_estimators=2000,
    learning_rate=0.01,
    max_depth=4,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    eval_metric="mlogloss",
    early_stopping_rounds=50,
)

xgb_model.fit(
    X_train,
    y_train_enc,
    eval_set=[(X_val, y_val_enc)],
    verbose=False,
)
print(f"XGBoost best iteration: {xgb_model.best_iteration}")

# ============================================================
# 8. ENSEMBLE PREDICTIONS VIA PROBABILITY AVERAGING
# ============================================================
# Get predicted probabilities from each model
lgb_val_probs = lgb_model.predict_proba(X_val)
xgb_val_probs = xgb_model.predict_proba(X_val)

# Combine via soft voting (average of predicted probabilities)
val_probs = 0.5 * lgb_val_probs + 0.5 * xgb_val_probs
val_preds_idx = val_probs.argmax(axis=1)
val_preds = np.array([idx_to_class[i] for i in val_preds_idx])

# Compute final validation score (accuracy - the official metric)
score = accuracy_score(y_val, val_preds)
print(f"Validation Accuracy: {score:.4f}")
print(
    f"Val predictions distribution: {np.bincount(val_preds_idx, minlength=num_classes)}"
)

# Test predictions (identical ensemble logic)
lgb_test_probs = lgb_model.predict_proba(X_test)
xgb_test_probs = xgb_model.predict_proba(X_test)
test_probs = 0.5 * lgb_test_probs + 0.5 * xgb_test_probs
test_preds_idx = test_probs.argmax(axis=1)
test_preds = np.array([idx_to_class[i] for i in test_preds_idx])

# ============================================================
# 9. SAVE SUBMISSION
# ============================================================
os.makedirs("./submission", exist_ok=True)
submission = pd.DataFrame({"id": test_ids.astype(int), "cannabis": test_preds})
submission.to_csv("./submission/submission.csv", index=False)
print(f"\nSubmission saved with {len(submission)} rows")
print(f"Submission target distribution:\n{submission['cannabis'].value_counts()}")

print(f"Final Validation Score: {score}")
