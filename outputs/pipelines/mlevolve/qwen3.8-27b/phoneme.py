import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import lightgbm as lgb
import os

# Create directories
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# ============================================================
# STEP 1: Data Loading & Feature Engineering
# ============================================================

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Remap class labels: data uses 1-based (1,2), task specifies 0-based (0,1)
# Class 0: Nasals, Class 1: Orals
train_df["class"] = train_df["class"].astype(int) - 1

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")
print(f"Columns: {train_df.columns.tolist()}")
print(f"Target distribution:\n{train_df['class'].value_counts()}")

# Identify feature columns (everything except 'class')
feature_cols = [c for c in train_df.columns if c != "class"]
print(f"Feature columns: {feature_cols}")

print(f"Missing values in train: {train_df.isnull().sum().sum()}")
print(f"Missing values in test: {test_df.isnull().sum().sum()}")


# Feature Engineering Function
def engineer_features(df, feature_cols):
    """Create domain-informed features from the 5 harmonic amplitudes."""
    df = df.copy()

    # Get the 5 harmonic features
    h = [df[c] for c in feature_cols]
    h0, h1, h2, h3, h4 = h

    # --- Sign pattern features ---
    df["num_positive"] = (
        (h0 > 0).astype(int)
        + (h1 > 0).astype(int)
        + (h2 > 0).astype(int)
        + (h3 > 0).astype(int)
        + (h4 > 0).astype(int)
    )
    df["num_negative"] = 5 - df["num_positive"]
    df["sign_change_01"] = np.sign(h0) * np.sign(h1)
    df["sign_change_12"] = np.sign(h1) * np.sign(h2)
    df["sign_change_23"] = np.sign(h2) * np.sign(h3)
    df["sign_change_34"] = np.sign(h3) * np.sign(h4)
    df["total_sign_changes"] = (
        (df["sign_change_01"] < 0).astype(int)
        + (df["sign_change_12"] < 0).astype(int)
        + (df["sign_change_23"] < 0).astype(int)
        + (df["sign_change_34"] < 0).astype(int)
    )

    # --- Magnitude features ---
    abs_h = [np.abs(x) for x in h]
    df["abs_h0"] = abs_h[0]
    df["abs_h1"] = abs_h[1]
    df["abs_h2"] = abs_h[2]
    df["abs_h3"] = abs_h[3]
    df["abs_h4"] = abs_h[4]

    df["total_abs_energy"] = sum(abs_h)

    df["max_abs"] = np.maximum(
        np.maximum(np.maximum(abs_h[0], abs_h[1]), np.maximum(abs_h[2], abs_h[3])),
        abs_h[4],
    )
    df["min_abs"] = np.minimum(
        np.minimum(np.minimum(abs_h[0], abs_h[1]), np.minimum(abs_h[2], abs_h[3])),
        abs_h[4],
    )
    df["abs_range"] = df["max_abs"] - df["min_abs"]

    # --- Ratio features (energy distribution) ---
    total = df["total_abs_energy"].replace(0, 1e-10)
    df["share_h0"] = abs_h[0] / total
    df["share_h1"] = abs_h[1] / total
    df["share_h2"] = abs_h[2] / total
    df["share_h3"] = abs_h[3] / total
    df["share_h4"] = abs_h[4] / total

    df["ratio_h1_h0"] = abs_h[1] / (abs_h[0] + 1e-10)
    df["ratio_h2_h1"] = abs_h[2] / (abs_h[1] + 1e-10)
    df["ratio_h3_h2"] = abs_h[3] / (abs_h[2] + 1e-10)
    df["ratio_h4_h3"] = abs_h[4] / (abs_h[3] + 1e-10)

    df["ratio_h0_h1"] = abs_h[0] / (abs_h[1] + 1e-10)
    df["ratio_h2_h0"] = abs_h[2] / (abs_h[0] + 1e-10)
    df["ratio_h4_h0"] = abs_h[4] / (abs_h[0] + 1e-10)

    # --- Polynomial/interaction features ---
    df["h0_sq"] = h0**2
    df["h1_sq"] = h1**2
    df["h2_sq"] = h2**2
    df["h3_sq"] = h3**2
    df["h4_sq"] = h4**2

    df["h0_h1"] = h0 * h1
    df["h1_h2"] = h1 * h2
    df["h2_h3"] = h2 * h3
    df["h3_h4"] = h3 * h4

    df["h0_h2"] = h0 * h2
    df["h1_h3"] = h1 * h3
    df["h2_h4"] = h2 * h4

    # --- Aggregate statistics ---
    df["mean_abs"] = np.mean(abs_h, axis=0)
    df["std_abs"] = np.std(abs_h, axis=0)
    df["cv_abs"] = df["std_abs"] / (df["mean_abs"] + 1e-10)

    df["energy_asymmetry"] = (abs_h[0] + abs_h[4]) / (abs_h[1] + abs_h[3] + 1e-10)

    # --- Signed magnitude features ---
    df["signed_energy_h0"] = h0 * abs_h[0]
    df["signed_energy_h1"] = h1 * abs_h[1]
    df["signed_energy_h2"] = h2 * abs_h[2]
    df["signed_energy_h3"] = h3 * abs_h[3]
    df["signed_energy_h4"] = h4 * abs_h[4]

    # --- Log features ---
    df["log_abs_h0"] = np.log1p(abs_h[0])
    df["log_abs_h1"] = np.log1p(abs_h[1])
    df["log_abs_h2"] = np.log1p(abs_h[2])
    df["log_abs_h3"] = np.log1p(abs_h[3])
    df["log_abs_h4"] = np.log1p(abs_h[4])

    return df


# Apply feature engineering
train_fe = engineer_features(train_df, feature_cols)
test_fe = engineer_features(test_df, feature_cols)

# Define all feature columns (exclude 'class')
all_feature_cols = [c for c in train_fe.columns if c != "class"]
print(f"Total features after engineering: {len(all_feature_cols)}")

# ============================================================
# STEP 2: Train/Val Split & Scaling (fit scaler on train only)
# ============================================================

train_data, val_data = train_test_split(
    train_fe, test_size=0.2, random_state=42, stratify=train_fe["class"]
)

print(f"Train split: {train_data.shape[0]} samples")
print(f"Validation split: {val_data.shape[0]} samples")
print(f"Test: {test_fe.shape[0]} samples")

# Fit scaler ONLY on training data (prevent leakage)
scaler = StandardScaler()
X_train = scaler.fit_transform(train_data[all_feature_cols].values)
X_val = scaler.transform(val_data[all_feature_cols].values)
X_test = scaler.transform(test_fe[all_feature_cols].values)

y_train = train_data["class"].values
y_val = val_data["class"].values

print(f"Class distribution in train: {np.bincount(y_train)}")
print(f"Class distribution in val: {np.bincount(y_val)}")

# ============================================================
# STEP 3: Model Training & Evaluation
# ============================================================

print(f"\nTraining & Evaluation Step")
print(f"Train: {X_train.shape[0]} samples, {X_train.shape[1]} features")
print(f"Val:   {X_val.shape[0]} samples")
print(f"Test:  {X_test.shape[0]} samples")

model = lgb.LGBMClassifier(
    n_estimators=2000,
    learning_rate=0.02,
    num_leaves=31,
    max_depth=6,
    min_child_samples=15,
    subsample=0.85,
    subsample_freq=1,
    colsample_bytree=0.85,
    reg_alpha=0.05,
    reg_lambda=0.1,
    random_state=42,
    verbose=-1,
)

# Train with early stopping on validation accuracy
callbacks = [
    lgb.early_stopping(stopping_rounds=100, verbose=False),
    lgb.log_evaluation(period=-1),
]

model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    eval_metric="accuracy",
    callbacks=callbacks,
)

best_iter = model.best_iteration_
print(f"Best iteration: {best_iter}")

# --- Validation inference ---
val_pred = model.predict(X_val)
val_acc = accuracy_score(y_val, val_pred)
print(f"Validation Accuracy: {val_acc:.6f}")

# --- Test inference (identical processing logic) ---
test_pred = model.predict(X_test)

# --- Generate submission ---
test_ids = np.arange(len(test_pred))
submission = pd.DataFrame({"id": test_ids.astype(int), "class": test_pred.astype(int)})
print(f"Submission class distribution:\n{submission['class'].value_counts()}")
submission.to_csv("./submission/submission.csv", index=False)
print(f"Submission saved: {len(submission)} rows")
print(f"Test prediction distribution: {np.bincount(test_pred.astype(int))}")

score = val_acc
print(f"Final Validation Score: {score}")
