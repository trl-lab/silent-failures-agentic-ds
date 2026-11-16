import pandas as pd
import numpy as np
import lightgbm as lgb
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from collections import Counter

# Create directories
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# ============================================================
# 1. LOAD DATA
# ============================================================
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")
print(f"\nTrain columns: {list(train.columns)}")
print(f"\nTrain dtypes:\n{train.dtypes}")
print(f"\nTarget distribution:\n{train['classification'].value_counts()}")
print(f"\nTarget nunique: {train['classification'].nunique()}")
print(f"\nMissing values (train):\n{train.isnull().sum()[train.isnull().sum() > 0]}")
print(f"\nMissing values (test):\n{test.isnull().sum()[test.isnull().sum() > 0]}")

# ============================================================
# 2. SEPARATE TARGET
# ============================================================
target_col = "classification"
y = train[target_col].copy()
train.drop(columns=[target_col], inplace=True)

# Identify if there's an 'id' column to preserve
id_col = None
if "id" in train.columns:
    id_col = "id"
    train_ids = train["id"].copy()
    test_ids = test["id"].copy()
    train.drop(columns=["id"], inplace=True)
    test.drop(columns=["id"], inplace=True)

# ============================================================
# 3. CREATE TRAIN/VALIDATION SPLIT (Stratified)
# ============================================================
X_train, X_val, y_train, y_val = train_test_split(
    train, y, test_size=0.15, random_state=42, stratify=y
)

print(f"\nX_train shape: {X_train.shape}")
print(f"X_val shape: {X_val.shape}")
print(f"X_test shape: {test.shape}")
print(f"\nVal target distribution:\n{y_val.value_counts()}")

# ============================================================
# 4. FEATURE ENGINEERING
# ============================================================
# Identify column types
numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = X_train.select_dtypes(
    include=["object", "category"]
).columns.tolist()
bool_cols = X_train.select_dtypes(include=["bool"]).columns.tolist()

# Handle boolean columns as numeric
for col in bool_cols:
    if col in X_train.columns:
        X_train[col] = X_train[col].astype(int)
    if col in X_val.columns:
        X_val[col] = X_val[col].astype(int)
    if col in test.columns:
        test[col] = test[col].astype(int)
    if col in categorical_cols:
        categorical_cols.remove(col)
    if col not in numeric_cols:
        numeric_cols.append(col)

print(f"\nNumeric columns ({len(numeric_cols)}): {numeric_cols[:10]}...")
print(f"Categorical columns ({len(categorical_cols)}): {categorical_cols[:10]}...")

# --- Handle Categorical Columns ---
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    combined = pd.concat([X_train[col], X_val[col]], axis=0)
    le.fit(combined)

    X_train[col] = le.transform(X_train[col])
    X_val[col] = le.transform(X_val[col])

    # For test, handle unseen categories
    test[col] = test[col].map(lambda x: x if x in le.classes_ else "__UNSEEN__")
    if "__UNSEEN__" not in le.classes_:
        le.fit(pd.concat([combined, pd.Series(["__UNSEEN__"])], axis=0))
    test[col] = le.transform(test[col])

    label_encoders[col] = le
    numeric_cols.append(col)
    categorical_cols.remove(col)

# --- Handle Missing Values ---
medians = {}
for col in numeric_cols:
    if col in X_train.columns:
        med = X_train[col].median()
        medians[col] = med
        X_train[col] = X_train[col].fillna(med)
    if col in X_val.columns:
        X_val[col] = X_val[col].fillna(medians.get(col, 0))
    if col in test.columns:
        test[col] = test[col].fillna(medians.get(col, 0))

# --- Scale Numeric Features ---
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train[numeric_cols]),
    columns=numeric_cols,
    index=X_train.index,
)
X_val_scaled = pd.DataFrame(
    scaler.transform(X_val[numeric_cols]), columns=numeric_cols, index=X_val.index
)
test_scaled = pd.DataFrame(
    scaler.transform(test[numeric_cols]), columns=numeric_cols, index=test.index
)

# Replace any inf values
X_train_scaled = X_train_scaled.replace([np.inf, -np.inf], 0).fillna(0)
X_val_scaled = X_val_scaled.replace([np.inf, -np.inf], 0).fillna(0)
test_scaled = test_scaled.replace([np.inf, -np.inf], 0).fillna(0)


# --- Additional Feature Engineering: Row-wise meta-features ---
def add_meta_features(df):
    df = df.copy()
    df["row_mean"] = df.mean(axis=1)
    df["row_std"] = df.std(axis=1)
    df["row_max"] = df.max(axis=1)
    df["row_min"] = df.min(axis=1)
    df["row_range"] = df["row_max"] - df["row_min"]
    df["row_skew"] = df.skew(axis=1)
    return df


X_train_scaled = add_meta_features(X_train_scaled)
X_val_scaled = add_meta_features(X_val_scaled)
test_scaled = add_meta_features(test_scaled)

print(f"\nFinal feature shapes:")
print(f"  X_train: {X_train_scaled.shape}")
print(f"  X_val: {X_val_scaled.shape}")
print(f"  X_test: {test_scaled.shape}")

# ============================================================
# 5. LABEL ENCODE TARGET
# ============================================================
target_le = LabelEncoder()
y_train_enc = target_le.fit_transform(y_train)
y_val_enc = target_le.transform(y_val)

print(f"\nTarget classes: {target_le.classes_}")
print(f"Target shape: {y_train_enc.shape}")

# ============================================================
# 6. SAVE PROCESSED DATA
# ============================================================
X_train_scaled.to_parquet("./working/X_train.parquet")
X_val_scaled.to_parquet("./working/X_val.parquet")
test_scaled.to_parquet("./working/X_test.parquet")

np.save("./working/y_train.npy", y_train_enc)
np.save("./working/y_val.npy", y_val_enc)

metadata = {
    "target_classes": target_le.classes_.tolist(),
    "numeric_cols": numeric_cols,
    "label_encoders": {k: v.classes_.tolist() for k, v in label_encoders.items()},
    "scaler_mean": scaler.mean_.tolist(),
    "scaler_scale": scaler.scale_.tolist(),
    "train_ids": train_ids.tolist() if id_col else None,
    "test_ids": test_ids.tolist() if id_col else None,
    "id_col": id_col,
    "n_features": X_train_scaled.shape[1],
    "feature_names": X_train_scaled.columns.tolist(),
}

with open("./working/metadata.pkl", "wb") as f:
    pickle.dump(metadata, f)

if id_col:
    test_ids.to_frame("id").to_csv("./working/test_ids.csv", index=False)

print(f"\n✅ Data processing complete!")
print(f"   Features: {X_train_scaled.shape[1]}")
print(f"   Train samples: {X_train_scaled.shape[0]}")
print(f"   Val samples: {X_val_scaled.shape[0]}")
print(f"   Test samples: {test_scaled.shape[0]}")
print(f"   Classes: {len(target_le.classes_)}")

# Baseline accuracy
most_common = Counter(y_train_enc).most_common(1)[0][1]
baseline_acc = (y_val_enc == most_common).mean()
print(f"\nBaseline (majority class) accuracy: {baseline_acc:.4f}")

# ============================================================
# 7. MODEL DESIGN: LightGBM with class weighting
# ============================================================
n_features = metadata["n_features"]
n_classes = len(metadata["target_classes"])

print(f"\nModel Design: LightGBM Classifier")
print(f"  Features: {n_features}")
print(f"  Classes: {n_classes}")
print(f"  Target classes: {metadata['target_classes']}")

# Compute class weights
class_counts = np.bincount(y_train_enc, minlength=n_classes)
class_weights = len(y_train_enc) / (n_classes * class_counts)
class_weights = np.clip(class_weights, 0.1, 10.0)

print(f"  Class counts: {class_counts}")
print(f"  Class weights: {np.round(class_weights, 3)}")

lgb_params = {
    "objective": "multiclass",
    "num_class": n_classes,
    "metric": "multi_logloss",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "max_depth": 6,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 20,
    "min_child_weight": 1e-3,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "class_weight": "balanced",
    "n_estimators": 2000,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}

model_config = {
    "model_type": "lightgbm",
    "params": lgb_params,
    "n_features": n_features,
    "n_classes": n_classes,
    "class_weights": class_weights.tolist(),
}

with open("./working/model_config.pkl", "wb") as f:
    pickle.dump(model_config, f)

print(f"\n✅ Model designed: LightGBM with {n_classes} classes")
print(
    f"   Key hyperparameters: num_leaves={lgb_params['num_leaves']}, max_depth={lgb_params['max_depth']}, lr={lgb_params['learning_rate']}"
)
print(f"   Regularization: L1={lgb_params['reg_alpha']}, L2={lgb_params['reg_lambda']}")

# ============================================================
# 8. TRAIN LIGHTGBM WITH EARLY STOPPING
# ============================================================
# Reload data (as designed by the pipeline)
X_train_loaded = pd.read_parquet("./working/X_train.parquet")
X_val_loaded = pd.read_parquet("./working/X_val.parquet")
X_test_loaded = pd.read_parquet("./working/X_test.parquet")

y_train_loaded = np.load("./working/y_train.npy")
y_val_loaded = np.load("./working/y_val.npy")

with open("./working/metadata.pkl", "rb") as f:
    metadata_loaded = pickle.load(f)

with open("./working/model_config.pkl", "rb") as f:
    model_config_loaded = pickle.load(f)

target_classes = np.array(metadata_loaded["target_classes"])
n_classes_loaded = len(target_classes)

print(
    f"\nLoaded data: X_train={X_train_loaded.shape}, X_val={X_val_loaded.shape}, X_test={X_test_loaded.shape}"
)
print(f"Classes: {n_classes_loaded}, Target classes: {target_classes}")

lgb_params_loaded = model_config_loaded["params"].copy()
n_estimators = lgb_params_loaded.pop("n_estimators")

model = lgb.LGBMClassifier(**lgb_params_loaded)

callbacks = [
    lgb.early_stopping(stopping_rounds=100, verbose=False),
    lgb.log_evaluation(period=50),
]

print("\nTraining LightGBM...")
model.fit(
    X_train_loaded,
    y_train_loaded,
    eval_set=[(X_val_loaded, y_val_loaded)],
    callbacks=callbacks,
)

best_iteration = model.best_iteration_
print(f"Best iteration: {best_iteration}")

# ============================================================
# 9. VALIDATION EVALUATION (Official Metric: Accuracy)
# ============================================================
val_pred_proba = model.predict_proba(X_val_loaded)
val_pred = np.argmax(val_pred_proba, axis=1)

val_accuracy = accuracy_score(y_val_loaded, val_pred)
print(f"\nValidation Accuracy: {val_accuracy:.6f}")

print(f"\nClassification Report (Validation):")
print(
    classification_report(
        y_val_loaded, val_pred, target_names=target_classes, zero_division=0
    )
)

# ============================================================
# 10. TEST INFERENCE
# ============================================================
test_pred_proba = model.predict_proba(X_test_loaded)
test_pred = np.argmax(test_pred_proba, axis=1)
test_pred_labels = target_classes[test_pred]

# ============================================================
# 11. SAVE SUBMISSION
# ============================================================
submission = pd.DataFrame(
    {"id": range(len(test_pred_labels)), "classification": test_pred_labels}
)

submission.to_csv("./submission/submission.csv", index=False)
print(f"\nSubmission saved: {submission.shape}")
print(f"Prediction distribution:\n{submission['classification'].value_counts()}")

# ============================================================
# FINAL SCORE
# ============================================================
score = val_accuracy
print(f"Final Validation Score: {score}")
