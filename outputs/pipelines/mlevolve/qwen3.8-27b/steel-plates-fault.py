import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import lightgbm as lgb
import warnings

warnings.filterwarnings("ignore")

# ── Directories ──────────────────────────────────────────────────────────────
os.makedirs("./submission", exist_ok=True)
os.makedirs("./working", exist_ok=True)

# ── Load data ────────────────────────────────────────────────────────────────
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

print(f"Train shape: {train_df.shape}")
print(f"Test  shape: {test_df.shape}")
print(f"Train columns: {train_df.columns.tolist()}")
print(f"Target dtype : {train_df['target'].dtype}")
print(f"Target unique: {train_df['target'].unique()}")
print(f"Target dist  :\n{train_df['target'].value_counts()}")
print(f"Missing (train): {int(train_df.isnull().sum().sum())}")
print(f"Missing (test) : {int(test_df.isnull().sum().sum())}")

# ── Feature / target split ───────────────────────────────────────────────────
target_col = "target"
feature_cols = [c for c in train_df.columns if c != target_col]

# Safety: ensure every feature column exists in test
for c in feature_cols:
    if c not in test_df.columns:
        print(f"WARNING: '{c}' missing in test, filling with 0")
        test_df[c] = 0.0

# ── Target encoding (handle both int and string targets) ─────────────────────
y_raw = train_df[target_col].values
target_is_str = (
    y_raw.dtype == object
    or y_raw.dtype.name == "category"
    or any(isinstance(v, str) for v in y_raw[:20])
)

if target_is_str:
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    print(f"Label-encoded target → classes: {list(le.classes_)}")
else:
    y = y_raw.astype(int)
    le = None
    print(f"Numeric target values: {sorted(np.unique(y))}")

X = train_df[feature_cols].values.astype(np.float64)
X_test = test_df[feature_cols].values.astype(np.float64)

# ── Clean ────────────────────────────────────────────────────────────────────
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

# ── Stratified validation split ──────────────────────────────────────────────
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)
num_classes = len(np.unique(y))
print(
    f"Train split: {X_train.shape}  |  Val split: {X_val.shape}  |  Classes: {num_classes}"
)

# ── Scale (fit on train split only to avoid leakage) ─────────────────────────
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)
X_scaled = scaler.transform(X)

# ── LightGBM params ──────────────────────────────────────────────────────────
params = {
    "objective": "multiclass",
    "num_class": num_classes,
    "metric": "multi_logloss",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 10,
    "lambda_l1": 0.1,
    "lambda_l2": 0.1,
    "verbose": -1,
    "seed": 42,
}

# ── Train with early stopping ────────────────────────────────────────────────
dtrain = lgb.Dataset(X_train, label=y_train)
dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

model = lgb.train(
    params,
    dtrain,
    num_boost_round=2000,
    valid_sets=[dval],
    valid_names=["val"],
    callbacks=[lgb.early_stopping(stopping_rounds=100), lgb.log_evaluation(period=200)],
)
best_iter = model.best_iteration
print(f"Best iteration: {best_iter}")

# ── Validation prediction & metric ───────────────────────────────────────────
val_pred_proba = model.predict(X_val, num_iteration=best_iter)
val_pred = np.argmax(val_pred_proba, axis=1)
val_accuracy = accuracy_score(y_val, val_pred)
print(f"Validation Accuracy: {val_accuracy:.6f}")

# ── Retrain on FULL training data (same scaler, same best_iter) ──────────────
dtrain_full = lgb.Dataset(X_scaled, label=y)
model_full = lgb.train(
    params,
    dtrain_full,
    num_boost_round=best_iter,
    callbacks=[lgb.log_evaluation(period=200)],
)

# ── Test inference ───────────────────────────────────────────────────────────
test_pred_proba = model_full.predict(X_test_scaled, num_iteration=best_iter)
test_pred = np.argmax(test_pred_proba, axis=1)

# Decode back to original target labels if we encoded them
if le is not None:
    test_pred_final = le.inverse_transform(test_pred)
else:
    test_pred_final = test_pred

# ── Write submission ─────────────────────────────────────────────────────────
submission = pd.DataFrame(
    {
        "id": np.arange(len(test_pred_final), dtype=int),
        "target": test_pred_final,
    }
)
submission.to_csv("./submission/submission.csv", index=False)

print(f"Submission shape: {submission.shape}")
print(f"Submission head:\n{submission.head(10)}")
print(f"Prediction distribution:\n{pd.Series(test_pred_final).value_counts()}")
print(f"Final Validation Score: {val_accuracy}")