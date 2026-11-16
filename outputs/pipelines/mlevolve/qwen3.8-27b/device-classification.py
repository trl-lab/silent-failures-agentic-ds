import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb

# ── Paths ──────────────────────────────────────────────────────────────
INPUT_DIR = "./input"
SUBMISSION_DIR = "./submission"
WORK_DIR = "./working"
os.makedirs(SUBMISSION_DIR, exist_ok=True)
os.makedirs(WORK_DIR, exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────
train_df = pd.read_parquet(os.path.join(INPUT_DIR, "train.parquet"))
test_df = pd.read_parquet(os.path.join(INPUT_DIR, "test.parquet"))

TARGET = "device_class"
RANDOM_STATE = 42

# ── Feature columns (union of train & test, excluding target) ─────────
feature_cols = [c for c in train_df.columns if c != TARGET]
for c in test_df.columns:
    if c != TARGET and c not in feature_cols:
        feature_cols.append(c)

# ── Stratified train / val split ───────────────────────────────────────
X_all = train_df[feature_cols].copy()
y_all = train_df[TARGET].copy()

X_train, X_val, y_train, y_val = train_test_split(
    X_all, y_all, test_size=0.15, random_state=RANDOM_STATE, stratify=y_all
)

# Test features (drop target column if it happens to be present)
X_test = test_df.drop(columns=[TARGET], errors="ignore").copy()

# ── Identify column types from training data ───────────────────────────
numeric_cols = []
categorical_cols = []
datetime_cols = []

for col in feature_cols:
    if col not in X_train.columns:
        continue
    if pd.api.types.is_datetime64_any_dtype(X_train[col]):
        datetime_cols.append(col)
    elif pd.api.types.is_numeric_dtype(X_train[col]):
        numeric_cols.append(col)
    else:
        categorical_cols.append(col)

# ── Fit encoders / statistics on TRAINING data only (no leakage) ───────
cat_mappings = {}  # col -> {category_str: int_code}
numeric_medians = {}  # col -> median

for col in categorical_cols:
    le = LabelEncoder()
    train_vals = (
        X_train[col]
        .astype(str)
        .replace({"": "MISSING", "nan": "MISSING", "None": "MISSING"})
    )
    le.fit(train_vals)
    cat_mappings[col] = {c: i for i, c in enumerate(le.classes_)}

for col in numeric_cols:
    numeric_medians[col] = float(X_train[col].median())


# ── Transform (applied identically to train / val / test) ──────────────
def transform(df):
    df = df.copy()

    # Ensure every feature column exists
    for col in feature_cols:
        if col not in df.columns:
            df[col] = np.nan

    # Datetime → calendar components
    for col in datetime_cols:
        dt = pd.to_datetime(df[col], errors="coerce")
        df[f"{col}_year"] = dt.dt.year.fillna(0).astype(int)
        df[f"{col}_month"] = dt.dt.month.fillna(0).astype(int)
        df[f"{col}_day"] = dt.dt.day.fillna(0).astype(int)
        df[f"{col}_dayofweek"] = dt.dt.dayofweek.fillna(0).astype(int)
        df[f"{col}_hour"] = dt.dt.hour.fillna(0).astype(int)
        df.drop(col, axis=1, inplace=True)

    # Categorical → integer codes (unseen categories → -1)
    for col in categorical_cols:
        vals = (
            df[col]
            .astype(str)
            .replace({"": "MISSING", "nan": "MISSING", "None": "MISSING"})
        )
        df[col] = vals.map(cat_mappings[col]).fillna(-1).astype(int)

    # Numeric → coerce + fill NaN with training median
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(numeric_medians[col])

    # Build ordered output column list
    out_cols = []
    for col in feature_cols:
        if col in datetime_cols:
            out_cols += [
                f"{col}_year",
                f"{col}_month",
                f"{col}_day",
                f"{col}_dayofweek",
                f"{col}_hour",
            ]
        else:
            out_cols.append(col)

    return df[out_cols]


X_train_p = transform(X_train)
X_val_p = transform(X_val)
X_test_p = transform(X_test)

# Align columns across all three sets
all_cols = X_train_p.columns.tolist()
for col in all_cols:
    if col not in X_val_p.columns:
        X_val_p[col] = 0
    if col not in X_test_p.columns:
        X_test_p[col] = 0
X_val_p = X_val_p[all_cols]
X_test_p = X_test_p[all_cols]

print(
    f"Features: {len(all_cols)} | Train: {len(X_train_p)} | Val: {len(X_val_p)} | Test: {len(X_test_p)}"
)

# ── Train LightGBM classifier ──────────────────────────────────────────
model = lgb.LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,
    num_leaves=31,
    min_child_samples=10,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=RANDOM_STATE,
    verbose=-1,
)

model.fit(
    X_train_p,
    y_train,
    eval_set=[(X_val_p, y_val)],
    callbacks=[
        lgb.early_stopping(stopping_rounds=50, verbose=False),
        lgb.log_evaluation(period=0),
    ],
)

# ── Validation (official metric: accuracy) ─────────────────────────────
val_pred = model.predict(X_val_p)
val_acc = accuracy_score(y_val, val_pred)
print(f"Validation Accuracy: {val_acc:.6f}")

# ── Test predictions ───────────────────────────────────────────────────
test_pred = model.predict(X_test_p)

# ── Save submission ────────────────────────────────────────────────────
submission = pd.DataFrame(
    {
        "id": np.arange(len(test_pred), dtype=int),
        "device_class": test_pred,
    }
)
submission.to_csv(os.path.join(SUBMISSION_DIR, "submission.csv"), index=False)
print(f"Submission saved: {len(submission)} rows")
print(f"Final Validation Score: {val_acc}")
