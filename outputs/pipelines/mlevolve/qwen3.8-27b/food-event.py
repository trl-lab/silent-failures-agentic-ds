import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import lightgbm as lgb
import warnings

warnings.filterwarnings("ignore")

# ── Directories ──────────────────────────────────────────────────────────────
os.makedirs("./submission", exist_ok=True)
os.makedirs("./working", exist_ok=True)

# ── Load data ────────────────────────────────────────────────────────────────
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

print(f"Train shape: {train.shape}")
print(f"Test  shape: {test.shape}")
print(f"Train columns: {list(train.columns)}")
print(f"Target distribution:\n{train['outcomes'].value_counts()}")

# ── Separate target ──────────────────────────────────────────────────────────
y = train["outcomes"].values


# ── Drop id-like columns ─────────────────────────────────────────────────────
def drop_id_cols(df):
    id_cols = [
        c
        for c in df.columns
        if c.lower() in ("id", "index", "row_id", "row_index", "idx")
    ]
    return df.drop(columns=id_cols, errors="ignore")


X = drop_id_cols(train.drop(columns=["outcomes"]))
X_test = drop_id_cols(test)

# ── Align columns between train and test ─────────────────────────────────────
common_cols = [c for c in X.columns if c in X_test.columns]
X = X[common_cols].copy()
X_test = X_test[common_cols].copy()
print(f"Using {len(common_cols)} features")

# ── Encode categorical features (label-encode using combined vocabulary) ─────
cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
for col in cat_cols:
    combined = pd.concat([X[col].astype(str), X_test[col].astype(str)], axis=0)
    vocab = {c: i for i, c in enumerate(sorted(combined.unique()))}
    X[col] = X[col].astype(str).map(vocab).fillna(-1).astype(int)
    X_test[col] = X_test[col].astype(str).map(vocab).fillna(-1).astype(int)

# ── Force all remaining columns to numeric ───────────────────────────────────
for col in X.columns:
    X[col] = pd.to_numeric(X[col], errors="coerce")
    X_test[col] = pd.to_numeric(X_test[col], errors="coerce")

# ── Validation split (before imputation to prevent leakage) ──────────────────
try:
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
except ValueError:
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
print(f"Train: {X_train.shape}  |  Val: {X_val.shape}")

# ── Fill missing values with train-only medians (fallback 0) ─────────────────
for col in X_train.columns:
    if X_train[col].isnull().any() or X_val[col].isnull().any() or X_test[col].isnull().any():
        fill_val = X_train[col].median()
        if pd.isna(fill_val):
            fill_val = 0
        X_train[col] = X_train[col].fillna(fill_val)
        X_val[col] = X_val[col].fillna(fill_val)
        X_test[col] = X_test[col].fillna(fill_val)

# ── Train LightGBM with early stopping ───────────────────────────────────────
params = {
    "n_estimators": 3000,
    "learning_rate": 0.05,
    "max_depth": -1,
    "num_leaves": 63,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}

model = lgb.LGBMClassifier(**params)
model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[
        lgb.early_stopping(stopping_rounds=100),
        lgb.log_evaluation(period=1000),
    ],
)

# ── Validation metric (official: accuracy) ───────────────────────────────────
val_preds = model.predict(X_val)
val_accuracy = accuracy_score(y_val, val_preds)
print(f"Validation Accuracy: {val_accuracy:.6f}")

# ── Retrain on full training data with best iteration count ──────────────────
best_iter = getattr(model, "best_iteration_", params["n_estimators"])
final_model = lgb.LGBMClassifier(**{**params, "n_estimators": max(int(best_iter), 100)})
final_model.fit(X, y)

# ── Test predictions ─────────────────────────────────────────────────────────
test_preds = final_model.predict(X_test)

# Match target dtype for submission
if y.dtype in (np.int64, np.int32, np.int16, np.int8):
    test_preds = test_preds.astype(int)

# ── Write submission ─────────────────────────────────────────────────────────
submission = pd.DataFrame(
    {
        "id": range(len(test_preds)),
        "outcomes": test_preds,
    }
)
submission.to_csv("./submission/submission.csv", index=False)
print(f"Submission saved: {len(submission)} rows → ./submission/submission.csv")

# ── Final score (must be last line) ──────────────────────────────────────────
print(f"Final Validation Score: {val_accuracy}")