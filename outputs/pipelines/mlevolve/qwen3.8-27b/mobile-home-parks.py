import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import lightgbm as lgb
import os

# ── Load data ──────────────────────────────────────────────────────
train = pd.read_parquet("./input/train.parquet")
test  = pd.read_parquet("./input/test.parquet")

print(f"Train shape: {train.shape}")
print(f"Test  shape: {test.shape}")
print(f"Target distribution:\n{train['SIZE'].value_counts()}")

# ── Prepare features ───────────────────────────────────────────────
target = train["SIZE"]

id_like = [c for c in train.columns if c.lower() in ("id", "index")]
drop_train = ["SIZE"] + [c for c in id_like if c in train.columns]
drop_test  = [c for c in id_like if c in test.columns]

X_all  = train.drop(columns=drop_train).copy()
X_test = test.drop(columns=drop_test).copy()

# keep only columns present in both sets
feature_cols = [c for c in X_all.columns if c in X_test.columns]
X_all  = X_all[feature_cols]
X_test = X_test[feature_cols]

# convert object / string columns to category (LightGBM native handling)
for col in feature_cols:
    if X_all[col].dtype == object:
        X_all[col]  = X_all[col].astype("category")
        X_test[col] = X_test[col].astype("category")

# ── Validation split (stratified) ──────────────────────────────────
X_tr, X_val, y_tr, y_val = train_test_split(
    X_all, target, test_size=0.2, random_state=42, stratify=target
)

# ── Train with early stopping ──────────────────────────────────────
model = lgb.LGBMClassifier(
    objective="multiclass",
    learning_rate=0.01,
    num_leaves=63,
    n_estimators=5000,
    min_child_samples=20,
    lambda_l1=0.5,
    lambda_l2=0.5,
    subsample=0.8,
    colsample_bytree=0.8,
    verbose=-1,
    random_state=42,
)
model.fit(
    X_tr, y_tr,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(200)],
)

# ── Validation score (official metric: accuracy) ───────────────────
val_pred  = model.predict(X_val)
val_acc   = accuracy_score(y_val, val_pred)

# ── Retrain on full training data using best iteration ─────────────
final_model = lgb.LGBMClassifier(
    objective="multiclass",
    learning_rate=0.01,
    num_leaves=63,
    n_estimators=model.best_iteration_,
    min_child_samples=20,
    lambda_l1=0.5,
    lambda_l2=0.5,
    subsample=0.8,
    colsample_bytree=0.8,
    verbose=-1,
    random_state=42,
)
final_model.fit(X_all, target)

# ── Test predictions ───────────────────────────────────────────────
test_pred = final_model.predict(X_test)

# ── Write submission ───────────────────────────────────────────────
os.makedirs("./submission", exist_ok=True)
submission = pd.DataFrame({
    "id":   range(len(test)),
    "SIZE": np.asarray(test_pred, dtype=object),
})
submission.to_csv("./submission/submission.csv", index=False)
print(f"Final Validation Score: {val_acc:.4f}")