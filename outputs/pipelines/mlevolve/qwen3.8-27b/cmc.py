import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ── Setup directories ──────────────────────────────────────────────
os.makedirs("./submission", exist_ok=True)
os.makedirs("./working", exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "Contraceptive_method_used"
feature_cols = [c for c in train_df.columns if c != target_col]

print(f"Train shape: {train_df.shape}")
print(f"Test shape:  {test_df.shape}")
print(f"Features:    {feature_cols}")
print(f"Target dist: {train_df[target_col].value_counts().to_dict()}")

# ── Stratified validation split (20 % held out) ───────────────────
X_train, X_val, y_train, y_val = train_test_split(
    train_df[feature_cols],
    train_df[target_col],
    test_size=0.2,
    random_state=42,
    stratify=train_df[target_col],
)

# ── Model hyper-parameters (conservative for small dataset) ───────
model_params = {
    "objective": "multiclass",
    "num_class": 3,
    "learning_rate": 0.05,
    "num_leaves": 15,
    "n_estimators": 1000,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "min_child_samples": 20,
    "verbose": -1,
    "random_state": 42,
}

# ── Train on train split, early-stop on validation ────────────────
val_model = lgb.LGBMClassifier(**model_params)
val_model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
)

# ── Validation metric (official: accuracy) ─────────────────────────
val_pred = val_model.predict(X_val)
val_accuracy = accuracy_score(y_val, val_pred)
print(f"Best iteration: {val_model.best_iteration_}")
print(f"Final Validation Score: {val_accuracy}")

# ── Retrain on FULL training data for best test generalisation ────
full_model = lgb.LGBMClassifier(**{**model_params, "n_estimators": val_model.best_iteration_})
full_model.fit(train_df[feature_cols], train_df[target_col])

# ── Test predictions (model forward pass) ─────────────────────────
test_pred = full_model.predict(test_df[feature_cols]).astype(int)

# ── Write submission ───────────────────────────────────────────────
submission = pd.DataFrame(
    {
        "id": np.arange(len(test_df)),
        "Contraceptive_method_used": test_pred,
    }
)
submission.to_csv("./submission/submission.csv", index=False)
print(f"Submission saved: {len(submission)} rows → ./submission/submission.csv")