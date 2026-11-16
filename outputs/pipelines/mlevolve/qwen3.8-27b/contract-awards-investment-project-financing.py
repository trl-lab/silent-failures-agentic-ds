import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split

# ── Paths ──────────────────────────────────────────────────────────────────
TRAIN_PATH = "./input/train.parquet"
TEST_PATH = "./input/test.parquet"
SUB_DIR = "./submission"
os.makedirs(SUB_DIR, exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────────
print("Loading data...")
train_df = pd.read_parquet(TRAIN_PATH)
test_df = pd.read_parquet(TEST_PATH)
print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

TARGET = "Supplier Contract Amount (USD)"

# ── Identify target and features ───────────────────────────────────────────
y = train_df[TARGET].values
train_df = train_df.drop(columns=[TARGET])

# Drop any ID-like columns that aren't features
id_cols = [c for c in train_df.columns if c.lower() in ("id", "index", "row_id")]
for c in id_cols:
    train_df = train_df.drop(columns=[c])
    test_df = test_df.drop(columns=[c])

# ── Handle categorical columns ─────────────────────────────────────────────
cat_cols = train_df.select_dtypes(include=["object", "category"]).columns.tolist()
print(
    f"Categorical columns ({len(cat_cols)}): {cat_cols[:10]}{'...' if len(cat_cols)>10 else ''}"
)

# Convert to category dtype for LightGBM
for c in cat_cols:
    train_df[c] = train_df[c].astype("category")
    test_df[c] = test_df[c].astype("category")

# ── Handle missing values (LightGBM handles NaN natively, but fill inf) ───
num_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
for c in num_cols:
    train_df[c] = train_df[c].replace([np.inf, -np.inf], np.nan)
    test_df[c] = test_df[c].replace([np.inf, -np.inf], np.nan)

# ── Create validation split ────────────────────────────────────────────────
X_train, X_val, y_train, y_val = train_test_split(
    train_df, y, test_size=0.15, random_state=42
)
print(f"Train split: {X_train.shape}, Val split: {X_val.shape}")

# ── LightGBM parameters ────────────────────────────────────────────────────
params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 255,
    "max_depth": -1,
    "min_child_samples": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "n_estimators": 5000,
    "verbose": -1,
    "n_jobs": -1,
    "seed": 42,
}

# ── Train model with early stopping ────────────────────────────────────────
print("Training LightGBM...")
train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_cols)
val_data = lgb.Dataset(
    X_val, label=y_val, categorical_feature=cat_cols, reference=train_data
)

model = lgb.train(
    params,
    train_data,
    valid_sets=[val_data],
    callbacks=[
        lgb.early_stopping(stopping_rounds=100, verbose=False),
        lgb.log_evaluation(period=0),  # suppress per-epoch logging
    ],
)

best_iter = model.best_iteration
print(f"Best iteration: {best_iter}")

# ── Validation metric (official: RMSE) ─────────────────────────────────────
val_pred = model.predict(X_val, num_iteration=best_iter)
val_rmse = root_mean_squared_error(y_val, val_pred)
print(f"Validation RMSE: {val_rmse:.4f}")

# ── Retrain on full training data with best iteration ──────────────────────
print("Retraining on full training data...")
full_train_data = lgb.Dataset(train_df, label=y, categorical_feature=cat_cols)

final_model = lgb.train(
    params,
    full_train_data,
    num_boost_round=best_iter,
    callbacks=[lgb.log_evaluation(period=0)],
)

# ── Predict on test set ────────────────────────────────────────────────────
test_pred = final_model.predict(test_df, num_iteration=best_iter)
# Ensure predictions are non-negative (amounts can't be negative)
test_pred = np.clip(test_pred, 0, None)

# ── Build submission ───────────────────────────────────────────────────────
submission = pd.DataFrame(
    {
        "id": np.arange(len(test_pred)),
        "Supplier Contract Amount (USD)": test_pred,
    }
)
submission.to_csv(os.path.join(SUB_DIR, "submission.csv"), index=False)
print(f"Submission saved: {submission.shape}")

# ── Final validation score (official metric: RMSE) ─────────────────────────
score = val_rmse
print(f"Final Validation Score: {score}")
