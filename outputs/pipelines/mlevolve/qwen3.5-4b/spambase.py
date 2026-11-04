import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

# Read data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")
print(f"Train columns: {train_df.columns.tolist()}")

# Separate features and target
X_train = train_df.drop("class", axis=1)
y_train = train_df["class"].astype(int)

# Split for validation
X_train_val, X_val, y_train_val, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

# Full test set for submission
X_test = test_df

# Train LightGBM model
lgb_train = lgb.Dataset(X_train_val, label=y_train_val)
lgb_eval = lgb.Dataset(X_val, label=y_val, reference=lgb_train)

model = lgb.train(
    {
        "objective": "binary",
        "metric": "binary_logloss",
        "verbose": -1,
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "early_stopping_rounds": 100,
        "seed": 42,
    },
    lgb_train,
    num_boost_round=500,
    valid_sets=[lgb_eval],
    callbacks=[lgb.log_evaluation(0)],
)

# Predict on validation set for metric
y_val_proba = model.predict(X_val)
y_val_pred = (y_val_proba >= 0.5).astype(int)

# Predict on full test set for submission
y_test_proba = model.predict(X_test)
y_test_pred = (y_test_proba >= 0.5).astype(int)

# Calculate validation metric
val_auc = roc_auc_score(y_val, y_val_proba)
print(f"Final Validation Score: {val_auc:.4f}")

# Save submission
submission = pd.DataFrame({"id": range(len(X_test)), "class": y_test_pred})
submission.to_csv("./submission/submission.csv", index=False)
print("Submission saved to ./submission/submission.csv")
