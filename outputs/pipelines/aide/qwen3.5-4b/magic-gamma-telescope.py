import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Separate features and target
X_train = train_df.drop("class", axis=1)
y_train = train_df["class"]
X_test = test_df.copy()  # Test data has no target column

# Handle potential missing values
X_train = X_train.fillna(X_train.median())
X_test = X_test.fillna(X_test.median())

# Create validation split
X_train_val, X_test_split, y_train_val, y_test_split = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_val)
X_test_split_scaled = scaler.transform(X_test_split)
X_test_scaled = scaler.transform(X_test)

# LightGBM model with callbacks
model = lgb.LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
)

# Train with early stopping and validation logging
callbacks = [lgb.early_stopping(50), lgb.log_evaluation(-1)]
model.fit(
    X_train_scaled,
    y_train_val,
    eval_set=[(X_test_split_scaled, y_test_split)],
    callbacks=callbacks,
)

# Evaluate on validation set
val_pred = model.predict_proba(X_test_split_scaled)[:, 1]
val_auc = roc_auc_score(y_test_split, val_pred)
print(f"Validation ROC AUC: {val_auc:.4f}")

# Generate test predictions
test_pred = model.predict_proba(X_test_scaled)[:, 1]

# Save submission
submission = pd.DataFrame({"id": range(len(test_pred)), "class": test_pred})
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved to ./working/submission.csv")
