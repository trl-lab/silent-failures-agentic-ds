import pandas as pd
import lightgbm as lgb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Separate features and target
X_train = train_df.drop("Decision type", axis=1)
y_train = train_df["Decision type"]
X_test = test_df.drop("Decision type", axis=1)

# Split into train/validation (80/20)
X_train_split, X_val, y_train_split, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

# LightGBM parameters
params = {
    "objective": "multiclass",
    "metric": "accuracy",
    "num_class": len(np.unique(y_train)),
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
}

# Create LightGBM datasets
train_data = lgb.Dataset(X_train_split, label=y_train_split)
val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

# Train model with callbacks
model = lgb.train(
    params,
    train_data,
    num_boost_round=1000,
    valid_sets=[train_data, val_data],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
)

# Evaluate on validation set
y_val_pred = model.predict(X_val)
y_val_pred = np.round(y_val_pred).astype(int)
val_accuracy = accuracy_score(y_val, y_val_pred)
print(f"Validation Accuracy: {val_accuracy:.4f}")

# Predict on test set
test_predictions = model.predict(X_test)
test_predictions = np.round(test_predictions).astype(int)

# Create submission
submission = pd.DataFrame(
    {"id": range(len(test_predictions)), "Decision type": test_predictions}
)
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved to ./working/submission.csv")
