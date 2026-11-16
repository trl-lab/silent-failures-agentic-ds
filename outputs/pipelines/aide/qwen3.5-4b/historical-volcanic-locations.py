import pandas as pd
import lightgbm as lgb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Target column
target_col = "DEATHS_TOTAL_DESCRIPTION"
id_col = "id" if id_col in test_df.columns else None

# Create feature matrix
X_train = train_df.drop(columns=[target_col])
y_train = train_df[target_col]

# Split into train/validation sets (70/30)
X_train_split, X_val, y_train_split, y_val_split = train_test_split(
    X_train, y_train, test_size=0.3, random_state=42, stratify=y_train
)

# Convert to LightGBM format
train_data = lgb.Dataset(X_train_split, label=y_train_split)
val_data = lgb.Dataset(X_val, label=y_val_split, reference=train_data)

# Train LightGBM model with early stopping
params = {
    "objective": "multiclass",
    "num_class": y_train_split.nunique(),
    "boosting_type": "gbdt",
    "verbose": -1,
    "learning_rate": 0.1,
    "num_leaves": 31,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
}

model = lgb.train(
    params,
    train_data,
    num_boost_round=500,
    valid_sets=[train_data, val_data],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
)

# Evaluate on validation set
y_val_pred = model.predict(X_val)
accuracy = accuracy_score(y_val_split, y_val_pred)
print(f"Validation Accuracy: {accuracy:.4f}")

# Prepare submission
if id_col is None:
    test_df = test_df.reset_index()
    test_df["id"] = test_df.index
    test_df = test_df.drop(columns=["index"])

submission = pd.DataFrame(
    {"id": test_df.index, target_col: model.predict(test_df.drop(columns=[target_col]))}
)

submission.to_csv("./working/submission.csv", index=False)
print("Submission saved to ./working/submission.csv")
