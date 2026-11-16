import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import os

# Set working directory
os.makedirs("./working", exist_ok=True)

# Load base tables
train_base = pd.read_parquet("./input/train/base.parquet")
test_base = pd.read_parquet("./input/test/base.parquet")

# Load all feature tables for training
train_applprev_1_0 = pd.read_parquet("./input/train/applprev_1_0.parquet")
train_applprev_1_1 = pd.read_parquet("./input/train/applprev_1_1.parquet")
train_applprev_2 = pd.read_parquet("./input/train/applprev_2.parquet")

# Load all feature tables for testing
test_applprev_1_0 = pd.read_parquet("./input/test/applprev_1_0.parquet")
test_applprev_1_1 = pd.read_parquet("./input/test/applprev_1_1.parquet")
test_applprev_2 = pd.read_parquet("./input/test/applprev_2.parquet")


import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import os

# Set working directory
os.makedirs("./working", exist_ok=True)

# Load base tables
train_base = pd.read_parquet("./input/train/base.parquet")
test_base = pd.read_parquet("./input/test/base.parquet")

# Load all feature tables for training
train_applprev_1_0 = pd.read_parquet("./input/train/applprev_1_0.parquet")
train_applprev_1_1 = pd.read_parquet("./input/train/applprev_1_1.parquet")
train_applprev_2 = pd.read_parquet("./input/train/applprev_2.parquet")

# Load all feature tables for testing
test_applprev_1_0 = pd.read_parquet("./input/test/applprev_1_0.parquet")
test_applprev_1_1 = pd.read_parquet("./input/test/applprev_1_1.parquet")
test_applprev_2 = pd.read_parquet("./input/test/applprev_2.parquet")

# Create a unified training dataset by joining on case_id
train_features = pd.concat(
    [train_base, train_applprev_1_0, train_applprev_1_1, train_applprev_2], axis=1
)
test_features = pd.concat(
    [test_base, test_applprev_1_0, test_applprev_1_1, test_applprev_2], axis=1
)

# Define feature columns (exclude case_id, date_decision, MONTH, WEEK_NUM, target)
feature_cols = [
    col
    for col in train_features.columns
    if col not in ["case_id", "date_decision", "MONTH", "WEEK_NUM", "target"]
]

# Create validation split (80% train, 20% validation)
X_train = train_features[feature_cols]
y_train = train_features["target"]
X_val = train_features[feature_cols].iloc[-(len(y_train) * 0.2) :]
y_val = train_features["target"].iloc[-(len(y_train) * 0.2) :]

# Prepare LightGBM datasets
train_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

# Train LightGBM model with early stopping
model = lgb.train(
    {"objective": "binary", "metric": "auc", "verbose": -1, "seed": 42},
    train_data,
    num_boost_round=1000,
    valid_sets=[train_data, val_data],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
)

# Evaluate on validation set
y_val_pred = model.predict(X_val)
val_auc = roc_auc_score(y_val, y_val_pred)
print(f"Validation ROC AUC: {val_auc:.4f}")

# Generate predictions for test set
test_features = pd.read_parquet("./input/test/base.parquet")
test_applprev_1_0 = pd.read_parquet("./input/test/applprev_1_0.parquet")
test_applprev_1_1 = pd.read_parquet("./input/test/applprev_1_1.parquet")
test_applprev_2 = pd.read_parquet("./input/test/applprev_2.parquet")

test_features = pd.concat(
    [test_features, test_applprev_1_0, test_applprev_1_1, test_applprev_2], axis=1
)
X_test = test_features[feature_cols]

# Generate predictions and save to submission.csv
test_predictions = model.predict(X_test)
submission = pd.DataFrame(
    {"id": range(len(test_predictions)), "target": test_predictions}
)
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved to ./working/submission.csv")
