import pandas as pd
import numpy as np
import catboost as cb

# Define paths based on the provided data files description
TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/openml/KDDCup09_appetency/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/openml/KDDCup09_appetency/data/test.parquet"

# Load data
train_df = pd.read_parquet(TRAIN_PATH)
test_df = pd.read_parquet(TEST_PATH)

# Extract feature columns (all columns except target APPETENCY)
feature_cols = [col for col in train_df.columns if col != "APPETENCY"]

# Prepare target variable for training
# Original target is {-1, 1}, convert to {0, 1} for CatBoost
y_train = (train_df["APPETENCY"] == 1).astype(int)

# Train model using CatBoost for automatic categorical and missing value handling
model = cb.CatBoostClassifier(
    iterations=1000,
    depth=5,
    learning_rate=0.1,
    verbose=0
)

model.fit(train_df[feature_cols], y_train, eval_set=test_df[feature_cols])

# Get predictions probabilities
# predict_proba returns shape (n_samples, n_classes), index 1 corresponds to class 1 (original 1)
pred_probs = model.predict_proba(test_df[feature_cols])[:, 1]

# Convert probabilities back to {-1, 1} labels
# Probability > 0.5 indicates class 1 (original 1), else class 0 (original -1)
pred_labels = (pred_probs > 0.5).astype(int) - 1

# Create results dataframe with 'id' starting from 0 and 'APPETENCY'
results_df = pd.DataFrame({
    "id": range(len(pred_labels)),
    "APPETENCY": pred_labels
})

# Save predictions to csv
results_df.to_csv("predictions.csv", index=False)