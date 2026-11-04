import pandas as pd
import numpy as np
import catboost as cb

# Define file paths as specified
train_path = "/path/to/user/workspace/ds_failures/datasets/kaggle/otto-group-product-classification-challenge/data/train.parquet"
test_path = "/path/to/user/workspace/ds_failures/datasets/kaggle/otto-group-product-classification-challenge/data/test.parquet"
output_path = "predictions.csv"

# Load data
train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

# Define columns to exclude
target_col = "target"
id_col = "id"

# Drop target column from train and test for processing
train_cols = [c for c in train_df.columns if c != target_col]
test_cols = [c for c in test_df.columns if c != target_col and c != id_col]

# Drop ID column from test if it exists (Otto dataset test file usually has ID)
# However, we need to generate it back for the output.
test_cols = [c for c in test_df.columns if c != target_col and c != id_col]

# Filter dataframes
train_data = train_df[train_cols].copy()
test_data = test_df[test_cols].copy()
y_train = train_df[target_col].values

# Initialize CatBoost model for multi-class classification
# Using 'MultiClass' as objective (older spelling) or 'multiclass' (newer).
# We use 'MultiClass' to ensure compatibility with older environments while handling proba automatically.
# However, standard catboost for multilabel/multiclass uses 'MultiClass' for single target multi-label.
# For this Kaggle competition (Multi-Class Single Target), 'MultiClass' is incorrect. 
# It should be 'MultiClass' if it was one-vs-rest, but actually it is just 'MultiClass' with n_jobs.
# Wait, CatBoost 'MultiClass' refers to one-vs-all or soft voting?
# The correct objective is 'MultiClass' for multi-class classification in CatBoost.
# Actually, 'MultiClass' is the objective name for this problem.

model = cb.CatBoostClassifier(
    iterations=1000,
    eval_metric='Logloss',
    learning_rate=0.05,
    verbosity=0,
    random_seed=42,
    depth=6,
    lost_best=0
)

# Train model
# catboost handles categorical and missing values automatically
model.fit(train_data, y_train, cat_features=test_cols, verbose=False)

# Predict on test set
# cat_features argument in fit helps identify string columns, but for predict we just pass data
y_pred_proba = model.predict_proba(test_data)

# Get predicted class labels (argmax)
y_pred_class = y_pred_proba.argmax(axis=1)

# Prepare output dataframe
output_df = pd.DataFrame({
    "id": range(len(test_data)),
    "target": y_pred_class
})

# Save predictions
output_df.to_csv(output_path, index=False)