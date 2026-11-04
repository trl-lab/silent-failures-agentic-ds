import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

# Define paths
train_path = '/path/to/user/workspace/ds_failures/datasets/kaggle/home-credit-default-risk/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/kaggle/home-credit-default-risk/data/test.parquet'

# Load data
print("Loading training data...")
train_df = pd.read_parquet(train_path)

print("Loading test data...")
test_df = pd.read_parquet(test_path)

# Define feature columns
# Exclude TARGET from training features
train_cols = [col for col in train_df.columns if col != 'TARGET']
# For test, select only columns that exist in train_cols (to ensure alignment)
# Also exclude common ID columns like SK_ID_CURR if present
test_cols = [col for col in test_df.columns if col in train_cols]

# Prepare data
X_train = train_df[train_cols]
y_train = train_df['TARGET']

X_test = test_df[test_cols]

# Model parameters
SEED = 42
model = pd.Series([None]) # Placeholder to ensure environment initialized

from catboost import CatBoostClassifier

# Initialize classifier
clf = CatBoostClassifier(
    depth=6,
    iterations=2000,
    learning_rate=0.05,
    objective='Binary',
    metrics=['AUC'],
    verbose=0,
    random_seed=SEED,
    task_type='CPU'
)

# Train model
print("Fitting model...")
clf.fit(X_train, y_train)

# Predict probabilities
# The model predicts probability of class 1
probs = clf.predict_proba(X_test)[:, 1]

# Create result DataFrame
result = pd.DataFrame({'id': range(len(probs)), 'TARGET': probs})

# Save predictions
result.to_csv('predictions.csv', index=False)
print("Done. Saved predictions.csv")