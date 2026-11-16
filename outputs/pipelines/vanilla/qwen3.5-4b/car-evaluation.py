import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier

# Define paths for train and test parquet files
train_path = '/path/to/user/workspace/ds_failures/datasets/uci/car-evaluation/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/uci/car-evaluation/data/test.parquet'

# Load data
df_train = pd.read_parquet(train_path)
df_test = pd.read_parquet(test_path)

# Define target column based on requirements ('car')
target_col = 'car'

# Identify feature columns (exclude target column)
feature_cols = [col for col in df_train.columns if col != target_col]

# Extract features and targets
X_train = df_train[feature_cols].copy()
y_train = df_train[target_col].copy()

X_test = df_test[feature_cols].copy()

# Initialize and train model
clf = LGBMClassifier(learning_rate=0.1, num_leaves=31, random_state=42, silent=True)
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Create output dataframe with 'id' (starting from 0) and 'car'
results_df = pd.DataFrame({
    'id': range(len(df_test)),
    'car': y_pred.astype(str)
})

# Save predictions to current working directory
results_df.to_csv('predictions.csv', index=False)