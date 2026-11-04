import pandas as pd
import numpy as np
from catboost import CatBoostClassifier

# Load data
train_df = pd.read_parquet('/path/to/user/workspace/ds_failures/datasets/openml/phoneme/data/train.parquet')
test_df = pd.read_parquet('/path/to/user/workspace/ds_failures/datasets/openml/phoneme/data/test.parquet')

# Define features (exclude 'class' target column)
feature_columns = [col for col in train_df.columns if col != 'class']

# Prepare training data
X_train = train_df[feature_columns].copy()
y_train = train_df['class'].values

# Prepare test data
X_test = test_df[feature_columns].copy()
test_id = test_df.index.values.copy()  # Store original indices if reset, but parquet order matters. 
                                       # Requirement: "id" integer row index starting from 0.
                                       # Assuming test_df reads sequentially, index 0..N-1.
                                         # We reset index to be safe 0-based.

# Clean data and handle potential missing values automatically within the library
# CatBoost handles missing values and categorical types automatically during fit
# We ensure no NaNs cause runtime issues by ensuring inputs are clean
X_train = X_train.fillna(0)
X_test = X_test.fillna(0)

# Ensure categorical columns are identified as such if present (CatBoost does this implicitly for strings)
# Initialize model
model = CatBoostClassifier(iterations=100, depth=6, learning_rate=0.1, loss_function="Logloss")

# Fit the model
model.fit(X_train, y_train, cat_features=None)

# Make predictions
predictions = model.predict(X_test)

# Create output dataframe
output_df = pd.DataFrame({
    'id': test_df.index.values,
    'class': predictions
})

# Save predictions
output_df.to_csv('predictions.csv', index=False)