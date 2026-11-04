import pandas as pd
import numpy as np
import catboost as cb

# Define paths for train and test data
train_path = '/path/to/user/workspace/ds_failures/datasets/openml/climate-model-simulation-crashes/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/openml/climate-model-simulation-crashes/data/test.parquet'

# Load data from parquet files
train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

# Identify target and feature columns
target_col = 'outcome'
# Select all columns that are not the target column as features
feature_columns = [col for col in train_df.columns if col != target_col]

# Prepare training and testing sets
X_train = train_df[feature_columns]
y_train = train_df[target_col]
X_test = test_df[feature_columns]

# Handle missing values by dropping rows if necessary, or pass to classifier
# CatBoost handles missing values natively, but we ensure clean input for robustness
# Convert target to appropriate type (usually binary integer)
y_train = y_train.astype('int')
y_test_label = y_train.values

# Initialize CatBoost classifier (handles categoricals and missing values automatically)
model = cb.CatBoostClassifier(
    verbose=False,
    iterations=100,
    learning_rate=0.1,
    loss_function='Logloss',
    depth=6,
    random_seed=42,
    early_stopping_rounds=20
)

# Train the model
model.fit(X_train, y_train)

# Make predictions on test set
# Returns array of probabilities, we threshold for binary outcome
predictions_prob = model.predict_proba(X_test)[:, 1]
predictions_binary = (predictions_prob >= 0.5).astype(int)

# Create output DataFrame
result_df = pd.DataFrame({
    'id': range(len(predictions_binary)),
    'outcome': predictions_binary
})

# Save predictions to csv
result_df.to_csv('predictions.csv', index=False)