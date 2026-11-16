import pandas as pd
import numpy as np
from catboost import CatBoostRegressor

# Load training and testing data
train_path = '/path/to/user/workspace/ds_failures/datasets/strable/discretionary-grant/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/strable/discretionary-grant/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

# Define target column and prepare feature set
target_column = 'AwardTotal'
feature_columns = [col for col in train_df.columns if col != target_column]

X_train = train_df[feature_columns]
y_train = train_df[target_column]
X_test = test_df[feature_columns]

# Automatically identify categorical columns for CatBoost
# CatBoost handles missing values and object types effectively if specified
categorical_indices = []
for idx, col in enumerate(feature_columns):
    if pd.api.types.is_object_dtype(train_df[col]):
        categorical_indices.append(idx)

# Initialize Regressor with RMSE loss function
model = CatBoostRegressor(
    iterations=500,
    depth=6,
    learning_rate=0.05,
    loss_function='RMSE',
    cat_features=categorical_indices,
    random_state=42
)

# Train the model
model.fit(
    X_train, 
    y_train,
    cat_features=categorical_indices,
    eval_metric='RMSE'
)

# Generate predictions
predictions = model.predict(X_test)

# Create output dataframe with 'id' (0-indexed row number) and 'AwardTotal'
predictions_df = pd.DataFrame({
    'id': range(len(predictions)),
    'AwardTotal': predictions
})

# Save predictions to CSV
predictions_df.to_csv('predictions.csv', index=False)