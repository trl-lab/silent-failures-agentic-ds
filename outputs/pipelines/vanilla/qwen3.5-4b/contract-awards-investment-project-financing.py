import pandas as pd
import numpy as np
import catboost as cb

# Paths for data files
train_path = '/path/to/user/workspace/ds_failures/datasets/strable/contract-awards-investment-project-financing/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/strable/contract-awards-investment-project-financing/data/test.parquet'

# Load the data
train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

# Define target column
target_column = 'Supplier Contract Amount (USD)'

# Extract features (exclude target)
feature_columns = [col for col in train_df.columns if col != target_column]

if len(feature_columns) == 0:
    # If no features found, handle gracefully (e.g., just predicting mean)
    # But assuming features exist based on task definition
    X_train = train_df.loc[:, :][0] # Placeholder logic
else:
    X_train = train_df[feature_columns]
    y_train = train_df[target_column]
    X_test = test_df[feature_columns]

# Set random seed for reproducibility
np.random.seed(42)
catboost.params['random_seed'] = 42

# Initialize CatBoost Regressor with RMSE loss function
# CatBoost automatically handles missing values and categorical columns
params = {
    'loss_function': 'RMSE',
    'objective': 'regression',
    'verbose': 0,
    'thread_count': -1,  # Use all available CPU cores
    'random_seed': 42,
    'n_estimators': 1000,
    'learning_rate': 0.05
}

model = cb.CatBoostRegressor(**params)

# Train the model
try:
    model.fit(X_train, y_train, eval_set=(X_test, None))
except Exception as e:
    # Fallback if training fails (though unlikely)
    model = cb.CatBoostRegressor(verbose=0, thread_count=1, random_seed=42)
    model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Prepare output dataframe
# Requirement: id (integer row index starting from 0) and target column
output_df = pd.DataFrame({
    'id': range(len(predictions)),
    'Supplier Contract Amount (USD)': predictions
})

# Save predictions to CSV in current working directory
output_df.to_csv('predictions.csv', index=False)