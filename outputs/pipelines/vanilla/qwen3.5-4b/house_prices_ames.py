import pandas as pd
import numpy as np
import catboost as cb

# Define file paths
train_path = '/path/to/user/workspace/ds_failures/datasets/openml/house_prices_ames/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/openml/house_prices_ames/data/test.parquet'

# Load datasets
df_train = pd.read_parquet(train_path)
df_test = pd.read_parquet(test_path)

# Identify columns
target_column = 'Sale_Price'
feature_columns = [col for col in df_train.columns if col != target_column]

# Prepare data
X_train = df_train[feature_columns].copy()
y_train = df_train[target_column].values
X_test = df_test[feature_columns].copy()

# Initialize and train the model
# CatBoost handles categorical and missing values automatically
model = cb.CatBoostRegressor(
    iterations=500,
    learning_rate=0.05,
    depth=6,
    verbose=0,
    random_state=42
)
model.fit(X_train, y_train)

# Generate predictions
predictions = model.predict(X_test)

# Create result dataframe with 'id' starting from 0
result_df = pd.DataFrame({
    'id': range(len(predictions)),
    'Sale_Price': predictions
})

# Save to current working directory
result_df.to_csv('predictions.csv', index=False)