import pandas as pd
import numpy as np
from xgboost import XGBClassifier

# Define file paths as provided in the problem description
train_path = '/path/to/user/workspace/ds_failures/datasets/openml/wall-robot-navigation/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/openml/wall-robot-navigation/data/test.parquet'

# Load the datasets
df_train = pd.read_parquet(train_path)
df_test = pd.read_parquet(test_path)

# Identify feature columns (exclude target 'class')
# Assuming all columns except 'class' are features
feature_columns = [col for col in df_train.columns if col != 'class']

# Extract features and target
X_train = df_train[feature_columns].copy()
y_train = df_train['class'].values
X_test = df_test[feature_columns].copy()

# Handle missing values automatically using median imputation for numeric columns
# This covers cases where sensor data might have NaN entries
X_train = X_train.fillna(X_train.median())
X_test = X_test.fillna(X_test.median())

# Initialize and train the model
# Parameters tuned for robustness on tabular data
model = XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbosity=0
)

model.fit(X_train, y_train)

# Make predictions on the test set
predictions = model.predict(X_test)

# Prepare output dataframe with 'id' and 'class'
output_data = pd.DataFrame({
    'id': range(len(predictions)),
    'class': predictions
})

# Save predictions to csv file
output_data.to_csv('predictions.csv', index=False)