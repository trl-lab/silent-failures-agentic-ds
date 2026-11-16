import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
import os

# Define paths
TRAIN_PATH = '/path/to/user/workspace/ds_failures/datasets/openml/madelon/data/train.parquet'
TEST_PATH = '/path/to/user/workspace/ds_failures/datasets/openml/madelon/data/test.parquet'
OUTPUT_FILE = 'predictions.csv'

# Load data
train_df = pd.read_parquet(TRAIN_PATH)
test_df = pd.read_parquet(TEST_PATH)

# Separate features and target
feature_cols = [col for col in train_df.columns if col != 'Class']
y_train = train_df['Class'].values
X_train = train_df[feature_cols].copy()
y_test = test_df['Class'].values
X_test = test_df[feature_cols].copy()

# Handle missing values automatically (impute with median)
for df in [X_train, X_test]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # Fill numeric missing values with median
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())
    
    # If there are object columns, we treat them as unknown or impute based on strategy
    # Since MADELON is described as continuous, we'll try to keep them if convertible
    # But to ensure model training success, we drop objects or coerce to numeric if possible
    # Strategy: Convert objects to numeric where possible, else fill with 0
    for col in categorical_cols:
        # Try to convert string object to numeric
        converted = pd.to_numeric(X_test[col], errors='coerce')
        if converted.isnull().any():
            filled_val = converted.mean() if not converted.all(np.nan) else 0
            X_test[col] = filled_val
        X_train[col] = converted.replace({np.nan: 0}, 0)

# Prepare final frames for model
X_train_final = X_train.astype('float32').to_numpy()
X_test_final = X_test.astype('float32').to_numpy()

# Train model
model = CatBoostClassifier(iterations=1000, verbose=False, task_name='SVM', loss_function='LogLoss')
model.fit(X_train_final, y_train)

# Make predictions
preds = model.predict(X_test_final)

# Prepare output dataframe with id (integer row index starting from 0)
output_df = pd.DataFrame({
    'id': range(len(test_df)),
    'Class': preds
})

# Ensure output directory exists (assumed current dir)
output_file_path = os.path.join(os.getcwd(), OUTPUT_FILE)
output_df.to_csv(output_file_path, index=False)