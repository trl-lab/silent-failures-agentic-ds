import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor

# Define file paths
train_path = '/path/to/user/workspace/ds_failures/datasets/strable/industry-payments-project/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/strable/industry-payments-project/data/test.parquet'

# Load data
train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

# Define target column
target_col = 'projectPayment'
features_df = train_df.drop(columns=[target_col])
test_features = test_df.copy()

# Get target vector for training
y_train = train_df[target_col]

# Helper function to process dataframes
def prepare_data(df):
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Handle missing values in numerical columns
    for col in numerical_cols:
        df[col] = df[col].fillna(df[col].median())
    
    # Handle missing values in categorical columns
    for col in categorical_cols:
        df[col] = df[col].fillna('__NULL_CATEGORICAL__')
    
    # Convert categorical columns to category type for LightGBM efficiency
    df[categorical_cols] = df[categorical_cols].astype('category')
    
    return df, categorical_cols

# Prepare training and test data
X_train, train_cats = prepare_data(features_df)
X_test, _ = prepare_data(test_features)

# Initialize LightGBM model
model = LGBMRegressor(
    objective='regression',
    metric='rmse',
    max_depth=6,
    num_leaves=31,
    learning_rate=0.05,
    n_estimators=1000,
    random_state=42,
    categorical_feature=train_cats,
    force_cat_weights=False,
    verbose=-1
)

# Train model
model.fit(X_train, y_train)

# Predict on test data
predictions = model.predict(X_test)

# Create output dataframe with 'id' column
output_df = pd.DataFrame({
    'id': range(len(predictions)),
    'projectPayment': predictions
})

# Save predictions
output_df.to_csv('predictions.csv', index=False)