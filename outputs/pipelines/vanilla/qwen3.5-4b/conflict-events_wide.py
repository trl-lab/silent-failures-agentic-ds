import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import lightgbm as lgb

# Define file paths
train_file = '/path/to/user/workspace/ds_failures/datasets/strable/conflict-events_wide/data/train.parquet'
test_file = '/path/to/user/workspace/ds_failures/datasets/strable/conflict-events_wide/data/test.parquet'
output_file = 'predictions.csv'

# Load data
train_df = pd.read_parquet(train_file)
test_df = pd.read_parquet(test_file)

# Define target column
target_col = 'fatalities'

# Prepare Feature Matrix
X_train = train_df.drop(columns=[target_col])
y_train = train_df[target_col]
X_test = test_df.drop(columns=[target_col])

# Ensure data types are handled for preprocessing
# Identify numerical and categorical columns automatically
numerical_columns = X_train.select_dtypes(include=[np.number]).columns.tolist()
categorical_columns = X_train.select_dtypes(include=['object', 'bool']).columns.tolist()

# Initialize transformers
# 1. Imputer for numerical columns
imputer = SimpleImputer(strategy='median')

# 2. OneHotEncoder for categorical columns
# handle_unknown='ignore' helps with test sets having different categories
ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

# Combine into ColumnTransformer
preprocessor = ColumnTransformer(transformers=[
    ('num', imputer, numerical_columns),
    ('cat', ohe, categorical_columns)
])

# Build Pipeline
pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('model', lgb.LGBRegressor(objective='regression', random_state=42, verbosity=-1))
])

# Fit the model
pipeline.fit(X_train, y_train)

# Predict on test set
# We use .predict() on the fitted pipeline
y_pred = pipeline.predict(X_test)

# Create output dataframe with 'id' and 'fatalities'
# The 'id' is the integer row index starting from 0
output_data = pd.DataFrame({
    'id': np.arange(len(y_pred)),
    'fatalities': y_pred
})

# Save predictions to CSV
output_data.to_csv(output_file, index=False)