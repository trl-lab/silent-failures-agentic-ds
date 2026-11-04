import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
import lightgbm as lgb

# Configuration
TRAIN_PATH = '/path/to/user/workspace/ds_failures/datasets/strable/workforce-demographics-wide/data/train.parquet'
TEST_PATH = '/path/to/user/workspace/ds_failures/datasets/strable/workforce-demographics-wide/data/test.parquet'
TARGET_COL = 'Female (%)'
PREDICTIONS_PATH = 'predictions.csv'

# Load data
print("Loading data...")
train_df = pd.read_parquet(TRAIN_PATH)
test_df = pd.read_parquet(TEST_PATH)

# Identify columns
y_col = TARGET_COL
X_cols = [col for col in train_df.columns if col != y_col]

# Extract features
X_train = train_df[X_cols].copy()
y_train = train_df[y_col].copy()
X_test = test_df[X_cols].copy()

# Determine numeric and categorical columns
num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

# Define preprocessing pipeline
# Handle missing values with median for numeric, mode or constant fill for categorical
preprocessor = ColumnTransformer(transformers=[
    ('num', Pipeline(steps=[
        ('impute', SimpleImputer(strategy='median')),
        ('scale', FunctionTransformer(lambda x: x / x.max(), passes=None)) # Simplified scaling or omit if not needed, let's skip scaling for robustness on skewed data unless specified
    ]), num_cols),
    ('cat', OneHotEncoder(categories='auto', sparse_output=False, handle_unknown='ignore'), cat_cols)
])

# Define model (LightGBM)
model = lgb.LGBMRegressor(random_state=42, n_estimators=500, learning_rate=0.05, verbose=-1, random_state=42)

# Prepare full pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', model)])

# Fit model
print("Training model...")
try:
    pipeline.fit(X_train, y_train)
except Exception as e:
    print(f"Error fitting model: {e}")
    raise

# Make predictions on test set
print("Making predictions...")
y_pred = pipeline.predict(X_test)

# Handle predictions data type
predictions_df = pd.DataFrame({
    'id': range(len(y_pred)),
    'Female (%)': y_pred
})

# Ensure column names match exactly
predictions_df.rename(columns={'id': 'id', 'Female (%)': 'Female (%)'}, inplace=True)

# Save predictions
print(f"Saving predictions to {PREDICTIONS_PATH}...")
predictions_df.to_csv(PREDICTIONS_PATH, index=False)

print("Done.")