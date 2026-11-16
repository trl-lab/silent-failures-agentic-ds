import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
import lightgbm as lgb

# Load data
train_df = pd.read_parquet('/path/to/user/workspace/ds_failures/datasets/openml/energy-efficiency/data/train.parquet')
test_df = pd.read_parquet('/path/to/user/workspace/ds_failures/datasets/openml/energy-efficiency/data/test.parquet')

# Define target variable
target_col = 'heating_load'
feature_columns = [col for col in train_df.columns if col != target_col]

# Identify categorical and numeric columns
numeric_cols = []
categorical_cols = []

for col in feature_columns:
    if train_df[col].dtype == 'object':
        categorical_cols.append(col)
    elif train_df[col].dtype == 'int64' or train_df[col].dtype == 'float64':
        numeric_cols.append(col)
    else:
        # Fallback for other numeric types
        numeric_cols.append(col)

# Prepare preprocessors
imp = SimpleImputer(strategy='median')
ohe = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', imp, numeric_cols),
        ('cat', ohe, categorical_cols)
    ],
    remainder='drop'
)

# Define model
model = lgb.LGBMRegressor(random_state=42, n_estimators=100, learning_rate=0.1)

# Fit pipeline on training data
pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('estimator', model)])
X_train = train_df[feature_columns]
y_train = train_df[target_col]

pipeline.fit(X_train, y_train)

# Transform test data and predict
X_test = test_df[feature_columns]
y_pred = pipeline.predict(X_test)

# Prepare output dataframe
output_df = pd.DataFrame({
    'id': range(len(y_pred)),
    'heating_load': y_pred
})

# Save predictions
output_df.to_csv('predictions.csv', index=False)