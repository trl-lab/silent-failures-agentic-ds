import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# 1. Load Data
path_train = '/path/to/user/workspace/ds_failures/datasets/strable/pol-terminal/data/train.parquet'
path_test = '/path/to/user/workspace/ds_failures/datasets/strable/pol-terminal/data/test.parquet'

train_df = pd.read_parquet(path_train)
test_df = pd.read_parquet(path_test)

# 2. Preprocessing Setup
target_col = 'CAPACITY'
X_train = train_df.drop(columns=[target_col])
X_test = test_df.drop(columns=[target_col])
y_train = train_df[target_col]

# Automatically identify column types to handle missing values and categoricals
numeric_features = X_train.select_dtypes(include=['number']).columns.tolist()
categorical_features = X_train.select_dtypes(exclude=['number']).columns.tolist()

# Configure transformers
imputer = SimpleImputer(strategy='median')
ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False, dtype=np.float32)

ct = ColumnTransformer(
    transformers=[
        ('num', imputer, numeric_features),
        ('cat', ohe, categorical_features)
    ],
    remainder='drop'
)

# Build Pipeline
# Using LightGBM for speed and efficiency
model = Pipeline([
    ('preprocessor', ct),
    ('model', lgb.LGBMRegressor(
        objective='regression',
        learning_rate=0.05,
        n_estimators=500,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
        device='cpu'
    ))
])

# 3. Train Model
model.fit(X_train, y_train)

# 4. Generate Predictions
predictions = model.predict(X_test)

# 5. Prepare Output DataFrame
# "id" (integer row index starting from 0)
output_df = pd.DataFrame({
    'id': range(len(test_df)),
    'CAPACITY': predictions
})

# 6. Save Results
output_df.to_csv('predictions.csv', index=False)