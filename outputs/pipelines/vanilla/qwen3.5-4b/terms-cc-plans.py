import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from lightgbm import LGBMRegressor

# File paths
train_path = '/path/to/user/workspace/ds_failures/datasets/strable/terms-cc-plans/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/strable/terms-cc-plans/data/test.parquet'
output_path = 'predictions.csv'

# Load data
df_train = pd.read_parquet(train_path)
df_test = pd.read_parquet(test_path)

# Target column
target_col = 'APR'

# Prepare features
X_train = df_train.drop(columns=[target_col])
X_test = df_test.drop(columns=[target_col])
y_train = df_train[target_col]

# Identify feature types
num_cols = []
cat_cols = []

for col in df_train.columns:
    dtype = X_train[col].dtype
    if pd.api.types.is_numeric_dtype(dtype):
        num_cols.append(col)
    else:
        cat_cols.append(col)

# Define transformers
num_transf = Pipeline([
    ('imputer', SimpleImputer(strategy='median'))
])

cat_transf = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transf, num_cols),
        ('cat', cat_transf, cat_cols)
    ],
    remainder='drop'
)

# Model
model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', LGBMRegressor(random_state=42, verbose=-1, n_estimators=1000))
])

# Train
model.fit(X_train, y_train)

# Predict
X_test_transformed = model.named_steps['preprocessor'].transform(X_test)
preds = model.named_steps['regressor'].predict(X_test_transformed)

# Prepare output dataframe with id
results_df = pd.DataFrame({
    'id': range(len(df_test)),
    'APR': preds
})

# Save
results_df.to_csv(output_path, index=False)