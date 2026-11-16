import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
import lightgbm as lgb

# Paths
train_path = '/path/to/user/workspace/ds_failures/datasets/strable/cosmetic-event/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/strable/cosmetic-event/data/test.parquet'

# Load Data
train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

# Target column
target_col = 'outcomes'

# Initialize Label Encoder for target
le = LabelEncoder()
y_train = le.fit_transform(train_df[target_col].values)

# Feature Identification
categorical_cols = []
numerical_cols = []

for col in train_df.columns:
    if col == target_col:
        continue
    if pd.api.types.is_object_dtype(train_df[col]):
        categorical_cols.append(col)
    elif pd.api.types.is_categorical_dtype(train_df[col]):
        categorical_cols.append(col)
    elif pd.api.types.is_numeric_dtype(train_df[col]):
        numerical_cols.append(col)

# Define Transformers
numeric_transformation = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
])

categorical_transformation = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
])

# Column Transformer setup
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformation, numerical_cols),
        ('cat', categorical_transformation, categorical_cols),
    ],
    remainder='drop',  # Drop unknown columns
)

# Define Model
model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', lgb.LGBClassifier(objective='multiclass', num_class=np.max(y_train)+1, random_state=42)),
])

# Fit Model
model.fit(train_df.drop(columns=[target_col]), y_train)

# Prepare Test Input
# Ensure test data target is not used
test_X = test_df.drop(columns=[target_col])

# Ensure categorical columns are kept if missing in test (unlikely but safe)
# Apply preprocessor
X_processed = model.named_steps['preprocessor'].transform(test_X)

# Predict
y_pred_indices = model.named_steps['classifier'].predict(X_processed)
y_pred_original = le.inverse_transform(y_pred_indices)

# Create Output DataFrame
output_df = pd.DataFrame({
    'id': range(len(test_df)),
    'outcomes': y_pred_original
})

# Ensure types
output_df['id'] = output_df['id'].astype(np.int64)

# Save
output_df.to_csv('predictions.csv', index=False)