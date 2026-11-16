import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import catboost as cb

# Load data
train_path = '/path/to/user/workspace/ds_failures/datasets/strable/local-government-renewable-action/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/strable/local-government-renewable-action/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

# Define target column
target_col = 'Size of Purchase (MW)'

# Prepare features
train_target = train_df[target_col]
features = train_df.drop(columns=[target_col])
test_features = test_df.drop(columns=[target_col])

# Determine column types to decide preprocessing
# We want to handle missing and categorical automatically
# Cols are either numeric or object/category
numeric_cols = features.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = features.select_dtypes(include=['object']).columns.tolist()

# Function to preprocess specific columns
def create_transformer():
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    return ColumnTransformer(transformers=[
        ('num', num_transformer, numeric_cols),
        ('cat', cat_transformer, cat_cols)
    ])

transformer = create_transformer()

# Apply transformation to train set
# Note: We apply transform to get a consistent representation for X_train and X_test
# However, transformer.fit is done first.

# Fit the preprocessing transformer on training data
preprocessor = transformer
X_train = pd.DataFrame(preprocessor.transform(features))

# Predict
model = lgb.LGBMRegressor(random_state=42, learning_rate=0.1, n_estimators=100, subsample_for_bin=0.8, min_child_samples=20, objective='regression', metric='rmse', max_depth=-1, importance_type='gain')

# Fit the model
model.fit(X_train, train_target.values.flatten(), eval_set=(X_train, train_target.values.flatten()), verbose=False)

# Transform test set
X_test = preprocessor.transform(test_features)
predictions = model.predict(X_test)

# Prepare output
output_data = {
    'id': range(len(predictions)),
    'Size of Purchase (MW)': predictions
}

output_df = pd.DataFrame(output_data)
output_df.to_csv('predictions.csv', index=False)