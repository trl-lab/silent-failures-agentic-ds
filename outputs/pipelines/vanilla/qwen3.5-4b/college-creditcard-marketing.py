import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDRegressor
import catboost as cb

train_path = '/path/to/user/workspace/ds_failures/datasets/strable/college-creditcard-marketing/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/strable/college-creditcard-marketing/data/test.parquet'

train_data = pd.read_parquet(train_path)
test_data = pd.read_parquet(test_path)

target_col = 'PAYMENTS_BY_ISSUER'
X_train = train_data.drop(columns=[target_col])
y_train = train_data[target_col]
X_test = test_data.copy()

# Identify numeric and categorical columns
numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

# Define transformers
numeric_transformer = SimpleImputer(strategy='median')
categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', numeric_transformer)
        ]), numeric_features),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='MISSING')),
            ('encoder', categorical_transformer)
        ]), categorical_features)
    ],
    remainder='passthrough'
)

# Build model pipeline
clf = cb.CatBoostRegressor(
    objective='RootMSE',
    metric='RMSE',
    learning_rate=0.05,
    n_estimators=1000,
    verbose=0,
    random_seed=42,
    early_stopping_rounds=100
)

full_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', clf)
])

# Fit model on train
full_pipeline.fit(X_train, y_train)

# Predict on test
y_pred = full_pipeline.predict(X_test)

# Prepare output dataframe with id
test_ids = list(range(len(test_data)))
output_df = pd.DataFrame({
    'id': test_ids,
    'PAYMENTS_BY_ISSUER': y_pred
})

# Save predictions
output_df.to_csv('predictions.csv', index=False)