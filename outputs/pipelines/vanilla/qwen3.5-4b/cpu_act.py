import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
import lightgbm as lgb

# Load training and test data
train_data = pd.read_parquet('/path/to/user/workspace/ds_failures/datasets/openml/cpu_act/data/train.parquet')
test_data = pd.read_parquet('/path/to/user/workspace/ds_failures/datasets/openml/cpu_act/data/test.parquet')

# Define target
target_column = 'usr'

# Separate features and target
X_train = train_data.drop(columns=[target_column])
y_train = train_data[target_column]
X_test = test_data

# Identify columns
columns = X_train.columns.tolist()

# Initialize imputer to handle missing values
imputer = SimpleImputer(strategy='median')
X_train_filled = imputer.fit_transform(X_train)
X_test_filled = imputer.transform(X_test)

# Convert back to DataFrame for LightGBM (or pass arrays)
# LightGBM supports array input
dataset_train = lgb.Dataset(X_train_filled, label=y_train.values, feature_name=columns)
dataset_test = lgb.Dataset(X_test_filled, reference=dataset_train, feature_name=columns)

# Define model parameters
params = {'verbosity': -1, 'random_state': 42, 'seed': 42}
model = lgb.train(params, dataset_train, num_boost_round=1000)

# Predictions on test data
preds = model.predict(dataset_test.to_internal_dataset())

# Create output dataframe
output_df = pd.DataFrame({
    'id': range(len(preds)),
    'usr': preds
})

# Save to CSV
output_df.to_csv('predictions.csv', index=False)