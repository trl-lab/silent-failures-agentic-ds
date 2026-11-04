import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from lightgbm import LGBMClassifier

# Define paths based on the prompt specifications
train_path = "/path/to/user/workspace/ds_failures/datasets/openml/satimage/data/train.parquet"
test_path = "/path/to/user/workspace/ds_failures/datasets/openml/satimage/data/test.parquet"

# Load data
df_train = pd.read_parquet(train_path)
df_test = pd.read_parquet(test_path)

# Identify target and features
target = 'class'
X_train = df_train.drop(columns=[target])
y_train = df_train[target]

X_test = df_test.drop(columns=[target])
y_test = df_test[target]

# Automatically handle categorical columns by converting to numeric (handles strings/normals)
for col in X_train.columns:
    X_train[col] = pd.to_numeric(X_train[col], errors='coerce')
for col in X_test.columns:
    X_test[col] = pd.to_numeric(X_test[col], errors='coerce')

# Handle missing values automatically
imputer = SimpleImputer(strategy='median')
X_train_imp = imputer.fit_transform(X_train.values)
X_test_imp = imputer.transform(X_test.values)

# Train model
model = LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
model.fit(X_train_imp, y_train)

# Generate predictions
preds = model.predict(X_test_imp)

# Create output dataframe with 'id' starting from 0
ids = list(range(len(preds)))
result = pd.DataFrame({'id': ids, 'class': preds})

# Save predictions
result.to_csv('predictions.csv', index=False)