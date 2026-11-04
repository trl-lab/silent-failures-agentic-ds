import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from lightgbm import LGBMClassifier

train_path = '/path/to/user/workspace/ds_failures/datasets/strable/device-classification/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/strable/device-classification/data/test.parquet'
output_path = 'predictions.csv'

df_train = pd.read_parquet(train_path)
df_test = pd.read_parquet(test_path)

target_col = 'device_class'
feature_columns = [col for col in df_train.columns if col != target_col]

# Identify numeric and categorical columns
numeric_features = df_train[feature_columns].select_dtypes(include=[np.number]).columns.tolist()
categorical_features = df_train[feature_columns].select_dtypes(include=['object', 'string']).columns.tolist()

# Preprocessor
preprocessor = ColumnTransformer(transformers=[
    ('num', Pipeline(steps=[('imputer', SimpleImputer(strategy='median'))]), numeric_features),
    ('cat', Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), categorical_features)
], remainder="drop")

# Train
X_train, y_train = df_train[feature_columns], df_train[target_col]
try:
    X_train_processed = preprocessor.fit_transform(X_train)
except Exception as e:
    print(f"Error fitting preprocessor: {e}")
    raise

model = LGBMClassifier(random_state=42, objective='multinomial', num_leaves=31)
model.fit(X_train_processed, y_train.values)

# Predict
X_test_processed = preprocessor.transform(df_test[feature_columns])
y_pred = model.predict(X_test_processed)

# Format Output
results = pd.DataFrame({
    'id': range(len(y_pred)),
    'device_class': y_pred
})
results.to_csv(output_path, index=False)