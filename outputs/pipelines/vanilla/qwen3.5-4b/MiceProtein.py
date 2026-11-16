import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression # Fallback if needed, but use LGB
from lightgbm import LGBMClassifier

# File paths
TRAIN_PATH = 'train.parquet'
TEST_PATH = 'test.parquet'

# Load data
df_train = pd.read_parquet(TRAIN_PATH)
df_test = pd.read_parquet(TEST_PATH)

# Identify target column
TARGET_COLUMN = 'class'

# Prepare features and target
y_train = df_train[TARGET_COLUMN]
X_train = df_train.drop(TARGET_COLUMN, axis=1)

y_test = df_test[TARGET_COLUMN]
X_test = df_test.drop(TARGET_COLUMN, axis=1)

# Define preprocessing
numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

transformer = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='median'), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='drop'
)

pipeline = Pipeline(steps=[
    ('preprocess', transformer),
    ('model', LGBMClassifier(objective='multiclass', verbose=-1, random_state=42))
])

# Train model
pipeline.fit(X_train, y_train)

# Predict
y_pred_proba = pipeline.predict(X_test)

# Create output dataframe
output_df = pd.DataFrame({
    'id': range(len(y_pred_proba)),
    'class': y_pred_proba
})

# Save predictions
output_df.to_csv('predictions.csv', index=False)