import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Configuration
TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/uci/drug-consumption/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/uci/drug-consumption/data/test.parquet"
OUTPUT_PATH = "predictions.csv"

# Load Data
train_df = pd.read_parquet(TRAIN_PATH)
test_df = pd.read_parquet(TEST_PATH)

# Define target column and exclude known target columns from features
target_col = 'cannabis'
all_targets = ['alcohol', 'amphet', 'amyl', 'benzos', 'caff', 'cannabis', 'choc', 
               'coke', 'crack', 'ecstasy', 'heroin', 'ketamine', 'legalh', 'lsd', 
               'meth', 'mushrooms', 'nicotine', 'semer', 'vsa']

# Create list of feature columns
feature_cols = [col for col in train_df.columns if col not in all_targets and col != 'id']

# Initialize Pipeline
preprocessor = ColumnTransformer(transformers=[
    ('num', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ], verbose=0), feature_cols)
], remainder='drop')

model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train
model.fit(train_df[feature_cols], train_df[target_col])

# Predict
test_df_features = test_df[[col for col in feature_cols if col in test_df.columns]]
predictions = model.predict(test_df_features)

# Save Results
output_df = pd.DataFrame({
    'id': range(len(predictions)),
    'cannabis': predictions
})
output_df.to_csv(OUTPUT_PATH, index=False)