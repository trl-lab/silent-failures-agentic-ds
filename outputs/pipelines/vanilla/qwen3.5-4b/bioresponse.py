import pandas as pd
import numpy as np
import warnings
from catboost import CatBoostClassifier
import os

# Suppress warnings for cleaner execution
warnings.filterwarnings("ignore")

# Define paths
train_path = '/path/to/user/workspace/ds_failures/datasets/kaggle/bioresponse/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/kaggle/bioresponse/data/test.parquet'

# Load data
df_train = pd.read_parquet(train_path)
df_test = pd.read_parquet(test_path)

# Identify features
target_column = 'Activity'
all_columns = list(df_train.columns)
feature_columns = [col for col in all_columns if col != target_column]

# Prepare dataframes
X_train = df_train[feature_columns]
y_train = df_train[target_column]
X_test = df_test[feature_columns]

# Initialize and Train Model
# CatBoost handles missing values and categorical features automatically
model = CatBoostClassifier(
    iterations=1000,
    depth=6,
    learning_rate=0.01,
    loss_function='Logloss',
    verbose=0,
    random_state=42
)

model.fit(X_train, y_train)

# Generate predictions (probabilities)
# LogLoss requires probability estimates for the positive class (Class 1)
proba = model.predict_proba(X_test)
positive_probs = proba[:, 1]

# Prepare output DataFrame
# Create 'id' column starting from 0 up to N
ids = np.arange(len(proba))
prediction_df = pd.DataFrame({
    'id': ids,
    'Activity': positive_probs
})

# Save predictions to current working directory
output_path = 'predictions.csv'
prediction_df.to_csv(output_path, index=False)