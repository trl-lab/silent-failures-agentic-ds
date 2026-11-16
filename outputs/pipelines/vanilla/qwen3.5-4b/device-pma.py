import pandas as pd
import numpy as np
from catboost import CatBoostClassifier

# Set random seed for reproducibility
np.random.seed(42)

# Define file paths
train_path = "/path/to/user/workspace/ds_failures/datasets/strable/device-pma/data/train.parquet"
test_path = "/path/to/user/workspace/ds_failures/datasets/strable/device-pma/data/test.parquet"

# Load data
try:
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)
except Exception as e:
    raise RuntimeError(f"Failed to load data: {e}")

# Identify features and target
feature_columns = [col for col in train_df.columns if col != 'decision_code']

X_train = train_df[feature_columns]
y_train = train_df['decision_code']
X_test = test_df[feature_columns]

# Initialize model
# CatBoost handles categorical and missing values automatically without explicit preprocessing
model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.1,
    depth=6,
    verbose=False,
    random_state=42
)

# Train model
try:
    model.fit(X_train, y_train)
except Exception as e:
    raise RuntimeError(f"Failed to train model: {e}")

# Make predictions
predictions = model.predict(X_test)

# Prepare output dataframe
output_df = pd.DataFrame({
    "id": range(len(predictions)),
    "decision_code": predictions
})

# Save to CSV in the current working directory
output_path = "predictions.csv"
output_df.to_csv(output_path, index=False)