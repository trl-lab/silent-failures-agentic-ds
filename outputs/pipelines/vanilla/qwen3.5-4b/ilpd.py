import pandas as pd
import numpy as np
import catboost as cb

# Load datasets
train_path = "/path/to/user/workspace/ds_failures/datasets/openml/ilpd/data/train.parquet"
test_path = "/path/to/user/workspace/ds_failures/datasets/openml/ilpd/data/test.parquet"

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

# Identify target column
target_col = 'Class'

# Prepare training data
X_train = train_df.drop(columns=[target_col])
y_train = train_df[target_col]

# Ensure consistent index names, drop 'index' if present in parquet columns to avoid confusion
if 'index' in X_train.columns:
    X_train = X_train.drop(columns=['index'])

if 'index' in test_df.columns:
    test_df = test_df.drop(columns=['index'])

# Prepare test data (ensure 'Class' is dropped for prediction training)
X_test = test_df.drop(columns=[target_col])

# Combine train and test to detect feature types and handle missing values
# CatBoost can handle missing and categorical automatically if columns are not converted incorrectly
# However, to ensure compatibility with the installed libraries and automatic handling,
# we will apply basic preprocessing for consistency.
# CatBoost handles object columns as categorical and numeric as numeric automatically.

# Initialize CatBoost Classifier
model = cb.CatBoostClassifier(
    iterations=100,
    depth=5,
    loss_function="Logloss",
    verbose=False,
    random_state=42
)

# Train model
model.fit(X_train, y_train)

# Make predictions
y_pred_prob = model.predict_proba(X_test)[:, 1]
# Threshold predictions based on majority class or probability
# Assuming class 0 and 1 or similar, we threshold at 0.5
y_pred_classes = (y_pred_prob >= 0.5).astype(int).astype(str)

# Generate IDs for output
id_series = range(len(y_pred_classes))
# Create DataFrame for output
output_df = pd.DataFrame({'id': id_series, 'Class': y_pred_classes})

# Save to CSV
output_df.to_csv("predictions.csv", index=False)