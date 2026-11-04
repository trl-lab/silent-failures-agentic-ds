import pandas as pd
import numpy as np
from catboost import CatBoostClassifier

# Define paths
TRAIN_PATH = '/path/to/user/workspace/ds_failures/datasets/openml/bank-marketing/data/train.parquet'
TEST_PATH = '/path/to/user/workspace/ds_failures/datasets/openml/bank-marketing/data/test.parquet'

# Load data
df_train = pd.read_parquet(TRAIN_PATH)
df_test = pd.read_parquet(TEST_PATH)

# Separate features and target
# Based on prompt: Target column is 'Class'
y_train = df_train['Class']
X_train = df_train.drop(columns=['Class'])

y_test = None  # Not needed for prediction
X_test = df_test.copy()

# Initialize model
# catboost handles categorical columns (strings) automatically and missing values
model = CatBoostClassifier(
    iterations=200,
    depth=6,
    learning_rate=0.05,
    loss_function='Logloss',
    verbose=False
)

# Train model
model.fit(X_train, y_train)

# Make predictions on test set
preds = model.predict(X_test)

# Convert probabilities to class labels (binary classification)
# CatBoost returns classes (integers for multiclass, probabilities for prob)
# Assuming binary 'yes'/'no' -> 1/0
# preds is array of labels (0/1) or probabilities.
# Need to check return type. predict() returns labels.
# Map labels back to 'yes'/'no' based on class mapping.
# We need to know unique classes.
unique_classes = sorted(list(np.unique(y_train)))
class_to_idx = {c: i for i, c in enumerate(unique_classes)}
idx_to_class = {i: c for i, c in enumerate(unique_classes)}

# Get actual predicted labels from model predict()
# If predict returns probabilities, we round. Usually predict returns class ids.
# Let's use .predict_proba() to determine the max class then map it.
probs = model.predict_proba(X_test)
predicted_indices = np.argmax(probs, axis=1)
predicted_classes = [idx_to_class[i] for i in predicted_indices]

# Prepare output dataframe
result_df = pd.DataFrame({
    'id': range(len(predicted_classes)),
    'Class': predicted_classes
})

# Save predictions
result_df.to_csv('predictions.csv', index=False)