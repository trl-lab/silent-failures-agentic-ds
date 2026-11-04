import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier

# File paths as provided in the dataset description
train_file = '/path/to/user/workspace/ds_failures/datasets/uci/electrical-grid-stability/data/train.parquet'
test_file = '/path/to/user/workspace/ds_failures/datasets/uci/electrical-grid-stability/data/test.parquet'

# Load the data
train_df = pd.read_parquet(train_file)
test_df = pd.read_parquet(test_file)

# Define target column
target_col = 'stabf'

# Extract feature columns (all columns except the target)
feature_cols = [col for col in train_df.columns if col != target_col]

# Prepare training and testing datasets
X_train = train_df[feature_cols].copy()
y_train = train_df[target_col].copy()
X_test = test_df[feature_cols].copy()

# Handle missing values (optional fill strategy, defaults to None in LGBM if configured, but we ensure consistency)
# LightGBM handles missing values automatically, but filling ensures numerical stability
X_train = X_train.fillna(np.nan)
X_test = X_test.fillna(np.nan)

# Initialize and train the model
model = LGBMClassifier(
    n_estimators=500,
    learning_rate=0.1,
    num_leaves=31,
    min_data_in_leaf=20,
    max_depth=-1,
    random_state=42,
    verbose=-1
)

model.fit(X_train, y_train)

# Make predictions
y_pred_proba = model.predict(X_test)

# Convert probabilities to binary predictions based on threshold
y_pred_classes = (y_pred_proba >= 0.5).astype(int)

# Create output dataframe with 'id' and 'stabf' columns
output_df = pd.DataFrame({
    'id': range(len(output_df := pd.DataFrame())),
    'stabf': y_pred_classes
})

# Correctly construct the ID and label dataframe based on test set length
ids = list(range(len(X_test)))
predictions_df = pd.DataFrame({
    'id': ids,
    'stabf': y_pred_classes
})

# Save to CSV in the current working directory
predictions_df.to_csv('predictions.csv', index=False)