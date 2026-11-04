import os
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import catboost as cb

# Define paths as specified in the dataset description
TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/openml/kc1/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/openml/kc1/data/test.parquet"
OUTPUT_PATH = "predictions.csv"

# Load data
train_df = pd.read_parquet(TRAIN_PATH)
test_df = pd.read_parquet(TEST_PATH)

# Identify target column and map boolean/strings to binary
target_col = 'defects'
train_y = train_df[target_col].map({'false': 0, 'true': 1}).values
test_y = test_df[target_col].map({'false': 0, 'true': 1}).values

# Prepare feature matrices
train_X = train_df.drop(columns=[target_col])
test_X = test_df.drop(columns=[target_col])

# Handle missing values automatically
train_X = train_X.fillna(train_X.median())
test_X = test_X.fillna(test_X.median())

# Initialize Model
model = cb.CatBoostClassifier(
    iterations=1000,
    learning_rate=0.1,
    depth=6,
    random_state=42,
    loss_function='Logloss',
    eval_metric='AUC',
    verbose=False
)

# Train model
model.fit(
    train_X.values,
    train_y,
    eval_set=[(test_X.values, test_y)],
    cat_features=[]
)

# Make predictions
# We need to predict classes (0 or 1), so we take argmax or thresholded probabilities
# Default CatBoost predict_proba returns probabilities, we want class labels.
# Using method='OneVsRest' internally or direct prediction.
# catboost predicts probabilities. We'll threshold at 0.5.
# Actually, let's use predict() for direct class prediction if available, but predict() sometimes gives classes directly.
# However, for AUC optimization, predicting probabilities is standard.
# We will generate the class label based on max probability.
y_pred_proba = model.predict_proba(test_X.values)[:, 1]
y_pred = (y_pred_proba >= 0.5).astype(int)

# Reset index to ensure id starts from 0 and matches row count
output_df = pd.DataFrame({
    'id': np.arange(len(test_df)),
    'defects': y_pred
})

# Save to CSV
output_df.to_csv(OUTPUT_PATH, index=False)