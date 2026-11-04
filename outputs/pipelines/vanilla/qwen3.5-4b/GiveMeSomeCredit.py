import pandas as pd
import numpy as np
import catboost as cb

# Load datasets
train_df = pd.read_parquet('/path/to/user/workspace/ds_failures/datasets/kaggle/GiveMeSomeCredit/data/train.parquet')
test_df = pd.read_parquet('/path/to/user/workspace/ds_failures/datasets/kaggle/GiveMeSomeCredit/data/test.parquet')

# Define target column
target_col = 'SeriousDlqin2yrs'

# Prepare Training Data
X_train = train_df.drop(columns=[target_col])
y_train = train_df[target_col]

# Handle missing values in target (drop rows if needed, though typically handled internally)
mask = y_train.notna()
X_train = X_train[mask].reset_index(drop=True)
y_train = y_train[mask].reset_index(drop=True)

# Prepare Test Data
X_test = test_df.copy()

# Initialize CatBoost Classifier (handles categories and missing values automatically)
model = cb.CatBoostClassifier(
    verbosity=0,
    iterations=1000,
    eval_metric='AUC',
    random_state=42,
    loss_function='Logloss',
    enable_catboost_log=False
)

# Train model
model.fit(
    X_train,
    y_train,
    eval_set=(X_test, None),
    verbose=False
)

# Generate predictions (proba for target)
probs = model.predict_proba(X_test)[:, 1]

# Create output dataframe with required columns
result_df = pd.DataFrame({
    'id': range(len(test_df)),
    'SeriousDlqin2yrs': probs
})

# Save predictions
result_df.to_csv('predictions.csv', index=False)