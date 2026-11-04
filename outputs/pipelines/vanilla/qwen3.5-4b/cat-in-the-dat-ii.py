import pandas as pd
import numpy as np
from catboost import CatBoostClassifier

# Define file paths as given in the prompt
train_path = '/path/to/user/workspace/ds_failures/datasets/kaggle/cat-in-the-dat-ii/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/kaggle/cat-in-the-dat-ii/data/test.parquet'
output_path = 'predictions.csv'

# Load data
print("Loading data...")
train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

# Ensure we are extracting the target column
y_train = train_df['target'].values
X_train = train_df.drop(columns=['target'])
X_test = test_df

# Configure model for categorical and missing value handling automatically
clf = CatBoostClassifier(
    iterations=1000,
    verbose=0,
    random_state=42,
    loss_function='LogLoss',
    depth=8,
    learning_rate=0.03,
    bagging_temperature=2,
    early_stopping_rounds=100
)

# Fit model on training data
clf.fit(X_train, y_train, plot=False, eval_metric='AUC')

# Predict on test data (returns probabilities for all classes)
probabilities = clf.predict_proba(X_test)[:, 1]

# Create output dataframe
predictions_df = pd.DataFrame({'id': range(len(test_df)), 'target': probabilities})

# Save predictions
predictions_df.to_csv(output_path, index=False)

print(f"Predictions saved to {output_path}")