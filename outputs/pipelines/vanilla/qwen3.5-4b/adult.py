import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Load the datasets
train_path = '/path/to/user/workspace/ds_failures/datasets/uci/adult/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/uci/adult/data/test.parquet'

try:
    train_df = pd.read_parquet(train_path)
except Exception:
    raise FileNotFoundError("Training parquet file not found.")

try:
    test_df = pd.read_parquet(test_path)
except Exception:
    raise FileNotFoundError("Test parquet file not found.")

# Define target column
target_column = 'income'

# Separate target and features
y_train = train_df[target_column].astype(str)
X_train = train_df.drop(columns=[target_column])

y_test = test_df[target_column].astype(str)
X_test = test_df.drop(columns=[target_column])

# Initialize CatBoost Classifier with parameters for mixed data
model = catboost.CatBoostClassifier(iterations=1000, depth=6, learning_rate=0.05, verbose=False, random_seed=42, loss_function='Logloss')

# Fit the model
model.fit(X_train, y_train)

# Predict on test set
predictions = model.predict(X_test)

# Prepare output dataframe
output_df = pd.DataFrame({
    'id': range(len(predictions)),
    'income': predictions
})

# Save predictions
output_path = 'predictions.csv'
output_df.to_csv(output_path, index=False)

print(f"Predictions saved to {output_path}")