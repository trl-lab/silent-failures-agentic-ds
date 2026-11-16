import pandas as pd
import numpy as np
from catboost import CatBoostRegressor

# Define paths to data
train_path = '/path/to/user/workspace/ds_failures/datasets/strable/ibrd-statement-loans-guarantees/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/strable/ibrd-statement-loans-guarantees/data/test.parquet'
target_col = 'Original Principal Amount (US$)'

# Load the datasets
train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

# Separate features and target
X_train = train_df.drop(columns=[target_col])
y_train = train_df[target_col]

# Apply model to predict
model = CatBoostRegressor(
    iterations=500,
    learning_rate=0.05,
    verbose=False,
    loss_function='RMSE'
)

model.fit(X_train, y_train)

# Make predictions on test set
y_pred = model.predict(test_df)

# Create predictions dataframe
id_range = list(range(len(test_df)))
result_df = pd.DataFrame({
    'id': id_range,
    target_col: y_pred
})

# Save predictions to CSV
result_df.to_csv('predictions.csv', index=False)