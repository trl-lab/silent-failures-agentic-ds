import pandas as pd
import numpy as np
import catboost as cb

# Define file paths as specified in the problem description
train_path = "/path/to/user/workspace/ds_failures/datasets/openml/diamonds/data/train.parquet"
test_path = "/path/to/user/workspace/ds_failures/datasets/openml/diamonds/data/test.parquet"

# Load data
train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

# Separate target and features
target_col = 'price'
feature_cols = [col for col in train_df.columns if col != target_col]
X_train = train_df[feature_cols]
y_train = train_df[target_col]
X_test = test_df[feature_cols]

# Initialize CatBoost Regressor
# Handles missing values and categorical encoding automatically
model = cb.CatBoostRegressor(
    objective='RMSE',
    verbose=False,
    random_seed=42,
    dev_iteration_counter=True, # Helps internal iteration tracking if needed
    learning_rate=0.05
)

# Train model
model.fit(X_train, y_train)

# Predict on test set
predictions = model.predict(X_test)

# Create output dataframe with 'id' starting from 0 and 'price'
result_df = pd.DataFrame({
    'id': range(len(predictions)),
    'price': predictions
})

# Save to predictions.csv
result_df.to_csv("predictions.csv", index=False)