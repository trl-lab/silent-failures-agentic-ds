import pandas as pd
import numpy as np
import catboost as cb

train_path = "/path/to/user/workspace/ds_failures/datasets/strable/china-overseas-finance-inventory/data/train.parquet"
test_path = "/path/to/user/workspace/ds_failures/datasets/strable/china-overseas-finance-inventory/data/test.parquet"

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = "installed_capacity"
exclude_cols = ["installed_capacity"]

# Remove 'id' column if it exists in training data to prevent leakage, 
# although 'id' in test set is handled separately for output.
if "id" in train_df.columns:
    exclude_cols.append("id")

X_train = train_df.drop(columns=exclude_cols)
y_train = train_df[target_col]
X_test = test_df.drop(columns=exclude_cols)

# Initialize CatBoost Regressor
# CatBoost handles missing values and categorical columns automatically by default.
model = cb.CatBoostRegressor(random_state=42, verbose=False)

# Fit the model
model.fit(X_train, y_train)

# Predict on test set
predictions = model.predict(X_test)

# Prepare output dataframe with id starting from 0
output_df = pd.DataFrame({
    "id": list(range(len(X_test))),
    "installed_capacity": predictions
})

# Save predictions to csv
output_df.to_csv("predictions.csv", index=False)