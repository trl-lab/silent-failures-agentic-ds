import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor

# File paths
train_file = "/path/to/user/workspace/ds_failures/datasets/uci/superconductivity/data/train.parquet"
test_file = "/path/to/user/workspace/ds_failures/datasets/uci/superconductivity/data/test.parquet"

# Load datasets
df_train = pd.read_parquet(train_file)
df_test = pd.read_parquet(test_file)

# Target and features
target_col = "critical_temp"
feature_cols = [col for col in df_train.columns if col != target_col]

# Prepare training data
X_train = df_train[feature_cols].copy()
y_train = df_train[target_col].values

# Prepare test data
X_test = df_test[feature_cols].copy()
y_test = df_test[target_col].values

# Initialize and train model
model = LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=20,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    verbose=-1,
    subsample=0.8,
    colsample_bytree=0.8
)

# Train
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Create output dataframe with 'id' column starting from 0
output_df = pd.DataFrame({
    "id": np.arange(len(predictions)),
    "critical_temp": predictions
})

# Save predictions
output_df.to_csv("predictions.csv", index=False)