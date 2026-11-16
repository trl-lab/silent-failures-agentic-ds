import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Define target and feature columns
target = "Amount in USD"
X_train = train_df.drop(columns=[target])
y_train = train_df[target]
X_test = test_df.drop(columns=[target])

# Split training data into train and validation sets
X_train_val, X_val, y_train_val, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

# Prepare datasets for LightGBM
train_data = lgb.Dataset(X_train_val, label=y_train_val)
val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

# Define model parameters
params = {"objective": "regression", "metric": "rmse", "verbose": -1}

# Train model with early stopping and logging
model = lgb.train(
    params,
    train_data,
    num_boost_round=1000,
    valid_sets=[val_data],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
)

# Evaluate on validation set
y_val_pred = model.predict(X_val)
rmse = mean_squared_error(y_val, y_val_pred, squared=False)
print(f"Validation RMSE: {rmse:.6f}")

# Predict on test set
y_test_pred = model.predict(X_test)

# Prepare submission file
submission = pd.DataFrame({"id": range(len(y_test_pred)), "Amount in USD": y_test_pred})

# Save submission
submission.to_csv("./working/submission.csv", index=False)
