import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Create working directory if it doesn't exist
os.makedirs("./working", exist_ok=True)

# Load data
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

# Separate target
target = "Strucht"
y_train = train[target]
X_train = train.drop(columns=[target])

# Identify categorical columns
categorical_cols = [
    col
    for col in X_train.columns
    if X_train[col].dtype == "object" or X_train[col].dtype.name == "category"
]

# Preprocess categorical columns: align categories between train and test
for col in categorical_cols:
    # Convert train column to category and get categories
    X_train[col] = X_train[col].astype("category")
    train_categories = X_train[col].cat.categories

    # Convert test column to category with the same categories
    test[col] = test[col].astype("category").cat.set_categories(train_categories)

# Split training data into train and validation sets
X_train_part, X_val_part, y_train_part, y_val_part = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

# Define LightGBM parameters
params = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "min_child_samples": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
    "seed": 42,
}

# Create datasets
train_dataset = lgb.Dataset(
    X_train_part, label=y_train_part, categorical_feature=categorical_cols
)
val_dataset = lgb.Dataset(
    X_val_part,
    label=y_val_part,
    categorical_feature=categorical_cols,
    reference=train_dataset,
)

# Train model with early stopping
model = lgb.train(
    params,
    train_dataset,
    num_boost_round=10000,
    valid_sets=[val_dataset],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
)

# Get best iteration
best_iteration = model.best_iteration

# Evaluate on validation set
y_val_pred = model.predict(X_val_part, num_iteration=best_iteration)
rmse_val = np.sqrt(mean_squared_error(y_val_part, y_val_pred))
print(f"Validation RMSE: {rmse_val:.6f}")

# Retrain on full training data with best iteration
full_train_dataset = lgb.Dataset(
    X_train, label=y_train, categorical_feature=categorical_cols
)
final_model = lgb.train(
    params,
    full_train_dataset,
    num_boost_round=best_iteration,
    callbacks=[lgb.log_evaluation(-1)],
)

# Predict on test set
y_test_pred = final_model.predict(test, num_iteration=best_iteration)

# Create submission
submission = pd.DataFrame({"id": range(len(test)), "Strucht": y_test_pred})

# Save submission
submission.to_csv("./working/submission.csv", index=False)
