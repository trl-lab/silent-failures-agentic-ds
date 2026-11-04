import pandas as pd
import lightgbm as lgb
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Check data structure
print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")
print(f"Target column: {train_df.columns.tolist()}")

# Identify features and target
target_col = "youSaveSpend"
feature_cols = [col for col in train_df.columns if col != target_col]

X_train = train_df[feature_cols]
y_train = train_df[target_col]
X_test = test_df[feature_cols]
test_ids = test_df["id"]

# Create validation split (80/20 for simplicity)
from sklearn.model_selection import train_test_split

X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

# Handle any missing values
X_train_split = X_train_split.dropna(axis=1)
X_val_split = X_val_split.dropna(axis=1)
X_test = X_test.dropna(axis=1)

# LightGBM model parameters
params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
    "seed": 42,
}

# Train model with early stopping
train_data = lgb.Dataset(X_train_split, label=y_train_split)
val_data = lgb.Dataset(X_val_split, label=y_val_split, reference=train_data)

callbacks = [lgb.early_stopping(100), lgb.log_evaluation(0)]

model = lgb.train(
    params,
    train_data,
    num_boost_round=1000,
    valid_sets=[train_data, val_data],
    callbacks=callbacks,
    verbose_eval=False,
)

# Predict on validation set
y_val_pred = model.predict(X_val_split)
rmse_val = np.sqrt(mean_squared_error(y_val_split, y_val_pred))
print(f"Validation RMSE: {rmse_val:.4f}")

# Predict on test set
y_test_pred = model.predict(X_test)

# Create submission file
submission = pd.DataFrame({"id": test_ids, "youSaveSpend": y_test_pred})

submission.to_csv("./working/submission.csv", index=False)
print("Submission saved to ./working/submission.csv")
