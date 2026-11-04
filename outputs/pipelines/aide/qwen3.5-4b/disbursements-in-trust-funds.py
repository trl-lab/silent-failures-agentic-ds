import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import os

# Set random seed for reproducibility
np.random.seed(42)

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Create submission directory if not exists
os.makedirs("./working", exist_ok=True)

# Identify feature columns (exclude target and id columns)
target_col = "Cash Disbursements (US$)"
id_col = "id" if "id" in test_df.columns else None

# If id column exists, separate it
if id_col:
    train_features = train_df.drop(columns=[id_col, target_col])
    test_features = test_df.drop(columns=[id_col])
else:
    train_features = train_df.drop(columns=[target_col])
    test_features = test_df

# Handle missing values
train_features = train_features.fillna(train_features.median())
test_features = test_features.fillna(train_features.median())

# Identify numeric feature columns
numeric_features = train_features.select_dtypes(include=[np.number]).columns.tolist()

# Scale features
scaler = StandardScaler()
train_features_scaled = scaler.fit_transform(train_features[numeric_features])
test_features_scaled = scaler.transform(test_features[numeric_features])

# Split training data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    train_features_scaled,
    train_df[target_col],
    test_size=0.2,
    random_state=42,
    stratify=None,
)

# Create LightGBM datasets
train_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

# Train LightGBM model with callbacks
params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
}

model = lgb.train(
    params,
    train_data,
    num_boost_round=1000,
    valid_sets=[train_data, val_data],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
    verbose_eval=False,
)

# Evaluate on validation set
val_predictions = model.predict(X_val)
val_rmse = mean_squared_error(y_val, val_predictions)
print(f"Validation RMSE: {val_rmse:.4f}")

# Generate predictions for test set
test_predictions = model.predict(test_features_scaled)

# Create submission file
submission = pd.DataFrame(
    {
        "id": test_features.index if id_col else range(len(test_features)),
        "Cash Disbursements (US$)": test_predictions,
    }
)

# Save submission
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved to ./working/submission.csv")
