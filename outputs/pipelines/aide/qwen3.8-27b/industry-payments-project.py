import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import LabelEncoder

# Ensure working directory exists
os.makedirs("./working", exist_ok=True)

# Load data
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")
print(f"Columns: {list(train.columns)}")
print(f"\nTarget stats:\n{train['projectPayment'].describe()}")

# Separate target
target_col = "projectPayment"
y = train[target_col].values
X = train.drop(columns=[target_col])

# Identify and handle categorical columns
cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
num_cols = X.select_dtypes(include=["number"]).columns.tolist()

print(f"\nCategorical columns ({len(cat_cols)}): {cat_cols}")
print(f"Numerical columns ({len(num_cols)}): {num_cols}")

# Encode categoricals
le_dict = {}
for col in cat_cols:
    le = LabelEncoder()
    # Fit on combined train+test to handle unseen categories
    combined = pd.concat([X[col], test[col]], axis=0).astype(str)
    le.fit(combined)
    X[col] = le.transform(X[col].astype(str))
    test[col] = le.transform(test[col].astype(str))
    le_dict[col] = le

# Handle missing values - fill numerical with median, categoricals with -1
for col in num_cols:
    if X[col].isna().sum() > 0:
        median_val = X[col].median()
        X[col] = X[col].fillna(median_val)
        test[col] = test[col].fillna(median_val)

for col in cat_cols:
    if X[col].isna().sum() > 0:
        X[col] = X[col].fillna(-1)
        test[col] = test[col].fillna(-1)

# Split into training and validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTrain split shape: {X_train.shape}")
print(f"Validation split shape: {X_val.shape}")

# Define LightGBM parameters
params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "max_depth": -1,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 20,
    "verbose": -1,
    "seed": 42,
}

# Create datasets
train_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

# Train with early stopping
model = lgb.train(
    params,
    train_data,
    num_boost_round=5000,
    valid_sets=[val_data],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
)

# Evaluate on validation set
val_pred = model.predict(X_val, num_iteration=model.best_iteration)
val_rmse = root_mean_squared_error(y_val, val_pred)
print(f"\nValidation RMSE: {val_rmse:.4f}")
print(f"Best iteration: {model.best_iteration}")

# Feature importance
fi = pd.DataFrame(
    {
        "feature": X.columns,
        "importance": model.feature_importance(importance_type="gain"),
    }
).sort_values("importance", ascending=False)
print(f"\nTop 15 features by gain:")
print(fi.head(15))

# Retrain on full training data with best iteration count
full_train_data = lgb.Dataset(X, label=y)
final_model = lgb.train(
    params,
    full_train_data,
    num_boost_round=model.best_iteration,
    callbacks=[lgb.log_evaluation(-1)],
)

# Generate test predictions
test_pred = final_model.predict(test, num_iteration=final_model.best_iteration)

# Create submission
submission = pd.DataFrame(
    {"id": np.arange(len(test_pred)), "projectPayment": test_pred}
)

submission.to_csv("./working/submission.csv", index=False)
print(f"\nSubmission saved to ./working/submission.csv")
print(f"Submission shape: {submission.shape}")
print(f"\nPrediction stats:\n{submission['projectPayment'].describe()}")
