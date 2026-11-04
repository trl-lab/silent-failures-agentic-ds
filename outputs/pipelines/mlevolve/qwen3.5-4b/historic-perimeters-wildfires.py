import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import os

# Ensure output directories exist
os.makedirs("./submission", exist_ok=True)

# 1. Load Data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# 2. Identify Target and Features
target_col = "gisacres"
id_col = "id"
# Only include numeric columns as features
feature_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
feature_cols = [c for c in feature_cols if c != target_col and c != id_col]

# 3. Handle Missing Values (Impute with median for numeric columns only)
train_df = train_df.copy()
test_df = test_df.copy()

for col in feature_cols:
    median_val = train_df[col].median()
    train_df[col].fillna(median_val, inplace=True)
    test_df[col].fillna(median_val, inplace=True)

# 4. Split Data (Train/Validation/Test)
# We split the training data into train and validation sets for evaluation
X_train, X_val, y_train, y_val = train_test_split(
    train_df[feature_cols], train_df[target_col], test_size=0.2, random_state=42
)

# 5. Feature Scaling (Fit on Train Only to prevent leakage)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(test_df[feature_cols])

# 6. Prepare LightGBM Data
train_data = lgb.Dataset(X_train_scaled, label=y_train, feature_name=feature_cols)
val_data = lgb.Dataset(
    X_val_scaled, label=y_val, feature_name=feature_cols, reference=train_data
)

# 7. Define Model Parameters
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

# 8. Train Model with Early Stopping Callbacks
model = lgb.train(
    params,
    train_data,
    num_boost_round=1000,
    valid_sets=[val_data],
    callbacks=[lgb.early_stopping(50)],
)

# 9. Evaluate on Validation Set
y_val_pred = model.predict(X_val_scaled)
val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
print(f"Final Validation Score: {val_rmse}")

# 10. Predict on Test Set
y_test_pred = model.predict(X_test_scaled)

# 11. Prepare Submission
submission_df = pd.DataFrame({id_col: range(len(test_df)), target_col: y_test_pred})
submission_df.to_csv("./submission/submission.csv", index=False)