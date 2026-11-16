import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Identify features and target
target_col = "SupStruc"
feature_cols = [col for col in train_df.columns if col != target_col]

# Convert object columns to numeric or categorical
for col in feature_cols:
    if train_df[col].dtype == "object":
        # Try to convert to numeric
        train_df[col] = pd.to_numeric(train_df[col], errors="coerce")
        test_df[col] = pd.to_numeric(test_df[col], errors="coerce")
        # Fill NaN with 0
        train_df[col] = train_df[col].fillna(0)
        test_df[col] = test_df[col].fillna(0)

# Identify categorical columns (remaining object columns after conversion)
cat_cols = [col for col in feature_cols if train_df[col].dtype == "object"]

# Create 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Store predictions for each fold
fold_preds = []
fold_scores = []

# Train and evaluate on each fold
for fold, (train_idx, val_idx) in enumerate(kf.split(train_df)):
    train_X = train_df.iloc[train_idx][feature_cols]
    train_y = train_df.iloc[train_idx][target_col]
    val_X = train_df.iloc[val_idx][feature_cols]
    val_y = train_df.iloc[val_idx][target_col]

    # Create LightGBM datasets
    train_data = lgb.Dataset(train_X, label=train_y)
    val_data = lgb.Dataset(val_X, label=val_y, reference=train_data)

    # Train model with early stopping
    params = {
        "objective": "regression",
        "metric": "mse",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
    }

    # Add categorical features if any
    if cat_cols:
        params["categorical_feature"] = cat_cols

    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )

    # Predict on validation set
    val_pred = model.predict(val_X)
    fold_scores.append(mean_squared_error(val_y, val_pred))
    fold_preds.append(val_pred)

# Calculate cross-validation RMSE
cv_rmse = np.sqrt(np.mean(fold_scores))
print(f"Cross-validation RMSE: {cv_rmse:.4f}")

# Save predictions for test set
test_X = test_df[feature_cols]
test_pred = model.predict(test_X)

# Create submission
submission = pd.DataFrame({"id": range(len(test_pred)), "SupStruc": test_pred})
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved to ./working/submission.csv")
