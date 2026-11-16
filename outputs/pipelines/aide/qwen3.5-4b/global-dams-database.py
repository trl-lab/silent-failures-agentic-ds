import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import os

# Set random seeds for reproducibility
np.random.seed(42)

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Define features (all columns except id and target)
feature_cols = [col for col in train_df.columns if col not in ["id", "Res_capacityM3"]]

# Handle missing values
train_df = train_df.dropna(subset=feature_cols)
test_df = test_df.dropna(subset=feature_cols)

# Create feature arrays
X_train = train_df[feature_cols].values
y_train = train_df["Res_capacityM3"].values
X_test = test_df[feature_cols].values

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize LightGBM parameters
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

# 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_scaled)):
    X_train_fold = X_train_scaled[train_idx]
    y_train_fold = y_train[train_idx]
    X_val_fold = X_train_scaled[val_idx]
    y_val_fold = y_train[val_idx]

    # Create LightGBM datasets
    train_data = lgb.Dataset(
        X_train_fold, label=y_train_fold, feature_name=feature_cols
    )
    val_data = lgb.Dataset(
        X_val_fold, label=y_val_fold, feature_name=feature_cols, reference=train_data
    )

    # Train model with early stopping
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
    )

    # Evaluate on validation set
    y_pred_val = model.predict(X_val_fold)
    rmse = np.sqrt(np.mean((y_val_fold - y_pred_val) ** 2))
    cv_scores.append(rmse)
    print(f"Fold {fold + 1}: RMSE = {rmse:.4f}")

# Calculate mean RMSE from cross-validation
mean_cv_rmse = np.mean(cv_scores)
print(f"\nMean Cross-Validation RMSE: {mean_cv_rmse:.4f}")

# Train final model on all training data
train_data_final = lgb.Dataset(X_train_scaled, label=y_train, feature_name=feature_cols)
final_model = lgb.train(
    params,
    train_data_final,
    num_boost_round=1000,
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
)

# Predict on test set
test_predictions = final_model.predict(X_test_scaled)

# Create submission file
submission = pd.DataFrame(
    {"id": test_df["id"].values, "Res_capacityM3": test_predictions}
)
submission.to_csv("./working/submission.csv", index=False)
print("\nSubmission saved to ./working/submission.csv")
