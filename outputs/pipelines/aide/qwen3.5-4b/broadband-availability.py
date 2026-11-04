import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import os

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Define target
target_col = "Population for whom broadband available, 2019 (%)"


# Feature engineering
def create_features(df):
    df = df.copy()
    # Create numeric features from categorical columns if any
    # Add some simple transformations
    df["log_population"] = np.log1p(df["Population"])
    df["population_squared"] = df["Population"] ** 2
    # Create interaction features
    if "Population" in df.columns:
        df["pop_per_area"] = df["Population"] / (df.get("Area", 1) + 1e-6)
    return df


train_features = create_features(train_df)
test_features = create_features(test_df)

# Select features (exclude target and non-numeric columns)
numeric_cols = train_features.select_dtypes(include=[np.number]).columns.tolist()
feature_cols = [c for c in numeric_cols if c != target_col]

# Prepare data
X_train = train_features[feature_cols].values
y_train = train_df[target_col].values
X_test = test_features[feature_cols].values

# 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
best_score = np.inf
best_params = {}

# Define base parameters
base_params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_data_in_leaf": 20,
    "min_sum_hessian_in_leaf": 3,
    "verbose": -1,
}

# Train with cross-validation
for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_train_fold = X_train[train_idx]
    y_train_fold = y_train[train_idx]
    X_val_fold = X_train[val_idx]
    y_val_fold = y_train[val_idx]

    # Create datasets
    train_data = lgb.Dataset(X_train_fold, label=y_train_fold)
    val_data = lgb.Dataset(X_val_fold, label=y_val_fold, reference=train_data)

    # Train model with callbacks
    params = base_params.copy()
    params["n_estimators"] = 1000
    params["early_stopping_rounds"] = 100

    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(-1)],
    )

    # Predict on validation set
    y_pred = model.predict(X_val_fold)
    fold_rmse = np.sqrt(mean_squared_error(y_val_fold, y_pred))

    print(f"Fold {fold+1} RMSE: {fold_rmse:.4f}")

    # Track best fold
    if fold_rmse < best_score:
        best_score = fold_rmse
        best_model = model
        best_params = params

# Final model training with all data
final_params = base_params.copy()
final_params["n_estimators"] = 1000
final_params["early_stopping_rounds"] = 50

final_train_data = lgb.Dataset(X_train, label=y_train)
final_model = lgb.train(
    final_params,
    final_train_data,
    num_boost_round=1000,
    callbacks=[lgb.early_stopping(50)],
)

# Generate test predictions
test_predictions = final_model.predict(X_test)

# Save to submission file
submission_df = pd.DataFrame({"id": range(len(test_df)), target_col: test_predictions})
submission_df.to_csv("./working/submission.csv", index=False)

print(f"\nBest validation RMSE: {best_score:.4f}")
print("Submission saved to ./working/submission.csv")
