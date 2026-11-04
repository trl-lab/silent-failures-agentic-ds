import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Define feature columns (exclude target and id columns)
target_col = "Enrollment_FT_Teachers_Ratio"
feature_cols = [col for col in train_df.columns if col not in [target_col, "id"]]

# Create feature matrix and target
X_train = train_df[feature_cols].values
y_train = train_df[target_col].values
X_test = test_df[feature_cols].values
test_ids = test_df["id"].values

# Set random seed for reproducibility
np.random.seed(42)
lgb.set_params(verbose=-1)

# Define LightGBM parameters
params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 20,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "verbose": -1,
}

# 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_train_fold = X_train[train_idx]
    y_train_fold = y_train[train_idx]
    X_val_fold = X_train[val_idx]
    y_val_fold = y_train[val_idx]

    # Create datasets
    train_data = lgb.Dataset(X_train_fold, label=y_train_fold, cat_features=[])
    val_data = lgb.Dataset(
        X_val_fold, label=y_val_fold, reference=train_data, cat_features=[]
    )

    # Train model with early stopping
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
    )

    # Evaluate on validation fold
    y_pred_val = model.predict(X_val_fold)
    fold_rmse = mean_squared_error(y_val_fold, y_pred_val, squared=False)
    cv_scores.append(fold_rmse)
    print(f"Fold {fold+1}: RMSE = {fold_rmse:.4f}")

# Calculate mean CV RMSE
mean_cv_rmse = np.mean(cv_scores)
print(f"CV RMSE (mean): {mean_cv_rmse:.4f}")

# Train final model on all training data
final_train_data = lgb.Dataset(X_train, label=y_train, cat_features=[])
final_model = lgb.train(
    params,
    final_train_data,
    num_boost_round=1000,
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
)

# Generate predictions for test data
y_pred_test = final_model.predict(X_test)

# Create submission dataframe
submission = pd.DataFrame({"id": test_ids, "Enrollment_FT_Teachers_Ratio": y_pred_test})

# Save submission
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved to ./working/submission.csv")
print(f"Final CV RMSE: {mean_cv_rmse:.4f}")
