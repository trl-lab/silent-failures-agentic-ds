import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Define feature columns (exclude target and id columns)
target_col = "IMU Score"
id_col = "id"

# Prepare features
train_features = train_df.drop(columns=[target_col, id_col])
test_features = test_df.drop(columns=[id_col])

# Split for validation (80/20)
X_train, X_val, y_train, y_val = (
    train_features.sample(frac=0.8, random_state=42),
    train_features.sample(frac=0.2, random_state=42),
    train_df[target_col].sample(frac=0.8, random_state=42),
    train_df[target_col].sample(frac=0.2, random_state=42),
)

# Better approach: use 5-fold cross-validation for final evaluation
# But we need to save predictions for test set

# LightGBM model
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

# Train model with 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_scores = []

for fold_idx, (train_idx, val_idx) in enumerate(kf.split(train_features)):
    X_train_fold = train_features.iloc[train_idx]
    y_train_fold = train_df[target_col].iloc[train_idx]
    X_val_fold = train_features.iloc[val_idx]
    y_val_fold = train_df[target_col].iloc[val_idx]

    train_data = lgb.Dataset(X_train_fold, label=y_train_fold)
    val_data = lgb.Dataset(X_val_fold, label=y_val_fold, reference=train_data)

    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[train_data, val_data],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
    )

    y_pred = model.predict(X_val_fold)
    fold_rmse = mean_squared_error(y_val_fold, y_pred, squared=False)
    fold_scores.append(fold_rmse)
    print(f"Fold {fold_idx+1}: RMSE = {fold_rmse:.4f}")

# Get final validation RMSE
final_rmse = np.sqrt(np.mean(fold_scores))
print(f"Final RMSE on validation set: {final_rmse:.4f}")

# Train final model on all training data
train_data = lgb.Dataset(train_features, label=train_df[target_col])
final_model = lgb.train(
    params, train_data, num_boost_round=1000, callbacks=[lgb.early_stopping(50)]
)

# Make predictions on test set
test_predictions = final_model.predict(test_features)

# Save to submission file
submission = pd.DataFrame({"id": test_df[id_col], "IMU Score": test_predictions})
submission.to_csv("./working/submission.csv", index=False)

print("Submission saved to ./working/submission.csv")


import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Define columns
target_col = "IMU Score"
id_col = "id"

# Prepare features
train_features = train_df.drop(columns=[target_col, id_col])
test_features = test_df.drop(columns=[id_col])

# Convert to appropriate numeric types
train_features = train_features.astype(np.float32)
test_features = test_features.astype(np.float32)

# 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_scores = []

for fold_idx, (train_idx, val_idx) in enumerate(kf.split(train_features)):
    X_train_fold = train_features.iloc[train_idx]
    y_train_fold = train_df[target_col].iloc[train_idx]
    X_val_fold = train_features.iloc[val_idx]
    y_val_fold = train_df[target_col].iloc[val_idx]

    train_data = lgb.Dataset(X_train_fold, label=y_train_fold)
    val_data = lgb.Dataset(X_val_fold, label=y_val_fold, reference=train_data)

    params = {
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

    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[train_data, val_data],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
    )

    y_pred = model.predict(X_val_fold)
    fold_rmse = mean_squared_error(y_val_fold, y_pred, squared=False)
    fold_scores.append(fold_rmse)
    print(f"Fold {fold_idx+1}: RMSE = {fold_rmse:.4f}")

# Final validation RMSE
final_rmse = np.sqrt(np.mean(fold_scores))
print(f"Final RMSE on validation set: {final_rmse:.4f}")

# Train final model on all training data
train_data = lgb.Dataset(train_features, label=train_df[target_col])
final_model = lgb.train(
    params, train_data, num_boost_round=1000, callbacks=[lgb.early_stopping(50)]
)

# Make predictions on test set
test_predictions = final_model.predict(test_features)

# Save to submission file
submission = pd.DataFrame({"id": test_df[id_col], "IMU Score": test_predictions})
submission.to_csv("./working/submission.csv", index=False)

print("Submission saved to ./working/submission.csv")
