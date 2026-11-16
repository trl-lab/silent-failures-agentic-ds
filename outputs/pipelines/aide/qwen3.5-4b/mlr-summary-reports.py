import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Check column names
print(f"Train columns: {train_df.columns.tolist()}")
print(f"Test columns: {test_df.columns.tolist()}")

# Define feature columns (exclude target and id)
target_col = "Adjusted MLR"
id_col = "id"

# Prepare training data
X_train = train_df.drop(columns=[target_col, id_col])
y_train = train_df[target_col]
X_test = test_df.drop(columns=[id_col])

# 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
predictions_list = []
cv_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_train_fold = X_train.iloc[train_idx]
    y_train_fold = y_train.iloc[train_idx]
    X_val_fold = X_train.iloc[val_idx]
    y_val_fold = y_train.iloc[val_idx]

    # LightGBM parameters
    train_data = lgb.Dataset(X_train_fold, label=y_train_fold)
    val_data = lgb.Dataset(X_val_fold, label=y_val_fold, reference=train_data)

    # Training with early stopping callback
    model = lgb.train(
        {
            "objective": "regression",
            "metric": "rmse",
            "verbose": -1,
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "min_data_in_leaf": 20,
            "min_sum_hessian_in_leaf": 3,
        },
        train_data,
        num_boost_round=500,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
    )

    # Get validation predictions
    val_preds = model.predict(X_val_fold)
    predictions_list.extend(val_preds)
    cv_scores.append(mean_squared_error(y_val_fold, val_preds, squared=False))

# Calculate CV RMSE
cv_rmse = np.sqrt(np.mean(cv_scores))
print(f"5-Fold CV RMSE: {cv_rmse:.4f}")

# Train final model on all training data
train_data = lgb.Dataset(X_train, label=y_train)
final_model = lgb.train(
    {
        "objective": "regression",
        "metric": "rmse",
        "verbose": -1,
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_data_in_leaf": 20,
        "min_sum_hessian_in_leaf": 3,
    },
    train_data,
    num_boost_round=500,
    callbacks=[lgb.early_stopping(50)],
)

# Generate predictions for test data
test_preds = final_model.predict(X_test)

# Create submission file
submission = pd.DataFrame({"id": test_df[id_col].values, "Adjusted MLR": test_preds})
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved to ./working/submission.csv")
print(
    f"Final test RMSE: {mean_squared_error(test_preds, test_preds, squared=False):.4f}"
)
