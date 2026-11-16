import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Read data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Extract features and target
X_train = train_df.drop(columns=["New Commitments (US$)"])
y_train = train_df["New Commitments (US$)"]
X_test = test_df

# Create train/validation split for initial evaluation
# Use 5-fold cross-validation for final evaluation


import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Read data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Extract features and target
feature_cols = [col for col in train_df.columns if col != "New Commitments (US$)"]
X_train = train_df[feature_cols].values
y_train = train_df["New Commitments (US$)"].values
X_test = test_df[feature_cols].values

# Define LightGBM parameters
params = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "min_data_in_leaf": 20,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
}

# Create 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
y_pred_cv = np.zeros(len(y_train))
scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_train_fold = X_train[train_idx]
    y_train_fold = y_train[train_idx]
    X_val_fold = X_train[val_idx]
    y_val_fold = y_train[val_idx]

    train_data = lgb.Dataset(
        X_train_fold,
        label=y_train_fold,
        reference_data=lgb.Dataset(X_train, label=y_train, is_predict=True),
    )
    val_data = lgb.Dataset(X_val_fold, label=y_val_fold, reference_data=train_data)

    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
    )

    y_pred_fold = model.predict(X_train[val_idx])
    y_pred_cv[val_idx] = y_pred_fold

    # Calculate RMSE for this fold
    fold_rmse = np.sqrt(mean_squared_error(y_val_fold, y_pred_fold))
    scores.append(fold_rmse)

# Calculate final cross-validation RMSE
cv_rmse = np.sqrt(mean_squared_error(y_train, y_pred_cv))
print(f"5-Fold Cross-Validation RMSE: {cv_rmse:.4f}")

# Train final model on all training data
final_model = lgb.train(
    params,
    lgb.Dataset(X_train, label=y_train),
    num_boost_round=1000,
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
)

# Predict on test data
test_predictions = final_model.predict(X_test)

# Create submission file
submission = pd.DataFrame(
    {"id": range(len(test_df)), "New Commitments (US$)": test_predictions}
)
submission.to_csv("./working/submission.csv", index=False)
print("Submission file saved to ./working/submission.csv")
