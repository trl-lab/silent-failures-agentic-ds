import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error

# Ensure working directory exists
os.makedirs("./working", exist_ok=True)

# Load data FIRST - this was the main bug
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Define feature columns and target
feature_cols = ["Date", "Dept", "IsHoliday", "Store"]
target = "Weekly_Sales"

# Preprocessing: Convert Date to datetime and ordinal representation
train_df["Date"] = pd.to_datetime(train_df["Date"])
test_df["Date"] = pd.to_datetime(test_df["Date"])

# Convert Date to ordinal (days since epoch) - numeric representation LightGBM can handle
train_df["Date"] = train_df["Date"].apply(lambda x: x.toordinal())
test_df["Date"] = test_df["Date"].apply(lambda x: x.toordinal())

# Convert IsHoliday to int (0/1)
train_df["IsHoliday"] = train_df["IsHoliday"].astype(int)
test_df["IsHoliday"] = test_df["IsHoliday"].astype(int)

# Drop target from test data
if "Weekly_Sales" in test_df.columns:
    test_df = test_df.drop(columns=["Weekly_Sales"])

X_train = train_df[feature_cols].values
y_train = train_df[target].values
X_test = test_df[feature_cols].values

# LightGBM Parameters with improved settings
params = {
    "objective": "regression",
    "metric": "mae",
    "verbose": -1,
    "boosting_type": "gbdt",
    "num_leaves": 63,
    "learning_rate": 0.05,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 20,
    "max_depth": -1,
    "seed": 42,
}

# 5-Fold Cross Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_train_fold = X_train[train_idx]
    y_train_fold = y_train[train_idx]
    X_val_fold = X_train[val_idx]
    y_val_fold = y_train[val_idx]

    # Define dataset for train and validation
    train_set = lgb.Dataset(X_train_fold, label=y_train_fold)
    valid_set = lgb.Dataset(X_val_fold, label=y_val_fold, reference=train_set)

    # Train model with callbacks (LightGBM 4.6.0 compatible)
    callbacks = [lgb.early_stopping(100, verbose=False), lgb.log_evaluation(-1)]
    model = lgb.train(
        params,
        train_set,
        num_boost_round=1000,
        valid_sets=[valid_set],
        callbacks=callbacks,
    )

    # Evaluate on validation set
    y_pred_val = model.predict(X_val_fold)
    mae = mean_absolute_error(y_val_fold, y_pred_val)
    cv_scores.append(mae)

# Print evaluation metric (Average MAE from CV)
print(f"Cross-Validation MAE: {np.mean(cv_scores):.4f}")

# Train final model on all training data
train_set_final = lgb.Dataset(X_train, label=y_train)
final_model = lgb.train(params, train_set_final, num_boost_round=1000)

# Predict on test data
y_test_pred = final_model.predict(X_test)

# Prepare submission
submission_df = pd.DataFrame({"id": range(len(test_df)), "Weekly_Sales": y_test_pred})

# Save submission
submission_df.to_csv("./working/submission.csv", index=False)

print("Submission saved to ./working/submission.csv")
