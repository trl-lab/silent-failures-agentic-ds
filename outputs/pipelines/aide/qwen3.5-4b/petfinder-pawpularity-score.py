import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
import os

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Create feature columns (exclude target and id)
feature_cols = [col for col in train_df.columns if col not in ["Pawpularity", "id"]]

# Split training data for validation
X_train = train_df[feature_cols]
y_train = train_df["Pawpularity"]
X_test = test_df[feature_cols]
X_test_ids = test_df["id"]

# Create 5-fold cross-validation for evaluation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# LightGBM parameters
params = {
    "objective": "regression",
    "metric": "rmse",
    "verbose": -1,
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
}

# Train LightGBM model with early stopping
callbacks = [lgb.early_stopping(50), lgb.log_evaluation(-1)]
model = lgb.train(
    params,
    lgb.Dataset(X_train, label=y_train),
    num_boost_round=500,
    callbacks=callbacks,
)

# Evaluate on validation set
y_val = train_df["Pawpularity"].values
y_pred = model.predict(X_train)
rmse_val = np.sqrt(mean_squared_error(y_val, y_pred))
print(f"Validation RMSE: {rmse_val:.4f}")

# 5-fold cross-validation for robust evaluation
cv_scores = []
for i, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_train_fold = X_train.iloc[train_idx]
    y_train_fold = y_train.iloc[train_idx]
    X_val_fold = X_train.iloc[val_idx]

    model_fold = lgb.train(
        params,
        lgb.Dataset(X_train_fold, label=y_train_fold),
        num_boost_round=500,
        callbacks=callbacks,
    )
    y_pred_fold = model_fold.predict(X_val_fold)
    cv_scores.append(np.sqrt(mean_squared_error(y_train_fold, y_pred_fold)))

print(f"5-fold CV RMSE: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")

# Predict on test set
test_predictions = model.predict(X_test)

# Create submission file
submission = pd.DataFrame({"id": X_test_ids, "Pawpularity": test_predictions})
submission.to_csv("./working/submission.csv", index=False)
print("Saved predictions to ./working/submission.csv")


import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import os

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Create feature columns (exclude target and id)
feature_cols = [col for col in train_df.columns if col not in ["Pawpularity", "id"]]

# Split training data for validation (70-30 split)
X_train_full = train_df[feature_cols]
y_train_full = train_df["Pawpularity"]
X_test = test_df[feature_cols]
X_test_ids = test_df["id"]

# Hold-out validation split (70% train, 30% val)
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.3, random_state=42, shuffle=False
)

# LightGBM parameters
params = {
    "objective": "regression",
    "metric": "rmse",
    "verbose": -1,
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_data_in_leaf": 20,
    "min_sum_hessian_in_leaf": 3,
}

# Train LightGBM model with early stopping
callbacks = [lgb.early_stopping(50), lgb.log_evaluation(-1)]
model = lgb.train(
    params,
    lgb.Dataset(X_train, label=y_train),
    num_boost_round=500,
    callbacks=callbacks,
)

# Evaluate on validation set
y_pred_val = model.predict(X_val)
rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_val))
print(f"Validation RMSE: {rmse_val:.4f}")

# 5-fold cross-validation for robust evaluation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []
for i, (train_idx, val_idx) in enumerate(kf.split(X_train_full)):
    X_train_fold = X_train_full.iloc[train_idx]
    y_train_fold = y_train_full.iloc[train_idx]
    X_val_fold = X_train_full.iloc[val_idx]

    model_fold = lgb.train(
        params,
        lgb.Dataset(X_train_fold, label=y_train_fold),
        num_boost_round=500,
        callbacks=callbacks,
    )
    y_pred_fold = model_fold.predict(X_val_fold)
    cv_scores.append(np.sqrt(mean_squared_error(y_train_fold, y_pred_fold)))

cv_rmse = np.mean(cv_scores)
cv_std = np.std(cv_scores)
print(f"5-fold CV RMSE: {cv_rmse:.4f} ± {cv_std:.4f}")

# Predict on test set
test_predictions = model.predict(X_test)

# Create submission file
submission = pd.DataFrame({"id": X_test_ids, "Pawpularity": test_predictions})
submission.to_csv("./working/submission.csv", index=False)
print("Saved predictions to ./working/submission.csv")
