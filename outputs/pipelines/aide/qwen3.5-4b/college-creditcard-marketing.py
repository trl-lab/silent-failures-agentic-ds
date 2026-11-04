import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error
import os

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Define features and target
target = "PAYMENTS_BY_ISSUER"
feature_cols = [col for col in train_df.columns if col != target]

# Create feature matrix and target
X_train = train_df[feature_cols].values
y_train = train_df[target].values
X_test = test_df[feature_cols].values
test_ids = test_df["id"].values if "id" in test_df.columns else np.arange(len(test_df))

# Create validation split for model evaluation
from sklearn.model_selection import train_test_split

X_train_val, X_test_val, y_train_val, y_test_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

# LightGBM parameters
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
    "seed": 42,
}

# Train model with early stopping
model = lgb.LGBMRegressor(
    n_estimators=1000, max_depth=6, random_state=42, n_jobs=-1, **params
)

# Use 5-fold cross-validation for evaluation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []
for train_idx, val_idx in kf.split(X_train_val):
    X_tr, X_val = X_train_val[train_idx], X_train_val[val_idx]
    y_tr, y_val = y_train_val[train_idx], y_train_val[val_idx]

    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
        verbose=False,
    )

    y_pred = model.predict(X_val)
    cv_scores.append(mean_squared_error(y_val, y_pred))

# Final model evaluation
y_pred_val = model.predict(X_test_val)
cv_rmse = np.sqrt(np.mean(cv_scores))
print(f"5-Fold Cross-Validation RMSE: {cv_rmse:.4f}")

# Final predictions on test set
y_pred_test = model.predict(X_test)

# Save submission
submission = pd.DataFrame({"id": test_ids, "PAYMENTS_BY_ISSUER": y_pred_test})
submission.to_csv("./working/submission.csv", index=False)


import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import os

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Define target and features
target = "PAYMENTS_BY_ISSUER"
feature_cols = [col for col in train_df.columns if col != target]

# Create feature matrices
X_train = train_df[feature_cols].values.astype(np.float32)
y_train = train_df[target].values.astype(np.float32)
X_test = test_df[feature_cols].values.astype(np.float32)

# Check if test has id column
if "id" not in test_df.columns:
    test_ids = np.arange(len(test_df))
else:
    test_ids = test_df["id"].values

# 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_rmse_scores = []

# LightGBM parameters
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
    "seed": 42,
}

# Train and evaluate on each fold
for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_tr, X_val = X_train[train_idx], X_train[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]

    train_data = lgb.Dataset(X_tr, label=y_tr)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
        verbose_eval=-1,
    )

    y_pred_val = model.predict(X_val)
    fold_rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
    cv_rmse_scores.append(fold_rmse)
    print(f"Fold {fold+1} RMSE: {fold_rmse:.4f}")

# Calculate final RMSE
cv_rmse = np.sqrt(np.mean(cv_rmse_scores))
print(f"5-Fold Cross-Validation RMSE: {cv_rmse:.4f}")

# Train final model on all training data
final_model = lgb.train(
    params,
    lgb.Dataset(X_train, label=y_train),
    num_boost_round=500,
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
    verbose_eval=-1,
)

# Generate test predictions
y_pred_test = final_model.predict(X_test)

# Save submission
submission = pd.DataFrame({"id": test_ids, "PAYMENTS_BY_ISSUER": y_pred_test})
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

# Define target and features
target = "PAYMENTS_BY_ISSUER"
feature_cols = [col for col in train_df.columns if col != target]

# Create feature matrices
X_train = train_df[feature_cols].values.astype(np.float32)
y_train = train_df[target].values.astype(np.float32)
X_test = test_df[feature_cols].values.astype(np.float32)

# Check if test has id column
if "id" not in test_df.columns:
    test_ids = np.arange(len(test_df))
else:
    test_ids = test_df["id"].values

# 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_rmse_scores = []

# LightGBM parameters
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
    "seed": 42,
}

# Train and evaluate on each fold
for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_tr, X_val = X_train[train_idx], X_train[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]

    train_data = lgb.Dataset(X_tr, label=y_tr)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
        verbose_eval=-1,
    )

    y_pred_val = model.predict(X_val)
    fold_rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
    cv_rmse_scores.append(fold_rmse)
    print(f"Fold {fold+1} RMSE: {fold_rmse:.4f}")

# Calculate final RMSE
cv_rmse = np.sqrt(np.mean(cv_rmse_scores))
print(f"5-Fold Cross-Validation RMSE: {cv_rmse:.4f}")

# Train final model on all training data
final_model = lgb.train(
    params,
    lgb.Dataset(X_train, label=y_train),
    num_boost_round=500,
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
    verbose_eval=-1,
)

# Generate test predictions
y_pred_test = final_model.predict(X_test)

# Save submission
submission = pd.DataFrame({"id": test_ids, "PAYMENTS_BY_ISSUER": y_pred_test})
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved to ./working/submission.csv")
