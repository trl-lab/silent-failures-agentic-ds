import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Feature engineering - use all columns except target
feature_cols = [
    col
    for col in train_df.columns
    if col != "Total IFC investment as approved by Board(Million - USD)"
]
target_col = "Total IFC investment as approved by Board(Million - USD)"

# Prepare train data
X_train = train_df[feature_cols]
y_train = train_df[target_col]

# Prepare test data
X_test = test_df[feature_cols]
test_ids = test_df.index

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
}

# 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []
for train_idx, val_idx in kf.split(X_train):
    X_train_fold = X_train.iloc[train_idx]
    y_train_fold = y_train.iloc[train_idx]
    X_val_fold = X_train.iloc[val_idx]
    y_val_fold = y_train.iloc[val_idx]

    train_data = lgb.Dataset(X_train_fold, label=y_train_fold)
    val_data = lgb.Dataset(X_val_fold, label=y_val_fold, reference=train_data)

    model = lgb.train(
        params,
        train_data,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
    )

    y_pred = model.predict(X_val_fold)
    cv_scores.append(mean_squared_error(y_val_fold, y_pred))

# Get best model scores
print(f"5-fold RMSE: {np.sqrt(np.mean(cv_scores))}")

# Train final model on all training data
train_data = lgb.Dataset(X_train, label=y_train)
model = lgb.train(
    params, train_data, callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)]
)

# Predict on test data
y_pred_test = model.predict(X_test)

# Create submission
submission = pd.DataFrame(
    {
        "id": test_ids,
        "Total IFC investment as approved by Board(Million - USD)": y_pred_test,
    }
)

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

# Define feature columns (exclude target)
target_col = "Total IFC investment as approved by Board(Million - USD)"
feature_cols = [col for col in train_df.columns if col != target_col]

# Prepare train data
X_train = train_df[feature_cols].copy()
y_train = train_df[target_col].copy()

# Prepare test data
X_test = test_df[feature_cols].copy()
test_ids = test_df.index

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
}

# 5-fold cross-validation
cv_scores = []
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_train_fold = X_train.iloc[train_idx]
    y_train_fold = y_train.iloc[train_idx]
    X_val_fold = X_train.iloc[val_idx]
    y_val_fold = y_train.iloc[val_idx]

    train_data = lgb.Dataset(X_train_fold, label=y_train_fold)
    val_data = lgb.Dataset(X_val_fold, label=y_val_fold, reference=train_data)

    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
    )

    y_pred = model.predict(X_val_fold)
    cv_scores.append(mean_squared_error(y_val_fold, y_pred))
    print(f"Fold {fold + 1} RMSE: {np.sqrt(np.mean(cv_scores)):.4f}")

print(f"5-fold Cross-Validation RMSE: {np.sqrt(np.mean(cv_scores)):.4f}")

# Train final model on all training data
train_data = lgb.Dataset(X_train, label=y_train)
model = lgb.train(
    params,
    train_data,
    num_boost_round=1000,
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
)

# Predict on test data
y_pred_test = model.predict(X_test)

# Create submission
submission = pd.DataFrame(
    {
        "id": test_ids,
        "Total IFC investment as approved by Board(Million - USD)": y_pred_test,
    }
)

submission.to_csv("./working/submission.csv", index=False)
print("Submission saved to ./working/submission.csv")
