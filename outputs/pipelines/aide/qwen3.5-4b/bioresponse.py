import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss
import numpy as np
import os

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Split features and target
y = train_df["Activity"]
X = train_df.drop("Activity", axis=1)
test_X = test_df

# Create 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# LightGBM parameters
params = {
    "objective": "regression",
    "metric": "l2",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
    "seed": 42,
}

# Training with callbacks
callbacks = [lgb.early_stopping(50), lgb.log_evaluation(-1)]

# Cross-validation evaluation
cv_scores = []
for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        callbacks=callbacks,
        valid_sets=[val_data],
    )

    pred_val = model.predict(X_val)
    cv_scores.append(log_loss(y_val, pred_val))

# Final model training on all data
train_data = lgb.Dataset(X, label=y)
model = lgb.train(params, train_data, num_boost_round=1000, callbacks=callbacks)

# Predict on test data
test_pred = model.predict(test_X)

# Save submission
submission_df = pd.DataFrame({"id": test_df.index, "Activity": test_pred})
submission_df.to_csv("./working/submission.csv", index=False)

# Print evaluation metric
print(f"Cross-validation log_loss: {np.mean(cv_scores):.4f}")
