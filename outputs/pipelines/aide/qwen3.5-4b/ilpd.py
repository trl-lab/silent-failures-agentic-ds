import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import os

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Define feature columns (exclude target and id)
feature_cols = ["V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10"]
# Add id column for test set
test_df["id"] = range(len(test_df))

# Split train into train and validation for final model evaluation
X_train_full = train_df[feature_cols]
y_train = train_df["Class"]
X_test = test_df[feature_cols]

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_full)
X_test_scaled = scaler.transform(X_test)

# 5-fold cross-validation for validation metric
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_scaled, y_train)):
    X_tr, X_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]

    # Create LightGBM datasets
    train_data = lgb.Dataset(X_tr, label=y_tr)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    # Train with callbacks
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
    }

    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)],
    )

    # Predict on validation set
    y_pred = model.predict(X_val, num_iteration=model.best_iteration)
    cv_scores.append(roc_auc_score(y_val, y_pred))

print(f"Cross-validation ROC-AUC scores: {cv_scores}")
print(f"Mean CV ROC-AUC: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")

# Train final model on all training data
final_train_data = lgb.Dataset(X_train_scaled, label=y_train)
final_model = lgb.train(
    params, final_train_data, num_boost_round=1000, callbacks=[lgb.early_stopping(100)]
)

# Generate predictions for test set
y_pred_test = final_model.predict(
    X_test_scaled, num_iteration=final_model.best_iteration
)

# Create submission file
submission = pd.DataFrame({"id": test_df["id"], "Class": y_pred_test})
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved to ./working/submission.csv")
