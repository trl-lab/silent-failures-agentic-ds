import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
import os

# Set random seed for reproducibility
np.random.seed(42)

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Store original IDs for submission
original_ids = test_df.index
train_ids = train_df.index

# Create feature ID mapping for test data if needed
if "id" in test_df.columns:
    test_ids = test_df["id"].values
else:
    test_ids = test_df.index.values

# Define feature columns
feature_cols = [col for col in train_df.columns if col not in ["id", "risk_decision"]]

# Encode categorical features
categorical_cols = [col for col in feature_cols if train_df[col].dtype == "object"]
numerical_cols = [
    col for col in feature_cols if train_df[col].dtype in ["int64", "float64"]
]

# Encode categorical features
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    train_df[col + "_encoded"] = le.fit_transform(train_df[col].astype(str))
    test_df[col + "_encoded"] = le.transform(test_df[col].astype(str))
    label_encoders[col] = le

# Convert to numeric (handle NaN)
for col in categorical_cols:
    train_df[col + "_encoded"] = pd.to_numeric(
        train_df[col + "_encoded"], errors="coerce"
    )
    test_df[col + "_encoded"] = pd.to_numeric(
        test_df[col + "_encoded"], errors="coerce"
    )

# Prepare features and target
X_train = train_df[feature_cols].fillna(0)
y_train = train_df["risk_decision"]

X_test = test_df[feature_cols].fillna(0)
X_test = X_test.copy()

# LightGBM parameters
params = {
    "objective": "multiclass",
    "num_class": len(np.unique(y_train)),
    "metric": "accuracy",
    "verbose": -1,
    "seed": 42,
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
}

# 5-fold cross-validation for evaluation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

    # Create datasets
    train_data = lgb.Dataset(X_tr, label=y_tr)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    # Train with early stopping
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
        verbose_eval=-1,
    )

    # Evaluate on validation set
    y_pred_val = model.predict(X_val, num_iteration=model.best_iteration)
    y_pred_val = np.round(y_pred_val).astype(int)
    y_pred_val = np.clip(y_pred_val, 0, len(np.unique(y_train)) - 1)

    fold_accuracy = (y_pred_val == y_val).mean()
    cv_scores.append(fold_accuracy)
    print(f"Fold {fold + 1}: {fold_accuracy:.4f}")

# Average CV score
cv_mean = np.mean(cv_scores)
print(f"\nAverage CV Accuracy: {cv_mean:.4f}")

# Train final model on all training data
final_model = lgb.train(
    params,
    lgb.Dataset(X_train, label=y_train),
    num_boost_round=1000,
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
    verbose_eval=-1,
)

# Get predictions for test data
test_predictions = final_model.predict(X_test, num_iteration=final_model.best_iteration)
test_predictions = np.round(test_predictions).astype(int)
test_predictions = np.clip(test_predictions, 0, len(np.unique(y_train)) - 1)

# Create submission file
submission = pd.DataFrame({"id": original_ids, "risk_decision": test_predictions})

# Save submission
submission.to_csv("./working/submission.csv", index=False)
print(f"\nSubmission saved to ./working/submission.csv")
print(f"Final CV Accuracy: {cv_mean:.4f}")
