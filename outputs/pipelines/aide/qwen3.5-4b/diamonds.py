import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
import os

# Load data
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

# Feature engineering
categorical_cols = ["cut", "color", "clarity"]
numeric_cols = ["carat", "x", "y", "z", "depth", "table"]

# Encode categorical features
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    le.fit(train[col])
    label_encoders[col] = le
    train[col] = le.transform(train[col])
    test[col] = le.transform(test[col])

# Create feature matrix
X_train = train[numeric_cols + categorical_cols]
X_test = test[numeric_cols + categorical_cols]
y_train = train["price"]

# 5-fold cross-validation for evaluation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_train_fold = X_train.iloc[train_idx]
    y_train_fold = y_train.iloc[train_idx]
    X_val_fold = X_train.iloc[val_idx]

    # Create LightGBM datasets
    train_data = lgb.Dataset(X_train_fold, label=y_train_fold)
    val_data = lgb.Dataset(
        X_val_fold, label=y_train.iloc[val_idx], reference=train_data
    )

    # Train model
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

    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
    )

    # Calculate RMSE on validation fold
    val_pred = model.predict(X_val_fold)
    fold_rmse = np.sqrt(np.mean((val_pred - y_train.iloc[val_idx]) ** 2))
    scores.append(fold_rmse)
    print(f"Fold {fold+1} RMSE: {fold_rmse:.4f}")

# Average cross-validation RMSE
cv_rmse = np.mean(scores)
print(f"\nAverage 5-fold Cross-Validation RMSE: {cv_rmse:.4f}")

# Train final model on all training data
train_data = lgb.Dataset(X_train, label=y_train)
final_model = lgb.train(
    params,
    train_data,
    num_boost_round=500,
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
)

# Generate predictions for test set
test_pred = final_model.predict(X_test)

# Create submission file
submission = pd.DataFrame({"id": test.index, "price": test_pred})
submission.to_csv("./working/submission.csv", index=False)
print("\nSubmission file saved to ./working/submission.csv")
