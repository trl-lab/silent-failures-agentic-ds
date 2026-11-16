import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import numpy as np

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Prepare features
X_train = train_df.drop("dea_schedule", axis=1)
y_train = train_df["dea_schedule"]
X_test = test_df.drop("dea_schedule", axis=1)

# Create KFold for 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Initialize LightGBM parameters
params = {
    "objective": "multiclass",
    "num_class": len(np.unique(y_train)),
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
    "seed": 42,
}

# Track best validation accuracy
best_accuracy = 0.0
best_model = None

# Cross-validation
for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_train_fold = X_train.iloc[train_idx]
    y_train_fold = y_train.iloc[train_idx]
    X_val_fold = X_train.iloc[val_idx]
    y_val_fold = y_train.iloc[val_idx]

    # Create datasets
    train_data = lgb.Dataset(X_train_fold, label=y_train_fold)
    val_data = lgb.Dataset(X_val_fold, label=y_val_fold, reference=train_data)

    # Train model
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
    )

    # Predict on validation set
    y_pred = model.predict(X_val_fold)
    y_pred_class = np.round(y_pred).astype(int)
    accuracy = accuracy_score(y_val_fold, y_pred_class)

    print(f"Fold {fold+1} accuracy: {accuracy:.4f}")

    # Track best model
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model

# Generate test predictions
test_data = lgb.Dataset(X_test, reference=best_model)
test_pred = best_model.predict(test_data)
test_pred_class = np.round(test_pred).astype(int)

# Create submission
submission = pd.DataFrame({"id": range(len(test_df)), "dea_schedule": test_pred_class})

# Save submission
submission.to_csv("./working/submission.csv", index=False)
print(f"\nBest validation accuracy: {best_accuracy:.4f}")
print("Submission saved to ./working/submission.csv")
