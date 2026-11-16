import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Identify features and target
target_col = "Stars"
feature_cols = [col for col in train_df.columns if col != target_col]

# Create feature arrays
X_train = train_df[feature_cols].values
y_train = train_df[target_col].values
X_test = test_df[feature_cols].values
test_ids = test_df.index

# 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_train_fold = X_train[train_idx]
    y_train_fold = y_train[train_idx]
    X_val_fold = X_train[val_idx]
    y_val_fold = y_train[val_idx]

    # Create LightGBM datasets
    train_data = lgb.Dataset(
        X_train_fold, label=y_train_fold, feature_name=feature_cols
    )
    val_data = lgb.Dataset(
        X_val_fold, label=y_val_fold, reference=train_data, feature_name=feature_cols
    )

    # Train model with callbacks
    params = {
        "objective": "multiclass",
        "num_class": len(np.unique(y_train)),
        "metric": "accuracy",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
    }

    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )

    # Predict on validation set
    y_pred = model.predict(X_val_fold)
    y_pred = np.round(y_pred).astype(int)
    score = accuracy_score(y_val_fold, y_pred)
    scores.append(score)
    print(f"Fold {fold+1} Accuracy: {score:.4f}")

# Calculate mean validation accuracy
mean_accuracy = np.mean(scores)
print(f"\nMean Validation Accuracy: {mean_accuracy:.4f}")

# Train final model on all training data
train_data_final = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols)
final_model = lgb.train(
    params,
    train_data_final,
    num_boost_round=1000,
    callbacks=[lgb.early_stopping(100, verbose=False)],
)

# Make predictions on test data
y_test_pred = final_model.predict(X_test)
y_test_pred = np.round(y_test_pred).astype(int)

# Create submission file
submission = pd.DataFrame({"id": test_ids, "Stars": y_test_pred})
submission.to_csv("./working/submission.csv", index=False)
print(f"\nSubmission saved to ./working/submission.csv")
