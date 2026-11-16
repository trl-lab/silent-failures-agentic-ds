import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Prepare features and targets
features = [
    "tau1",
    "tau2",
    "tau3",
    "tau4",
    "p1",
    "p2",
    "p3",
    "p4",
    "g1",
    "g2",
    "g3",
    "g4",
]
X_train = train_df[features].values
y_train = train_df["stabf"].values
X_test = test_df[features].values

# Create 5-fold stratified cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Store predictions for validation
cv_predictions = np.zeros(len(y_train))

# Train model with 5-fold CV
for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    X_fold_train = X_train[train_idx]
    y_fold_train = y_train[train_idx]
    X_fold_val = X_train[val_idx]
    y_fold_val = y_train[val_idx]

    # Create LightGBM datasets
    train_data = lgb.Dataset(X_fold_train, label=y_fold_train)
    val_data = lgb.Dataset(X_fold_val, label=y_fold_val, reference=train_data)

    # Train model with callbacks
    params = {
        "objective": "binary",
        "metric": "auc",
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
        num_boost_round=500,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )

    # Get validation predictions
    val_preds = model.predict(X_fold_val)
    cv_predictions[val_idx] = val_preds

# Calculate validation metric
val_auc = roc_auc_score(y_train, cv_predictions)
print(f"5-Fold Cross-Validation ROC-AUC: {val_auc:.4f}")

# Train final model on all training data
final_train_data = lgb.Dataset(X_train, label=y_train)
final_model = lgb.train(
    params,
    final_train_data,
    num_boost_round=500,
    callbacks=[lgb.early_stopping(50, verbose=False)],
)

# Generate test predictions
test_preds = final_model.predict(X_test)

# Create submission file
submission = pd.DataFrame({"id": range(len(test_df)), "stabf": test_preds})
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved to ./working/submission.csv")
print(f"Test predictions saved with shape: {submission.shape}")
