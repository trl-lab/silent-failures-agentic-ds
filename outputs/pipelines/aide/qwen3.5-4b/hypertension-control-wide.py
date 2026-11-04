import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Prepare training data
y_train = train_df["Estimated Percentage of Patients with Controlled Blood Pressure"]
X_train = train_df.drop(
    columns=["Estimated Percentage of Patients with Controlled Blood Pressure"]
)
X_test = test_df.drop(
    columns=["Estimated Percentage of Patients with Controlled Blood Pressure"]
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
}

# 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_predictions = np.zeros(len(X_train))
cv_roc_auc = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_train_fold = X_train.iloc[train_idx]
    y_train_fold = y_train.iloc[train_idx]
    X_val_fold = X_train.iloc[val_idx]
    y_val_fold = y_train.iloc[val_idx]

    # Create LightGBM datasets
    train_data = lgb.Dataset(X_train_fold, label=y_train_fold)
    val_data = lgb.Dataset(X_val_fold, label=y_val_fold, reference=train_data)

    # Train model with early stopping
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )

    # Get predictions
    oof_predictions[val_idx] = model.predict(X_val_fold)

    # Calculate ROC AUC on validation set
    if len(np.unique(y_val_fold)) > 1:
        cv_roc_auc.append(roc_auc_score(y_val_fold, oof_predictions[val_idx]))

# Calculate final ROC AUC
final_roc_auc = np.mean(cv_roc_auc)
print(f"5-Fold Cross-Validation ROC AUC: {final_roc_auc:.4f}")

# Train final model on all data
final_model = lgb.train(
    params, lgb.Dataset(X_train, label=y_train), num_boost_round=1000
)

# Generate predictions for test set
test_predictions = final_model.predict(X_test)

# Create submission file
submission = pd.DataFrame(
    {
        "id": range(len(X_test)),
        "Estimated Percentage of Patients with Controlled Blood Pressure": test_predictions,
    }
)
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved to ./working/submission.csv")
