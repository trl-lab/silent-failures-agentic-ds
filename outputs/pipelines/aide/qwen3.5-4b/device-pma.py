import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Define feature columns (exclude target and id if present)
feature_cols = [col for col in train_df.columns if col not in ["decision_code", "id"]]
X_train = train_df[feature_cols].values
y_train = train_df["decision_code"].values
X_test = test_df[feature_cols].values
test_ids = test_df["id"].values if "id" in test_df.columns else np.arange(len(test_df))

# Create train/val split
X_train_split, y_train_split = (
    X_train[: int(len(X_train) * 0.8)],
    y_train[: int(len(y_train) * 0.8)],
)
X_val, y_val = X_train[int(len(X_train) * 0.8) :], y_train[int(len(y_train) * 0.8) :]

# LightGBM parameters
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
    "seed": 42,
}

# Train model with early stopping
train_data = lgb.Dataset(X_train_split, label=y_train_split)
val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

model = lgb.train(
    params,
    train_data,
    num_boost_round=5000,
    valid_sets=[train_data, val_data],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
)

# 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_roc_auc_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_fold_train = X_train[train_idx]
    y_fold_train = y_train[train_idx]
    X_fold_val = X_train[val_idx]
    y_fold_val = y_train[val_idx]

    train_data_fold = lgb.Dataset(X_fold_train, label=y_fold_train)
    val_data_fold = lgb.Dataset(X_fold_val, label=y_fold_val, reference=train_data_fold)

    model_fold = lgb.train(
        params,
        train_data_fold,
        num_boost_round=model.best_iteration,
        valid_sets=[train_data_fold, val_data_fold],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
    )

    y_val_pred = model_fold.predict(X_fold_val)
    cv_roc_auc_scores.append(roc_auc_score(y_fold_val, y_val_pred))

print(
    f"5-Fold Cross-Validation ROC-AUC: {np.mean(cv_roc_auc_scores):.4f} (+/- {np.std(cv_roc_auc_scores):.4f})"
)

# Generate predictions for test set
test_pred = model.predict(X_test)

# Create submission file
submission = pd.DataFrame({"id": test_ids, "decision_code": test_pred})
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved to ./working/submission.csv")
