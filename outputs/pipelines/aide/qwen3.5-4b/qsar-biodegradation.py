import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import os

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Extract features and target
X_train = train_df.drop(columns=["Class"])
y_train = train_df["Class"]
X_test = test_df.drop(columns=["Class"])
test_ids = test_df.index

# Define feature columns
feature_cols = X_train.columns.tolist()

# 5-fold cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
all_predictions = []

# LightGBM model parameters
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
    "seed": 42,
}


import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import os

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Extract features and target
X_train = train_df.drop(columns=["Class"])
y_train = train_df["Class"].astype(int)
X_test = test_df.drop(columns=["Class"])
test_ids = test_df.index

# Define feature columns
feature_cols = X_train.columns.tolist()

# LightGBM model parameters
params = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_data_in_leaf": 20,
    "max_depth": -1,
    "verbose": -1,
    "seed": 42,
}

# Initialize LightGBM dataset
train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols)
test_data = lgb.Dataset(X_test, feature_name=feature_cols, reference=train_data)

# 5-fold cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_roc_auc_scores = []
all_preds = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    X_train_fold = X_train.iloc[train_idx]
    y_train_fold = y_train.iloc[train_idx]
    X_val_fold = X_train.iloc[val_idx]
    y_val_fold = y_train.iloc[val_idx]

    train_data_fold = lgb.Dataset(
        X_train_fold, label=y_train_fold, feature_name=feature_cols
    )
    val_data_fold = lgb.Dataset(
        X_val_fold,
        label=y_val_fold,
        feature_name=feature_cols,
        reference=train_data_fold,
    )

    model = lgb.train(
        params,
        train_data_fold,
        valid_sets=[val_data_fold],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(-1)],
        num_boost_round=1000,
    )

    val_preds = model.predict(X_val_fold)
    fold_roc_auc_scores.append(roc_auc_score(y_val_fold, val_preds))
    all_preds.extend(val_preds)

mean_cv_auc = np.mean(fold_roc_auc_scores)
print(f"5-Fold Cross-Validation ROC-AUC: {mean_cv_auc:.4f}")

# Train final model on all training data
final_model = lgb.train(
    params,
    train_data,
    callbacks=[lgb.early_stopping(100), lgb.log_evaluation(-1)],
    num_boost_round=1000,
)

# Generate predictions for test set
test_preds = final_model.predict(X_test)

# Save to submission file
submission = pd.DataFrame({"id": test_ids, "Class": test_preds})
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved to ./working/submission.csv")
print(f"Final Test ROC-AUC: {roc_auc_score(y_train, test_preds):.4f}")
