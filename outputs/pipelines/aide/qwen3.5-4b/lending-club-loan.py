import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Define target column and positive class
target_col = "loan_status"
positive_class = "Charged Off"

# Preprocess target: Binary (1 if Charged Off, else 0)
train_df[target_col] = (train_df[target_col] == positive_class).astype("int8")

# Define feature columns (exclude target)
feature_cols = [c for c in train_df.columns if c != target_col]

# Prepare LightGBM datasets
train_data = lgb.Dataset(train_df[feature_cols], label=train_df[target_col])
test_data = lgb.Dataset(test_df[feature_cols], reference=train_data)

# Hyperparameters
params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
    "seed": 42,
}

# Cross-validation
cv_auc = []
kf = KFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_idx, val_idx) in enumerate(kf.split(train_df)):
    train_fold = train_df.iloc[train_idx]
    val_fold = train_df.iloc[val_idx]

    train_fold_data = lgb.Dataset(
        train_fold[feature_cols], label=train_fold[target_col]
    )
    val_fold_data = lgb.Dataset(
        val_fold[feature_cols], label=val_fold[target_col], reference=train_fold_data
    )

    # Train model
    model = lgb.train(
        params,
        train_fold_data,
        num_boost_round=1000,
        valid_sets=[val_fold_data],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )

    # Predict on validation set
    val_pred = model.predict(val_fold_data)
    # Calculate AUC
    if len(np.unique(val_fold[target_col])) > 1:
        cv_auc.append(roc_auc_score(val_fold[target_col], val_pred))
    else:
        cv_auc.append(0)

mean_cv_auc = np.mean(cv_auc)
print(f"5-Fold Cross-Validation ROC AUC: {mean_cv_auc:.4f}")

# Train final model on all training data
final_model = lgb.train(
    params,
    train_data,
    num_boost_round=1000,
    callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
)

# Predict on test set
test_pred = final_model.predict(test_data)

# Prepare submission
submission = pd.DataFrame({"id": range(len(test_df)), "loan_status": test_pred})

# Save submission
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved to ./working/submission.csv")
