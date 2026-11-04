import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# Load data
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

# Data cleaning
# Drop Unnamed: 0 column
train = train.drop(columns=["Unnamed: 0"])
test = test.drop(columns=["Unnamed: 0"])

# Impute missing values with median for numerical columns
train = train.fillna(train.median(numeric_only=True))
test = test.fillna(test.median(numeric_only=True))

# Define features and target
feature_cols = train.drop(columns=["SeriousDlqin2yrs"]).columns
X_train = train[feature_cols]
y_train = train["SeriousDlqin2yrs"]
X_test = test[feature_cols]

# Initialize LightGBM model
params = {
    "objective": "binary",
    "metric": "auc",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "max_depth": -1,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "verbose": -1,
}

# Set callbacks for LightGBM 4.6.0
callbacks = [lgb.early_stopping(50), lgb.log_evaluation(-1)]

# 5-fold cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
auc_scores = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
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
        valid_sets=[val_data],
        callbacks=callbacks,
        num_boost_round=1000,
    )

    # Evaluate on validation fold
    y_val_pred = model.predict(X_val_fold, num_iteration=model.best_iteration)
    auc_scores.append(roc_auc_score(y_val_fold, y_val_pred))

# Print evaluation metric
mean_auc = np.mean(auc_scores)
print(f"5-Fold Cross-Validation ROC AUC: {mean_auc:.4f}")

# Predict on test set
y_test_pred = model.predict(X_test, num_iteration=model.best_iteration)

# Save predictions
submission = pd.DataFrame({"id": test.index, "SeriousDlqin2yrs": y_test_pred})
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved to ./working/submission.csv")
