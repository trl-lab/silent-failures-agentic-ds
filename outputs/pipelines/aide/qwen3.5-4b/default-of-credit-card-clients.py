import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Separate features and target
X_train = train_df.drop("Y", axis=1)
y_train = train_df["Y"]
X_test = test_df.drop("ID", axis=1)
test_ids = test_df["ID"].values

# Prepare LightGBM dataset
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, reference=train_data)

# Define callbacks
callbacks = [lgb.early_stopping(50), lgb.log_evaluation(-1)]

# 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
predictions = np.zeros(len(y_train))
auc_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_train_fold = X_train.iloc[train_idx]
    y_train_fold = y_train.iloc[train_idx]
    X_val_fold = X_train.iloc[val_idx]
    y_val_fold = y_train.iloc[val_idx]

    train_data_fold = lgb.Dataset(X_train_fold, label=y_train_fold)
    val_data_fold = lgb.Dataset(X_val_fold, reference=train_data_fold)

    # Train LightGBM model
    model = lgb.train(
        {"objective": "binary", "metric": "auc", "verbose": -1},
        train_data=train_data_fold,
        num_boost_round=1000,
        valid_sets=[val_data_fold],
        callbacks=callbacks,
    )

    # Get predictions for validation set
    val_pred = model.predict(X_val_fold)
    predictions[val_idx] = val_pred

    # Store AUC scores
    auc_scores.append(roc_auc_score(y_val_fold, val_pred))

# Calculate mean AUC across folds
mean_auc = np.mean(auc_scores)
print(f"Cross-Validation ROC AUC: {mean_auc:.4f}")

# Train final model on all training data
final_train_data = lgb.Dataset(X_train, label=y_train)
final_model = lgb.train(
    {"objective": "binary", "metric": "auc", "verbose": -1},
    train_data=final_train_data,
    num_boost_round=1000,
    callbacks=callbacks,
)

# Get predictions for test set
test_pred = final_model.predict(X_test)

# Create submission file
submission = pd.DataFrame({"id": test_ids, "default payment next month": test_pred})

# Save to working directory
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved to ./working/submission.csv")
