import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Get feature names (exclude target)
feature_cols = [col for col in train_df.columns if col != "class"]
X_train = train_df[feature_cols]
y_train = train_df["class"]

X_test = test_df[feature_cols]

# Use 5-fold cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    X_train_fold = X_train.iloc[train_idx]
    y_train_fold = y_train.iloc[train_idx]
    X_val_fold = X_train.iloc[val_idx]
    y_val_fold = y_train.iloc[val_idx]

    # Create LightGBM datasets
    train_data = lgb.Dataset(X_train_fold, label=y_train_fold)
    val_data = lgb.Dataset(X_val_fold, label=y_val_fold, reference=train_data)

    # Train model
    model = lgb.train(
        {
            "objective": "multiclass",
            "num_class": 4,
            "metric": "accuracy",
            "verbose": -1,
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "min_child_samples": 20,
            "min_data_in_leaf": 20,
            "max_depth": -1,
            "seed": 42,
        },
        train_set=train_data,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
    )

    # Get predictions on validation fold
    y_pred = model.predict(X_val_fold)
    y_pred_class = np.argmax(y_pred, axis=1)
    cv_scores.append(accuracy_score(y_val_fold, y_pred_class))

    print(f"Fold {fold + 1}: {accuracy_score(y_val_fold, y_pred_class):.4f}")

# Calculate mean CV score
mean_cv_score = np.mean(cv_scores)
print(f"Mean CV Score: {mean_cv_score:.4f}")

# Train final model on all training data
final_train_data = lgb.Dataset(X_train, label=y_train)
final_model = lgb.train(
    {
        "objective": "multiclass",
        "num_class": 4,
        "metric": "accuracy",
        "verbose": -1,
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_child_samples": 20,
        "min_data_in_leaf": 20,
        "max_depth": -1,
        "seed": 42,
    },
    train_set=final_train_data,
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
)

# Get predictions on test data
y_test_pred = final_model.predict(X_test)
y_test_pred_class = np.argmax(y_test_pred, axis=1)

# Create submission file
submission = pd.DataFrame(
    {"id": range(len(y_test_pred_class)), "class": y_test_pred_class}
)
submission.to_csv("./working/submission.csv", index=False)

print("Submission saved to ./working/submission.csv")
print(f"Final CV Score: {mean_cv_score:.4f}")


import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Get feature names (exclude target)
feature_cols = [col for col in train_df.columns if col != "class"]
X_train = train_df[feature_cols].values
y_train = train_df["class"].values
X_test = test_df[feature_cols].values

# Use 5-fold cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    X_train_fold = X_train[train_idx]
    y_train_fold = y_train[train_idx]
    X_val_fold = X_train[val_idx]
    y_val_fold = y_train[val_idx]

    # Create LightGBM datasets
    train_data = lgb.Dataset(X_train_fold, label=y_train_fold)
    val_data = lgb.Dataset(X_val_fold, label=y_val_fold, reference=train_data)

    # Train model with early stopping
    model = lgb.train(
        {
            "objective": "multiclass",
            "num_class": 4,
            "metric": "accuracy",
            "verbose": -1,
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "min_child_samples": 20,
            "min_data_in_leaf": 20,
            "max_depth": 6,
            "seed": 42,
        },
        train_set=train_data,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
    )

    # Get predictions on validation fold
    y_pred = model.predict(X_val_fold)
    y_pred_class = np.argmax(y_pred, axis=1)
    cv_scores.append(accuracy_score(y_val_fold, y_pred_class))

    print(f"Fold {fold + 1}: {accuracy_score(y_val_fold, y_pred_class):.4f}")

# Calculate mean CV score
mean_cv_score = np.mean(cv_scores)
print(f"Mean CV Score: {mean_cv_score:.4f}")

# Train final model on all training data
final_train_data = lgb.Dataset(X_train, label=y_train)
final_model = lgb.train(
    {
        "objective": "multiclass",
        "num_class": 4,
        "metric": "accuracy",
        "verbose": -1,
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_child_samples": 20,
        "min_data_in_leaf": 20,
        "max_depth": 6,
        "seed": 42,
    },
    train_set=final_train_data,
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
)

# Get predictions on test data
y_test_pred = final_model.predict(X_test)
y_test_pred_class = np.argmax(y_test_pred, axis=1)

# Create submission file
submission = pd.DataFrame(
    {"id": range(len(y_test_pred_class)), "class": y_test_pred_class}
)
submission.to_csv("./working/submission.csv", index=False)

print("Submission saved to ./working/submission.csv")
print(f"Final CV Score: {mean_cv_score:.4f}")
