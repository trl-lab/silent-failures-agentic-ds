import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Define features and target
feature_cols = [
    col for col in train_df.columns if col not in ["id", "Damage Description (total)"]
]
target_col = "Damage Description (total)"


# Extract features and target
X_train = train_df[feature_cols]
y_train = train_df[target_col]
X_test = test_df[feature_cols]
test_ids = test_df["id"]

# Split data for validation (80/20 split)
from sklearn.model_selection import train_test_split

X_train_val, X_test_temp, y_train_val, y_test_temp = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

# LightGBM parameters
params = {
    "objective": "multiclass",
    "num_class": len(y_train_val.unique()),
    "boosting_type": "gbdt",
    "metric": "accuracy",
    "verbose": -1,
}

# Create LightGBM dataset
train_data = lgb.Dataset(X_train_val, label=y_train_val)
valid_data = lgb.Dataset(X_test_temp, label=y_test_temp, reference=train_data)

# Train with early stopping
model = lgb.train(
    params,
    train_data,
    num_boost_round=1000,
    valid_sets=[valid_data],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
)

# Evaluate on validation set
y_pred_val = model.predict(X_test_temp)
accuracy = accuracy_score(y_test_temp, y_pred_val)
print(f"Validation Accuracy: {accuracy:.4f}")

# Save test predictions
y_pred_test = model.predict(X_test)
submission = pd.DataFrame({"id": test_ids, "Damage Description (total)": y_pred_test})
submission.to_csv("./working/submission.csv", index=False)


import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import numpy as np

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Define features and target
feature_cols = [
    col for col in train_df.columns if col not in ["id", "Damage Description (total)"]
]
target_col = "Damage Description (total)"

# Extract features and target
X_train = train_df[feature_cols]
y_train = train_df[target_col]
X_test = test_df[feature_cols]
test_ids = test_df["id"]

# Number of classes
n_classes = len(y_train.unique())

# 5-fold cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Train and evaluate
fold_scores = []
for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    X_train_fold = X_train.iloc[train_idx]
    y_train_fold = y_train.iloc[train_idx]
    X_val_fold = X_train.iloc[val_idx]
    y_val_fold = y_train.iloc[val_idx]

    # LightGBM parameters
    params = {
        "objective": "multiclass",
        "num_class": n_classes,
        "boosting_type": "gbdt",
        "metric": "accuracy",
        "verbose": -1,
    }

    train_data = lgb.Dataset(X_train_fold, label=y_train_fold)
    val_data = lgb.Dataset(X_val_fold, label=y_val_fold, reference=train_data)

    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
    )

    y_pred = model.predict(X_val_fold)
    fold_scores.append(accuracy_score(y_val_fold, y_pred))

# Calculate mean CV accuracy
cv_accuracy = np.mean(fold_scores)
print(f"5-Fold Cross-Validation Accuracy: {cv_accuracy:.4f}")

# Save test predictions
y_pred_test = model.predict(X_test)
submission = pd.DataFrame({"id": test_ids, "Damage Description (total)": y_pred_test})
submission.to_csv("./working/submission.csv", index=False)


import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import numpy as np

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Define features and target
feature_cols = [
    col for col in train_df.columns if col not in ["id", "Damage Description (total)"]
]
target_col = "Damage Description (total)"

# Extract features and target
X_train = train_df[feature_cols]
y_train = train_df[target_col]
X_test = test_df[feature_cols]
test_ids = test_df["id"]

# Number of classes
n_classes = len(y_train.unique())

# 5-fold cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
best_model = None
best_score = 0

for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    X_train_fold = X_train.iloc[train_idx]
    y_train_fold = y_train.iloc[train_idx]
    X_val_fold = X_train.iloc[val_idx]
    y_val_fold = y_train.iloc[val_idx]

    # LightGBM parameters
    params = {
        "objective": "multiclass",
        "num_class": n_classes,
        "boosting_type": "gbdt",
        "metric": "accuracy",
        "verbose": -1,
        "random_state": 42,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_child_samples": 20,
        "min_child_weight": 1.0,
    }

    train_data = lgb.Dataset(X_train_fold, label=y_train_fold)
    val_data = lgb.Dataset(X_val_fold, label=y_val_fold, reference=train_data)

    # Train with early stopping
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
    )

    y_pred = model.predict(X_val_fold)
    fold_score = accuracy_score(y_val_fold, y_pred)

    print(f"Fold {fold_idx + 1} CV Accuracy: {fold_score:.4f}")

    if fold_score > best_score:
        best_score = fold_score
        best_model = model

# Calculate mean CV accuracy
cv_accuracy = np.mean(
    [
        accuracy_score(y_train.iloc[val_idx], model.predict(X_train.iloc[val_idx]))
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train))
    ]
)
print(f"5-Fold Cross-Validation Accuracy: {cv_accuracy:.4f}")

# Save test predictions
y_pred_test = best_model.predict(X_test)
submission = pd.DataFrame({"id": test_ids, "Damage Description (total)": y_pred_test})
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved to ./working/submission.csv")
