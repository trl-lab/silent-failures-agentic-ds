import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Identify target column
target = "class"

# Feature columns (exclude id, target, and the first two variables mentioned as irrelevant)
feature_cols = [
    col
    for col in train_df.columns
    if col != target and col not in ["id", "image_id", "image_index"]
]

# Create stratified k-fold cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Initialize model
model = lgb.LGBMClassifier(
    n_estimators=100, learning_rate=0.1, num_leaves=31, random_state=42, verbose=-1
)

# Cross-validation to find best model
cv_scores = []
for fold, (train_idx, val_idx) in enumerate(
    skf.split(train_df[feature_cols], train_df[target])
):
    X_train, X_val = (
        train_df[feature_cols].iloc[train_idx],
        train_df[feature_cols].iloc[val_idx],
    )
    y_train = train_df[target].iloc[train_idx]
    y_val = train_df[target].iloc[val_idx]

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=50,
        verbose=-1,
    )
    y_pred = model.predict(X_val)
    cv_scores.append(accuracy_score(y_val, y_pred))

    print(f"Fold {fold+1}: {cv_scores[-1]:.4f}")

# Calculate mean accuracy
mean_accuracy = np.mean(cv_scores)
print(f"Mean CV Accuracy: {mean_accuracy:.4f}")

# Generate predictions for test set
X_test = test_df[feature_cols]
y_pred_test = model.predict(X_test)

# Create submission file
submission = pd.DataFrame({"id": range(len(test_df)), "class": y_pred_test})
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved to ./working/submission.csv")


import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Identify target column
target = "class"

# Feature columns (exclude id, target, and the first two variables mentioned as irrelevant)
feature_cols = [
    col
    for col in train_df.columns
    if col != target
    and col
    not in [
        "id",
        "image_id",
        "image_index",
        "short-line-density-5",
        "short-line-density-2",
    ]
]

# Create stratified k-fold cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Initialize model
model = lgb.LGBMClassifier(
    n_estimators=100,
    learning_rate=0.1,
    num_leaves=31,
    max_depth=5,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbose=-1,
)

# Cross-validation to find best model and compute validation accuracy
cv_scores = []
for fold, (train_idx, val_idx) in enumerate(
    skf.split(train_df[feature_cols], train_df[target])
):
    X_train, X_val = (
        train_df[feature_cols].iloc[train_idx],
        train_df[feature_cols].iloc[val_idx],
    )
    y_train = train_df[target].iloc[train_idx]
    y_val = train_df[target].iloc[val_idx]

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    model.fit(
        train_data,
        eval_set=[val_data],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
    )

    y_pred = model.predict(X_val)
    cv_scores.append(accuracy_score(y_val, y_pred))

    print(f"Fold {fold+1}: {accuracy_score(y_val, y_pred):.4f}")

# Calculate mean accuracy
mean_accuracy = np.mean(cv_scores)
print(f"Mean CV Accuracy: {mean_accuracy:.4f}")

# Generate predictions for test set
X_test = test_df[feature_cols]
y_pred_test = model.predict(X_test)

# Create submission file
submission = pd.DataFrame({"id": range(len(test_df)), "class": y_pred_test})
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved to ./working/submission.csv")
