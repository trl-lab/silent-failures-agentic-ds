import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Prepare features and target
X_train = train_df.drop("outcomes", axis=1)
y_train = train_df["outcomes"]
X_test = test_df

# Convert object columns to appropriate types for LightGBM
# Identify object columns
object_cols = X_train.select_dtypes(include=["object"]).columns.tolist()

# Convert object columns to numeric where possible, otherwise handle as categorical
for col in object_cols:
    # Try to convert to numeric (e.g., dates, numbers stored as strings)
    X_train[col] = pd.to_numeric(X_train[col], errors="coerce")
    X_test[col] = pd.to_numeric(X_test[col], errors="coerce")

    # Fill NaN with a safe default (0 or median)
    X_train[col] = X_train[col].fillna(X_train[col].median())
    X_test[col] = X_test[col].fillna(X_test[col].median())

# Identify remaining object columns that couldn't be converted to numeric
remaining_object_cols = X_train.select_dtypes(include=["object"]).columns.tolist()

# Create categorical feature list for LightGBM
categorical_feature = remaining_object_cols
print(f"Categorical features: {categorical_feature}")

# 5-fold cross-validation with LightGBM
kf = KFold(n_splits=5, shuffle=True, random_state=42)
predictions = []
scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_train_fold = X_train.iloc[train_idx]
    y_train_fold = y_train.iloc[train_idx]
    X_val_fold = X_train.iloc[val_idx]
    y_val_fold = y_train.iloc[val_idx]

    # Create LightGBM model
    model = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        verbose=-1,
        categorical_feature=categorical_feature,
    )

    # Callbacks
    callbacks = [lgb.early_stopping(50), lgb.log_evaluation(-1)]

    # Train model
    model.fit(
        X_train_fold,
        y_train_fold,
        eval_set=[(X_val_fold, y_val_fold)],
        callbacks=callbacks,
    )

    # Get predictions for validation fold
    val_pred = model.predict(X_val_fold)
    val_acc = accuracy_score(y_val_fold, val_pred)
    scores.append(val_acc)
    predictions.extend(val_pred)

# Calculate mean accuracy
mean_accuracy = np.mean(scores)
print(f"Mean Cross-Validation Accuracy: {mean_accuracy:.4f}")

# Save predictions for test data
test_pred = model.predict(X_test)
submission = pd.DataFrame({"id": range(len(test_pred)), "outcomes": test_pred})
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved to ./working/submission.csv")
