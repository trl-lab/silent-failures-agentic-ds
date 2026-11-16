import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

# Load data
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

# Extract features and target
X_train = train.drop("VOLT_CLASS", axis=1)
y_train = train["VOLT_CLASS"]
X_test = test.drop("VOLT_CLASS", axis=1) if "VOLT_CLASS" in test.columns else test

# Create 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Initialize model
model = lgb.LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    max_depth=-1,
    verbose=-1,
)

# Train with 5-fold CV and print validation accuracy
best_score = 0
for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_train_fold = X_train.iloc[train_idx]
    y_train_fold = y_train.iloc[train_idx]
    X_val_fold = X_train.iloc[val_idx]
    y_val_fold = y_train.iloc[val_idx]

    # Train model on fold
    model.fit(
        X_train_fold,
        y_train_fold,
        eval_set=[(X_val_fold, y_val_fold)],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
    )

    # Get predictions
    val_pred = model.predict(X_val_fold)
    fold_acc = accuracy_score(y_val_fold, val_pred)

    print(f"Fold {fold+1} validation accuracy: {fold_acc:.4f}")
    best_score = max(best_score, fold_acc)

print(f"Best validation accuracy: {best_score:.4f}")

# Generate predictions for test data
test_pred = model.predict(X_test)

# Save to submission.csv
submission = pd.DataFrame({"id": range(len(test_pred)), "VOLT_CLASS": test_pred})
submission.to_csv("./working/submission.csv", index=False)


import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

# Load data
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

# Extract features and target
X_train = train.drop("VOLT_CLASS", axis=1)
y_train = train["VOLT_CLASS"]
X_test = test.drop("VOLT_CLASS", axis=1)

# Identify categorical columns for encoding
categorical_cols = X_train.select_dtypes(include=["object"]).columns.tolist()
numerical_cols = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()

# One-hot encode categorical columns
X_train_encoded = pd.get_dummies(X_train, columns=categorical_cols, drop_first=True)
X_test_encoded = pd.get_dummies(X_test, columns=categorical_cols, drop_first=True)

# Ensure same columns in test as train
X_test_encoded = X_test_encoded.reindex(columns=X_train_encoded.columns, fill_value=0)

# Create 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Initialize model
model = lgb.LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    max_depth=-1,
    verbose=-1,
)

# Train with 5-fold CV and print validation accuracy
best_score = 0
for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_encoded)):
    X_train_fold = X_train_encoded.iloc[train_idx]
    y_train_fold = y_train.iloc[train_idx]
    X_val_fold = X_train_encoded.iloc[val_idx]
    y_val_fold = y_train.iloc[val_idx]

    # Train model on fold
    model.fit(
        X_train_fold,
        y_train_fold,
        eval_set=[(X_val_fold, y_val_fold)],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
    )

    # Get predictions
    val_pred = model.predict(X_val_fold)
    fold_acc = accuracy_score(y_val_fold, val_pred)

    print(f"Fold {fold+1} validation accuracy: {fold_acc:.4f}")
    best_score = max(best_score, fold_acc)

print(f"Best validation accuracy: {best_score:.4f}")

# Generate predictions for test data
test_pred = model.predict(X_test_encoded)

# Save to submission.csv
submission = pd.DataFrame({"id": range(len(test_pred)), "VOLT_CLASS": test_pred})
submission.to_csv("./working/submission.csv", index=False)
