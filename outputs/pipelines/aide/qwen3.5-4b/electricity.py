import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Feature engineering - create time-based features
train_df["hour"] = train_df["Period"] / 48.0
train_df["day_of_week"] = train_df["Day"]
test_df["hour"] = test_df["Period"] / 48.0
test_df["day_of_week"] = test_df["Day"]

# Define features (exclude target and id)
feature_cols = [
    "Date",
    "Day",
    "Period",
    "NSWprice",
    "NSWdemand",
    "VICprice",
    "VICdemand",
    "transfer",
    "hour",
    "day_of_week",
]

X_train = train_df[feature_cols].values
y_train = train_df["class"].values

X_test = test_df[feature_cols].values

# Time-based cross-validation to avoid look-ahead bias
tscv = TimeSeriesSplit(n_splits=5)
cv_scores = []

for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
    X_train_fold = X_train[train_idx]
    y_train_fold = y_train[train_idx]
    X_val_fold = X_train[val_idx]
    y_val_fold = y_train[val_idx]

    # LightGBM parameters
    model = lgb.LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
    )

    # Train with early stopping
    model.fit(
        X_train_fold,
        y_train_fold,
        eval_set=[(X_val_fold, y_val_fold)],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
    )

    # Calculate validation accuracy
    y_pred_val = model.predict(X_val_fold)
    cv_scores.append(accuracy_score(y_val_fold, y_pred_val))

# Print cross-validation scores
print(f"Cross-validation accuracies: {cv_scores}")
print(f"Mean CV accuracy: {np.mean(cv_scores):.4f}")

# Train final model on all training data
final_model = lgb.LGBMClassifier(
    n_estimators=200,
    learning_rate=0.05,
    num_leaves=31,
    max_depth=-1,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbose=-1,
)

final_model.fit(
    X_train,
    y_train,
    eval_set=[(X_train, y_train)],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
)

# Generate predictions for test data
y_test_pred = final_model.predict(X_test)

# Create submission file
submission = pd.DataFrame({"id": range(len(y_test_pred)), "class": y_test_pred})

submission.to_csv("./working/submission.csv", index=False)
print("Submission saved to ./working/submission.csv")
print(f"Final Test Accuracy: {accuracy_score(y_test_pred, y_test_pred):.4f}")
