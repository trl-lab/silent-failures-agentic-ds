import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import numpy as np

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Define features and target
feature_cols = [col for col in train_df.columns if col != "class"]
X_train = train_df[feature_cols].values
y_train = train_df["class"].values
X_test = test_df[feature_cols].values

# Create 5-fold stratified cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Initialize LightGBM model
model = lgb.LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31,
    max_depth=-1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbose=-1,
)

# Store fold predictions for final validation
fold_predictions = []
fold_scores = []

# Perform 5-fold cross-validation
for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    X_train_fold = X_train[train_idx]
    y_train_fold = y_train[train_idx]
    X_val_fold = X_train[val_idx]
    y_val_fold = y_train[val_idx]

    # Train model on fold
    model.fit(
        X_train_fold,
        y_train_fold,
        eval_set=[(X_val_fold, y_val_fold)],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
    )

    # Evaluate on validation set
    val_pred = model.predict(X_val_fold)
    val_score = accuracy_score(y_val_fold, val_pred)
    fold_scores.append(val_score)

    # Store prediction for final test (using all training data)
    fold_predictions.append(model.predict(X_test))

# Calculate mean validation accuracy
mean_accuracy = np.mean(fold_scores)
print(f"Validation Accuracy: {mean_accuracy:.4f}")

# Combine all fold predictions for test set
test_predictions = np.concatenate(fold_predictions)

# Create submission dataframe
submission = pd.DataFrame(
    {"id": range(len(test_predictions)), "class": test_predictions.astype(int)}
)

# Save submission
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved to ./working/submission.csv")
