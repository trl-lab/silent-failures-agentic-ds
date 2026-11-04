import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Feature engineering and preparation
feature_cols = [f for f in train_df.columns if f != "class"]
X_train = train_df[feature_cols].values
y_train = train_df["class"].values
X_test = test_df[feature_cols].values

# 5-fold cross-validation for evaluation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
predictions = []
scores = []

for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    X_tr, X_val = X_train[train_idx], X_train[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]

    # LightGBM parameters
    model = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        verbose=-1,
    )

    callbacks = [lgb.early_stopping(50), lgb.log_evaluation(-1)]
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=callbacks)

    y_pred = model.predict(X_val)
    scores.append(accuracy_score(y_val, y_pred))
    predictions.extend(y_pred)

# Compute mean CV accuracy
cv_accuracy = np.mean(scores)
print(f"5-Fold CV Accuracy: {cv_accuracy:.4f}")

# Final predictions on test set
y_test_pred = model.predict(X_test)
test_df["class"] = y_test_pred
submission = pd.DataFrame({"id": range(len(test_df)), "class": y_test_pred})
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved to ./working/submission.csv")
print(f"Test predictions shape: {submission.shape}")
print(f"Final CV Accuracy: {cv_accuracy:.4f}")
