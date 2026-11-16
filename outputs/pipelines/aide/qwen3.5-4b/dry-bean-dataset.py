import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Preprocessing: Fill NaN values in ShapeFactor columns
for col in ["ShapeFactor1", "ShapeFactor2", "ShapeFactor3", "ShapeFactor4"]:
    if col in train_df.columns:
        train_df[col] = train_df[col].fillna(train_df[col].median())
        test_df[col] = test_df[col].fillna(test_df[col].median())

# Feature columns (exclude Class)
feature_cols = [col for col in train_df.columns if col != "Class"]
X_train = train_df[feature_cols].copy()
y_train = train_df["Class"]
X_test = test_df[feature_cols].copy()

# Create train-validation split for validation
X_train_val, X_test_temp, y_train_val, y_test_temp = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

# LightGBM model
lgbm_model = lgb.LGBMClassifier(
    n_estimators=100,
    learning_rate=0.05,
    num_leaves=31,
    max_depth=6,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbose=-1,
)

# Train on train-validation split with early stopping
lgbm_model.fit(
    X_train_val,
    y_train_val,
    eval_set=[(X_test_temp, y_test_temp)],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
)

# Evaluate on validation set
y_pred_val = lgbm_model.predict(X_test_temp)
accuracy_val = accuracy_score(y_test_temp, y_pred_val)

# Cross-validation for final evaluation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(lgbm_model, X_train, y_train, cv=kf, scoring="accuracy")
accuracy_cv = cv_scores.mean()

# Predict on test set
y_pred_test = lgbm_model.predict(X_test)

# Save predictions
submission = pd.DataFrame({"id": range(len(y_pred_test)), "Class": y_pred_test})
submission.to_csv("./working/submission.csv", index=False)

# Print evaluation metrics
print(f"Validation accuracy: {accuracy_val:.4f}")
print(f"Cross-validation mean accuracy: {accuracy_cv:.4f}")
print(f"Cross-validation std: {cv_scores.std():.4f}")
print("Submission saved to ./working/submission.csv")
