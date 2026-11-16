import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import lightgbm as lgb
import os

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Identify features and target
target_col = "Contraceptive_method_used"
feature_cols = [col for col in train_df.columns if col != target_col]

# Encode categorical features
label_encoders = {}
for col in feature_cols:
    le = LabelEncoder()
    train_df[col + "_encoded"] = le.fit_transform(train_df[col])
    test_df[col + "_encoded"] = le.transform(test_df[col])
    label_encoders[col] = le

# Create feature matrix and target
X_train = train_df[feature_cols + [col + "_encoded" for col in feature_cols]]
y_train = train_df[target_col]

# Split for validation (80-20 for initial validation)
X_train_val, X_test_temp, y_train_val, y_test_temp = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

# LightGBM model
model = lgb.LGBMClassifier(
    n_estimators=100,
    learning_rate=0.1,
    num_leaves=31,
    max_depth=-1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbose=-1,
)

# Train with early stopping
model.fit(
    X_train_val,
    y_train_val,
    eval_set=[(X_train_val, y_train_val)],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
)

# Evaluate on validation set
y_pred_val = model.predict(X_test_temp)
val_accuracy = accuracy_score(y_test_temp, y_pred_val)
print(f"Validation Accuracy: {val_accuracy:.4f}")

# Train final model on all training data
X_train_final = train_df[feature_cols + [col + "_encoded" for col in feature_cols]]
model.fit(
    X_train_final, y_train, callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)]
)

# Predict on test data
test_predictions = model.predict(
    test_df[feature_cols + [col + "_encoded" for col in feature_cols]]
)

# Create submission file
submission = pd.DataFrame(
    {"id": range(len(test_predictions)), "Contraceptive_method_used": test_predictions}
)
submission.to_csv("./working/submission.csv", index=False)

print("Submission saved to ./working/submission.csv")
print(f"Final Validation Accuracy: {val_accuracy:.4f}")
