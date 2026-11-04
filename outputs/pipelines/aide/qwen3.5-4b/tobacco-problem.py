import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Prepare features and target
feature_cols = [
    col for col in train_df.columns if col not in ["number_health_problems"]
]
X_train = train_df[feature_cols].copy()
y_train = train_df["number_health_problems"].values
X_test = test_df[feature_cols].copy()

# Create train/validation split (80/20)
X_train_val, X_val, y_train_val, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

# Handle missing values
X_train_val = X_train_val.fillna(X_train_val.mean(numeric_only=True))
X_val = X_val.fillna(X_val.mean(numeric_only=True))
X_test = X_test.fillna(X_test.mean(numeric_only=True))

# Encode categorical features
le = LabelEncoder()
for col in feature_cols:
    if X_train_val[col].dtype == "object":
        X_train_val[col] = le.fit_transform(X_train_val[col].astype(str))
        X_val[col] = le.transform(X_val[col].astype(str))
        X_test[col] = le.transform(X_test[col].astype(str))

# LightGBM model with callbacks
model = lgb.LGBMClassifier(
    n_estimators=500, learning_rate=0.05, num_leaves=31, random_state=42, verbose=-1
)

callbacks = [lgb.early_stopping(50), lgb.log_evaluation(-1)]

# Train with validation set
model.fit(
    X_train_val, y_train_val, eval_set=[(X_val, y_val)], callbacks=callbacks, verbose=0
)

# Evaluate on validation set
y_val_pred = model.predict(X_val)
val_accuracy = accuracy_score(y_val, y_val_pred)
print(f"Validation Accuracy: {val_accuracy:.4f}")

# Generate test predictions
test_predictions = model.predict(X_test)

# Create submission file
submission = pd.DataFrame(
    {"id": range(len(test_predictions)), "number_health_problems": test_predictions}
)
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved to ./working/submission.csv")
