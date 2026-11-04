import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Load data
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

# Define features and target
feature_cols = [
    "Age",
    "CAEC",
    "CALC",
    "CH2O",
    "FAF",
    "FAVC",
    "FCVC",
    "Gender",
    "Height",
    "MTRANS",
    "NCP",
    "SCC",
    "SMOKE",
    "TUE",
    "Weight",
    "family_history_with_overweight",
]
target_col = "NObeyesdad"
id_col = "id"

# Encode categorical features
categorical_cols = [
    "CAEC",
    "CALC",
    "FAVC",
    "Gender",
    "MTRANS",
    "SCC",
    "SMOKE",
    "family_history_with_overweight",
]
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col].astype(str))
    test[col] = le.transform(test[col].astype(str))
    label_encoders[col] = le

# Train set for training
X_train = train[feature_cols]
y_train = train[target_col]
X_test = test[feature_cols]
test_ids = test[id_col]

# Convert target to one-hot encoding for multi-class
y_train_encoded = pd.get_dummies(y_train, columns=[target_col])
y_train_encoded = y_train_encoded.values

# LightGBM parameters
params = {
    "objective": "multiclass",
    "num_class": 7,
    "metric": "accuracy",
    "verbose": -1,
    "seed": 42,
}

# 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
best_cv_score = 0
best_model = None

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_train_fold = X_train.iloc[train_idx]
    y_train_fold = y_train_encoded.iloc[train_idx]
    X_val_fold = X_train.iloc[val_idx]
    y_val_fold = y_train_encoded.iloc[val_idx]

    # Create datasets
    train_data = lgb.Dataset(X_train_fold, label=y_train_fold)
    val_data = lgb.Dataset(X_val_fold, label=y_val_fold, reference=train_data)

    # Train model
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
    )

    # Evaluate on validation fold
    y_pred = model.predict(X_val_fold)
    y_pred_classes = np.argmax(y_pred, axis=1)
    fold_score = accuracy_score(y_val_fold.argmax(axis=1), y_pred_classes)

    print(f"Fold {fold+1}: Accuracy = {fold_score:.4f}")

    if fold_score > best_cv_score:
        best_cv_score = fold_score
        best_model = model

# Generate predictions on test set
y_test_pred = best_model.predict(X_test)
y_test_pred_classes = np.argmax(y_test_pred, axis=1)

# Create submission
submission = pd.DataFrame({id_col: test_ids, target_col: y_test_pred_classes})
submission.to_csv("./working/submission.csv", index=False)

print(f"\nBest CV Accuracy: {best_cv_score:.4f}")
print("Submission saved to ./working/submission.csv")
