import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score

# Load data
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

# Define features and target
target_col = "Heart Disease"
id_col = "id"

# Encode target
train[target_col] = train[target_col].map({"Presence": 1, "Absence": 0})
test[target_col] = test[target_col].map({"Presence": 1, "Absence": 0})

# Encode categorical features
categorical_cols = [
    "Chest pain type",
    "EKG results",
    "Exercise angina",
    "FBS over 120",
    "Number of vessels fluro",
    "Sex",
    "Slope of ST",
    "Thallium",
]
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col])
    test[col] = le.transform(test[col])
    label_encoders[col] = le

# Define feature columns
feature_cols = [c for c in train.columns if c not in [id_col, target_col]]

# Prepare training data
X_train = train[feature_cols]
y_train = train[target_col]
X_test = test[feature_cols]

# 5-fold cross-validation for evaluation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
best_score = 0.0

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]

    # Prepare LightGBM datasets
    train_data = lgb.Dataset(X_train_fold, label=y_train_fold)
    val_data = lgb.Dataset(X_val_fold, label=y_val_fold, reference=train_data)

    # Train model with early stopping
    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
    }

    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[train_data, val_data],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
    )

    # Predict on validation set
    y_pred = model.predict(X_val_fold)
    fold_auc = roc_auc_score(y_val_fold, y_pred)

    print(f"Fold {fold+1} AUC: {fold_auc:.4f}")

    if fold_auc > best_score:
        best_score = fold_auc

# Train final model on all training data
train_data_final = lgb.Dataset(X_train, label=y_train)
model_final = lgb.train(
    params,
    train_data_final,
    num_boost_round=1000,
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
)

# Predict on test set
y_test_pred = model_final.predict(X_test)

# Create submission
submission = pd.DataFrame({"id": test[id_col], "Heart Disease": y_test_pred})

# Save submission
submission.to_csv("./working/submission.csv", index=False)

print(f"\nFinal Cross-Validation AUC: {best_score:.4f}")
print("Submission saved to ./working/submission.csv")
