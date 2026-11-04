import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Extract features and target
feature_cols = [
    col
    for col in train_df.columns
    if col not in ["Mouse ID", "Genotype", "Treatment type", "Behavior", "class"]
]
target_col = "class"

# Encode categorical features
categorical_cols = ["Genotype", "Treatment type", "Behavior"]
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    le.fit(train_df[col].astype(str))
    label_encoders[col] = le
    train_df[col] = le.transform(train_df[col].astype(str))
    test_df[col] = le.transform(test_df[col].astype(str))

# Prepare features and target
X_train = train_df[feature_cols].values
y_train = train_df[target_col].values
X_test = test_df[feature_cols].values

# Multi-class LightGBM with 5-fold stratified CV
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
best_score = 0
best_params = {
    "num_leaves": 31,
    "learning_rate": 0.05,
    "n_estimators": 1000,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
}

for train_idx, val_idx in skf.split(X_train, y_train):
    X_tr, X_val = X_train[train_idx], X_train[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]

    train_data = lgb.Dataset(X_tr, label=y_tr)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    params = {
        "objective": "multiclass",
        "num_class": len(np.unique(y_train)),
        "metric": "accuracy",
        **best_params,
    }

    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[train_data, val_data],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )

    val_pred = model.predict(X_val)
    val_acc = accuracy_score(y_val, np.round(val_pred).astype(int))
    print(f"Fold {train_idx}: Validation Accuracy = {val_acc:.4f}")
    best_score = max(best_score, val_acc)

# Train final model on all training data
train_data = lgb.Dataset(X_train, label=y_train)
final_params = {
    "objective": "multiclass",
    "num_class": len(np.unique(y_train)),
    "metric": "accuracy",
    **best_params,
}
final_model = lgb.train(
    final_params,
    train_data,
    num_boost_round=1000,
    callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
)

# Generate predictions for test set
test_pred = final_model.predict(X_test)
test_pred_labels = np.round(test_pred).astype(int)

# Create submission file
submission = pd.DataFrame({"id": test_df["Mouse ID"].values, "class": test_pred_labels})
submission.to_csv("./working/submission.csv", index=False)
print(f"\nFinal Test Predictions Saved to ./working/submission.csv")
print(f"Best Cross-Validation Accuracy: {best_score:.4f}")
