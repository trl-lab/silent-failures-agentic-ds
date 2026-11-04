import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score
import os

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Separate features and target
y = train_df["current_status"]
X = train_df.drop("current_status", axis=1)
X_test = (
    test_df.drop("current_status", axis=1)
    if "current_status" in test_df.columns
    else test_df
)

# Add ID column for test data if not present
if "id" not in test_df.columns:
    test_df["id"] = range(len(test_df))

# Handle categorical features
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

le = LabelEncoder()
for col in cat_cols:
    X[col] = le.fit_transform(X[col].astype(str))
    X_test[col] = le.transform(X_test[col].astype(str))

# Add one-hot encoding for categorical features
X_cat_encoded = pd.get_dummies(X[cat_cols], columns=cat_cols, drop_first=True)
X_test_encoded = pd.get_dummies(X_test[cat_cols], columns=cat_cols, drop_first=True)

# Combine with numerical features
X_combined = (
    pd.concat([X_num, X_cat_encoded], axis=1) if "X_num" in locals() else X_cat_encoded
)

# Scale numerical features
scaler = StandardScaler()
X_combined[num_cols] = scaler.fit_transform(X[num_cols])

# 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
roc_auc_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_combined)):
    X_train_fold = X_combined.iloc[train_idx]
    y_train_fold = y.iloc[train_idx]
    X_val_fold = X_combined.iloc[val_idx]
    y_val_fold = y.iloc[val_idx]

    # Prepare LightGBM data
    train_data = lgb.Dataset(X_train_fold, label=y_train_fold)
    val_data = lgb.Dataset(X_val_fold, label=y_val_fold, reference=train_data)

    # Train model
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
        "seed": 42,
    }

    callbacks = [
        lgb.early_stopping(50, verbose=False),
        lgb.log_evaluation(0, verbose=False),
    ]

    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        callbacks=callbacks,
    )

    # Get predictions on validation set
    y_pred = model.predict(X_val_fold)
    if len(np.unique(y)) == 2:
        fold_auc = roc_auc_score(y_val_fold, y_pred)
        roc_auc_scores.append(fold_auc)
        print(f"Fold {fold+1} ROC-AUC: {fold_auc:.4f}")

# Calculate mean ROC-AUC
mean_auc = np.mean(roc_auc_scores)
print(f"\nMean ROC-AUC (5-fold CV): {mean_auc:.4f}")

# Train final model on all training data
final_train_data = lgb.Dataset(X_combined, label=y)
final_model = lgb.train(
    params,
    final_train_data,
    num_boost_round=1000,
    callbacks=[lgb.early_stopping(50, verbose=False)],
)

# Generate predictions for test data
test_pred = final_model.predict(X_test_encoded)

# Create submission file
submission = pd.DataFrame({"id": range(len(test_df)), "current_status": test_pred})
submission.to_csv("./working/submission.csv", index=False)
print("\nSubmission saved to ./working/submission.csv")
