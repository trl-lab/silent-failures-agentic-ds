import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import numpy as np

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Define feature columns (exclude target and id if present)
feature_cols = [col for col in train_df.columns if col not in ["Rating", "id"]]
X_train = train_df[feature_cols]
y_train = train_df["Rating"]
X_test = test_df[feature_cols]

# 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    # Split for this fold
    X_train_fold = X_train.iloc[train_idx]
    y_train_fold = y_train.iloc[train_idx]
    X_val_fold = X_train.iloc[val_idx]
    y_val_fold = y_train.iloc[val_idx]

    # Create LightGBM datasets
    train_data = lgb.Dataset(X_train_fold, label=y_train_fold)
    val_data = lgb.Dataset(X_val_fold, label=y_val_fold, reference=train_data)

    # Train model with early stopping
    params = {
        "objective": "multiclass",
        "num_class": len(y_train_fold.unique()),
        "metric": "accuracy",
        "verbose": -1,
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_child_samples": 20,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
    }

    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
    )

    # Predict on validation set
    y_pred = model.predict(X_val_fold, num_iteration=model.best_iteration)
    y_pred = np.round(y_pred).astype(int)
    fold_scores.append(accuracy_score(y_val_fold, y_pred))

# Calculate average CV score
cv_score = np.mean(fold_scores)
print(f"5-Fold Cross-Validation Score: {cv_score:.4f}")

# Train final model on all training data
train_data_final = lgb.Dataset(X_train, label=y_train)
final_model = lgb.train(
    params,
    train_data_final,
    num_boost_round=500,
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
)

# Predict on test set
y_pred_test = final_model.predict(X_test, num_iteration=final_model.best_iteration)
y_pred_test = np.round(y_pred_test).astype(int)

# Create submission
submission = pd.DataFrame({"id": range(len(y_pred_test)), "Rating": y_pred_test})
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved to ./working/submission.csv")
