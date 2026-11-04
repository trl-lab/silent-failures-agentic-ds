import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# Load data
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

# Identify features and target
target_col = "price"
feature_cols = [col for col in train.columns if col != target_col]

# Create feature matrix and target
X_train = train[feature_cols].copy()
y_train = train[target_col].copy()
X_test = test[feature_cols].copy()

# Handle missing values
X_train = X_train.fillna(X_train.median())
X_test = X_test.fillna(X_test.median())

# Create parameter set
params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
}

# 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_train_fold = X_train.iloc[train_idx]
    y_train_fold = y_train.iloc[train_idx]
    X_val_fold = X_train.iloc[val_idx]

    train_data = lgb.Dataset(X_train_fold, label=y_train_fold)
    val_data = lgb.Dataset(
        X_val_fold, label=y_train.iloc[val_idx], reference=train_data
    )

    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[train_data, val_data],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )

    y_pred_val = model.predict(X_val_fold)
    cv_scores.append(mean_squared_error(y_train.iloc[val_idx], y_pred_val))

# Average CV score
cv_rmse = np.sqrt(np.mean(cv_scores))
print(f"5-Fold CV RMSE: {cv_rmse:.4f}")

# Train final model on all training data
train_data_full = lgb.Dataset(X_train, label=y_train)
final_model = lgb.train(
    params,
    train_data_full,
    num_boost_round=1000,
    callbacks=[lgb.early_stopping(50, verbose=False)],
)

# Predict on test data
y_test_pred = final_model.predict(X_test)

# Create submission
submission = pd.DataFrame({"id": range(len(X_test)), "price": y_test_pred})

# Save submission
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved to ./working/submission.csv")
