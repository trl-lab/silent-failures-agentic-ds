import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import numpy as np

# Load data
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

# Identify target and features
target = "Debt-to-Earnings Annual Rate"
X_train = train.drop(columns=[target])
y_train = train[target]
X_test = (
    test.drop(columns=["Debt-to-Earnings Annual Rate"])
    if "Debt-to-Earnings Annual Rate" in test.columns
    else test
)

# Ensure column alignment
columns = X_train.columns
X_test = X_test[columns]

# 5-Fold Cross-Validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

for train_idx, val_idx in kfold.split(X_train):
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

    train_data = lgb.Dataset(X_tr, label=y_tr)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    # Train model with early stopping and logging
    model = lgb.train(
        {"objective": "regression", "metric": "rmse", "verbose": -1},
        train_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
    )

    preds = model.predict(X_val)
    rmse = mean_squared_error(y_val, preds, squared=False)
    cv_scores.append(rmse)

# Print evaluation metric
print(f"Cross-Validation RMSE: {np.sqrt(np.mean(cv_scores)):.4f}")

# Final Model Training on Full Training Data
train_data_full = lgb.Dataset(X_train, label=y_train)
final_model = lgb.train(
    {"objective": "regression", "metric": "rmse", "verbose": -1},
    train_data_full,
    num_boost_round=1000,
    callbacks=[lgb.early_stopping(50)],
)

# Generate Predictions for Test Set
preds_test = final_model.predict(X_test)

# Create Submission
submission = pd.DataFrame(
    {"id": range(len(X_test)), "Debt-to-Earnings Annual Rate": preds_test}
)

# Save Submission
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved to ./working/submission.csv")
