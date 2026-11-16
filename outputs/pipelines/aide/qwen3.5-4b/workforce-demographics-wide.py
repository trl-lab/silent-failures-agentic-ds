import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Prepare features and target
feature_cols = [col for col in train_df.columns if col != "Female (%)"]
X_train = train_df[feature_cols]
y_train = train_df["Female (%)"]
X_test = test_df[feature_cols]

# Create 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = []

# Train and evaluate on each fold
for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_train_fold = X_train.iloc[train_idx]
    y_train_fold = y_train.iloc[train_idx]
    X_val_fold = X_train.iloc[val_idx]
    y_val_fold = y_train.iloc[val_idx]

    # Create LightGBM datasets
    train_data = lgb.Dataset(X_train_fold, label=y_train_fold)
    val_data = lgb.Dataset(X_val_fold, label=y_val_fold, reference=train_data)

    # Train model with callbacks
    params = {
        "objective": "regression",
        "metric": "rmse",
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
        num_boost_round=500,
        valid_sets=[train_data, val_data],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
    )

    # Evaluate on validation set
    y_pred_val = model.predict(X_val_fold)
    rmse = mean_squared_error(y_val_fold, y_pred_val)
    scores.append(rmse)
    print(f"Fold {fold+1} RMSE: {rmse:.4f}")

# Calculate mean RMSE across folds
mean_rmse = np.mean(scores)
print(f"\nMean RMSE on 5-fold CV: {mean_rmse:.4f}")

# Generate predictions on test set
test_predictions = model.predict(X_test)

# Create submission file
submission = pd.DataFrame(
    {"id": range(len(test_predictions)), "Female (%)": test_predictions}
)
submission.to_csv("./working/submission.csv", index=False)
print("\nSubmission saved to ./working/submission.csv")
print(f"Final RMSE: {mean_rmse:.4f}")
