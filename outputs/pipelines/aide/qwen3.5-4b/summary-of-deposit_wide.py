import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Handle missing values
train_df = train_df.dropna()
test_df = test_df.dropna()

# Extract features and target
X_train = train_df.drop("DEPSUM", axis=1)
y_train = train_df["DEPSUM"]
X_test = test_df

# 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
best_score = np.inf
best_model = None

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_train_fold = X_train.iloc[train_idx]
    y_train_fold = y_train.iloc[train_idx]
    X_val_fold = X_train.iloc[val_idx]
    y_val_fold = y_train.iloc[val_idx]

    # Create LightGBM datasets
    train_data = lgb.Dataset(X_train_fold, label=y_train_fold)
    val_data = lgb.Dataset(X_val_fold, label=y_val_fold, reference=train_data)

    # Train model with callbacks
    model = lgb.train(
        params={
            "objective": "regression",
            "metric": "mse",
            "verbose": -1,
            "num_leaves": 31,
            "learning_rate": 0.05,
            "min_child_samples": 20,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
        },
        train_set=train_data,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(-1)],
        num_boost_round=1000,
    )

    # Evaluate on validation fold
    y_pred_val = model.predict(X_val_fold)
    fold_rmse = np.sqrt(mean_squared_error(y_val_fold, y_pred_val))
    print(f"Fold {fold+1} RMSE: {fold_rmse:.4f}")

    if fold_rmse < best_score:
        best_score = fold_rmse
        best_model = model

print(f"Best 5-fold CV RMSE: {best_score:.4f}")

# Make predictions on test set
test_predictions = best_model.predict(X_test)

# Create submission file
submission = pd.DataFrame({"id": range(len(test_df)), "DEPSUM": test_predictions})
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved to ./working/submission.csv")
