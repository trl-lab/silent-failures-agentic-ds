import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Define features and target
feature_cols = train_df.columns.drop(["id", "State Share"])
X_train = train_df[feature_cols]
y_train = train_df["State Share"]
X_test = test_df[feature_cols]
test_ids = test_df["id"]

# 5-fold cross-validation to estimate generalization performance
kf = KFold(n_splits=5, shuffle=True, random_state=42)
rmse_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_train_fold = X_train.iloc[train_idx]
    y_train_fold = y_train.iloc[train_idx]
    X_val_fold = X_train.iloc[val_idx]

    # LightGBM parameters
    params = {
        "objective": "regression",
        "metric": "rmse",
        "verbose": -1,
        "force_col_wise": True,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
    }

    train_data = lgb.Dataset(X_train_fold, label=y_train_fold)
    val_data = lgb.Dataset(X_val_fold, reference=train_data)

    # Train with callbacks
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
        valid_sets=[val_data],
    )

    # Calculate RMSE on validation fold
    y_pred = model.predict(X_val_fold)
    rmse = mean_squared_error(y_train.iloc[val_idx], y_pred)
    rmse_scores.append(rmse)

    print(f"Fold {fold+1}: RMSE = {rmse:.4f}")

# Average RMSE across folds
avg_rmse = np.mean(rmse_scores)
print(f"\nAverage Validation RMSE: {avg_rmse:.4f}")

# Train final model on all training data
final_params = {
    "objective": "regression",
    "metric": "rmse",
    "verbose": -1,
    "force_col_wise": True,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
}

final_model = lgb.train(
    final_params,
    lgb.Dataset(X_train, label=y_train),
    num_boost_round=1000,
    callbacks=[lgb.early_stopping(50)],
)

# Generate predictions for test set
test_predictions = final_model.predict(X_test)

# Create submission file
submission = pd.DataFrame({"id": test_ids, "State Share": test_predictions})
submission.to_csv("./working/submission.csv", index=False)
print("\nSubmission saved to ./working/submission.csv")
