import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Identify features (exclude target and id)
target_col = "WATERDEPTH"
feature_cols = [col for col in train_df.columns if col != target_col and col != "id"]

# Prepare features and target
X_train = train_df[feature_cols].copy()
y_train = train_df[target_col].copy()
X_test = test_df[feature_cols].copy()

# Create 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
predictions = []
scores = []

# Train model with cross-validation
for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_train_fold = X_train.iloc[train_idx]
    y_train_fold = y_train.iloc[train_idx]
    X_val_fold = X_train.iloc[val_idx]
    y_val_fold = y_train.iloc[val_idx]

    # Create LightGBM datasets
    train_data = lgb.Dataset(X_train_fold, label=y_train_fold)
    val_data = lgb.Dataset(X_val_fold, label=y_val_fold, reference=train_data)

    # Train model
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
        "seed": 42,
    }

    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[train_data, val_data],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )

    # Get validation predictions
    val_pred = model.predict(X_val_fold)
    fold_rmse = mean_squared_error(y_val_fold, val_pred, squared=False)
    scores.append(fold_rmse)
    predictions.append(val_pred)

# Combine all predictions
all_predictions = np.concatenate(predictions)
cv_rmse = np.sqrt(np.mean(scores))

# Now train final model on all training data for test predictions
final_params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
    "seed": 42,
}

final_model = lgb.train(
    final_params,
    lgb.Dataset(X_train, label=y_train),
    num_boost_round=1000,
    callbacks=[lgb.early_stopping(50, verbose=False)],
)

# Predict on test data
test_predictions = final_model.predict(X_test)

# Create submission file
submission = pd.DataFrame({"id": range(len(test_df)), "WATERDEPTH": test_predictions})
submission.to_csv("./working/submission.csv", index=False)

# Print evaluation metric
print(f"Cross-validation RMSE: {cv_rmse:.4f}")
