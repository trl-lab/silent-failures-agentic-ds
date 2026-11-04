import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Define feature columns (exclude target, identifiers, and fold)
target = "ViolentCrimesPerPop"
exclude_cols = ["state", "county", "community", "fold", target]
feature_cols = [col for col in train_df.columns if col not in exclude_cols]

X_train = train_df[feature_cols]
y_train = train_df[target]
X_test = test_df[feature_cols]

# Prepare LightGBM dataset
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, reference=train_data)

# Define callbacks
callbacks = [lgb.early_stopping(50), lgb.log_evaluation(-1)]

# Train LightGBM model with 5-fold cross-validation
best_score = np.inf
best_model = None

for fold_idx, (train_idx, val_idx) in enumerate(
    KFold(n_splits=5, shuffle=True, random_state=42)
):
    # Split for this fold
    X_train_fold = X_train.iloc[train_idx]
    y_train_fold = y_train.iloc[train_idx]
    X_val_fold = X_train.iloc[val_idx]
    y_val_fold = y_train.iloc[val_idx]

    # Prepare datasets for this fold
    train_data_fold = lgb.Dataset(X_train_fold, label=y_train_fold)
    val_data_fold = lgb.Dataset(X_val_fold, reference=train_data_fold)

    # Train model for this fold
    model = lgb.train(
        {
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
        },
        train_set=train_data_fold,
        valid_sets=[val_data_fold],
        callbacks=callbacks,
        num_boost_round=500,
    )

    # Evaluate on validation set
    val_pred = model.predict(X_val_fold)
    fold_rmse = mean_squared_error(y_val_fold, val_pred)
    print(f"Fold {fold_idx + 1} RMSE: {fold_rmse:.4f}")

    # Keep track of best model
    if fold_rmse < best_score:
        best_score = fold_rmse
        best_model = model

# Evaluate on full validation set (average of all folds)
final_rmse = np.mean(
    [
        mean_squared_error(y_train.iloc[val_idx], model.predict(X_train.iloc[val_idx]))
        for fold_idx, (train_idx, val_idx) in enumerate(
            KFold(n_splits=5, shuffle=True, random_state=42)
        )
    ]
)
print(f"Average Validation RMSE: {final_rmse:.4f}")

# Generate predictions for test set
test_pred = best_model.predict(X_test)

# Create submission file
submission = pd.DataFrame({"id": range(len(test_df)), "ViolentCrimesPerPop": test_pred})
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved to ./working/submission.csv")
print(
    f"Final Test RMSE: {np.sqrt(mean_squared_error(y_train.iloc[:0], test_pred)):.4f}"
)
