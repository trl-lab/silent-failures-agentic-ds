import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import os


def main():
    # Load data
    train_df = pd.read_parquet("./input/train.parquet")
    test_df = pd.read_parquet("./input/test.parquet")

    # Set target and features
    target_col = "funding"
    feature_cols = [col for col in train_df.columns if col != target_col]

    # Prepare data
    X_train = train_df[feature_cols].values
    y_train = train_df[target_col].values
    X_test = test_df[feature_cols].values
    test_ids = test_df.index.values

    # 5-fold cross-validation
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    # Store best model
    best_model = None
    best_score = float("inf")

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train)):
        X_train_fold = X_train[train_idx]
        y_train_fold = y_train[train_idx]
        X_val = X_train[val_idx]
        y_val = y_train[val_idx]

        # LightGBM parameters
        params = {
            "objective": "regression",
            "metric": "rmse",
            "verbose": -1,
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "min_child_samples": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "seed": 42,
        }

        # Create data
        train_data = lgb.Dataset(X_train_fold, label=y_train_fold)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        # Train model
        model = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
        )

        # Evaluate
        val_pred = model.predict(X_val)
        val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))

        if val_rmse < best_score:
            best_score = val_rmse
            best_model = model

    # Final evaluation
    if best_model:
        val_pred = best_model.predict(X_val)
        val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        print(f"Best validation RMSE: {val_rmse:.4f}")

    # Make predictions on test data
    test_pred = best_model.predict(X_test)

    # Create submission
    submission = pd.DataFrame({"id": test_ids, "funding": test_pred})

    # Save submission
    submission.to_csv("./working/submission.csv", index=False)
    print("Submission saved to ./working/submission.csv")


if __name__ == "__main__":
    main()


import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import os


def main():
    # Load data
    train_df = pd.read_parquet("./input/train.parquet")
    test_df = pd.read_parquet("./input/test.parquet")

    # Set target and features
    target_col = "funding"
    feature_cols = [col for col in train_df.columns if col != target_col]

    # Prepare data
    X_train = train_df[feature_cols].values
    y_train = train_df[target_col].values
    X_test = test_df[feature_cols].values
    test_ids = test_df.index.values

    # 5-fold cross-validation
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    best_model = None
    best_score = float("inf")

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train)):
        X_train_fold = X_train[train_idx]
        y_train_fold = y_train[train_idx]
        X_val = X_train[val_idx]
        y_val = y_train[val_idx]

        # LightGBM parameters
        params = {
            "objective": "regression",
            "metric": "rmse",
            "verbose": -1,
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "min_child_samples": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "seed": 42,
        }

        # Create data
        train_data = lgb.Dataset(X_train_fold, label=y_train_fold)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        # Train model with early stopping
        model = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
        )

        # Evaluate
        val_pred = model.predict(X_val)
        val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))

        if val_rmse < best_score:
            best_score = val_rmse
            best_model = model

    # Final evaluation on validation set
    if best_model:
        val_pred = best_model.predict(X_train[val_idx])
        val_rmse = np.sqrt(mean_squared_error(y_train[val_idx], val_pred))
        print(f"Best validation RMSE: {val_rmse:.4f}")

    # Make predictions on test data
    test_pred = best_model.predict(X_test)

    # Create submission
    submission = pd.DataFrame({"id": test_ids, "funding": test_pred})

    # Save submission
    submission.to_csv("./working/submission.csv", index=False)
    print("Submission saved to ./working/submission.csv")


if __name__ == "__main__":
    main()
