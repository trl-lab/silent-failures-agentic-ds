import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error


def main():
    # Load data
    train_df = pd.read_parquet("./input/train.parquet")
    test_df = pd.read_parquet("./input/test.parquet")

    # Define features
    feature_columns = [col for col in train_df.columns if col != "strength"]
    X_train = train_df[feature_columns]
    y_train = train_df["strength"]
    X_test = test_df[feature_columns]

    # 5-fold CV for evaluation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    rmse_scores = []

    for train_idx, val_idx in kf.split(X_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        # Create datasets for LightGBM
        train_data = lgb.Dataset(
            X_tr, label=y_tr, feature_name=feature_columns, reference_name=None
        )
        val_data = lgb.Dataset(
            X_val, label=y_val, feature_name=feature_columns, reference_name=None
        )

        # Define parameters
        params = {
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "min_child_samples": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        }

        # Train model
        callbacks = [lgb.early_stopping(50), lgb.log_evaluation(-1)]

        model = lgb.train(
            params,
            train_data,
            valid_data=val_data,
            callbacks=callbacks,
            num_boost_round=1000,  # Max rounds
        )

        # Predict on validation set
        y_pred = model.predict(X_val)
        rmse = mean_squared_error(y_val, y_pred, squared=False)
        rmse_scores.append(rmse)

    # Print evaluation metric
    print(f"5-Fold CV RMSE: {sum(rmse_scores) / len(rmse_scores):.4f}")

    # Train final model on full training data
    final_model = lgb.train(
        params,
        lgb.Dataset(X_train, label=y_train, feature_name=feature_columns),
        callbacks=callbacks,
        num_boost_round=1000,
    )

    # Predict on test set
    y_test_pred = final_model.predict(X_test)

    # Prepare submission
    submission = pd.DataFrame({"id": range(len(test_df)), "strength": y_test_pred})
    submission.to_csv("./working/submission.csv", index=False)
    print("Submission saved to ./working/submission.csv")


if __name__ == "__main__":
    main()
