import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cohen_kappa_score


def calculate_qwk(y_true, y_pred):
    """Calculate weighted kappa score."""
    y_pred_binary = (y_pred >= 0.5).astype(int)
    return cohen_kappa_score(y_true, y_pred_binary, weights="quartile")


def main():
    # Load data
    train_data = pd.read_parquet("./input/train.parquet")
    test_data = pd.read_parquet("./input/test.parquet")

    # Preprocessing
    feature_cols = [col for col in train_data.columns if col not in ["Id", "Response"]]

    # Handle missing values efficiently
    train_data[feature_cols] = train_data[feature_cols].fillna(
        train_data[feature_cols].median()
    )
    test_data[feature_cols] = test_data[feature_cols].fillna(
        train_data[feature_cols].median()
    )

    # Encode categorical variables
    for col in feature_cols:
        if train_data[col].dtype == "object":
            train_data[col] = train_data[col].astype("category")
            test_data[col] = test_data[col].astype("category")

    # Prepare training data
    X_train = train_data[feature_cols].astype("float32")
    y_train = train_data["Response"]

    # Prepare test data
    X_test = test_data[feature_cols].astype("float32")
    test_ids = test_data["Id"] if "Id" in test_data.columns else range(len(test_data))

    # 5-Fold Cross Validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    model = None

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_train_fold = X_train.iloc[train_idx]
        y_train_fold = y_train.iloc[train_idx]
        X_val_fold = X_train.iloc[val_idx]
        y_val_fold = y_train.iloc[val_idx]

        # Create datasets with refine=False for memory efficiency
        train_set = lgb.Dataset(X_train_fold, label=y_train_fold, reference=None)
        val_set = lgb.Dataset(
            X_val_fold, label=y_val_fold, reference=train_set, refine=False
        )

        # Train model with callbacks
        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
        }

        callbacks = [lgb.early_stopping(50), lgb.log_evaluation(-1)]

        # Train
        model = lgb.train(params, train_set, valid_sets=[val_set], callbacks=callbacks)

        # Predict on validation set
        y_pred_fold = model.predict(X_val_fold)
        score = calculate_qwk(y_val_fold, y_pred_fold)
        cv_scores.append(score)

    # Calculate average CV score
    avg_cv_score = np.mean(cv_scores)
    print(f"Average Cross-Validation QWK Score: {avg_cv_score:.4f}")

    # Train final model on full training data
    train_set_final = lgb.Dataset(X_train, label=y_train, reference=None)
    final_model = lgb.train(
        params,
        train_set_final,
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
    )

    # Predict on test set
    y_pred_test = final_model.predict(X_test)

    # Prepare submission
    submission = pd.DataFrame({"id": test_ids, "Response": y_pred_test})
    submission.to_csv("./working/submission.csv", index=False)

    print("Submission saved to ./working/submission.csv")


if __name__ == "__main__":
    main()
