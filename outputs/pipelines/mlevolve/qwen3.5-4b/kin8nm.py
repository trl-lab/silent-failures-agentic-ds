import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import warnings

warnings.filterwarnings("ignore")

# Set random seeds for reproducibility
np.random.seed(42)


def main():
    # Load data
    train_df = pd.read_parquet("./input/train.parquet")
    test_df = pd.read_parquet("./input/test.parquet")

    # Separate features and target
    X_train = train_df.drop("y", axis=1)
    y_train = train_df["y"]

    X_test = test_df.drop("id", axis=1) if "id" in test_df.columns else test_df

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Create validation set
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train_scaled, y_train, test_size=0.2, random_state=42
    )

    # Define ensemble configurations
    configs = [
        {"num_leaves": 31, "learning_rate": 0.05, "bagging_fraction": 0.8},
        {"num_leaves": 63, "learning_rate": 0.03, "bagging_fraction": 0.7},
        {"num_leaves": 128, "learning_rate": 0.01, "bagging_fraction": 0.9},
    ]

    # Create datasets
    train_data = lgb.Dataset(X_train_split, label=y_train_split)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    # Train ensemble of 3 models with different configurations
    models = []
    for i, config in enumerate(configs):
        # Create model-specific parameters
        params = {
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "num_leaves": config["num_leaves"],
            "learning_rate": config["learning_rate"],
            "feature_fraction": 0.9,
            "bagging_fraction": config["bagging_fraction"],
            "bagging_freq": 5,
            "verbose": -1,
            "seed": 42,
        }

        # Train model
        model = lgb.train(
            params,
            train_data,
            num_boost_round=500,
            valid_sets=[train_data, val_data],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
        )
        models.append(model)

        # Validate each model
        y_val_pred = model.predict(X_val)
        rmse_score = np.sqrt(mean_squared_error(y_val, y_val_pred))
        print(f"Model {i+1} Validation Score: {rmse_score:.6f}")

    # Average predictions from ensemble models
    y_test_pred = np.mean([model.predict(X_test_scaled) for model in models], axis=0)

    # Create submission DataFrame
    submission_df = pd.DataFrame(
        {
            "id": (
                test_df["id"].values if "id" in test_df.columns else range(len(test_df))
            ),
            "y": y_test_pred,
        }
    )

    # Save submission
    submission_df.to_csv("./submission/submission.csv", index=False)
    print("Submission saved to ./submission/submission.csv")


if __name__ == "__main__":
    main()
