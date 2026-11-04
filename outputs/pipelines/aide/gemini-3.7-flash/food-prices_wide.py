import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error


def preprocess_data(train_df, test_df, target_col):
    df = pd.concat(
        [train_df.assign(is_train=1), test_df.assign(is_train=0)], ignore_index=True
    )

    # Identify and extract datetime features
    for col in df.columns:
        if col in [target_col, "is_train", "id"]:
            continue
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[f"{col}_year"] = df[col].dt.year
            df[f"{col}_month"] = df[col].dt.month
            df[f"{col}_day"] = df[col].dt.day
            df[f"{col}_dayofweek"] = df[col].dt.dayofweek
            df[f"{col}_quarter"] = df[col].dt.quarter
            df.drop(columns=[col], inplace=True)
        elif df[col].dtype == "object":
            # Check if it can be parsed as datetime
            try:
                converted = pd.to_datetime(df[col], errors="coerce")
                if converted.notna().sum() > 0.8 * len(df):
                    df[f"{col}_year"] = converted.dt.year
                    df[f"{col}_month"] = converted.dt.month
                    df[f"{col}_day"] = converted.dt.day
                    df[f"{col}_dayofweek"] = converted.dt.dayofweek
                    df[f"{col}_quarter"] = converted.dt.quarter
                    df.drop(columns=[col], inplace=True)
                else:
                    df[col] = df[col].astype("category")
            except Exception:
                df[col] = df[col].astype("category")
        elif df[col].dtype == "category":
            df[col] = df[col].astype("category")

    train_processed = df[df["is_train"] == 1].drop(columns=["is_train"])
    test_processed = df[df["is_train"] == 0].drop(columns=["is_train", target_col])

    return train_processed, test_processed


def main():
    target_col = "usdprice"

    # Load data
    train_df = pd.read_parquet("./input/train.parquet")
    test_df = pd.read_parquet("./input/test.parquet")

    # Preprocess
    train_processed, test_processed = preprocess_data(train_df, test_df, target_col)

    feature_cols = [c for c in train_processed.columns if c != target_col]

    X = train_processed[feature_cols]
    y = train_processed[target_col].values
    X_test = test_processed[feature_cols]

    # 5-Fold Cross Validation
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    oof_predictions = np.zeros(len(train_processed))
    test_predictions = np.zeros(len(test_processed))

    params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "n_estimators": 2000,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }

    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        X_train, y_train = X.iloc[train_idx], y[train_idx]
        X_val, y_val = X.iloc[val_idx], y[val_idx]

        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
        )

        val_pred = model.predict(X_val)
        oof_predictions[val_idx] = val_pred
        test_predictions += model.predict(X_test) / n_splits

        fold_rmse = root_mean_squared_error(y_val, val_pred)
        print(f"Fold {fold + 1} RMSE: {fold_rmse:.5f}")

    overall_rmse = root_mean_squared_error(y, oof_predictions)
    print(f"Overall OOF RMSE: {overall_rmse:.5f}")

    # Save submission
    os.makedirs("./working", exist_ok=True)
    submission = pd.DataFrame(
        {"id": np.arange(len(test_predictions)), "usdprice": test_predictions}
    )
    submission.to_csv("./working/submission.csv", index=False)
    print("Submission saved to ./working/submission.csv")


if __name__ == "__main__":
    main()
