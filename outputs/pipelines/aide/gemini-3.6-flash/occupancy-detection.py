import os
import numpy as np
import pandas as pd

import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score


def preprocess_and_feature_engineering(train_df, test_df):
    train_df = train_df.copy()
    test_df = test_df.copy()

    # Convert date to datetime
    train_df["date"] = pd.to_datetime(train_df["date"])
    test_df["date"] = pd.to_datetime(test_df["date"])

    # Retain test ordering for predictions
    test_df["orig_idx"] = np.arange(len(test_df))

    # Add temporal features
    for df in [train_df, test_df]:
        df.sort_values("date", inplace=True)
        df["hour"] = df["date"].dt.hour
        df["minute"] = df["date"].dt.minute
        df["second"] = df["date"].dt.second
        df["dayofweek"] = df["date"].dt.dayofweek
        df["time_in_seconds"] = df["hour"] * 3600 + df["minute"] * 60 + df["second"]

        # Temporal difference and rolling features
        sensor_cols = ["Temperature", "Humidity", "Light", "CO2", "HumidityRatio"]
        for col in sensor_cols:
            df[f"{col}_diff1"] = df[col].diff(1).fillna(0)
            df[f"{col}_diff2"] = df[col].diff(2).fillna(0)
            df[f"{col}_diff_next1"] = df[col].diff(-1).fillna(0)
            df[f"{col}_roll_mean3"] = (
                df[col].rolling(window=3, min_periods=1, center=True).mean()
            )
            df[f"{col}_roll_std3"] = (
                df[col].rolling(window=3, min_periods=1, center=True).std().fillna(0)
            )
            df[f"{col}_roll_mean5"] = (
                df[col].rolling(window=5, min_periods=1, center=True).mean()
            )
            df[f"{col}_roll_std5"] = (
                df[col].rolling(window=5, min_periods=1, center=True).std().fillna(0)
            )

    # Re-sort test back to original index order
    test_df = test_df.sort_values("orig_idx").drop(columns=["orig_idx"])

    return train_df, test_df


def main():
    # Load dataset
    train_path = "./input/train.parquet"
    test_path = "./input/test.parquet"

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    # Preprocess
    train_df, test_df = preprocess_and_feature_engineering(train_df, test_df)

    # Define feature columns and target
    ignore_cols = {"id", "date", "Occupancy"}
    feature_cols = [c for c in train_df.columns if c not in ignore_cols]
    target_col = "Occupancy"

    X = train_df[feature_cols].copy()
    y = train_df[target_col].copy()
    X_test = test_df[feature_cols].copy()

    # Stratified K-Fold setup
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    oof_preds = np.zeros(len(train_df))
    test_preds = np.zeros(len(test_df))

    lgb_params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "learning_rate": 0.03,
        "num_leaves": 31,
        "random_state": 42,
        "n_estimators": 1000,
        "verbose": -1,
        "n_jobs": -1,
    }

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]

        model = lgb.LGBMClassifier(**lgb_params)
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )

        val_probs = model.predict_proba(X_va)[:, 1]
        oof_preds[val_idx] = val_probs
        test_preds += model.predict_proba(X_test)[:, 1] / n_splits

    cv_score = roc_auc_score(y, oof_preds)
    print(f"5-Fold Cross-Validation ROC AUC Score: {cv_score:.6f}")

    # Prepare submission
    os.makedirs("./working", exist_ok=True)
    if "id" in test_df.columns:
        test_ids = test_df["id"].values
    else:
        test_ids = np.arange(len(test_df))

    sub = pd.DataFrame({"id": test_ids, "Occupancy": test_preds})
    sub.to_csv("./working/submission.csv", index=False)


if __name__ == "__main__":
    main()
