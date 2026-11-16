import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error

TARGET = "critical_temp"


def load_and_prepare_data():
    train = pd.read_parquet("./input/train.parquet")
    test = pd.read_parquet("./input/test.parquet")

    train = train.dropna(subset=[TARGET]).reset_index(drop=True)

    exclude = {TARGET, "id", "ID"}
    feature_cols = [c for c in train.columns if c not in exclude and c in test.columns]

    X = train[feature_cols].apply(pd.to_numeric, errors="coerce")
    y = train[TARGET].apply(pd.to_numeric, errors="coerce")
    X_test = test[feature_cols].apply(pd.to_numeric, errors="coerce")

    X = X.replace([np.inf, -np.inf], np.nan)
    X_test = X_test.replace([np.inf, -np.inf], np.nan)

    feature_cols = [c for c in feature_cols if not X[c].isna().all()]

    return X[feature_cols], y, X_test[feature_cols], test


def main():
    X, y, X_test, test = load_and_prepare_data()

    params = {
        "objective": "regression_l2",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "max_depth": -1,
        "min_child_samples": 20,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l1": 1.0,
        "lambda_l2": 1.0,
        "seed": 42,
        "feature_fraction_seed": 42,
        "bagging_seed": 42,
        "verbosity": -1,
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_rmses = []
    best_iterations = []
    num_boost_round = 5000

    for fold, (train_idx, valid_idx) in enumerate(kf.split(X), start=1):
        X_train_fold = X.iloc[train_idx]
        y_train_fold = y.iloc[train_idx]
        X_valid_fold = X.iloc[valid_idx]
        y_valid_fold = y.iloc[valid_idx]

        train_dataset = lgb.Dataset(X_train_fold, label=y_train_fold)
        valid_dataset = lgb.Dataset(
            X_valid_fold,
            label=y_valid_fold,
            reference=train_dataset,
        )

        booster = lgb.train(
            params,
            train_dataset,
            num_boost_round=num_boost_round,
            valid_sets=[valid_dataset],
            valid_names=["validation"],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
        )

        best_iteration = getattr(booster, "best_iteration", 0) or 0
        if best_iteration <= 0:
            best_iteration = num_boost_round

        valid_pred = booster.predict(X_valid_fold, num_iteration=best_iteration)
        fold_rmse = root_mean_squared_error(y_valid_fold, valid_pred)

        fold_rmses.append(fold_rmse)
        best_iterations.append(best_iteration)
        print(f"Fold {fold} validation RMSE: {fold_rmse:.6f}")

    cv_rmse = float(np.mean(fold_rmses))
    print(f"5-Fold CV RMSE: {cv_rmse:.6f}")

    final_rounds = int(round(np.mean(best_iterations)))
    if final_rounds < 1:
        final_rounds = 100

    final_dataset = lgb.Dataset(X, label=y)
    final_booster = lgb.train(params, final_dataset, num_boost_round=final_rounds)

    test_pred = final_booster.predict(X_test, num_iteration=final_rounds)

    os.makedirs("./working", exist_ok=True)
    submission = pd.DataFrame(
        {
            "id": np.arange(len(test)),
            "critical_temp": test_pred,
        }
    )
    submission.to_csv("./working/submission.csv", index=False)
    print("Saved ./working/submission.csv")


if __name__ == "__main__":
    main()
