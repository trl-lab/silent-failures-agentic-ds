import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score


def main() -> None:
    input_dir = "./input"
    working_dir = "./working"
    os.makedirs(working_dir, exist_ok=True)

    train = pd.read_parquet(os.path.join(input_dir, "train.parquet"))
    test = pd.read_parquet(os.path.join(input_dir, "test.parquet"))

    target_col = "stabf"
    exclude_cols = {"stab", "stabf", "id", "index"}

    feature_cols = [
        col for col in train.columns if col not in exclude_cols and col in test.columns
    ]

    train = train.dropna(subset=[target_col]).copy()
    y = train[target_col].astype(int).to_numpy()
    X = train[feature_cols].copy()
    X_test = test[feature_cols].copy()

    for col in feature_cols:
        X[col] = pd.to_numeric(X[col], errors="coerce")
        X_test[col] = pd.to_numeric(X_test[col], errors="coerce")

    X = X.replace([np.inf, -np.inf], np.nan)
    X_test = X_test.replace([np.inf, -np.inf], np.nan)

    n_splits = 5
    random_state = 42
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    oof_preds = np.zeros(len(X), dtype=np.float64)
    best_iterations = []

    params = {
        "objective": "binary",
        "metric": ["auc"],
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "min_data_in_leaf": 20,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "seed": random_state,
        "verbose": -1,
    }

    callbacks = [
        lgb.early_stopping(stopping_rounds=50, verbose=False),
        lgb.log_evaluation(-1),
    ]

    for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y), start=1):
        X_train_fold = X.iloc[train_idx]
        y_train_fold = y[train_idx]
        X_valid_fold = X.iloc[valid_idx]
        y_valid_fold = y[valid_idx]

        dtrain = lgb.Dataset(X_train_fold, label=y_train_fold, free_raw_data=False)
        dvalid = lgb.Dataset(X_valid_fold, label=y_valid_fold, free_raw_data=False)

        booster = lgb.train(
            params,
            dtrain,
            num_boost_round=3000,
            valid_sets=[dvalid],
            valid_names=["valid"],
            callbacks=callbacks,
        )

        best_iter = max(1, int(booster.best_iteration))
        best_iterations.append(best_iter)

        oof_preds[valid_idx] = booster.predict(
            X_valid_fold,
            num_iteration=best_iter,
        )

        print(f"Fold {fold}: best_iteration={best_iter}")

    validation_auc = roc_auc_score(y, oof_preds)
    print(f"Validation ROC AUC (5-fold OOF): {validation_auc:.6f}")

    final_rounds = max(100, int(round(np.mean(best_iterations))))

    dtrain_full = lgb.Dataset(X, label=y, free_raw_data=False)
    final_model = lgb.train(
        params,
        dtrain_full,
        num_boost_round=final_rounds,
        callbacks=[lgb.log_evaluation(-1)],
    )

    test_preds = np.asarray(
        final_model.predict(X_test, num_iteration=final_rounds)
    ).reshape(-1)
    test_preds = np.clip(test_preds, 0.0, 1.0)

    submission = pd.DataFrame(
        {
            "id": np.arange(len(test), dtype=int),
            "stabf": test_preds,
        }
    )

    submission_path = os.path.join(working_dir, "submission.csv")
    submission.to_csv(submission_path, index=False)
    print(f"Saved submission to {submission_path} with {len(submission)} rows")


if __name__ == "__main__":
    main()
