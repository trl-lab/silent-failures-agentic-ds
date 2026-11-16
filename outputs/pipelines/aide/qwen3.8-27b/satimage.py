import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score


def main():
    train = pd.read_parquet("./input/train.parquet")
    test = pd.read_parquet("./input/test.parquet")

    target = "class"
    exclude_columns = {target, "id", "Id", "ID"}
    feature_cols = [c for c in train.columns if c not in exclude_columns]

    train = train.dropna(subset=[target]).reset_index(drop=True)

    X = train[feature_cols].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = train[target].astype(int)

    X_test = (
        test[feature_cols].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    )

    classes = np.unique(y)
    class_to_idx = {int(c): i for i, c in enumerate(classes)}
    idx_to_class = {i: int(c) for c, i in class_to_idx.items()}
    y_idx = y.map(class_to_idx).to_numpy()

    params = {
        "objective": "multiclass",
        "num_class": len(classes),
        "metric": "multi_logloss",
        "learning_rate": 0.05,
        "num_leaves": 15,
        "max_depth": 6,
        "min_data_in_leaf": 10,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "lambda_l1": 0.1,
        "lambda_l2": 1.0,
        "seed": 42,
        "verbosity": -1,
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    val_accuracies = []
    best_iterations = []
    num_boost_round = 1000

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_idx)):
        dtrain = lgb.Dataset(X.iloc[train_idx], label=y_idx[train_idx])
        dvalid = lgb.Dataset(X.iloc[val_idx], label=y_idx[val_idx], reference=dtrain)

        booster = lgb.train(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            valid_sets=[dvalid],
            callbacks=[
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(-1),
            ],
        )

        val_pred_idx = np.argmax(booster.predict(X.iloc[val_idx]), axis=1)
        val_accuracies.append(accuracy_score(y_idx[val_idx], val_pred_idx))

        best_iter = getattr(booster, "best_iteration", 0)
        best_iterations.append(best_iter if best_iter > 0 else num_boost_round)

    cv_accuracy = float(np.mean(val_accuracies))
    print(f"Validation accuracy (5-fold CV mean): {cv_accuracy:.4f}")

    final_rounds = max(1, int(round(np.mean(best_iterations))))
    dtrain_full = lgb.Dataset(X, label=y_idx)

    final_booster = lgb.train(
        params,
        dtrain_full,
        num_boost_round=final_rounds,
        callbacks=[lgb.log_evaluation(-1)],
    )

    test_pred_idx = np.argmax(final_booster.predict(X_test), axis=1)
    test_pred_class = np.array([idx_to_class[i] for i in test_pred_idx], dtype=int)

    os.makedirs("./working", exist_ok=True)
    submission = pd.DataFrame(
        {
            "id": np.arange(len(test_pred_class), dtype=int),
            "class": test_pred_class,
        }
    )
    submission.to_csv("./working/submission.csv", index=False)


if __name__ == "__main__":
    main()
