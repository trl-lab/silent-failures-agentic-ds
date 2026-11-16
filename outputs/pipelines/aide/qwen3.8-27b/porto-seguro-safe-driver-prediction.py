import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score


def normalized_gini(y_true, y_score):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=np.float64)
    y_score = np.nan_to_num(y_score, nan=0.5)

    if len(np.unique(y_true)) < 2 or np.all(y_score == y_score[0]):
        return 0.0

    auc = roc_auc_score(y_true, y_score)
    return float(2.0 * auc - 1.0)


def main():
    os.makedirs("./working", exist_ok=True)

    train = pd.read_parquet("./input/train.parquet")
    test = pd.read_parquet("./input/test.parquet")

    target_col = "target"
    id_col = "id"

    y = train[target_col].astype(int)
    feature_cols = [c for c in train.columns if c not in (id_col, target_col)]

    X = train[feature_cols].copy()
    X_test = test[feature_cols].copy()

    for col in feature_cols:
        if (X[col] == -1).any():
            X[col] = X[col].replace(-1, np.nan)
        if (X_test[col] == -1).any():
            X_test[col] = X_test[col].replace(-1, np.nan)

    X = X.astype(np.float32)
    X_test = X_test.astype(np.float32)

    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "max_depth": -1,
        "min_child_samples": 50,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "seed": 42,
        "num_threads": -1,
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(X.shape[0], dtype=np.float64)
    best_iters = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        dtrain = lgb.Dataset(
            X.iloc[train_idx],
            label=y.iloc[train_idx],
            feature_name=feature_cols,
        )
        dvalid = lgb.Dataset(
            X.iloc[val_idx],
            label=y.iloc[val_idx],
            feature_name=feature_cols,
        )

        model = lgb.train(
            params,
            dtrain,
            valid_set=dvalid,
            num_boost_round=5000,
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
        )

        oof[val_idx] = model.predict(
            X.iloc[val_idx], num_iteration=model.best_iteration
        )
        best_iters.append(model.best_iteration)
        print(f"Fold {fold} best_iteration={model.best_iteration}")

    cv_metric = normalized_gini(y.values, oof)
    print(f"Normalized Gini (5-fold CV): {cv_metric:.6f}")

    best_rounds = max(1, int(np.mean(best_iters)))

    dtrain_full = lgb.Dataset(X, label=y, feature_name=feature_cols)
    final_model = lgb.train(
        params,
        dtrain_full,
        num_boost_round=best_rounds,
        callbacks=[lgb.log_evaluation(-1)],
    )

    test_pred = final_model.predict(X_test, num_iteration=best_rounds)
    test_pred = np.clip(test_pred, 0.0, 1.0)

    submission = pd.DataFrame(
        {
            "id": np.arange(len(test_pred), dtype=int),
            "target": test_pred,
        }
    )
    submission.to_csv("./working/submission.csv", index=False)
    print("Saved ./working/submission.csv")


if __name__ == "__main__":
    main()
