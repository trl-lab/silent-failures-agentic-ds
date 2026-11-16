import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss


def main():
    os.makedirs("./working", exist_ok=True)

    train = pd.read_parquet("./input/train.parquet")
    test = pd.read_parquet("./input/test.parquet")

    feature_cols = [c for c in train.columns if c.startswith("feat_")]
    feature_cols = [c for c in feature_cols if c in test.columns]

    if not feature_cols:
        feature_cols = [c for c in train.columns if c not in ("id", "target")]

    for col in feature_cols:
        train[col] = pd.to_numeric(train[col], errors="coerce").fillna(0.0)
        test[col] = pd.to_numeric(test[col], errors="coerce").fillna(0.0)

    class_labels = sorted(train["target"].unique().tolist())
    class_to_idx = {label: i for i, label in enumerate(class_labels)}
    y = train["target"].map(class_to_idx).astype(int).to_numpy()

    X = train[feature_cols].to_numpy(dtype=np.float32)
    X_test = test[feature_cols].to_numpy(dtype=np.float32)

    params = {
        "objective": "multiclass",
        "num_class": len(class_labels),
        "metric": "multi_logloss",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "min_child_samples": 20,
        "lambda_l1": 0.1,
        "lambda_l2": 0.1,
        "verbose": -1,
        "seed": 42,
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    val_losses = []
    best_iterations = []
    num_boost_round = 1000

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        dtrain = lgb.Dataset(
            X[train_idx],
            label=y[train_idx],
            feature_name=feature_cols,
            free_raw_data=False,
        )
        dval = lgb.Dataset(
            X[val_idx],
            label=y[val_idx],
            feature_name=feature_cols,
            reference=dtrain,
            free_raw_data=False,
        )

        booster = lgb.train(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            valid_set=[dval],
            callbacks=[
                lgb.early_stopping(50, first_metric_only=True, verbose=False),
                lgb.log_evaluation(-1),
            ],
        )

        best_iter = booster.best_iteration
        if best_iter is None or best_iter <= 0:
            best_iter = num_boost_round

        best_iterations.append(best_iter)
        val_pred = booster.predict(X[val_idx], num_iteration=best_iter)
        fold_loss = log_loss(
            y[val_idx], val_pred, labels=list(range(len(class_labels)))
        )
        val_losses.append(fold_loss)
        print(f"Fold {fold} log_loss: {fold_loss:.6f}")

    mean_log_loss = float(np.mean(val_losses))
    std_log_loss = float(np.std(val_losses))
    print(f"Validation log_loss: {mean_log_loss:.6f} +/- {std_log_loss:.6f}")

    final_rounds = int(np.median(best_iterations))
    if final_rounds <= 0:
        final_rounds = num_boost_round

    dtrain_full = lgb.Dataset(
        X,
        label=y,
        feature_name=feature_cols,
        free_raw_data=False,
    )

    final_booster = lgb.train(
        params,
        dtrain_full,
        num_boost_round=final_rounds,
        callbacks=[lgb.log_evaluation(-1)],
    )

    test_pred = final_booster.predict(X_test)
    test_pred = np.clip(test_pred, 1e-15, 1.0)
    test_pred /= test_pred.sum(axis=1, keepdims=True)

    pred_labels = np.array(class_labels)[test_pred.argmax(axis=1)]

    submission = pd.DataFrame(
        {
            "id": np.arange(len(test), dtype=int),
            "target": pred_labels,
        }
    )
    submission.to_csv("./working/submission.csv", index=False)
    print("Saved ./working/submission.csv")


if __name__ == "__main__":
    main()
