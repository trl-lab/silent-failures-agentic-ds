import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error


def main() -> None:
    os.makedirs("./working", exist_ok=True)

    train = pd.read_parquet("./input/train.parquet")
    test = pd.read_parquet("./input/test.parquet")

    target_col = "loss"
    id_col = "id"

    feature_cols = [
        col
        for col in train.columns
        if col not in {target_col, id_col} and col in test.columns
    ]

    X = train[feature_cols].copy()
    y = train[target_col].astype(float)
    X_test = test[feature_cols].copy()

    params = {
        "objective": "regression_l1",
        "metric": "l1",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "max_depth": -1,
        "min_child_samples": 20,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "lambda_l1": 1.0,
        "lambda_l2": 1.0,
        "max_cat_threshold": 255,
        "max_cat_to_onehot": 4,
        "seed": 42,
    }

    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    val_maes = []
    test_predictions = np.zeros(len(X_test), dtype=float)

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X), start=1):
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_val = X.iloc[val_idx]
        y_val = y.iloc[val_idx]

        dtrain = lgb.Dataset(X_train, label=y_train, categorical_feature=feature_cols)
        dvalid = lgb.Dataset(
            X_val,
            label=y_val,
            categorical_feature=feature_cols,
            reference=dtrain,
        )

        model = lgb.train(
            params,
            dtrain,
            num_boost_round=5000,
            valid_sets=[dvalid],
            valid_names=["valid"],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
        )

        val_pred = model.predict(X_val, num_iteration=model.best_iteration)
        fold_mae = mean_absolute_error(y_val, val_pred)
        val_maes.append(fold_mae)

        print(f"Fold {fold} MAE: {fold_mae:.6f}")

        test_predictions += (
            model.predict(X_test, num_iteration=model.best_iteration) / 5.0
        )

    cv_mae = float(np.mean(val_maes))
    print(f"CV MAE: {cv_mae:.6f}")

    submission = pd.DataFrame(
        {
            "id": np.arange(len(X_test), dtype=int),
            "loss": test_predictions,
        }
    )
    submission.to_csv("./working/submission.csv", index=False)


if __name__ == "__main__":
    main()
