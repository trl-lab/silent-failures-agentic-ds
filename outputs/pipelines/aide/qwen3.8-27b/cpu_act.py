import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error


def main():
    os.makedirs("./working", exist_ok=True)

    train = pd.read_parquet("./input/train.parquet")
    test = pd.read_parquet("./input/test.parquet")

    target = "usr"
    drop_cols = {target}
    for col in ("id", "Id", "ID"):
        if col in train.columns:
            drop_cols.add(col)

    features = [c for c in train.columns if c not in drop_cols]

    X = train[features].copy()
    y = train[target].astype(float)
    X_test = test.reindex(columns=features).copy()

    for col in features:
        X[col] = pd.to_numeric(X[col], errors="coerce")
        X_test[col] = pd.to_numeric(X_test[col], errors="coerce")

    X = X.replace([np.inf, -np.inf], np.nan)
    X_test = X_test.replace([np.inf, -np.inf], np.nan)
    y = y.replace([np.inf, -np.inf], np.nan)

    valid_mask = y.notna()
    X = X.loc[valid_mask].copy().reset_index(drop=True)
    y = y.loc[valid_mask].copy().reset_index(drop=True)

    params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 20,
        "subsample": 0.9,
        "subsample_freq": 1,
        "colsample_bytree": 0.9,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "seed": 42,
        "bagging_seed": 42,
        "feature_fraction_seed": 42,
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_pred = np.zeros(len(y), dtype=float)
    best_iterations = []
    max_rounds = 3000

    for fold, (train_idx, val_idx) in enumerate(kf.split(X), start=1):
        train_data = lgb.Dataset(X.iloc[train_idx], label=y.iloc[train_idx])
        val_data = lgb.Dataset(
            X.iloc[val_idx],
            label=y.iloc[val_idx],
            reference=train_data,
        )

        model = lgb.train(
            params,
            train_data,
            num_boost_round=max_rounds,
            valid_sets=[val_data],
            valid_names=["valid"],
            callbacks=[
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(-1),
            ],
        )

        best_iter = model.best_iteration if model.best_iteration > 0 else max_rounds
        best_iterations.append(best_iter)

        val_pred = model.predict(X.iloc[val_idx], num_iteration=best_iter)
        val_pred = np.nan_to_num(val_pred, nan=0.0, posinf=100.0, neginf=0.0)
        val_pred = np.clip(val_pred, 0.0, 100.0)

        oof_pred[val_idx] = val_pred

        fold_rmse = float(np.sqrt(mean_squared_error(y.iloc[val_idx], val_pred)))
        print(f"Fold {fold} RMSE: {fold_rmse:.6f}, best iteration: {best_iter}")

    cv_rmse = float(np.sqrt(mean_squared_error(y, oof_pred)))
    print(f"5-fold CV RMSE: {cv_rmse:.6f}")

    median_best = float(np.median(best_iterations))
    final_rounds = int(median_best) if median_best > 0 else 100

    full_data = lgb.Dataset(X, label=y)
    final_model = lgb.train(params, full_data, num_boost_round=final_rounds)

    test_pred = final_model.predict(X_test)
    test_pred = np.nan_to_num(test_pred, nan=0.0, posinf=100.0, neginf=0.0)
    test_pred = np.clip(test_pred, 0.0, 100.0)

    submission = pd.DataFrame(
        {
            "id": np.arange(len(test_pred)),
            "usr": test_pred,
        }
    )
    submission.to_csv("./working/submission.csv", index=False)
    print("Saved ./working/submission.csv")


if __name__ == "__main__":
    main()
