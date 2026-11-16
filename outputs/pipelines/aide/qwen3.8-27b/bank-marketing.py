import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score


def main() -> None:
    os.makedirs("./working", exist_ok=True)

    train = pd.read_parquet("./input/train.parquet")
    test = pd.read_parquet("./input/test.parquet")

    target = "Class"

    y = train[target].copy()
    X = train.drop(columns=[target]).copy()
    X_test = test.drop(columns=[target], errors="ignore").copy()

    # Ignore identifier-like columns and keep only features present in both splits.
    ignore_cols = {"id", "index", "row_id", "customer_id", "client_id"}
    feature_cols = [
        c for c in X.columns if c in X_test.columns and c.lower() not in ignore_cols
    ]

    # Drop features that are entirely missing in the training data.
    feature_cols = [c for c in feature_cols if X[c].notna().any()]

    X = X[feature_cols]
    X_test = X_test[feature_cols]

    categorical_candidates = [
        "job",
        "marital",
        "education",
        "default",
        "housing",
        "loan",
        "contact",
        "month",
        "poutcome",
    ]

    cat_cols = [c for c in categorical_candidates if c in feature_cols]
    num_cols = [c for c in feature_cols if c not in cat_cols]

    # Clean numeric columns.
    for col in num_cols:
        X[col] = pd.to_numeric(X[col], errors="coerce")
        X_test[col] = pd.to_numeric(X_test[col], errors="coerce")

    # Align categorical categories between train and test.
    for col in cat_cols:
        X[col] = X[col].astype("category")
        X_test[col] = pd.Categorical(
            X_test[col],
            categories=X[col].cat.categories,
        )

    # Encode target as 1 for yes, 0 for no.
    if y.dtype == object:
        y = y.astype(str).str.strip().str.lower().eq("yes").astype(int)
    else:
        y = y.astype(int)

    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_child_samples": 20,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "seed": 42,
        "verbosity": -1,
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []
    best_iterations = []
    num_boost_round = 2000

    for train_idx, valid_idx in skf.split(X, y):
        X_train = X.iloc[train_idx]
        X_valid = X.iloc[valid_idx]
        y_train = y.iloc[train_idx]
        y_valid = y.iloc[valid_idx]

        train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_cols)
        valid_data = lgb.Dataset(
            X_valid,
            label=y_valid,
            categorical_feature=cat_cols,
            reference=train_data,
        )

        model = lgb.train(
            params,
            train_data,
            num_boost_round=num_boost_round,
            valid_sets=[valid_data],
            valid_names=["valid"],
            callbacks=[
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(-1),
            ],
        )

        best_iter = (
            model.best_iteration if model.best_iteration > 0 else num_boost_round
        )
        valid_pred = model.predict(X_valid, num_iteration=best_iter)

        if len(np.unique(y_valid)) < 2:
            aucs.append(0.5)
        else:
            aucs.append(roc_auc_score(y_valid, valid_pred))

        best_iterations.append(best_iter)

    print(f"5-fold CV ROC AUC: {np.mean(aucs):.6f} +/- {np.std(aucs):.6f}")

    # Refit on the full training set using the average best iteration from CV.
    best_rounds = int(np.clip(np.mean(best_iterations), 50, num_boost_round))

    full_data = lgb.Dataset(X, label=y, categorical_feature=cat_cols)
    final_model = lgb.train(
        params,
        full_data,
        num_boost_round=best_rounds,
        callbacks=[lgb.log_evaluation(-1)],
    )

    test_pred = np.asarray(final_model.predict(X_test)).reshape(-1)
    test_pred = np.clip(test_pred, 0.0, 1.0)

    submission = pd.DataFrame(
        {
            "id": np.arange(len(X_test)),
            "Class": test_pred,
        }
    )

    submission.to_csv("./working/submission.csv", index=False)
    print(f"Saved submission with {len(submission)} rows.")


if __name__ == "__main__":
    main()
