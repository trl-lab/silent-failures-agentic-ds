import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

TARGET = "APPETENCY"
RANDOM_STATE = 42
N_SPLITS = 5
EARLY_STOPPING_ROUNDS = 50
MAX_BOOST_ROUND = 2000


def ensure_columns(df: pd.DataFrame, feature_order: list[str]) -> pd.DataFrame:
    df = df.copy()
    for col in feature_order:
        if col not in df.columns:
            df[col] = np.nan
    return df[feature_order]


def clean_features(
    df: pd.DataFrame,
    feature_order: list[str],
    categorical_features: list[str],
    categorical_dtypes: dict[str, pd.CategoricalDtype] | None = None,
) -> pd.DataFrame:
    df = ensure_columns(df, feature_order)
    categorical_set = set(categorical_features)

    for col in feature_order:
        if col in categorical_set:
            if categorical_dtypes is not None and col in categorical_dtypes:
                df[col] = df[col].astype(categorical_dtypes[col])
            else:
                df[col] = df[col].astype("category")
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df


def main() -> None:
    os.makedirs("./working", exist_ok=True)

    train = pd.read_parquet("./input/train.parquet")
    test = pd.read_parquet("./input/test.parquet")

    if TARGET not in train.columns:
        raise ValueError(f"Target column '{TARGET}' not found in train data.")

    y = (train[TARGET] > 0).astype(int)
    X = train.drop(columns=[TARGET])

    if "id" in X.columns:
        X = X.drop(columns=["id"])

    feature_order = list(X.columns)

    if len(feature_order) >= 40:
        categorical_features = feature_order[-40:]
    else:
        categorical_features = []

    X = clean_features(
        X,
        feature_order=feature_order,
        categorical_features=categorical_features,
        categorical_dtypes=None,
    )

    categorical_dtypes = {
        col: X[col].cat.dtype for col in categorical_features if col in X.columns
    }

    test = test.drop(columns=["id"], errors="ignore")
    test = clean_features(
        test,
        feature_order=feature_order,
        categorical_features=categorical_features,
        categorical_dtypes=categorical_dtypes,
    )

    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "max_depth": -1,
        "min_child_samples": 50,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l1": 1.0,
        "lambda_l2": 1.0,
        "seed": RANDOM_STATE,
        "verbosity": -1,
        "num_threads": -1,
    }

    callbacks = [
        lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
        lgb.log_evaluation(-1),
    ]

    skf = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    oof_predictions = np.zeros(len(X), dtype=float)
    fold_aucs = []
    best_iterations = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        train_data = X.iloc[train_idx]
        val_data = X.iloc[val_idx]
        train_labels = y.iloc[train_idx]
        val_labels = y.iloc[val_idx]

        train_set = lgb.Dataset(
            data=train_data,
            label=train_labels,
            categorical_feature=categorical_features,
        )
        val_set = lgb.Dataset(
            data=val_data,
            label=val_labels,
            categorical_feature=categorical_features,
            reference=train_set,
        )

        model = lgb.train(
            params,
            train_set,
            num_boost_round=MAX_BOOST_ROUND,
            valid_sets=[val_set],
            valid_names=["valid"],
            callbacks=callbacks,
        )

        best_iteration = max(1, int(model.best_iteration))
        best_iterations.append(best_iteration)

        val_pred = model.predict(val_data, num_iteration=best_iteration)
        oof_predictions[val_idx] = val_pred

        if len(np.unique(val_labels)) >= 2:
            fold_auc = roc_auc_score(val_labels, val_pred)
        else:
            fold_auc = 0.5

        fold_aucs.append(fold_auc)

    validation_auc = float(np.mean(fold_aucs))
    print(f"Validation ROC AUC: {validation_auc:.6f}")

    final_boost_round = int(np.max(best_iterations)) if best_iterations else 500
    final_boost_round = max(1, min(final_boost_round, MAX_BOOST_ROUND))

    full_train_set = lgb.Dataset(
        data=X,
        label=y,
        categorical_feature=categorical_features,
    )

    final_model = lgb.train(
        params,
        full_train_set,
        num_boost_round=final_boost_round,
        callbacks=[lgb.log_evaluation(-1)],
    )

    test_predictions = final_model.predict(test)
    test_predictions = np.clip(test_predictions, 0.0, 1.0)

    submission = pd.DataFrame(
        {
            "id": np.arange(len(test), dtype=int),
            "APPETENCY": test_predictions,
        }
    )

    submission.to_csv("./working/submission.csv", index=False)


if __name__ == "__main__":
    main()
