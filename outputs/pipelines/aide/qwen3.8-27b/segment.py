import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Remove target and any explicit id column before feature selection.
    df = df.drop(columns=["class", "id"], errors="ignore")

    # Ignore the first three variables as described in the dataset notes.
    if df.shape[1] > 3:
        df = df.iloc[:, 3:]

    def to_numeric(col):
        if not pd.api.types.is_numeric_dtype(col):
            col = col.astype(object)
        return pd.to_numeric(col, errors="coerce")

    df = df.apply(to_numeric)
    df = df.replace([np.inf, -np.inf], np.nan)
    return df


def main() -> None:
    os.makedirs("./working", exist_ok=True)

    train = pd.read_parquet("./input/train.parquet")
    test = pd.read_parquet("./input/test.parquet")

    X_train = prepare_features(train)
    X_test = prepare_features(test)

    # Ensure train/test feature alignment.
    X_test = X_test.reindex(columns=X_train.columns)
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)

    # Encode target to integer labels for LightGBM.
    y_raw = train["class"].reset_index(drop=True)
    encoder = LabelEncoder()
    y_train = encoder.fit_transform(y_raw.to_numpy())
    n_classes = len(encoder.classes_)

    params = {
        "objective": "multiclass",
        "num_class": n_classes,
        "metric": "multi_logloss",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 5,
        "min_child_samples": 20,
        "verbose": -1,
        "seed": 42,
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    best_iterations = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), start=1):
        dtrain = lgb.Dataset(X_train.iloc[train_idx], label=y_train[train_idx])
        dval = lgb.Dataset(
            X_train.iloc[val_idx], label=y_train[val_idx], reference=dtrain
        )

        model = lgb.train(
            params,
            dtrain,
            num_boost_round=1000,
            valid_sets=[dval],
            valid_names=["val"],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
        )

        best_iter = getattr(model, "best_iteration", 0)
        if best_iter is None:
            best_iter = 0
        best_iter = max(1, int(best_iter))

        val_pred = model.predict(X_train.iloc[val_idx], num_iteration=best_iter)
        val_pred_class = np.argmax(val_pred, axis=1)
        acc = accuracy_score(y_train[val_idx], val_pred_class)

        cv_scores.append(acc)
        best_iterations.append(best_iter)
        print(f"Fold {fold} Accuracy: {acc:.4f}")

    mean_acc = float(np.mean(cv_scores))
    print(f"Mean CV Accuracy: {mean_acc:.4f}")

    # Train final model on all training data using the average best iteration from CV.
    avg_best_iteration = max(1, int(np.mean(best_iterations)))
    dtrain_full = lgb.Dataset(X_train, label=y_train)
    model_full = lgb.train(
        params,
        dtrain_full,
        num_boost_round=avg_best_iteration,
        callbacks=[lgb.log_evaluation(-1)],
    )

    test_pred = model_full.predict(X_test)
    test_pred_class = np.argmax(test_pred, axis=1)
    test_pred_labels = encoder.inverse_transform(test_pred_class)

    submission = pd.DataFrame(
        {
            "id": np.arange(len(test_pred_labels), dtype=int),
            "class": test_pred_labels,
        }
    )
    submission.to_csv("./working/submission.csv", index=False)


if __name__ == "__main__":
    main()
