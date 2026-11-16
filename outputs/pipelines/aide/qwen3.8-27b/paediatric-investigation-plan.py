import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score

TARGET = "Decision type"
RANDOM_STATE = 42
NUM_BOOST_ROUND = 2000
EARLY_STOPPING_ROUNDS = 50


def prepare_features(train: pd.DataFrame, test: pd.DataFrame, target: str):
    """
    Prepare train/test feature matrices with train-only encoding.
    Returns X_train, X_test, categorical_columns.
    """
    if target in train.columns:
        train = train.drop(columns=[target])
    if target in test.columns:
        test = test.drop(columns=[target])

    # Keep only columns present in both train and test.
    common_cols = [c for c in train.columns if c in test.columns]

    # Drop obvious identifier-like columns, if present.
    id_like = {"id", "index", "row_id", "rowid", "identifier", "uid", "sample_id"}
    common_cols = [c for c in common_cols if c.lower() not in id_like]

    train = train[common_cols].copy()
    test = test[common_cols].copy()

    # Drop columns that are all missing or constant in the training set.
    drop_cols = []
    for col in train.columns:
        if train[col].isna().all() or train[col].nunique(dropna=True) <= 1:
            drop_cols.append(col)
    train.drop(columns=drop_cols, inplace=True)
    test.drop(columns=drop_cols, inplace=True)

    if train.shape[1] == 0:
        return train, test, []

    categorical_cols = []
    for col in train.columns:
        dtype_name = train[col].dtype.name
        if (
            train[col].dtype == object
            or dtype_name in {"category", "string"}
            or train[col].dtype == bool
        ):
            categorical_cols.append(col)
        else:
            train[col] = pd.to_numeric(train[col], errors="coerce")
            test[col] = pd.to_numeric(test[col], errors="coerce")
            train[col] = train[col].replace([np.inf, -np.inf], np.nan)
            test[col] = test[col].replace([np.inf, -np.inf], np.nan)

    # Encode categoricals using mappings learned only from the training data.
    for col in categorical_cols:
        train[col] = train[col].astype(object).fillna("missing").astype(str)
        test[col] = test[col].astype(object).fillna("missing").astype(str)

        categories = sorted(train[col].unique())
        mapping = {cat: i for i, cat in enumerate(categories)}

        train[col] = train[col].map(mapping).astype(int)
        test[col] = test[col].map(mapping).fillna(-1).astype(int)

    return train, test, categorical_cols


def main():
    os.makedirs("./working", exist_ok=True)

    train = pd.read_parquet("./input/train.parquet")
    test = pd.read_parquet("./input/test.parquet")

    if TARGET not in train.columns:
        raise ValueError(f"Target column '{TARGET}' not found in train data.")

    y_raw = train[TARGET].copy()
    valid_mask = y_raw.notna()
    train = train[valid_mask].reset_index(drop=True)
    y_raw = y_raw[valid_mask].reset_index(drop=True)

    X_train, X_test, categorical_cols = prepare_features(train, test, TARGET)

    classes = np.unique(y_raw.astype(str))
    y = np.searchsorted(classes, y_raw.astype(str))

    # Degenerate case: only one class in the training data.
    if len(classes) == 1:
        print("CV accuracy: 1.000000")
        submission = pd.DataFrame({"id": np.arange(len(test)), TARGET: classes[0]})
        submission.to_csv("./working/submission.csv", index=False)
        return

    # Degenerate case: no usable features.
    if X_train.shape[1] == 0:
        majority_class_idx = int(np.argmax(np.bincount(y)))
        majority_class = classes[majority_class_idx]

        if len(y) > 1:
            try:
                _, val_idx = train_test_split(
                    np.arange(len(y)),
                    test_size=0.2,
                    random_state=RANDOM_STATE,
                    stratify=y,
                )
            except ValueError:
                _, val_idx = train_test_split(
                    np.arange(len(y)), test_size=0.2, random_state=RANDOM_STATE
                )
            val_acc = accuracy_score(
                classes[y[val_idx]], np.full(len(val_idx), majority_class)
            )
        else:
            val_acc = 1.0

        print(f"CV accuracy: {val_acc:.6f}")

        submission = pd.DataFrame({"id": np.arange(len(test)), TARGET: majority_class})
        submission.to_csv("./working/submission.csv", index=False)
        return

    # Choose an appropriate number of folds. Prefer 5-fold, but reduce if a class is rare.
    class_counts = np.bincount(y)
    min_class_count = int(class_counts.min())

    if len(y) >= 10 and min_class_count >= 2:
        n_splits = min(5, min_class_count)
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
        splits = list(cv.split(X_train, y))
    else:
        try:
            train_idx, val_idx = train_test_split(
                np.arange(len(y)), test_size=0.2, random_state=RANDOM_STATE, stratify=y
            )
        except ValueError:
            train_idx, val_idx = train_test_split(
                np.arange(len(y)), test_size=0.2, random_state=RANDOM_STATE
            )
        splits = [(train_idx, val_idx)]

    params = {
        "objective": "multiclass",
        "num_class": len(classes),
        "metric": "multi_logloss",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 20,
        "subsample": 0.9,
        "subsample_freq": 1,
        "colsample_bytree": 0.9,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "verbosity": -1,
    }

    accuracies = []
    best_iterations = []

    for fold_train_idx, fold_val_idx in splits:
        X_fold_train = X_train.iloc[fold_train_idx]
        X_fold_val = X_train.iloc[fold_val_idx]
        y_fold_train = y[fold_train_idx]
        y_fold_val = y[fold_val_idx]

        train_set = lgb.Dataset(
            X_fold_train,
            label=y_fold_train,
            categorical_feature=categorical_cols if categorical_cols else None,
        )
        val_set = lgb.Dataset(
            X_fold_val,
            label=y_fold_val,
            categorical_feature=categorical_cols if categorical_cols else None,
            reference=train_set,
        )

        model = lgb.train(
            params,
            train_set,
            num_boost_round=NUM_BOOST_ROUND,
            valid_sets=[val_set],
            valid_names=["val"],
            callbacks=[
                lgb.early_stopping(EARLY_STOPPING_ROUNDS),
                lgb.log_evaluation(-1),
            ],
        )

        best_iteration = model.best_iteration
        if best_iteration is None or best_iteration <= 0:
            best_iteration = NUM_BOOST_ROUND
        best_iterations.append(best_iteration)

        val_proba = model.predict(X_fold_val, num_iteration=best_iteration)
        val_pred = classes[np.argmax(val_proba, axis=1)]
        val_true = classes[y_fold_val]

        accuracies.append(accuracy_score(val_true, val_pred))

    cv_accuracy = float(np.mean(accuracies))
    print(f"CV accuracy: {cv_accuracy:.6f}")

    # Train final model on all available training data.
    final_boost_rounds = int(np.median(best_iterations)) + EARLY_STOPPING_ROUNDS
    final_boost_rounds = max(1, min(final_boost_rounds, NUM_BOOST_ROUND))

    final_train_set = lgb.Dataset(
        X_train,
        label=y,
        categorical_feature=categorical_cols if categorical_cols else None,
    )

    final_model = lgb.train(
        params,
        final_train_set,
        num_boost_round=final_boost_rounds,
        callbacks=[lgb.log_evaluation(-1)],
    )

    test_proba = final_model.predict(X_test, num_iteration=final_boost_rounds)
    test_pred = classes[np.argmax(test_proba, axis=1)]

    submission = pd.DataFrame({"id": np.arange(len(test)), TARGET: test_pred})
    submission.to_csv("./working/submission.csv", index=False)


if __name__ == "__main__":
    main()
