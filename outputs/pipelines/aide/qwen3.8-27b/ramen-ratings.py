import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import accuracy_score

WORK_DIR = "./working"
TRAIN_PATH = "./input/train.parquet"
TEST_PATH = "./input/test.parquet"
TARGET = "Stars"
ID_COLS = {"id", "Id", "ID", "index", "Index"}


def get_feature_columns(train, test, target):
    train_cols = [c for c in train.columns if c != target and c not in ID_COLS]
    return [c for c in train_cols if c in test.columns]


def get_categorical_columns(train, cols):
    cat_cols = []
    for c in cols:
        if c in train.columns:
            if not pd.api.types.is_numeric_dtype(
                train[c]
            ) and not pd.api.types.is_bool_dtype(train[c]):
                cat_cols.append(c)
    return cat_cols


def get_categories(X, cat_cols):
    cats_map = {}
    for col in cat_cols:
        if col not in X.columns:
            cats = ["missing"]
        else:
            vals = X[col].astype("string").fillna("missing")
            cats = list(pd.unique(vals))
            if "missing" not in cats:
                cats.append("missing")
        cats_map[col] = cats
    return cats_map


def encode_features(X, cat_cols, cats_map):
    X = X.copy()
    for col in X.columns:
        if col in cat_cols:
            cats = cats_map.get(col, ["missing"])
            vals = X[col].astype("string").fillna("missing")
            vals = vals.where(vals.isin(cats), "missing")
            X[col] = pd.Categorical(vals, categories=cats)
        else:
            X[col] = pd.to_numeric(X[col], errors="coerce")
            X[col] = X[col].replace([np.inf, -np.inf], np.nan)
    return X


def make_labels(y):
    y = y.copy()
    if not pd.api.types.is_numeric_dtype(y):
        y_num = pd.to_numeric(y, errors="coerce")
        if y_num.notna().all():
            y = y_num

    if pd.api.types.is_numeric_dtype(y):
        if np.allclose(y, y.round()):
            y = y.round().astype(int)

    unique_labels = list(pd.unique(y))
    try:
        unique_labels = sorted(unique_labels)
    except TypeError:
        unique_labels = sorted(unique_labels, key=lambda x: str(x))

    label_to_idx = {v: i for i, v in enumerate(unique_labels)}
    y_idx = y.map(label_to_idx).astype(int)
    return y_idx, unique_labels


def save_submission(pred_labels, n_rows):
    os.makedirs(WORK_DIR, exist_ok=True)
    submission = pd.DataFrame({"id": np.arange(n_rows), "Stars": pred_labels})
    submission.to_csv(os.path.join(WORK_DIR, "submission.csv"), index=False)


def main():
    os.makedirs(WORK_DIR, exist_ok=True)

    train = pd.read_parquet(TRAIN_PATH)
    test = pd.read_parquet(TEST_PATH)

    if TARGET not in train.columns:
        raise ValueError(f"Target column {TARGET} not found in train data.")

    train = train.dropna(subset=[TARGET]).reset_index(drop=True)

    if train.empty:
        save_submission(np.array([], dtype=object), len(test))
        print("CV accuracy: 0.0000")
        return

    y_raw = train[TARGET]
    y, unique_labels = make_labels(y_raw)
    num_classes = len(unique_labels)

    common = get_feature_columns(train, test, TARGET)
    cat_cols = get_categorical_columns(train, common)

    keep = []
    for c in common:
        if c in cat_cols:
            keep.append(c)
        elif not train[c].isna().all():
            keep.append(c)
    common = keep

    X_train_raw = train[common].copy()
    X_test_raw = test[common].copy()

    if len(common) == 0 or num_classes < 2:
        if num_classes > 0:
            majority_idx = int(np.bincount(y).argmax())
            pred_labels = np.array([unique_labels[majority_idx]] * len(test))
        else:
            pred_labels = np.array([], dtype=object)

        cv_acc = 1.0 if num_classes == 1 else 0.0
        print(f"CV accuracy: {cv_acc:.4f}")
        save_submission(pred_labels, len(test))
        return

    min_class = int(y.value_counts().min())
    n_splits = min(5, max(2, min_class))

    try:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        splits = list(splitter.split(X_train_raw, y))
    except Exception:
        n_splits = min(5, max(2, len(y)))
        splits = list(
            KFold(n_splits=n_splits, shuffle=True, random_state=42).split(
                X_train_raw, y
            )
        )

    params = {
        "objective": "multiclass",
        "num_class": num_classes,
        "metric": "multi_logloss",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 5,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "verbose": -1,
        "seed": 42,
    }

    accs = []
    best_iters = []

    for tr_idx, va_idx in splits:
        X_tr_raw = X_train_raw.iloc[tr_idx]
        X_va_raw = X_train_raw.iloc[va_idx]
        y_tr = y.iloc[tr_idx]
        y_va = y.iloc[va_idx]

        cats_map = get_categories(X_tr_raw, cat_cols)
        X_tr = encode_features(X_tr_raw, cat_cols, cats_map)
        X_va = encode_features(X_va_raw, cat_cols, cats_map)

        dtrain = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat_cols)
        dvalid = lgb.Dataset(
            X_va, label=y_va, categorical_feature=cat_cols, reference=dtrain
        )

        model = lgb.train(
            params,
            dtrain,
            num_boost_round=1000,
            valid_sets=[dvalid],
            valid_names=["valid"],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
        )

        best_iter = model.best_iteration
        if best_iter is None or best_iter <= 0:
            best_iter = 1000
        best_iters.append(best_iter)

        proba = model.predict(X_va, num_iteration=best_iter)
        pred = np.argmax(proba, axis=1)
        accs.append(accuracy_score(y_va, pred))

    cv_acc = float(np.mean(accs))
    cv_std = float(np.std(accs))
    print(f"CV accuracy: {cv_acc:.4f} (+/- {cv_std:.4f})")

    if len(test) == 0:
        save_submission(np.array([], dtype=object), 0)
        return

    cats_map_all = get_categories(X_train_raw, cat_cols)
    X_train_all = encode_features(X_train_raw, cat_cols, cats_map_all)
    X_test_all = encode_features(X_test_raw, cat_cols, cats_map_all)

    final_rounds = int(np.mean(best_iters) + 50)
    final_rounds = max(100, min(final_rounds, 2000))

    dtrain_all = lgb.Dataset(X_train_all, label=y, categorical_feature=cat_cols)
    final_model = lgb.train(params, dtrain_all, num_boost_round=final_rounds)

    test_proba = final_model.predict(X_test_all)
    pred_idx = np.argmax(test_proba, axis=1)
    pred_labels = np.array(unique_labels)[pred_idx]

    save_submission(pred_labels, len(test))


if __name__ == "__main__":
    main()
