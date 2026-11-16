import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier

TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/kaggle/prudential-life-insurance-assessment/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/kaggle/prudential-life-insurance-assessment/data/test.parquet"
TARGET = "Response"
ID_COL = "Id"
RANDOM_STATE = 42


def load_data():
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)
    return train_df, test_df


def prepare_features(train_df, test_df):
    X_train = train_df.drop(columns=[TARGET, ID_COL], errors="ignore").copy()
    X_test = test_df.drop(columns=[TARGET, ID_COL], errors="ignore").copy()

    cat_cols = []
    for col in X_train.columns:
        if X_train[col].dtype == object or str(X_train[col].dtype) == "category":
            cat_cols.append(col)
    for col in X_test.columns:
        if col not in cat_cols and (X_test[col].dtype == object or str(X_test[col].dtype) == "category"):
            cat_cols.append(col)

    for col in cat_cols:
        if col in X_train.columns:
            X_train[col] = X_train[col].astype("object").fillna("missing")
        if col in X_test.columns:
            X_test[col] = X_test[col].astype("object").fillna("missing")

    train_cats = [col for col in cat_cols if col in X_train.columns]
    test_cats = [col for col in cat_cols if col in X_test.columns]

    if train_cats:
        X_train = pd.get_dummies(X_train, columns=train_cats, dummy_na=False, dtype=float)
    if test_cats:
        X_test = pd.get_dummies(X_test, columns=test_cats, dummy_na=False, dtype=float)

    X_train, X_test = X_train.align(X_test, join="outer", axis=1)

    for col in X_train.columns:
        if not pd.api.types.is_numeric_dtype(X_train[col]):
            X_train[col] = pd.to_numeric(X_train[col], errors="coerce")
            X_test[col] = pd.to_numeric(X_test[col], errors="coerce")

    X_train = X_train.replace([np.inf, -np.inf], np.nan)
    X_test = X_test.replace([np.inf, -np.inf], np.nan)

    for col in X_train.columns:
        if X_train[col].isna().any() or X_test[col].isna().any():
            fill_value = X_train[col].median()
            if pd.isna(fill_value):
                fill_value = 0
            X_train[col] = X_train[col].fillna(fill_value)
            X_test[col] = X_test[col].fillna(fill_value)

    constant_cols = [col for col in X_train.columns if X_train[col].nunique(dropna=True) <= 1]
    if constant_cols:
        X_train = X_train.drop(columns=constant_cols)
        X_test = X_test.drop(columns=constant_cols)

    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)

    return X_train, X_test


def make_weight_matrix(n_classes):
    idx = np.arange(n_classes)
    diff = np.abs(idx[:, None] - idx[None, :])
    return (diff / max(n_classes - 1, 1)) ** 2


def fast_qwk(y_true_idx, y_pred_idx, n_classes, weight_matrix):
    y_true_idx = np.clip(np.asarray(y_true_idx, dtype=np.int64), 0, n_classes - 1)
    y_pred_idx = np.clip(np.asarray(y_pred_idx, dtype=np.int64), 0, n_classes - 1)

    flat = y_true_idx * n_classes + y_pred_idx
    contingency = np.bincount(flat, minlength=n_classes * n_classes).reshape(n_classes, n_classes)

    row = contingency.sum(axis=1)[:, None]
    col = contingency.sum(axis=0)[None, :]
    expected = row * col / max(contingency.sum(), 1)

    observed = np.sum(weight_matrix * contingency)
    expected_score = np.sum(weight_matrix * expected)

    if expected_score == 0:
        return 1.0
    return 1.0 - observed / expected_score


def cumulative_probabilities(proba):
    n_samples, n_classes = proba.shape
    cum = np.empty((n_samples, n_classes - 1), dtype=np.float64)
    for i in range(1, n_classes):
        cum[:, i - 1] = proba[:, i:].sum(axis=1)
    return cum


def predict_from_cumulative(cum, thresholds):
    return (cum > thresholds).sum(axis=1)


def initial_thresholds(cum, y_true_idx, n_classes):
    thresholds = np.full(n_classes - 1, 0.5, dtype=np.float64)
    for i in range(n_classes - 1):
        true_frac = float(np.mean(y_true_idx >= i + 1))
        if true_frac <= 0.0:
            thresholds[i] = 1.0
        elif true_frac >= 1.0:
            thresholds[i] = 0.0
        else:
            thresholds[i] = float(np.quantile(cum[:, i], 1.0 - true_frac))
    return thresholds


def tune_thresholds(cum, y_true_idx, n_classes, weight_matrix, max_iter=4, n_candidates=21):
    starts = [
        np.full(n_classes - 1, 0.5, dtype=np.float64),
        initial_thresholds(cum, y_true_idx, n_classes),
    ]

    for q in (0.25, 0.75):
        start = np.empty(n_classes - 1, dtype=np.float64)
        for i in range(n_classes - 1):
            start[i] = float(np.quantile(cum[:, i], q))
        starts.append(start)

    best_thresholds = None
    best_score = -np.inf

    for start in starts:
        thresholds = np.clip(start.copy(), 0.0, 1.0)
        current_score = fast_qwk(
            y_true_idx,
            predict_from_cumulative(cum, thresholds),
            n_classes,
            weight_matrix,
        )

        for _ in range(max_iter):
            improved = False
            for i in range(n_classes - 1):
                old_threshold = thresholds[i]
                candidates = np.unique(
                    np.concatenate(
                        [
                            np.quantile(cum[:, i], np.linspace(0.0, 1.0, n_candidates)),
                            [old_threshold, 0.0, 1.0],
                        ]
                    )
                )

                best_local_threshold = old_threshold
                best_local_score = current_score

                for candidate in candidates:
                    thresholds[i] = float(candidate)
                    score = fast_qwk(
                        y_true_idx,
                        predict_from_cumulative(cum, thresholds),
                        n_classes,
                        weight_matrix,
                    )
                    if score > best_local_score:
                        best_local_score = score
                        best_local_threshold = float(candidate)

                thresholds[i] = best_local_threshold
                if best_local_threshold != old_threshold:
                    improved = True
                    current_score = best_local_score

            if not improved:
                break

        if current_score > best_score:
            best_score = current_score
            best_thresholds = thresholds.copy()

    if best_thresholds is None:
        best_thresholds = np.full(n_classes - 1, 0.5, dtype=np.float64)

    return best_thresholds, best_score


def reorder_proba(model, X, classes):
    proba = model.predict_proba(X)
    model_classes = np.asarray(model.classes_)

    if len(model_classes) == len(classes):
        idx = []
        for c in classes:
            matches = np.where(model_classes == c)[0]
            if len(matches) == 0:
                idx.append(0)
            else:
                idx.append(int(matches[0]))
        return proba[:, idx]

    out = np.zeros((proba.shape[0], len(classes)), dtype=np.float64)
    class_to_pos = {int(c): i for i, c in enumerate(classes)}
    for j, c in enumerate(model_classes):
        if int(c) in class_to_pos:
            out[:, class_to_pos[int(c)]] = proba[:, j]
    return out


def train_model(X, y):
    model = LGBMClassifier(
        n_estimators=800,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=8,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        min_child_samples=50,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(X, y)
    return model


def save_predictions(predictions):
    pd.DataFrame(
        {
            "id": np.arange(len(predictions), dtype=int),
            "Response": predictions.astype(int),
        }
    ).to_csv("predictions.csv", index=False)


def main():
    train_df, test_df = load_data()
    X_full, X_test = prepare_features(train_df, test_df)
    y_full = train_df[TARGET].astype(int)

    classes = np.sort(y_full.unique())
    n_classes = len(classes)

    if n_classes == 0:
        save_predictions(np.zeros(len(X_test), dtype=int))
        return

    if n_classes == 1 or X_full.shape[1] == 0:
        save_predictions(np.full(len(X_test), int(classes[0]), dtype=int))
        return

    class_to_idx = {int(c): i for i, c in enumerate(classes)}

    try:
        X_train, X_val, y_train, y_val = train_test_split(
            X_full,
            y_full,
            test_size=0.2,
            stratify=y_full,
            random_state=RANDOM_STATE,
        )
    except ValueError:
        X_train, X_val, y_train, y_val = train_test_split(
            X_full,
            y_full,
            test_size=0.2,
            random_state=RANDOM_STATE,
        )

    y_val_idx = np.array([class_to_idx[int(v)] for v in y_val], dtype=np.int64)
    weight_matrix = make_weight_matrix(n_classes)

    val_model = train_model(X_train, y_train)
    val_proba = reorder_proba(val_model, X_val, classes)
    val_cum = cumulative_probabilities(val_proba)

    try:
        thresholds, _ = tune_thresholds(val_cum, y_val_idx, n_classes, weight_matrix)
    except Exception:
        thresholds = np.full(n_classes - 1, 0.5, dtype=np.float64)

    final_model = train_model(X_full, y_full)
    test_proba = reorder_proba(final_model, X_test, classes)
    test_cum = cumulative_probabilities(test_proba)

    pred_idx = predict_from_cumulative(test_cum, thresholds)
    pred_idx = np.clip(pred_idx, 0, n_classes - 1)
    predictions = classes[pred_idx]

    save_predictions(predictions)


if __name__ == "__main__":
    main()