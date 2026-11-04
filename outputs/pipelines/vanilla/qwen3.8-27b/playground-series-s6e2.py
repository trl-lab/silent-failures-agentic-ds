import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/kaggle/playground-series-s6e2/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/kaggle/playground-series-s6e2/data/test.parquet"
TARGET = "Heart Disease"
ID_COL = "id"
RANDOM_STATE = 42


def positive_proba(model, X):
    proba = model.predict_proba(X)
    if proba.ndim == 1:
        return proba

    classes = getattr(model, "classes_", None)
    if classes is not None and len(classes) == proba.shape[1]:
        if 1 in classes:
            return proba[:, list(classes).index(1)]
        if 0 in classes:
            return 1.0 - proba[:, list(classes).index(0)]

    if proba.shape[1] == 2:
        return proba[:, 1]

    return proba[:, -1]


def make_models():
    return [
        LGBMClassifier(
            n_estimators=800,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=-1,
            subsample=0.9,
            subsample_freq=1,
            colsample_bytree=0.9,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        ),
        XGBClassifier(
            n_estimators=800,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            tree_method="hist",
            eval_metric="auc",
        ),
        CatBoostClassifier(
            iterations=800,
            learning_rate=0.05,
            depth=6,
            l2_leaf_reg=3.0,
            random_seed=RANDOM_STATE,
            verbose=0,
            thread_count=-1,
            allow_writing_files=False,
        ),
    ]


def save_predictions(predictions, n_rows):
    predictions = np.asarray(predictions, dtype=float)

    if predictions.size == 0:
        submission = pd.DataFrame(
            {
                "id": np.arange(0, dtype=int),
                TARGET: np.array([], dtype=float),
            }
        )
    else:
        if not np.all(np.isfinite(predictions)):
            predictions = np.nan_to_num(predictions, nan=0.5, posinf=1.0, neginf=0.0)
        predictions = np.clip(predictions, 0.0, 1.0)

        submission = pd.DataFrame(
            {
                "id": np.arange(n_rows, dtype=int),
                TARGET: predictions,
            }
        )

    submission.to_csv("predictions.csv", index=False)


def main():
    train = pd.read_parquet(TRAIN_PATH)
    test = pd.read_parquet(TEST_PATH)

    train = train.dropna(subset=[TARGET]).copy()

    if len(train) == 0:
        save_predictions(np.full(len(test), 0.5), len(test))
        return

    y_raw = train[TARGET]
    unique_targets = np.unique(y_raw)

    if len(unique_targets) == 2:
        if set(unique_targets.tolist()) != {0, 1}:
            train[TARGET] = (y_raw == unique_targets[-1]).astype(int)
        else:
            train[TARGET] = y_raw.astype(int)
    elif len(unique_targets) == 1:
        train[TARGET] = y_raw.astype(int)
    else:
        train[TARGET] = (y_raw == unique_targets[-1]).astype(int)

    y = train[TARGET].astype(int)

    feature_cols = [col for col in train.columns if col not in (ID_COL, TARGET)]

    for col in feature_cols:
        if col not in test.columns:
            test[col] = np.nan

    X_all = train[feature_cols].copy()
    X_test = test[feature_cols].copy()

    categorical_cols = []
    for col in feature_cols:
        if (
            pd.api.types.is_string_dtype(X_all[col])
            or pd.api.types.is_categorical_dtype(X_all[col])
            or pd.api.types.is_bool_dtype(X_all[col])
            or pd.api.types.is_datetime64_any_dtype(X_all[col])
        ):
            categorical_cols.append(col)

    for col in categorical_cols:
        X_all[col] = X_all[col].astype("string").fillna("missing")
        X_test[col] = X_test[col].astype("string").fillna("missing")

        categories = list(X_all[col].unique())
        mapping = {category: idx for idx, category in enumerate(categories)}

        X_all[col] = X_all[col].map(mapping).astype(float)
        X_test[col] = X_test[col].map(mapping).astype(float)

    X_all = X_all.astype(float).replace([np.inf, -np.inf], np.nan)
    X_test = X_test.astype(float).replace([np.inf, -np.inf], np.nan)

    if X_all.shape[1] == 0:
        save_predictions(np.full(len(X_test), 0.5), len(X_test))
        return

    if y.nunique() < 2:
        save_predictions(np.full(len(X_test), float(y.iloc[0]) if len(y) else 0.5), len(X_test))
        return

    min_class_count = int(y.value_counts().min())
    n_splits = min(3, min_class_count) if min_class_count >= 2 else 0

    oof_predictions = np.zeros(len(X_all), dtype=float)
    cv_test_predictions = np.zeros(len(X_test), dtype=float)

    if n_splits >= 2:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

        for fold, (train_idx, val_idx) in enumerate(skf.split(X_all, y)):
            X_train_fold = X_all.iloc[train_idx]
            X_val_fold = X_all.iloc[val_idx]
            y_train_fold = y.iloc[train_idx]
            y_val_fold = y.iloc[val_idx]

            fold_oof = np.zeros(len(X_val_fold), dtype=float)
            fold_test = np.zeros(len(X_test), dtype=float)
            n_models = 0

            for model in make_models():
                try:
                    model.fit(X_train_fold, y_train_fold)
                    fold_oof += positive_proba(model, X_val_fold)
                    fold_test += positive_proba(model, X_test)
                    n_models += 1
                except Exception:
                    pass

            if n_models > 0:
                oof_predictions[val_idx] = fold_oof / n_models
                cv_test_predictions += fold_test / n_models

        cv_test_predictions /= n_splits

    try:
        oof_auc = roc_auc_score(y, oof_predictions)
        print(f"OOF AUC: {oof_auc:.6f}")
    except Exception:
        print("OOF AUC: nan")

    final_predictions = np.zeros(len(X_test), dtype=float)
    n_models = 0

    for model in make_models():
        try:
            model.fit(X_all, y)
            final_predictions += positive_proba(model, X_test)
            n_models += 1
        except Exception:
            pass

    if n_models > 0:
        final_predictions /= n_models
    else:
        if np.any(cv_test_predictions != 0):
            final_predictions = cv_test_predictions
        else:
            final_predictions = np.full(len(X_test), 0.5)

    save_predictions(final_predictions, len(X_test))


if __name__ == "__main__":
    main()