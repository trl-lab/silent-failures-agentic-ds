import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import log_loss
from lightgbm import LGBMClassifier

TRAIN_PATH = '/path/to/user/workspace/ds_failures/datasets/kaggle/otto-group-product-classification-challenge/data/train.parquet'
TEST_PATH = '/path/to/user/workspace/ds_failures/datasets/kaggle/otto-group-product-classification-challenge/data/test.parquet'
TARGET = 'target'
ID = 'id'
RANDOM_STATE = 42


def make_onehot():
    try:
        return OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown='ignore', sparse=False)


def label_to_float(label):
    label = str(label)
    if label.startswith('Class_'):
        try:
            return float(label.split('_', 1)[1])
        except Exception:
            pass
    try:
        return float(label)
    except Exception:
        return 0.0


def prepare(df, feature_cols, numeric_cols, categorical_cols):
    out = df[feature_cols].copy()
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors='coerce').astype('float64')
    for col in categorical_cols:
        out[col] = out[col].astype('object').fillna('missing')
    return out


def build_preprocessor(numeric_cols, categorical_cols):
    transformers = []
    if numeric_cols:
        transformers.append(
            (
                'num',
                Pipeline([('imputer', SimpleImputer(strategy='median'))]),
                numeric_cols,
            )
        )
    if categorical_cols:
        transformers.append(
            (
                'cat',
                Pipeline(
                    [
                        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                        ('onehot', make_onehot()),
                    ]
                ),
                categorical_cols,
            )
        )
    return ColumnTransformer(transformers, remainder='drop')


def to_dense(X):
    if hasattr(X, 'toarray'):
        X = X.toarray()
    X = np.asarray(X, dtype='float64')
    return np.where(np.isfinite(X), X, 0.0)


def align_proba(proba, model_classes, target_classes):
    target_classes = np.asarray(target_classes)
    proba = np.asarray(proba, dtype='float64')
    out = np.zeros((proba.shape[0], len(target_classes)), dtype='float64')

    target_index = {c: i for i, c in enumerate(target_classes)}
    for j, c in enumerate(model_classes):
        if c in target_index:
            out[:, target_index[c]] = proba[:, j]

    row_sum = out.sum(axis=1, keepdims=True)
    empty = row_sum[:, 0] == 0
    if np.any(empty):
        out[empty] = 1.0 / len(target_classes)
        row_sum = out.sum(axis=1, keepdims=True)

    out = np.clip(out, 1e-15, 1.0)
    out /= out.sum(axis=1, keepdims=True)
    return out


def main():
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    feature_cols = [c for c in train_df.columns if c not in (ID, TARGET)]
    numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(train_df[c])]
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]

    y_str = train_df[TARGET].astype(str)
    classes = np.sort(y_str.unique())
    y = pd.Categorical(y_str, categories=classes).codes.astype(int)
    target_classes = np.arange(len(classes), dtype=int)

    X = train_df[feature_cols]
    try:
        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=RANDOM_STATE,
            stratify=y,
        )
    except ValueError:
        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=RANDOM_STATE,
        )

    X_train = prepare(X_train, feature_cols, numeric_cols, categorical_cols)
    X_val = prepare(X_val, feature_cols, numeric_cols, categorical_cols)

    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    X_train = to_dense(preprocessor.fit_transform(X_train))
    X_val = to_dense(preprocessor.transform(X_val))

    val_model = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.9,
        min_child_samples=50,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )
    val_model.fit(X_train, y_train)
    val_proba = align_proba(val_model.predict_proba(X_val), val_model.classes_, target_classes)
    _ = log_loss(y_val, val_proba, labels=target_classes)

    X_full = prepare(train_df, feature_cols, numeric_cols, categorical_cols)
    X_test = prepare(test_df, feature_cols, numeric_cols, categorical_cols)

    final_preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    X_full = to_dense(final_preprocessor.fit_transform(X_full))
    X_test = to_dense(final_preprocessor.transform(X_test))

    final_model = LGBMClassifier(
        n_estimators=800,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.9,
        min_child_samples=50,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )
    final_model.fit(X_full, y)

    test_proba = align_proba(final_model.predict_proba(X_test), final_model.classes_, target_classes)
    pred_idx = np.argmax(test_proba, axis=1)

    class_numbers = np.array([label_to_float(c) for c in classes], dtype='float64')
    pred_numeric = class_numbers[pred_idx]

    pd.DataFrame(
        {
            'id': np.arange(len(test_df), dtype=int),
            'target': pred_numeric.astype('float64'),
        }
    ).to_csv('predictions.csv', index=False)


if __name__ == '__main__':
    main()