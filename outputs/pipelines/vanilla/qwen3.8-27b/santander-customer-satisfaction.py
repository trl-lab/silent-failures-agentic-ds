import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
import lightgbm as lgb

TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/kaggle/santander-customer-satisfaction/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/kaggle/santander-customer-satisfaction/data/test.parquet"
TARGET_COL = "TARGET"
ID_COL = "ID"
RANDOM_STATE = 42


def _is_categorical(series: pd.Series) -> bool:
    return series.dtype == object or series.dtype.name == "category"


def _encode_categorical(train_series: pd.Series, test_series: pd.Series):
    train_series = train_series.astype("object").fillna("__MISSING__").astype(str)
    test_series = test_series.astype("object").fillna("__MISSING__").astype(str)

    categories = sorted(train_series.unique())
    train_codes = pd.Categorical(train_series, categories=categories).codes.astype(np.int32)
    test_codes = pd.Categorical(test_series, categories=categories).codes.astype(np.int32)

    return train_codes, test_codes


def _clean_numeric(train_series: pd.Series, test_series: pd.Series):
    train_series = pd.to_numeric(train_series, errors="coerce").astype(np.float32)
    test_series = pd.to_numeric(test_series, errors="coerce").astype(np.float32)

    train_series = train_series.replace([np.inf, -np.inf], np.nan)
    test_series = test_series.replace([np.inf, -np.inf], np.nan)

    median = train_series.median()
    if np.isnan(median):
        median = 0.0

    train_series = train_series.fillna(median)
    test_series = test_series.fillna(median)

    return train_series, test_series


def prepare_data(train_df: pd.DataFrame, test_df: pd.DataFrame):
    train_df = train_df.copy()
    test_df = test_df.copy()

    if TARGET_COL in train_df.columns:
        y = train_df.pop(TARGET_COL)
    else:
        y = pd.Series(0, index=train_df.index)

    train_df = train_df.drop(columns=[ID_COL], errors="ignore")
    test_df = test_df.drop(columns=[ID_COL], errors="ignore")
    test_df = test_df.drop(columns=[TARGET_COL], errors="ignore")

    common_cols = [col for col in train_df.columns if col in test_df.columns]
    train_df = train_df[common_cols]
    test_df = test_df[common_cols]

    cat_cols = [col for col in common_cols if _is_categorical(train_df[col])]
    num_cols = [col for col in common_cols if col not in cat_cols]

    for col in cat_cols:
        train_df[col], test_df[col] = _encode_categorical(train_df[col], test_df[col])

    for col in num_cols:
        train_df[col], test_df[col] = _clean_numeric(train_df[col], test_df[col])

    if train_df.shape[1] > 1:
        variances = train_df.var()
        keep_cols = variances[variances > 0].index.tolist()
        if len(keep_cols) > 0:
            train_df = train_df[keep_cols]
            test_df = test_df[keep_cols]

    train_df = train_df.astype(np.float32)
    test_df = test_df.astype(np.float32)

    return train_df, y, test_df


def make_model(n_estimators: int) -> LGBMClassifier:
    return LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=0.05,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=50,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=RANDOM_STATE,
        n_jobs=1,
        verbose=-1,
    )


def _save_predictions(n_rows: int, target_values) -> None:
    target_values = np.asarray(target_values, dtype=np.float64)

    if target_values.ndim == 0:
        target_values = np.full(n_rows, float(target_values), dtype=np.float64)
    else:
        target_values = target_values[:n_rows]
        if len(target_values) < n_rows:
            pad_value = float(target_values[-1]) if len(target_values) > 0 else 0.5
            target_values = np.concatenate(
                [target_values, np.full(n_rows - len(target_values), pad_value, dtype=np.float64)]
            )

    target_values = np.where(np.isfinite(target_values), target_values, 0.5)
    target_values = np.clip(target_values, 0.0, 1.0)

    predictions = pd.DataFrame(
        {
            "id": np.arange(n_rows, dtype=int),
            "TARGET": target_values,
        }
    )
    predictions.to_csv("predictions.csv", index=False)


def _predict_with_model(model: LGBMClassifier, X_test: pd.DataFrame) -> np.ndarray:
    proba = model.predict_proba(X_test)
    classes = model.classes_

    if 1 in classes:
        positive_idx = int(np.where(classes == 1)[0][0])
    else:
        positive_idx = -1

    test_pred = np.asarray(proba[:, positive_idx], dtype=np.float64)
    test_pred = np.where(np.isfinite(test_pred), test_pred, 0.5)
    return np.clip(test_pred, 0.0, 1.0)


def main() -> None:
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    X_full, y, X_test = prepare_data(train_df, test_df)

    valid_mask = y.notna()
    X_full = X_full.loc[valid_mask]
    y = y.loc[valid_mask]

    if len(X_full) == 0 or X_full.shape[1] == 0:
        fallback = float(y.mean()) if len(y) > 0 else 0.5
        _save_predictions(len(X_test), fallback)
        return

    if len(np.unique(y)) < 2:
        fallback = float(y.iloc[0]) if len(y) > 0 else 0.5
        _save_predictions(len(X_test), fallback)
        return

    if len(X_full) < 10:
        model = make_model(2000)
        model.fit(X_full, y)
        _save_predictions(len(X_test), _predict_with_model(model, X_test))
        return

    X_train, X_val, y_train, y_val = train_test_split(
        X_full,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model = make_model(2000)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(stopping_rounds=100)],
    )

    best_iter = int(getattr(model, "best_iteration_", 0) or 0)
    if best_iter <= 0:
        best_iter = 2000
    best_iter = min(max(best_iter, 1), 2000)

    final_model = make_model(best_iter)
    final_model.fit(X_full, y)

    _save_predictions(len(X_test), _predict_with_model(final_model, X_test))


if __name__ == "__main__":
    main()