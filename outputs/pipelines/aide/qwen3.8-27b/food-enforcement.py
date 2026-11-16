import os
import warnings
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

WORKING_DIR = "./working"
os.makedirs(WORKING_DIR, exist_ok=True)

TARGET = "classification"
ID_COLS = ["id", "Id", "ID", "index"]


def is_probably_date_column(col: str, series: pd.Series) -> bool:
    if series.dropna().empty:
        return False

    sample = series.dropna().astype(str).head(200)
    name = col.lower()
    date_name = any(token in name for token in ["date", "time", "year", "month", "day"])
    date_regex = r"\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{4}"
    regex_match = sample.str.contains(date_regex, regex=True, na=False).mean() > 0.5

    if not (date_name or regex_match):
        return False

    parsed = pd.to_datetime(sample, errors="coerce")
    return parsed.notna().mean() > 0.7


def is_probably_text_column(series: pd.Series) -> bool:
    if series.dropna().empty:
        return False

    s = series.dropna().astype(str)
    median_len = float(s.str.len().median())
    n_unique = int(s.nunique())
    return median_len > 50 or (n_unique > 5000 and median_len > 10)


def identify_feature_types(
    raw_train: pd.DataFrame,
) -> Tuple[List[str], List[str], List[str]]:
    date_cols: List[str] = []
    text_cols: List[str] = []
    cat_cols: List[str] = []

    for col in raw_train.columns:
        if col == TARGET or col in ID_COLS:
            continue

        series = raw_train[col]

        if pd.api.types.is_numeric_dtype(series):
            continue

        if pd.api.types.is_datetime64_any_dtype(series):
            date_cols.append(col)
            continue

        if pd.api.types.is_object_dtype(series) or pd.api.types.is_categorical_dtype(
            series
        ):
            if is_probably_date_column(col, series):
                date_cols.append(col)
            elif is_probably_text_column(series):
                text_cols.append(col)
            else:
                cat_cols.append(col)
        else:
            cat_cols.append(col)

    date_cols = list(dict.fromkeys(date_cols))
    text_cols = [c for c in dict.fromkeys(text_cols) if c not in date_cols]
    cat_cols = [
        c for c in dict.fromkeys(cat_cols) if c not in date_cols and c not in text_cols
    ]

    return date_cols, text_cols, cat_cols


def preprocess_features(
    df: pd.DataFrame,
    date_cols: List[str],
    text_cols: List[str],
    cat_cols: List[str],
) -> pd.DataFrame:
    df = df.copy()
    df = df.drop(
        columns=[c for c in ID_COLS + [TARGET] if c in df.columns],
        errors="ignore",
    )

    for col in date_cols:
        if col not in df.columns:
            continue

        dt = pd.to_datetime(df[col], errors="coerce")
        df[f"{col}_year"] = dt.dt.year
        df[f"{col}_month"] = dt.dt.month
        df[f"{col}_day"] = dt.dt.day
        df[f"{col}_weekday"] = dt.dt.dayofweek
        df = df.drop(columns=[col])

    for col in text_cols:
        if col not in df.columns:
            continue

        s = df[col].fillna("").astype(str)
        df[f"{col}_len"] = s.str.len()
        df[f"{col}_words"] = s.str.split().str.len().fillna(0)
        df[f"{col}_digits"] = s.str.count(r"\d")
        df[f"{col}_upper"] = s.str.count(r"[A-Z]")
        df[f"{col}_special"] = s.str.count(r"[^a-zA-Z0-9]")
        df = df.drop(columns=[col])

    for col in df.columns:
        if col not in cat_cols and not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in cat_cols:
        if col not in df.columns:
            df[col] = np.nan

    return df


def encode_categoricals(
    train_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame],
    cat_columns: List[str],
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    train_df = train_df.copy()
    val_df = val_df.copy() if val_df is not None else None

    for col in cat_columns:
        if col not in train_df.columns:
            continue

        train_df[col] = train_df[col].fillna("missing").astype(str)
        cats = list(train_df[col].unique())
        train_df[col] = pd.Categorical(train_df[col], categories=cats)

        if val_df is not None and col in val_df.columns:
            val_df[col] = val_df[col].fillna("missing").astype(str)
            val_df[col] = val_df[col].where(val_df[col].isin(cats), "missing")
            val_df[col] = pd.Categorical(val_df[col], categories=cats)

    return train_df, val_df


def save_submission(n_test: int, predictions: np.ndarray) -> None:
    if len(predictions) != n_test:
        if len(predictions) < n_test:
            predictions = np.concatenate(
                [predictions, np.full(n_test - len(predictions), "", dtype=object)]
            )
        else:
            predictions = predictions[:n_test]

    submission = pd.DataFrame(
        {
            "id": np.arange(n_test, dtype=int),
            "classification": predictions,
        }
    )
    submission.to_csv(os.path.join(WORKING_DIR, "submission.csv"), index=False)
    print(f"Saved submission to {os.path.join(WORKING_DIR, 'submission.csv')}")


def main() -> None:
    train = pd.read_parquet("./input/train.parquet")
    test = pd.read_parquet("./input/test.parquet")
    n_test = len(test)

    train = train.drop(
        columns=[c for c in ID_COLS if c in train.columns], errors="ignore"
    )
    test = test.drop(
        columns=[c for c in ID_COLS + [TARGET] if c in test.columns],
        errors="ignore",
    )

    if TARGET not in train.columns:
        save_submission(n_test, np.array([""] * n_test, dtype=object))
        print("CV Accuracy: 0.00000")
        return

    train = train.dropna(subset=[TARGET]).reset_index(drop=True)

    if train.empty:
        save_submission(n_test, np.array([""] * n_test, dtype=object))
        print("CV Accuracy: 0.00000")
        return

    y_raw = train[TARGET]
    X_train_raw = train.drop(columns=[TARGET])
    X_test_raw = test.copy()

    date_cols, text_cols, cat_cols = identify_feature_types(X_train_raw)

    X_train_pre = preprocess_features(X_train_raw, date_cols, text_cols, cat_cols)
    X_test_pre = preprocess_features(X_test_raw, date_cols, text_cols, cat_cols)

    # Ensure train/test have exactly the same feature columns before modeling.
    X_test_pre = X_test_pre.reindex(columns=X_train_pre.columns)

    keep_mask = X_train_pre.nunique(dropna=True) > 1
    X_train_pre = X_train_pre.loc[:, keep_mask]
    X_test_pre = X_test_pre.loc[:, keep_mask]

    cat_cols = [c for c in cat_cols if c in X_train_pre.columns]

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    n_classes = len(label_encoder.classes_)

    if n_classes == 1:
        only_label = label_encoder.classes_[0]
        save_submission(n_test, np.full(n_test, only_label, dtype=object))
        print("CV Accuracy: 1.00000")
        return

    if X_train_pre.shape[1] == 0:
        if len(y) >= 2:
            tr_idx, va_idx = train_test_split(
                np.arange(len(y)),
                test_size=0.2,
                random_state=42,
            )
            majority_enc = int(np.bincount(y[tr_idx]).argmax())
            val_pred = np.full(len(va_idx), majority_enc, dtype=int)
            acc = accuracy_score(y[va_idx], val_pred)
        else:
            majority_enc = 0
            acc = 0.0

        full_majority_enc = int(np.bincount(y).argmax())
        test_pred = label_encoder.inverse_transform(
            np.full(n_test, full_majority_enc, dtype=int)
        )
        save_submission(n_test, test_pred)
        print(f"CV Accuracy: {acc:.5f}")
        return

    if n_classes == 2:
        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": -1,
            "min_child_samples": 20,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "lambda_l1": 0.1,
            "lambda_l2": 0.1,
            "seed": 42,
            "verbose": -1,
        }
    else:
        params = {
            "objective": "multiclass",
            "num_class": n_classes,
            "metric": "multi_logloss",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": -1,
            "min_child_samples": 20,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "lambda_l1": 0.1,
            "lambda_l2": 0.1,
            "seed": 42,
            "verbose": -1,
        }

    min_class_count = int(pd.Series(y).value_counts().min())

    if min_class_count < 2:
        tr_idx, va_idx = train_test_split(
            np.arange(len(y)),
            test_size=0.2,
            random_state=42,
        )
        folds = [(tr_idx, va_idx)]
    else:
        n_splits = min(5, min_class_count)
        if n_splits == 1:
            tr_idx, va_idx = train_test_split(
                np.arange(len(y)),
                test_size=0.2,
                random_state=42,
                stratify=y,
            )
            folds = [(tr_idx, va_idx)]
        else:
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            folds = list(skf.split(X_train_pre, y))

    scores = []
    best_iterations = []

    for fold_idx, (tr_idx, va_idx) in enumerate(folds, start=1):
        Xtr_raw = X_train_pre.iloc[tr_idx]
        Xva_raw = X_train_pre.iloc[va_idx]
        ytr = y[tr_idx]
        yva = y[va_idx]

        Xtr, Xva = encode_categoricals(Xtr_raw, Xva_raw, cat_cols)

        train_dataset = lgb.Dataset(
            Xtr,
            label=ytr,
            categorical_feature=cat_cols,
            free_raw_data=False,
        )
        valid_dataset = lgb.Dataset(
            Xva,
            label=yva,
            reference=train_dataset,
            free_raw_data=False,
        )

        model = lgb.train(
            params,
            train_dataset,
            num_boost_round=2000,
            valid_sets=[valid_dataset],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
        )

        best_iterations.append(max(1, int(model.best_iteration)))

        if n_classes == 2:
            proba = model.predict(Xva, num_iteration=model.best_iteration)
            pred = (proba > 0.5).astype(int)
        else:
            proba = model.predict(Xva, num_iteration=model.best_iteration)
            pred = np.argmax(proba, axis=1)

        acc = accuracy_score(yva, pred)
        scores.append(acc)
        print(f"Fold {fold_idx} accuracy: {acc:.5f}")

    cv_accuracy = float(np.mean(scores))
    cv_std = float(np.std(scores))
    print(f"CV Accuracy: {cv_accuracy:.5f} (+/- {cv_std:.5f})")

    X_train_full, _ = encode_categoricals(X_train_pre, None, cat_cols)
    full_dataset = lgb.Dataset(
        X_train_full,
        label=y,
        categorical_feature=cat_cols,
        free_raw_data=False,
    )

    final_num_boost_round = int(np.clip(np.mean(best_iterations) + 50, 100, 3000))
    final_model = lgb.train(
        params,
        full_dataset,
        num_boost_round=final_num_boost_round,
        callbacks=[lgb.log_evaluation(-1)],
    )

    # Bugfix: the encoded test frame is the second return value.
    _, X_test_final = encode_categoricals(X_train_pre, X_test_pre, cat_cols)

    if n_classes == 2:
        test_proba = final_model.predict(
            X_test_final, num_iteration=final_num_boost_round
        )
        test_pred_enc = (test_proba > 0.5).astype(int)
    else:
        test_proba = final_model.predict(
            X_test_final, num_iteration=final_num_boost_round
        )
        test_pred_enc = np.argmax(test_proba, axis=1)

    test_pred = label_encoder.inverse_transform(test_pred_enc)
    save_submission(n_test, test_pred)


if __name__ == "__main__":
    main()
