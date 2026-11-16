import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier

TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/strable/medicines/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/strable/medicines/data/test.parquet"
TARGET = "Medicine status"
POSITIVE_LABEL = "Authorised"


def encode_target(series: pd.Series) -> pd.Series:
    s = series.fillna("").astype(str).str.strip().str.lower()
    return (s == POSITIVE_LABEL.lower()).astype(int)


def is_boolean_column(series: pd.Series) -> bool:
    vals = set(series.dropna().astype(str).str.strip().str.lower().unique())
    return len(vals) > 0 and vals.issubset({"yes", "no"})


def make_boolean(series: pd.Series) -> pd.Series:
    return series.fillna("no").astype(str).str.strip().str.lower().eq("yes").astype(int)


def clean_text(series: pd.Series) -> pd.Series:
    s = series.fillna("missing").astype(str).str.strip()
    s = s.replace(
        {
            "": "missing",
            "nan": "missing",
            "NaN": "missing",
            "None": "missing",
            "none": "missing",
        }
    )
    return s


class FrequencyEncoder:
    def __init__(self):
        self.stats_ = {}

    def fit(self, X: pd.DataFrame, y=None):
        X = X.astype(str)
        self.stats_ = {col: X[col].value_counts(normalize=True) for col in X.columns}
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.astype(str)
        out = pd.DataFrame(index=X.index)
        for col in X.columns:
            out[col] = X[col].map(self.stats_[col]).fillna(0.0).astype(float)
        return out


def build_feature_matrix(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame | None = None,
    low_cardinality_threshold: int = 50,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    X_train = X_train.copy()
    X_test = X_test.copy() if X_test is not None else None

    if X_train.shape[1] == 0:
        X_train = pd.DataFrame(index=X_train.index)
        if X_test is not None:
            X_test = pd.DataFrame(index=X_test.index)
        X_train["feature_0"] = 0.0
        if X_test is not None:
            X_test["feature_0"] = 0.0
        return X_train, X_test

    bool_cols = [c for c in X_train.columns if is_boolean_column(X_train[c])]
    other_cols = [c for c in X_train.columns if c not in bool_cols]

    for col in bool_cols:
        X_train[col] = make_boolean(X_train[col])
        if X_test is not None:
            X_test[col] = make_boolean(X_test[col])

    for col in other_cols:
        X_train[col] = clean_text(X_train[col])
        if X_test is not None:
            X_test[col] = clean_text(X_test[col])

    low_cols = []
    high_cols = []
    drop_cols = []

    n_rows = max(len(X_train), 1)
    near_unique_threshold = max(100, int(0.95 * n_rows))

    for col in other_cols:
        n_unique = X_train[col].nunique(dropna=True)
        if n_unique <= 1 or n_unique >= near_unique_threshold:
            drop_cols.append(col)
        elif n_unique <= low_cardinality_threshold:
            low_cols.append(col)
        else:
            high_cols.append(col)

    X_train = X_train.drop(columns=drop_cols, errors="ignore")
    if X_test is not None:
        X_test = X_test.drop(columns=drop_cols, errors="ignore")

    parts_train = []
    parts_test = []

    if bool_cols:
        parts_train.append(X_train[bool_cols].astype(float))
        if X_test is not None:
            parts_test.append(X_test[bool_cols].astype(float))

    if low_cols:
        try:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

        ohe.fit(X_train[low_cols])

        try:
            feature_names = ohe.get_feature_names_out(low_cols)
        except AttributeError:
            feature_names = ohe.get_feature_names(low_cols)

        low_train = ohe.transform(X_train[low_cols])
        low_train = pd.DataFrame(
            low_train,
            columns=feature_names,
            index=X_train.index,
        )
        parts_train.append(low_train.astype(float))

        if X_test is not None:
            low_test = ohe.transform(X_test[low_cols])
            low_test = pd.DataFrame(
                low_test,
                columns=feature_names,
                index=X_test.index,
            )
            parts_test.append(low_test.astype(float))

    if high_cols:
        freq = FrequencyEncoder().fit(X_train[high_cols])
        parts_train.append(freq.transform(X_train[high_cols]))
        if X_test is not None:
            parts_test.append(freq.transform(X_test[high_cols]))

    if parts_train:
        X_out_train = pd.concat(parts_train, axis=1)
    else:
        X_out_train = pd.DataFrame(index=X_train.index)

    if X_test is not None:
        if parts_test:
            X_out_test = pd.concat(parts_test, axis=1)
        else:
            X_out_test = pd.DataFrame(index=X_test.index)
    else:
        X_out_test = None

    if X_out_train.shape[1] == 0:
        X_out_train = pd.DataFrame({"feature_0": 0.0}, index=X_out_train.index)

    if X_out_test is not None:
        if X_out_test.shape[1] == 0:
            X_out_test = pd.DataFrame(index=X_out_test.index)
        X_out_test = X_out_test.reindex(columns=X_out_train.columns, fill_value=0.0)
        if X_out_test.shape[1] == 0:
            X_out_test = pd.DataFrame({"feature_0": 0.0}, index=X_out_test.index)

    n_features = X_out_train.shape[1]
    X_out_train = X_out_train.copy()
    X_out_train.columns = [f"feature_{i}" for i in range(n_features)]

    if X_out_test is not None:
        X_out_test = X_out_test.copy()
        X_out_test.columns = [f"feature_{i}" for i in range(n_features)]

    X_out_train = X_out_train.replace([np.inf, -np.inf], 0.0).fillna(0.0).astype(float)
    if X_out_test is not None:
        X_out_test = X_out_test.replace([np.inf, -np.inf], 0.0).fillna(0.0).astype(float)

    return X_out_train, X_out_test


def make_model(n_estimators: int = 1000) -> LGBMClassifier:
    return LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=20,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )


def save_predictions(test_df: pd.DataFrame, predictions: np.ndarray) -> None:
    predictions = np.asarray(predictions, dtype=float)
    if predictions.ndim == 2:
        predictions = predictions[:, 1]

    if len(predictions) != len(test_df):
        predictions = np.resize(predictions, len(test_df))

    submission = pd.DataFrame(
        {
            "id": np.arange(len(test_df), dtype=int),
            "Medicine status": predictions,
        }
    )
    submission.to_csv("predictions.csv", index=False)


def main() -> None:
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    if TARGET in test_df.columns:
        test_df = test_df.drop(columns=[TARGET])

    if TARGET not in train_df.columns:
        save_predictions(test_df, np.full(len(test_df), 0.5))
        return

    train_df = train_df.dropna(subset=[TARGET])

    if train_df.empty:
        save_predictions(test_df, np.full(len(test_df), 0.5))
        return

    y_full = encode_target(train_df[TARGET])
    X_full = train_df.drop(columns=[TARGET])

    common_cols = [c for c in X_full.columns if c in test_df.columns]
    X_full = X_full[common_cols]
    test_df = test_df[common_cols]

    if y_full.nunique() < 2:
        pred_value = float(y_full.iloc[0]) if len(y_full) else 0.5
        save_predictions(test_df, np.full(len(test_df), pred_value))
        return

    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X_full,
        y_full,
        test_size=0.2,
        random_state=42,
        stratify=y_full,
    )

    X_train_proc, X_val_proc = build_feature_matrix(X_train_raw, X_val_raw)

    val_model = make_model(n_estimators=1000)
    if y_val.nunique() > 1:
        val_model.fit(
            X_train_proc,
            y_train,
            eval_set=[(X_val_proc, y_val)],
            eval_metric="auc",
        )
        val_proba = val_model.predict_proba(X_val_proc)[:, 1]
        try:
            val_auc = roc_auc_score(y_val, val_proba)
            print(f"Validation ROC AUC: {val_auc:.6f}")
        except Exception:
            pass
    else:
        val_model.fit(X_train_proc, y_train)

    X_train_final, X_test_proc = build_feature_matrix(X_full, test_df)

    final_model = make_model(n_estimators=1000)
    final_model.fit(X_train_final, y_full)

    test_proba = final_model.predict_proba(X_test_proc)[:, 1]
    save_predictions(test_df, test_proba)


if __name__ == "__main__":
    main()