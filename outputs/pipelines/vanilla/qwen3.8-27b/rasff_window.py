import pandas as pd
import numpy as np
import inspect
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier

TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/strable/rasff_window/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/strable/rasff_window/data/test.parquet"
TARGET = "risk_decision"
RANDOM_STATE = 42
EMPTY_STRINGS = {"", "none", "nan", "null", "na"}


def parse_date_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(s):
        return s
    s = s.fillna("").astype(str)
    dt = pd.to_datetime(s, format="%d-%m-%Y %H:%M:%S", errors="coerce")
    if len(dt) > 0 and dt.isna().mean() > 0.5:
        dt = pd.to_datetime(s, errors="coerce")
    return dt


def add_date_features(train: pd.DataFrame, test: pd.DataFrame) -> None:
    if "date" not in train.columns or "date" not in test.columns:
        return

    train_dt = parse_date_series(train["date"])
    test_dt = parse_date_series(test["date"])

    for df, dt in ((train, train_dt), (test, test_dt)):
        df["date_year"] = dt.dt.year.fillna(0).astype(int)
        df["date_month"] = dt.dt.month.fillna(0).astype(int)
        df["date_day"] = dt.dt.day.fillna(0).astype(int)
        df["date_weekday"] = dt.dt.weekday.fillna(0).astype(int)
        df["date_hour"] = dt.dt.hour.fillna(0).astype(int)


def list_count(s: pd.Series) -> pd.Series:
    s = s.fillna("").astype(str)

    def count(x: str) -> int:
        x = str(x).strip()
        if x.lower() in EMPTY_STRINGS:
            return 0
        parts = [p.strip() for p in x.split(",")]
        return len([p for p in parts if p.lower() not in EMPTY_STRINGS])

    return s.apply(count).astype(int)


def list_has(s: pd.Series) -> pd.Series:
    s = s.fillna("").astype(str)
    return (~s.str.strip().str.lower().isin(EMPTY_STRINGS)).astype(int)


def add_subject_features(train: pd.DataFrame, test: pd.DataFrame, keep_raw_subject: bool) -> None:
    if "subject" not in train.columns or "subject" not in test.columns:
        return

    train_subj = train["subject"].fillna("").astype(str)
    test_subj = test["subject"].fillna("").astype(str)

    train["subject_len"] = train_subj.str.len().astype(int)
    test["subject_len"] = test_subj.str.len().astype(int)

    train["subject_word_count"] = train_subj.str.split().str.len().fillna(0).astype(int)
    test["subject_word_count"] = test_subj.str.split().str.len().fillna(0).astype(int)

    train["subject_has_digit"] = train_subj.str.contains(r"\d", regex=True).astype(int)
    test["subject_has_digit"] = test_subj.str.contains(r"\d", regex=True).astype(int)

    if not keep_raw_subject:
        train.drop(columns=["subject"], inplace=True, errors="ignore")
        test.drop(columns=["subject"], inplace=True, errors="ignore")


def add_list_features(train: pd.DataFrame, test: pd.DataFrame) -> None:
    list_cols = ["distribution", "forAttention", "forFollowUp", "operator", "origin", "hazards"]
    for col in list_cols:
        if col not in train.columns or col not in test.columns:
            continue
        train[f"{col}_count"] = list_count(train[col]).astype(int)
        test[f"{col}_count"] = list_count(test[col]).astype(int)
        train[f"{col}_has"] = list_has(train[col]).astype(int)
        test[f"{col}_has"] = list_has(test[col]).astype(int)


def clean_categoricals(train: pd.DataFrame, test: pd.DataFrame, exclude: set) -> None:
    for col in train.columns:
        if col == TARGET or col in exclude:
            continue
        if col not in test.columns:
            continue
        if train[col].dtype == object or str(train[col].dtype) == "category":
            train[col] = train[col].fillna("missing").astype(str)
            test[col] = test[col].fillna("missing").astype(str)


def clean_numerics(train: pd.DataFrame, test: pd.DataFrame, exclude: set) -> None:
    for col in train.columns:
        if col == TARGET or col in exclude:
            continue
        if col not in test.columns:
            continue
        if pd.api.types.is_numeric_dtype(train[col]):
            train[col] = pd.to_numeric(train[col], errors="coerce")
            test[col] = pd.to_numeric(test[col], errors="coerce")
            train[col] = train[col].replace([np.inf, -np.inf], np.nan)
            test[col] = test[col].replace([np.inf, -np.inf], np.nan)
            median = train[col].median()
            if pd.isna(median):
                median = 0.0
            train[col] = train[col].fillna(median).astype(float)
            test[col] = test[col].fillna(median).astype(float)


def drop_constant_columns(train: pd.DataFrame, test: pd.DataFrame, feature_cols: list, protect: set) -> list:
    keep = []
    for col in feature_cols:
        if col in protect:
            keep.append(col)
            continue
        if col not in train.columns or col not in test.columns:
            continue
        if train[col].nunique(dropna=True) <= 1:
            continue
        keep.append(col)
    return keep


def supports_text_features() -> bool:
    try:
        return "text_features" in inspect.signature(CatBoostClassifier.fit).parameters
    except Exception:
        return False


def filter_init_params(params: dict) -> dict:
    try:
        sig = inspect.signature(CatBoostClassifier.__init__)
        return {k: v for k, v in params.items() if k in sig.parameters}
    except Exception:
        return params


def build_model(iterations: int) -> CatBoostClassifier:
    params = {
        "iterations": iterations,
        "learning_rate": 0.05,
        "depth": 6,
        "l2_leaf_reg": 3.0,
        "random_seed": RANDOM_STATE,
        "verbose": 0,
        "eval_metric": "Logloss",
        "allow_writing_files": False,
    }
    return CatBoostClassifier(**filter_init_params(params))


def get_fit_extra(cat_features: list, use_text: bool) -> dict:
    extra = {"cat_features": cat_features}
    if use_text and supports_text_features():
        extra["text_features"] = ["subject"]
    return extra


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    cat_features: list,
    use_text: bool
) -> np.ndarray:
    if len(X_train) == 0 or y_train.nunique() <= 1:
        pred = y_train.mode().iloc[0] if len(y_train) > 0 else "unknown"
        return np.full(len(X_test), pred, dtype=str)

    fit_extra = get_fit_extra(cat_features, use_text)

    if len(X_train) > 200:
        stratify = y_train if y_train.value_counts().min() > 1 else None
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=RANDOM_STATE, stratify=stratify
        )
        model = build_model(1000)
        model.fit(
            X_tr,
            y_tr,
            eval_set=(X_val, y_val),
            early_stopping_rounds=100,
            **fit_extra
        )
        best_iter = getattr(model, "best_iteration_", None)
        if best_iter is None or best_iter <= 0:
            best_iter = 1000
        iterations = int(best_iter) + 1
    else:
        iterations = 1000

    model = build_model(iterations)
    model.fit(X_train, y_train, **fit_extra)
    return model.predict(X_test)


def fallback_predict(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame) -> np.ndarray:
    try:
        X_train2 = X_train.drop(columns=["subject"], errors="ignore")
        X_test2 = X_test.drop(columns=["subject"], errors="ignore")

        if X_train2.shape[1] == 0:
            pred = y_train.mode().iloc[0] if len(y_train) > 0 else "unknown"
            return np.full(len(X_test2), pred, dtype=str)

        cat_features2 = [
            c for c in X_train2.columns
            if X_train2[c].dtype == object or str(X_train2[c].dtype) == "category"
        ]

        model = CatBoostClassifier(**filter_init_params({
            "iterations": 500,
            "learning_rate": 0.05,
            "depth": 6,
            "l2_leaf_reg": 3.0,
            "random_seed": RANDOM_STATE,
            "verbose": 0,
            "allow_writing_files": False,
        }))
        model.fit(X_train2, y_train, cat_features=cat_features2)
        return model.predict(X_test2)
    except Exception:
        pred = y_train.mode().iloc[0] if len(y_train) > 0 else "unknown"
        return np.full(len(X_test), pred, dtype=str)


def to_prediction_array(predictions, n: int) -> np.ndarray:
    if n == 0:
        return np.empty(0, dtype=str)

    arr = np.asarray(predictions)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    else:
        arr = arr.reshape(-1)

    if arr.dtype == object:
        if arr.size > 0:
            first = arr.flat[0]
            if isinstance(first, (list, tuple, np.ndarray)):
                arr = np.array([
                    (x[0] if len(x) > 0 else "unknown") if isinstance(x, (list, tuple, np.ndarray)) else x
                    for x in arr.tolist()
                ], dtype=str)
            else:
                arr = arr.astype(str)
        else:
            arr = np.empty(0, dtype=str)
    else:
        arr = arr.astype(str)

    if arr.size == n:
        return arr
    if arr.size == 0:
        return np.full(n, "unknown", dtype=str)
    if arr.size < n:
        pad = np.full(n - arr.size, arr[-1], dtype=str)
        arr = np.concatenate([arr, pad])
    else:
        arr = arr[:n]

    return arr


def main() -> None:
    train = pd.read_parquet(TRAIN_PATH)
    test = pd.read_parquet(TEST_PATH)

    if TARGET not in train.columns:
        raise ValueError(f"Target column {TARGET!r} not found in train data.")

    if TARGET in test.columns:
        test.drop(columns=[TARGET], inplace=True)

    train = train.dropna(subset=[TARGET]).copy()
    train[TARGET] = train[TARGET].astype(str)

    for col in train.columns:
        if col not in test.columns:
            test[col] = np.nan
    for col in test.columns:
        if col not in train.columns:
            train[col] = np.nan

    drop_cols = {c for c in ["reference", "id", "Id", "ID"] if c in train.columns}
    train.drop(columns=drop_cols, inplace=True, errors="ignore")
    test.drop(columns=drop_cols, inplace=True, errors="ignore")

    use_text = (
        supports_text_features()
        and "subject" in train.columns
        and train["subject"].notna().sum() > 0
        and train["subject"].nunique(dropna=True) > 1
    )

    add_date_features(train, test)
    add_subject_features(train, test, keep_raw_subject=use_text)
    add_list_features(train, test)

    clean_exclude = set(drop_cols) | {TARGET, "date"}
    if use_text:
        train["subject"] = train["subject"].fillna("").astype(str)
        test["subject"] = test["subject"].fillna("").astype(str)
        clean_exclude.add("subject")

    clean_categoricals(train, test, clean_exclude)
    clean_numerics(train, test, clean_exclude)

    feature_cols = [c for c in train.columns if c != TARGET and c not in drop_cols and c != "date"]
    feature_cols = [c for c in feature_cols if c in test.columns]
    protect = {"subject"} if use_text else set()
    feature_cols = drop_constant_columns(train, test, feature_cols, protect)

    if use_text and "subject" not in feature_cols:
        use_text = False
        if "subject" in train.columns:
            train.drop(columns=["subject"], inplace=True, errors="ignore")
            test.drop(columns=["subject"], inplace=True, errors="ignore")
            feature_cols = [c for c in feature_cols if c != "subject"]

    X_train = train[feature_cols].copy()
    y_train = train[TARGET].copy()
    X_test = test[feature_cols].copy()

    if use_text:
        X_train["subject"] = X_train["subject"].fillna("").astype(str)
        X_test["subject"] = X_test["subject"].fillna("").astype(str)

    cat_features = [
        c for c in feature_cols
        if X_train[c].dtype == object or str(X_train[c].dtype) == "category"
    ]
    if use_text and "subject" in cat_features:
        cat_features.remove("subject")

    if len(test) == 0:
        pd.DataFrame({"id": [], "risk_decision": []}).to_csv("predictions.csv", index=False)
        return

    if len(feature_cols) == 0:
        pred = y_train.mode().iloc[0] if len(y_train) > 0 else "unknown"
        predictions = np.full(len(test), pred, dtype=str)
    else:
        try:
            predictions = train_model(X_train, y_train, X_test, cat_features, use_text)
        except Exception:
            predictions = fallback_predict(X_train, y_train, X_test)

    predictions = to_prediction_array(predictions, len(test))

    pd.DataFrame({
        "id": range(len(test)),
        "risk_decision": predictions
    }).to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    main()