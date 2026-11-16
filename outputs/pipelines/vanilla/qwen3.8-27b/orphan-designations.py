import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier

TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/strable/orphan-designations/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/strable/orphan-designations/data/test.parquet"
TARGET = "Status"
RANDOM_STATE = 42

MISSING_STRINGS = ["", "None", "nan", "NaN", "null", "NULL", "NaT"]


def clean_feature_name(name):
    name = re.sub(r"[^0-9a-zA-Z_]+", "_", str(name)).strip("_")
    return name if name else "feature"


def add_feature(frame, name, values):
    name = clean_feature_name(name)
    if name in frame.columns:
        base = name
        i = 1
        while name in frame.columns:
            i += 1
            name = f"{base}_{i}"
    frame[name] = values
    return frame


def present_series(series):
    s = series.fillna("").astype(str).str.strip()
    return (~s.isin(MISSING_STRINGS)).astype(int)


def parse_date_series(series):
    if series is None:
        return pd.Series(pd.NaT, index=[])

    s = series.fillna("").astype(str).str.strip()
    s = s.replace(MISSING_STRINGS, np.nan)

    dt = pd.to_datetime(s, format="%d/%m/%Y", errors="coerce")

    mask = dt.isna() & s.notna()
    if mask.any():
        dt = dt.copy()
        dt[mask] = pd.to_datetime(s[mask], errors="coerce")

    return dt


def extract_eu_parts(series):
    s = series.fillna("").astype(str)
    pattern = r"(?i)eu[-/]?3[-/]?(\d{2})[-/](\d+)"
    extracted = s.str.extract(pattern)

    if extracted is None or extracted.shape[1] < 2:
        year = pd.Series(0, index=series.index, dtype=int)
        number = pd.Series(0, index=series.index, dtype=int)
    else:
        year = pd.to_numeric(extracted[0], errors="coerce").fillna(0).astype(int)
        number = pd.to_numeric(extracted[1], errors="coerce").fillna(0).astype(int)

    full_year = np.where(year > 0, 2000 + year, 0).astype(int)
    return year, number, full_year


def clean_text_series(series):
    s = series.fillna("").astype(str).str.strip()
    s = s.replace(MISSING_STRINGS, "")
    return s.str.lower()


def build_numeric_features(df):
    out = pd.DataFrame(index=df.index)

    for col in df.columns:
        add_feature(out, f"{col}_present", present_series(df[col]))

    date_cols = [
        "Date of designation / refusal",
        "First published date",
        "Last updated date",
    ]
    parsed_dates = {}

    for col in date_cols:
        if col not in df.columns:
            continue

        dt = parse_date_series(df[col])
        parsed_dates[col] = dt

        add_feature(out, f"{col}_year", dt.dt.year.fillna(0).astype(int))
        add_feature(out, f"{col}_month", dt.dt.month.fillna(0).astype(int))
        add_feature(out, f"{col}_day", dt.dt.day.fillna(0).astype(int))
        add_feature(out, f"{col}_dayofweek", dt.dt.dayofweek.fillna(0).astype(int))
        add_feature(out, f"{col}_missing", dt.isna().astype(int))

    designation = "Date of designation / refusal"
    first_published = "First published date"
    last_updated = "Last updated date"

    if designation in parsed_dates and first_published in parsed_dates:
        diff = (parsed_dates[first_published] - parsed_dates[designation]).dt.days
        add_feature(out, "days_designation_to_publish", diff.fillna(0).astype(int))

    if first_published in parsed_dates and last_updated in parsed_dates:
        diff = (parsed_dates[last_updated] - parsed_dates[first_published]).dt.days
        add_feature(out, "days_publish_to_update", diff.fillna(0).astype(int))

    if designation in parsed_dates and last_updated in parsed_dates:
        diff = (parsed_dates[last_updated] - parsed_dates[designation]).dt.days
        add_feature(out, "days_designation_to_update", diff.fillna(0).astype(int))

    eu_col = "EU designation number"
    if eu_col in df.columns:
        add_feature(out, "eu_present", present_series(df[eu_col]))
        eu_year, eu_number, eu_full_year = extract_eu_parts(df[eu_col])
        add_feature(out, "eu_year", eu_year)
        add_feature(out, "eu_number", eu_number)
        add_feature(out, "eu_full_year", eu_full_year)

    url_col = "Orphan designation URL"
    if url_col in df.columns:
        add_feature(out, "url_present", present_series(df[url_col]))
        url_year, url_number, url_full_year = extract_eu_parts(df[url_col])
        add_feature(out, "url_year", url_year)
        add_feature(out, "url_number", url_number)
        add_feature(out, "url_full_year", url_full_year)

    ema_col = "Related EMA product number"
    if ema_col in df.columns:
        add_feature(out, "ema_present", present_series(df[ema_col]))
        ema_number = df[ema_col].fillna("").astype(str).str.extract(r"(\d+)")[0]
        add_feature(out, "ema_number", pd.to_numeric(ema_number, errors="coerce").fillna(0).astype(int))

    text_cols = ["Medicine name", "Active substance", "Intended use"]
    for col in text_cols:
        if col not in df.columns:
            continue

        s = clean_text_series(df[col])
        add_feature(out, f"{col}_length", s.str.len().astype(int))
        add_feature(out, f"{col}_word_count", s.str.split().str.len().fillna(0).astype(int))
        add_feature(out, f"{col}_has_digit", s.str.contains(r"\d", regex=True).astype(int))
        add_feature(out, f"{col}_has_upper", s.str.contains(r"[A-Z]", regex=True).astype(int))

    return out.fillna(0).astype(float)


def build_text_features(df):
    cols = ["Medicine name", "Active substance", "Intended use"]
    parts = []

    for col in cols:
        if col in df.columns:
            parts.append(clean_text_series(df[col]))
        else:
            parts.append(pd.Series("", index=df.index))

    if not parts:
        return pd.Series("", index=df.index)

    if len(parts) == 1:
        return parts[0]

    return pd.concat(parts, axis=1).agg(" ".join, axis=1)


def make_lgbm():
    return LGBMClassifier(
        n_estimators=600,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )


def make_logistic():
    return LogisticRegression(
        C=1.0,
        max_iter=2000,
        class_weight="balanced",
        solver="lbfgs",
        random_state=RANDOM_STATE,
    )


def make_vectorizer():
    return TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True,
        strip_accents="unicode",
        lowercase=True,
        token_pattern=r"(?u)\b[a-zA-Z0-9]+\b",
    )


def safe_auc(y_true, y_score):
    if len(np.unique(y_true)) < 2:
        return 0.5
    return roc_auc_score(y_true, y_score)


def fit_transform_text(vectorizer, texts):
    try:
        return vectorizer.fit_transform(texts)
    except ValueError:
        return np.zeros((len(texts), 0), dtype=np.float64)


def transform_text(vectorizer, texts):
    try:
        return vectorizer.transform(texts)
    except ValueError:
        return np.zeros((len(texts), 0), dtype=np.float64)


def main():
    train_df = pd.read_parquet(TRAIN_PATH).reset_index(drop=True)
    test_df = pd.read_parquet(TEST_PATH).reset_index(drop=True)

    if TARGET in test_df.columns:
        test_df = test_df.drop(TARGET, axis=1)

    y = (
        train_df[TARGET]
        .astype(str)
        .str.strip()
        .str.lower()
        .eq("positive")
        .astype(int)
    )

    train_df = train_df.drop(TARGET, axis=1)

    train_num = build_numeric_features(train_df)
    test_num = build_numeric_features(test_df)

    num_cols = train_num.columns.union(test_num.columns)
    train_num = train_num.reindex(columns=num_cols, fill_value=0).astype(float)
    test_num = test_num.reindex(columns=num_cols, fill_value=0).astype(float)

    train_text = build_text_features(train_df)
    test_text = build_text_features(test_df)

    if y.nunique() < 2:
        constant = float(y.iloc[0]) if len(y) > 0 else 0.5
        predictions = pd.DataFrame(
            {
                "id": np.arange(len(test_df), dtype=int),
                "Status": np.full(len(test_df), constant, dtype=float),
            }
        )
        predictions.to_csv("predictions.csv", index=False)
        return

    class_counts = y.value_counts()
    if class_counts.min() >= 2:
        train_idx, val_idx = train_test_split(
            np.arange(len(train_df)),
            test_size=0.2,
            stratify=y,
            random_state=RANDOM_STATE,
        )
    else:
        train_idx, val_idx = train_test_split(
            np.arange(len(train_df)),
            test_size=0.2,
            random_state=RANDOM_STATE,
        )

    val_vectorizer = make_vectorizer()
    X_text_train_val = fit_transform_text(val_vectorizer, train_text.iloc[train_idx])
    X_text_val = transform_text(val_vectorizer, train_text.iloc[val_idx])

    X_num_train_val = train_num.iloc[train_idx]
    X_num_val = train_num.iloc[val_idx]
    y_train_val = y.iloc[train_idx]
    y_val = y.iloc[val_idx]

    lgbm_val_model = make_lgbm()
    lgbm_val_model.fit(X_num_train_val, y_train_val)
    lgbm_val_score = lgbm_val_model.predict_proba(X_num_val)[:, 1]

    if X_text_train_val.shape[1] > 0 and y_train_val.nunique() > 1:
        log_val_model = make_logistic()
        log_val_model.fit(X_text_train_val, y_train_val)
        log_val_score = log_val_model.predict_proba(X_text_val)[:, 1]
    else:
        log_val_score = np.full(len(val_idx), 0.5, dtype=float)

    candidates = {
        "lgbm": lgbm_val_score,
        "log": log_val_score,
        "avg": 0.5 * lgbm_val_score + 0.5 * log_val_score,
        "lgbm_heavy": 0.7 * lgbm_val_score + 0.3 * log_val_score,
        "log_heavy": 0.3 * lgbm_val_score + 0.7 * log_val_score,
    }

    best_name = max(candidates, key=lambda name: safe_auc(y_val, candidates[name]))

    final_vectorizer = make_vectorizer()
    X_text_train_full = fit_transform_text(final_vectorizer, train_text)
    X_text_test = transform_text(final_vectorizer, test_text)

    lgbm_full_model = make_lgbm()
    lgbm_full_model.fit(train_num, y)
    lgbm_test_score = lgbm_full_model.predict_proba(test_num)[:, 1]

    if X_text_train_full.shape[1] > 0 and y.nunique() > 1:
        log_full_model = make_logistic()
        log_full_model.fit(X_text_train_full, y)
        log_test_score = log_full_model.predict_proba(X_text_test)[:, 1]
    else:
        log_test_score = np.full(len(test_df), 0.5, dtype=float)

    if best_name == "lgbm":
        final_score = lgbm_test_score
    elif best_name == "log":
        final_score = log_test_score
    elif best_name == "avg":
        final_score = 0.5 * lgbm_test_score + 0.5 * log_test_score
    elif best_name == "lgbm_heavy":
        final_score = 0.7 * lgbm_test_score + 0.3 * log_test_score
    else:
        final_score = 0.3 * lgbm_test_score + 0.7 * log_test_score

    final_score = np.clip(final_score, 0.0, 1.0)

    predictions = pd.DataFrame(
        {
            "id": np.arange(len(test_df), dtype=int),
            "Status": final_score.astype(float),
        }
    )
    predictions.to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    main()