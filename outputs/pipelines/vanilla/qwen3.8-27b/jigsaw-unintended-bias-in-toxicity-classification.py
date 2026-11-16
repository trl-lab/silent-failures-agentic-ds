import gc
import warnings

import numpy as np
import pandas as pd
import lightgbm as lgb
from lightgbm import LGBMClassifier
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore")

TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/mlebench/jigsaw-unintended-bias-in-toxicity-classification/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/mlebench/jigsaw-unintended-bias-in-toxicity-classification/data/test.parquet"
OUT_PATH = "predictions.csv"

RANDOM_STATE = 42
TEXT_COL = "comment_text"
TARGET_COL = "target"

EXCLUDE_COLS = {
    "id",
    "target",
    "comment_text",
    "severe_toxicity",
    "obscene",
    "identity_attack",
    "insult",
    "threat",
    "asian",
    "atheist",
    "bisexual",
    "black",
    "buddhist",
    "christian",
    "female",
    "heterosexual",
    "hindu",
    "homosexual_gay_or_lesbian",
    "intellectual_or_learning_disability",
    "jewish",
    "latino",
    "male",
    "muslim",
    "other_disability",
    "other_gender",
    "other_race_or_ethnicity",
    "other_religion",
    "other_sexual_orientation",
    "physical_disability",
    "psychiatric_or_mental_illness",
    "transgender",
    "white",
    "sexual_explicit",
    "publication_id",
    "parent_id",
    "article_id",
    "created_date",
}


def clean_text(series: pd.Series) -> pd.Series:
    if series is None:
        return pd.Series(dtype=str)
    return series.fillna("").astype(str).str.strip()


def count_chars(series: pd.Series, char: str) -> pd.Series:
    return series.str.replace(char, "", regex=False).str.len() * 0 + series.str.len() - series.str.replace(char, "", regex=False).str.len()


def add_text_features(df: pd.DataFrame) -> pd.DataFrame:
    text = df[TEXT_COL]

    df["text_len"] = text.str.len()
    df["space_count"] = text.str.len() - text.str.replace(" ", "", regex=False).str.len()
    df["word_count"] = df["space_count"] + (text != "").astype(int)
    df["avg_word_len"] = np.where(df["word_count"] > 0, df["text_len"] / df["word_count"], 0.0)

    df["exclamation_count"] = text.str.len() - text.str.replace("!", "", regex=False).str.len()
    df["question_count"] = text.str.len() - text.str.replace("?", "", regex=False).str.len()
    df["newline_count"] = text.str.len() - text.str.replace("\n", "", regex=False).str.len()

    df["has_url"] = text.str.contains("http", case=False, regex=False).astype(int)
    df["has_email"] = text.str.contains("@", regex=False).astype(int)

    return df


def select_extra_columns(train_df: pd.DataFrame, test_df: pd.DataFrame):
    common_cols = [c for c in train_df.columns if c in test_df.columns and c not in EXCLUDE_COLS]

    numeric_cols = []
    categorical_cols = []

    for col in common_cols:
        series = train_df[col]

        if pd.api.types.is_numeric_dtype(series):
            if series.notna().any() and series.nunique(dropna=True) > 1:
                numeric_cols.append(col)
        else:
            nunique = series.nunique(dropna=True)
            if 1 < nunique <= 50:
                categorical_cols.append(col)

    return numeric_cols, categorical_cols


def prepare_extra_columns(train_df: pd.DataFrame, test_df: pd.DataFrame, numeric_cols, categorical_cols):
    for col in numeric_cols:
        train_df[col] = pd.to_numeric(train_df[col], errors="coerce").astype(np.float32)
        test_df[col] = pd.to_numeric(test_df[col], errors="coerce").astype(np.float32)

    for col in categorical_cols:
        train_df[col] = train_df[col].fillna("missing").astype(str)
        test_df[col] = test_df[col].fillna("missing").astype(str)


def build_matrix(df: pd.DataFrame, vectorizer, encoder, numeric_cols, categorical_cols, fit: bool = False):
    if fit:
        X_text = vectorizer.fit_transform(df[TEXT_COL])
    else:
        X_text = vectorizer.transform(df[TEXT_COL])

    parts = [X_text]

    if numeric_cols:
        X_num = df[numeric_cols].to_numpy(dtype=np.float32, copy=False)
        X_num = np.nan_to_num(X_num, nan=0.0, posinf=0.0, neginf=0.0)
        parts.append(csr_matrix(X_num))

    if categorical_cols and encoder is not None:
        if fit:
            X_cat = encoder.fit_transform(df[categorical_cols])
        else:
            X_cat = encoder.transform(df[categorical_cols])
        parts.append(X_cat)

    if len(parts) == 1:
        return parts[0].tocsr()

    return hstack(parts, format="csr")


def make_model(n_estimators: int) -> LGBMClassifier:
    return LGBMClassifier(
        objective="binary",
        n_estimators=n_estimators,
        learning_rate=0.1,
        num_leaves=31,
        max_depth=6,
        min_child_samples=100,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        max_bin=255,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )


def main():
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    if TARGET_COL not in train_df.columns:
        raise ValueError(f"Target column {TARGET_COL!r} not found in train data.")

    train_df[TEXT_COL] = clean_text(train_df.get(TEXT_COL))
    test_df[TEXT_COL] = clean_text(test_df.get(TEXT_COL))

    train_df = add_text_features(train_df)
    test_df = add_text_features(test_df)

    y = (train_df[TARGET_COL].fillna(0.0) > 0.5).astype(int)

    numeric_cols, categorical_cols = select_extra_columns(train_df, test_df)
    prepare_extra_columns(train_df, test_df, numeric_cols, categorical_cols)

    feature_cols = [TEXT_COL] + numeric_cols + categorical_cols
    feature_cols = [c for c in feature_cols if c in train_df.columns and c in test_df.columns]

    train_df = train_df[feature_cols + [TARGET_COL]]
    test_df = test_df[feature_cols]

    train_idx, val_idx = train_test_split(
        np.arange(len(train_df)),
        test_size=0.15,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    X_train_df = train_df.iloc[train_idx]
    X_val_df = train_df.iloc[val_idx]
    y_train = y.iloc[train_idx]
    y_val = y.iloc[val_idx]

    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        stop_words="english",
        ngram_range=(1, 1),
        min_df=15,
        max_df=0.9,
        max_features=5000,
        sublinear_tf=True,
        dtype=np.float32,
    )

    encoder = OneHotEncoder(handle_unknown="ignore") if categorical_cols else None

    X_train = build_matrix(X_train_df, vectorizer, encoder, numeric_cols, categorical_cols, fit=True)
    X_val = build_matrix(X_val_df, vectorizer, encoder, numeric_cols, categorical_cols, fit=False)
    X_test = build_matrix(test_df, vectorizer, encoder, numeric_cols, categorical_cols, fit=False)

    del X_train_df, X_val_df
    gc.collect()

    eval_metric = "auc" if len(np.unique(y_val)) > 1 else "binary_logloss"

    model = make_model(n_estimators=200)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric=eval_metric,
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
    )

    best_iteration = int(getattr(model, "best_iteration_", 0) or 0)
    if best_iteration <= 0:
        best_iteration = 100

    val_pred = model.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, val_pred) if len(np.unique(y_val)) > 1 else float("nan")

    print(f"Validation AUC: {val_auc:.6f}")
    print(f"Best iteration: {best_iteration}")

    del model, X_train, X_val
    gc.collect()

    X_full = build_matrix(train_df, vectorizer, encoder, numeric_cols, categorical_cols, fit=False)

    final_model = make_model(n_estimators=max(best_iteration, 1))
    final_model.fit(X_full, y)

    test_pred = final_model.predict_proba(X_test)[:, 1]
    test_pred = np.clip(test_pred, 0.0, 1.0)

    submission = pd.DataFrame(
        {
            "id": np.arange(len(test_pred), dtype=int),
            "target": test_pred.astype(float),
        }
    )

    submission.to_csv(OUT_PATH, index=False)


if __name__ == "__main__":
    main()