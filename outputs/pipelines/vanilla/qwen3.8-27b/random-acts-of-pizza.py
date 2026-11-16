import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier

TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/mlebench/random-acts-of-pizza/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/mlebench/random-acts-of-pizza/data/test.parquet"
TARGET = "requester_received_pizza"

TEXT_COLS = ["request_title", "request_text", "request_text_edit_aware"]
DROP_COLS = [
    "request_id",
    "requester_username",
    "giver_username_if_known",
    "requester_subreddits_at_request",
] + TEXT_COLS

KEYWORDS = [
    "broke", "hungry", "food", "please", "thank", "thanks", "college", "student",
    "family", "sick", "ill", "homeless", "pay it forward", "miscarriage",
    "depression", "anxiety", "stress", "job", "work", "money", "rent", "bills",
    "no money", "cant afford", "cannot afford", "help", "grateful", "appreciate",
    "pizza", "hungry howie", "dominos", "papa johns", "little caesars"
]

TOP_SUBREDDITS = [
    "random_acts_of_pizza", "askreddit", "pics", "funny", "self", "wtf",
    "adviceanimals", "twoxchromosomes", "offmychest", "depression", "aww",
    "gaming", "technology", "science", "todayilearned", "videos",
    "programming", "linux", "ubuntu", "comics", "anime", "starcraft",
    "tf2", "buildapc", "hardware", "software", "reddit.com"
]


def parse_subreddits(value):
    if value is None:
        return []

    if isinstance(value, np.ndarray):
        value = value.tolist()

    if isinstance(value, (list, tuple, set)):
        return [str(x).strip() for x in value if str(x).strip()]

    if not isinstance(value, str):
        return []

    value = value.strip()
    if value in ("", "[]", "nan", "None"):
        return []

    if value.startswith("[") and value.endswith("]"):
        value = value[1:-1]

    parts = [part.strip().strip("'\"") for part in value.split(",")]
    return [part for part in parts if part]


def add_features(df):
    df = df.copy()

    for col in TEXT_COLS:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str).str.slice(0, 5000)

    df["combined_text"] = (
        df["request_title"] + " " +
        df["request_text"] + " " +
        df["request_text_edit_aware"]
    )
    df["combined_text"] = df["combined_text"].fillna("").astype(str)

    df["text_len"] = df["combined_text"].str.len()
    df["text_word_count"] = df["combined_text"].str.split().str.len().fillna(0)
    df["title_len"] = df["request_title"].str.len()
    df["text_exclamation_count"] = df["combined_text"].str.count("!")
    df["text_question_count"] = df["combined_text"].str.count(r"\?")

    lower_text = df["combined_text"].str.lower()
    for kw in KEYWORDS:
        df[f"kw_{kw.replace(' ', '_')}"] = lower_text.str.contains(kw, regex=False).astype(int)

    if "requester_subreddits_at_request" in df.columns:
        subreddits = df["requester_subreddits_at_request"].apply(parse_subreddits)
    else:
        subreddits = pd.Series([[] for _ in range(len(df))], index=df.index)

    df["subreddits_count"] = subreddits.apply(len)
    for sub in TOP_SUBREDDITS:
        df[f"sub_{sub}"] = subreddits.apply(
            lambda lst, s=sub: int(any(x.lower() == s for x in lst))
        )

    for ts_col in ["unix_timestamp_of_request_utc", "unix_timestamp_of_request"]:
        if ts_col in df.columns:
            ts_num = pd.to_numeric(df[ts_col], errors="coerce")
            ts = pd.to_datetime(ts_num, unit="s", utc=True, errors="coerce")
            df[f"{ts_col}_year"] = ts.dt.year
            df[f"{ts_col}_month"] = ts.dt.month
            df[f"{ts_col}_day"] = ts.dt.day
            df[f"{ts_col}_dayofweek"] = ts.dt.dayofweek
            df[f"{ts_col}_hour"] = ts.dt.hour

    if "unix_timestamp_of_request" in df.columns and "unix_timestamp_of_request_utc" in df.columns:
        df["timestamp_diff"] = (
            pd.to_numeric(df["unix_timestamp_of_request"], errors="coerce") -
            pd.to_numeric(df["unix_timestamp_of_request_utc"], errors="coerce")
        )

    drop_cols = [c for c in DROP_COLS if c in df.columns]
    if TARGET in df.columns:
        drop_cols = [c for c in drop_cols if c != TARGET]
    df = df.drop(columns=drop_cols)

    return df


def make_model(n_estimators=1000):
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
        verbose=-1
    )


def save_predictions(n, probs):
    probs = np.asarray(probs, dtype=float).reshape(-1)

    if probs.size != n:
        if probs.size == 0:
            probs = np.full(n, 0.5, dtype=float)
        else:
            probs = np.resize(probs, n)

    probs = np.nan_to_num(probs, nan=0.5, posinf=1.0, neginf=0.0)
    probs = np.clip(probs, 0.0, 1.0)

    pd.DataFrame({
        "id": np.arange(n, dtype=int),
        "requester_received_pizza": probs
    }).to_csv("predictions.csv", index=False)


def baseline_predictions(n, y_train):
    if len(y_train) > 0:
        p = float(y_train.mean())
        p = min(max(p, 0.01), 0.99)
    else:
        p = 0.5
    return np.full(n, p, dtype=float)


def main():
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    if TARGET in test_df.columns:
        test_df = test_df.drop(columns=[TARGET])

    train_df = add_features(train_df)
    test_df = add_features(test_df)

    if TARGET in train_df.columns:
        train_df = train_df.dropna(subset=[TARGET])
        y_train = train_df[TARGET].astype(int)
        X_train = train_df.drop(columns=[TARGET])
    else:
        y_train = pd.Series([], dtype=int)
        X_train = train_df

    common_cols = [c for c in X_train.columns if c in test_df.columns]
    X_train = X_train[common_cols].copy()
    X_test = test_df[common_cols].copy()

    if len(X_test) == 0:
        save_predictions(0, [])
        return

    if len(X_train) == 0 or y_train.nunique() <= 1:
        save_predictions(len(X_test), baseline_predictions(len(X_test), y_train))
        return

    if "combined_text" not in X_train.columns:
        X_train["combined_text"] = ""
        X_test["combined_text"] = ""

    X_train["combined_text"] = X_train["combined_text"].fillna("").astype(str)
    X_test["combined_text"] = X_test["combined_text"].fillna("").astype(str)

    object_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    cat_cols = [
        col for col in object_cols
        if col != "combined_text" and X_train[col].nunique(dropna=True) <= 50
    ]

    for col in cat_cols:
        X_train[col] = X_train[col].astype(str).replace({"nan": "missing", "None": "missing", "": "missing"})
        X_test[col] = X_test[col].astype(str).replace({"nan": "missing", "None": "missing", "": "missing"})

    numeric_cols = [
        col for col in X_train.columns
        if col not in cat_cols and col != "combined_text" and pd.api.types.is_numeric_dtype(X_train[col])
    ]

    for col in numeric_cols:
        X_train[col] = pd.to_numeric(X_train[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        X_test[col] = pd.to_numeric(X_test[col], errors="coerce").replace([np.inf, -np.inf], np.nan)

    transformers = []

    if numeric_cols:
        transformers.append(("num", SimpleImputer(strategy="median"), numeric_cols))

    if X_train["combined_text"].str.len().sum() > 0:
        transformers.append((
            "text",
            TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                min_df=1,
                stop_words="english",
                sublinear_tf=True,
                strip_accents="unicode",
                dtype=np.float32
            ),
            ["combined_text"]
        ))

    if cat_cols:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols))

    if not transformers:
        save_predictions(len(X_test), baseline_predictions(len(X_test), y_train))
        return

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.0,
        n_jobs=1
    )

    best_iteration = 1000

    if len(X_train) > 20 and y_train.nunique() > 1:
        vc = y_train.value_counts()
        stratify = y_train if vc.min() >= 2 else None

        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
            X_train,
            y_train,
            test_size=0.2,
            random_state=42,
            stratify=stratify
        )

        if y_val_split.nunique() > 1:
            val_pipeline = Pipeline([
                ("preprocessor", preprocessor),
                ("model", make_model())
            ])

            try:
                val_pipeline.fit(
                    X_train_split,
                    y_train_split,
                    model__eval_set=[(X_val_split, y_val_split)],
                    model__eval_metric="auc",
                    model__early_stopping_rounds=50,
                    model__verbose=False
                )

                val_pred = val_pipeline.predict_proba(X_val_split)[:, 1]
                val_auc = roc_auc_score(y_val_split, val_pred)
                print(f"Validation AUC: {val_auc:.6f}")

                best_iter = getattr(val_pipeline.named_steps["model"], "best_iteration_", None)
                if best_iter is not None and best_iter > 0:
                    best_iteration = max(int(best_iter), 100)
            except Exception as exc:
                print(f"Validation failed: {exc}")

    try:
        final_pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", make_model(n_estimators=best_iteration))
        ])

        final_pipeline.fit(X_train, y_train)
        test_pred = final_pipeline.predict_proba(X_test)[:, 1]
        save_predictions(len(X_test), test_pred)
        return

    except Exception as exc:
        print(f"Final model failed: {exc}")

    simple_transformers = [t for t in transformers if t[0] != "text"]
    if simple_transformers:
        try:
            simple_preprocessor = ColumnTransformer(
                transformers=simple_transformers,
                remainder="drop",
                sparse_threshold=0.0,
                n_jobs=1
            )

            simple_pipeline = Pipeline([
                ("preprocessor", simple_preprocessor),
                ("model", make_model(n_estimators=min(best_iteration, 500)))
            ])

            simple_pipeline.fit(X_train, y_train)
            test_pred = simple_pipeline.predict_proba(X_test)[:, 1]
            save_predictions(len(X_test), test_pred)
            return

        except Exception as exc:
            print(f"Simple fallback failed: {exc}")

    save_predictions(len(X_test), baseline_predictions(len(X_test), y_train))


if __name__ == "__main__":
    main()