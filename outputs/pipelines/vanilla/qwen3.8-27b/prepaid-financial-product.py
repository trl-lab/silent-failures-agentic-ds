import warnings
import re
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/strable/prepaid-financial-product/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/strable/prepaid-financial-product/data/test.parquet"
TARGET = "current_status"
RANDOM_STATE = 42

TEXT_COLS = [
    "issuer_name",
    "product_name",
    "program_manager",
    "other_relevant_parties",
    "path",
    "direct_download",
]

CAT_COLS = [
    "issuer_name",
    "product_name",
    "prepaid_product_type",
    "program_manager_exists",
    "program_manager",
    "other_relevant_parties",
]


def safe_col_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", str(name))


def clean_text(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    eff = pd.to_datetime(df.get("agreement_effective_date"), errors="coerce")
    created = pd.to_datetime(df.get("created_date"), errors="coerce")

    out["eff_year"] = eff.dt.year
    out["eff_month"] = eff.dt.month
    out["eff_day"] = eff.dt.day
    out["eff_dayofweek"] = eff.dt.dayofweek
    out["eff_quarter"] = eff.dt.quarter
    out["eff_dayofyear"] = eff.dt.dayofyear

    out["created_year"] = created.dt.year
    out["created_month"] = created.dt.month
    out["created_day"] = created.dt.day
    out["created_dayofweek"] = created.dt.dayofweek
    out["created_quarter"] = created.dt.quarter
    out["created_dayofyear"] = created.dt.dayofyear
    out["created_hour"] = created.dt.hour
    out["created_minute"] = created.dt.minute
    out["created_second"] = created.dt.second

    out["created_minus_eff_days"] = (created - eff).dt.days
    out["created_minus_eff_hours"] = (created - eff).dt.total_seconds() / 3600.0

    return out


def add_text_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    for col in TEXT_COLS:
        if col not in df.columns:
            continue

        s = clean_text(df[col])
        prefix = safe_col_name(col)

        out[f"{prefix}_len"] = s.str.len()
        out[f"{prefix}_words"] = s.str.split().str.len()
        out[f"{prefix}_digits"] = s.str.count(r"\d")
        out[f"{prefix}_upper"] = s.str.count(r"[A-Z]")
        out[f"{prefix}_lower"] = s.str.count(r"[a-z]")
        out[f"{prefix}_special"] = s.str.count(r"[^A-Za-z0-9]")
        out[f"{prefix}_semicolon"] = s.str.count(";")
        out[f"{prefix}_comma"] = s.str.count(",")
        out[f"{prefix}_paren"] = s.str.count(r"\(")
        out[f"{prefix}_slash"] = s.str.count("/")
        out[f"{prefix}_dot"] = s.str.count(".")
        out[f"{prefix}_has_no_info"] = s.str.contains("no information provided", case=False, na=False).astype(int)

    if "other_relevant_parties" in df.columns:
        s = clean_text(df["other_relevant_parties"])
        out["other_parties_count"] = s.str.count(";") + 1
        out["other_parties_count"] = out["other_parties_count"].where(
            ~s.str.contains("no information provided", case=False, na=False),
            0,
        )

    if "path" in df.columns:
        s = clean_text(df["path"])
        out["path_segments"] = s.str.count("/") + 1
        out["path_has_paren"] = s.str.contains(r"\(", regex=True, na=False).astype(int)

    if "direct_download" in df.columns:
        s = clean_text(df["direct_download"])
        out["download_is_zip"] = s.str.endswith(".zip", na=False).astype(int)
        out["download_has_percent"] = s.str.contains("%", regex=False, na=False).astype(int)

    return out


def add_categorical_features(fit_df: pd.DataFrame, train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_out = pd.DataFrame(index=train_df.index)
    test_out = pd.DataFrame(index=test_df.index)

    for col in CAT_COLS:
        if col not in fit_df.columns:
            continue

        fit_s = clean_text(fit_df[col])
        train_s = clean_text(train_df[col])
        test_s = clean_text(test_df[col])

        prefix = safe_col_name(col)
        nunique = fit_s.nunique()

        if nunique <= 25:
            categories = sorted(fit_s.unique())
            for cat in categories:
                cat_name = safe_col_name(cat)
                train_out[f"{prefix}_{cat_name}"] = (train_s == cat).astype(int)
                test_out[f"{prefix}_{cat_name}"] = (test_s == cat).astype(int)
        else:
            freq = fit_s.value_counts(normalize=True)
            train_out[f"{prefix}_freq"] = train_s.map(freq).fillna(0.0)
            test_out[f"{prefix}_freq"] = test_s.map(freq).fillna(0.0)

    return train_out, test_out


def add_tfidf_features(fit_df: pd.DataFrame, train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    matrices_train = []
    matrices_test = []

    for col in TEXT_COLS:
        if col not in fit_df.columns:
            continue

        fit_s = clean_text(fit_df[col])
        train_s = clean_text(train_df[col])
        test_s = clean_text(test_df[col])

        vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            max_features=800,
            min_df=1,
            sublinear_tf=True,
        )

        X_fit = vectorizer.fit_transform(fit_s)
        X_train = vectorizer.transform(train_s)
        X_test = vectorizer.transform(test_s)

        matrices_train.append(X_train)
        matrices_test.append(X_test)

    if not matrices_train:
        return (
            sparse.csr_matrix((len(train_df), 0)),
            sparse.csr_matrix((len(test_df), 0)),
        )

    return sparse.hstack(matrices_train, format="csr"), sparse.hstack(matrices_test, format="csr")


def build_features(fit_df: pd.DataFrame, train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_parts = [
        add_date_features(train_df),
        add_text_features(train_df),
    ]
    test_parts = [
        add_date_features(test_df),
        add_text_features(test_df),
    ]

    cat_train, cat_test = add_categorical_features(fit_df, train_df, test_df)
    train_parts.append(cat_train)
    test_parts.append(cat_test)

    tfidf_train, tfidf_test = add_tfidf_features(fit_df, train_df, test_df)

    train_df_out = pd.concat(train_parts, axis=1)
    test_df_out = pd.concat(test_parts, axis=1)

    train_df_out = train_df_out.replace([np.inf, -np.inf], np.nan)
    test_df_out = test_df_out.replace([np.inf, -np.inf], np.nan)

    return train_df_out, test_df_out, tfidf_train, tfidf_test


def make_model():
    return LGBMClassifier(
        n_estimators=1500,
        learning_rate=0.03,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=20,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )


def make_xgb_model():
    return XGBClassifier(
        n_estimators=1500,
        learning_rate=0.03,
        max_depth=6,
        min_child_weight=5,
        subsample=0.9,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist",
        eval_metric="auc",
    )


def main():
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    y_full = train_df[TARGET].astype(str)
    classes = sorted(y_full.unique())

    if len(classes) == 2:
        positive_class = classes[1]
    else:
        positive_class = classes[-1]

    y_full_binary = (y_full == positive_class).astype(int)

    train_part, val_part = train_test_split(
        train_df,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y_full_binary,
    )

    y_train_part = train_part[TARGET].astype(str)
    y_val_part = val_part[TARGET].astype(str)

    X_train_part, X_val_part, tfidf_train_part, tfidf_val_part = build_features(
        fit_df=train_part,
        train_df=train_part,
        test_df=val_part,
    )

    X_train_part = sparse.hstack([X_train_part.astype("float32").values, tfidf_train_part], format="csr")
    X_val_part = sparse.hstack([X_val_part.astype("float32").values, tfidf_val_part], format="csr")

    y_train_part_binary = (y_train_part == positive_class).astype(int)
    y_val_part_binary = (y_val_part == positive_class).astype(int)

    lgbm_val = make_model()
    lgbm_val.fit(X_train_part, y_train_part_binary)
    lgbm_val_proba = lgbm_val.predict_proba(X_val_part)[:, 1]

    xgb_val = make_xgb_model()
    xgb_val.fit(X_train_part, y_train_part_binary)
    xgb_val_proba = xgb_val.predict_proba(X_val_part)[:, 1]

    val_proba = 0.5 * lgbm_val_proba + 0.5 * xgb_val_proba
    val_auc = roc_auc_score(y_val_part_binary, val_proba)
    print(f"Validation ROC AUC: {val_auc:.6f}")

    X_train_full, X_test, tfidf_train_full, tfidf_test = build_features(
        fit_df=train_df,
        train_df=train_df,
        test_df=test_df,
    )

    X_train_full = sparse.hstack([X_train_full.astype("float32").values, tfidf_train_full], format="csr")
    X_test = sparse.hstack([X_test.astype("float32").values, tfidf_test], format="csr")

    lgbm_full = make_model()
    lgbm_full.fit(X_train_full, y_full_binary)
    lgbm_full_proba = lgbm_full.predict_proba(X_test)[:, 1]

    xgb_full = make_xgb_model()
    xgb_full.fit(X_train_full, y_full_binary)
    xgb_full_proba = xgb_full.predict_proba(X_test)[:, 1]

    test_proba = 0.5 * lgbm_full_proba + 0.5 * xgb_full_proba
    test_proba = np.clip(test_proba, 0.0, 1.0)

    submission = pd.DataFrame({
        "id": np.arange(len(test_df), dtype=int),
        "current_status": test_proba,
    })

    submission.to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    main()