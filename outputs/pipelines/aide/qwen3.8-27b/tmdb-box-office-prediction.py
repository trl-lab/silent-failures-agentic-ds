import os
import json
import ast
import re
import warnings

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")
os.makedirs("./working", exist_ok=True)

TRAIN_PATH = "./input/train.parquet"
TEST_PATH = "./input/test.parquet"
SUBMISSION_PATH = "./working/submission.csv"

TARGET = "revenue"
ID_COL = "id"

GENRES = [
    "Adventure",
    "Animation",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Family",
    "Fantasy",
    "History",
    "Horror",
    "Music",
    "Mystery",
    "Romance",
    "Science Fiction",
    "TV Movie",
    "Thriller",
    "War",
    "Western",
]

TOP_COMPANIES = [
    "Warner Bros. Pictures",
    "Universal Pictures",
    "Walt Disney Pictures",
    "20th Century Fox",
    "Sony Pictures",
    "Paramount Pictures",
    "Columbia Pictures",
    "Lionsgate",
    "DreamWorks",
    "New Line Cinema",
    "Miramax",
    "Fox Searchlight",
    "A24",
    "Searchlight Pictures",
    "Blumhouse Productions",
    "Legendary Pictures",
    "Marvel Studios",
    "DC Studios",
    "Pixar",
    "Disney",
]

COUNTRY_MAP = {
    "united states": "US",
    "usa": "US",
    "united kingdom": "GB",
    "uk": "GB",
    "england": "GB",
    "france": "FR",
    "germany": "DE",
    "canada": "CA",
    "japan": "JP",
    "australia": "AU",
    "italy": "IT",
    "spain": "ES",
    "mexico": "MX",
    "south korea": "KR",
    "korea": "KR",
    "china": "CN",
    "india": "IN",
    "brazil": "BR",
    "sweden": "SE",
    "norway": "NO",
    "denmark": "DK",
    "finland": "FI",
    "netherlands": "NL",
    "belgium": "BE",
    "switzerland": "CH",
    "austria": "AT",
    "ireland": "IE",
    "new zealand": "NZ",
    "singapore": "SG",
    "hong kong": "HK",
    "taiwan": "TW",
    "philippines": "PH",
    "thailand": "TH",
    "vietnam": "VN",
    "indonesia": "ID",
    "malaysia": "MY",
    "argentina": "AR",
    "chile": "CL",
    "colombia": "CO",
    "south africa": "ZA",
    "israel": "IL",
    "turkey": "TR",
    "russia": "RU",
    "poland": "PL",
    "czech republic": "CZ",
    "hungary": "HU",
    "romania": "RO",
    "bulgaria": "BG",
    "greece": "GR",
    "portugal": "PT",
    "croatia": "HR",
    "serbia": "RS",
    "bosnia and herzegovina": "BA",
    "slovenia": "SI",
    "slovakia": "SK",
    "latvia": "LV",
    "lithuania": "LT",
    "estonia": "EE",
    "iceland": "IS",
    "monaco": "MC",
    "liechtenstein": "LI",
    "luxembourg": "LU",
    "moldova": "MD",
    "north macedonia": "MK",
    "albania": "AL",
    "montenegro": "ME",
    "kosovo": "XK",
    "belarus": "BY",
    "ukraine": "UA",
    "georgia": "GE",
    "armenia": "AM",
    "azerbaijan": "AZ",
    "kazakhstan": "KZ",
    "uzbekistan": "UZ",
    "mongolia": "MN",
    "macau": "MO",
    "north korea": "KP",
    "laos": "LA",
    "myanmar": "MM",
    "cambodia": "KH",
    "brunei": "BN",
    "greenland": "GL",
    "faroe islands": "FO",
}

TOP_COUNTRIES = [
    "US",
    "GB",
    "FR",
    "DE",
    "CA",
    "JP",
    "AU",
    "IT",
    "ES",
    "MX",
    "KR",
    "CN",
    "IN",
    "BR",
    "SE",
    "NO",
    "DK",
    "FI",
    "NL",
    "BE",
    "CH",
    "AT",
    "IE",
    "NZ",
    "SG",
    "HK",
    "TW",
    "PH",
    "TH",
    "VN",
    "ID",
    "MY",
    "AR",
    "CL",
    "CO",
    "ZA",
    "IL",
    "TR",
    "RU",
    "PL",
    "CZ",
    "HU",
    "RO",
    "BG",
    "GR",
    "PT",
    "HR",
    "RS",
    "BA",
    "SI",
    "SK",
]

LANGUAGES = [
    "en",
    "fr",
    "es",
    "de",
    "it",
    "ja",
    "ko",
    "zh",
    "hi",
    "ru",
    "pt",
    "nl",
    "sv",
    "no",
    "da",
    "fi",
    "pl",
    "tr",
    "ar",
    "th",
    "vi",
    "id",
    "he",
    "cs",
    "el",
    "hu",
    "ro",
    "sk",
    "sl",
    "bg",
    "hr",
    "sr",
    "uk",
    "bn",
    "ta",
    "te",
    "ml",
    "kn",
    "gu",
    "pa",
    "mr",
    "ne",
    "si",
    "ur",
    "fa",
    "ps",
    "ku",
    "sq",
    "mk",
    "bs",
]


def is_missing(x) -> bool:
    if x is None:
        return True
    if isinstance(x, float) and np.isnan(x):
        return True
    if isinstance(x, str):
        return x.strip().lower() in {"", "nan", "none", "null", "nat"}
    return False


def parse_list_field(x):
    """
    Parse TMDB-style list fields, which may be Python lists, JSON strings,
    literal strings, or comma-separated strings. Returns a list of names/codes.
    """
    if is_missing(x):
        return []

    if isinstance(x, (list, tuple, np.ndarray)):
        raw_items = list(x)
    else:
        s = str(x).strip()
        if not s or s.lower() in {"nan", "none", "null", "[]"}:
            return []

        raw_items = None

        if s.startswith("["):
            try:
                raw_items = json.loads(s)
            except Exception:
                try:
                    raw_items = ast.literal_eval(s)
                except Exception:
                    raw_items = None

        if raw_items is None:
            names = re.findall(r'"name"\s*:\s*"([^"]+)"', s)
            if names:
                return names
            return [part.strip() for part in s.split(",") if part.strip()]

        if isinstance(raw_items, dict):
            raw_items = [raw_items]

        if not isinstance(raw_items, list):
            return []

    parsed = []
    for item in raw_items:
        if item is None:
            continue
        if isinstance(item, dict):
            value = item.get("name")
            if value is None:
                value = item.get("iso_3166_1")
            if value is None:
                value = item.get("id")
            if value is not None:
                parsed.append(str(value))
        else:
            parsed.append(str(item))

    return parsed


def country_code_from_name(name: str):
    n = str(name).strip().lower()
    if n in COUNTRY_MAP:
        return COUNTRY_MAP[n]
    if n.isalpha() and len(n) == 2:
        return n.upper()
    return None


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    numeric_cols = ["budget", "popularity", "runtime", "vote_average", "vote_count"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "budget" in df.columns:
        df["budget_log"] = np.log1p(df["budget"].clip(lower=0))

    text_cols = [
        "overview",
        "original_title",
        "title",
        "tagline",
        "belongs_to_collection",
        "homepage",
        "poster_path",
        "imdb_id",
    ]
    for col in text_cols:
        if col in df.columns:
            s = df[col].where(df[col].notna(), "").astype(str)
            df[f"{col}_len"] = s.str.len()
            df[f"{col}_words"] = s.str.split().str.len().fillna(0)

    boolean_text_cols = [
        "homepage",
        "poster_path",
        "tagline",
        "overview",
        "belongs_to_collection",
    ]
    for col in boolean_text_cols:
        if col in df.columns:
            s = df[col].where(df[col].notna(), "").astype(str).str.strip()
            df[f"has_{col}"] = (~s.isin(["", "nan", "none", "null"])).astype(int)

    if "video" in df.columns:
        df["video"] = pd.to_numeric(df["video"], errors="coerce").fillna(0).astype(int)

    list_fields = {
        "genres": "genres",
        "production_companies": "companies",
        "production_countries": "countries",
        "spoken_languages": "languages",
    }

    for col, prefix in list_fields.items():
        if col not in df.columns:
            continue

        parsed = df[col].apply(parse_list_field)
        df[f"{prefix}_count"] = parsed.apply(len)

        if prefix == "genres":
            for genre in GENRES:
                g = genre.lower()
                df[f"genre_{genre}"] = parsed.apply(
                    lambda names, g=g: int(
                        any(g in str(name).lower() for name in names)
                    )
                )

        if prefix == "companies":
            for company in TOP_COMPANIES:
                c = company.lower()
                df[f"company_{company}"] = parsed.apply(
                    lambda names, c=c: int(
                        any(c in str(name).lower() for name in names)
                    )
                )

        if prefix == "countries":
            for country in TOP_COUNTRIES:

                def has_country(names, country=country):
                    for name in names:
                        code = country_code_from_name(name)
                        if code == country:
                            return 1
                    return 0

                df[f"country_{country}"] = parsed.apply(has_country)

    if "release_date" in df.columns:
        dates = pd.to_datetime(df["release_date"], errors="coerce")
        df["release_year"] = dates.dt.year
        df["release_month"] = dates.dt.month
        df["release_day"] = dates.dt.day
        df["release_dayofweek"] = dates.dt.dayofweek
        df["release_missing"] = dates.isna().astype(int)

    if "original_language" in df.columns:
        lang = df["original_language"].fillna("xx").astype(str).str.strip().str.lower()
        for language in LANGUAGES:
            df[f"lang_{language}"] = (lang == language).astype(int)

    if "status" in df.columns:
        status = df["status"].fillna("").astype(str).str.strip().str.lower()
        df["status_released"] = (status == "released").astype(int)
        df["status_post_production"] = (status == "post production").astype(int)
        df["status_planned"] = (status == "planned").astype(int)
        df["status_in_production"] = (status == "in production").astype(int)
        df["status_rumored"] = (status == "rumored").astype(int)

    return df


def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_true = np.clip(y_true, 0.0, None)
    y_pred = np.clip(y_pred, 0.0, None)
    return float(np.sqrt(mean_squared_error(np.log1p(y_true), np.log1p(y_pred))))


def main():
    train = pd.read_parquet(TRAIN_PATH)
    test = pd.read_parquet(TEST_PATH)

    train = train.dropna(subset=[TARGET]).copy()
    train[TARGET] = (
        pd.to_numeric(train[TARGET], errors="coerce").fillna(0.0).clip(lower=0.0)
    )

    if ID_COL not in train.columns:
        train[ID_COL] = np.arange(len(train))
    if ID_COL not in test.columns:
        test[ID_COL] = np.arange(len(test))

    train_feat = add_features(train)
    test_feat = add_features(test)

    exclude_cols = {
        ID_COL,
        TARGET,
        "belongs_to_collection",
        "genres",
        "homepage",
        "imdb_id",
        "original_language",
        "original_title",
        "overview",
        "poster_path",
        "production_companies",
        "production_countries",
        "release_date",
        "spoken_languages",
        "status",
        "tagline",
        "title",
    }

    feature_cols = [
        col
        for col in train_feat.columns
        if col not in exclude_cols
        and pd.api.types.is_numeric_dtype(train_feat[col])
        and train_feat[col].notna().any()
    ]

    for col in feature_cols:
        if col not in test_feat.columns:
            test_feat[col] = 0.0

    X_train = train_feat[feature_cols].astype(float).replace([np.inf, -np.inf], np.nan)
    X_test = test_feat[feature_cols].astype(float).replace([np.inf, -np.inf], np.nan)
    y = train[TARGET].to_numpy(dtype=float)
    log_y = np.log1p(y)

    params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_child_samples": 20,
        "lambda_l1": 0.1,
        "lambda_l2": 0.1,
        "seed": 42,
        "verbose": -1,
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_log = np.zeros(len(X_train), dtype=float)
    best_iterations = []

    for fold, (train_idx, valid_idx) in enumerate(kf.split(X_train), start=1):
        dtrain = lgb.Dataset(X_train.iloc[train_idx], label=log_y[train_idx])
        dvalid = lgb.Dataset(
            X_train.iloc[valid_idx], label=log_y[valid_idx], reference=dtrain
        )

        model = lgb.train(
            params,
            dtrain,
            num_boost_round=5000,
            valid_sets=[dvalid],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
        )

        oof_log[valid_idx] = model.predict(
            X_train.iloc[valid_idx],
            num_iteration=model.best_iteration,
        )
        best_iterations.append(model.best_iteration)

    oof_rev = np.expm1(oof_log)
    oof_rev = np.clip(oof_rev, 0.0, None)
    cv_rmsle = rmsle(y, oof_rev)
    print(f"CV RMSLE: {cv_rmsle:.6f}")

    final_rounds = int(np.mean(best_iterations)) if best_iterations else 1000
    dtrain_full = lgb.Dataset(X_train, label=log_y)
    final_model = lgb.train(
        params,
        dtrain_full,
        num_boost_round=final_rounds,
        callbacks=[lgb.log_evaluation(-1)],
    )

    test_pred_log = final_model.predict(X_test)
    test_pred_rev = np.expm1(test_pred_log)
    test_pred_rev = np.clip(test_pred_rev, 0.0, None)

    submission = pd.DataFrame(
        {
            "id": np.arange(len(test), dtype=int),
            "revenue": test_pred_rev.astype(float),
        }
    )
    submission.to_csv(SUBMISSION_PATH, index=False)


if __name__ == "__main__":
    main()
