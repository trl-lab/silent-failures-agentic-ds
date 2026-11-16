import ast
import os
import re
import warnings
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold

try:
    from catboost import CatBoostRegressor

    HAS_CATBOOST = True
except ImportError:
    from sklearn.ensemble import HistGradientBoostingRegressor

    HAS_CATBOOST = False

warnings.filterwarnings("ignore")

# 1. Load Data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

train_size = len(train_df)
combined = pd.concat([train_df, test_df], ignore_index=True)


# 2. JSON Parsing Utilities
def safe_parse_json(val):
    if pd.isna(val) or val is None or val == "":
        return []
    if isinstance(val, (list, dict)):
        return val
    try:
        return ast.literal_eval(str(val))
    except Exception:
        return []


# Feature Extraction Pipeline
def process_data(df):
    data = pd.DataFrame()

    # Id
    data["id"] = df["id"] if "id" in df.columns else np.arange(len(df))

    # Numerical Base Features
    data["budget"] = df["budget"].fillna(0).astype(float)
    data["budget_log1p"] = np.log1p(data["budget"])
    data["is_budget_zero"] = (data["budget"] == 0).astype(int)

    data["popularity"] = df["popularity"].fillna(0).astype(float)
    data["popularity_log1p"] = np.log1p(data["popularity"])

    data["runtime"] = df["runtime"].fillna(0).astype(float)
    data["is_runtime_zero"] = (data["runtime"] == 0).astype(int)

    # Interaction Ratios
    data["budget_to_popularity"] = data["budget"] / (data["popularity"] + 1.0)
    data["budget_to_runtime"] = data["budget"] / (data["runtime"] + 1.0)
    data["popularity_to_runtime"] = data["popularity"] / (data["runtime"] + 1.0)

    # Dates & Temporal Signals
    dates = pd.to_datetime(df["release_date"], errors="coerce")
    years = dates.dt.year.apply(lambda y: y - 100 if y > 2020 else y)

    data["release_year"] = years.fillna(-1).astype(int)
    data["release_month"] = dates.dt.month.fillna(-1).astype(int)
    data["release_day"] = dates.dt.day.fillna(-1).astype(int)
    data["release_dayofweek"] = dates.dt.dayofweek.fillna(-1).astype(int)
    data["release_quarter"] = dates.dt.quarter.fillna(-1).astype(int)
    data["release_dayofyear"] = dates.dt.dayofyear.fillna(-1).astype(int)
    data["release_is_weekend"] = dates.dt.dayofweek.isin([5, 6]).astype(int).fillna(0)
    data["has_release_date"] = dates.notna().astype(int)

    # Categoricals
    data["original_language"] = df["original_language"].fillna("missing").astype(str)
    data["has_homepage"] = df["homepage"].notna().astype(int)
    data["has_tagline"] = df["tagline"].notna().astype(int)
    data["has_overview"] = df["overview"].notna().astype(int)

    # Text Statistics
    data["overview_len"] = df["overview"].fillna("").apply(len)
    data["overview_word_count"] = (
        df["overview"].fillna("").apply(lambda x: len(x.split()))
    )
    data["tagline_len"] = df["tagline"].fillna("").apply(len)
    data["title_len"] = df["title"].fillna("").apply(len)

    # JSON Extraction
    parsed_collection = df["belongs_to_collection"].apply(safe_parse_json)
    data["has_collection"] = parsed_collection.apply(lambda x: 1 if len(x) > 0 else 0)
    data["collection_name"] = parsed_collection.apply(
        lambda x: x[0]["name"] if len(x) > 0 and "name" in x[0] else "None"
    )

    parsed_genres = df["genres"].apply(safe_parse_json)
    data["num_genres"] = parsed_genres.apply(len)
    data["primary_genre"] = parsed_genres.apply(
        lambda x: x[0]["name"] if len(x) > 0 and "name" in x[0] else "None"
    )

    parsed_companies = df["production_companies"].apply(safe_parse_json)
    data["num_companies"] = parsed_companies.apply(len)
    data["primary_company"] = parsed_companies.apply(
        lambda x: x[0]["name"] if len(x) > 0 and "name" in x[0] else "None"
    )

    parsed_countries = df["production_countries"].apply(safe_parse_json)
    data["num_countries"] = parsed_countries.apply(len)
    data["is_usa_prod"] = parsed_countries.apply(
        lambda x: (
            1
            if any(
                c.get("iso_3166_1") == "US" or "United States" in c.get("name", "")
                for c in x
            )
            else 0
        )
    )

    parsed_spoken = df["spoken_languages"].apply(safe_parse_json)
    data["num_languages"] = parsed_spoken.apply(len)
    data["is_english_spoken"] = parsed_spoken.apply(
        lambda x: (
            1
            if any(
                l.get("iso_639_1") == "en" or "English" in l.get("name", "") for l in x
            )
            else 0
        )
    )

    parsed_cast = df["cast"].apply(safe_parse_json)
    data["cast_size"] = parsed_cast.apply(len)
    data["female_cast_count"] = parsed_cast.apply(
        lambda x: sum(1 for c in x if c.get("gender") == 1)
    )
    data["male_cast_count"] = parsed_cast.apply(
        lambda x: sum(1 for c in x if c.get("gender") == 2)
    )
    data["lead_actor"] = parsed_cast.apply(
        lambda x: x[0]["name"] if len(x) > 0 and "name" in x[0] else "None"
    )

    parsed_crew = df["crew"].apply(safe_parse_json)
    data["crew_size"] = parsed_crew.apply(len)

    def get_director(crew_list):
        for member in crew_list:
            if member.get("job") == "Director":
                return member.get("name", "None")
        return "None"

    data["director"] = parsed_crew.apply(get_director)

    return data


df_processed = process_data(combined)

# Text TF-IDF + SVD features
text_data = (
    combined["title"].fillna("")
    + " "
    + combined["tagline"].fillna("")
    + " "
    + combined["overview"].fillna("")
)
tfidf = TfidfVectorizer(max_features=5000, stop_words="english")
tfidf_feats = tfidf.fit_transform(text_data)
svd = TruncatedSVD(n_components=16, random_state=42)
svd_feats = svd.fit_transform(tfidf_feats)

for i in range(16):
    df_processed[f"text_svd_{i}"] = svd_feats[:, i]

# Split Train / Test
X_all = df_processed.drop(columns=["id"])
cat_cols = [
    "original_language",
    "collection_name",
    "primary_genre",
    "primary_company",
    "lead_actor",
    "director",
]
for col in cat_cols:
    X_all[col] = X_all[col].astype(str).fillna("missing")

X_train = X_all.iloc[:train_size].copy()
X_test = X_all.iloc[train_size:].copy()

y_train = train_df["revenue"].fillna(0).values
y_train_log = np.log1p(y_train)

# Model Training with 5-Fold Cross Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(train_size)
test_preds = np.zeros(len(test_df))

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train_log)):
    X_tr, y_tr = X_train.iloc[train_idx], y_train_log[train_idx]
    X_va, y_va = X_train.iloc[val_idx], y_train_log[val_idx]

    if HAS_CATBOOST:
        model = CatBoostRegressor(
            iterations=800,
            learning_rate=0.05,
            depth=6,
            cat_features=cat_cols,
            verbose=0,
            random_seed=42 + fold,
        )
        model.fit(X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=50)
        oof_preds[val_idx] = model.predict(X_va)
        test_preds += model.predict(X_test) / kf.n_splits
    else:
        # Fallback to target encoding + HistGradientBoosting
        for col in cat_cols:
            freq = X_tr[col].value_counts().to_dict()
            X_tr[col + "_freq"] = X_tr[col].map(freq).fillna(0)
            X_va[col + "_freq"] = X_va[col].map(freq).fillna(0)
            X_test_mod = X_test.copy()
            X_test_mod[col + "_freq"] = X_test_mod[col].map(freq).fillna(0)

        X_tr_num = X_tr.drop(columns=cat_cols)
        X_va_num = X_va.drop(columns=cat_cols)
        X_te_num = X_test_mod.drop(columns=cat_cols)

        model = HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.05, random_state=42 + fold
        )
        model.fit(X_tr_num, y_tr)
        oof_preds[val_idx] = model.predict(X_va_num)
        test_preds += model.predict(X_te_num) / kf.n_splits

oof_preds_clipped = np.clip(oof_preds, 0, None)
cv_rmsle = root_mean_squared_error(y_train_log, oof_preds_clipped)

print(f"5-Fold Cross-Validation RMSLE: {cv_rmsle:.5f}")

# Output Submission File
test_revenue_pred = np.expm1(np.clip(test_preds, 0, None))
sub = pd.DataFrame(
    {
        "id": test_df["id"] if "id" in test_df.columns else np.arange(len(test_df)),
        "revenue": test_revenue_pred,
    }
)

os.makedirs("./working", exist_ok=True)
sub.to_csv("./working/submission.csv", index=False)
