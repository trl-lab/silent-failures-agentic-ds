import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
import os
import re
import json

# ============================================================
# 1. LOAD DATA
# ============================================================
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")
print(
    f"Revenue stats: mean={train_df['revenue'].mean():.0f}, median={train_df['revenue'].median():.0f}, max={train_df['revenue'].max():.0f}"
)

# ============================================================
# 2. CREATE TRAIN/VAL SPLIT
# ============================================================
train_with_target = train_df.copy()
train_with_target["log_revenue"] = np.log1p(train_with_target["revenue"])
train_with_target["revenue_bin"] = pd.qcut(
    train_with_target["log_revenue"], q=10, labels=False, duplicates="drop"
)

train_split, val_split = train_test_split(
    train_with_target,
    test_size=0.15,
    random_state=42,
    stratify=train_with_target["revenue_bin"],
)
print(f"Train split: {len(train_split)}, Val split: {len(val_split)}")

# ============================================================
# 3. FEATURE ENGINEERING
# ============================================================


def extract_list_length(series):
    def _count(s):
        if pd.isna(s) or s == "":
            return 0
        if isinstance(s, list):
            return len(s)
        s = str(s)
        if s.startswith("["):
            return max(0, s.count(",") + 1)
        return 0

    return series.apply(_count)


def extract_unique_words(series):
    def _count(s):
        if pd.isna(s) or s == "":
            return 0
        return len(set(str(s).lower().split()))

    return series.apply(_count)


def extract_word_count(series):
    def _count(s):
        if pd.isna(s) or s == "":
            return 0
        return len(str(s).split())

    return series.apply(_count)


def extract_sentence_count(series):
    def _count(s):
        if pd.isna(s) or s == "":
            return 0
        return len(re.split(r"[.!?]+", str(s)))

    return series.apply(_count)


def extract_has_numbers(series):
    return series.apply(lambda s: 1 if pd.notna(s) and re.search(r"\d", str(s)) else 0)


def extract_has_exclamation(series):
    return series.apply(lambda s: 1 if pd.notna(s) and "!" in str(s) else 0)


def extract_has_question(series):
    return series.apply(lambda s: 1 if pd.notna(s) and "?" in str(s) else 0)


def extract_has_quotes(series):
    return series.apply(
        lambda s: 1 if pd.notna(s) and ('"' in str(s) or "'" in str(s)) else 0
    )


def extract_has_sequel_keywords(series):
    keywords = [
        "sequel",
        "prequel",
        "remake",
        "reboot",
        "spin-off",
        "spinoff",
        "part 2",
        "part 3",
        "part 4",
        "chapter",
        "book 1",
        "book 2",
        "returns",
        "continues",
        "follows",
        "based on",
    ]

    def _check(s):
        if pd.isna(s):
            return 0
        s_lower = str(s).lower()
        return 1 if any(kw in s_lower for kw in keywords) else 0

    return series.apply(_check)


def extract_has_franchise_keywords(series):
    keywords = ["part", "chapter", "rise", "fall", "origin", "return", "revenge"]

    def _check(s):
        if pd.isna(s):
            return 0
        s_lower = str(s).lower()
        if re.search(r"\b\d+\b", s_lower):
            return 1
        if any(kw in s_lower for kw in keywords):
            return 1
        return 0

    return series.apply(_check)


def extract_title_length(series):
    return series.apply(lambda s: len(str(s)) if pd.notna(s) else 0)


def extract_title_has_number(series):
    return series.apply(lambda s: 1 if pd.notna(s) and re.search(r"\d", str(s)) else 0)


def extract_title_has_colon(series):
    return series.apply(lambda s: 1 if pd.notna(s) and ":" in str(s) else 0)


def extract_release_year(series):
    def _get_year(s):
        if pd.isna(s) or s == "":
            return np.nan
        try:
            return pd.Timestamp(s).year
        except:
            return np.nan

    return series.apply(_get_year)


def extract_release_month(series):
    def _get_month(s):
        if pd.isna(s) or s == "":
            return np.nan
        try:
            return pd.Timestamp(s).month
        except:
            return np.nan

    return series.apply(_get_month)


def extract_release_dayofweek(series):
    def _get_dow(s):
        if pd.isna(s) or s == "":
            return np.nan
        try:
            return pd.Timestamp(s).dayofweek
        except:
            return np.nan

    return series.apply(_get_dow)


def extract_release_decade(series):
    def _get_decade(s):
        if pd.isna(s) or s == "":
            return np.nan
        try:
            year = pd.Timestamp(s).year
            return (year // 10) * 10
        except:
            return np.nan

    return series.apply(_get_decade)


def extract_is_summer(series):
    def _check(s):
        if pd.isna(s) or s == "":
            return np.nan
        try:
            month = pd.Timestamp(s).month
            return 1 if 5 <= month <= 8 else 0
        except:
            return np.nan

    return series.apply(_check)


def extract_is_holiday_season(series):
    def _check(s):
        if pd.isna(s) or s == "":
            return np.nan
        try:
            month = pd.Timestamp(s).month
            return 1 if month in [11, 12, 1] else 0
        except:
            return np.nan

    return series.apply(_check)


def extract_is_valentine(series):
    def _check(s):
        if pd.isna(s) or s == "":
            return np.nan
        try:
            ts = pd.Timestamp(s)
            valentine = pd.Timestamp(f"{ts.year}-02-14")
            return 1 if abs((ts - valentine).days) <= 7 else 0
        except:
            return np.nan

    return series.apply(_check)


def extract_is_easter(series):
    def _check(s):
        if pd.isna(s) or s == "":
            return np.nan
        try:
            ts = pd.Timestamp(s)
            easter_month = 3 if ts.year % 4 == 0 else 4
            easter_day = 25 if easter_month == 3 else 15
            easter = pd.Timestamp(f"{ts.year}-{easter_month:02d}-{easter_day:02d}")
            return 1 if abs((ts - easter).days) <= 14 else 0
        except:
            return np.nan

    return series.apply(_check)


def extract_has_collection(series):
    return series.apply(lambda s: 1 if pd.notna(s) and s != "" and s != "[]" else 0)


def extract_has_homepage(series):
    return series.apply(lambda s: 1 if pd.notna(s) and s != "" else 0)


def extract_has_imdb(series):
    return series.apply(lambda s: 1 if pd.notna(s) and s != "" else 0)


def extract_has_poster(series):
    return series.apply(lambda s: 1 if pd.notna(s) and s != "" else 0)


def extract_has_tagline(series):
    return series.apply(lambda s: 1 if pd.notna(s) and s != "" else 0)


def extract_tagline_length(series):
    return series.apply(lambda s: len(str(s)) if pd.notna(s) and s != "" else 0)


def extract_has_budget(series):
    return series.apply(lambda s: 1 if pd.notna(s) and s > 0 else 0)


def extract_log_budget(series):
    return np.log1p(series.fillna(0).clip(lower=0))


def extract_budget_per_minute(series_budget, series_runtime):
    runtime = series_runtime.fillna(
        series_runtime.median() if series_runtime.notna().any() else 100
    )
    runtime = runtime.replace(0, np.nan)
    return series_budget.fillna(0) / runtime


def parse_cast_names(series):
    """Extract top 5 actor names from cast JSON string."""
    def _parse(s):
        if pd.isna(s) or s == "" or s == "[]":
            return []
        try:
            data = json.loads(str(s))
            names = [item.get("name", "") for item in data[:5]]
            return names
        except:
            return []
    return series.apply(_parse)


def parse_crew_director(series):
    """Extract director name from crew JSON string."""
    def _parse(s):
        if pd.isna(s) or s == "" or s == "[]":
            return "unknown"
        try:
            data = json.loads(str(s))
            for item in data:
                if item.get("job", "").lower() == "director":
                    return item.get("name", "unknown")
            return "unknown"
        except:
            return "unknown"
    return series.apply(_parse)


def parse_crew_names(series):
    """Extract top 5 crew member names from crew JSON string."""
    def _parse(s):
        if pd.isna(s) or s == "" or s == "[]":
            return []
        try:
            data = json.loads(str(s))
            names = [item.get("name", "") for item in data[:5]]
            return names
        except:
            return []
    return series.apply(_parse)


def parse_genres(series):
    """Parse genres JSON string into list of genre names."""
    def _parse(s):
        if pd.isna(s) or s == "" or s == "[]":
            return []
        try:
            data = json.loads(str(s))
            return [item.get("name", "") for item in data]
        except:
            # Try regex fallback
            names = re.findall(r'"name"\s*:\s*"([^"]*)"', str(s))
            return names
    return series.apply(_parse)


def engineer_features(df, train_df_for_encoding=None):
    df = df.copy()

    # Numerical features
    df["log_budget"] = extract_log_budget(df["budget"])
    df["has_budget"] = extract_has_budget(df["budget"])

    if "runtime" in df.columns:
        df["budget_per_minute"] = extract_budget_per_minute(df["budget"], df["runtime"])

    if "popularity" in df.columns:
        df["log_popularity"] = np.log1p(df["popularity"].fillna(0).clip(lower=0))
        df["popularity_sq"] = (df["popularity"].fillna(0)) ** 2

    if "runtime" in df.columns:
        df["runtime_missing"] = df["runtime"].isna().astype(int)
        df["log_runtime"] = np.log1p(df["runtime"].fillna(0).clip(lower=0))

    # Temporal features
    if "release_date" in df.columns:
        df["release_year"] = extract_release_year(df["release_date"])
        df["release_month"] = extract_release_month(df["release_date"])
        df["release_dayofweek"] = extract_release_dayofweek(df["release_date"])
        df["release_decade"] = extract_release_decade(df["release_date"])
        df["is_summer_release"] = extract_is_summer(df["release_date"])
        df["is_holiday_release"] = extract_is_holiday_season(df["release_date"])
        df["is_valentine_release"] = extract_is_valentine(df["release_date"])
        df["is_easter_release"] = extract_is_easter(df["release_date"])
        df["release_date_missing"] = df["release_date"].isna().astype(int)
        df["month_sin"] = np.sin(2 * np.pi * df["release_month"].fillna(6) / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["release_month"].fillna(6) / 12)

    # Text features
    if "overview" in df.columns:
        df["overview_word_count"] = extract_word_count(df["overview"])
        df["overview_unique_words"] = extract_unique_words(df["overview"])
        df["overview_sentence_count"] = extract_sentence_count(df["overview"])
        df["overview_has_numbers"] = extract_has_numbers(df["overview"])
        df["overview_has_exclamation"] = extract_has_exclamation(df["overview"])
        df["overview_has_question"] = extract_has_question(df["overview"])
        df["overview_has_quotes"] = extract_has_quotes(df["overview"])
        df["overview_has_sequel_keywords"] = extract_has_sequel_keywords(df["overview"])
        df["overview_missing"] = df["overview"].isna().astype(int)
        df["overview_length"] = df["overview"].apply(
            lambda s: len(str(s)) if pd.notna(s) else 0
        )

    # Title features
    if "original_title" in df.columns:
        df["title_length"] = extract_title_length(df["original_title"])
        df["title_has_number"] = extract_title_has_number(df["original_title"])
        df["title_has_colon"] = extract_title_has_colon(df["original_title"])
        df["title_has_franchise_keywords"] = extract_has_franchise_keywords(
            df["original_title"]
        )

    # Collection / metadata
    if "belongs_to_collection" in df.columns:
        df["has_collection"] = extract_has_collection(df["belongs_to_collection"])
        df["collection_name_length"] = df["belongs_to_collection"].apply(
            lambda s: len(str(s)) if pd.notna(s) and s != "" and s != "[]" else 0
        )
    if "homepage" in df.columns:
        df["has_homepage"] = extract_has_homepage(df["homepage"])
    if "imdb_id" in df.columns:
        df["has_imdb"] = extract_has_imdb(df["imdb_id"])
    if "poster_path" in df.columns:
        df["has_poster"] = extract_has_poster(df["poster_path"])
    if "tagline" in df.columns:
        df["has_tagline"] = extract_has_tagline(df["tagline"])
        df["tagline_length"] = extract_tagline_length(df["tagline"])

    # List-like features
    if "genres" in df.columns:
        df["genre_count"] = extract_list_length(df["genres"])
        # Genre binary indicators
        genre_list = parse_genres(df["genres"])
        all_genres = set()
        for g in genre_list:
            for genre in g:
                all_genres.add(genre.lower().replace(" ", "_"))
        # Use top 15 most common genres
        genre_counts = {}
        for g in genre_list:
            for genre in g:
                key = genre.lower().replace(" ", "_")
                genre_counts[key] = genre_counts.get(key, 0) + 1
        top_genres = sorted(genre_counts.keys(), key=lambda x: genre_counts[x], reverse=True)[:15]
        for genre in top_genres:
            df[f"genre_{genre}"] = genre_list.apply(lambda x: 1 if genre in [g.lower().replace(" ", "_") for g in x] else 0)
    if "production_companies" in df.columns:
        df["production_company_count"] = extract_list_length(df["production_companies"])
    if "production_countries" in df.columns:
        df["production_country_count"] = extract_list_length(df["production_countries"])
    if "cast" in df.columns:
        df["cast_count"] = extract_list_length(df["cast"])
        # Cast name count (number of unique names in cast)
        cast_names = parse_cast_names(df["cast"])
        df["cast_name_count"] = cast_names.apply(lambda x: len(set(x)) if x else 0)
    if "crew" in df.columns:
        df["crew_count"] = extract_list_length(df["crew"])
        # Crew name count (number of unique names in crew)
        crew_names = parse_crew_names(df["crew"])
        df["crew_name_count"] = crew_names.apply(lambda x: len(set(x)) if x else 0)
        # Director name
        df["director_name"] = parse_crew_director(df["crew"])
    if "keywords" in df.columns:
        df["keyword_count"] = extract_list_length(df["keywords"])

    # Frequency encoding (fit on train only)
    if train_df_for_encoding is not None:
        if "genres" in df.columns and "genres" in train_df_for_encoding.columns:

            def get_first_genre(s):
                if pd.isna(s) or s == "":
                    return "unknown"
                s = str(s)
                if s.startswith("["):
                    match = re.search(r'"([^"]*)"', s)
                    if match:
                        return match.group(1)
                return s

            train_first_genre = train_df_for_encoding["genres"].apply(get_first_genre)
            genre_freq = train_first_genre.value_counts()
            df["first_genre"] = df["genres"].apply(get_first_genre)
            df["first_genre_freq"] = df["first_genre"].map(genre_freq).fillna(0)

        if (
            "production_companies" in df.columns
            and "production_companies" in train_df_for_encoding.columns
        ):

            def get_first_company(s):
                if pd.isna(s) or s == "":
                    return "unknown"
                s = str(s)
                if s.startswith("["):
                    match = re.search(r'"([^"]*)"', s)
                    if match:
                        return match.group(1)
                return s

            train_first_company = train_df_for_encoding["production_companies"].apply(
                get_first_company
            )
            company_freq = train_first_company.value_counts()
            df["first_company"] = df["production_companies"].apply(get_first_company)
            df["first_company_freq"] = df["first_company"].map(company_freq).fillna(0)

        if (
            "original_language" in df.columns
            and "original_language" in train_df_for_encoding.columns
        ):
            lang_freq = train_df_for_encoding["original_language"].value_counts()
            df["language_freq"] = df["original_language"].map(lang_freq).fillna(0)

        # Cast actor frequency encoding
        if "cast" in df.columns and "cast" in train_df_for_encoding.columns:
            train_cast_names = parse_cast_names(train_df_for_encoding["cast"])
            actor_freq = {}
            for names in train_cast_names:
                for name in names:
                    actor_freq[name] = actor_freq.get(name, 0) + 1
            df["top_actor_freq"] = parse_cast_names(df["cast"]).apply(
                lambda x: actor_freq.get(x[0], 0) if x else 0
            )
            df["top_actor2_freq"] = parse_cast_names(df["cast"]).apply(
                lambda x: actor_freq.get(x[1], 0) if len(x) > 1 else 0
            )
            df["top_actor3_freq"] = parse_cast_names(df["cast"]).apply(
                lambda x: actor_freq.get(x[2], 0) if len(x) > 2 else 0
            )
            df["max_actor_freq"] = parse_cast_names(df["cast"]).apply(
                lambda x: max([actor_freq.get(n, 0) for n in x]) if x else 0
            )

        # Director frequency encoding
        if "crew" in df.columns and "crew" in train_df_for_encoding.columns:
            train_directors = parse_crew_director(train_df_for_encoding["crew"])
            director_freq = train_directors.value_counts()
            if "director_name" not in df.columns:
                df["director_name"] = parse_crew_director(df["crew"])
            df["director_freq"] = df["director_name"].map(director_freq).fillna(0)

    # Interaction features
    if "log_budget" in df.columns and "log_popularity" in df.columns:
        df["budget_popularity_interaction"] = df["log_budget"] * df["log_popularity"]
    if "log_budget" in df.columns and "log_runtime" in df.columns:
        df["budget_runtime_interaction"] = df["log_budget"] * df["log_runtime"]
    if "log_popularity" in df.columns and "genre_count" in df.columns:
        df["popularity_genre_interaction"] = df["log_popularity"] * df["genre_count"]
    if "has_collection" in df.columns and "log_budget" in df.columns:
        df["franchise_budget"] = df["has_collection"] * df["log_budget"]

    # Drop original columns
    cols_to_drop = [
        "belongs_to_collection",
        "homepage",
        "imdb_id",
        "original_language",
        "original_title",
        "overview",
        "poster_path",
        "production_companies",
        "production_countries",
        "release_date",
        "genres",
        "cast",
        "crew",
        "keywords",
        "tagline",
        "first_genre",
        "first_company",
        "director_name",
    ]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors="ignore")

    # Robustly drop any remaining non-numeric columns (object, string, etc.)
    # This ensures only numeric features remain for modelling
    non_numeric_cols = [
        c
        for c in df.columns
        if df[c].dtype == "object" or df[c].dtype == "string" or str(df[c].dtype) == "category"
    ]
    if non_numeric_cols:
        df = df.drop(columns=non_numeric_cols, errors="ignore")

    return df


# Apply feature engineering
train_feat = engineer_features(train_split, train_df_for_encoding=train_split)
val_feat = engineer_features(val_split, train_df_for_encoding=train_split)
test_feat = engineer_features(test_df, train_df_for_encoding=train_split)

# Handle missing values
exclude_cols = ["id", "revenue", "log_revenue", "revenue_bin"]
feature_cols = [c for c in train_feat.columns if c not in exclude_cols]

# Safety: ensure all feature columns are numeric
for col in feature_cols:
    if not pd.api.types.is_numeric_dtype(train_feat[col]):
        train_feat[col] = pd.to_numeric(train_feat[col], errors="coerce")
        val_feat[col] = pd.to_numeric(val_feat[col], errors="coerce")
        test_feat[col] = pd.to_numeric(test_feat[col], errors="coerce")

for col in feature_cols:
    train_median = train_feat[col].median()
    if pd.isna(train_median):
        train_median = 0.0
    train_feat[col] = train_feat[col].fillna(train_median)
    val_feat[col] = val_feat[col].fillna(train_median)
    test_feat[col] = test_feat[col].fillna(train_median)

for df_temp in [train_feat, val_feat, test_feat]:
    df_temp.replace([np.inf, -np.inf], 0, inplace=True)

print(f"Final feature count: {len(feature_cols)}")

# ============================================================
# 4. PREPARE FEATURES FOR GRADIENT BOOSTING
# ============================================================
X_train = train_feat[feature_cols].values
X_val = val_feat[feature_cols].values
X_test = test_feat[feature_cols].values

y_train = np.log1p(train_feat["revenue"].values)
y_val = np.log1p(val_feat["revenue"].values)

print(f"Feature matrix shapes: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")
print(f"Target range: [{y_train.min():.2f}, {y_train.max():.2f}]")

# ============================================================
# 5. TRAIN LIGHTGBM MODEL
# ============================================================
print("\n--- Training LightGBM ---")
lgb_model = lgb.LGBMRegressor(
    n_estimators=5000,
    learning_rate=0.005,
    num_leaves=63,
    max_depth=-1,
    min_child_samples=10,
    subsample=0.8,
    colsample_bytree=0.8,
    feature_fraction_bynode=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)

lgb_model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    eval_metric="rmse",
    callbacks=[lgb.early_stopping(200, verbose=False)],
)

lgb_val_pred_log = lgb_model.predict(X_val)
lgb_val_rmsle = np.sqrt(np.mean((y_val - lgb_val_pred_log) ** 2))
lgb_best_iter = lgb_model.best_iteration_
print(f"LightGBM best iteration: {lgb_best_iter}")
print(f"LightGBM Val RMSLE: {lgb_val_rmsle:.6f}")

# ============================================================
# 6. TRAIN XGBOOST MODEL
# ============================================================
print("\n--- Training XGBoost ---")
xgb_model = xgb.XGBRegressor(
    n_estimators=5000,
    learning_rate=0.005,
    max_depth=7,
    min_child_weight=3,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    early_stopping_rounds=200,
)

xgb_model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    verbose=False,
)

xgb_val_pred_log = xgb_model.predict(X_val)
xgb_val_rmsle = np.sqrt(np.mean((y_val - xgb_val_pred_log) ** 2))
xgb_best_iter = xgb_model.best_iteration
print(f"XGBoost best iteration: {xgb_best_iter}")
print(f"XGBoost Val RMSLE: {xgb_val_rmsle:.6f}")

# ============================================================
# 7. TRAIN CATBOOST MODEL
# ============================================================
print("\n--- Training CatBoost ---")
cat_model = CatBoostRegressor(
    iterations=5000,
    learning_rate=0.005,
    depth=7,
    l2_leaf_reg=3,
    subsample=0.8,
    colsample_bylevel=0.8,
    random_seed=42,
    verbose=0,
    early_stopping_rounds=200,
)

cat_model.fit(
    X_train,
    y_train,
    eval_set=(X_val, y_val),
    verbose=0,
)

cat_val_pred_log = cat_model.predict(X_val)
cat_val_rmsle = np.sqrt(np.mean((y_val - cat_val_pred_log) ** 2))
cat_best_iter = cat_model.get_best_iteration()
print(f"CatBoost best iteration: {cat_best_iter}")
print(f"CatBoost Val RMSLE: {cat_val_rmsle:.6f}")

# ============================================================
# 8. WEIGHTED ENSEMBLE PREDICTIONS
# ============================================================
# Weights inversely proportional to each model's validation RMSE
rmsles = np.array([lgb_val_rmsle, xgb_val_rmsle, cat_val_rmsle])
weights = (1.0 / rmsles) / np.sum(1.0 / rmsles)
print(f"\nEnsemble weights: LGB={weights[0]:.4f}, XGB={weights[1]:.4f}, CAT={weights[2]:.4f}")

ensemble_val_pred_log = (
    weights[0] * lgb_val_pred_log
    + weights[1] * xgb_val_pred_log
    + weights[2] * cat_val_pred_log
)
val_rmsle = np.sqrt(np.mean((y_val - ensemble_val_pred_log) ** 2))
print(f"Weighted Ensemble Val RMSLE: {val_rmsle:.6f}")


# ============================================================
# 9. SECOND ROUND: TRAIN WITH FULL DATA (train + val)
# ============================================================
print("\n--- Second Round: Training with full data ---")

# Combine train and val for full training
X_full = np.vstack([X_train, X_val])
y_full = np.concatenate([y_train, y_val])

# LightGBM full
lgb_model_full = lgb.LGBMRegressor(
    n_estimators=lgb_best_iter,
    learning_rate=0.005,
    num_leaves=63,
    max_depth=-1,
    min_child_samples=10,
    subsample=0.8,
    colsample_bytree=0.8,
    feature_fraction_bynode=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)
lgb_model_full.fit(X_full, y_full)

# XGBoost full
xgb_model_full = xgb.XGBRegressor(
    n_estimators=xgb_best_iter,
    learning_rate=0.005,
    max_depth=7,
    min_child_weight=3,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
)
xgb_model_full.fit(X_full, y_full, verbose=False)

# CatBoost full
cat_model_full = CatBoostRegressor(
    iterations=cat_best_iter,
    learning_rate=0.005,
    depth=7,
    l2_leaf_reg=3,
    subsample=0.8,
    colsample_bylevel=0.8,
    random_seed=42,
    verbose=0,
)
cat_model_full.fit(X_full, y_full, verbose=0)

print("Full data models trained.")

# ============================================================
# 10. GENERATE TEST PREDICTIONS
# ============================================================
lgb_test_pred_log = lgb_model_full.predict(X_test)
xgb_test_pred_log = xgb_model_full.predict(X_test)
cat_test_pred_log = cat_model_full.predict(X_test)

# Weighted ensemble for test predictions
test_preds_log = (
    weights[0] * lgb_test_pred_log
    + weights[1] * xgb_test_pred_log
    + weights[2] * cat_test_pred_log
)
test_preds_revenue = np.expm1(test_preds_log)
test_preds_revenue = np.clip(test_preds_revenue, 0, None)

# ============================================================
# 11. SAVE SUBMISSION
# ============================================================
os.makedirs("./submission", exist_ok=True)
submission = pd.DataFrame(
    {"id": range(len(test_preds_revenue)), "revenue": test_preds_revenue}
)
submission.to_csv("./submission/submission.csv", index=False)

print(f"Submission saved: {submission.shape}")
print(
    f"Prediction stats: mean={test_preds_revenue.mean():.0f}, median={np.median(test_preds_revenue):.0f}, max={test_preds_revenue.max():.0f}"
)
print(f"Final Validation Score: {val_rmsle:.6f}")
