import os
import re
import warnings
from catboost import CatBoostRegressor
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold
import xgboost as xgb

warnings.filterwarnings("ignore")

# =========================================================================
# 1. Directory and Data Loading
# =========================================================================
INPUT_DIR = "./input"
WORKING_DIR = "./working"
SUBMISSION_DIR = "./submission"
MODEL_DIR = os.path.join(WORKING_DIR, "models")

os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

train_path = os.path.join(INPUT_DIR, "train.parquet")
test_path = os.path.join(INPUT_DIR, "test.parquet")

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = "Total interior livable area"
id_col = (
    "id" if "id" in train_df.columns else ("Id" if "Id" in train_df.columns else None)
)

# Clean train target: remove missing or non-positive target rows
if target_col in train_df.columns:
    train_df = train_df[
        train_df[target_col].notnull() & (train_df[target_col] > 0)
    ].copy()
    train_df.reset_index(drop=True, inplace=True)
test_df.reset_index(drop=True, inplace=True)


# =========================================================================
# 2. Domain-Specific Feature Extraction Helpers
# =========================================================================
def parse_lot_size(val):
    """Parse lot size strings (e.g., '0.5 Acres', '5,000 sqft') to uniform square footage."""
    if pd.isnull(val):
        return np.nan
    if isinstance(val, (int, float)):
        return float(val)
    val_str = str(val).strip().replace(",", "")
    match = re.search(r"([0-9.]+)\s*([a-zA-Z]*)", val_str)
    if not match:
        return np.nan
    num, unit = match.groups()
    try:
        num = float(num)
    except ValueError:
        return np.nan
    unit = unit.lower()
    if "acre" in unit:
        return num * 43560.0
    return num


def parse_price(val):
    """Parse price/value strings with currency symbols and commas."""
    if pd.isnull(val):
        return np.nan
    if isinstance(val, (int, float)):
        return float(val)
    val_str = str(val).replace("$", "").replace(",", "").strip()
    try:
        return float(val_str)
    except ValueError:
        return np.nan


def parse_numeric(val):
    """Parse general numeric fields that might contain text or symbols."""
    if pd.isnull(val):
        return np.nan
    if isinstance(val, (int, float)):
        return float(val)
    val_str = str(val).replace(",", "").strip()
    match = re.search(r"[-+]?[0-9]*\.?[0-9]+", val_str)
    if match:
        try:
            return float(match.group())
        except ValueError:
            return np.nan
    return np.nan


def extract_keyword_features(series, keywords):
    """Extract binary indicator flags and total counts for a list of domain keywords."""
    clean_series = series.fillna("").astype(str).str.lower()
    prefix = re.sub(r"[^a-zA-Z0-9_]", "_", str(series.name)).lower()
    feat_dict = {}
    counts = np.zeros(len(series), dtype=np.int32)
    for kw in keywords:
        clean_kw = re.sub(r"[^a-zA-Z0-9_]", "_", kw)
        col_name = f"{prefix}_has_{clean_kw}"
        has_kw = clean_series.str.contains(kw, regex=False).astype(np.int32)
        feat_dict[col_name] = has_kw
        counts += has_kw.values
    feat_dict[f"{prefix}_amenity_count"] = counts
    return pd.DataFrame(feat_dict, index=series.index)


def get_listing_text(df):
    """Combine informative text description fields into consolidated string series."""
    text_cols = [
        c
        for c in df.columns
        if df[c].dtype == "object"
        or str(df[c].dtype) == "category"
        or any(k in c.lower() for k in ["summary", "desc", "detail", "about"])
    ]
    combined_text = pd.Series("", index=df.index)
    for col in text_cols:
        s = df[col].fillna("").astype(str)
        if s.str.len().mean() > 3:
            combined_text = combined_text + " " + s
    return combined_text.str.strip()


# =========================================================================
# 3. Base Feature Engineering Pipeline
# =========================================================================
def create_base_features(df):
    """Construct domain-rich tabular, interaction, structural, and amenity features."""
    feats = pd.DataFrame(index=df.index)

    # Store ID if present
    if id_col and id_col in df.columns:
        feats["id"] = df[id_col].values
    elif "id" in df.columns:
        feats["id"] = df["id"].values
    else:
        feats["id"] = df.index.values

    # 1. Parsing numeric / physical area features
    lot_cols = [c for c in df.columns if "lot" in c.lower()]
    if lot_cols:
        feats["lot_size_sqft"] = df[lot_cols[0]].apply(parse_lot_size)

    # Price & Tax values
    price_cols = [
        c
        for c in df.columns
        if any(p in c.lower() for p in ["price", "tax assessed", "annual tax"])
    ]
    for pc in price_cols:
        clean_name = re.sub(r"[^a-zA-Z0-9_]", "_", pc).lower()
        feats[clean_name] = df[pc].apply(parse_price)

    # Vertical Housing Structure Features (Stories & Levels)
    story_cols = [
        c for c in df.columns if any(s in c.lower() for s in ["stori", "level", "floors"])
    ]
    for sc in story_cols:
        clean_name = f'num_{re.sub(r"[^a-zA-Z0-9_]", "_", sc).lower()}'
        feats[clean_name] = df[sc].apply(parse_numeric)

    # Tax structure / improvement values
    tax_struct_cols = [
        c
        for c in df.columns
        if any(
            k in c.lower()
            for k in ["structure", "improvement", "land value", "building value"]
        )
    ]
    for tc in tax_struct_cols:
        clean_name = f'val_{re.sub(r"[^a-zA-Z0-9_]", "_", tc).lower()}'
        feats[clean_name] = df[tc].apply(parse_price)

    # Rooms & Bathrooms
    bed_cols = [c for c in df.columns if "bed" in c.lower()]
    bath_cols = [c for c in df.columns if "bath" in c.lower()]
    full_bath_cols = [
        c for c in df.columns if "full" in c.lower() and "bath" in c.lower()
    ]
    half_bath_cols = [
        c for c in df.columns if "half" in c.lower() and "bath" in c.lower()
    ]

    for c in full_bath_cols:
        feats[f'num_{re.sub(r"[^a-zA-Z0-9_]", "_", c).lower()}'] = df[c].apply(
            parse_numeric
        )
    for c in half_bath_cols:
        feats[f'num_{re.sub(r"[^a-zA-Z0-9_]", "_", c).lower()}'] = df[c].apply(
            parse_numeric
        )

    for c in bed_cols:
        clean_name = f'num_{re.sub(r"[^a-zA-Z0-9_]", "_", c).lower()}'
        feats[clean_name] = df[c].apply(parse_numeric)
    for c in bath_cols:
        clean_name = f'num_{re.sub(r"[^a-zA-Z0-9_]", "_", c).lower()}'
        feats[clean_name] = df[c].apply(parse_numeric)

    primary_beds = (
        feats[f'num_{re.sub(r"[^a-zA-Z0-9_]", "_", bed_cols[0]).lower()}']
        if bed_cols
        else pd.Series(np.nan, index=df.index)
    )
    primary_baths = (
        feats[f'num_{re.sub(r"[^a-zA-Z0-9_]", "_", bath_cols[0]).lower()}']
        if bath_cols
        else pd.Series(np.nan, index=df.index)
    )

    feats["est_total_rooms"] = primary_beds.fillna(0) + primary_baths.fillna(0)
    feats["bed_bath_ratio"] = (primary_beds + 1e-4) / (primary_baths + 1e-4)
    feats["bed_bath_mult"] = primary_beds.fillna(0) * primary_baths.fillna(0)
    feats["bed_squared"] = (primary_beds.fillna(0)) ** 2
    feats["bath_squared"] = (primary_baths.fillna(0)) ** 2

    # Vertical story interaction ratios
    story_feat_cols = [
        c
        for c in feats.columns
        if c.startswith("num_")
        and any(s in c for s in ["stori", "level", "floor"])
    ]
    if story_feat_cols:
        primary_story_val = feats[story_feat_cols[0]].fillna(1.0).clip(lower=1.0)
        feats["rooms_per_story"] = feats["est_total_rooms"] / primary_story_val
        feats["beds_per_story"] = primary_beds.fillna(0) / primary_story_val
    else:
        feats["rooms_per_story"] = feats["est_total_rooms"]
        feats["beds_per_story"] = primary_beds.fillna(0)

    # Tax assessment structure-to-land ratio
    struct_val_cols = [
        c
        for c in feats.columns
        if c.startswith("val_")
        and any(k in c for k in ["struct", "improv", "build"])
    ]
    land_val_cols = [
        c for c in feats.columns if c.startswith("val_") and "land" in c
    ]
    if struct_val_cols and land_val_cols:
        feats["struct_to_land_tax_ratio"] = (
            feats[struct_val_cols[0]].fillna(0) + 1.0
        ) / (feats[land_val_cols[0]].fillna(0) + 1.0)
    elif struct_val_cols:
        feats["struct_to_land_tax_ratio"] = feats[struct_val_cols[0]].fillna(0)
    else:
        feats["struct_to_land_tax_ratio"] = 0.0

    # Lot coverage proxy
    if "lot_size_sqft" in feats.columns:
        feats["lot_coverage_proxy"] = (
            primary_beds.fillna(1.0) * 350.0
        ) / (feats["lot_size_sqft"].fillna(5000.0) + 10.0)

    # Garage & Parking spaces
    parking_cols = [
        c
        for c in df.columns
        if any(p in c.lower() for p in ["garage", "parking", "spaces", "carport"])
    ]
    for pc in parking_cols:
        clean_name = f'parking_{re.sub(r"[^a-zA-Z0-9_]", "_", pc).lower()}'
        feats[clean_name] = df[pc].apply(parse_numeric)

    # Year Built & Age calculations
    year_cols = [c for c in df.columns if "year" in c.lower() and "built" in c.lower()]
    if year_cols:
        year_built = df[year_cols[0]].apply(parse_numeric)
        feats["year_built"] = year_built
        feats["property_age"] = 2024.0 - year_built
        feats["is_new_construction"] = (feats["property_age"] <= 5).astype(np.int32)
        feats["is_historic"] = (feats["property_age"] >= 70).astype(np.int32)
        feats["decade_built"] = (year_built // 10) * 10

    # School ratings & distances
    school_score_cols = [
        c for c in df.columns if "score" in c.lower() or "rating" in c.lower()
    ]
    school_dist_cols = [c for c in df.columns if "distance" in c.lower()]

    for sc in school_score_cols:
        feats[f'school_score_{re.sub(r"[^a-zA-Z0-9_]", "_", sc).lower()}'] = df[
            sc
        ].apply(parse_numeric)
    for dc in school_dist_cols:
        feats[f'school_dist_{re.sub(r"[^a-zA-Z0-9_]", "_", dc).lower()}'] = df[
            dc
        ].apply(parse_numeric)

    score_feats = [c for c in feats.columns if c.startswith("school_score_")]
    if score_feats:
        feats["school_score_mean"] = feats[score_feats].mean(axis=1)
        feats["school_score_max"] = feats[score_feats].max(axis=1)
        feats["school_score_min"] = feats[score_feats].min(axis=1)
        feats["school_score_std"] = feats[score_feats].std(axis=1).fillna(0)

    dist_feats = [c for c in feats.columns if c.startswith("school_dist_")]
    if dist_feats:
        feats["school_dist_mean"] = feats[dist_feats].mean(axis=1)
        feats["school_dist_min"] = feats[dist_feats].min(axis=1)

    # 2. Financial & Valuation Ratios
    listed_price_cols = [
        c for c in feats.columns if "price" in c and "tax" not in c and "bed" not in c
    ]

    if listed_price_cols and bed_cols:
        feats["price_per_bedroom"] = feats[listed_price_cols[0]] / (
            primary_beds.fillna(0) + 1.0
        )
    if listed_price_cols and bath_cols:
        feats["price_per_bathroom"] = feats[listed_price_cols[0]] / (
            primary_baths.fillna(0) + 1.0
        )
    if "lot_size_sqft" in feats.columns and listed_price_cols:
        feats["price_per_lot_sqft"] = feats[listed_price_cols[0]] / (
            feats["lot_size_sqft"].fillna(0) + 10.0
        )

    # 3. Amenity and Room Space Text Parsing
    amenity_keywords = [
        "pool",
        "spa",
        "hardwood",
        "central air",
        "cooling",
        "fireplace",
        "granite",
        "stainless",
        "solar",
        "view",
        "ocean",
        "mountain",
        "patio",
        "balcony",
        "deck",
        "porch",
        "remodeled",
        "renovated",
        "upgraded",
        "gated",
        "adu",
        "guest house",
        "casita",
        "bonus room",
        "loft",
        "den",
        "office",
        "study",
        "master suite",
        "primary suite",
        "finished basement",
        "basement",
        "attic",
        "in-law",
        "walk-in closet",
        "vaulted ceiling",
        "open floor",
        "dining room",
        "family room",
        "living room",
        "workshop",
        "storage",
    ]
    text_cols = [
        c
        for c in df.columns
        if df[c].dtype == "object" or str(df[c].dtype) == "category"
    ]
    for tc in text_cols:
        sample_vals = df[tc].dropna().astype(str).head(100)
        if any(len(str(v)) > 15 for v in sample_vals):
            kw_df = extract_keyword_features(df[tc], amenity_keywords)
            feats = pd.concat([feats, kw_df], axis=1)

    # 4. Categorical Categorization
    cat_columns = [
        c
        for c in df.columns
        if c
        in [
            "City",
            "State",
            "Zip",
            "County",
            "Home type",
            "Property condition",
            "Elementary School",
            "Middle School",
            "High School",
            "Heating",
            "Cooling",
            "Flooring",
        ]
        or (df[c].dtype == "object" and df[c].nunique() < 500)
    ]

    for cc in set(cat_columns):
        if cc in df.columns:
            clean_name = f'cat_{re.sub(r"[^a-zA-Z0-9_]", "_", cc).lower()}'
            feats[clean_name] = df[cc].fillna("Missing").astype(str)

    feats = feats.loc[:, ~feats.columns.duplicated()]
    return feats


# =========================================================================
# 4. Build Base Feature Sets & Out-of-Fold Encoding
# =========================================================================
X_train_raw = create_base_features(train_df)
X_test_raw = create_base_features(test_df)
y_train = train_df[target_col].values.astype(np.float64)

N_SPLITS = 5
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

X_train_processed = X_train_raw.copy()
X_test_processed = X_test_raw.copy()

# Construct compound spatial-type interaction keys
if "cat_zip" in X_train_processed.columns and "cat_home_type" in X_train_processed.columns:
    X_train_processed["cat_zip_home_type"] = (
        X_train_processed["cat_zip"].astype(str) + "___" + X_train_processed["cat_home_type"].astype(str)
    )
    X_test_processed["cat_zip_home_type"] = (
        X_test_processed["cat_zip"].astype(str) + "___" + X_test_processed["cat_home_type"].astype(str)
    )

if "cat_city" in X_train_processed.columns and "cat_home_type" in X_train_processed.columns:
    X_train_processed["cat_city_home_type"] = (
        X_train_processed["cat_city"].astype(str) + "___" + X_train_processed["cat_home_type"].astype(str)
    )
    X_test_processed["cat_city_home_type"] = (
        X_test_processed["cat_city"].astype(str) + "___" + X_test_processed["cat_home_type"].astype(str)
    )

cat_cols = [c for c in X_train_processed.columns if c.startswith("cat_")]

# Frequency Encoding
for col in cat_cols:
    freq_map = X_train_processed[col].value_counts(normalize=True).to_dict()
    X_train_processed[f"{col}_freq"] = (
        X_train_processed[col].map(freq_map).fillna(0).astype(np.float32)
    )
    X_test_processed[f"{col}_freq"] = (
        X_test_processed[col].map(freq_map).fillna(0).astype(np.float32)
    )

# Out-of-fold Smoothed Bayesian Target Encoding
target_enc_cols = [
    c
    for c in cat_cols
    if any(k in c for k in ["city", "zip", "county", "home_type", "school"])
]
global_target_mean = float(np.mean(y_train))
SMOOTHING = 10.0

for col in target_enc_cols:
    oof_target_enc = np.full(
        len(X_train_processed), global_target_mean, dtype=np.float32
    )

    for tr_idx, val_idx in kf.split(X_train_processed):
        tr_df = X_train_processed.iloc[tr_idx]
        tr_y = y_train[tr_idx]

        stats = (
            pd.DataFrame({"cat": tr_df[col], "target": tr_y})
            .groupby("cat", observed=False)["target"]
            .agg(["count", "mean"])
        )
        smoothed = (stats["count"] * stats["mean"] + SMOOTHING * global_target_mean) / (
            stats["count"] + SMOOTHING
        )
        enc_map = smoothed.to_dict()

        oof_target_enc[val_idx] = (
            X_train_processed.iloc[val_idx][col]
            .map(enc_map)
            .fillna(global_target_mean)
            .values
        )

    X_train_processed[f"{col}_target_enc"] = oof_target_enc

    full_stats = (
        pd.DataFrame({"cat": X_train_processed[col], "target": y_train})
        .groupby("cat", observed=False)["target"]
        .agg(["count", "mean"])
    )
    full_smoothed = (
        full_stats["count"] * full_stats["mean"] + SMOOTHING * global_target_mean
    ) / (full_stats["count"] + SMOOTHING)
    full_enc_map = full_smoothed.to_dict()
    X_test_processed[f"{col}_target_enc"] = (
        X_test_processed[col]
        .map(full_enc_map)
        .fillna(global_target_mean)
        .astype(np.float32)
    )

# Multi-feature geographic & spatial compound aggregations
agg_num_cols = [
    c
    for c in [
        "lot_size_sqft",
        "est_total_rooms",
        "rooms_per_story",
        "price_per_bedroom",
        "price_per_bathroom",
        "property_age",
    ]
    if c in X_train_processed.columns
]
for loc_key in ["cat_zip", "cat_city", "cat_zip_home_type", "cat_city_home_type"]:
    if loc_key in cat_cols:
        for num_c in agg_num_cols:
            stats_df = (
                X_train_processed.groupby(loc_key, observed=False)[num_c]
                .agg(mean="mean", median="median", std="std")
                .reset_index()
            )
            mean_map = dict(zip(stats_df[loc_key], stats_df["mean"]))
            median_map = dict(zip(stats_df[loc_key], stats_df["median"]))
            std_map = dict(zip(stats_df[loc_key], stats_df["std"]))

            X_train_processed[f"{loc_key}_{num_c}_mean"] = (
                X_train_processed[loc_key].map(mean_map).astype(np.float32)
            )
            X_train_processed[f"{loc_key}_{num_c}_median"] = (
                X_train_processed[loc_key].map(median_map).astype(np.float32)
            )
            X_train_processed[f"{loc_key}_{num_c}_std"] = (
                X_train_processed[loc_key].map(std_map).astype(np.float32)
            )

            X_test_processed[f"{loc_key}_{num_c}_mean"] = (
                X_test_processed[loc_key].map(mean_map).astype(np.float32)
            )
            X_test_processed[f"{loc_key}_{num_c}_median"] = (
                X_test_processed[loc_key].map(median_map).astype(np.float32)
            )
            X_test_processed[f"{loc_key}_{num_c}_std"] = (
                X_test_processed[loc_key].map(std_map).astype(np.float32)
            )

# Text semantic decomposition (TF-IDF + SVD) & Out-of-fold Text Ridge Meta-Feature
train_text = get_listing_text(train_df)
test_text = get_listing_text(test_df)

tfidf = TfidfVectorizer(
    ngram_range=(1, 2), max_features=10000, min_df=3, stop_words="english"
)
train_tfidf = tfidf.fit_transform(train_text)
test_tfidf = tfidf.transform(test_text)

svd = TruncatedSVD(n_components=25, random_state=42)
train_svd = svd.fit_transform(train_tfidf)
test_svd = svd.transform(test_tfidf)

for i in range(25):
    X_train_processed[f"svd_text_{i}"] = train_svd[:, i].astype(np.float32)
    X_test_processed[f"svd_text_{i}"] = test_svd[:, i].astype(np.float32)

oof_text_ridge = np.zeros(len(X_train_processed), dtype=np.float32)
test_text_ridge = np.zeros(len(X_test_processed), dtype=np.float32)

for tr_idx, val_idx in kf.split(X_train_processed):
    ridge_model = Ridge(alpha=10.0, random_state=42)
    ridge_model.fit(train_tfidf[tr_idx], y_train[tr_idx])
    oof_text_ridge[val_idx] = ridge_model.predict(train_tfidf[val_idx])
    test_text_ridge += ridge_model.predict(test_tfidf) / N_SPLITS

X_train_processed["ridge_text_pred"] = oof_text_ridge
X_test_processed["ridge_text_pred"] = test_text_ridge

# Assign CV Folds to training data
X_train_processed["fold"] = -1
for fold_num, (tr_idx, val_idx) in enumerate(kf.split(X_train_processed)):
    X_train_processed.loc[val_idx, "fold"] = fold_num

X_train_processed["target"] = y_train

exclude_cols = {"id", "Id", "ID", "fold", "target"}
feature_cols = [c for c in X_train_processed.columns if c not in exclude_cols]

# Align categorical dtypes across train and test for models
for col in feature_cols:
    if (
        X_train_processed[col].dtype == "object"
        or X_test_processed[col].dtype == "object"
    ):
        combined = pd.concat([X_train_processed[col], X_test_processed[col]]).dropna()
        cat_type = pd.CategoricalDtype(categories=combined.unique())
        X_train_processed[col] = X_train_processed[col].astype(cat_type)
        X_test_processed[col] = X_test_processed[col].astype(cat_type)


# =========================================================================
# 5. Model Hyperparameters & Training Loop
# =========================================================================
def evaluate_rmse(y_true, y_pred):
    """Compute Root Mean Squared Error."""
    return root_mean_squared_error(y_true, y_pred)


lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.03,
    "num_leaves": 127,
    "max_depth": 9,
    "min_child_samples": 25,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.75,
    "reg_alpha": 0.5,
    "reg_lambda": 2.0,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}

xgb_params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "tree_method": "hist",
    "learning_rate": 0.03,
    "max_depth": 8,
    "min_child_weight": 5,
    "subsample": 0.75,
    "colsample_bytree": 0.75,
    "reg_alpha": 0.5,
    "reg_lambda": 3.0,
    "random_state": 42,
    "n_estimators": 3000,
    "n_jobs": -1,
    "enable_categorical": True,
}

cb_params = {
    "iterations": 3000,
    "learning_rate": 0.03,
    "depth": 7,
    "l2_leaf_reg": 5.0,
    "random_strength": 0.5,
    "bagging_temperature": 0.2,
    "loss_function": "RMSE",
    "eval_metric": "RMSE",
    "random_seed": 42,
    "verbose": False,
}

n_train = len(X_train_processed)
n_test = len(X_test_processed)

oof_preds_lgb = np.zeros(n_train, dtype=np.float64)
oof_preds_xgb = np.zeros(n_train, dtype=np.float64)
oof_preds_cb = np.zeros(n_train, dtype=np.float64)

test_preds_lgb = np.zeros(n_test, dtype=np.float64)
test_preds_xgb = np.zeros(n_test, dtype=np.float64)
test_preds_cb = np.zeros(n_test, dtype=np.float64)

cb_cat_cols = [
    c
    for c in feature_cols
    if isinstance(X_train_processed[c].dtype, pd.CategoricalDtype)
]

for fold in range(N_SPLITS):
    val_mask = X_train_processed["fold"] == fold
    train_mask = ~val_mask

    X_tr = X_train_processed.loc[train_mask, feature_cols]
    y_tr = y_train[train_mask]
    X_va = X_train_processed.loc[val_mask, feature_cols]
    y_va = y_train[val_mask]

    # 1. LightGBM Training
    lgb_train_data = lgb.Dataset(X_tr, label=y_tr)
    lgb_val_data = lgb.Dataset(X_va, label=y_va, reference=lgb_train_data)

    lgb_model = lgb.train(
        params=lgb_params,
        train_set=lgb_train_data,
        num_boost_round=3000,
        valid_sets=[lgb_val_data],
        valid_names=["val"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )

    val_pred_lgb = lgb_model.predict(X_va, num_iteration=lgb_model.best_iteration)
    oof_preds_lgb[val_mask] = val_pred_lgb
    test_preds_lgb += (
        lgb_model.predict(
            X_test_processed[feature_cols], num_iteration=lgb_model.best_iteration
        )
        / N_SPLITS
    )
    joblib.dump(lgb_model, os.path.join(MODEL_DIR, f"lgb_model_fold_{fold}.pkl"))

    # 2. XGBoost Training
    xgb_model = xgb.XGBRegressor(
        **xgb_params,
        early_stopping_rounds=50,
    )
    xgb_model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        verbose=False,
    )

    val_pred_xgb = xgb_model.predict(X_va)
    oof_preds_xgb[val_mask] = val_pred_xgb
    test_preds_xgb += xgb_model.predict(X_test_processed[feature_cols]) / N_SPLITS
    joblib.dump(xgb_model, os.path.join(MODEL_DIR, f"xgb_model_fold_{fold}.pkl"))

    # 3. CatBoost Training
    cb_model = CatBoostRegressor(
        **cb_params,
        early_stopping_rounds=50,
        cat_features=cb_cat_cols if cb_cat_cols else None,
    )
    cb_model.fit(
        X_tr,
        y_tr,
        eval_set=(X_va, y_va),
        verbose=False,
    )

    val_pred_cb = cb_model.predict(X_va)
    oof_preds_cb[val_mask] = val_pred_cb
    test_preds_cb += (
        cb_model.predict(X_test_processed[feature_cols]) / N_SPLITS
    )
    joblib.dump(cb_model, os.path.join(MODEL_DIR, f"cb_model_fold_{fold}.pkl"))

    fold_lgb_rmse = evaluate_rmse(y_va, val_pred_lgb)
    fold_xgb_rmse = evaluate_rmse(y_va, val_pred_xgb)
    fold_cb_rmse = evaluate_rmse(y_va, val_pred_cb)
    print(
        f"Fold {fold + 1}/{N_SPLITS} | LightGBM RMSE: {fold_lgb_rmse:.4f} | XGBoost RMSE: {fold_xgb_rmse:.4f} | CatBoost RMSE: {fold_cb_rmse:.4f}"
    )

# =========================================================================
# 6. Model Blending and Validation Scoring
# =========================================================================
lgb_overall_rmse = evaluate_rmse(y_train, oof_preds_lgb)
xgb_overall_rmse = evaluate_rmse(y_train, oof_preds_xgb)
cb_overall_rmse = evaluate_rmse(y_train, oof_preds_cb)
print(f"Overall LightGBM OOF RMSE: {lgb_overall_rmse:.4f}")
print(f"Overall XGBoost OOF RMSE:  {xgb_overall_rmse:.4f}")
print(f"Overall CatBoost OOF RMSE: {cb_overall_rmse:.4f}")


def ensemble_obj(weights):
    w = np.array(weights)
    pred = w[0] * oof_preds_lgb + w[1] * oof_preds_xgb + w[2] * oof_preds_cb
    return evaluate_rmse(y_train, pred)


init_weights = [1.0 / 3, 1.0 / 3, 1.0 / 3]
bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]
constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

opt_res = minimize(
    ensemble_obj,
    init_weights,
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
)
best_w = opt_res.x
best_w = np.maximum(best_w, 0.0)
best_w = best_w / np.sum(best_w)

final_oof_preds = (
    best_w[0] * oof_preds_lgb + best_w[1] * oof_preds_xgb + best_w[2] * oof_preds_cb
)
final_score = evaluate_rmse(y_train, final_oof_preds)

print(
    f"Optimal Weights: LightGBM = {best_w[0]:.4f}, XGBoost = {best_w[1]:.4f}, CatBoost = {best_w[2]:.4f}"
)

# =========================================================================
# 7. Generate Final Test Predictions and Submission File
# =========================================================================
final_test_preds = (
    best_w[0] * test_preds_lgb
    + best_w[1] * test_preds_xgb
    + best_w[2] * test_preds_cb
)

# Clip predictions to valid positive domain
min_train_area = float(np.percentile(y_train, 0.1))
final_test_preds = np.clip(final_test_preds, a_min=min_train_area, a_max=None)

submission_df = pd.DataFrame(
    {
        "id": np.arange(len(final_test_preds), dtype=int),
        "Total interior livable area": final_test_preds,
    }
)

submission_path = os.path.join(SUBMISSION_DIR, "submission.csv")
submission_df.to_csv(submission_path, index=False)

print(f"Submission saved to {submission_path} with shape {submission_df.shape}")
print(f"Final Validation Score: {final_score}")