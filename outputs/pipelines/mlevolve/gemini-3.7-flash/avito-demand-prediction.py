import gc
import os
import re
import string
from catboost import CatBoostRegressor
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, train_test_split

# Set random seed for reproducibility
SEED = 42
np.random.seed(SEED)

print("Starting Avito Demand Prediction pipeline...")

# Ensure required directories exist
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# ---------------------------------------------------------
# 1. Load Data
# ---------------------------------------------------------
train_path = "./input/train.parquet"
test_path = "./input/test.parquet"

print(f"Loading datasets from {train_path} and {test_path}...")
df_train = pd.read_parquet(train_path)
df_test = pd.read_parquet(test_path)

print(f"Train raw shape: {df_train.shape}, Test raw shape: {df_test.shape}")

# ---------------------------------------------------------
# 2. Strict Train/Validation Split (Leak-Free Protocol)
# ---------------------------------------------------------
target_bins = pd.qcut(
    df_train["deal_probability"], q=10, labels=False, duplicates="drop"
)
train_indices, val_indices = train_test_split(
    df_train.index, test_size=0.15, random_state=SEED, stratify=target_bins
)

train_df = df_train.loc[train_indices].copy().reset_index(drop=True)
val_df = df_train.loc[val_indices].copy().reset_index(drop=True)
test_df = df_test.copy().reset_index(drop=True)

del df_train, df_test, target_bins
gc.collect()

print(
    f"Split completed: Train={train_df.shape[0]} rows, Validation={val_df.shape[0]} rows, Test={test_df.shape[0]} rows"
)

# ---------------------------------------------------------
# 3. Text Feature Engineering & Listing Completeness
# ---------------------------------------------------------
# Listing completeness: missingness counts across structural fields
missing_check_cols = ["param_1", "param_2", "param_3", "description", "price", "image", "image_top_1"]
for df in [train_df, val_df, test_df]:
    df["null_count"] = df[missing_check_cols].isna().sum(axis=1).astype(np.float32)

text_cols = ["title", "description"]

for col in text_cols:
    train_df[col] = train_df[col].fillna("").astype(str)
    val_df[col] = val_df[col].fillna("").astype(str)
    test_df[col] = test_df[col].fillna("").astype(str)

# Construct unified listing text representation
for df in [train_df, val_df, test_df]:
    df["full_text"] = (
        df["title"] + " " +
        df["param_1"].fillna("").astype(str) + " " +
        df["param_2"].fillna("").astype(str) + " " +
        df["param_3"].fillna("").astype(str) + " " +
        df["description"]
    )


def extract_text_stats(df, col):
    """Extract structural and syntactic metadata from text columns."""
    stats = pd.DataFrame(index=df.index)
    s = df[col].astype(str)

    char_len = s.str.len()
    word_count = s.str.count(r"\S+").clip(lower=0)

    stats[f"{col}_char_len"] = char_len.astype(np.float32)
    stats[f"{col}_word_count"] = word_count.astype(np.float32)
    stats[f"{col}_avg_word_len"] = (char_len / (word_count + 1e-5)).astype(np.float32)

    upper_chars = s.str.findall(r"[A-ZА-ЯЁ]").str.len()
    stats[f"{col}_upper_count"] = upper_chars.astype(np.float32)
    stats[f"{col}_upper_ratio"] = (upper_chars / (char_len + 1e-5)).astype(np.float32)

    digit_count = s.str.count(r"\d")
    stats[f"{col}_digit_count"] = digit_count.astype(np.float32)
    stats[f"{col}_digit_ratio"] = (digit_count / (char_len + 1e-5)).astype(np.float32)

    punct_count = s.str.count(r"[^\w\s]")
    stats[f"{col}_punct_count"] = punct_count.astype(np.float32)
    stats[f"{col}_punct_ratio"] = (punct_count / (char_len + 1e-5)).astype(np.float32)
    stats[f"{col}_newline_count"] = s.str.count(r"\n").astype(np.float32)
    stats[f"{col}_exclamation_count"] = s.str.count("!").astype(np.float32)
    stats[f"{col}_question_count"] = s.str.count(r"\?").astype(np.float32)

    return stats


print("Computing text statistical features...")
train_text_stats = pd.concat(
    [
        extract_text_stats(train_df, "title"),
        extract_text_stats(train_df, "description"),
    ],
    axis=1,
)
val_text_stats = pd.concat(
    [extract_text_stats(val_df, "title"), extract_text_stats(val_df, "description")],
    axis=1,
)
test_text_stats = pd.concat(
    [extract_text_stats(test_df, "title"), extract_text_stats(test_df, "description")],
    axis=1,
)

# Cross-text ratios & patterns
for stats_df in [train_text_stats, val_text_stats, test_text_stats]:
    stats_df["title_desc_char_ratio"] = (
        stats_df["title_char_len"] / (stats_df["description_char_len"] + 1e-5)
    ).astype(np.float32)
    stats_df["title_desc_word_ratio"] = (
        stats_df["title_word_count"] / (stats_df["description_word_count"] + 1e-5)
    ).astype(np.float32)
    stats_df["desc_digit_symbol_ratio"] = (
        (stats_df["description_digit_count"] + stats_df["description_punct_count"])
        / (stats_df["description_char_len"] + 1e-5)
    ).astype(np.float32)


def compute_text_overlap(df):
    """Compute word-level overlap and Jaccard similarity between title and description."""
    titles = df["title"].astype(str).str.lower().str.findall(r"\w+")
    descs = df["description"].astype(str).str.lower().str.findall(r"\w+")

    jaccard_list = []
    overlap_count_list = []
    title_overlap_ratio_list = []

    for t_words, d_words in zip(titles, descs):
        s_t = set(t_words)
        s_d = set(d_words)
        intersect = len(s_t & s_d)
        union = len(s_t | s_d)
        jaccard = intersect / union if union > 0 else 0.0
        t_ratio = intersect / len(s_t) if len(s_t) > 0 else 0.0

        jaccard_list.append(jaccard)
        overlap_count_list.append(intersect)
        title_overlap_ratio_list.append(t_ratio)

    return pd.DataFrame(
        {
            "title_desc_jaccard": np.array(jaccard_list, dtype=np.float32),
            "title_desc_overlap_count": np.array(overlap_count_list, dtype=np.float32),
            "title_desc_overlap_ratio": np.array(title_overlap_ratio_list, dtype=np.float32),
        },
        index=df.index,
    )


print("Computing title-description Jaccard overlap features...")
train_overlap_df = compute_text_overlap(train_df)
val_overlap_df = compute_text_overlap(val_df)
test_overlap_df = compute_text_overlap(test_df)

train_text_stats = pd.concat([train_text_stats, train_overlap_df], axis=1)
val_text_stats = pd.concat([val_text_stats, val_overlap_df], axis=1)
test_text_stats = pd.concat([test_text_stats, test_overlap_df], axis=1)

del train_overlap_df, val_overlap_df, test_overlap_df
gc.collect()

print("Extracting high-dimensional TF-IDF, SVD, and OOF Ridge text meta-features...")
# 1. Title TF-IDF & SVD & Ridge
tfidf_title = TfidfVectorizer(
    max_features=50000,
    ngram_range=(1, 2),
    min_df=3,
    sublinear_tf=True,
    token_pattern=r"(?u)\b\w+\b",
)
train_title_tfidf = tfidf_title.fit_transform(train_df["title"])
val_title_tfidf = tfidf_title.transform(val_df["title"])
test_title_tfidf = tfidf_title.transform(test_df["title"])

svd_title = TruncatedSVD(n_components=16, algorithm="randomized", random_state=SEED)
train_title_svd = svd_title.fit_transform(train_title_tfidf).astype(np.float32)
val_title_svd = svd_title.transform(val_title_tfidf).astype(np.float32)
test_title_svd = svd_title.transform(test_title_tfidf).astype(np.float32)

title_svd_cols = [f"svd_title_{i}" for i in range(16)]
train_title_svd_df = pd.DataFrame(train_title_svd, columns=title_svd_cols, index=train_df.index)
val_title_svd_df = pd.DataFrame(val_title_svd, columns=title_svd_cols, index=val_df.index)
test_title_svd_df = pd.DataFrame(test_title_svd, columns=title_svd_cols, index=test_df.index)

del svd_title, train_title_svd, val_title_svd, test_title_svd
gc.collect()

print("Generating Out-Of-Fold Ridge predictions for title...")
oof_ridge_title_train = np.zeros(len(train_df), dtype=np.float32)
oof_ridge_title_val = np.zeros(len(val_df), dtype=np.float32)
oof_ridge_title_test = np.zeros(len(test_df), dtype=np.float32)

kf_text = KFold(n_splits=5, shuffle=True, random_state=SEED)
y_train_target = train_df["deal_probability"].values.astype(np.float32)

for tr_idx, oof_idx in kf_text.split(train_df):
    ridge = Ridge(alpha=10.0, solver="auto", random_state=SEED)
    ridge.fit(train_title_tfidf[tr_idx], y_train_target[tr_idx])
    oof_ridge_title_train[oof_idx] = ridge.predict(train_title_tfidf[oof_idx])
    oof_ridge_title_val += ridge.predict(val_title_tfidf) / 5.0
    oof_ridge_title_test += ridge.predict(test_title_tfidf) / 5.0

del tfidf_title, train_title_tfidf, val_title_tfidf, test_title_tfidf
gc.collect()

# 2. Description TF-IDF & SVD & Ridge
tfidf_desc = TfidfVectorizer(
    max_features=75000,
    ngram_range=(1, 2),
    min_df=3,
    sublinear_tf=True,
    token_pattern=r"(?u)\b\w+\b",
)
train_desc_tfidf = tfidf_desc.fit_transform(train_df["description"])
val_desc_tfidf = tfidf_desc.transform(val_df["description"])
test_desc_tfidf = tfidf_desc.transform(test_df["description"])

svd_desc = TruncatedSVD(n_components=16, algorithm="randomized", random_state=SEED)
train_desc_svd = svd_desc.fit_transform(train_desc_tfidf).astype(np.float32)
val_desc_svd = svd_desc.transform(val_desc_tfidf).astype(np.float32)
test_desc_svd = svd_desc.transform(test_desc_tfidf).astype(np.float32)

desc_svd_cols = [f"svd_desc_{i}" for i in range(16)]
train_desc_svd_df = pd.DataFrame(train_desc_svd, columns=desc_svd_cols, index=train_df.index)
val_desc_svd_df = pd.DataFrame(val_desc_svd, columns=desc_svd_cols, index=val_df.index)
test_desc_svd_df = pd.DataFrame(test_desc_svd, columns=desc_svd_cols, index=test_df.index)

del svd_desc, train_desc_svd, val_desc_svd, test_desc_svd
gc.collect()

print("Generating Out-Of-Fold Ridge predictions for description...")
oof_ridge_desc_train = np.zeros(len(train_df), dtype=np.float32)
oof_ridge_desc_val = np.zeros(len(val_df), dtype=np.float32)
oof_ridge_desc_test = np.zeros(len(test_df), dtype=np.float32)

for tr_idx, oof_idx in kf_text.split(train_df):
    ridge = Ridge(alpha=10.0, solver="auto", random_state=SEED)
    ridge.fit(train_desc_tfidf[tr_idx], y_train_target[tr_idx])
    oof_ridge_desc_train[oof_idx] = ridge.predict(train_desc_tfidf[oof_idx])
    oof_ridge_desc_val += ridge.predict(val_desc_tfidf) / 5.0
    oof_ridge_desc_test += ridge.predict(test_desc_tfidf) / 5.0

del tfidf_desc, train_desc_tfidf, val_desc_tfidf, test_desc_tfidf
gc.collect()

print("Generating Out-Of-Fold Ridge predictions for unified full text...")
tfidf_full = TfidfVectorizer(
    max_features=100000,
    ngram_range=(1, 2),
    min_df=3,
    sublinear_tf=True,
    token_pattern=r"(?u)\b\w+\b",
)
train_full_tfidf = tfidf_full.fit_transform(train_df["full_text"])
val_full_tfidf = tfidf_full.transform(val_df["full_text"])
test_full_tfidf = tfidf_full.transform(test_df["full_text"])

oof_ridge_full_train = np.zeros(len(train_df), dtype=np.float32)
oof_ridge_full_val = np.zeros(len(val_df), dtype=np.float32)
oof_ridge_full_test = np.zeros(len(test_df), dtype=np.float32)

for tr_idx, oof_idx in kf_text.split(train_df):
    ridge = Ridge(alpha=10.0, solver="auto", random_state=SEED)
    ridge.fit(train_full_tfidf[tr_idx], y_train_target[tr_idx])
    oof_ridge_full_train[oof_idx] = ridge.predict(train_full_tfidf[oof_idx])
    oof_ridge_full_val += ridge.predict(val_full_tfidf) / 5.0
    oof_ridge_full_test += ridge.predict(test_full_tfidf) / 5.0

del tfidf_full, train_full_tfidf, val_full_tfidf, test_full_tfidf
gc.collect()

train_ridge_df = pd.DataFrame(
    {
        "ridge_pred_title": oof_ridge_title_train,
        "ridge_pred_desc": oof_ridge_desc_train,
        "ridge_pred_full": oof_ridge_full_train,
    },
    index=train_df.index,
)
val_ridge_df = pd.DataFrame(
    {
        "ridge_pred_title": oof_ridge_title_val,
        "ridge_pred_desc": oof_ridge_desc_val,
        "ridge_pred_full": oof_ridge_full_val,
    },
    index=val_df.index,
)
test_ridge_df = pd.DataFrame(
    {
        "ridge_pred_title": oof_ridge_title_test,
        "ridge_pred_desc": oof_ridge_desc_test,
        "ridge_pred_full": oof_ridge_full_test,
    },
    index=test_df.index,
)

train_svd_df = pd.concat([train_title_svd_df, train_desc_svd_df, train_ridge_df], axis=1)
val_svd_df = pd.concat([val_title_svd_df, val_desc_svd_df, val_ridge_df], axis=1)
test_svd_df = pd.concat([test_title_svd_df, test_desc_svd_df, test_ridge_df], axis=1)

del train_title_svd_df, val_title_svd_df, test_title_svd_df, train_desc_svd_df, val_desc_svd_df, test_desc_svd_df
del train_ridge_df, val_ridge_df, test_ridge_df
gc.collect()

# ---------------------------------------------------------
# 4. Temporal Features
# ---------------------------------------------------------
print("Extracting temporal features...")
for df in [train_df, val_df, test_df]:
    df["activation_date"] = pd.to_datetime(df["activation_date"])
    df["day_of_week"] = df["activation_date"].dt.dayofweek.astype(np.int8)
    df["day_of_month"] = df["activation_date"].dt.day.astype(np.int8)
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(np.int8)

# ---------------------------------------------------------
# 5. Price & Numerical Engineering
# ---------------------------------------------------------
print("Engineering price & category relative metrics...")
for df in [train_df, val_df, test_df]:
    df["price_is_nan"] = df["price"].isna().astype(np.int8)
    df["price_is_zero"] = (df["price"] == 0).astype(np.int8)
    df["log_price"] = np.log1p(df["price"].fillna(-1).clip(lower=0)).astype(np.float32)
    df["log_item_seq_number"] = np.log1p(df["item_seq_number"]).astype(np.float32)
    df["has_image"] = df["image"].notna().astype(np.int8)
    df["image_top_1_clean"] = df["image_top_1"].fillna(-1).astype(np.float32)

# Fill string missings early for interaction groupings
cat_cols = [
    "region",
    "city",
    "parent_category_name",
    "category_name",
    "param_1",
    "param_2",
    "param_3",
    "user_type",
]

for col in cat_cols:
    train_df[col] = train_df[col].fillna("missing").astype(str)
    val_df[col] = val_df[col].fillna("missing").astype(str)
    test_df[col] = test_df[col].fillna("missing").astype(str)

# Interaction features
for df in [train_df, val_df, test_df]:
    df["cat_param1"] = df["category_name"] + "_" + df["param_1"]
    df["region_city"] = df["region"] + "_" + df["city"]
    df["region_cat"] = df["region"] + "_" + df["category_name"]

all_cat_cols = cat_cols + ["cat_param1", "region_city", "region_cat"]

# Category-relative price statistics (fit strictly on train_df)
price_group_key = ["parent_category_name", "category_name"]
group_stats = (
    train_df.groupby(price_group_key)["log_price"]
    .agg(cat_price_mean="mean", cat_price_std="std", cat_price_median="median")
    .reset_index()
)
group_stats["cat_price_std"] = group_stats["cat_price_std"].fillna(0.0)

train_df = train_df.merge(group_stats, on=price_group_key, how="left")
val_df = val_df.merge(group_stats, on=price_group_key, how="left")
test_df = test_df.merge(group_stats, on=price_group_key, how="left")

for df in [train_df, val_df, test_df]:
    df["price_diff_cat_mean"] = (df["log_price"] - df["cat_price_mean"]).astype(np.float32)
    df["price_ratio_cat_mean"] = (df["log_price"] / (df["cat_price_mean"] + 1e-4)).astype(np.float32)
    df["price_diff_cat_med"] = (df["log_price"] - df["cat_price_median"]).astype(np.float32)
    df["cat_price_mean"] = df["cat_price_mean"].astype(np.float32)
    df["cat_price_std"] = df["cat_price_std"].astype(np.float32)
    df["cat_price_median"] = df["cat_price_median"].astype(np.float32)

# Subcategory (category_name + param_1) price statistics & z-scores
subcat_stats = (
    train_df.groupby("cat_param1")["log_price"]
    .agg(subcat_price_mean="mean", subcat_price_std="std")
    .reset_index()
)
subcat_stats["subcat_price_std"] = subcat_stats["subcat_price_std"].fillna(0.0)

train_df = train_df.merge(subcat_stats, on="cat_param1", how="left")
val_df = val_df.merge(subcat_stats, on="cat_param1", how="left")
test_df = test_df.merge(subcat_stats, on="cat_param1", how="left")

for df in [train_df, val_df, test_df]:
    df["subcat_price_mean"] = df["subcat_price_mean"].fillna(df["cat_price_mean"]).astype(np.float32)
    df["subcat_price_std"] = df["subcat_price_std"].fillna(0.0).astype(np.float32)
    df["subcat_price_diff"] = (df["log_price"] - df["subcat_price_mean"]).astype(np.float32)
    df["subcat_price_ratio"] = (df["log_price"] / (df["subcat_price_mean"] + 1e-4)).astype(np.float32)
    df["subcat_price_zscore"] = (
        (df["log_price"] - df["subcat_price_mean"]) / (df["subcat_price_std"] + 1e-4)
    ).astype(np.float32)

# Region-Category (region + category_name) price statistics & z-scores
region_cat_price_stats = (
    train_df.groupby("region_cat")["log_price"]
    .agg(region_cat_price_mean="mean", region_cat_price_std="std")
    .reset_index()
)
region_cat_price_stats["region_cat_price_std"] = region_cat_price_stats["region_cat_price_std"].fillna(0.0)

train_df = train_df.merge(region_cat_price_stats, on="region_cat", how="left")
val_df = val_df.merge(region_cat_price_stats, on="region_cat", how="left")
test_df = test_df.merge(region_cat_price_stats, on="region_cat", how="left")

for df in [train_df, val_df, test_df]:
    df["region_cat_price_mean"] = df["region_cat_price_mean"].fillna(df["cat_price_mean"]).astype(np.float32)
    df["region_cat_price_std"] = df["region_cat_price_std"].fillna(0.0).astype(np.float32)
    df["price_diff_region_cat"] = (df["log_price"] - df["region_cat_price_mean"]).astype(np.float32)
    df["price_ratio_region_cat"] = (df["log_price"] / (df["region_cat_price_mean"] + 1e-4)).astype(np.float32)
    df["price_zscore_region_cat"] = (
        (df["log_price"] - df["region_cat_price_mean"]) / (df["region_cat_price_std"] + 1e-4)
    ).astype(np.float32)

# User-level price and activity statistics (strictly from train_df)
user_stats = (
    train_df.groupby("user_id")
    .agg(
        user_price_mean=("log_price", "mean"),
        user_listing_count=("log_price", "count"),
        user_seq_mean=("item_seq_number", "mean"),
    )
    .reset_index()
)
train_df = train_df.merge(user_stats, on="user_id", how="left")
val_df = val_df.merge(user_stats, on="user_id", how="left")
test_df = test_df.merge(user_stats, on="user_id", how="left")

for df in [train_df, val_df, test_df]:
    df["user_price_mean"] = df["user_price_mean"].fillna(df["cat_price_mean"]).astype(np.float32)
    df["user_listing_count"] = df["user_listing_count"].fillna(0).astype(np.float32)
    df["user_seq_mean"] = df["user_seq_mean"].fillna(df["item_seq_number"]).astype(np.float32)
    df["user_price_diff"] = (df["log_price"] - df["user_price_mean"]).astype(np.float32)
    df["user_seq_diff"] = (df["item_seq_number"] - df["user_seq_mean"]).astype(np.float32)

# Image_top_1 price aggregations & z-scores (strictly from train_df)
img_stats = (
    train_df[train_df["image_top_1"].notna()]
    .groupby("image_top_1")["log_price"]
    .agg(img_price_mean="mean", img_price_std="std")
    .reset_index()
)
img_stats["img_price_std"] = img_stats["img_price_std"].fillna(0.0)

train_df = train_df.merge(img_stats, on="image_top_1", how="left")
val_df = val_df.merge(img_stats, on="image_top_1", how="left")
test_df = test_df.merge(img_stats, on="image_top_1", how="left")

for df in [train_df, val_df, test_df]:
    df["img_price_mean"] = df["img_price_mean"].fillna(df["cat_price_mean"]).astype(np.float32)
    df["img_price_std"] = df["img_price_std"].fillna(0.0).astype(np.float32)
    df["image_price_diff"] = (df["log_price"] - df["img_price_mean"]).astype(np.float32)
    df["image_price_ratio"] = (df["log_price"] / (df["img_price_mean"] + 1e-4)).astype(np.float32)
    df["image_price_zscore"] = (
        (df["log_price"] - df["img_price_mean"]) / (df["img_price_std"] + 1e-4)
    ).astype(np.float32)

# ---------------------------------------------------------
# 6. Categorical Encodings & Hierarchy Interactions
# ---------------------------------------------------------
print("Encoding categorical and interaction variables...")

# Frequency encoding based strictly on train frequency
for col in ["user_id", "city", "cat_param1", "region_city", "region_cat", "category_name", "image_top_1_clean"]:
    freq_map = train_df[col].value_counts().to_dict()
    train_df[f"{col}_freq"] = train_df[col].map(freq_map).fillna(1).astype(np.float32)
    val_df[f"{col}_freq"] = val_df[col].map(freq_map).fillna(1).astype(np.float32)
    test_df[f"{col}_freq"] = test_df[col].map(freq_map).fillna(1).astype(np.float32)

# Bayesian Smoothed K-Fold Target Encoding
print("Computing Bayesian smoothed target encodings...")
te_cols = ["image_top_1_clean", "param_1", "cat_param1", "region_city", "region_cat", "user_type"]
smoothing = 15.0
global_mean = float(train_df["deal_probability"].mean())
kf_te = KFold(n_splits=5, shuffle=True, random_state=SEED)

for col in te_cols:
    train_df[f"{col}_te"] = np.nan
    for tr_idx, val_fold_idx in kf_te.split(train_df):
        fold_tr = train_df.iloc[tr_idx]
        stats = fold_tr.groupby(col)["deal_probability"].agg(["count", "mean"])
        smoothed = (stats["count"] * stats["mean"] + smoothing * global_mean) / (stats["count"] + smoothing)
        train_df.iloc[val_fold_idx, train_df.columns.get_loc(f"{col}_te")] = (
            train_df.iloc[val_fold_idx][col].map(smoothed).fillna(global_mean).values
        )
    train_df[f"{col}_te"] = train_df[f"{col}_te"].astype(np.float32)

    full_stats = train_df.groupby(col)["deal_probability"].agg(["count", "mean"])
    full_smoothed = (full_stats["count"] * full_stats["mean"] + smoothing * global_mean) / (full_stats["count"] + smoothing)
    val_df[f"{col}_te"] = val_df[col].map(full_smoothed).fillna(global_mean).astype(np.float32)
    test_df[f"{col}_te"] = test_df[col].map(full_smoothed).fillna(global_mean).astype(np.float32)

# Vectorized integer category encoding (strictly mapped from train vocabulary)
for col in all_cat_cols:
    unique_vals = train_df[col].unique()
    cat_mapping = {val: idx for idx, val in enumerate(unique_vals)}
    train_df[col] = train_df[col].map(cat_mapping).fillna(-1).astype(np.int32)
    val_df[col] = val_df[col].map(cat_mapping).fillna(-1).astype(np.int32)
    test_df[col] = test_df[col].map(cat_mapping).fillna(-1).astype(np.int32)

# ---------------------------------------------------------
# 7. Assemble Final Feature Matrices
# ---------------------------------------------------------
feature_cols = [
    *all_cat_cols,
    "null_count",
    "user_id_freq",
    "city_freq",
    "cat_param1_freq",
    "region_city_freq",
    "region_cat_freq",
    "category_name_freq",
    "image_top_1_clean_freq",
    "image_top_1_clean",
    "has_image",
    "day_of_week",
    "day_of_month",
    "is_weekend",
    "log_price",
    "price_is_nan",
    "price_is_zero",
    "log_item_seq_number",
    "cat_price_mean",
    "cat_price_std",
    "cat_price_median",
    "price_diff_cat_mean",
    "price_ratio_cat_mean",
    "price_diff_cat_med",
    "subcat_price_mean",
    "subcat_price_std",
    "subcat_price_diff",
    "subcat_price_ratio",
    "subcat_price_zscore",
    "region_cat_price_mean",
    "region_cat_price_std",
    "price_diff_region_cat",
    "price_ratio_region_cat",
    "price_zscore_region_cat",
    "user_price_mean",
    "user_listing_count",
    "user_seq_mean",
    "user_price_diff",
    "user_seq_diff",
    "img_price_mean",
    "img_price_std",
    "image_price_diff",
    "image_price_ratio",
    "image_price_zscore",
    *[f"{col}_te" for col in te_cols],
]

print("Assembling final feature DataFrames...")
X_train = pd.concat([train_df[feature_cols], train_text_stats, train_svd_df], axis=1)
y_train = train_df["deal_probability"].astype(np.float32)

X_val = pd.concat([val_df[feature_cols], val_text_stats, val_svd_df], axis=1)
y_val = val_df["deal_probability"].astype(np.float32)

X_test = pd.concat([test_df[feature_cols], test_text_stats, test_svd_df], axis=1)

del train_df, val_df, test_df, train_text_stats, val_text_stats, test_text_stats
del train_svd_df, val_svd_df, test_svd_df
gc.collect()

print(
    f"X_train shape: {X_train.shape}, X_val shape: {X_val.shape}, X_test shape: {X_test.shape}"
)

# ---------------------------------------------------------
# 8. Model Architecture & Ensemble Design
# ---------------------------------------------------------
categorical_features = [
    "region",
    "city",
    "parent_category_name",
    "category_name",
    "param_1",
    "param_2",
    "param_3",
    "user_type",
    "cat_param1",
    "region_city",
]

# Model 1: Primary LightGBM with deep capacity
lgb_params_1 = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.03,
    "num_leaves": 256,
    "max_depth": -1,
    "min_child_samples": 40,
    "subsample": 0.85,
    "subsample_freq": 1,
    "colsample_bytree": 0.68,
    "reg_alpha": 0.2,
    "reg_lambda": 1.5,
    "n_estimators": 4000,
    "n_jobs": -1,
    "random_state": 42,
    "verbose": -1,
}

# Model 2: Secondary LightGBM with constrained tree structure and diverse subsampling
lgb_params_2 = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.035,
    "num_leaves": 96,
    "max_depth": 8,
    "min_child_samples": 80,
    "subsample": 0.75,
    "subsample_freq": 1,
    "colsample_bytree": 0.60,
    "reg_alpha": 1.0,
    "reg_lambda": 3.0,
    "n_estimators": 4000,
    "n_jobs": -1,
    "random_state": 2024,
    "verbose": -1,
}

# Model 3: CatBoost Regressor with oblivious decision trees
cat_params = {
    "iterations": 3000,
    "learning_rate": 0.07,
    "depth": 7,
    "l2_leaf_reg": 4.0,
    "loss_function": "RMSE",
    "eval_metric": "RMSE",
    "random_seed": 42,
    "thread_count": -1,
    "verbose": 200,
}

model_lgb1 = lgb.LGBMRegressor(**lgb_params_1)
model_lgb2 = lgb.LGBMRegressor(**lgb_params_2)
model_cat = CatBoostRegressor(**cat_params)

callbacks_lgb = [
    lgb.early_stopping(stopping_rounds=50, verbose=False),
    lgb.log_evaluation(period=200),
]

# ---------------------------------------------------------
# 9. Model Training
# ---------------------------------------------------------
print("Training Model 1 (LightGBM Deep)...")
model_lgb1.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    eval_names=["valid"],
    categorical_feature=categorical_features,
    callbacks=callbacks_lgb,
)
raw_val_lgb1 = model_lgb1.predict(X_val)
raw_test_lgb1 = model_lgb1.predict(X_test)
rmse_lgb1 = float(np.sqrt(mean_squared_error(y_val, np.clip(raw_val_lgb1, 0.0, 1.0))))
print(f"Model 1 (LightGBM Deep) Validation RMSE: {rmse_lgb1:.5f}")

print("Training Model 2 (LightGBM Constrained)...")
model_lgb2.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    eval_names=["valid"],
    categorical_feature=categorical_features,
    callbacks=callbacks_lgb,
)
raw_val_lgb2 = model_lgb2.predict(X_val)
raw_test_lgb2 = model_lgb2.predict(X_test)
rmse_lgb2 = float(np.sqrt(mean_squared_error(y_val, np.clip(raw_val_lgb2, 0.0, 1.0))))
print(f"Model 2 (LightGBM Constrained) Validation RMSE: {rmse_lgb2:.5f}")

print("Training Model 3 (CatBoost Regressor)...")
model_cat.fit(
    X_train,
    y_train,
    eval_set=(X_val, y_val),
    early_stopping_rounds=50,
    verbose=200,
)
raw_val_cat = model_cat.predict(X_val)
raw_test_cat = model_cat.predict(X_test)
rmse_cat = float(np.sqrt(mean_squared_error(y_val, np.clip(raw_val_cat, 0.0, 1.0))))
print(f"Model 3 (CatBoost) Validation RMSE: {rmse_cat:.5f}")

# ---------------------------------------------------------
# 10. Ensemble Optimization & Validation Evaluation
# ---------------------------------------------------------
print("Optimizing ensemble blend weights on validation predictions...")
val_preds_matrix = [raw_val_lgb1, raw_val_lgb2, raw_val_cat]
test_preds_matrix = [raw_test_lgb1, raw_test_lgb2, raw_test_cat]

def blend_loss(weights):
    w = np.array(weights)
    pred = sum(w[i] * val_preds_matrix[i] for i in range(len(w)))
    pred = np.clip(pred, 0.0, 1.0)
    return np.sqrt(mean_squared_error(y_val, pred))

init_weights = [1.0 / len(val_preds_matrix)] * len(val_preds_matrix)
bounds = [(0.0, 1.0)] * len(val_preds_matrix)
constraints = {"type": "eq", "fun": lambda w: 1.0 - sum(w)}

opt_res = minimize(
    blend_loss,
    init_weights,
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
)

opt_weights = opt_res.x / np.sum(opt_res.x)
print(f"Optimal Ensemble Weights: LGB1={opt_weights[0]:.4f}, LGB2={opt_weights[1]:.4f}, Cat={opt_weights[2]:.4f}")

raw_val_preds = sum(opt_weights[i] * val_preds_matrix[i] for i in range(len(opt_weights)))
val_preds = np.clip(raw_val_preds, 0.0, 1.0)
val_rmse = float(np.sqrt(mean_squared_error(y_val, val_preds)))

print(
    f"Validation Evaluation: Target Mean={y_val.mean():.5f}, Pred Mean={val_preds.mean():.5f}"
)

# ---------------------------------------------------------
# 11. Test Inference & Submission Generation
# ---------------------------------------------------------
print("Generating test predictions with optimized ensemble...")
raw_test_preds = sum(opt_weights[i] * test_preds_matrix[i] for i in range(len(opt_weights)))
test_preds = np.clip(raw_test_preds, 0.0, 1.0)

submission_df = pd.DataFrame(
    {"id": np.arange(len(test_preds), dtype=np.int64), "deal_probability": test_preds}
)

submission_path = "./submission/submission.csv"
submission_df.to_csv(submission_path, index=False)
print(
    f"Submission successfully saved to {submission_path} with shape {submission_df.shape}"
)

# Final validation metric output
print(f"Final Validation Score: {val_rmse}")
