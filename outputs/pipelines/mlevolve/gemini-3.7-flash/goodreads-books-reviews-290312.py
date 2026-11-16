import gc
import os
import re
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import StratifiedKFold

# Set random seeds for reproducibility
np.random.seed(42)
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# 1. Load Data
print("Loading raw parquet datasets...")
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")
print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

# 2. Setup Stratified 5-Fold Split on Target
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
train_df["fold"] = -1
for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df["rating"])):
    train_df.loc[val_idx, "fold"] = fold


# 3. Temporal Feature Extraction Function
def extract_temporal_features(df):
    date_cols = ["date_added", "date_updated", "read_at", "started_at"]
    parsed = {}
    for col in date_cols:
        if col in df.columns:
            parsed[col] = pd.to_datetime(
                df[col], format="%a %b %d %H:%M:%S %z %Y", errors="coerce", utc=True
            )

    feats = pd.DataFrame(index=df.index)

    feats["has_started_at"] = (
        parsed["started_at"].notna().astype(np.int8) if "started_at" in parsed else 0
    )
    feats["has_read_at"] = (
        parsed["read_at"].notna().astype(np.int8) if "read_at" in parsed else 0
    )

    if "date_added" in parsed:
        feats["added_year"] = parsed["date_added"].dt.year.fillna(-1).astype(np.int16)
        feats["added_month"] = parsed["date_added"].dt.month.fillna(-1).astype(np.int8)
        feats["added_dayofweek"] = (
            parsed["date_added"].dt.dayofweek.fillna(-1).astype(np.int8)
        )
        feats["added_hour"] = parsed["date_added"].dt.hour.fillna(-1).astype(np.int8)

    if "read_at" in parsed and "started_at" in parsed:
        reading_days = (
            parsed["read_at"] - parsed["started_at"]
        ).dt.total_seconds() / 86400.0
        feats["reading_duration_days"] = (
            reading_days.clip(lower=0.0, upper=3650.0).fillna(-1.0).astype(np.float32)
        )

    if "date_added" in parsed and "read_at" in parsed:
        review_delay = (
            parsed["date_added"] - parsed["read_at"]
        ).dt.total_seconds() / 86400.0
        feats["review_delay_days"] = (
            review_delay.clip(lower=-365.0, upper=3650.0)
            .fillna(-999.0)
            .astype(np.float32)
        )

    if "date_updated" in parsed and "date_added" in parsed:
        update_delay = (
            parsed["date_updated"] - parsed["date_added"]
        ).dt.total_seconds() / 86400.0
        feats["update_delay_days"] = (
            update_delay.clip(lower=0.0, upper=3650.0).fillna(-1.0).astype(np.float32)
        )

    return feats


# 4. Text Meta-Features & Regex Rating Extractor
POS_WORDS = {
    "love",
    "loved",
    "loving",
    "amazing",
    "awesome",
    "favorite",
    "favourite",
    "best",
    "fantastic",
    "excellent",
    "beautiful",
    "perfect",
    "enjoyed",
    "masterpiece",
    "brilliant",
    "wonderful",
    "stellar",
    "superb",
    "highly",
}
NEG_WORDS = {
    "boring",
    "hate",
    "hated",
    "worst",
    "disappointed",
    "disappointing",
    "terrible",
    "awful",
    "poor",
    "bad",
    "dnf",
    "dropped",
    "waste",
    "annoying",
    "unreadable",
    "horrible",
    "dull",
    "cliche",
    "underwhelming",
}

RATING_REGEX_1 = re.compile(
    r"(\b[0-5](?:\.[0-9]+)?)\s*(?:/|\bout of\b)\s*5", re.IGNORECASE
)
RATING_REGEX_2 = re.compile(r"(\b[0-5](?:\.[0-9]+)?)\s*stars?", re.IGNORECASE)


def extract_explicit_rating(text):
    if not isinstance(text, str) or not text:
        return -1.0
    m1 = RATING_REGEX_1.search(text)
    if m1:
        try:
            val = float(m1.group(1))
            if 0.0 <= val <= 5.0:
                return val
        except ValueError:
            pass
    m2 = RATING_REGEX_2.search(text)
    if m2:
        try:
            val = float(m2.group(1))
            if 0.0 <= val <= 5.0:
                return val
        except ValueError:
            pass
    return -1.0


def extract_text_meta_features(df):
    text_s = df["review_text"].fillna("")
    feats = pd.DataFrame(index=df.index)

    feats["review_char_len"] = text_s.str.len().astype(np.int32)
    feats["review_word_count"] = text_s.apply(lambda x: len(x.split())).astype(np.int32)
    feats["avg_word_len"] = (
        feats["review_char_len"] / (feats["review_word_count"] + 1)
    ).astype(np.float32)

    feats["exclamation_count"] = text_s.str.count("!").astype(np.int16)
    feats["question_count"] = text_s.str.count(r"\?").astype(np.int16)
    feats["ellipsis_count"] = text_s.str.count(r"\.\.\.").astype(np.int16)
    feats["quote_count"] = text_s.str.count('"').astype(np.int16)

    upper_chars = text_s.str.findall(r"[A-Z]").str.len()
    feats["upper_char_ratio"] = (upper_chars / (feats["review_char_len"] + 1)).astype(
        np.float32
    )

    def get_sent_counts(text):
        if not text:
            return 0, 0
        words = set(re.findall(r"\b[a-z]+\b", text.lower()))
        pos = len(words.intersection(POS_WORDS))
        neg = len(words.intersection(NEG_WORDS))
        return pos, neg

    sent_tuples = [get_sent_counts(t) for t in text_s]
    pos_counts = np.array([t[0] for t in sent_tuples], dtype=np.int16)
    neg_counts = np.array([t[1] for t in sent_tuples], dtype=np.int16)

    feats["pos_word_count"] = pos_counts
    feats["neg_word_count"] = neg_counts
    feats["sentiment_balance"] = (
        (pos_counts - neg_counts) / (pos_counts + neg_counts + 1)
    ).astype(np.float32)

    explicit_ratings = np.array(
        [extract_explicit_rating(t) for t in text_s], dtype=np.float32
    )
    feats["explicit_rating_extracted"] = explicit_ratings
    feats["has_explicit_rating"] = (explicit_ratings >= 0.0).astype(np.int8)

    return feats


# 5. Engagement Features
def extract_engagement_features(df):
    feats = pd.DataFrame(index=df.index)
    v = np.maximum(0, df["n_votes"].fillna(0).values)
    c = np.maximum(0, df["n_comments"].fillna(0).values)

    feats["n_votes_clean"] = v.astype(np.int32)
    feats["n_comments_clean"] = c.astype(np.int32)
    feats["log_votes"] = np.log1p(v).astype(np.float32)
    feats["log_comments"] = np.log1p(c).astype(np.float32)
    feats["votes_per_comment"] = (v / (c + 1.0)).astype(np.float32)
    return feats


print("Extracting base features (temporal, text-meta, engagement)...")
train_temporal = extract_temporal_features(train_df)
test_temporal = extract_temporal_features(test_df)

train_text_meta = extract_text_meta_features(train_df)
test_text_meta = extract_text_meta_features(test_df)

train_eng = extract_engagement_features(train_df)
test_eng = extract_engagement_features(test_df)

# 6. TF-IDF + OOF Ridge Regression & TruncatedSVD Text Embeddings
print("Fitting TF-IDF, OOF Ridge Regression, and TruncatedSVD on reviews...")
tfidf = TfidfVectorizer(
    max_features=100000,
    ngram_range=(1, 2),
    min_df=3,
    sublinear_tf=True,
    stop_words="english",
    dtype=np.float32,
)

train_text_clean = train_df["review_text"].fillna("")
test_text_clean = test_df["review_text"].fillna("")

tfidf_train = tfidf.fit_transform(train_text_clean)
tfidf_test = tfidf.transform(test_text_clean)

oof_ridge_text = np.zeros(len(train_df), dtype=np.float32)
test_ridge_text = np.zeros(len(test_df), dtype=np.float32)

print("Training OOF Ridge Regression on 100k TF-IDF features...")
for fold in range(n_splits):
    trn_mask = train_df["fold"] != fold
    val_mask = train_df["fold"] == fold

    ridge = Ridge(alpha=3.0, solver="auto", random_state=42)
    ridge.fit(tfidf_train[trn_mask], train_df.loc[trn_mask, "rating"].values)
    oof_ridge_text[val_mask] = ridge.predict(tfidf_train[val_mask]).astype(np.float32)
    test_ridge_text += ridge.predict(tfidf_test).astype(np.float32) / n_splits

n_components = 64
svd = TruncatedSVD(n_components=n_components, random_state=42)
svd_train = svd.fit_transform(tfidf_train).astype(np.float32)
svd_test = svd.transform(tfidf_test).astype(np.float32)

svd_cols = [f"svd_dim_{i}" for i in range(n_components)]
train_svd_df = pd.DataFrame(svd_train, columns=svd_cols, index=train_df.index)
test_svd_df = pd.DataFrame(svd_test, columns=svd_cols, index=test_df.index)

train_svd_df["ridge_text_pred"] = oof_ridge_text
test_svd_df["ridge_text_pred"] = test_ridge_text

del tfidf_train, tfidf_test, train_text_clean, test_text_clean
gc.collect()

# 7. Leak-Free Out-of-Fold Target Encoding and Aggregation Statistics
print("Computing leak-free out-of-fold target and entity statistics...")
global_rating_mean = float(train_df["rating"].mean())
global_rating_std = float(train_df["rating"].std())


def compute_oof_target_encoding(
    train_data, test_data, group_col, target_col, smoothing=10.0
):
    oof_means = np.full(len(train_data), global_rating_mean, dtype=np.float32)
    oof_stds = np.full(len(train_data), global_rating_std, dtype=np.float32)
    oof_counts = np.zeros(len(train_data), dtype=np.float32)

    for fold in range(n_splits):
        trn_idx = train_data["fold"] != fold
        val_idx = train_data["fold"] == fold

        trn_subset = train_data.loc[trn_idx]
        stats = (
            trn_subset.groupby(group_col)[target_col]
            .agg(["count", "mean", "std"])
            .reset_index()
        )
        stats["std"] = stats["std"].fillna(global_rating_std)
        stats["smooth_mean"] = (
            stats["count"] * stats["mean"] + smoothing * global_rating_mean
        ) / (stats["count"] + smoothing)

        val_merged = train_data.loc[val_idx, [group_col]].merge(
            stats, on=group_col, how="left"
        )
        oof_means[val_idx] = val_merged["smooth_mean"].fillna(global_rating_mean).values
        oof_stds[val_idx] = val_merged["std"].fillna(global_rating_std).values
        oof_counts[val_idx] = val_merged["count"].fillna(0).values

    full_stats = (
        train_data.groupby(group_col)[target_col]
        .agg(["count", "mean", "std"])
        .reset_index()
    )
    full_stats["std"] = full_stats["std"].fillna(global_rating_std)
    full_stats["smooth_mean"] = (
        full_stats["count"] * full_stats["mean"] + smoothing * global_rating_mean
    ) / (full_stats["count"] + smoothing)

    test_merged = test_data[[group_col]].merge(full_stats, on=group_col, how="left")
    test_means = (
        test_merged["smooth_mean"].fillna(global_rating_mean).values.astype(np.float32)
    )
    test_stds = test_merged["std"].fillna(global_rating_std).values.astype(np.float32)
    test_counts = test_merged["count"].fillna(0).values.astype(np.float32)

    return oof_means, oof_stds, oof_counts, test_means, test_stds, test_counts


u_mean_tr, u_std_tr, u_cnt_tr, u_mean_te, u_std_te, u_cnt_te = (
    compute_oof_target_encoding(train_df, test_df, "user_id", "rating", smoothing=5.0)
)
b_mean_tr, b_std_tr, b_cnt_tr, b_mean_te, b_std_te, b_cnt_te = (
    compute_oof_target_encoding(train_df, test_df, "book_id", "rating", smoothing=10.0)
)


def compute_oof_activity_encoding(train_data, test_data, group_col, value_s):
    df_temp = train_data[["fold", group_col]].copy()
    df_temp["val"] = value_s.values
    global_m = float(df_temp["val"].mean())

    oof_val = np.full(len(train_data), global_m, dtype=np.float32)
    for fold in range(n_splits):
        trn_idx = train_data["fold"] != fold
        val_idx = train_data["fold"] == fold

        trn_sub = df_temp.loc[trn_idx]
        stats = trn_sub.groupby(group_col)["val"].mean().reset_index()
        val_m = df_temp.loc[val_idx, [group_col]].merge(stats, on=group_col, how="left")
        oof_val[val_idx] = val_m["val"].fillna(global_m).values

    full_stats = df_temp.groupby(group_col)["val"].mean().reset_index()
    test_m = test_data[[group_col]].merge(full_stats, on=group_col, how="left")
    test_val = test_m["val"].fillna(global_m).values.astype(np.float32)

    return oof_val, test_val


u_word_tr, u_word_te = compute_oof_activity_encoding(
    train_df, test_df, "user_id", train_text_meta["review_word_count"]
)
b_word_tr, b_word_te = compute_oof_activity_encoding(
    train_df, test_df, "book_id", train_text_meta["review_word_count"]
)

# Multi-dimensional entity aggregations: sentiment & engagement
u_sent_tr, u_sent_te = compute_oof_activity_encoding(
    train_df, test_df, "user_id", train_text_meta["sentiment_balance"]
)
b_sent_tr, b_sent_te = compute_oof_activity_encoding(
    train_df, test_df, "book_id", train_text_meta["sentiment_balance"]
)

u_vote_tr, u_vote_te = compute_oof_activity_encoding(
    train_df, test_df, "user_id", train_eng["log_votes"]
)
b_vote_tr, b_vote_te = compute_oof_activity_encoding(
    train_df, test_df, "book_id", train_eng["log_votes"]
)

u_comm_tr, u_comm_te = compute_oof_activity_encoding(
    train_df, test_df, "user_id", train_eng["log_comments"]
)
b_comm_tr, b_comm_te = compute_oof_activity_encoding(
    train_df, test_df, "book_id", train_eng["log_comments"]
)

# Joint Sparse User-Book Collaborative Ridge Bias Extraction
print("Computing joint sparse user-book collaborative ridge biases...")
unique_users = pd.Series(pd.concat([train_df["user_id"], test_df["user_id"]]).unique())
unique_books = pd.Series(pd.concat([train_df["book_id"], test_df["book_id"]]).unique())
user_map = {u: i for i, u in enumerate(unique_users)}
book_map = {b: i for i, b in enumerate(unique_books)}
n_users = len(user_map)
n_books = len(book_map)


def build_collab_matrix(df):
    u_idx = df["user_id"].map(user_map).values
    b_idx = df["book_id"].map(book_map).values + n_users
    row_ind = np.repeat(np.arange(len(df)), 2)
    col_ind = np.empty(len(df) * 2, dtype=np.int32)
    col_ind[0::2] = u_idx
    col_ind[1::2] = b_idx
    data = np.ones(len(df) * 2, dtype=np.float32)
    return sparse.csr_matrix(
        (data, (row_ind, col_ind)), shape=(len(df), n_users + n_books), dtype=np.float32
    )


X_collab_tr = build_collab_matrix(train_df)
X_collab_te = build_collab_matrix(test_df)

oof_collab_pred = np.zeros(len(train_df), dtype=np.float32)
test_collab_pred = np.zeros(len(test_df), dtype=np.float32)

for fold in range(n_splits):
    trn_mask = train_df["fold"] != fold
    val_mask = train_df["fold"] == fold
    ridge_collab = Ridge(alpha=12.0, solver="lsqr", random_state=42)
    ridge_collab.fit(X_collab_tr[trn_mask], train_df.loc[trn_mask, "rating"].values)
    oof_collab_pred[val_mask] = ridge_collab.predict(X_collab_tr[val_mask]).astype(
        np.float32
    )
    test_collab_pred += (
        ridge_collab.predict(X_collab_te).astype(np.float32) / n_splits
    )

del X_collab_tr, X_collab_te, unique_users, unique_books, user_map, book_map
gc.collect()

# 8. Assemble Full Feature DataFrames
print("Assembling final feature matrices with interaction residuals...")
train_features = pd.concat(
    [
        train_df[["review_id", "user_id", "book_id", "fold", "rating"]],
        train_temporal,
        train_text_meta,
        train_eng,
        train_svd_df,
    ],
    axis=1,
)

train_features["collab_ridge_pred"] = oof_collab_pred

train_features["user_rating_mean_te"] = u_mean_tr
train_features["user_rating_std_te"] = u_std_tr
train_features["user_rating_log_cnt"] = np.log1p(u_cnt_tr).astype(np.float32)
train_features["user_avg_word_count"] = u_word_tr
train_features["user_avg_sentiment"] = u_sent_tr
train_features["user_avg_log_votes"] = u_vote_tr
train_features["user_avg_log_comments"] = u_comm_tr

train_features["book_rating_mean_te"] = b_mean_tr
train_features["book_rating_std_te"] = b_std_tr
train_features["book_rating_log_cnt"] = np.log1p(b_cnt_tr).astype(np.float32)
train_features["book_avg_word_count"] = b_word_tr
train_features["book_avg_sentiment"] = b_sent_tr
train_features["book_avg_log_votes"] = b_vote_tr
train_features["book_avg_log_comments"] = b_comm_tr

# Relative entity differences
train_features["user_book_mean_diff"] = (
    train_features["user_rating_mean_te"] - train_features["book_rating_mean_te"]
).astype(np.float32)
train_features["user_book_mean_sum"] = (
    train_features["user_rating_mean_te"] + train_features["book_rating_mean_te"]
).astype(np.float32)

train_features["user_sentiment_diff"] = (
    train_features["sentiment_balance"] - train_features["user_avg_sentiment"]
).astype(np.float32)
train_features["book_sentiment_diff"] = (
    train_features["sentiment_balance"] - train_features["book_avg_sentiment"]
).astype(np.float32)
train_features["user_vote_ratio"] = (
    (train_features["log_votes"] + 0.1) / (train_features["user_avg_log_votes"] + 0.1)
).astype(np.float32)
train_features["book_vote_ratio"] = (
    (train_features["log_votes"] + 0.1) / (train_features["book_avg_log_votes"] + 0.1)
).astype(np.float32)

# Text-to-entity and collaborative prediction residuals
train_features["ridge_text_minus_user_te"] = (
    train_features["ridge_text_pred"] - train_features["user_rating_mean_te"]
).astype(np.float32)
train_features["ridge_text_minus_book_te"] = (
    train_features["ridge_text_pred"] - train_features["book_rating_mean_te"]
).astype(np.float32)
train_features["ridge_text_minus_collab"] = (
    train_features["ridge_text_pred"] - train_features["collab_ridge_pred"]
).astype(np.float32)
train_features["collab_minus_user_te"] = (
    train_features["collab_ridge_pred"] - train_features["user_rating_mean_te"]
).astype(np.float32)
train_features["collab_minus_book_te"] = (
    train_features["collab_ridge_pred"] - train_features["book_rating_mean_te"]
).astype(np.float32)

train_features["explicit_diff_user"] = np.where(
    train_features["explicit_rating_extracted"] >= 0,
    train_features["explicit_rating_extracted"] - train_features["user_rating_mean_te"],
    -999.0,
).astype(np.float32)
train_features["explicit_diff_book"] = np.where(
    train_features["explicit_rating_extracted"] >= 0,
    train_features["explicit_rating_extracted"] - train_features["book_rating_mean_te"],
    -999.0,
).astype(np.float32)
train_features["explicit_diff_collab"] = np.where(
    train_features["explicit_rating_extracted"] >= 0,
    train_features["explicit_rating_extracted"] - train_features["collab_ridge_pred"],
    -999.0,
).astype(np.float32)

test_features = pd.concat(
    [
        test_df[["review_id", "user_id", "book_id"]],
        test_temporal,
        test_text_meta,
        test_eng,
        test_svd_df,
    ],
    axis=1,
)

test_features["collab_ridge_pred"] = test_collab_pred

test_features["user_rating_mean_te"] = u_mean_te
test_features["user_rating_std_te"] = u_std_te
test_features["user_rating_log_cnt"] = np.log1p(u_cnt_te).astype(np.float32)
test_features["user_avg_word_count"] = u_word_te
test_features["user_avg_sentiment"] = u_sent_te
test_features["user_avg_log_votes"] = u_vote_te
test_features["user_avg_log_comments"] = u_comm_te

test_features["book_rating_mean_te"] = b_mean_te
test_features["book_rating_std_te"] = b_std_te
test_features["book_rating_log_cnt"] = np.log1p(b_cnt_te).astype(np.float32)
test_features["book_avg_word_count"] = b_word_te
test_features["book_avg_sentiment"] = b_sent_te
test_features["book_avg_log_votes"] = b_vote_te
test_features["book_avg_log_comments"] = b_comm_te

test_features["user_book_mean_diff"] = (
    test_features["user_rating_mean_te"] - test_features["book_rating_mean_te"]
).astype(np.float32)
test_features["user_book_mean_sum"] = (
    test_features["user_rating_mean_te"] + test_features["book_rating_mean_te"]
).astype(np.float32)

test_features["user_sentiment_diff"] = (
    test_features["sentiment_balance"] - test_features["user_avg_sentiment"]
).astype(np.float32)
test_features["book_sentiment_diff"] = (
    test_features["sentiment_balance"] - test_features["book_avg_sentiment"]
).astype(np.float32)
test_features["user_vote_ratio"] = (
    (test_features["log_votes"] + 0.1) / (test_features["user_avg_log_votes"] + 0.1)
).astype(np.float32)
test_features["book_vote_ratio"] = (
    (test_features["log_votes"] + 0.1) / (test_features["book_avg_log_votes"] + 0.1)
).astype(np.float32)

test_features["ridge_text_minus_user_te"] = (
    test_features["ridge_text_pred"] - test_features["user_rating_mean_te"]
).astype(np.float32)
test_features["ridge_text_minus_book_te"] = (
    test_features["ridge_text_pred"] - test_features["book_rating_mean_te"]
).astype(np.float32)
test_features["ridge_text_minus_collab"] = (
    test_features["ridge_text_pred"] - test_features["collab_ridge_pred"]
).astype(np.float32)
test_features["collab_minus_user_te"] = (
    test_features["collab_ridge_pred"] - test_features["user_rating_mean_te"]
).astype(np.float32)
test_features["collab_minus_book_te"] = (
    test_features["collab_ridge_pred"] - test_features["book_rating_mean_te"]
).astype(np.float32)

test_features["explicit_diff_user"] = np.where(
    test_features["explicit_rating_extracted"] >= 0,
    test_features["explicit_rating_extracted"] - test_features["user_rating_mean_te"],
    -999.0,
).astype(np.float32)
test_features["explicit_diff_book"] = np.where(
    test_features["explicit_rating_extracted"] >= 0,
    test_features["explicit_rating_extracted"] - test_features["book_rating_mean_te"],
    -999.0,
).astype(np.float32)
test_features["explicit_diff_collab"] = np.where(
    test_features["explicit_rating_extracted"] >= 0,
    test_features["explicit_rating_extracted"] - test_features["collab_ridge_pred"],
    -999.0,
).astype(np.float32)

del (
    train_temporal,
    test_temporal,
    train_text_meta,
    test_text_meta,
    train_eng,
    test_eng,
    train_svd_df,
    test_svd_df,
)
gc.collect()

# 9. Model Training and Cross-Validation
IGNORE_COLS = {"review_id", "user_id", "book_id", "rating", "fold"}
feature_cols = [c for c in train_features.columns if c not in IGNORE_COLS]
print(f"Total predictive features: {len(feature_cols)}")

oof_preds = np.zeros(len(train_features), dtype=np.float32)
test_preds = np.zeros(len(test_features), dtype=np.float32)

y_true = train_features["rating"].values.astype(np.float32)
folds = train_features["fold"].values

print("Starting 5-Fold Stratified LightGBM Training...")
for fold in range(n_splits):
    trn_mask = folds != fold
    val_mask = folds == fold

    X_train = train_features.loc[trn_mask, feature_cols]
    y_train = y_true[trn_mask]
    X_val = train_features.loc[val_mask, feature_cols]
    y_val = y_true[val_mask]

    model = lgb.LGBMRegressor(
        objective="regression",
        metric="rmse",
        boosting_type="gbdt",
        n_estimators=2500,
        learning_rate=0.03,
        num_leaves=95,
        max_depth=-1,
        min_child_samples=40,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.7,
        reg_alpha=0.1,
        reg_lambda=1.0,
        n_jobs=-1,
        random_state=42 + fold,
        verbose=-1,
    )

    callbacks = [
        lgb.early_stopping(stopping_rounds=50, verbose=False),
        lgb.log_evaluation(period=0),
    ]

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_names=["val"],
        callbacks=callbacks,
    )

    val_pred = model.predict(X_val)
    val_pred_clipped = np.clip(val_pred, 0.0, 5.0)
    oof_preds[val_mask] = val_pred_clipped

    fold_rmse = root_mean_squared_error(y_val, val_pred_clipped)
    print(
        f"Fold {fold} | Best Iteration: {model.best_iteration_} | Validation RMSE: {fold_rmse:.5f}"
    )

    test_fold_pred = model.predict(test_features[feature_cols])
    test_preds += np.clip(test_fold_pred, 0.0, 5.0) / n_splits

    del X_train, y_train, X_val, y_val, val_pred, test_fold_pred, model
    gc.collect()

# 10. Final Out-Of-Fold Validation Evaluation
final_oof_rmse = root_mean_squared_error(y_true, oof_preds)

# 11. Format and Save Final Submission
submission_df = pd.DataFrame(
    {"id": np.arange(len(test_preds), dtype=np.int64), "rating": test_preds}
)

submission_path = "./submission/submission.csv"
submission_df.to_csv(submission_path, index=False)

assert len(submission_df) == len(
    test_df
), f"Mismatch in test sample count: {len(submission_df)} vs {len(test_df)}"
assert not submission_df.isnull().values.any(), "NaN values found in submission"
print(f"Submission successfully written to {submission_path}")

# 12. Final Validation Score Output
print(f"Final Validation Score: {final_oof_rmse}")
