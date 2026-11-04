import gc
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import lightgbm as lgb

warnings.filterwarnings("ignore")

# 1. Load Data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

y = train_df["rating"].values.astype(np.float32)
train_len = len(train_df)
test_len = len(test_df)

combined = pd.concat(
    [train_df.drop(columns=["rating"], errors="ignore"), test_df],
    ignore_index=True,
)

# 2. Text Meta Features & Sentiment Lexicons
text_col = combined["review_text"].fillna("")

combined["text_len"] = text_col.str.len().astype(np.float32)
combined["word_count"] = text_col.str.split().str.len().fillna(0).astype(np.float32)
combined["avg_word_len"] = (
    combined["text_len"] / (combined["word_count"] + 1e-5)
).astype(np.float32)
combined["exclamation_count"] = text_col.str.count("!").astype(np.float32)
combined["question_count"] = text_col.str.count(r"\?").astype(np.float32)
combined["upper_count"] = text_col.apply(
    lambda s: sum(1 for c in s if c.isupper())
).astype(np.float32)
combined["upper_ratio"] = (
    combined["upper_count"] / (combined["text_len"] + 1e-5)
).astype(np.float32)
combined["digit_count"] = text_col.apply(
    lambda s: sum(1 for c in s if c.isdigit())
).astype(np.float32)

pos_pattern = r"\b(love|loved|great|amazing|favorite|favourite|awesome|enjoyed|best|fantastic|excellent|5 stars|4 stars)\b"
neg_pattern = r"\b(boring|disappointed|disappointing|dnf|worst|terrible|hate|hated|awful|horrible|1 star|2 stars|meh)\b"

text_lower = text_col.str.lower()
combined["pos_lex_count"] = text_lower.str.count(pos_pattern).astype(np.float32)
combined["neg_lex_count"] = text_lower.str.count(neg_pattern).astype(np.float32)
combined["sentiment_diff"] = (
    combined["pos_lex_count"] - combined["neg_lex_count"]
).astype(np.float32)
del text_lower
gc.collect()

# 3. Temporal Attributes & Reading Durations
date_cols = ["date_added", "date_updated", "read_at", "started_at"]
ts_dict = {}

for col in date_cols:
    dt = pd.to_datetime(
        combined[col], format="%a %b %d %H:%M:%S %z %Y", errors="coerce", utc=True
    )
    combined[f"{col}_year"] = dt.dt.year.fillna(-1).astype(np.float32)
    combined[f"{col}_month"] = dt.dt.month.fillna(-1).astype(np.float32)
    combined[f"{col}_dayofweek"] = dt.dt.dayofweek.fillna(-1).astype(np.float32)
    valid_mask = dt.notna()
    ts_array = np.full(len(combined), -1.0, dtype=np.float64)
    ts_array[valid_mask] = dt[valid_mask].astype("int64").values // 10**9
    ts_dict[col] = ts_array

combined["reading_duration_days"] = np.where(
    (ts_dict["read_at"] > 0) & (ts_dict["started_at"] > 0),
    (ts_dict["read_at"] - ts_dict["started_at"]) / 86400.0,
    -1.0,
).astype(np.float32)

combined["update_delay_days"] = np.where(
    (ts_dict["date_updated"] > 0) & (ts_dict["date_added"] > 0),
    (ts_dict["date_updated"] - ts_dict["date_added"]) / 86400.0,
    -1.0,
).astype(np.float32)

combined["add_to_read_days"] = np.where(
    (ts_dict["read_at"] > 0) & (ts_dict["date_added"] > 0),
    (ts_dict["read_at"] - ts_dict["date_added"]) / 86400.0,
    -1.0,
).astype(np.float32)

del ts_dict
gc.collect()

combined["log_n_votes"] = np.log1p(
    np.maximum(combined["n_votes"].fillna(0).values, 0)
).astype(np.float32)
combined["log_n_comments"] = np.log1p(
    np.maximum(combined["n_comments"].fillna(0).values, 0)
).astype(np.float32)
combined["vote_comment_ratio"] = (
    (combined["n_votes"].fillna(0) + 1.0) / (combined["n_comments"].fillna(0) + 1.0)
).astype(np.float32)

# 4. TF-IDF & SVD Feature Extraction
word_vec = TfidfVectorizer(
    max_features=25000, ngram_range=(1, 2), min_df=5, sublinear_tf=True
)
X_tfidf_word = word_vec.fit_transform(text_col)

char_vec = TfidfVectorizer(
    max_features=12000,
    analyzer="char",
    ngram_range=(3, 4),
    min_df=10,
    sublinear_tf=True,
)
X_tfidf_char = char_vec.fit_transform(text_col)

svd = TruncatedSVD(n_components=12, random_state=42)
svd_feats = svd.fit_transform(X_tfidf_word).astype(np.float32)
for i in range(12):
    combined[f"svd_word_{i}"] = svd_feats[:, i]

train_meta = combined.iloc[:train_len].copy()
test_meta = combined.iloc[train_len:].copy()

X_word_tr = X_tfidf_word[:train_len]
X_word_te = X_tfidf_word[train_len:]

X_char_tr = X_tfidf_char[:train_len]
X_char_te = X_tfidf_char[train_len:]

del combined, text_col, X_tfidf_word, X_tfidf_char, svd, svd_feats
gc.collect()

# 5. Out-Of-Fold Ridge Predictions
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

ridge_word_oof = np.zeros(train_len, dtype=np.float32)
ridge_word_test = np.zeros(test_len, dtype=np.float32)
ridge_char_oof = np.zeros(train_len, dtype=np.float32)
ridge_char_test = np.zeros(test_len, dtype=np.float32)

for fold, (trn_idx, val_idx) in enumerate(skf.split(train_meta, y.astype(int))):
    ridge_w = Ridge(alpha=3.0, random_state=42)
    ridge_w.fit(X_word_tr[trn_idx], y[trn_idx])
    ridge_word_oof[val_idx] = ridge_w.predict(X_word_tr[val_idx])
    ridge_word_test += ridge_w.predict(X_word_te) / n_splits

    ridge_c = Ridge(alpha=3.0, random_state=42)
    ridge_c.fit(X_char_tr[trn_idx], y[trn_idx])
    ridge_char_oof[val_idx] = ridge_c.predict(X_char_tr[val_idx])
    ridge_char_test += ridge_c.predict(X_char_te) / n_splits

train_meta["ridge_word_pred"] = ridge_word_oof
test_meta["ridge_word_pred"] = ridge_word_test
train_meta["ridge_char_pred"] = ridge_char_oof
test_meta["ridge_char_pred"] = ridge_char_test

del X_word_tr, X_word_te, X_char_tr, X_char_te
gc.collect()

# Target Encoding for user_id and book_id
m_user = 10.0
m_book = 10.0
global_mean = float(np.mean(y))

u_stats_g = (
    train_meta.assign(rating=y)
    .groupby("user_id")["rating"]
    .agg(["count", "mean", "std"])
    .reset_index()
)
u_stats_g["std"] = u_stats_g["std"].fillna(0.0)
u_stats_g["smooth_mean"] = (
    u_stats_g["count"] * u_stats_g["mean"] + m_user * global_mean
) / (u_stats_g["count"] + m_user)

b_stats_g = (
    train_meta.assign(rating=y)
    .groupby("book_id")["rating"]
    .agg(["count", "mean", "std"])
    .reset_index()
)
b_stats_g["std"] = b_stats_g["std"].fillna(0.0)
b_stats_g["smooth_mean"] = (
    b_stats_g["count"] * b_stats_g["mean"] + m_book * global_mean
) / (b_stats_g["count"] + m_book)

u_map_mean = dict(zip(u_stats_g["user_id"], u_stats_g["smooth_mean"]))
u_map_std = dict(zip(u_stats_g["user_id"], u_stats_g["std"]))
u_map_count = dict(zip(u_stats_g["user_id"], u_stats_g["count"]))

b_map_mean = dict(zip(b_stats_g["book_id"], b_stats_g["smooth_mean"]))
b_map_std = dict(zip(b_stats_g["book_id"], b_stats_g["std"]))
b_map_count = dict(zip(b_stats_g["book_id"], b_stats_g["count"]))

test_meta["user_target_enc_mean"] = (
    test_meta["user_id"].map(u_map_mean).fillna(global_mean).astype(np.float32)
)
test_meta["user_target_enc_std"] = (
    test_meta["user_id"].map(u_map_std).fillna(0.0).astype(np.float32)
)
test_meta["user_target_enc_count"] = (
    test_meta["user_id"].map(u_map_count).fillna(0.0).astype(np.float32)
)

test_meta["book_target_enc_mean"] = (
    test_meta["book_id"].map(b_map_mean).fillna(global_mean).astype(np.float32)
)
test_meta["book_target_enc_std"] = (
    test_meta["book_id"].map(b_map_std).fillna(0.0).astype(np.float32)
)
test_meta["book_target_enc_count"] = (
    test_meta["book_id"].map(b_map_count).fillna(0.0).astype(np.float32)
)

u_enc_mean_oof = np.zeros(train_len, dtype=np.float32)
u_enc_std_oof = np.zeros(train_len, dtype=np.float32)
u_enc_count_oof = np.zeros(train_len, dtype=np.float32)

b_enc_mean_oof = np.zeros(train_len, dtype=np.float32)
b_enc_std_oof = np.zeros(train_len, dtype=np.float32)
b_enc_count_oof = np.zeros(train_len, dtype=np.float32)

for fold, (trn_idx, val_idx) in enumerate(skf.split(train_meta, y.astype(int))):
    trn_df = train_meta.iloc[trn_idx].assign(rating=y[trn_idx])

    u_stats = (
        trn_df.groupby("user_id")["rating"].agg(["count", "mean", "std"]).reset_index()
    )
    u_stats["std"] = u_stats["std"].fillna(0.0)
    u_stats["smooth_mean"] = (
        u_stats["count"] * u_stats["mean"] + m_user * global_mean
    ) / (u_stats["count"] + m_user)

    u_m = dict(zip(u_stats["user_id"], u_stats["smooth_mean"]))
    u_s = dict(zip(u_stats["user_id"], u_stats["std"]))
    u_c = dict(zip(u_stats["user_id"], u_stats["count"]))

    val_users = train_meta.iloc[val_idx]["user_id"]
    u_enc_mean_oof[val_idx] = val_users.map(u_m).fillna(global_mean).values
    u_enc_std_oof[val_idx] = val_users.map(u_s).fillna(0.0).values
    u_enc_count_oof[val_idx] = val_users.map(u_c).fillna(0.0).values

    b_stats = (
        trn_df.groupby("book_id")["rating"].agg(["count", "mean", "std"]).reset_index()
    )
    b_stats["std"] = b_stats["std"].fillna(0.0)
    b_stats["smooth_mean"] = (
        b_stats["count"] * b_stats["mean"] + m_book * global_mean
    ) / (b_stats["count"] + m_book)

    b_m = dict(zip(b_stats["book_id"], b_stats["smooth_mean"]))
    b_s = dict(zip(b_stats["book_id"], b_stats["std"]))
    b_c = dict(zip(b_stats["book_id"], b_stats["count"]))

    val_books = train_meta.iloc[val_idx]["book_id"]
    b_enc_mean_oof[val_idx] = val_books.map(b_m).fillna(global_mean).values
    b_enc_std_oof[val_idx] = val_books.map(b_s).fillna(0.0).values
    b_enc_count_oof[val_idx] = val_books.map(b_c).fillna(0.0).values

train_meta["user_target_enc_mean"] = u_enc_mean_oof
train_meta["user_target_enc_std"] = u_enc_std_oof
train_meta["user_target_enc_count"] = u_enc_count_oof

train_meta["book_target_enc_mean"] = b_enc_mean_oof
train_meta["book_target_enc_std"] = b_enc_std_oof
train_meta["book_target_enc_count"] = b_enc_count_oof

# 6. Additive Baseline & Text Rating Deviation Features
train_meta["additive_baseline"] = (
    train_meta["user_target_enc_mean"]
    + train_meta["book_target_enc_mean"]
    - global_mean
)
test_meta["additive_baseline"] = (
    test_meta["user_target_enc_mean"] + test_meta["book_target_enc_mean"] - global_mean
)

train_meta["ridge_word_user_diff"] = (
    train_meta["ridge_word_pred"] - train_meta["user_target_enc_mean"]
)
test_meta["ridge_word_user_diff"] = (
    test_meta["ridge_word_pred"] - test_meta["user_target_enc_mean"]
)

train_meta["ridge_word_book_diff"] = (
    train_meta["ridge_word_pred"] - train_meta["book_target_enc_mean"]
)
test_meta["ridge_word_book_diff"] = (
    test_meta["ridge_word_pred"] - test_meta["book_target_enc_mean"]
)

train_meta["ridge_char_user_diff"] = (
    train_meta["ridge_char_pred"] - train_meta["user_target_enc_mean"]
)
test_meta["ridge_char_user_diff"] = (
    test_meta["ridge_char_pred"] - test_meta["user_target_enc_mean"]
)

train_meta["ridge_char_book_diff"] = (
    train_meta["ridge_char_pred"] - train_meta["book_target_enc_mean"]
)
test_meta["ridge_char_book_diff"] = (
    test_meta["ridge_char_pred"] - test_meta["book_target_enc_mean"]
)

drop_cols = [
    "review_id",
    "user_id",
    "book_id",
    "review_text",
    "date_added",
    "date_updated",
    "read_at",
    "started_at",
]
feature_cols = [c for c in train_meta.columns if c not in drop_cols]

# 7. LightGBM Training & Cross-Validation
oof_preds = np.zeros(train_len, dtype=np.float32)
test_preds = np.zeros(test_len, dtype=np.float32)

lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "n_estimators": 1200,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}

for fold, (trn_idx, val_idx) in enumerate(skf.split(train_meta, y.astype(int))):
    X_train, y_train = train_meta.iloc[trn_idx][feature_cols], y[trn_idx]
    X_val, y_val = train_meta.iloc[val_idx][feature_cols], y[val_idx]

    model = lgb.LGBMRegressor(**lgb_params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )

    oof_preds[val_idx] = model.predict(X_val)
    test_preds += model.predict(test_meta[feature_cols]) / n_splits

cv_rmse = float(np.sqrt(mean_squared_error(y, oof_preds)))
print(f"Validation RMSE: {cv_rmse:.5f}")

# 8. Save Submission Predictions
test_preds_clipped = np.clip(test_preds, 0.0, 5.0)
submission = pd.DataFrame(
    {"id": np.arange(len(test_preds_clipped)), "rating": test_preds_clipped}
)
submission.to_csv("./working/submission.csv", index=False)
