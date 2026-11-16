import os
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import warnings

warnings.filterwarnings("ignore")

os.makedirs("./working", exist_ok=True)

# 1. Load Data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "requester_received_pizza"
y = train_df[target_col].astype(int).values
n_train = len(train_df)
n_test = len(test_df)

combined_df = pd.concat(
    [train_df.drop(columns=[target_col], errors="ignore"), test_df],
    ignore_index=True,
)

# 2. Text Features
text_title = combined_df["request_title"].fillna("").astype(str)
if "request_text_edit_aware" in combined_df.columns:
    text_body = combined_df["request_text_edit_aware"].fillna("").astype(str)
elif "request_text" in combined_df.columns:
    text_body = combined_df["request_text"].fillna("").astype(str)
else:
    text_body = pd.Series([""] * len(combined_df))

full_text = text_title + " " + text_body

features_df = pd.DataFrame()

# Text length and formatting features
features_df["title_char_len"] = text_title.apply(len)
features_df["body_char_len"] = text_body.apply(len)
features_df["full_char_len"] = full_text.apply(len)
features_df["title_word_count"] = text_title.apply(lambda x: len(x.split()))
features_df["body_word_count"] = text_body.apply(lambda x: len(x.split()))
features_df["caps_ratio"] = full_text.apply(
    lambda x: sum(1 for c in x if c.isupper()) / (len(x) + 1)
)
features_df["digit_ratio"] = full_text.apply(
    lambda x: sum(1 for c in x if c.isdigit()) / (len(x) + 1)
)

# Keyword & heuristic indicators
full_text_lower = full_text.str.lower()
features_df["has_question"] = full_text.apply(lambda x: 1 if "?" in x else 0)
features_df["has_exclamation"] = full_text.apply(lambda x: 1 if "!" in x else 0)
features_df["has_proof_link"] = full_text_lower.apply(
    lambda x: (
        1
        if any(k in x for k in ["http", "imgur", "jpg", "png", "proof", "photo"])
        else 0
    )
)
features_df["has_reciprocity"] = full_text_lower.apply(
    lambda x: (
        1
        if any(
            k in x
            for k in [
                "pay it forward",
                "return the favor",
                "pay forward",
                "repay",
                "will pay",
            ]
        )
        else 0
    )
)
features_df["has_gratitude"] = full_text_lower.apply(
    lambda x: (
        1
        if any(
            k in x for k in ["thank", "thanks", "appreciate", "grateful", "god bless"]
        )
        else 0
    )
)
features_df["has_need_words"] = full_text_lower.apply(
    lambda x: (
        1
        if any(
            k in x
            for k in [
                "broke",
                "hungry",
                "paycheck",
                "unemployed",
                "rent",
                "student",
                "fridge",
                "empty",
                "starving",
            ]
        )
        else 0
    )
)

# TF-IDF + SVD on request text
tfidf_text = TfidfVectorizer(
    max_features=25000,
    ngram_range=(1, 2),
    stop_words="english",
    min_df=2,
    sublinear_tf=True,
)
tfidf_text_mat = tfidf_text.fit_transform(full_text)
svd_text = TruncatedSVD(n_components=35, random_state=42)
svd_text_feats = svd_text.fit_transform(tfidf_text_mat)

for i in range(svd_text_feats.shape[1]):
    features_df[f"svd_text_{i}"] = svd_text_feats[:, i]

# 3. Subreddit History Embeddings
if "requester_subreddits_at_request" in combined_df.columns:
    subreddit_strings = combined_df["requester_subreddits_at_request"].apply(
        lambda x: (
            " ".join(x)
            if isinstance(x, (list, np.ndarray))
            else str(x) if pd.notnull(x) else ""
        )
    )
    tfidf_sub = TfidfVectorizer(max_features=5000, min_df=2, binary=True)
    tfidf_sub_mat = tfidf_sub.fit_transform(subreddit_strings)
    svd_sub = TruncatedSVD(n_components=25, random_state=42)
    svd_sub_feats = svd_sub.fit_transform(tfidf_sub_mat)

    for i in range(svd_sub_feats.shape[1]):
        features_df[f"svd_sub_{i}"] = svd_sub_feats[:, i]

# 4. Tabular User Activity Features
numeric_cols = [
    "requester_account_age_in_days_at_request",
    "requester_days_since_first_post_on_raop_at_request",
    "requester_number_of_comments_at_request",
    "requester_number_of_comments_in_raop_at_request",
    "requester_number_of_posts_at_request",
    "requester_number_of_posts_on_raop_at_request",
    "requester_number_of_subreddits_at_request",
    "requester_upvotes_minus_downvotes_at_request",
    "requester_upvotes_plus_downvotes_at_request",
]

for col in numeric_cols:
    if col in combined_df.columns:
        features_df[col] = (
            pd.to_numeric(combined_df[col], errors="coerce").fillna(0).astype(float)
        )

# Derived ratio and interaction features
upvotes_plus = features_df.get(
    "requester_upvotes_plus_downvotes_at_request",
    pd.Series(0.0, index=features_df.index),
)
upvotes_diff = features_df.get(
    "requester_upvotes_minus_downvotes_at_request",
    pd.Series(0.0, index=features_df.index),
)
acct_age = features_df.get(
    "requester_account_age_in_days_at_request",
    pd.Series(0.0, index=features_df.index),
)
num_comments = features_df.get(
    "requester_number_of_comments_at_request",
    pd.Series(0.0, index=features_df.index),
)
num_posts = features_df.get(
    "requester_number_of_posts_at_request",
    pd.Series(0.0, index=features_df.index),
)

features_df["upvote_ratio"] = np.where(
    upvotes_plus > 0,
    (upvotes_diff + upvotes_plus) / (2.0 * upvotes_plus + 1e-5),
    0.5,
)
features_df["comments_per_day"] = num_comments / (acct_age + 1.0)
features_df["posts_per_day"] = num_posts / (acct_age + 1.0)
features_df["total_activity"] = num_comments + num_posts

# 5. Model Training & Out-of-Fold Evaluation
X_train = features_df.iloc[:n_train].copy().fillna(0).values
X_test = features_df.iloc[n_train:].copy().fillna(0).values

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_lgb = np.zeros(n_train)
oof_lr = np.zeros(n_train)
test_preds_lgb = np.zeros(n_test)
test_preds_lr = np.zeros(n_test)

lgb_params = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "learning_rate": 0.03,
    "num_leaves": 15,
    "max_depth": 4,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "min_child_samples": 20,
    "reg_alpha": 0.5,
    "reg_lambda": 1.0,
    "random_state": 42,
    "verbose": -1,
    "n_jobs": -1,
}

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y)):
    X_tr, y_tr = X_train[train_idx], y[train_idx]
    X_va, y_val = X_train[val_idx], y[val_idx]

    X_tr_sc, X_va_sc = X_train_scaled[train_idx], X_train_scaled[val_idx]

    # LightGBM
    dtrain = lgb.Dataset(X_tr, label=y_tr)
    dval = lgb.Dataset(X_va, label=y_val, reference=dtrain)

    model_lgb = lgb.train(
        lgb_params,
        dtrain,
        num_boost_round=1000,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )

    oof_lgb[val_idx] = model_lgb.predict(X_va, num_iteration=model_lgb.best_iteration)
    test_preds_lgb += (
        model_lgb.predict(X_test, num_iteration=model_lgb.best_iteration) / n_splits
    )

    # Logistic Regression
    lr = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
    lr.fit(X_tr_sc, y_tr)
    oof_lr[val_idx] = lr.predict_proba(X_va_sc)[:, 1]
    test_preds_lr += lr.predict_proba(X_test_scaled)[:, 1] / n_splits

# Ensemble blend (0.6 LightGBM + 0.4 Logistic Regression)
oof_blend = 0.6 * oof_lgb + 0.4 * oof_lr
final_test_preds = 0.6 * test_preds_lgb + 0.4 * test_preds_lr

cv_auc = roc_auc_score(y, oof_blend)
print(f"5-Fold CV ROC-AUC: {cv_auc:.5f}")

# 6. Generate Submission
submission_df = pd.DataFrame(
    {
        "id": np.arange(len(test_df)),
        "requester_received_pizza": final_test_preds,
    }
)
submission_df.to_csv("./working/submission.csv", index=False)
print("Submission saved successfully to ./working/submission.csv")
