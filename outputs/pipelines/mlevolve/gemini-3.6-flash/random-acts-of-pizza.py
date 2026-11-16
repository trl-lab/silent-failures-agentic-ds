import json
import os
import random
import re
import numpy as np
import pandas as pd
from scipy.stats import rankdata

from scipy.sparse import hstack as sp_hstack
from scipy.stats import rankdata

from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier


# Ensure reproducible random seed across all libraries
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


seed_everything(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Ensure required directories exist
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# ---------------------------------------------------------
# STEP 1: DATA PROCESSING AND FEATURE ENGINEERING
# ---------------------------------------------------------
print("Loading raw parquet files...")
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

y_train = train_df["requester_received_pizza"].astype(int)

train_ids = (
    train_df["request_id"]
    if "request_id" in train_df.columns
    else pd.Series(train_df.index)
)
test_ids = (
    test_df["request_id"]
    if "request_id" in test_df.columns
    else pd.Series(test_df.index)
)


def subreddits_to_string(sr):
    if isinstance(sr, (list, np.ndarray)):
        return " ".join([str(x).lower() for x in sr])
    if isinstance(sr, str):
        return sr.lower()
    return ""


def extract_lexical_features(df):
    features = pd.DataFrame(index=df.index)

    title = df["request_title"].fillna("")
    body = (
        df["request_text_edit_aware"].fillna("")
        if "request_text_edit_aware" in df.columns
        else df["request_text"].fillna("")
    )

    features["title_char_len"] = title.apply(len)
    features["title_word_count"] = title.apply(lambda x: len(x.split()))
    features["body_char_len"] = body.apply(len)
    features["body_word_count"] = body.apply(lambda x: len(x.split()))
    features["title_body_ratio"] = features["body_word_count"] / (
        features["title_word_count"] + 1.0
    )

    # Readability and formatting statistics
    features["exclamation_count"] = body.apply(lambda x: x.count("!"))
    features["question_count"] = body.apply(lambda x: x.count("?"))
    features["caps_ratio_body"] = body.apply(
        lambda x: sum(1 for c in x if c.isupper()) / (len(x) + 1.0)
    )
    features["newline_count"] = body.apply(lambda x: x.count("\n"))
    features["paragraph_count"] = body.apply(lambda x: x.count("\n\n") + 1)
    features["digit_count"] = body.apply(lambda x: sum(1 for c in x if c.isdigit()))
    features["char_word_ratio"] = features["body_char_len"] / (
        features["body_word_count"] + 1.0
    )

    # Proof and link indicators
    proof_domains = [
        "imgur.com", "i.imgur", "gyazo.com", "puu.sh", "prntscr.com",
        "photobucket", "postimg", "tinypic", "reddit.com/r/",
        "drive.google.com", "dropbox.com"
    ]
    body_lower = body.str.lower()

    features["has_imgur"] = body_lower.apply(lambda x: float("imgur" in x))
    features["has_proof_link"] = body_lower.apply(
        lambda x: float(any(domain in x for domain in proof_domains))
    )
    features["link_count"] = body_lower.apply(
        lambda x: x.count("http://") + x.count("https://") + x.count("www.")
    )

    # First-person pronoun ratios
    first_person_words = ["i", "me", "my", "mine", "myself", "we", "us", "our", "ours", "ourselves"]
    features["first_person_count"] = body_lower.apply(
        lambda x: sum(len(re.findall(r"\b" + w + r"\b", x)) for w in first_person_words)
    )
    features["first_person_ratio"] = features["first_person_count"] / (
        features["body_word_count"] + 1.0
    )

    # Currency indicators
    currency_symbols = ["$", "usd", "dollar", "dollars", "buck", "bucks", "cent", "cents", "€", "£"]
    features["currency_count"] = body_lower.apply(
        lambda x: sum(x.count(c) for c in currency_symbols)
    )
    features["has_currency"] = (features["currency_count"] > 0).astype(float)

    # Narrative sentiment lexicons
    positive_sentiment_words = [
        "happy", "grateful", "blessed", "hopeful", "thankful", "kind", "love",
        "smile", "great", "wonderful", "appreciate", "sincerely", "honestly"
    ]
    negative_sentiment_words = [
        "sad", "sorry", "struggling", "depressed", "crying", "broke", "desperate",
        "hard", "tough", "bad", "pain", "hopeless", "sick", "lost", "awful", "hungry"
    ]

    features["pos_sentiment_count"] = body_lower.apply(
        lambda x: sum(x.count(w) for w in positive_sentiment_words)
    )
    features["neg_sentiment_count"] = body_lower.apply(
        lambda x: sum(x.count(w) for w in negative_sentiment_words)
    )
    features["sentiment_balance"] = (features["pos_sentiment_count"] - features["neg_sentiment_count"]) / (
        features["body_word_count"] + 1.0
    )

    # Expanded domain dictionaries
    polite_words = [
        "please", "thanks", "thank", "appreciate", "kind", "grateful", "bless", "thankful", "kindness"
    ]
    reciprocity_words = [
        "pay forward", "repay", "return", "next week", "payday", "tomorrow", "promise", "future"
    ]
    financial_need_words = [
        "broke", "hungry", "bank", "account", "money", "food", "eat", "empty", "paycheck",
        "bills", "rent", "debt", "overdrawn", "negative", "dollar", "afford", "starving", "groceries"
    ]
    student_words = [
        "student", "college", "university", "campus", "tuition", "dorm", "finals", "exams", "class", "school"
    ]
    employment_words = [
        "job", "work", "unemployed", "fired", "layoff", "working", "boss", "salary", "wage", "interview"
    ]
    urgency_words = [
        "tonight", "today", "now", "urgent", "asap", "immediately", "starving", "days"
    ]
    family_words = [
        "kids", "family", "children", "daughter", "son", "wife", "husband", "roommate", "friend", "baby", "parent"
    ]

    features["polite_count"] = body_lower.apply(
        lambda x: sum(x.count(w) for w in polite_words)
    )
    features["reciprocity_count"] = body_lower.apply(
        lambda x: sum(x.count(w) for w in reciprocity_words)
    )
    features["financial_need_count"] = body_lower.apply(
        lambda x: sum(x.count(w) for w in financial_need_words)
    )
    features["student_count"] = body_lower.apply(
        lambda x: sum(x.count(w) for w in student_words)
    )
    features["employment_count"] = body_lower.apply(
        lambda x: sum(x.count(w) for w in employment_words)
    )
    features["urgency_count"] = body_lower.apply(
        lambda x: sum(x.count(w) for w in urgency_words)
    )
    features["family_count"] = body_lower.apply(
        lambda x: sum(x.count(w) for w in family_words)
    )

    return features


def extract_user_features(df):
    features = pd.DataFrame(index=df.index)

    age = df["requester_account_age_in_days_at_request"].fillna(0.0)
    raop_age = df["requester_days_since_first_post_on_raop_at_request"].fillna(0.0)
    comments = df["requester_number_of_comments_at_request"].fillna(0.0)
    posts = df["requester_number_of_posts_at_request"].fillna(0.0)
    subreddits = df["requester_number_of_subreddits_at_request"].fillna(0.0)
    up_minus_down = df["requester_upvotes_minus_downvotes_at_request"].fillna(0.0)
    up_plus_down = df["requester_upvotes_plus_downvotes_at_request"].fillna(0.0)

    features["account_age_days"] = age
    features["raop_age_days"] = raop_age
    features["raop_tenure_ratio"] = raop_age / (age + 1.0)
    features["comments_count"] = comments
    features["posts_count"] = posts
    features["subreddits_count"] = subreddits
    features["total_activity"] = comments + posts
    features["posts_per_day"] = posts / (age + 1.0)
    features["comments_per_day"] = comments / (age + 1.0)

    features["upvotes_minus_downvotes"] = up_minus_down
    features["upvotes_plus_downvotes"] = up_plus_down
    features["upvote_ratio"] = (up_minus_down + up_plus_down) / (
        2.0 * up_plus_down + 1e-5
    )
    features["is_first_raop_post"] = (raop_age <= 0.0).astype(float)

    # Community ratio interaction features
    features["comment_post_ratio"] = comments / (posts + 1.0)
    features["upvote_comment_ratio"] = up_minus_down / (comments + 1.0)
    features["subreddit_post_ratio"] = subreddits / (posts + 1.0)
    features["activity_age_ratio"] = (comments + posts) / (age + 1.0)

    return features


def extract_temporal_features(df):
    features = pd.DataFrame(index=df.index)

    timestamps = pd.to_datetime(
        df["unix_timestamp_of_request_utc"], unit="s", errors="coerce"
    )

    hour = timestamps.dt.hour.fillna(12.0)
    dayofweek = timestamps.dt.dayofweek.fillna(0.0)
    month = timestamps.dt.month.fillna(1.0)

    features["hour"] = hour
    features["dayofweek"] = dayofweek
    features["month"] = month
    features["is_weekend"] = (dayofweek >= 5).astype(float)

    features["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    features["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    features["day_sin"] = np.sin(2 * np.pi * dayofweek / 7.0)
    features["day_cos"] = np.cos(2 * np.pi * dayofweek / 7.0)

    return features


def extract_subreddit_domain_features(df):
    features = pd.DataFrame(index=df.index)

    hardship_sr = {
        "assistance",
        "food_pantry",
        "borrow",
        "almosthomeless",
        "povertyfinance",
        "foodstamps",
        "needs",
        "cheapmeals",
    }
    gifting_sr = {
        "randomkindness",
        "random_acts_of_amazon",
        "giftofgames",
        "wishlist",
        "snackexchange",
        "random_acts_of_pizza",
    }

    sr_str_series = df["requester_subreddits_at_request"].apply(subreddits_to_string)

    features["hardship_sr_count"] = sr_str_series.apply(
        lambda s: sum(1 for w in s.split() if w in hardship_sr)
    )
    features["gifting_sr_count"] = sr_str_series.apply(
        lambda s: sum(1 for w in s.split() if w in gifting_sr)
    )

    return features, sr_str_series


print("Extracting domain-specific tabular features...")
train_lexical = extract_lexical_features(train_df)
test_lexical = extract_lexical_features(test_df)

train_user = extract_user_features(train_df)
test_user = extract_user_features(test_df)

train_temp = extract_temporal_features(train_df)
test_temp = extract_temporal_features(test_df)

train_sr_tax, train_sr_str = extract_subreddit_domain_features(train_df)
test_sr_tax, test_sr_str = extract_subreddit_domain_features(test_df)

# Extract top-100 subreddit binary presence features
top100_vec = CountVectorizer(max_features=100, binary=True)
train_top100 = pd.DataFrame(
    top100_vec.fit_transform(train_sr_str).toarray(),
    columns=[f"sr_top100_{w}" for w in top100_vec.get_feature_names_out()],
    index=train_df.index,
)
test_top100 = pd.DataFrame(
    top100_vec.transform(test_sr_str).toarray(),
    columns=[f"sr_top100_{w}" for w in top100_vec.get_feature_names_out()],
    index=test_df.index,
)

# Combine tabular features without global SVD fitting to avoid data leakage
X_train = pd.concat(
    [
        train_lexical,
        train_user,
        train_temp,
        train_sr_tax,
        train_top100,
    ],
    axis=1,
)
X_test = pd.concat(
    [test_lexical, test_user, test_temp, test_sr_tax, test_top100],
    axis=1,
)

train_full_text = (
    train_df["request_title"].fillna("")
    + " "
    + train_df["request_text_edit_aware"].fillna("")
).str.lower().values
test_full_text = (
    test_df["request_title"].fillna("")
    + " "
    + test_df["request_text_edit_aware"].fillna("")
).str.lower().values

train_sr_str_vals = train_sr_str.values
test_sr_str_vals = test_sr_str.values

X_train = X_train.fillna(0.0).replace([np.inf, -np.inf], 0.0)
X_test = X_test.fillna(0.0).replace([np.inf, -np.inf], 0.0)

print("Extracting text embeddings using Hugging Face Transformers...")

def extract_hf_embeddings(text_list, model_name="BAAI/bge-base-en-v1.5", batch_size=32):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    embeddings = []
    for i in range(0, len(text_list), batch_size):
        batch_texts = list(text_list[i:i + batch_size])
        inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            token_embeddings = outputs.last_hidden_state
            attention_mask = inputs["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * attention_mask, 1)
            sum_mask = torch.clamp(attention_mask.sum(1), min=1e-9)
            mean_pooled = sum_embeddings / sum_mask
            normalized = F.normalize(mean_pooled, p=2, dim=1)
            embeddings.append(normalized.cpu().numpy())
    return np.vstack(embeddings)

train_embeds = extract_hf_embeddings(train_full_text)
test_embeds = extract_hf_embeddings(test_full_text)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
train_folds = np.zeros(len(X_train), dtype=int)
for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    train_folds[val_idx] = fold

feature_cols = list(X_train.columns)

X_all = X_train.values
y_all = y_train.values
folds = train_folds
X_test_raw = X_test.values


# ---------------------------------------------------------
# STEP 2: MODEL ARCHITECTURE AND HYBRID TRAINING
# ---------------------------------------------------------
class TabularResNet(nn.Module):

    def __init__(self, input_dim):
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
        )
        self.res_block = nn.Sequential(
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
        )
        self.gelu = nn.GELU()
        self.layer128 = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.2),
        )
        self.layer64 = nn.Sequential(
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.2),
        )
        self.head = nn.Linear(64, 1)

    def forward(self, x):
        x = self.in_proj(x)
        x = self.gelu(x + self.res_block(x))
        x = self.layer128(x)
        x = self.layer64(x)
        out = torch.sigmoid(self.head(x))
        return out


model_names = ["lgb", "xgb", "cat", "et", "pytorch"]
oof_preds = {m: np.zeros(len(y_all)) for m in model_names}
test_preds = {m: np.zeros(len(X_test_raw)) for m in model_names}

n_folds = 5

print("Beginning 5-fold cross-validation training across 5 model families...")

for fold in range(n_folds):
    tr_idx = np.where(folds != fold)[0]
    val_idx = np.where(folds == fold)[0]

    # High-dimensional TF-IDF (word + char n-grams)
    text_tfidf = TfidfVectorizer(max_features=10000, stop_words="english", ngram_range=(1, 2))
    tr_word_vec = text_tfidf.fit_transform(train_full_text[tr_idx])
    val_word_vec = text_tfidf.transform(train_full_text[val_idx])
    te_word_vec = text_tfidf.transform(test_full_text)

    char_tfidf = TfidfVectorizer(max_features=10000, analyzer="char", ngram_range=(3, 5))
    tr_char_vec = char_tfidf.fit_transform(train_full_text[tr_idx])
    val_char_vec = char_tfidf.transform(train_full_text[val_idx])
    te_char_vec = char_tfidf.transform(test_full_text)

    tr_text_vec = sp_hstack([tr_word_vec, tr_char_vec]).tocsr()
    val_text_vec = sp_hstack([val_word_vec, val_char_vec]).tocsr()
    te_text_vec = sp_hstack([te_word_vec, te_char_vec]).tocsr()

    # Fold-fitted linear meta-features on high-dim TF-IDF + embeddings
    tr_highdim = sp_hstack([tr_text_vec, train_embeds[tr_idx]]).tocsr()
    val_highdim = sp_hstack([val_text_vec, train_embeds[val_idx]]).tocsr()
    te_highdim = sp_hstack([te_text_vec, test_embeds]).tocsr()

    lr_text = LogisticRegression(C=1.0, max_iter=1000, random_state=42 + fold)
    tr_lr_oof = cross_val_predict(lr_text, tr_highdim, y_all[tr_idx], cv=3, method="predict_proba")[:, 1:2]
    lr_text.fit(tr_highdim, y_all[tr_idx])
    val_lr_oof = lr_text.predict_proba(val_highdim)[:, 1:2]
    te_lr_oof = lr_text.predict_proba(te_highdim)[:, 1:2]

    ridge_text = RidgeClassifier(alpha=1.0, random_state=42 + fold)
    tr_ridge_oof = cross_val_predict(ridge_text, tr_highdim, y_all[tr_idx], cv=3, method="decision_function")[:, None]
    ridge_text.fit(tr_highdim, y_all[tr_idx])
    val_ridge_oof = ridge_text.decision_function(val_highdim)[:, None]
    te_ridge_oof = ridge_text.decision_function(te_highdim)[:, None]

    # Compress 768-dim embeddings using fold-fitted TruncatedSVD (32 components)
    embed_svd = TruncatedSVD(n_components=32, random_state=42 + fold)
    tr_embed_svd = embed_svd.fit_transform(train_embeds[tr_idx])
    val_embed_svd = embed_svd.transform(train_embeds[val_idx])
    te_embed_svd = embed_svd.transform(test_embeds)

    # TruncatedSVD features on text
    text_svd = TruncatedSVD(n_components=32, random_state=42 + fold)
    tr_text_svd = text_svd.fit_transform(tr_text_vec)
    val_text_svd = text_svd.transform(val_text_vec)
    te_text_svd = text_svd.transform(te_text_vec)

    sr_tfidf = TfidfVectorizer(max_features=1000, lowercase=True)
    sr_svd = TruncatedSVD(n_components=16, random_state=42 + fold)
    tr_sr_vec = sr_tfidf.fit_transform(train_sr_str_vals[tr_idx])
    val_sr_vec = sr_tfidf.transform(train_sr_str_vals[val_idx])
    te_sr_vec = sr_tfidf.transform(test_sr_str_vals)
    tr_sr_svd = sr_svd.fit_transform(tr_sr_vec)
    val_sr_svd = sr_svd.transform(val_sr_vec)
    te_sr_svd = sr_svd.transform(te_sr_vec)

    # Concatenate tabular, compressed SVD embeddings, linear meta-features, and text SVD
    X_tr = np.hstack([
        X_all[tr_idx],
        tr_embed_svd,
        tr_lr_oof,
        tr_ridge_oof,
        tr_text_svd,
        tr_sr_svd
    ])
    X_val = np.hstack([
        X_all[val_idx],
        val_embed_svd,
        val_lr_oof,
        val_ridge_oof,
        val_text_svd,
        val_sr_svd
    ])
    X_te = np.hstack([
        X_test_raw,
        te_embed_svd,
        te_lr_oof,
        te_ridge_oof,
        te_text_svd,
        te_sr_svd
    ])
    y_tr, y_val = y_all[tr_idx], y_all[val_idx]

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_val_s = scaler.transform(X_val)
    X_te_s = scaler.transform(X_te)

    # 1. LightGBM Classifier
    lgb_model = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.5,
        reg_alpha=1.0,
        reg_lambda=2.0,
        random_state=42 + fold,
        verbose=-1,
    )
    lgb_model.fit(
        X_tr_s,
        y_tr,
        eval_set=[(X_val_s, y_val)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(0),
        ],
    )
    oof_preds["lgb"][val_idx] = lgb_model.predict_proba(X_val_s)[:, 1]
    test_preds["lgb"] += lgb_model.predict_proba(X_te_s)[:, 1] / n_folds

    # 2. XGBoost Classifier
    xgb_model = xgb.XGBClassifier(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.5,
        reg_alpha=1.0,
        reg_lambda=2.0,
        early_stopping_rounds=50,
        eval_metric="auc",
        random_state=42 + fold,
    )
    xgb_model.fit(X_tr_s, y_tr, eval_set=[(X_val_s, y_val)], verbose=False)
    oof_preds["xgb"][val_idx] = xgb_model.predict_proba(X_val_s)[:, 1]
    test_preds["xgb"] += xgb_model.predict_proba(X_te_s)[:, 1] / n_folds

    # 3. CatBoost Classifier
    cat_model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.03,
        depth=5,
        rsm=0.5,
        l2_leaf_reg=3.0,
        eval_metric="AUC",
        early_stopping_rounds=50,
        random_seed=42 + fold,
        verbose=0,
    )
    cat_model.fit(X_tr_s, y_tr, eval_set=(X_val_s, y_val), verbose=False)
    oof_preds["cat"][val_idx] = cat_model.predict_proba(X_val_s)[:, 1]
    test_preds["cat"] += cat_model.predict_proba(X_te_s)[:, 1] / n_folds

    # 4. ExtraTrees Classifier
    et_model = ExtraTreesClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_split=5,
        random_state=42 + fold,
        n_jobs=-1,
    )
    et_model.fit(X_tr_s, y_tr)
    oof_preds["et"][val_idx] = et_model.predict_proba(X_val_s)[:, 1]
    test_preds["et"] += et_model.predict_proba(X_te_s)[:, 1] / n_folds

    # 5. PyTorch TabularResNet
    train_dataset = TensorDataset(
        torch.tensor(X_tr_s, dtype=torch.float32),
        torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1),
    )
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    net = TabularResNet(input_dim=X_tr_s.shape[1]).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=35)

    best_val_auc = 0.0
    best_val_preds = None
    best_te_preds = None

    for epoch in range(35):
        net.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            out = net(batch_x)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()
        scheduler.step()

        net.eval()
        with torch.no_grad():
            val_x_t = torch.tensor(X_val_s, dtype=torch.float32).to(device)
            val_out = net(val_x_t).cpu().numpy().flatten()
            val_auc = roc_auc_score(y_val, val_out)

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_val_preds = val_out
                te_x_t = torch.tensor(X_te_s, dtype=torch.float32).to(device)
                best_te_preds = net(te_x_t).cpu().numpy().flatten()

    oof_preds["pytorch"][val_idx] = best_val_preds
    test_preds["pytorch"] += best_te_preds / n_folds

    fold_lgb_auc = roc_auc_score(y_val, oof_preds["lgb"][val_idx])
    fold_xgb_auc = roc_auc_score(y_val, oof_preds["xgb"][val_idx])
    fold_cat_auc = roc_auc_score(y_val, oof_preds["cat"][val_idx])
    print(
        f"Fold {fold+1}/{n_folds} - LGB: {fold_lgb_auc:.4f}, XGB: {fold_xgb_auc:.4f}, CAT: {fold_cat_auc:.4f}, PyTorch: {best_val_auc:.4f}"
    )

print("\n--- Individual Out-of-Fold Model ROC-AUC Scores ---")
for m in model_names:
    score = roc_auc_score(y_all, oof_preds[m])
    print(f"Model [{m.upper()}]: OOF ROC-AUC = {score:.5f}")

# ---------------------------------------------------------
# STEP 3: META-LEARNER ENSEMBLE BLENDING & SUBMISSION
# ---------------------------------------------------------
# Collect OOF probability matrices and rank-transformed versions
oof_matrix = np.column_stack([oof_preds[m] for m in model_names])
test_matrix = np.column_stack([test_preds[m] for m in model_names])

oof_ranks = np.column_stack(
    [rankdata(oof_preds[m]) / len(oof_preds[m]) for m in model_names]
)
test_ranks = np.column_stack(
    [rankdata(test_preds[m]) / len(test_preds[m]) for m in model_names]
)

# Ridge classification meta-learner with positive non-negative weights constraint on probability and rank meta-features
meta_model = RidgeClassifier(alpha=10.0, positive=True, random_state=42)
meta_model.fit(np.hstack([oof_matrix, oof_ranks]), y_all)

final_oof_scores = meta_model.decision_function(np.hstack([oof_matrix, oof_ranks]))
final_test_scores = meta_model.decision_function(np.hstack([test_matrix, test_ranks]))

# Rank normalize final test predictions to preserve uniform output distribution
final_test_preds = rankdata(final_test_scores) / len(final_test_scores)
final_val_score = roc_auc_score(y_all, final_oof_scores)

submission_df = pd.DataFrame(
    {
        "id": np.arange(len(final_test_preds)),
        "requester_received_pizza": final_test_preds,
    }
)
submission_df.to_csv("./submission/submission.csv", index=False)

print(f"Final Validation Score: {final_val_score}")