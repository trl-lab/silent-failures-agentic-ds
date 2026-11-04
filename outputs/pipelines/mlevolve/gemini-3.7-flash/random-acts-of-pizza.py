import copy
import json
import os
import re
from catboost import CatBoostClassifier
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModel, AutoTokenizer

# Set deterministic random seeds
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using compute device: {device}")

# Create output directories
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Load Raw Datasets
# ---------------------------------------------------------------------------
print("Loading raw parquet datasets...")
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "requester_received_pizza"
if target_col in train_df.columns:
    y_train = train_df[target_col].astype(int).values
else:
    raise ValueError(f"Target column '{target_col}' not found in train data.")

print(f"Train samples: {len(train_df)}, Test samples: {len(test_df)}")
print(f"Target positive rate: {y_train.mean():.4f}")


# ---------------------------------------------------------------------------
# 2. Text Preprocessing & Lexicon Feature Extraction
# ---------------------------------------------------------------------------
def clean_and_combine_text(df: pd.DataFrame) -> pd.Series:
    title = df.get("request_title", pd.Series([""] * len(df))).fillna("").astype(str)
    if "request_text_edit_aware" in df.columns:
        text = df["request_text_edit_aware"].fillna("").astype(str)
    elif "request_text" in df.columns:
        text = df["request_text"].fillna("").astype(str)
    else:
        text = pd.Series([""] * len(df))
    return (title + " " + text).str.strip()


def parse_subreddits(val) -> str:
    if isinstance(val, (list, np.ndarray)):
        return " ".join([str(x).strip() for x in val if x])
    elif isinstance(val, str):
        if val.startswith("[") and val.endswith("]"):
            try:
                items = json.loads(val.replace("'", '"'))
                return " ".join([str(x).strip() for x in items if x])
            except Exception:
                return re.sub(r"[\[\]',]", " ", val).strip()
        return val.strip()
    return ""


GRATITUDE_WORDS = {
    "thank",
    "thanks",
    "thankful",
    "grateful",
    "appreciate",
    "appreciated",
    "bless",
    "blessing",
    "god",
}
RECIPROCITY_WORDS = {
    "pay",
    "forward",
    "return",
    "repay",
    "favor",
    "paycheck",
    "payday",
    "next",
    "week",
    "soon",
    "promise",
}
URGENCY_NEED_WORDS = {
    "hungry",
    "starving",
    "empty",
    "fridge",
    "broke",
    "struggling",
    "lost",
    "job",
    "rent",
    "ramen",
    "cents",
    "dollar",
    "bills",
    "kids",
    "family",
    "foodbank",
    "pantry",
    "starve",
    "eviction",
    "overdraft",
}
POLITENESS_WORDS = {
    "please",
    "kindly",
    "generous",
    "kindness",
    "anyone",
    "willing",
    "help",
    "awesome",
    "friendly",
}
STUDENT_WORDS = {
    "student",
    "college",
    "university",
    "finals",
    "exam",
    "tuition",
    "campus",
    "dorm",
    "semester",
}
UNEMPLOYMENT_WORDS = {
    "unemployed",
    "laid",
    "fired",
    "jobless",
    "interview",
    "hire",
    "unemployment",
    "severance",
}
PROOF_WORDS = {
    "proof",
    "verification",
    "verify",
    "picture",
    "photo",
    "receipt",
    "pic",
    "camera",
    "timestamp",
}


def extract_text_meta_features(
    text_series: pd.Series, prefix: str = "text_"
) -> pd.DataFrame:
    features = {}
    char_len = text_series.apply(len)
    word_count = text_series.apply(lambda x: len(x.split()))
    features[f"{prefix}char_len"] = char_len
    features[f"{prefix}word_count"] = word_count
    features[f"{prefix}mean_word_len"] = char_len / (word_count + 1)

    features[f"{prefix}caps_count"] = text_series.apply(
        lambda x: sum(1 for c in x if c.isupper())
    )
    features[f"{prefix}caps_ratio"] = features[f"{prefix}caps_count"] / (char_len + 1)
    features[f"{prefix}exclamation_count"] = text_series.apply(lambda x: x.count("!"))
    features[f"{prefix}question_count"] = text_series.apply(lambda x: x.count("?"))
    features[f"{prefix}has_url"] = text_series.apply(
        lambda x: int("http" in x.lower() or "www." in x.lower())
    )
    features[f"{prefix}has_proof_link"] = text_series.apply(
        lambda x: int(
            any(
                img_host in x.lower()
                for img_host in ["imgur", "gyazo", "tinypic", "photobucket", "i.redd.it", "reddit.com/gallery"]
            )
        )
    )

    lower_texts = text_series.str.lower()

    def count_matches(s: str, vocab: set) -> int:
        words = re.findall(r"\b\w+\b", s)
        return sum(1 for w in words if w in vocab)

    features[f"{prefix}gratitude_count"] = lower_texts.apply(
        lambda x: count_matches(x, GRATITUDE_WORDS)
    )
    features[f"{prefix}reciprocity_count"] = lower_texts.apply(
        lambda x: count_matches(x, RECIPROCITY_WORDS)
    )
    features[f"{prefix}urgency_count"] = lower_texts.apply(
        lambda x: count_matches(x, URGENCY_NEED_WORDS)
    )
    features[f"{prefix}politeness_count"] = lower_texts.apply(
        lambda x: count_matches(x, POLITENESS_WORDS)
    )
    features[f"{prefix}student_count"] = lower_texts.apply(
        lambda x: count_matches(x, STUDENT_WORDS)
    )
    features[f"{prefix}unemployment_count"] = lower_texts.apply(
        lambda x: count_matches(x, UNEMPLOYMENT_WORDS)
    )
    features[f"{prefix}proof_count"] = lower_texts.apply(
        lambda x: count_matches(x, PROOF_WORDS)
    )

    features[f"{prefix}reciprocity_ratio"] = features[f"{prefix}reciprocity_count"] / (
        word_count + 1
    )
    features[f"{prefix}gratitude_ratio"] = features[f"{prefix}gratitude_count"] / (
        word_count + 1
    )
    features[f"{prefix}urgency_ratio"] = features[f"{prefix}urgency_count"] / (
        word_count + 1
    )
    features[f"{prefix}politeness_ratio"] = features[f"{prefix}politeness_count"] / (
        word_count + 1
    )
    features[f"{prefix}hardship_total"] = (
        features[f"{prefix}urgency_count"]
        + features[f"{prefix}unemployment_count"]
        + features[f"{prefix}student_count"]
    )
    features[f"{prefix}hardship_ratio"] = features[f"{prefix}hardship_total"] / (
        word_count + 1
    )

    return pd.DataFrame(features)


# ---------------------------------------------------------------------------
# 3. User History and Tabular Feature Engineering
# ---------------------------------------------------------------------------
def extract_tabular_features(df: pd.DataFrame) -> pd.DataFrame:
    feats = pd.DataFrame(index=df.index)

    account_age = (
        df.get("requester_account_age_in_days_at_request", pd.Series(0, index=df.index))
        .fillna(0)
        .astype(float)
    )
    feats["account_age_days"] = account_age
    feats["log_account_age_days"] = np.log1p(np.maximum(0, account_age))
    feats["is_brand_new_account"] = (account_age < 3).astype(int)
    feats["is_veteran_account"] = (account_age > 365).astype(int)

    up_minus_down = (
        df.get(
            "requester_upvotes_minus_downvotes_at_request", pd.Series(0, index=df.index)
        )
        .fillna(0)
        .astype(float)
    )
    up_plus_down = (
        df.get(
            "requester_upvotes_plus_downvotes_at_request", pd.Series(0, index=df.index)
        )
        .fillna(0)
        .astype(float)
    )
    feats["karma_diff"] = up_minus_down
    feats["karma_total"] = up_plus_down
    feats["log_karma_total"] = np.log1p(np.maximum(0, up_plus_down))
    feats["karma_ratio"] = (up_minus_down + 10.0) / (up_plus_down + 20.0)

    comments = (
        df.get("requester_number_of_comments_at_request", pd.Series(0, index=df.index))
        .fillna(0)
        .astype(float)
    )
    posts = (
        df.get("requester_number_of_posts_at_request", pd.Series(0, index=df.index))
        .fillna(0)
        .astype(float)
    )
    feats["num_comments"] = comments
    feats["num_posts"] = posts
    feats["log_comments"] = np.log1p(np.maximum(0, comments))
    feats["log_posts"] = np.log1p(np.maximum(0, posts))

    feats["comments_per_day"] = comments / (account_age + 1.0)
    feats["posts_per_day"] = posts / (account_age + 1.0)
    feats["total_activity_per_day"] = (comments + posts) / (account_age + 1.0)

    raop_comments = (
        df.get(
            "requester_number_of_comments_in_raop_at_request",
            pd.Series(0, index=df.index),
        )
        .fillna(0)
        .astype(float)
    )
    raop_posts = (
        df.get(
            "requester_number_of_posts_on_raop_at_request", pd.Series(0, index=df.index)
        )
        .fillna(0)
        .astype(float)
    )
    feats["raop_comments"] = raop_comments
    feats["raop_posts"] = raop_posts
    feats["raop_comment_ratio"] = raop_comments / (comments + 1.0)
    feats["raop_post_ratio"] = raop_posts / (posts + 1.0)
    feats["has_prior_raop_activity"] = ((raop_comments > 0) | (raop_posts > 0)).astype(
        int
    )

    days_since_first_raop = (
        df.get(
            "requester_days_since_first_post_on_raop_at_request",
            pd.Series(0, index=df.index),
        )
        .fillna(0)
        .astype(float)
    )
    feats["days_since_first_raop"] = days_since_first_raop
    feats["log_days_since_first_raop"] = np.log1p(np.maximum(0, days_since_first_raop))

    num_subreddits = (
        df.get(
            "requester_number_of_subreddits_at_request", pd.Series(0, index=df.index)
        )
        .fillna(0)
        .astype(float)
    )
    feats["num_subreddits"] = num_subreddits
    feats["log_num_subreddits"] = np.log1p(np.maximum(0, num_subreddits))
    feats["posts_per_subreddit"] = posts / (num_subreddits + 1.0)
    feats["comments_per_subreddit"] = comments / (num_subreddits + 1.0)

    timestamp_utc = df.get(
        "unix_timestamp_of_request_utc",
        df.get("unix_timestamp_of_request", pd.Series(0, index=df.index)),
    ).fillna(0)
    dt = pd.to_datetime(timestamp_utc, unit="s", errors="coerce")

    hour = dt.dt.hour.fillna(12).astype(float)
    day = dt.dt.day.fillna(15).astype(float)
    dayofweek = dt.dt.dayofweek.fillna(3).astype(float)
    month = dt.dt.month.fillna(6).astype(float)

    feats["hour"] = hour
    feats["day"] = day
    feats["dayofweek"] = dayofweek
    feats["month"] = month
    feats["is_weekend"] = (dayofweek >= 5).astype(int)

    feats["is_month_start"] = (day <= 5).astype(int)
    feats["is_month_mid"] = ((day >= 13) & (day <= 17)).astype(int)
    feats["is_month_end"] = (day >= 26).astype(int)

    feats["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    feats["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    feats["dow_sin"] = np.sin(2 * np.pi * dayofweek / 7.0)
    feats["dow_cos"] = np.cos(2 * np.pi * dayofweek / 7.0)

    return feats


# ---------------------------------------------------------------------------
# 4. Dense Contextual Text Embeddings (Transformer) & Lexical Features
# ---------------------------------------------------------------------------
print("Extracting metadata, lexical, and tabular features...")
train_text = clean_and_combine_text(train_df)
test_text = clean_and_combine_text(test_df)

train_title = (
    train_df.get("request_title", pd.Series([""] * len(train_df)))
    .fillna("")
    .astype(str)
)
test_title = (
    test_df.get("request_title", pd.Series([""] * len(test_df))).fillna("").astype(str)
)

train_text_meta = extract_text_meta_features(train_text, prefix="body_")
test_text_meta = extract_text_meta_features(test_text, prefix="body_")

train_title_meta = extract_text_meta_features(train_title, prefix="title_")
test_title_meta = extract_text_meta_features(test_title, prefix="title_")

train_tab_feats = extract_tabular_features(train_df)
test_tab_feats = extract_tabular_features(test_df)

train_subreddits_str = train_df.get(
    "requester_subreddits_at_request", pd.Series([""] * len(train_df))
).apply(parse_subreddits)
test_subreddits_str = test_df.get(
    "requester_subreddits_at_request", pd.Series([""] * len(test_df))
).apply(parse_subreddits)


def extract_transformer_embeddings(
    texts: pd.Series,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 64,
    device: torch.device = device,
) -> np.ndarray:
    print(f"Extracting transformer embeddings using {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    text_list = texts.fillna("").astype(str).tolist()
    all_embeddings = []

    for i in range(0, len(text_list), batch_size):
        batch = text_list[i : i + batch_size]
        batch = [t if len(t.strip()) > 0 else "empty" for t in batch]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model(**encoded)
            attention_mask = encoded["attention_mask"].unsqueeze(-1)
            token_embeddings = outputs.last_hidden_state
            sum_embeddings = (token_embeddings * attention_mask).sum(dim=1)
            sum_mask = attention_mask.sum(dim=1).clamp(min=1e-9)
            mean_pooled = sum_embeddings / sum_mask
            normalized = nn.functional.normalize(mean_pooled, p=2, dim=1)
            all_embeddings.append(normalized.cpu().numpy())

    return np.vstack(all_embeddings).astype(np.float32)


# Extract dense contextual representations
train_text_emb = extract_transformer_embeddings(train_text)
test_text_emb = extract_transformer_embeddings(test_text)

train_sub_emb = extract_transformer_embeddings(train_subreddits_str)
test_sub_emb = extract_transformer_embeddings(test_subreddits_str)

emb_dim = train_text_emb.shape[1]
text_emb_cols = [f"text_emb_{i}" for i in range(emb_dim)]
sub_emb_cols = [f"sub_emb_{i}" for i in range(emb_dim)]

train_text_emb_df = pd.DataFrame(
    train_text_emb, columns=text_emb_cols, index=train_df.index
)
test_text_emb_df = pd.DataFrame(
    test_text_emb, columns=text_emb_cols, index=test_df.index
)

train_sub_emb_df = pd.DataFrame(
    train_sub_emb, columns=sub_emb_cols, index=train_df.index
)
test_sub_emb_df = pd.DataFrame(test_sub_emb, columns=sub_emb_cols, index=test_df.index)

print("Fitting TF-IDF and SVD on subreddit history strictly without leakage...")
tfidf_sub = TfidfVectorizer(
    max_features=2000, ngram_range=(1, 1), token_pattern=r"(?u)\b[\w-]+\b", min_df=2
)
svd_sub = TruncatedSVD(n_components=25, random_state=42)

train_sub_mat = tfidf_sub.fit_transform(train_subreddits_str)
test_sub_mat = tfidf_sub.transform(test_subreddits_str)

train_sub_svd = svd_sub.fit_transform(train_sub_mat)
test_sub_svd = svd_sub.transform(test_sub_mat)

sub_svd_cols = [f"sub_svd_{i}" for i in range(25)]
train_sub_svd_df = pd.DataFrame(
    train_sub_svd, columns=sub_svd_cols, index=train_df.index
)
test_sub_svd_df = pd.DataFrame(test_sub_svd, columns=sub_svd_cols, index=test_df.index)

X_train = pd.concat(
    [
        train_tab_feats,
        train_text_meta,
        train_title_meta,
        train_text_emb_df,
        train_sub_svd_df,
    ],
    axis=1,
)
X_test = pd.concat(
    [
        test_tab_feats,
        test_text_meta,
        test_title_meta,
        test_text_emb_df,
        test_sub_svd_df,
    ],
    axis=1,
)

X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0.0)
X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(0.0)

feature_cols = list(X_train.columns)
n_features = len(feature_cols)
print(f"Total features assembled: {n_features}")

# 5-fold Stratified CV split
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
folds = np.zeros(len(train_df), dtype=int)
for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    folds[val_idx] = fold

n_folds = 5


# ---------------------------------------------------------------------------
# 5. Multi-Paradigm Model Architectures
# ---------------------------------------------------------------------------
class TabularResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout_rate: float = 0.25):
        super(TabularResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout_rate),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class ResidualPizzaMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 192,
        num_blocks: int = 2,
        dropout_rate: float = 0.25,
    ):
        super(ResidualPizzaMLP, self).__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )
        self.blocks = nn.ModuleList(
            [
                TabularResidualBlock(hidden_dim, dropout_rate=dropout_rate)
                for _ in range(num_blocks)
            ]
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout_rate / 2.0),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_layer(x)
        for block in self.blocks:
            h = block(h)
        return self.head(h).squeeze(-1)


def get_residual_mlp_components(
    input_dim: int,
    hidden_dim: int = 192,
    num_blocks: int = 2,
    dropout_rate: float = 0.25,
    lr: float = 8e-4,
    weight_decay: float = 1e-3,
    num_epochs: int = 35,
):
    model = ResidualPizzaMLP(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_blocks=num_blocks,
        dropout_rate=dropout_rate,
    )
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999)
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    return model, criterion, optimizer, scheduler


# ---------------------------------------------------------------------------
# 6. Stratified 5-Fold Multi-Model Training & Inference
# ---------------------------------------------------------------------------
print("Starting 5-fold cross-validation training across 4 model families...")
oof_lgb = np.zeros(len(X_train))
oof_cat = np.zeros(len(X_train))
oof_lr = np.zeros(len(X_train))
oof_mlp = np.zeros(len(X_train))

test_preds_lgb = np.zeros(len(X_test))
test_preds_cat = np.zeros(len(X_test))
test_preds_lr = np.zeros(len(X_test))
test_preds_mlp = np.zeros(len(X_test))

X_test_feat = X_test[feature_cols].values.astype(np.float32)

for fold in range(n_folds):
    train_mask = folds != fold
    val_mask = folds == fold

    X_tr = X_train.loc[train_mask, feature_cols].values.astype(np.float32)
    y_tr = y_train[train_mask].astype(float)
    X_va = X_train.loc[val_mask, feature_cols].values.astype(np.float32)
    y_va = y_train[val_mask].astype(float)

    # 1) LightGBM
    lgb_params = {
        "n_estimators": 700,
        "learning_rate": 0.025,
        "num_leaves": 15,
        "max_depth": 4,
        "min_child_samples": 30,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.6,
        "reg_alpha": 1.0,
        "reg_lambda": 2.5,
        "objective": "binary",
        "random_state": 42 + fold,
        "n_jobs": -1,
        "verbose": -1,
    }
    clf_lgb = lgb.LGBMClassifier(**lgb_params)
    callbacks = [lgb.early_stopping(40, verbose=False), lgb.log_evaluation(period=0)]
    clf_lgb.fit(
        X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric="auc", callbacks=callbacks
    )

    oof_lgb[val_mask] = clf_lgb.predict_proba(X_va)[:, 1]
    test_preds_lgb += clf_lgb.predict_proba(X_test_feat)[:, 1] / n_folds

    # 2) CatBoost Classifier
    clf_cat = CatBoostClassifier(
        iterations=750,
        learning_rate=0.03,
        depth=4,
        l2_leaf_reg=4.0,
        random_seed=42 + fold,
        eval_metric="AUC",
        verbose=False,
        early_stopping_rounds=40,
    )
    clf_cat.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=False)

    oof_cat[val_mask] = clf_cat.predict_proba(X_va)[:, 1]
    test_preds_cat += clf_cat.predict_proba(X_test_feat)[:, 1] / n_folds

    # 3) Regularized Logistic Regression on Standardized Multimodal Features
    scaler_lr = StandardScaler()
    X_tr_scaled = scaler_lr.fit_transform(X_tr)
    X_va_scaled = scaler_lr.transform(X_va)
    X_te_scaled = scaler_lr.transform(X_test_feat)

    clf_lr = LogisticRegression(
        C=0.05, penalty="l2", solver="lbfgs", max_iter=1000, random_state=42 + fold
    )
    clf_lr.fit(X_tr_scaled, y_tr)

    oof_lr[val_mask] = clf_lr.predict_proba(X_va_scaled)[:, 1]
    test_preds_lr += clf_lr.predict_proba(X_te_scaled)[:, 1] / n_folds

    # 4) Residual PizzaMLP (PyTorch)
    scaler_mlp = StandardScaler()
    X_tr_mlp = scaler_mlp.fit_transform(X_tr)
    X_va_mlp = scaler_mlp.transform(X_va)
    X_te_mlp = scaler_mlp.transform(X_test_feat)

    train_ds = TensorDataset(
        torch.tensor(X_tr_mlp, dtype=torch.float32),
        torch.tensor(y_tr, dtype=torch.float32),
    )
    val_ds = TensorDataset(
        torch.tensor(X_va_mlp, dtype=torch.float32),
        torch.tensor(y_va, dtype=torch.float32),
    )

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)

    mlp_model, mlp_criterion, mlp_opt, mlp_sched = get_residual_mlp_components(
        input_dim=n_features,
        hidden_dim=192,
        num_blocks=2,
        dropout_rate=0.25,
        lr=8e-4,
        weight_decay=1e-3,
        num_epochs=35,
    )
    mlp_model.to(device)

    best_val_auc = 0.0
    best_weights = None

    for epoch in range(1, 36):
        mlp_model.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            mlp_opt.zero_grad()
            logits = mlp_model(bx)
            loss = mlp_criterion(logits, by)
            loss.backward()
            mlp_opt.step()
        mlp_sched.step()

        mlp_model.eval()
        val_preds_list = []
        with torch.no_grad():
            for bx, _ in val_loader:
                bx = bx.to(device)
                val_probs = torch.sigmoid(mlp_model(bx)).cpu().numpy()
                val_preds_list.extend(val_probs)

        epoch_val_auc = roc_auc_score(y_va, np.array(val_preds_list))
        if epoch_val_auc > best_val_auc:
            best_val_auc = epoch_val_auc
            best_weights = copy.deepcopy(mlp_model.state_dict())

    mlp_model.load_state_dict(best_weights)
    mlp_model.eval()
    with torch.no_grad():
        val_probs = (
            torch.sigmoid(
                mlp_model(torch.tensor(X_va_mlp, dtype=torch.float32).to(device))
            )
            .cpu()
            .numpy()
        )
        test_probs = (
            torch.sigmoid(
                mlp_model(torch.tensor(X_te_mlp, dtype=torch.float32).to(device))
            )
            .cpu()
            .numpy()
        )

    oof_mlp[val_mask] = val_probs
    test_preds_mlp += test_probs / n_folds

    fold_auc_lgb = roc_auc_score(y_va, oof_lgb[val_mask])
    fold_auc_cat = roc_auc_score(y_va, oof_cat[val_mask])
    fold_auc_lr = roc_auc_score(y_va, oof_lr[val_mask])
    fold_auc_mlp = roc_auc_score(y_va, oof_mlp[val_mask])
    print(
        f"Fold {fold+1}/{n_folds} - LGB: {fold_auc_lgb:.4f} | Cat: {fold_auc_cat:.4f} | LR: {fold_auc_lr:.4f} | ResMLP: {fold_auc_mlp:.4f}"
    )

# ---------------------------------------------------------------------------
# 7. Validation Evaluation and Ensemble Weight Optimization
# ---------------------------------------------------------------------------
full_auc_lgb = roc_auc_score(y_train, oof_lgb)
full_auc_cat = roc_auc_score(y_train, oof_cat)
full_auc_lr = roc_auc_score(y_train, oof_lr)
full_auc_mlp = roc_auc_score(y_train, oof_mlp)
print(f"Overall OOF - LightGBM: {full_auc_lgb:.4f}")
print(f"Overall OOF - CatBoost: {full_auc_cat:.4f}")
print(f"Overall OOF - LogisticRegression: {full_auc_lr:.4f}")
print(f"Overall OOF - ResPizzaMLP: {full_auc_mlp:.4f}")


def ensemble_loss(weights):
    w1, w2, w3, w4 = weights
    pred = w1 * oof_lgb + w2 * oof_cat + w3 * oof_lr + w4 * oof_mlp
    return -roc_auc_score(y_train, pred)


opt_res = minimize(
    ensemble_loss,
    [0.25, 0.25, 0.25, 0.25],
    bounds=[(0, 1), (0, 1), (0, 1), (0, 1)],
    constraints={"type": "eq", "fun": lambda w: sum(w) - 1.0},
    method="SLSQP",
)
w_opt = opt_res.x / np.sum(opt_res.x)
print(
    f"Optimal Weights -> LGB: {w_opt[0]:.3f}, Cat: {w_opt[1]:.3f}, LR: {w_opt[2]:.3f}, ResMLP: {w_opt[3]:.3f}"
)

oof_final = (
    w_opt[0] * oof_lgb
    + w_opt[1] * oof_cat
    + w_opt[2] * oof_lr
    + w_opt[3] * oof_mlp
)
final_score = roc_auc_score(y_train, oof_final)

# ---------------------------------------------------------------------------
# 8. Test Prediction & Submission File Generation
# ---------------------------------------------------------------------------
test_final = (
    w_opt[0] * test_preds_lgb
    + w_opt[1] * test_preds_cat
    + w_opt[2] * test_preds_lr
    + w_opt[3] * test_preds_mlp
)

sub_df = pd.DataFrame(
    {"id": np.arange(len(test_final), dtype=int), target_col: test_final}
)
sub_path = "./submission/submission.csv"
sub_df.to_csv(sub_path, index=False)

print(f"Submission saved to {sub_path} with {len(sub_df)} predictions.")
print(f"Final Validation Score: {final_score}")
