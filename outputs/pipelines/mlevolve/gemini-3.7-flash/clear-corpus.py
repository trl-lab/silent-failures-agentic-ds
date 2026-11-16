import math
import os
import random
import re
import string
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, TensorDataset
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
    logging as hf_logging,
)

hf_logging.set_verbosity_error()

# =============================================================================
# 1. Reproducibility & Environment Setup
# =============================================================================
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# =============================================================================
# 2. Data Loading & Identification
# =============================================================================
print("Loading datasets from ./input/...")
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Identify target column
target_col = "BT Easiness"
if target_col not in train_df.columns:
    for candidate in ["BT_easiness", "target", "easiness", "ease"]:
        if candidate in train_df.columns:
            target_col = candidate
            break

# Identify text column
text_candidates = [
    "Excerpt",
    "excerpt",
    "text",
    "Text",
    "sentence",
    "content",
    "passage",
]
text_col = None
for candidate in text_candidates:
    if candidate in train_df.columns:
        text_col = candidate
        break

if text_col is None:
    str_cols = [
        c
        for c in train_df.columns
        if train_df[c].dtype == "object" or train_df[c].dtype == "string"
    ]
    text_col = max(str_cols, key=lambda c: train_df[c].astype(str).str.len().mean())

print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")
print(f"Target column: '{target_col}', Text column: '{text_col}'")

train_df[text_col] = train_df[text_col].fillna("").astype(str)
test_df[text_col] = test_df[text_col].fillna("").astype(str)

# =============================================================================
# 3. Linguistic and Readability Feature Extraction
# =============================================================================
COMMON_STOPWORDS = {
    "a",
    "about",
    "above",
    "after",
    "again",
    "against",
    "all",
    "am",
    "an",
    "and",
    "any",
    "are",
    "aren't",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "being",
    "below",
    "between",
    "both",
    "but",
    "by",
    "can't",
    "cannot",
    "could",
    "couldn't",
    "did",
    "didn't",
    "do",
    "does",
    "doesn't",
    "doing",
    "don't",
    "down",
    "during",
    "each",
    "few",
    "for",
    "from",
    "further",
    "had",
    "hadn't",
    "has",
    "hasn't",
    "have",
    "haven't",
    "having",
    "he",
    "he'd",
    "he'll",
    "he's",
    "her",
    "here",
    "here's",
    "hers",
    "herself",
    "him",
    "himself",
    "his",
    "how",
    "how's",
    "i",
    "i'd",
    "i'll",
    "i'm",
    "i've",
    "if",
    "in",
    "into",
    "is",
    "isn't",
    "it",
    "it's",
    "its",
    "itself",
    "let's",
    "me",
    "more",
    "most",
    "mustn't",
    "my",
    "myself",
    "no",
    "nor",
    "not",
    "of",
    "off",
    "on",
    "once",
    "only",
    "or",
    "other",
    "ought",
    "our",
    "ours",
    "ourselves",
    "out",
    "over",
    "own",
    "same",
    "shan't",
    "she",
    "she'd",
    "she'll",
    "she's",
    "should",
    "shouldn't",
    "so",
    "some",
    "such",
    "than",
    "that",
    "that's",
    "the",
    "their",
    "theirs",
    "them",
    "themselves",
    "then",
    "there",
    "there's",
    "these",
    "they",
    "they'd",
    "they'll",
    "they're",
    "they've",
    "this",
    "those",
    "through",
    "to",
    "too",
    "under",
    "until",
    "up",
    "very",
    "was",
    "wasn't",
    "we",
    "we'd",
    "we'll",
    "we're",
    "we've",
    "were",
    "weren't",
    "what",
    "what's",
    "when",
    "when's",
    "where",
    "where's",
    "which",
    "while",
    "who",
    "who's",
    "whom",
    "why",
    "why's",
    "with",
    "won't",
    "would",
    "wouldn't",
    "you",
    "you'd",
    "you'll",
    "you're",
    "you've",
    "your",
    "yours",
    "yourself",
    "yourselves",
}


def count_syllables(word: str) -> int:
    """Estimate syllable count in an English word."""
    word = word.lower()
    if len(word) <= 3:
        return 1
    word = re.sub(r"(?:[^laeiouy]|ed|es|e)$", "", word)
    word = re.sub(r"^y", "", word)
    syllable_matches = re.findall(r"[aeiouy]{1,2}", word)
    return max(1, len(syllable_matches))


def extract_linguistic_features(texts: list) -> pd.DataFrame:
    """Extract extensive readability metrics, surface statistics, and lexical indicators."""
    features_list = []

    for text in texts:
        text_len = len(text)
        sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
        num_sentences = max(1, len(sentences))

        words = re.findall(r"\b[a-zA-Z]+\b", text)
        num_words = max(1, len(words))

        word_lengths = [len(w) for w in words]
        avg_word_length = float(np.mean(word_lengths)) if word_lengths else 0.0
        std_word_length = float(np.std(word_lengths)) if word_lengths else 0.0
        max_word_length = float(max(word_lengths)) if word_lengths else 0.0

        sentence_lengths = [len(re.findall(r"\b[a-zA-Z]+\b", s)) for s in sentences]
        avg_sentence_length = (
            float(np.mean(sentence_lengths)) if sentence_lengths else 0.0
        )
        std_sentence_length = (
            float(np.std(sentence_lengths)) if sentence_lengths else 0.0
        )
        max_sentence_length = float(max(sentence_lengths)) if sentence_lengths else 0.0

        syllables = [count_syllables(w) for w in words]
        total_syllables = sum(syllables) if syllables else 0
        avg_syllables_per_word = total_syllables / num_words if num_words > 0 else 0.0

        poly_syllable_words = sum(1 for s in syllables if s >= 3)
        pct_complex_words = poly_syllable_words / num_words if num_words > 0 else 0.0

        # Readability formula indices
        flesch_reading_ease = (
            206.835 - 1.015 * avg_sentence_length - 84.6 * avg_syllables_per_word
        )
        flesch_kincaid = (
            0.39 * avg_sentence_length + 11.8 * avg_syllables_per_word - 15.59
        )
        gunning_fog = 0.4 * (avg_sentence_length + 100.0 * pct_complex_words)
        letters_per_100_words = (
            (sum(word_lengths) / num_words) * 100.0 if num_words > 0 else 0.0
        )
        sentences_per_100_words = (
            (num_sentences / num_words) * 100.0 if num_words > 0 else 0.0
        )
        coleman_liau = (
            0.0588 * letters_per_100_words - 0.296 * sentences_per_100_words - 15.8
        )
        ari = (
            4.71 * (sum(word_lengths) / num_words) + 0.5 * avg_sentence_length - 21.43
            if num_words > 0
            else 0.0
        )
        smog = (
            1.0430 * math.sqrt(poly_syllable_words * (30.0 / num_sentences)) + 3.1291
            if num_sentences >= 3
            else flesch_kincaid
        )

        easy_words = sum(1 for s in syllables if s <= 2)
        hard_words = sum(1 for s in syllables if s >= 3)
        linsear_write = (
            (easy_words + 3 * hard_words) / num_sentences if num_sentences > 0 else 0.0
        )

        # Lexical Diversity & Vocabulary Richness
        lower_words = [w.lower() for w in words]
        unique_words = set(lower_words)
        num_unique_words = len(unique_words)
        ttr = num_unique_words / num_words if num_words > 0 else 0.0
        root_ttr = num_unique_words / math.sqrt(num_words) if num_words > 0 else 0.0

        word_freqs = {}
        for w in lower_words:
            word_freqs[w] = word_freqs.get(w, 0) + 1
        hapax_count = sum(1 for freq in word_freqs.values() if freq == 1)
        hapax_ratio = hapax_count / num_words if num_words > 0 else 0.0

        short_words_ratio = (
            sum(1 for w in words if len(w) <= 3) / num_words if num_words > 0 else 0.0
        )
        long_words_ratio = (
            sum(1 for w in words if len(w) >= 7) / num_words if num_words > 0 else 0.0
        )
        very_long_words_ratio = (
            sum(1 for w in words if len(w) >= 10) / num_words if num_words > 0 else 0.0
        )

        stopwords_count = sum(1 for w in lower_words if w in COMMON_STOPWORDS)
        stopwords_ratio = stopwords_count / num_words if num_words > 0 else 0.0
        content_words_ratio = 1.0 - stopwords_ratio

        # Punctuation counts and densities
        comma_count = text.count(",")
        semicolon_count = text.count(";")
        colon_count = text.count(":")
        dash_count = text.count("-") + text.count("—")
        quote_count = (
            text.count('"') + text.count("'") + text.count("“") + text.count("”")
        )
        excl_count = text.count("!")
        ques_count = text.count("?")
        punct_count = sum(text.count(p) for p in string.punctuation)
        punct_ratio = punct_count / max(1, text_len)

        # Capitalization & Digit Ratios
        upper_chars = sum(1 for c in text if c.isupper())
        upper_char_ratio = upper_chars / max(1, text_len)
        capitalized_words = sum(1 for w in words if w[0].isupper())
        capitalized_word_ratio = capitalized_words / num_words if num_words > 0 else 0.0
        digits_count = sum(1 for c in text if c.isdigit())
        digit_ratio = digits_count / max(1, text_len)

        features = {
            "char_count": text_len,
            "num_words": num_words,
            "num_sentences": num_sentences,
            "avg_word_length": avg_word_length,
            "std_word_length": std_word_length,
            "max_word_length": max_word_length,
            "avg_sentence_length": avg_sentence_length,
            "std_sentence_length": std_sentence_length,
            "max_sentence_length": max_sentence_length,
            "total_syllables": total_syllables,
            "avg_syllables_per_word": avg_syllables_per_word,
            "poly_syllable_words": poly_syllable_words,
            "pct_complex_words": pct_complex_words,
            "flesch_reading_ease": flesch_reading_ease,
            "flesch_kincaid": flesch_kincaid,
            "gunning_fog": gunning_fog,
            "coleman_liau": coleman_liau,
            "ari": ari,
            "smog": smog,
            "linsear_write": linsear_write,
            "ttr": ttr,
            "root_ttr": root_ttr,
            "hapax_ratio": hapax_ratio,
            "short_words_ratio": short_words_ratio,
            "long_words_ratio": long_words_ratio,
            "very_long_words_ratio": very_long_words_ratio,
            "stopwords_ratio": stopwords_ratio,
            "content_words_ratio": content_words_ratio,
            "comma_count": comma_count,
            "semicolon_count": semicolon_count,
            "colon_count": colon_count,
            "dash_count": dash_count,
            "quote_count": quote_count,
            "excl_count": excl_count,
            "ques_count": ques_count,
            "punct_ratio": punct_ratio,
            "upper_char_ratio": upper_char_ratio,
            "capitalized_word_ratio": capitalized_word_ratio,
            "digit_ratio": digit_ratio,
        }
        features_list.append(features)

    return pd.DataFrame(features_list)


print("Extracting linguistic and readability features...")
train_ling_df = extract_linguistic_features(train_df[text_col].values)
test_ling_df = extract_linguistic_features(test_df[text_col].values)

# =============================================================================
# 4. Text Vectorization & Latent Semantic Analysis (SVD)
# =============================================================================
print("Fitting TF-IDF and SVD representations on training corpus...")
train_texts = train_df[text_col].values
test_texts = test_df[text_col].values

# Word TF-IDF
word_tfidf = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=10000,
    sublinear_tf=True,
    min_df=2,
    strip_accents="unicode",
    analyzer="word",
    token_pattern=r"\w{1,}",
)
# Char TF-IDF
char_tfidf = TfidfVectorizer(
    ngram_range=(3, 5),
    max_features=10000,
    sublinear_tf=True,
    min_df=3,
    strip_accents="unicode",
    analyzer="char",
)

word_tfidf.fit(train_texts)
char_tfidf.fit(train_texts)

train_word_mat = word_tfidf.transform(train_texts)
test_word_mat = word_tfidf.transform(test_texts)

train_char_mat = char_tfidf.transform(train_texts)
test_char_mat = char_tfidf.transform(test_texts)

svd_word = TruncatedSVD(n_components=64, random_state=SEED)
svd_char = TruncatedSVD(n_components=64, random_state=SEED)

train_word_svd = svd_word.fit_transform(train_word_mat)
test_word_svd = svd_word.transform(test_word_mat)

train_char_svd = svd_char.fit_transform(train_char_mat)
test_char_svd = svd_char.transform(test_char_mat)

word_svd_cols = [f"svd_word_{i}" for i in range(64)]
char_svd_cols = [f"svd_char_{i}" for i in range(64)]

train_word_svd_df = pd.DataFrame(train_word_svd, columns=word_svd_cols)
test_word_svd_df = pd.DataFrame(test_word_svd, columns=word_svd_cols)

train_char_svd_df = pd.DataFrame(train_char_svd, columns=char_svd_cols)
test_char_svd_df = pd.DataFrame(test_char_svd, columns=char_svd_cols)

# Combine engineered tabular features
train_features_df = pd.concat(
    [train_ling_df, train_word_svd_df, train_char_svd_df], axis=1
)
test_features_df = pd.concat([test_ling_df, test_word_svd_df, test_char_svd_df], axis=1)

feature_cols = list(train_features_df.columns)
X_train_raw = train_features_df[feature_cols].values
X_test_raw = test_features_df[feature_cols].values

y_train_all = train_df[target_col].values.astype(np.float64)
n_samples = len(train_df)
n_test = len(test_df)

# =============================================================================
# 5. Stratified Cross-Validation Splitting
# =============================================================================
n_splits = 5
num_bins = int(np.floor(1 + np.log2(len(y_train_all))))
bins = pd.qcut(y_train_all, q=num_bins, labels=False, duplicates="drop")
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
folds = np.zeros(n_samples, dtype=int)
for fold, (train_idx, val_idx) in enumerate(skf.split(train_features_df, bins)):
    folds[val_idx] = fold


# =============================================================================
# 6. Model Architectures & Parameter Configurations
# =============================================================================
class RMSELoss(nn.Module):
    """Root Mean Squared Error loss."""

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(self.mse(y_pred.view(-1), y_true.view(-1)) + self.eps)


class AttentionPooling(nn.Module):
    """Attention pooling layer for transformer sequence outputs."""

    def __init__(self, in_features: int, hidden_dim: int = 512):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        weights = self.attn(last_hidden_state)  # (B, L, 1)
        weights = weights.masked_fill(attention_mask.unsqueeze(-1) == 0, -10000.0)
        weights = torch.softmax(weights, dim=1)  # (B, L, 1)
        pooled = torch.sum(last_hidden_state * weights, dim=1)  # (B, D)
        return pooled


class DeBERTaReadabilityModel(nn.Module):
    """Transformer Readability model with Attention Pooling and Multi-Sample Dropout."""

    def __init__(
        self,
        model_name: str = "microsoft/deberta-v3-small",
        dropouts: tuple = (0.1, 0.2, 0.3, 0.4, 0.5),
    ):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.config.update(
            {
                "output_hidden_states": True,
                "hidden_dropout_prob": 0.0,
                "attention_probs_dropout_prob": 0.0,
            }
        )
        self.encoder = AutoModel.from_pretrained(
            model_name, config=self.config
        )
        hidden_size = self.config.hidden_size
        self.pooler = AttentionPooling(hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropouts = nn.ModuleList([nn.Dropout(p) for p in dropouts])
        self.regressor = nn.Linear(hidden_size, 1)
        self._init_weights(self.regressor)
        self._init_weights(self.layer_norm)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(
                mean=0.0, std=self.config.initializer_range
            )
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        outputs = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        )
        last_hidden_state = outputs.last_hidden_state
        pooled = self.pooler(last_hidden_state, attention_mask)
        pooled = self.layer_norm(pooled)
        out = torch.mean(
            torch.stack(
                [self.regressor(drop(pooled)) for drop in self.dropouts], dim=0
            ),
            dim=0,
        )
        return out.squeeze(-1)


class ReadabilityDataset(Dataset):
    """Dataset for tokenized text inputs and target readability scores."""

    def __init__(self, texts, targets=None, tokenizer=None, max_len=256):
        self.texts = list(texts)
        self.targets = (
            np.array(targets, dtype=np.float32) if targets is not None else None
        )
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )
        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }
        if self.targets is not None:
            item["target"] = torch.tensor(self.targets[idx], dtype=torch.float32)
        return item


def get_lightgbm_params() -> dict:
    return {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": -1,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l1": 0.1,
        "lambda_l2": 1.0,
        "min_child_samples": 20,
        "verbosity": -1,
        "n_jobs": -1,
        "random_state": SEED,
    }


# =============================================================================
# 7. Model Training & Cross-Validation
# =============================================================================
# --- Model 1: LightGBM Regressor ---
print("\n--- Training LightGBM Models ---")
lgb_oof = np.zeros(n_samples, dtype=np.float64)
lgb_test_preds = np.zeros(n_test, dtype=np.float64)
lgb_params = get_lightgbm_params()

for fold in range(n_splits):
    trn_idx = folds != fold
    val_idx = folds == fold

    scaler_fold = StandardScaler()
    X_tr = scaler_fold.fit_transform(X_train_raw[trn_idx])
    X_va = scaler_fold.transform(X_train_raw[val_idx])
    X_te = scaler_fold.transform(X_test_raw)
    y_tr = y_train_all[trn_idx]
    y_val_f = y_train_all[val_idx]

    trn_data = lgb.Dataset(X_tr, label=y_tr)
    val_data = lgb.Dataset(X_va, label=y_val_f, reference=trn_data)

    model_lgb = lgb.train(
        lgb_params,
        trn_data,
        num_boost_round=1500,
        valid_sets=[val_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )

    val_pred = model_lgb.predict(X_va, num_iteration=model_lgb.best_iteration)
    lgb_oof[val_idx] = val_pred
    lgb_test_preds += (
        model_lgb.predict(X_te, num_iteration=model_lgb.best_iteration)
        / n_splits
    )

    fold_rmse = root_mean_squared_error(y_val_f, val_pred)
    print(f"LGBM Fold {fold + 1}/{n_splits} RMSE: {fold_rmse:.5f}")

lgb_total_rmse = root_mean_squared_error(y_train_all, lgb_oof)
print(f"LGBM Full OOF RMSE: {lgb_total_rmse:.5f}")

# --- Model 2: Regularized Ridge Regression ---
print("\n--- Training Ridge Models ---")
ridge_oof = np.zeros(n_samples, dtype=np.float64)
ridge_test_preds = np.zeros(n_test, dtype=np.float64)

for fold in range(n_splits):
    trn_idx = folds != fold
    val_idx = folds == fold

    scaler_fold = StandardScaler()
    X_tr = scaler_fold.fit_transform(X_train_raw[trn_idx])
    X_va = scaler_fold.transform(X_train_raw[val_idx])
    X_te = scaler_fold.transform(X_test_raw)
    y_tr = y_train_all[trn_idx]
    y_val_f = y_train_all[val_idx]

    ridge = Ridge(alpha=10.0, random_state=SEED)
    ridge.fit(X_tr, y_tr)

    val_pred = ridge.predict(X_va)
    ridge_oof[val_idx] = val_pred
    ridge_test_preds += ridge.predict(X_te) / n_splits

    fold_rmse = root_mean_squared_error(y_val_f, val_pred)
    print(f"Ridge Fold {fold + 1}/{n_splits} RMSE: {fold_rmse:.5f}")

ridge_total_rmse = root_mean_squared_error(y_train_all, ridge_oof)
print(f"Ridge Full OOF RMSE: {ridge_total_rmse:.5f}")

# --- Model 3: DeBERTa Transformer Readability Model ---
print("\n--- Training DeBERTa-v3 Transformer Models ---")
transformer_model_name = "microsoft/deberta-v3-small"
tokenizer = AutoTokenizer.from_pretrained(transformer_model_name)

deberta_oof = np.zeros(n_samples, dtype=np.float64)
deberta_test_preds = np.zeros(n_test, dtype=np.float64)

test_dataset = ReadabilityDataset(
    test_texts, tokenizer=tokenizer, max_len=256
)
test_loader = DataLoader(
    test_dataset, batch_size=32, shuffle=False, pin_memory=torch.cuda.is_available()
)

def get_optimizer_params(model, encoder_lr=2e-5, head_lr=1e-4, weight_decay=0.01):
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.encoder.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "lr": encoder_lr,
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.encoder.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "lr": encoder_lr,
            "weight_decay": 0.0,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not n.startswith("encoder.")
                and not any(nd in n for nd in no_decay)
            ],
            "lr": head_lr,
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not n.startswith("encoder.")
                and any(nd in n for nd in no_decay)
            ],
            "lr": head_lr,
            "weight_decay": 0.0,
        },
    ]
    return optimizer_grouped_parameters

transformer_batch_size = 16
transformer_epochs = 4

for fold in range(n_splits):
    trn_idx = folds != fold
    val_idx = folds == fold

    trn_texts_f = train_texts[trn_idx]
    trn_targets_f = y_train_all[trn_idx]
    val_texts_f = train_texts[val_idx]
    val_targets_f = y_train_all[val_idx]

    train_dataset = ReadabilityDataset(
        trn_texts_f, trn_targets_f, tokenizer=tokenizer, max_len=256
    )
    val_dataset = ReadabilityDataset(
        val_texts_f, val_targets_f, tokenizer=tokenizer, max_len=256
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=transformer_batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
    )

    model_transformer = DeBERTaReadabilityModel(
        model_name=transformer_model_name
    ).to(device)
    criterion = RMSELoss().to(device)

    opt_params = get_optimizer_params(
        model_transformer, encoder_lr=2e-5, head_lr=1e-4, weight_decay=0.01
    )
    optimizer = torch.optim.AdamW(opt_params, eps=1e-6)
    total_steps = len(train_loader) * transformer_epochs
    warmup_steps = int(total_steps * 0.1)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    scaler = GradScaler(enabled=torch.cuda.is_available())

    best_val_rmse = float("inf")
    best_weights_path = f"./working/deberta_fold_{fold}.pt"

    for epoch in range(transformer_epochs):
        model_transformer.train()
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["target"].to(device)

            optimizer.zero_grad()
            with autocast(enabled=torch.cuda.is_available()):
                preds = model_transformer(input_ids, attention_mask)
                loss = criterion(preds, targets)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model_transformer.parameters(), max_norm=1.0
            )
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

        model_transformer.eval()
        val_preds_epoch = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                with autocast(enabled=torch.cuda.is_available()):
                    preds = model_transformer(input_ids, attention_mask)
                val_preds_epoch.extend(preds.cpu().numpy().tolist())

        val_rmse_epoch = root_mean_squared_error(
            val_targets_f, np.array(val_preds_epoch)
        )
        if val_rmse_epoch < best_val_rmse:
            best_val_rmse = val_rmse_epoch
            torch.save(model_transformer.state_dict(), best_weights_path)

    # Load best checkpoint and generate out-of-fold & test predictions
    model_transformer.load_state_dict(
        torch.load(best_weights_path, map_location=device)
    )
    model_transformer.eval()

    fold_val_preds = []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            with autocast(enabled=torch.cuda.is_available()):
                preds = model_transformer(input_ids, attention_mask)
            fold_val_preds.extend(preds.cpu().numpy().tolist())
    deberta_oof[val_idx] = np.array(fold_val_preds)

    fold_test_preds = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            with autocast(enabled=torch.cuda.is_available()):
                preds = model_transformer(input_ids, attention_mask)
            fold_test_preds.extend(preds.cpu().numpy().tolist())
    deberta_test_preds += np.array(fold_test_preds) / n_splits

    fold_rmse = root_mean_squared_error(val_targets_f, deberta_oof[val_idx])
    print(f"DeBERTa Fold {fold + 1}/{n_splits} RMSE: {fold_rmse:.5f}")

    del model_transformer, optimizer, scheduler
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

deberta_total_rmse = root_mean_squared_error(y_train_all, deberta_oof)
print(f"DeBERTa Full OOF RMSE: {deberta_total_rmse:.5f}")

# =============================================================================
# 8. Ensemble Optimization (Constrained SLSQP on OOF)
# =============================================================================
oof_matrix = np.column_stack([lgb_oof, ridge_oof, deberta_oof])
test_matrix = np.column_stack([lgb_test_preds, ridge_test_preds, deberta_test_preds])


def loss_func(weights):
    w = np.array(weights)
    pred = oof_matrix @ w
    return root_mean_squared_error(y_train_all, pred)


initial_weights = [0.1, 0.1, 0.8]
bounds = [(0.0, 1.0) for _ in range(3)]
constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

opt_res = minimize(
    loss_func,
    initial_weights,
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
)
optimal_weights = opt_res.x
print(f"\nOptimal Blending Weights [LGBM, Ridge, DeBERTa]: {optimal_weights}")

final_oof_preds = oof_matrix @ optimal_weights
final_test_preds = test_matrix @ optimal_weights

final_validation_score = root_mean_squared_error(y_train_all, final_oof_preds)

# =============================================================================
# 9. Submission File Generation
# =============================================================================
id_col = None
for candidate in ["id", "Id", "ID", "excerpt_id"]:
    if candidate in test_df.columns:
        id_col = candidate
        break

if id_col is not None:
    submission_df = pd.DataFrame(
        {"id": test_df[id_col].values, "BT Easiness": final_test_preds}
    )
else:
    submission_df = pd.DataFrame(
        {"id": np.arange(len(test_df)), "BT Easiness": final_test_preds}
    )

submission_df.to_csv("./submission/submission.csv", index=False)
print(f"Saved ./submission/submission.csv ({len(submission_df)} rows).")

# =============================================================================
# 10. Final Validation Score Output
# =============================================================================
print(f"Final Validation Score: {final_validation_score}")