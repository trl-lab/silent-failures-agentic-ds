import os
import re
import warnings
import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from sklearn.metrics import accuracy_score, log_loss
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
import scipy.sparse as sp
from scipy.optimize import minimize
import lightgbm as lgb
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, get_cosine_schedule_with_warmup
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Ensure necessary directories exist
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# Set random seeds for strict reproducibility
np.random.seed(42)

# ==========================================
# 1. LOAD DATA & TARGET ENCODING
# ==========================================
print("Loading datasets...")
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")
print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

target_col = "classification"
if target_col not in train_df.columns:
    raise ValueError(f"Target column '{target_col}' not found in training data.")

label_encoder = LabelEncoder()
y_train_raw = train_df[target_col].astype(str)
y_train_encoded = label_encoder.fit_transform(y_train_raw)
num_classes = len(label_encoder.classes_)
joblib.dump(label_encoder, "./working/target_label_encoder.joblib")
print(f"Target classes ({num_classes}): {list(label_encoder.classes_)}")

# Stratified K-Fold setup
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_indices = np.zeros(len(train_df), dtype=int)
for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, y_train_encoded)):
    fold_indices[val_idx] = fold
train_df["fold"] = fold_indices

# ==========================================
# 2. DOMAIN FEATURE ENGINEERING
# ==========================================
CLASS1_KEYWORDS = [
    "listeria", "monocytogenes", "salmonella", "e. coli", "escherichia", "botulinum",
    "botulism", "shiga", "pathogen", "death", "fatal", "life threatening", "hospitalization",
    "severe illness", "anaphylaxis", "peanut", "tree nut", "milk", "egg", "wheat", "soy",
    "sesame", "fish", "crustacean", "shellfish", "undeclared allergen", "undeclared peanut",
    "undeclared milk", "undeclared egg", "undeclared tree nut", "undeclared soy", "undeclared wheat",
    "lead", "cyanide", "toxin", "toxic", "poison", "choking hazard", "infant formula"
]

CLASS2_KEYWORDS = [
    "norovirus", "staphylococcus", "bacillus", "mold", "yeast", "coliform", "spoilage",
    "foreign material", "foreign object", "plastic", "rubber", "glass", "metal", "wood",
    "filth", "insect", "rodent", "temperature abuse", "swollen", "leaking", "off odor",
    "fermentation", "rancid", "unapproved color", "fd&c", "yellow 5", "sulfite", "unapproved additive",
    "potency", "superpotent", "subpotent", "particulate", "adverse reaction", "temporary"
]

CLASS3_KEYWORDS = [
    "labeling", "misbranded", "mislabeling", "net quantity", "net weight", "missing english",
    "font size", "incorrect label", "omission of ingredient", "packaging defect", "aesthetic",
    "label peeling", "typographical error", "date code", "batch number", "standard of identity",
    "cgmp deviation", "minor violation", "non health hazard", "technical violation"
]

FOOD_SAFETY_KEYWORDS = list(dict.fromkeys(CLASS1_KEYWORDS + CLASS2_KEYWORDS + CLASS3_KEYWORDS))


def extract_keyword_features(series):
    features = {}
    clean_series = series.fillna("").astype(str).str.lower()

    # Base individual keywords
    for kw in FOOD_SAFETY_KEYWORDS:
        col_name = f"kw_{re.sub(r'[^a-z0-9]', '_', kw)}"
        features[col_name] = clean_series.str.contains(
            r"\b" + re.escape(kw) + r"\b", regex=True
        ).astype(np.int8)

    # Domain Hazard Severity Aggregate Indicators
    c1_pattern = "|".join([r"\b" + re.escape(k) + r"\b" for k in CLASS1_KEYWORDS])
    c2_pattern = "|".join([r"\b" + re.escape(k) + r"\b" for k in CLASS2_KEYWORDS])
    c3_pattern = "|".join([r"\b" + re.escape(k) + r"\b" for k in CLASS3_KEYWORDS])

    c1_counts = clean_series.apply(lambda x: len(re.findall(c1_pattern, x)) if x else 0)
    c2_counts = clean_series.apply(lambda x: len(re.findall(c2_pattern, x)) if x else 0)
    c3_counts = clean_series.apply(lambda x: len(re.findall(c3_pattern, x)) if x else 0)

    features["hazard_c1_count"] = c1_counts.astype(np.float32)
    features["hazard_c2_count"] = c2_counts.astype(np.float32)
    features["hazard_c3_count"] = c3_counts.astype(np.float32)
    features["hazard_total_count"] = (c1_counts + c2_counts + c3_counts).astype(np.float32)
    features["hazard_ratio_c1"] = (c1_counts / (c1_counts + c2_counts + c3_counts + 1e-5)).astype(np.float32)
    features["hazard_ratio_c2"] = (c2_counts / (c1_counts + c2_counts + c3_counts + 1e-5)).astype(np.float32)
    features["hazard_ratio_c3"] = (c3_counts / (c1_counts + c2_counts + c3_counts + 1e-5)).astype(np.float32)
    features["has_c1_hazard"] = (c1_counts > 0).astype(np.int8)
    features["has_c2_hazard"] = (c2_counts > 0).astype(np.int8)
    features["has_c3_hazard"] = (c3_counts > 0).astype(np.int8)

    return pd.DataFrame(features, index=series.index)


feature_cols = [c for c in train_df.columns if c not in [target_col, "fold", "id"]]

date_cols = []
text_cols = []
cat_cols = []
num_cols = []

for c in feature_cols:
    if "date" in c.lower() or "time" in c.lower():
        date_cols.append(c)
    elif train_df[c].dtype == "object" or train_df[c].dtype.name == "category":
        sample_lens = train_df[c].dropna().astype(str).str.len()
        avg_len = sample_lens.mean() if len(sample_lens) > 0 else 0
        if avg_len > 35 or c.lower() in [
            "reason_for_recall",
            "product_description",
            "code_info",
            "distribution_pattern",
        ]:
            text_cols.append(c)
        else:
            cat_cols.append(c)
    elif np.issubdtype(train_df[c].dtype, np.number):
        num_cols.append(c)

print(
    f"Feature Types - Text: {len(text_cols)}, Categorical: {len(cat_cols)}, Date: {len(date_cols)}, Numeric: {len(num_cols)}"
)


def process_date_features(df, cols):
    res = pd.DataFrame(index=df.index)
    parsed_dates = {}
    for col in cols:
        dt = pd.to_datetime(df[col], errors="coerce", format="mixed")
        parsed_dates[col] = dt
        res[f"{col}_year"] = dt.dt.year.fillna(-1).astype(np.int16)
        res[f"{col}_month"] = dt.dt.month.fillna(-1).astype(np.int8)
        res[f"{col}_day"] = dt.dt.day.fillna(-1).astype(np.int8)
        res[f"{col}_dayofweek"] = dt.dt.dayofweek.fillna(-1).astype(np.int8)
        res[f"{col}_quarter"] = dt.dt.quarter.fillna(-1).astype(np.int8)
        res[f"{col}_dayofyear"] = dt.dt.dayofyear.fillna(-1).astype(np.int16)
        res[f"{col}_is_null"] = dt.isnull().astype(np.int8)

    date_keys = list(parsed_dates.keys())
    for i in range(len(date_keys)):
        for j in range(i + 1, len(date_keys)):
            c1, c2 = date_keys[i], date_keys[j]
            delta_days = (parsed_dates[c1] - parsed_dates[c2]).dt.days
            res[f"delta_days_{c1}_vs_{c2}"] = delta_days.fillna(-999).astype(np.float32)
    return res


def process_text_meta_features(df, cols):
    res = pd.DataFrame(index=df.index)
    for col in cols:
        s = df[col].fillna("").astype(str)
        res[f"{col}_char_len"] = s.str.len().astype(np.float32)
        res[f"{col}_word_count"] = s.str.split().str.len().astype(np.float32)
        res[f"{col}_uppercase_count"] = s.apply(
            lambda x: sum(1 for ch in x if ch.isupper())
        ).astype(np.float32)
        res[f"{col}_digit_count"] = s.apply(
            lambda x: sum(1 for ch in x if ch.isdigit())
        ).astype(np.float32)
        res[f"{col}_has_exclamation"] = s.str.contains("!").astype(np.int8)
        res[f"{col}_has_question"] = s.str.contains(r"\?").astype(np.int8)
        res[f"{col}_is_empty"] = (s.str.strip() == "").astype(np.int8)
    return res


# Compute Date & Text Meta Features
date_feat_train = process_date_features(train_df, date_cols)
date_feat_test = process_date_features(test_df, date_cols)

text_meta_train = process_text_meta_features(train_df, text_cols)
text_meta_test = process_text_meta_features(test_df, text_cols)

all_train_text = (
    train_df[text_cols].fillna("").astype(str).agg(" ".join, axis=1)
    if text_cols
    else pd.Series("", index=train_df.index)
)
all_test_text = (
    test_df[text_cols].fillna("").astype(str).agg(" ".join, axis=1)
    if text_cols
    else pd.Series("", index=test_df.index)
)

kw_feat_train = extract_keyword_features(all_train_text)
kw_feat_test = extract_keyword_features(all_test_text)

# TF-IDF + TruncatedSVD for Text Columns with Expanded Word & Char N-Grams
svd_train_dfs = []
svd_test_dfs = []

for col in text_cols:
    train_text_col = train_df[col].fillna("").astype(str)
    test_text_col = test_df[col].fillna("").astype(str)

    # Word TF-IDF
    tfidf_word = TfidfVectorizer(
        max_features=35000, ngram_range=(1, 2), min_df=2, sublinear_tf=True
    )
    tfidf_word_train = tfidf_word.fit_transform(train_text_col)
    tfidf_word_test = tfidf_word.transform(test_text_col)

    # Char TF-IDF to capture typos, chemical names, codes
    tfidf_char = TfidfVectorizer(
        max_features=20000, ngram_range=(3, 5), analyzer="char", min_df=3, sublinear_tf=True
    )
    tfidf_char_train = tfidf_char.fit_transform(train_text_col)
    tfidf_char_test = tfidf_char.transform(test_text_col)

    tfidf_train = sp.hstack([tfidf_word_train, tfidf_char_train], format="csr")
    tfidf_test = sp.hstack([tfidf_word_test, tfidf_char_test], format="csr")

    n_comp = 80 if col == "reason_for_recall" else 35
    n_components = min(n_comp, max(5, tfidf_train.shape[1] - 1)) if tfidf_train.shape[1] > 5 else tfidf_train.shape[1]

    if n_components > 0:
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        svd_train_mat = svd.fit_transform(tfidf_train)
        svd_test_mat = svd.transform(tfidf_test)

        svd_cols = [f"{col}_svd_{i}" for i in range(n_components)]
        svd_train_dfs.append(
            pd.DataFrame(svd_train_mat, columns=svd_cols, index=train_df.index)
        )
        svd_test_dfs.append(
            pd.DataFrame(svd_test_mat, columns=svd_cols, index=test_df.index)
        )

if len(text_cols) > 1:
    tfidf_comb_word = TfidfVectorizer(
        max_features=40000, ngram_range=(1, 2), min_df=2, sublinear_tf=True
    )
    tfidf_comb_char = TfidfVectorizer(
        max_features=25000, ngram_range=(3, 5), analyzer="char", min_df=3, sublinear_tf=True
    )

    comb_w_train = tfidf_comb_word.fit_transform(all_train_text)
    comb_w_test = tfidf_comb_word.transform(all_test_text)
    comb_c_train = tfidf_comb_char.fit_transform(all_train_text)
    comb_c_test = tfidf_comb_char.transform(all_test_text)

    tfidf_comb_train = sp.hstack([comb_w_train, comb_c_train], format="csr")
    tfidf_comb_test = sp.hstack([comb_w_test, comb_c_test], format="csr")

    svd_comb = TruncatedSVD(n_components=80, random_state=42)
    svd_comb_train_mat = svd_comb.fit_transform(tfidf_comb_train)
    svd_comb_test_mat = svd_comb.transform(tfidf_comb_test)

    svd_comb_cols = [f"all_text_svd_{i}" for i in range(80)]
    svd_train_dfs.append(
        pd.DataFrame(svd_comb_train_mat, columns=svd_comb_cols, index=train_df.index)
    )
    svd_test_dfs.append(
        pd.DataFrame(svd_comb_test_mat, columns=svd_comb_cols, index=test_df.index)
    )

# Pretrained Transformer Semantic Embeddings (all-MiniLM-L6-v2)
def extract_transformer_embeddings(train_series, test_series, model_name="sentence-transformers/all-MiniLM-L6-v2", batch_size=128):
    print(f"Extracting transformer embeddings using {model_name}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(device)
        model.eval()

        def encode_series(series):
            texts = series.fillna("").astype(str).tolist()
            embs = []
            with torch.no_grad():
                for i in range(0, len(texts), batch_size):
                    batch = [t if t.strip() else "none" for t in texts[i : i + batch_size]]
                    encoded = tokenizer(
                        batch,
                        padding=True,
                        truncation=True,
                        max_length=128,
                        return_tensors="pt",
                    ).to(device)
                    out = model(**encoded)
                    mask = encoded["attention_mask"].unsqueeze(-1)
                    emb = (out.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
                    emb = torch.nn.functional.normalize(emb, p=2, dim=1)
                    embs.append(emb.cpu().numpy().astype(np.float32))
            return np.vstack(embs)

        train_emb = encode_series(train_series)
        test_emb = encode_series(test_series)
        return train_emb, test_emb
    except Exception as e:
        print(f"Transformer embedding extraction fallback: {e}")
        return np.zeros((len(train_series), 384), dtype=np.float32), np.zeros((len(test_series), 384), dtype=np.float32)

transformer_train_dfs = []
transformer_test_dfs = []

if "reason_for_recall" in train_df.columns:
    r_tr_emb, r_te_emb = extract_transformer_embeddings(train_df["reason_for_recall"], test_df["reason_for_recall"])
    r_cols = [f"reason_emb_{i}" for i in range(r_tr_emb.shape[1])]
    transformer_train_dfs.append(pd.DataFrame(r_tr_emb, columns=r_cols, index=train_df.index))
    transformer_test_dfs.append(pd.DataFrame(r_te_emb, columns=r_cols, index=test_df.index))

if len(text_cols) > 0:
    all_tr_emb, all_te_emb = extract_transformer_embeddings(all_train_text, all_test_text)
    svd_transformer = TruncatedSVD(n_components=64, random_state=42)
    all_tr_emb_svd = svd_transformer.fit_transform(all_tr_emb)
    all_te_emb_svd = svd_transformer.transform(all_te_emb)
    all_emb_cols = [f"all_text_emb_svd_{i}" for i in range(64)]
    transformer_train_dfs.append(pd.DataFrame(all_tr_emb_svd, columns=all_emb_cols, index=train_df.index))
    transformer_test_dfs.append(pd.DataFrame(all_te_emb_svd, columns=all_emb_cols, index=test_df.index))

# Categorical Frequency, Label Encodings, and Out-of-Fold Smoothed Target Encoding
cat_feat_train = pd.DataFrame(index=train_df.index)
cat_feat_test = pd.DataFrame(index=test_df.index)
smoothing_factor = 10.0

for col in cat_cols:
    train_cat = train_df[col].fillna("__MISSING__").astype(str)
    test_cat = test_df[col].fillna("__MISSING__").astype(str)

    # Frequency Encoding
    freq_map = train_cat.value_counts(normalize=True).to_dict()
    cat_feat_train[f"{col}_freq"] = train_cat.map(freq_map).fillna(0).astype(np.float32)
    cat_feat_test[f"{col}_freq"] = test_cat.map(freq_map).fillna(0).astype(np.float32)

    # Label Encoding
    le = LabelEncoder()
    le.fit(list(train_cat.unique()) + ["__UNKNOWN__"])
    test_cat_mapped = test_cat.apply(lambda x: x if x in le.classes_ else "__UNKNOWN__")
    cat_feat_train[f"{col}_encoded"] = le.transform(train_cat).astype(np.int32)
    cat_feat_test[f"{col}_encoded"] = le.transform(test_cat_mapped).astype(np.int32)

    # Multi-class Out-of-Fold Target Encoding
    for c_idx in range(num_classes):
        te_train_col = np.zeros(len(train_df), dtype=np.float32)
        global_prior = (y_train_encoded == c_idx).mean()

        for fold in range(skf.n_splits):
            tr_indices = np.where(train_df["fold"].values != fold)[0]
            va_indices = np.where(train_df["fold"].values == fold)[0]

            cats_tr = train_cat.iloc[tr_indices]
            y_tr_c = (y_train_encoded[tr_indices] == c_idx).astype(float)

            stats = pd.DataFrame({"cat": cats_tr, "target": y_tr_c}).groupby("cat")["target"].agg(["count", "mean"])
            smoothed_map = (stats["count"] * stats["mean"] + smoothing_factor * global_prior) / (stats["count"] + smoothing_factor)

            te_train_col[va_indices] = train_cat.iloc[va_indices].map(smoothed_map).fillna(global_prior).values

        cat_feat_train[f"{col}_te_class_{c_idx}"] = te_train_col

        # Test set target encoding from full training data
        full_stats = pd.DataFrame({"cat": train_cat, "target": (y_train_encoded == c_idx).astype(float)}).groupby("cat")["target"].agg(["count", "mean"])
        full_smoothed_map = (full_stats["count"] * full_stats["mean"] + smoothing_factor * global_prior) / (full_stats["count"] + smoothing_factor)
        cat_feat_test[f"{col}_te_class_{c_idx}"] = test_cat.map(full_smoothed_map).fillna(global_prior).astype(np.float32)

# Numeric Features Processing
num_feat_train = pd.DataFrame(index=train_df.index)
num_feat_test = pd.DataFrame(index=test_df.index)

for col in num_cols:
    median_val = train_df[col].median()
    num_feat_train[f"{col}_imputed"] = (
        train_df[col].fillna(median_val).astype(np.float32)
    )
    num_feat_test[f"{col}_imputed"] = test_df[col].fillna(median_val).astype(np.float32)
    num_feat_train[f"{col}_is_null"] = train_df[col].isnull().astype(np.int8)
    num_feat_test[f"{col}_is_null"] = test_df[col].isnull().astype(np.int8)

# Combine All Features
train_parts = [
    date_feat_train,
    text_meta_train,
    kw_feat_train,
    cat_feat_train,
    num_feat_train,
] + svd_train_dfs + transformer_train_dfs
test_parts = [
    date_feat_test,
    text_meta_test,
    kw_feat_test,
    cat_feat_test,
    num_feat_test,
] + svd_test_dfs + transformer_test_dfs

X_full_df = pd.concat([df for df in train_parts if not df.empty], axis=1)
X_test_df = pd.concat([df for df in test_parts if not df.empty], axis=1)

X_full_df = X_full_df.loc[:, ~X_full_df.columns.duplicated()].copy()
X_test_df = X_test_df.loc[:, ~X_test_df.columns.duplicated()].copy()

missing_in_test = [c for c in X_full_df.columns if c not in X_test_df.columns]
for c in missing_in_test:
    X_test_df[c] = 0.0
X_test_df = X_test_df[X_full_df.columns].copy()

feature_names = list(X_full_df.columns)
input_dim = len(feature_names)
n_samples = len(X_full_df)
n_test = len(X_test_df)
folds = train_df["fold"].values
y_full = y_train_encoded

X_full = np.nan_to_num(
    X_full_df[feature_names].values.astype(np.float32),
    nan=0.0,
    posinf=0.0,
    neginf=0.0,
)
X_test = np.nan_to_num(
    X_test_df[feature_names].values.astype(np.float32),
    nan=0.0,
    posinf=0.0,
    neginf=0.0,
)

print(f"Preprocessed Feature Matrices -> Train: {X_full.shape}, Test: {X_test.shape}")


# ==========================================
# 3. MODEL ARCHITECTURES
# ==========================================
def get_lgbm_model(num_classes=num_classes, random_state=42):
    params = {
        "objective": "multiclass",
        "num_class": num_classes,
        "metric": "multi_logloss",
        "boosting_type": "gbdt",
        "n_estimators": 1500,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": 6,
        "min_child_samples": 20,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": random_state,
        "n_jobs": -1,
        "verbose": -1,
    }
    model = lgb.LGBMClassifier(**params)
    return model, params


def get_xgboost_model(num_classes=num_classes, random_state=42):
    params = {
        "objective": "multi:softprob",
        "num_class": num_classes,
        "eval_metric": "mlogloss",
        "n_estimators": 1200,
        "learning_rate": 0.03,
        "max_depth": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": random_state,
        "n_jobs": -1,
        "tree_method": "hist",
    }
    model = XGBClassifier(**params)
    return model, params


class TabularResidualBlock(nn.Module):

    def __init__(self, hidden_dim, dropout=0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x):
        return x + self.block(x)


class TabularResNet(nn.Module):

    def __init__(
        self,
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dim=256,
        num_blocks=2,
        dropout=0.25,
    ):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.blocks = nn.ModuleList(
            [TabularResidualBlock(hidden_dim, dropout) for _ in range(num_blocks)]
        )
        self.head = nn.Sequential(
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        h = self.input_layer(x)
        for block in self.blocks:
            h = block(h)
        return self.head(h)


def get_catboost_model(num_classes=num_classes, random_state=42):
    params = {
        "loss_function": "MultiClass",
        "iterations": 1200,
        "learning_rate": 0.04,
        "depth": 5,
        "l2_leaf_reg": 3.0,
        "random_seed": random_state,
        "verbose": False,
        "thread_count": -1,
    }
    model = CatBoostClassifier(**params)
    return model, params


# ==========================================
# 4. TRAINING & EVALUATION PIPELINE
# ==========================================
oof_lgb = np.zeros((n_samples, num_classes), dtype=np.float32)
oof_xgb = np.zeros((n_samples, num_classes), dtype=np.float32)
oof_cat = np.zeros((n_samples, num_classes), dtype=np.float32)
oof_nn = np.zeros((n_samples, num_classes), dtype=np.float32)

test_preds_lgb = np.zeros((n_test, num_classes), dtype=np.float32)
test_preds_xgb = np.zeros((n_test, num_classes), dtype=np.float32)
test_preds_cat = np.zeros((n_test, num_classes), dtype=np.float32)
test_preds_nn = np.zeros((n_test, num_classes), dtype=np.float32)

n_splits = int(np.max(folds) + 1)
print(f"Beginning {n_splits}-Fold Cross-Validation...")

for fold in range(n_splits):
    train_idx = np.where(folds != fold)[0]
    val_idx = np.where(folds == fold)[0]

    X_train_f, y_train_f = X_full[train_idx], y_full[train_idx]
    X_val_f, y_val_f = X_full[val_idx], y_full[val_idx]

    # --- A. LightGBM ---
    lgb_model, _ = get_lgbm_model(num_classes=num_classes, random_state=42 + fold)
    lgb_model.fit(
        X_train_f,
        y_train_f,
        eval_set=[(X_val_f, y_val_f)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )
    val_prob_lgb = lgb_model.predict_proba(X_val_f)
    oof_lgb[val_idx] = val_prob_lgb
    test_preds_lgb += lgb_model.predict_proba(X_test) / n_splits

    # --- B. XGBoost ---
    xgb_model, xgb_params = get_xgboost_model(
        num_classes=num_classes, random_state=42 + fold
    )
    xgb_params["early_stopping_rounds"] = 50
    xgb_model = XGBClassifier(**xgb_params)
    xgb_model.fit(X_train_f, y_train_f, eval_set=[(X_val_f, y_val_f)], verbose=False)
    val_prob_xgb = xgb_model.predict_proba(X_val_f)
    oof_xgb[val_idx] = val_prob_xgb
    test_preds_xgb += xgb_model.predict_proba(X_test) / n_splits

    # --- C. CatBoost ---
    cat_model, _ = get_catboost_model(
        num_classes=num_classes, random_state=42 + fold
    )
    cat_model.fit(
        X_train_f,
        y_train_f,
        eval_set=(X_val_f, y_val_f),
        early_stopping_rounds=50,
        verbose=False,
    )
    val_prob_cat = cat_model.predict_proba(X_val_f)
    oof_cat[val_idx] = val_prob_cat
    test_preds_cat += cat_model.predict_proba(X_test) / n_splits

    # --- D. Tabular ResNet (PyTorch) ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_f)
    X_val_scaled = scaler.transform(X_val_f)
    X_test_scaled = scaler.transform(X_test)

    train_ds = TensorDataset(
        torch.tensor(X_train_scaled, dtype=torch.float32),
        torch.tensor(y_train_f, dtype=torch.long),
    )
    val_ds = TensorDataset(
        torch.tensor(X_val_scaled, dtype=torch.float32),
        torch.tensor(y_val_f, dtype=torch.long),
    )

    train_loader = DataLoader(
        train_ds, batch_size=128, shuffle=True, drop_last=False, num_workers=0
    )
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0)

    nn_model = TabularResNet(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dim=256,
        num_blocks=2,
        dropout=0.25,
    ).to(device)

    epochs = 25
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = AdamW(nn_model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(
        optimizer, T_max=epochs * len(train_loader), eta_min=1e-6
    )

    best_val_loss = float("inf")
    best_weights_path = f"./working/nn_best_fold_{fold}.pt"

    for epoch in range(epochs):
        nn_model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = nn_model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(nn_model.parameters(), max_norm=2.0)
            optimizer.step()
            scheduler.step()

        nn_model.eval()
        val_loss = 0.0
        total_val = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                logits = nn_model(batch_x)
                loss = nn.CrossEntropyLoss()(logits, batch_y)
                val_loss += loss.item() * len(batch_y)
                total_val += len(batch_y)

        val_loss /= total_val
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(nn_model.state_dict(), best_weights_path)

    nn_model.load_state_dict(torch.load(best_weights_path))
    nn_model.eval()
    with torch.no_grad():
        val_logits = nn_model(
            torch.tensor(X_val_scaled, dtype=torch.float32).to(device)
        )
        val_prob_nn = torch.softmax(val_logits, dim=1).cpu().numpy()
        oof_nn[val_idx] = val_prob_nn

        test_logits = nn_model(
            torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
        )
        test_prob_nn = torch.softmax(test_logits, dim=1).cpu().numpy()
        test_preds_nn += test_prob_nn / n_splits

    fold_acc_lgb = accuracy_score(y_val_f, np.argmax(val_prob_lgb, axis=1))
    fold_acc_xgb = accuracy_score(y_val_f, np.argmax(val_prob_xgb, axis=1))
    fold_acc_nn = accuracy_score(y_val_f, np.argmax(val_prob_nn, axis=1))
    print(
        f"Fold {fold} -> LGB: {fold_acc_lgb:.4f} | XGB: {fold_acc_xgb:.4f} | NN: {fold_acc_nn:.4f}"
    )

# Out-of-Fold Evaluation
acc_lgb = accuracy_score(y_full, np.argmax(oof_lgb, axis=1))
acc_xgb = accuracy_score(y_full, np.argmax(oof_xgb, axis=1))
acc_nn = accuracy_score(y_full, np.argmax(oof_nn, axis=1))
print(
    f"Overall OOF Accuracies -> LightGBM: {acc_lgb:.4f} | XGBoost: {acc_xgb:.4f} | TabularResNet: {acc_nn:.4f}"
)

# Search optimal ensemble weights
best_ensemble_acc = 0.0
best_weights = (0.5, 0.5, 0.0)

for w_lgb in np.linspace(0.1, 0.8, 15):
    for w_xgb in np.linspace(0.1, 0.8, 15):
        w_nn = 1.0 - (w_lgb + w_xgb)
        if w_nn < 0:
            continue
        blended_oof = w_lgb * oof_lgb + w_xgb * oof_xgb + w_nn * oof_nn
        blend_preds = np.argmax(blended_oof, axis=1)
        score_candidate = accuracy_score(y_full, blend_preds)
        if score_candidate > best_ensemble_acc:
            best_ensemble_acc = score_candidate
            best_weights = (w_lgb, w_xgb, w_nn)

w1, w2, w3 = best_weights
print(
    f"Optimal Weights: LGB={w1:.2f}, XGB={w2:.2f}, NN={w3:.2f} -> Blended OOF Acc: {best_ensemble_acc:.5f}"
)

# Generate and save submission
final_test_probs = w1 * test_preds_lgb + w2 * test_preds_xgb + w3 * test_preds_nn
final_test_classes_encoded = np.argmax(final_test_probs, axis=1)
final_test_classes = label_encoder.inverse_transform(final_test_classes_encoded)

submission_df = pd.DataFrame(
    {"id": np.arange(len(final_test_classes)), "classification": final_test_classes}
)
submission_df.to_csv("./submission/submission.csv", index=False)
print(f"Saved submission to ./submission/submission.csv ({len(submission_df)} rows)")

final_score = float(best_ensemble_acc)
print(f"Final Validation Score: {final_score}")