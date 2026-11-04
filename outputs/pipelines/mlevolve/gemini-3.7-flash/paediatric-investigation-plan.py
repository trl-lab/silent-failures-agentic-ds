import os
import re
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from scipy.optimize import minimize
from scipy.special import softmax
from scipy.sparse import hstack, csr_matrix
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import log_loss
import lightgbm as lgb
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

warnings.filterwarnings("ignore")

# Ensure required directories exist
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# =========================================================================
# 1. Load Datasets & Column Inspection
# =========================================================================
train_path = "./input/train.parquet"
test_path = "./input/test.parquet"

print(f"Loading data from {train_path} and {test_path}...")
df_train = pd.read_parquet(train_path)
df_test = pd.read_parquet(test_path)

target_col = "Decision type"
if target_col not in df_train.columns:
    candidates = [
        c for c in df_train.columns if "decision" in c.lower() and "type" in c.lower()
    ]
    if candidates:
        target_col = candidates[0]

print(
    f"Train shape: {df_train.shape}, Test shape: {df_test.shape}, Target column: {target_col}"
)

# Encode Target
label_encoder = LabelEncoder()
df_train["target"] = label_encoder.fit_transform(df_train[target_col].astype(str))
num_classes = len(label_encoder.classes_)
print(f"Target classes ({num_classes}): {list(label_encoder.classes_)}")

# Combine datasets for schema inspection and feature structure
df_train["is_train"] = 1
df_test["is_train"] = 0
df_test["target"] = -1

combined = pd.concat([df_train, df_test], axis=0, ignore_index=True)

# Identify Modalities
date_cols = []
text_cols = []
cat_cols = []
num_cols = []

cols_to_ignore = {"target", target_col, "is_train", "id", "ID", "Id", "index"}

for col in combined.columns:
    if col in cols_to_ignore:
        continue

    if pd.api.types.is_datetime64_any_dtype(combined[col]):
        date_cols.append(col)
        continue

    if combined[col].dtype == "object":
        sample = combined[col].dropna().astype(str).head(50)
        if len(sample) > 0:
            is_iso = sample.str.match(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}").mean() > 0.6
            is_eu = sample.str.match(r"^\d{1,2}[-/]\d{1,2}[-/]\d{2,4}").mean() > 0.6
            if is_iso or is_eu:
                date_cols.append(col)
                continue

    if pd.api.types.is_numeric_dtype(combined[col]):
        num_cols.append(col)
    else:
        val_lens = combined[col].dropna().astype(str).str.len()
        avg_len = val_lens.mean() if len(val_lens) > 0 else 0
        nunique = combined[col].nunique()

        if avg_len > 40 or nunique > 500:
            text_cols.append(col)
        else:
            cat_cols.append(col)

print(
    f"Identified modalities - Dates: {len(date_cols)}, Categoricals: {len(cat_cols)}, Texts: {len(text_cols)}, Numerical: {len(num_cols)}"
)

# =========================================================================
# 2. Feature Engineering
# =========================================================================
engineered_features = pd.DataFrame(index=combined.index)

# Date Features
parsed_dates = {}
for col in date_cols:
    dt_series = pd.to_datetime(combined[col], errors="coerce")
    parsed_dates[col] = dt_series
    engineered_features[f"{col}_year"] = dt_series.dt.year.fillna(-1).astype(float)
    engineered_features[f"{col}_month"] = dt_series.dt.month.fillna(-1).astype(float)
    engineered_features[f"{col}_day"] = dt_series.dt.day.fillna(-1).astype(float)
    engineered_features[f"{col}_dayofweek"] = dt_series.dt.dayofweek.fillna(-1).astype(
        float
    )
    engineered_features[f"{col}_quarter"] = dt_series.dt.quarter.fillna(-1).astype(
        float
    )
    engineered_features[f"{col}_is_month_end"] = dt_series.dt.is_month_end.fillna(
        False
    ).astype(float)

    min_date = dt_series.min()
    if pd.notnull(min_date):
        engineered_features[f"{col}_days_from_start"] = (
            dt_series - min_date
        ).dt.days.fillna(-1.0)

# Pairwise Date Differences
date_col_list = list(parsed_dates.keys())
for i in range(len(date_col_list)):
    for j in range(i + 1, len(date_col_list)):
        c1, c2 = date_col_list[i], date_col_list[j]
        delta_col = f"delta_days_{c1}_vs_{c2}"
        engineered_features[delta_col] = (
            (parsed_dates[c1] - parsed_dates[c2]).dt.days.fillna(-9999.0).astype(float)
        )

# Categorical Frequency and Count Encoding
for col in cat_cols:
    combined_col = combined[col].fillna("__MISSING__").astype(str)
    freq_map = combined_col.value_counts(normalize=True).to_dict()
    count_map = combined_col.value_counts().to_dict()
    engineered_features[f"{col}_freq"] = (
        combined_col.map(freq_map).fillna(0.0).astype(float)
    )
    engineered_features[f"{col}_count"] = (
        combined_col.map(count_map).fillna(0.0).astype(float)
    )

    cat_le = LabelEncoder()
    engineered_features[f"{col}_cat_code"] = cat_le.fit_transform(combined_col).astype(
        float
    )

# Categorical Pairwise Interaction Frequencies
if len(cat_cols) >= 2:
    for i in range(min(3, len(cat_cols))):
        for j in range(i + 1, min(4, len(cat_cols))):
            c1, c2 = cat_cols[i], cat_cols[j]
            inter_series = combined[c1].astype(str) + "_" + combined[c2].astype(str)
            inter_freq = inter_series.value_counts(normalize=True).to_dict()
            engineered_features[f"inter_{c1}_{c2}_freq"] = (
                inter_series.map(inter_freq).fillna(0.0).astype(float)
            )

# Domain-Specific Regulatory Regex & Keyword Signals
combined_raw_text = combined[text_cols + cat_cols].fillna("").astype(str).agg(" ".join, axis=1)

regex_patterns = {
    "has_mod_m_code": r"(?:-M|/M|\bM)(\d+)\b",
    "has_pip_code": r"\bPIP(?:-\d+)?\b",
    "has_emea_code": r"\bEMEA-\d+\b",
    "has_modification_word": r"\b(?:modif(?:ication|ied|y|ying)?|amend(?:ment|ed|ing)?|revision|revised)\b",
    "has_waiver_word": r"\b(?:waiver|waivers|waived|full\s+waiver|class\s+waiver|ground\(s\)\s+waiver)\b",
    "has_deferral_word": r"\b(?:deferral|deferrals|deferred|deferring)\b",
    "has_refusal_word": r"\b(?:refus(?:al|ed|ing|e)?|negative\s+opinion|reject(?:ed|ion)?|not\s+agree(?:d)?)\b",
    "has_agree_word": r"\b(?:agree(?:d|ment|ing)?|favourable\s+opinion|positive\s+opinion|investigation\s+plan)\b",
    "has_partial_waiver": r"\bpartial\s+waiver\b",
    "has_all_age_groups": r"\ball\s+(?:age\s+groups|conditions|indications)\b",
    "has_paediatric_subset": r"\bpaediatric\s+subset\b",
    "has_compliance": r"\bcompliance\s+check\b",
}

for feat_name, pat in regex_patterns.items():
    regex_obj = re.compile(pat, flags=re.IGNORECASE)
    engineered_features[f"regex_{feat_name}"] = combined_raw_text.apply(
        lambda t: float(len(regex_obj.findall(t)))
    )

def extract_max_mod_num(text):
    matches = re.findall(r"(?:-M|/M|\bM)(\d{1,3})\b", text, flags=re.IGNORECASE)
    if matches:
        return float(max(int(m) for m in matches))
    return -1.0

engineered_features["mod_max_number"] = combined_raw_text.apply(extract_max_mod_num)

# Text Statistics
for col in text_cols:
    text_s = combined[col].fillna("").astype(str)
    engineered_features[f"{col}_len"] = text_s.str.len().astype(float)
    engineered_features[f"{col}_word_count"] = text_s.apply(
        lambda x: len(x.split())
    ).astype(float)
    engineered_features[f"{col}_cap_count"] = text_s.apply(
        lambda x: sum(1 for c in x if c.isupper())
    ).astype(float)
    engineered_features[f"{col}_digit_count"] = text_s.apply(
        lambda x: sum(1 for c in x if c.isdigit())
    ).astype(float)
    for kw in ["modification", "waiver", "deferral", "refusal", "agree", "plan"]:
        engineered_features[f"{col}_kw_{kw}"] = text_s.str.count(f"(?i){kw}").astype(float)

# Numerical Imputation
for col in num_cols:
    med_val = combined[col].median()
    engineered_features[col] = (
        combined[col].fillna(med_val if pd.notnull(med_val) else 0.0).astype(float)
    )

# Split train and test feature matrices
train_idx = combined["is_train"] == 1
test_idx = combined["is_train"] == 0

X_train_df = engineered_features[train_idx].copy().reset_index(drop=True)
y_train = combined.loc[train_idx, "target"].values.astype(np.int64)
X_test_df = engineered_features[test_idx].copy().reset_index(drop=True)

# NLP Feature Extraction (Dual Representation: Transformer Embeddings + Word/Char TF-IDF SVD & Sparse Matrices)
train_sparse_list = []
test_sparse_list = []

# Extract Transformer Dense Embeddings if available
embedder = None
try:
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    if torch.cuda.is_available():
        embedder = embedder.to("cuda")
except Exception as e:
    embedder = None

def build_prefixed_text(row, cols):
    parts = []
    for c in cols:
        val = str(row[c]).strip()
        if val and val != "nan" and val != "__MISSING__":
            parts.append(f"{c}: {val}")
    return " | ".join(parts)

all_combined_texts = [
    build_prefixed_text(combined.iloc[i], text_cols + cat_cols)
    for i in range(len(combined))
]
train_combined_text = [all_combined_texts[i] for i in range(len(combined)) if train_idx.iloc[i]]
test_combined_text = [all_combined_texts[i] for i in range(len(combined)) if test_idx.iloc[i]]

if embedder is not None:
    try:
        train_emb = embedder.encode(train_combined_text, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
        test_emb = embedder.encode(test_combined_text, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
        for d in range(train_emb.shape[1]):
            X_train_df[f"transformer_emb_{d}"] = train_emb[:, d].astype(float)
            X_test_df[f"transformer_emb_{d}"] = test_emb[:, d].astype(float)
        print(f"Extracted {train_emb.shape[1]} transformer embedding features.")
    except Exception as e:
        print(f"Transformer embedding extraction skipped: {e}")

# High-Fidelity Word (1-3) & Char (3-5) TF-IDF across unified text
unified_tfidf_word = TfidfVectorizer(
    max_features=8000, ngram_range=(1, 3), stop_words="english", min_df=2, sublinear_tf=True
)
train_uni_word = unified_tfidf_word.fit_transform(train_combined_text)
test_uni_word = unified_tfidf_word.transform(test_combined_text)

unified_tfidf_char = TfidfVectorizer(
    max_features=5000, analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True
)
train_uni_char = unified_tfidf_char.fit_transform(train_combined_text)
test_uni_char = unified_tfidf_char.transform(test_combined_text)

train_sparse_list.extend([train_uni_word, train_uni_char])
test_sparse_list.extend([test_uni_word, test_uni_char])

uni_sparse_train = hstack([train_uni_word, train_uni_char])
uni_sparse_test = hstack([test_uni_word, test_uni_char])
n_comp_uni = min(48, uni_sparse_train.shape[1] - 1)
svd_uni = TruncatedSVD(n_components=n_comp_uni, random_state=42)
train_svd_uni = svd_uni.fit_transform(uni_sparse_train)
test_svd_uni = svd_uni.transform(uni_sparse_test)

for k in range(n_comp_uni):
    X_train_df[f"unified_svd_{k}"] = train_svd_uni[:, k].astype(float)
    X_test_df[f"unified_svd_{k}"] = test_svd_uni[:, k].astype(float)

for col in text_cols:
    train_texts = combined.loc[train_idx, col].fillna("").astype(str).tolist()
    test_texts = combined.loc[test_idx, col].fillna("").astype(str).tolist()

    # Word TF-IDF
    tfidf_word = TfidfVectorizer(
        max_features=3000, ngram_range=(1, 3), stop_words="english", min_df=2, sublinear_tf=True
    )
    train_tfidf_word = tfidf_word.fit_transform(train_texts)
    test_tfidf_word = tfidf_word.transform(test_texts)

    # Char n-gram TF-IDF for clinical & regulatory phrases
    tfidf_char = TfidfVectorizer(
        max_features=2000, analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True
    )
    train_tfidf_char = tfidf_char.fit_transform(train_texts)
    test_tfidf_char = tfidf_char.transform(test_texts)

    train_col_sparse = hstack([train_tfidf_word, train_tfidf_char])
    test_col_sparse = hstack([test_tfidf_word, test_tfidf_char])
    train_sparse_list.append(train_col_sparse)
    test_sparse_list.append(test_col_sparse)

    n_features_tfidf = train_col_sparse.shape[1]
    if n_features_tfidf > 2:
        n_comp = min(20, n_features_tfidf - 1)
        svd = TruncatedSVD(n_components=n_comp, random_state=42)
        train_svd = svd.fit_transform(train_col_sparse)
        test_svd = svd.transform(test_col_sparse)

        for k in range(n_comp):
            X_train_df[f"{col}_svd_{k}"] = train_svd[:, k].astype(float)
            X_test_df[f"{col}_svd_{k}"] = test_svd[:, k].astype(float)

# Sparse Text Matrices for Linear Classifiers
X_train_sparse = hstack(train_sparse_list).tocsr()
X_test_sparse = hstack(test_sparse_list).tocsr()
print(f"Constructed sparse text matrix of shape: {X_train_sparse.shape}")

# 5-Fold Stratification & Out-of-Fold Multi-Class Target Encoding
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
X_train_df["fold"] = -1

te_candidates = cat_cols[:4] if len(cat_cols) > 0 else []
for c in te_candidates:
    for cls in range(num_classes):
        X_train_df[f"{c}_te_class_{cls}"] = 0.0
        X_test_df[f"{c}_te_class_{cls}"] = 0.0

raw_train_cat = combined.loc[train_idx, te_candidates].copy().reset_index(drop=True)
raw_test_cat = combined.loc[test_idx, te_candidates].copy().reset_index(drop=True)

for fold, (trn_idx_f, val_idx_f) in enumerate(skf.split(X_train_df, y_train)):
    X_train_df.loc[val_idx_f, "fold"] = fold

    for c in te_candidates:
        train_cats = raw_train_cat.loc[trn_idx_f, c].astype(str)
        val_cats = raw_train_cat.loc[val_idx_f, c].astype(str)
        train_y_f = y_train[trn_idx_f]

        for cls in range(num_classes):
            binary_y = pd.Series((train_y_f == cls).astype(float), index=trn_idx_f)
            global_mean = binary_y.mean()
            stats = binary_y.groupby(train_cats).agg(["mean", "count"])
            smoothing = 10.0
            smooth_enc = (stats["count"] * stats["mean"] + smoothing * global_mean) / (
                stats["count"] + smoothing
            )
            smooth_dict = smooth_enc.to_dict()

            X_train_df.loc[val_idx_f, f"{c}_te_class_{cls}"] = (
                val_cats.map(smooth_dict).fillna(global_mean).values
            )

# Full train target encoding mapped to test
for c in te_candidates:
    full_cats = raw_train_cat[c].astype(str)
    test_cats = raw_test_cat[c].astype(str)
    for cls in range(num_classes):
        binary_y = pd.Series((y_train == cls).astype(float))
        global_mean = binary_y.mean()
        stats = binary_y.groupby(full_cats).agg(["mean", "count"])
        smoothing = 10.0
        smooth_enc = (stats["count"] * stats["mean"] + smoothing * global_mean) / (
            stats["count"] + smoothing
        )
        smooth_dict = smooth_enc.to_dict()
        X_test_df[f"{c}_te_class_{cls}"] = (
            test_cats.map(smooth_dict).fillna(global_mean).values
        )

# Prepare numpy feature matrices
feature_cols = [c for c in X_train_df.columns if c not in ["target", "fold"]]
num_features = len(feature_cols)
num_samples = len(X_train_df)
num_test = len(X_test_df)
folds = X_train_df["fold"].values.astype(int)

X_train_full = np.nan_to_num(
    X_train_df[feature_cols].values.astype(np.float32),
    nan=0.0,
    posinf=1e5,
    neginf=-1e5,
)
X_test_full = np.nan_to_num(
    X_test_df[feature_cols].values.astype(np.float32), nan=0.0, posinf=1e5, neginf=-1e5
)
y_train_full = y_train

print(f"Engineered {num_features} tabular features across {n_splits} stratified folds.")


# =========================================================================
# 3. Model Architecture Definitions
# =========================================================================
def get_lgbm_model(num_classes: int, random_state: int = 42) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="multiclass",
        num_class=num_classes,
        metric="multi_logloss",
        n_estimators=1200,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=6,
        subsample=0.85,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        min_child_samples=20,
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
    )


def get_xgb_model(num_classes: int, random_state: int = 42) -> XGBClassifier:
    return XGBClassifier(
        objective="multi:softprob",
        num_class=num_classes,
        eval_metric="mlogloss",
        n_estimators=1200,
        learning_rate=0.03,
        max_depth=5,
        subsample=0.85,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        min_child_weight=3,
        early_stopping_rounds=50,
        random_state=random_state,
        n_jobs=-1,
        tree_method="hist",
    )


def get_catboost_model(num_classes: int, random_state: int = 42) -> CatBoostClassifier:
    return CatBoostClassifier(
        loss_function="MultiClass",
        eval_metric="MultiClass",
        iterations=1200,
        learning_rate=0.04,
        depth=6,
        l2_leaf_reg=3.0,
        random_seed=random_state,
        verbose=False,
        thread_count=-1,
    )


class TabularMLP(nn.Module):
    def __init__(self, in_features: int, num_classes: int, hidden_dims=[256, 128, 64], dropout=0.2):
        super().__init__()
        layers = []
        prev_dim = in_features
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.SiLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
        self.encoder = nn.Sequential(*layers)
        self.head = nn.Linear(prev_dim, num_classes)

    def forward(self, x):
        return self.head(self.encoder(x))


def train_mlp_fold(X_trn, y_trn, X_val, y_val, X_tst, num_classes, seed=42):
    torch.manual_seed(seed)
    scaler = StandardScaler()
    X_trn_sc = scaler.fit_transform(X_trn)
    X_val_sc = scaler.transform(X_val)
    X_tst_sc = scaler.transform(X_tst)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TabularMLP(in_features=X_trn_sc.shape[1], num_classes=num_classes).to(device)

    trn_ds = TensorDataset(torch.tensor(X_trn_sc, dtype=torch.float32), torch.tensor(y_trn, dtype=torch.long))
    val_ds = TensorDataset(torch.tensor(X_val_sc, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))

    trn_loader = DataLoader(trn_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=35, eta_min=1e-5)

    best_val_loss = float("inf")
    best_val_probs = None
    best_weights = None

    for epoch in range(35):
        model.train()
        for bx, by in trn_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        val_losses = []
        val_probs_list = []
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                logits = model(bx)
                l = criterion(logits, by)
                val_losses.append(l.item())
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                val_probs_list.append(probs)

        mean_val_loss = np.mean(val_losses)
        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            best_val_probs = np.vstack(val_probs_list)
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_weights)
    model.eval()
    with torch.no_grad():
        tst_tensor = torch.tensor(X_tst_sc, dtype=torch.float32).to(device)
        test_probs = torch.softmax(model(tst_tensor), dim=-1).cpu().numpy()

    return best_val_probs, test_probs


def train_transformer_fold(trn_texts, y_trn, val_texts, y_val, tst_texts, num_classes, seed=42):
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_cosine_schedule_with_warmup
        from torch.optim import AdamW
        model_id = "distilbert-base-uncased"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=num_classes).to(device)

        def encode_texts(texts):
            return tokenizer(texts, padding="max_length", truncation=True, max_length=160, return_tensors="pt")

        trn_enc = encode_texts(trn_texts)
        val_enc = encode_texts(val_texts)
        tst_enc = encode_texts(tst_texts)

        trn_ds = TensorDataset(trn_enc["input_ids"], trn_enc["attention_mask"], torch.tensor(y_trn, dtype=torch.long))
        val_ds = TensorDataset(val_enc["input_ids"], val_enc["attention_mask"], torch.tensor(y_val, dtype=torch.long))
        tst_ds = TensorDataset(tst_enc["input_ids"], tst_enc["attention_mask"])

        trn_loader = DataLoader(trn_ds, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
        tst_loader = DataLoader(tst_ds, batch_size=32, shuffle=False)

        epochs = 3
        optimizer = AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)
        total_steps = len(trn_loader) * epochs
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps)
        criterion = nn.CrossEntropyLoss()

        best_val_loss = float("inf")
        best_val_probs = None
        best_state = None

        for epoch in range(epochs):
            model.train()
            for b_ids, b_mask, b_y in trn_loader:
                b_ids, b_mask, b_y = b_ids.to(device), b_mask.to(device), b_y.to(device)
                optimizer.zero_grad()
                out = model(input_ids=b_ids, attention_mask=b_mask)
                loss = criterion(out.logits, b_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

            model.eval()
            val_losses = []
            val_probs_l = []
            with torch.no_grad():
                for b_ids, b_mask, b_y in val_loader:
                    b_ids, b_mask, b_y = b_ids.to(device), b_mask.to(device), b_y.to(device)
                    out = model(input_ids=b_ids, attention_mask=b_mask)
                    loss = criterion(out.logits, b_y)
                    val_losses.append(loss.item())
                    val_probs_l.append(torch.softmax(out.logits, dim=-1).cpu().numpy())

            m_loss = np.mean(val_losses)
            if m_loss < best_val_loss:
                best_val_loss = m_loss
                best_val_probs = np.vstack(val_probs_l)
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        tst_probs_l = []
        with torch.no_grad():
            for b_ids, b_mask in tst_loader:
                b_ids, b_mask = b_ids.to(device), b_mask.to(device)
                out = model(input_ids=b_ids, attention_mask=b_mask)
                tst_probs_l.append(torch.softmax(out.logits, dim=-1).cpu().numpy())
        test_probs = np.vstack(tst_probs_l)
        return best_val_probs, test_probs
    except Exception as e:
        print(f"Transformer fine-tuning skipped: {e}")
        return None, None


# =========================================================================
# 4. Stratified K-Fold Training & Evaluation
# =========================================================================
oof_lgb = np.zeros((num_samples, num_classes), dtype=np.float32)
oof_xgb = np.zeros((num_samples, num_classes), dtype=np.float32)
oof_cat = np.zeros((num_samples, num_classes), dtype=np.float32)
oof_lr = np.zeros((num_samples, num_classes), dtype=np.float32)
oof_ridge = np.zeros((num_samples, num_classes), dtype=np.float32)
oof_mlp = np.zeros((num_samples, num_classes), dtype=np.float32)
oof_tf = np.zeros((num_samples, num_classes), dtype=np.float32)

test_preds_lgb = np.zeros((num_test, num_classes), dtype=np.float32)
test_preds_xgb = np.zeros((num_test, num_classes), dtype=np.float32)
test_preds_cat = np.zeros((num_test, num_classes), dtype=np.float32)
test_preds_lr = np.zeros((num_test, num_classes), dtype=np.float32)
test_preds_ridge = np.zeros((num_test, num_classes), dtype=np.float32)
test_preds_mlp = np.zeros((num_test, num_classes), dtype=np.float32)
test_preds_tf = np.zeros((num_test, num_classes), dtype=np.float32)

has_transformer = True

for fold in range(n_splits):
    trn_idx_f = np.where(folds != fold)[0]
    val_idx_f = np.where(folds == fold)[0]

    X_trn, y_trn = X_train_full[trn_idx_f], y_train_full[trn_idx_f]
    X_val, y_val = X_train_full[val_idx_f], y_train_full[val_idx_f]

    X_trn_sp = X_train_sparse[trn_idx_f]
    X_val_sp = X_train_sparse[val_idx_f]

    # --- LightGBM ---
    lgb_model = get_lgbm_model(num_classes=num_classes, random_state=42 + fold)
    callbacks = [
        lgb.early_stopping(stopping_rounds=50, verbose=False),
        lgb.log_evaluation(period=0),
    ]
    lgb_model.fit(
        X_trn,
        y_trn,
        eval_set=[(X_val, y_val)],
        callbacks=callbacks,
    )
    oof_lgb[val_idx_f] = lgb_model.predict_proba(X_val)
    test_preds_lgb += lgb_model.predict_proba(X_test_full) / n_splits

    # --- XGBoost ---
    xgb_model = get_xgb_model(num_classes=num_classes, random_state=42 + fold)
    xgb_model.fit(X_trn, y_trn, eval_set=[(X_val, y_val)], verbose=False)
    oof_xgb[val_idx_f] = xgb_model.predict_proba(X_val)
    test_preds_xgb += xgb_model.predict_proba(X_test_full) / n_splits

    # --- CatBoost ---
    cat_model = get_catboost_model(num_classes=num_classes, random_state=42 + fold)
    cat_model.fit(
        X_trn,
        y_trn,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=50,
        verbose=False,
    )
    oof_cat[val_idx_f] = cat_model.predict_proba(X_val)
    test_preds_cat += cat_model.predict_proba(X_test_full) / n_splits

    # --- Logistic Regression on Sparse Text ---
    lr_model = LogisticRegression(C=2.0, max_iter=1000, solver="lbfgs", random_state=42 + fold)
    lr_model.fit(X_trn_sp, y_trn)
    oof_lr[val_idx_f] = lr_model.predict_proba(X_val_sp)
    test_preds_lr += lr_model.predict_proba(X_test_sparse) / n_splits

    # --- Calibrated Ridge Classifier on Sparse Text ---
    ridge_base = RidgeClassifier(alpha=2.0, random_state=42 + fold)
    ridge_model = CalibratedClassifierCV(ridge_base, cv=3)
    ridge_model.fit(X_trn_sp, y_trn)
    oof_ridge[val_idx_f] = ridge_model.predict_proba(X_val_sp)
    test_preds_ridge += ridge_model.predict_proba(X_test_sparse) / n_splits

    # --- PyTorch Tabular Deep MLP ---
    val_mlp_p, tst_mlp_p = train_mlp_fold(X_trn, y_trn, X_val, y_val, X_test_full, num_classes, seed=42 + fold)
    oof_mlp[val_idx_f] = val_mlp_p
    test_preds_mlp += tst_mlp_p / n_splits

    # --- Fine-Tuned Pretrained Transformer Sequence Classifier ---
    if has_transformer:
        trn_txt_f = [train_combined_text[i] for i in trn_idx_f]
        val_txt_f = [train_combined_text[i] for i in val_idx_f]
        val_tf_p, tst_tf_p = train_transformer_fold(trn_txt_f, y_trn, val_txt_f, y_val, test_combined_text, num_classes, seed=42 + fold)
        if val_tf_p is not None:
            oof_tf[val_idx_f] = val_tf_p
            test_preds_tf += tst_tf_p / n_splits
        else:
            has_transformer = False

    f_lgb = accuracy_score(y_val, np.argmax(oof_lgb[val_idx_f], axis=1))
    f_xgb = accuracy_score(y_val, np.argmax(oof_xgb[val_idx_f], axis=1))
    f_cat = accuracy_score(y_val, np.argmax(oof_cat[val_idx_f], axis=1))
    f_lr = accuracy_score(y_val, np.argmax(oof_lr[val_idx_f], axis=1))
    f_rd = accuracy_score(y_val, np.argmax(oof_ridge[val_idx_f], axis=1))
    f_mlp = accuracy_score(y_val, np.argmax(oof_mlp[val_idx_f], axis=1))
    log_str = f"Fold {fold+1}/{n_splits} | LGB: {f_lgb:.4f} | XGB: {f_xgb:.4f} | CAT: {f_cat:.4f} | LR: {f_lr:.4f} | Ridge: {f_rd:.4f} | MLP: {f_mlp:.4f}"
    if has_transformer and oof_tf[val_idx_f].sum() > 0:
        f_tf = accuracy_score(y_val, np.argmax(oof_tf[val_idx_f], axis=1))
        log_str += f" | TF: {f_tf:.4f}"
    print(log_str)

# Individual Model Performances
model_names = ["LightGBM", "XGBoost", "CatBoost", "LogisticRegression", "RidgeClassifier", "PyTorch_MLP"]
oofs = [oof_lgb, oof_xgb, oof_cat, oof_lr, oof_ridge, oof_mlp]
test_preds = [test_preds_lgb, test_preds_xgb, test_preds_cat, test_preds_lr, test_preds_ridge, test_preds_mlp]

if has_transformer and oof_tf.sum() > 0:
    model_names.append("FineTunedTransformer")
    oofs.append(oof_tf)
    test_preds.append(test_preds_tf)

# Constrained Optimization for Ensemble Weights
def ensemble_loss(weights):
    w = np.array(weights)
    w = np.maximum(w, 0.0)
    if w.sum() == 0:
        return 999.0
    w = w / w.sum()
    blend = sum(w[i] * oofs[i] for i in range(len(oofs)))
    blend = np.clip(blend, 1e-15, 1 - 1e-15)
    blend = blend / blend.sum(axis=1, keepdims=True)
    return log_loss(y_train_full, blend)

constraints = ({'type': 'eq', 'fun': lambda w: 1.0 - sum(w)})
bounds = [(0.0, 1.0) for _ in range(len(oofs))]
init_weights = [1.0 / len(oofs)] * len(oofs)

opt_res = minimize(
    ensemble_loss,
    init_weights,
    method='SLSQP',
    bounds=bounds,
    constraints=constraints,
    options={'maxiter': 500, 'ftol': 1e-9}
)

opt_weights = np.maximum(opt_res.x, 0.0)
opt_weights = opt_weights / opt_weights.sum()

for name, oof, w in zip(model_names, oofs, opt_weights):
    acc = accuracy_score(y_train_full, np.argmax(oof, axis=1))
    print(f"OOF Accuracy - {name:18s}: {acc:.5f} | Optimal Weight: {w:.4f}")

# Weighted Probability Blend
oof_ensemble = sum(w * oof for w, oof in zip(opt_weights, oofs))
oof_predictions = np.argmax(oof_ensemble, axis=1)

overall_score = float(accuracy_score(y_train_full, oof_predictions))
print(f"Overall Optimized Ensemble OOF Accuracy: {overall_score:.5f}")

# =========================================================================
# 5. Final Test Submission Generation
# =========================================================================
test_ensemble = sum(w * t_pred for w, t_pred in zip(opt_weights, test_preds))
test_predictions_encoded = np.argmax(test_ensemble, axis=1)
test_predictions_labels = label_encoder.inverse_transform(test_predictions_encoded)

submission_df = pd.DataFrame(
    {
        "id": np.arange(len(test_predictions_labels)),
        "Decision type": test_predictions_labels,
    }
)

submission_path = "./submission/submission.csv"
submission_df.to_csv(submission_path, index=False)

print(f"Submission successfully written to {submission_path}")
print(f"Submission rows: {len(submission_df)}, columns: {list(submission_df.columns)}")
print(f"Final Validation Score: {overall_score}")