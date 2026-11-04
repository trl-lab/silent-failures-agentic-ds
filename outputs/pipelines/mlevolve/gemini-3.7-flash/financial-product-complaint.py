import os
import re
import warnings
import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.special import softmax
from scipy.optimize import minimize
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import copy

warnings.filterwarnings("ignore")

# 1. Environment Setup
os.makedirs("./submission", exist_ok=True)
os.makedirs("./working", exist_ok=True)

# 2. Data Ingestion
print("Loading train and test datasets from ./input...")
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "Company response to consumer"
if target_col not in train_df.columns:
    raise ValueError(f"Target column '{target_col}' not found in training dataset.")

# Target encoding
label_encoder = LabelEncoder()
train_df["target"] = label_encoder.fit_transform(train_df[target_col].astype(str))
num_classes = len(label_encoder.classes_)
target_classes = label_encoder.classes_.tolist()
print(f"Target classes ({num_classes}): {target_classes}")

# 5-Fold Stratified Split
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
train_df["fold"] = -1
for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df["target"])):
    train_df.loc[val_idx, "fold"] = fold


# 3. Feature Engineering Functions
def extract_datetime_features(df):
    """Extract temporal, lag, and calendar features from date columns."""
    features = pd.DataFrame(index=df.index)
    date_cols = [c for c in df.columns if "date" in c.lower()]

    parsed_dates = {}
    for col in date_cols:
        parsed = pd.to_datetime(df[col], errors="coerce")
        parsed_dates[col] = parsed
        col_prefix = re.sub(r"[^a-zA-Z0-9_]", "_", col)
        features[f"{col_prefix}_year"] = parsed.dt.year.fillna(-1).astype(np.int32)
        features[f"{col_prefix}_month"] = parsed.dt.month.fillna(-1).astype(np.int32)
        features[f"{col_prefix}_day"] = parsed.dt.day.fillna(-1).astype(np.int32)
        features[f"{col_prefix}_dayofweek"] = parsed.dt.dayofweek.fillna(-1).astype(
            np.int32
        )
        features[f"{col_prefix}_is_weekend"] = parsed.dt.dayofweek.isin([5, 6]).astype(
            np.int32
        )
        features[f"{col_prefix}_quarter"] = parsed.dt.quarter.fillna(-1).astype(
            np.int32
        )
        features[f"{col_prefix}_dayofyear"] = parsed.dt.dayofyear.fillna(-1).astype(
            np.int32
        )
        features[f"{col_prefix}_is_null"] = df[col].isnull().astype(np.int32)

    # Response lag calculation between received and sent dates
    rec_cols = [c for c in date_cols if "received" in c.lower()]
    sent_cols = [c for c in date_cols if "sent" in c.lower()]
    if rec_cols and sent_cols:
        rec_col = rec_cols[0]
        sent_col = sent_cols[0]
        diff_days = (
            parsed_dates[sent_col] - parsed_dates[rec_col]
        ).dt.total_seconds() / (24 * 3600)
        features["days_to_send_company"] = diff_days.fillna(-1).astype(np.float32)
        features["sent_same_day"] = (diff_days == 0).astype(np.int32)
        features["negative_lag_flag"] = (diff_days < 0).astype(np.int32)

    return features


def extract_text_statistics(df):
    """Extract lexical and structural statistics from text/narrative columns."""
    features = pd.DataFrame(index=df.index)
    text_cols = [
        c
        for c in df.columns
        if "narrative" in c.lower()
        or "complaint" in c.lower()
        or "text" in c.lower()
        or "response" in c.lower()
    ]

    for col in text_cols:
        if col in [target_col, "target", "fold"]:
            continue
        col_prefix = re.sub(r"[^a-zA-Z0-9_]", "_", col)
        s = df[col].fillna("").astype(str)
        features[f"{col_prefix}_is_missing"] = (s.str.strip() == "").astype(np.int32)
        features[f"{col_prefix}_char_len"] = s.apply(len).astype(np.float32)
        features[f"{col_prefix}_word_count"] = s.apply(lambda x: len(x.split())).astype(
            np.float32
        )
        features[f"{col_prefix}_avg_word_len"] = (
            features[f"{col_prefix}_char_len"]
            / (features[f"{col_prefix}_word_count"] + 1e-5)
        ).astype(np.float32)
        features[f"{col_prefix}_num_uppercase"] = s.apply(
            lambda x: sum(1 for c in x if c.isupper())
        ).astype(np.float32)
        features[f"{col_prefix}_num_digits"] = s.apply(
            lambda x: sum(1 for c in x if c.isdigit())
        ).astype(np.float32)
        features[f"{col_prefix}_num_exclamations"] = s.apply(
            lambda x: x.count("!")
        ).astype(np.float32)
        features[f"{col_prefix}_num_questions"] = s.apply(
            lambda x: x.count("?")
        ).astype(np.float32)
        features[f"{col_prefix}_num_dollars"] = s.apply(lambda x: x.count("$")).astype(
            np.float32
        )
        features[f"{col_prefix}_redacted_count"] = s.apply(
            lambda x: x.count("X") + x.count("x")
        ).astype(np.float32)

    return features


def extract_interaction_features(df):
    """Construct hierarchical domain interaction features."""
    features = pd.DataFrame(index=df.index)
    cols = df.columns

    pairs = [
        ("Company", "Issue"),
        ("Company", "Product"),
        ("State", "Issue"),
        ("Submitted via", "Company"),
        ("Product", "Issue"),
        ("Product", "Sub-product"),
        ("Issue", "Sub-issue"),
        ("Submitted via", "Product"),
        ("State", "Product"),
    ]

    for col1, col2 in pairs:
        if col1 in cols and col2 in cols:
            combined = (
                df[col1].fillna("Missing").astype(str)
                + "___"
                + df[col2].fillna("Missing").astype(str)
            )
            feat_name = f"inter_{re.sub(r'[^a-zA-Z0-9_]', '_', col1)}_{re.sub(r'[^a-zA-Z0-9_]', '_', col2)}"
            features[feat_name] = combined

    return features


print(
    "Extracting features (temporal, lexical statistics, and categorical interactions)..."
)
train_temporal = extract_datetime_features(train_df)
test_temporal = extract_datetime_features(test_df)

train_text_stats = extract_text_statistics(train_df)
test_text_stats = extract_text_statistics(test_df)

train_interactions = extract_interaction_features(train_df)
test_interactions = extract_interaction_features(test_df)

# Categorical column identification
categorical_cols = [
    c
    for c in train_df.columns
    if c not in [target_col, "target", "fold"]
    and c not in train_temporal.columns
    and c not in train_text_stats.columns
    and (train_df[c].dtype == "object" or train_df[c].dtype.name == "category")
]

train_all_cats = pd.concat(
    [train_df[categorical_cols].fillna("Missing").astype(str), train_interactions],
    axis=1,
)
test_all_cats = pd.concat(
    [test_df[categorical_cols].fillna("Missing").astype(str), test_interactions], axis=1
)
all_cat_cols = list(train_all_cats.columns)

train_cat_encoded = pd.DataFrame(index=train_df.index)
test_cat_encoded = pd.DataFrame(index=test_df.index)

print(f"Encoding {len(all_cat_cols)} categorical & interaction features...")
for col in all_cat_cols:
    clean_name = re.sub(r"[^a-zA-Z0-9_]", "_", col)

    # Frequency encoding strictly derived from training partition
    freq_map = train_all_cats[col].value_counts().to_dict()
    train_cat_encoded[f"{clean_name}_freq"] = (
        train_all_cats[col].map(freq_map).fillna(0).astype(np.float32)
    )
    test_cat_encoded[f"{clean_name}_freq"] = (
        test_all_cats[col].map(freq_map).fillna(0).astype(np.float32)
    )

    # Ordinal label encoding
    le = LabelEncoder()
    unique_train_vals = list(train_all_cats[col].unique())
    if "<UNKNOWN>" not in unique_train_vals:
        unique_train_vals.append("<UNKNOWN>")
    le.fit(unique_train_vals)

    val_map = {val: idx for idx, val in enumerate(le.classes_)}
    unknown_idx = val_map["<UNKNOWN>"]

    train_cat_encoded[f"{clean_name}_code"] = (
        train_all_cats[col].map(lambda x: val_map.get(x, unknown_idx)).astype(np.int32)
    )
    test_cat_encoded[f"{clean_name}_code"] = (
        test_all_cats[col].map(lambda x: val_map.get(x, unknown_idx)).astype(np.int32)
    )

# Group Aggregation Statistics
print("Computing multi-level group aggregation statistics...")
train_temp_merged = train_df.copy()
test_temp_merged = test_df.copy()
if "days_to_send_company" in train_temporal.columns:
    train_temp_merged["days_to_send"] = train_temporal["days_to_send_company"]
    test_temp_merged["days_to_send"] = test_temporal["days_to_send_company"]
else:
    train_temp_merged["days_to_send"] = 0
    test_temp_merged["days_to_send"] = 0

train_group_feats = pd.DataFrame(index=train_df.index)
test_group_feats = pd.DataFrame(index=test_df.index)

if "Company" in train_df.columns:
    company_counts = train_df["Company"].value_counts().to_dict()
    train_group_feats["company_complaint_count"] = train_df["Company"].map(company_counts).fillna(0).astype(np.float32)
    test_group_feats["company_complaint_count"] = test_df["Company"].map(company_counts).fillna(0).astype(np.float32)

    company_days_mean = train_temp_merged.groupby("Company")["days_to_send"].mean().to_dict()
    company_days_std = train_temp_merged.groupby("Company")["days_to_send"].std().fillna(0).to_dict()

    train_group_feats["company_days_to_send_mean"] = train_df["Company"].map(company_days_mean).fillna(-1).astype(np.float32)
    test_group_feats["company_days_to_send_mean"] = test_df["Company"].map(company_days_mean).fillna(-1).astype(np.float32)

    train_group_feats["company_days_to_send_std"] = train_df["Company"].map(company_days_std).fillna(0).astype(np.float32)
    test_group_feats["company_days_to_send_std"] = test_df["Company"].map(company_days_std).fillna(0).astype(np.float32)

    if "Product" in train_df.columns:
        company_prod_diversity = train_df.groupby("Company")["Product"].nunique().to_dict()
        train_group_feats["company_product_diversity"] = train_df["Company"].map(company_prod_diversity).fillna(0).astype(np.float32)
        test_group_feats["company_product_diversity"] = test_df["Company"].map(company_prod_diversity).fillna(0).astype(np.float32)

    if "Issue" in train_df.columns:
        company_issue_diversity = train_df.groupby("Company")["Issue"].nunique().to_dict()
        train_group_feats["company_issue_diversity"] = train_df["Company"].map(company_issue_diversity).fillna(0).astype(np.float32)
        test_group_feats["company_issue_diversity"] = test_df["Company"].map(company_issue_diversity).fillna(0).astype(np.float32)

if "Product" in train_df.columns:
    prod_counts = train_df["Product"].value_counts().to_dict()
    train_group_feats["product_complaint_count"] = train_df["Product"].map(prod_counts).fillna(0).astype(np.float32)
    test_group_feats["product_complaint_count"] = test_df["Product"].map(prod_counts).fillna(0).astype(np.float32)

    prod_days_mean = train_temp_merged.groupby("Product")["days_to_send"].mean().to_dict()
    train_group_feats["product_days_to_send_mean"] = train_df["Product"].map(prod_days_mean).fillna(-1).astype(np.float32)
    test_group_feats["product_days_to_send_mean"] = test_df["Product"].map(prod_days_mean).fillna(-1).astype(np.float32)

if "Issue" in train_df.columns:
    issue_counts = train_df["Issue"].value_counts().to_dict()
    train_group_feats["issue_complaint_count"] = train_df["Issue"].map(issue_counts).fillna(0).astype(np.float32)
    test_group_feats["issue_complaint_count"] = test_df["Issue"].map(issue_counts).fillna(0).astype(np.float32)

# Out-of-Fold Multiclass Target Encoding with Interaction Pairs
print("Generating regularized out-of-fold multi-class target encodings for interaction pairs...")
candidate_te_cols = [
    "Company",
    "Product",
    "Issue",
    "Sub-issue",
    "State",
    "Submitted via",
    "Company public response",
    "inter_Company_Issue",
    "inter_Company_Product",
    "inter_State_Issue",
    "inter_Submitted_via_Company",
    "inter_Product_Issue",
]
te_cols = [c for c in candidate_te_cols if c in train_all_cats.columns]

train_te_feats = pd.DataFrame(index=train_df.index)
test_te_feats = pd.DataFrame(index=test_df.index)

smoothing = 10.0
global_priors = [(train_df["target"] == c).mean() for c in range(num_classes)]

for col in te_cols:
    clean_col = re.sub(r"[^a-zA-Z0-9_]", "_", col)
    for c in range(num_classes):
        train_te_feats[f"te_{clean_col}_c{c}"] = np.float32(global_priors[c])
        test_te_feats[f"te_{clean_col}_c{c}"] = np.float32(0.0)

for fold in range(n_splits):
    trn_mask = (train_df["fold"] != fold).values
    val_mask = (train_df["fold"] == fold).values

    for col in te_cols:
        clean_col = re.sub(r"[^a-zA-Z0-9_]", "_", col)
        series_trn = train_all_cats.loc[trn_mask, col]
        y_trn = train_df.loc[trn_mask, "target"]
        series_val = train_all_cats.loc[val_mask, col]
        series_test = test_all_cats[col]

        counts = series_trn.value_counts()
        for c in range(num_classes):
            class_counts = series_trn[y_trn == c].value_counts()
            smoothed_prob = ((class_counts + smoothing * global_priors[c]) / (counts + smoothing)).to_dict()

            train_te_feats.loc[val_mask, f"te_{clean_col}_c{c}"] = (
                series_val.map(smoothed_prob).fillna(global_priors[c]).astype(np.float32)
            )
            test_te_feats[f"te_{clean_col}_c{c}"] += (
                series_test.map(smoothed_prob).fillna(global_priors[c]).astype(np.float32) / n_splits
            )

# Multi-Field Semantic Text Embeddings (TF-IDF + SVD) and Linear Classifiers
print("Extracting multi-field TF-IDF, SVD representations, and out-of-fold linear text models...")
train_text_proc_dfs = []
test_text_proc_dfs = []

text_field_configs = [
    ("Consumer complaint narrative", {"max_features": 30000, "ngram_range": (1, 2), "min_df": 3, "svd_dim": 32}),
    ("Issue", {"max_features": 6000, "ngram_range": (1, 2), "min_df": 2, "svd_dim": 16}),
    ("Sub-issue", {"max_features": 6000, "ngram_range": (1, 2), "min_df": 2, "svd_dim": 16}),
    ("Company public response", {"max_features": 4000, "ngram_range": (1, 2), "min_df": 2, "svd_dim": 12}),
]

for field_name, cfg in text_field_configs:
    if field_name not in train_df.columns:
        continue
    clean_field = re.sub(r"[^a-zA-Z0-9_]", "_", field_name)
    trn_raw = train_df[field_name].fillna("").astype(str)
    tst_raw = test_df[field_name].fillna("").astype(str)

    vec = TfidfVectorizer(
        max_features=cfg["max_features"],
        ngram_range=cfg["ngram_range"],
        min_df=cfg["min_df"],
        sublinear_tf=True,
    )
    trn_tfidf_mat = vec.fit_transform(trn_raw)
    tst_tfidf_mat = vec.transform(tst_raw)

    svd_dim = min(cfg["svd_dim"], trn_tfidf_mat.shape[1] - 1)
    if svd_dim > 1:
        svd_model = TruncatedSVD(n_components=svd_dim, random_state=42)
        trn_svd_arr = svd_model.fit_transform(trn_tfidf_mat)
        tst_svd_arr = svd_model.transform(tst_tfidf_mat)

        svd_cols = [f"svd_{clean_field}_{i}" for i in range(svd_dim)]
        train_text_proc_dfs.append(pd.DataFrame(trn_svd_arr, columns=svd_cols, index=train_df.index).astype(np.float32))
        test_text_proc_dfs.append(pd.DataFrame(tst_svd_arr, columns=svd_cols, index=test_df.index).astype(np.float32))

    oof_lr_probs = np.zeros((len(train_df), num_classes), dtype=np.float32)
    test_lr_probs = np.zeros((len(test_df), num_classes), dtype=np.float32)

    for fold in range(n_splits):
        trn_mask = (train_df["fold"] != fold).values
        val_mask = (train_df["fold"] == fold).values

        lr = LogisticRegression(C=1.0, max_iter=200, solver="lbfgs", multi_class="multinomial", random_state=42 + fold)
        lr.fit(trn_tfidf_mat[trn_mask], train_df.loc[trn_mask, "target"])
        oof_lr_probs[val_mask] = lr.predict_proba(trn_tfidf_mat[val_mask]).astype(np.float32)
        test_lr_probs += lr.predict_proba(tst_tfidf_mat).astype(np.float32) / n_splits

    lr_cols = [f"lr_{clean_field}_prob_{c}" for c in range(num_classes)]
    train_text_proc_dfs.append(pd.DataFrame(oof_lr_probs, columns=lr_cols, index=train_df.index))
    test_text_proc_dfs.append(pd.DataFrame(test_lr_probs, columns=lr_cols, index=test_df.index))

# Also extract combined full-narrative representation
all_text_cols = [c for c in train_df.columns if "narrative" in c.lower() or "issue" in c.lower() or "response" in c.lower()]
all_text_cols = [c for c in all_text_cols if c not in [target_col, "target", "fold"]]
train_comb_text = train_df[all_text_cols].fillna("").agg(" ".join, axis=1)
test_comb_text = test_df[all_text_cols].fillna("").agg(" ".join, axis=1)

comb_tfidf = TfidfVectorizer(max_features=40000, ngram_range=(1, 2), min_df=3, sublinear_tf=True)
train_comb_tfidf = comb_tfidf.fit_transform(train_comb_text)
test_comb_tfidf = comb_tfidf.transform(test_comb_text)

comb_svd = TruncatedSVD(n_components=48, random_state=42)
train_comb_svd = comb_svd.fit_transform(train_comb_tfidf)
test_comb_svd = comb_svd.transform(test_comb_tfidf)

comb_svd_cols = [f"svd_comb_{i}" for i in range(48)]
train_text_proc_dfs.append(pd.DataFrame(train_comb_svd, columns=comb_svd_cols, index=train_df.index).astype(np.float32))
test_text_proc_dfs.append(pd.DataFrame(test_comb_svd, columns=comb_svd_cols, index=test_df.index).astype(np.float32))

train_text_engineered_df = pd.concat(train_text_proc_dfs, axis=1)
test_text_engineered_df = pd.concat(test_text_proc_dfs, axis=1)

# Numeric Features
numeric_cols = [
    c
    for c in train_df.columns
    if c not in [target_col, "target", "fold"]
    and c not in categorical_cols
    and c not in all_text_cols
    and np.issubdtype(train_df[c].dtype, np.number)
]

train_num_df = (
    train_df[numeric_cols].fillna(-1).astype(np.float32)
    if numeric_cols
    else pd.DataFrame(index=train_df.index)
)
test_num_df = (
    test_df[numeric_cols].fillna(-1).astype(np.float32)
    if numeric_cols
    else pd.DataFrame(index=test_df.index)
)

# Assemble feature matrices
X = pd.concat(
    [
        train_temporal,
        train_text_stats,
        train_cat_encoded,
        train_group_feats,
        train_te_feats,
        train_text_engineered_df,
        train_num_df,
    ],
    axis=1,
)
X_test = pd.concat(
    [
        test_temporal,
        test_text_stats,
        test_cat_encoded,
        test_group_feats,
        test_te_feats,
        test_text_engineered_df,
        test_num_df,
    ],
    axis=1,
)

# Sanitize column names for compatibility across tree-based algorithms
feature_names = list(X.columns)
X_test = X_test[feature_names]
clean_names = [re.sub(r'[ ,:;{}()=\[\]"<>]+', "_", str(c)) for c in feature_names]
X.columns = clean_names
X_test.columns = clean_names

X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(0)

y = train_df["target"].values.astype(np.int64)
folds = train_df["fold"].values.astype(np.int32)
n_splits = len(np.unique(folds))
n_train = len(train_df)
n_test = len(test_df)

print(f"Engineered feature shape: X={X.shape}, X_test={X_test.shape}")

# 4. Deep Tabular Neural Network Definition
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"PyTorch training device: {device}")


class TabularResNet(nn.Module):
    """Deep Tabular ResNet architecture with SiLU, BatchNorm, and LayerNorm."""

    def __init__(self, in_features, num_classes, hidden_dim=256, dropout=0.25):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.block1_lin1 = nn.Linear(hidden_dim, hidden_dim)
        self.block1_bn1 = nn.BatchNorm1d(hidden_dim)
        self.block1_act = nn.SiLU()
        self.block1_drop = nn.Dropout(dropout)
        self.block1_lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.block1_bn2 = nn.BatchNorm1d(hidden_dim)

        self.block2_lin1 = nn.Linear(hidden_dim, hidden_dim)
        self.block2_bn1 = nn.BatchNorm1d(hidden_dim)
        self.block2_act = nn.SiLU()
        self.block2_drop = nn.Dropout(dropout)
        self.block2_lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.block2_bn2 = nn.BatchNorm1d(hidden_dim)

        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        x = self.input_layer(x)
        res1 = x
        out1 = self.block1_drop(self.block1_act(self.block1_bn1(self.block1_lin1(x))))
        out1 = self.block1_bn2(self.block1_lin2(out1))
        x = nn.functional.silu(res1 + out1)

        res2 = x
        out2 = self.block2_drop(self.block2_act(self.block2_bn1(self.block2_lin1(x))))
        out2 = self.block2_bn2(self.block2_lin2(out2))
        x = nn.functional.silu(res2 + out2)

        return self.head(x)


# 5. Multi-Model 5-Fold Cross-Validation
oof_preds_lgb = np.zeros((n_train, num_classes), dtype=np.float32)
oof_preds_xgb = np.zeros((n_train, num_classes), dtype=np.float32)
oof_preds_cat = np.zeros((n_train, num_classes), dtype=np.float32)
oof_preds_nn = np.zeros((n_train, num_classes), dtype=np.float32)

test_preds_lgb = np.zeros((n_test, num_classes), dtype=np.float32)
test_preds_xgb = np.zeros((n_test, num_classes), dtype=np.float32)
test_preds_cat = np.zeros((n_test, num_classes), dtype=np.float32)
test_preds_nn = np.zeros((n_test, num_classes), dtype=np.float32)

print(
    f"Starting {n_splits}-fold training across LightGBM, XGBoost, CatBoost, and Deep Tabular Neural Network..."
)

for fold in range(n_splits):
    train_mask = folds != fold
    val_mask = folds == fold

    X_train_fold, y_train_fold = X.iloc[train_mask], y[train_mask]
    X_val_fold, y_val_fold = X.iloc[val_mask], y[val_mask]

    # --- LightGBM ---
    lgb_clf = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=num_classes,
        metric="multi_logloss",
        n_estimators=1200,
        learning_rate=0.035,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=30,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42 + fold,
        n_jobs=-1,
        verbose=-1,
    )
    lgb_clf.fit(
        X_train_fold,
        y_train_fold,
        eval_set=[(X_val_fold, y_val_fold)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )
    val_lgb_prob = lgb_clf.predict_proba(X_val_fold)
    oof_preds_lgb[val_mask] = val_lgb_prob
    test_preds_lgb += lgb_clf.predict_proba(X_test) / n_splits
    acc_lgb = accuracy_score(y_val_fold, np.argmax(val_lgb_prob, axis=1))

    # --- XGBoost ---
    xgb_clf = XGBClassifier(
        objective="multi:softprob",
        num_class=num_classes,
        eval_metric="mlogloss",
        n_estimators=1000,
        learning_rate=0.035,
        max_depth=6,
        min_child_weight=3,
        subsample=0.85,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        tree_method="hist",
        early_stopping_rounds=50,
        random_state=42 + fold,
        n_jobs=-1,
    )
    xgb_clf.fit(
        X_train_fold,
        y_train_fold,
        eval_set=[(X_val_fold, y_val_fold)],
        verbose=False,
    )
    val_xgb_prob = xgb_clf.predict_proba(X_val_fold)
    oof_preds_xgb[val_mask] = val_xgb_prob
    test_preds_xgb += xgb_clf.predict_proba(X_test) / n_splits
    acc_xgb = accuracy_score(y_val_fold, np.argmax(val_xgb_prob, axis=1))

    # --- CatBoost ---
    cat_clf = CatBoostClassifier(
        loss_function="MultiClass",
        eval_metric="MultiClass",
        iterations=1000,
        learning_rate=0.045,
        depth=6,
        l2_leaf_reg=3.0,
        random_seed=42 + fold,
        early_stopping_rounds=50,
        verbose=False,
        thread_count=-1,
    )
    cat_clf.fit(
        X_train_fold,
        y_train_fold,
        eval_set=(X_val_fold, y_val_fold),
        verbose=False,
    )
    val_cat_prob = cat_clf.predict_proba(X_val_fold)
    oof_preds_cat[val_mask] = val_cat_prob
    test_preds_cat += cat_clf.predict_proba(X_test) / n_splits
    acc_cat = accuracy_score(y_val_fold, np.argmax(val_cat_prob, axis=1))

    # --- Deep Tabular Neural Network ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_fold)
    X_val_scaled = scaler.transform(X_val_fold)
    X_test_scaled = scaler.transform(X_test)

    train_dataset = TensorDataset(
        torch.tensor(X_train_scaled, dtype=torch.float32),
        torch.tensor(y_train_fold, dtype=torch.long),
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val_scaled, dtype=torch.float32),
        torch.tensor(y_val_fold, dtype=torch.long),
    )

    train_loader = DataLoader(
        train_dataset, batch_size=256, shuffle=True, drop_last=False
    )
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)
    test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)

    nn_model = TabularResNet(
        in_features=X.shape[1],
        num_classes=num_classes,
        hidden_dim=256,
        dropout=0.25,
    ).to(device)
    optimizer = optim.AdamW(
        nn_model.parameters(), lr=1e-3, weight_decay=1e-4, eps=1e-8
    )
    epochs = 20
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-5
    )
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    best_nn_val_probs = None
    best_model_state = None

    for epoch in range(epochs):
        nn_model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = nn_model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(nn_model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        nn_model.eval()
        val_losses = []
        epoch_val_probs = []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                logits = nn_model(batch_x)
                loss = criterion(logits, batch_y)
                val_losses.append(loss.item())
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                epoch_val_probs.append(probs)

        mean_val_loss = np.mean(val_losses)
        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            best_nn_val_probs = np.vstack(epoch_val_probs)
            best_model_state = copy.deepcopy(nn_model.state_dict())

    oof_preds_nn[val_mask] = best_nn_val_probs
    acc_nn = accuracy_score(y_val_fold, np.argmax(best_nn_val_probs, axis=1))

    if best_model_state is not None:
        nn_model.load_state_dict(best_model_state)
    nn_model.eval()
    with torch.no_grad():
        test_nn_fold_probs = (
            torch.softmax(nn_model(test_tensor), dim=1).cpu().numpy()
        )
        test_preds_nn += test_nn_fold_probs / n_splits

    print(
        f"Fold {fold + 1}/{n_splits} - Accuracies: LGBM={acc_lgb:.5f}, XGB={acc_xgb:.5f}, CatBoost={acc_cat:.5f}, TabularNN={acc_nn:.5f}"
    )

# 6. Out-of-Fold Probability Stacking & Meta-Learning
print("Training regularized meta-learner on out-of-fold probability representations...")
X_meta_train = np.hstack(
    [oof_preds_lgb, oof_preds_xgb, oof_preds_cat, oof_preds_nn]
)
X_meta_test = np.hstack(
    [test_preds_lgb, test_preds_xgb, test_preds_cat, test_preds_nn]
)

oof_preds_meta = np.zeros((n_train, num_classes), dtype=np.float32)
test_preds_meta = np.zeros((n_test, num_classes), dtype=np.float32)

for fold in range(n_splits):
    trn_mask = folds != fold
    val_mask = folds == fold

    meta_clf = LogisticRegression(
        C=0.5,
        max_iter=500,
        solver="lbfgs",
        multi_class="multinomial",
        random_state=42 + fold,
    )
    meta_clf.fit(X_meta_train[trn_mask], y[trn_mask])

    oof_preds_meta[val_mask] = meta_clf.predict_proba(
        X_meta_train[val_mask]
    ).astype(np.float32)
    test_preds_meta += (
        meta_clf.predict_proba(X_meta_test).astype(np.float32) / n_splits
    )

meta_oof_acc = accuracy_score(y, np.argmax(oof_preds_meta, axis=1))
print(f"Meta-Learner Stacking OOF Accuracy: {meta_oof_acc:.5f}")

# Weighted combination of meta-learner and model blend
blend_base_oof = (
    0.25 * oof_preds_lgb
    + 0.35 * oof_preds_xgb
    + 0.30 * oof_preds_cat
    + 0.10 * oof_preds_nn
)
blend_base_test = (
    0.25 * test_preds_lgb
    + 0.35 * test_preds_xgb
    + 0.30 * test_preds_cat
    + 0.10 * test_preds_nn
)

final_oof_prob = 0.60 * oof_preds_meta + 0.40 * blend_base_oof
final_test_prob = 0.60 * test_preds_meta + 0.40 * blend_base_test
pre_calib_acc = accuracy_score(y, np.argmax(final_oof_prob, axis=1))
print(f"Ensemble OOF Accuracy before calibration: {pre_calib_acc:.5f}")

# Class Multiplier Calibration via Nelder-Mead Optimization
print("Calibrating decision thresholds via Nelder-Mead...")


def eval_multipliers(weights):
    scaled_probs = final_oof_prob * weights
    preds = np.argmax(scaled_probs, axis=1)
    return -accuracy_score(y, preds)


init_multipliers = np.ones(num_classes, dtype=np.float64)
opt_res = minimize(
    eval_multipliers,
    init_multipliers,
    method="Nelder-Mead",
    options={"maxiter": 600, "disp": False},
)
optimal_multipliers = opt_res.x
print(f"Optimal multipliers: {optimal_multipliers}")

calibrated_oof_preds = np.argmax(final_oof_prob * optimal_multipliers, axis=1)
calibrated_acc = accuracy_score(y, calibrated_oof_preds)
print(f"OOF Accuracy after multiplier optimization: {calibrated_acc:.5f}")

if calibrated_acc >= pre_calib_acc:
    final_oof_pred_indices = calibrated_oof_preds
    final_test_pred_indices = np.argmax(
        final_test_prob * optimal_multipliers, axis=1
    )
else:
    final_oof_pred_indices = np.argmax(final_oof_prob, axis=1)
    final_test_pred_indices = np.argmax(final_test_prob, axis=1)

final_test_pred_labels = label_encoder.inverse_transform(final_test_pred_indices)

submission_df = pd.DataFrame(
    {"id": np.arange(n_test), "Company response to consumer": final_test_pred_labels}
)

submission_path = "./submission/submission.csv"
submission_df.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path} (shape: {submission_df.shape})")

# 7. Final Validation Metric Computation
final_score = accuracy_score(y, final_oof_pred_indices)
print(f"Final Validation Score: {final_score}")