import json
import os
import re
import catboost as cb
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModel, AutoTokenizer
from xgboost import XGBClassifier

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Directories setup
output_dir = "./working"
submission_dir = "./submission"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(submission_dir, exist_ok=True)

# 1. Load Data
train_path = "./input/train.parquet"
test_path = "./input/test.parquet"

df_train = pd.read_parquet(train_path).reset_index(drop=True)
df_test = pd.read_parquet(test_path).reset_index(drop=True)

# Target Variable Preparation
target_col = "classification"
le_target = LabelEncoder()
df_train["target"] = le_target.fit_transform(df_train[target_col].astype(str))

# Save target class mapping
target_mapping = {i: str(c) for i, c in enumerate(le_target.classes_)}
num_classes = len(le_target.classes_)

# Identify column types dynamically
all_cols = [c for c in df_train.columns if c not in [target_col, "target"]]
text_cols = [
    c
    for c in ["reason_for_recall", "product_description", "code_info", "more_code_info"]
    if c in all_cols
]
cat_cols = [
    c
    for c in [
        "recalling_firm",
        "city",
        "state",
        "country",
        "status",
        "voluntary_mandated",
    ]
    if c in all_cols
]
date_cols = [
    c
    for c in [
        "initiation_date",
        "report_date",
        "termination_date",
        "recall_initiation_date",
    ]
    if c in all_cols
]

# 2. Domain FDA Hazard Keyword Extraction
FDA_HAZARDS = [
    "listeria",
    "salmonella",
    "e. coli",
    "ecoli",
    "botulinum",
    "allergen",
    "undeclared",
    "peanut",
    "milk",
    "egg",
    "soy",
    "wheat",
    "almond",
    "walnut",
    "tree nut",
    "sesame",
    "metal",
    "glass",
    "plastic",
    "foreign material",
    "foreign object",
    "piece",
    "fragment",
    "mold",
    "yeast",
    "bacteria",
    "filth",
    "insect",
    "rodent",
    "temperature",
    "spoilage",
    "mislabel",
    "misbrand",
    "substitute",
    "unapproved",
    "potency",
    "decay",
    "decomposition",
]


def build_hazard_features(df, text_series):
    hazard_feats = {}
    lowered = text_series.fillna("").str.lower()
    for hazard in FDA_HAZARDS:
        hazard_feats[f"hazard_{hazard}"] = lowered.str.contains(
            hazard, regex=False
        ).astype(np.float32)

    hazard_df = pd.DataFrame(hazard_feats, index=df.index)
    hazard_df["hazard_total_count"] = hazard_df.sum(axis=1)

    high_risk_cols = [
        f"hazard_{h}"
        for h in [
            "listeria",
            "salmonella",
            "e. coli",
            "ecoli",
            "botulinum",
            "metal",
            "glass",
        ]
    ]
    existing_high_risk = [c for c in high_risk_cols if c in hazard_df.columns]
    hazard_df["hazard_high_risk_score"] = hazard_df[existing_high_risk].sum(axis=1)

    return hazard_df


train_text = df_train[text_cols].fillna("").agg(" ".join, axis=1)
test_text = df_test[text_cols].fillna("").agg(" ".join, axis=1)

train_hazards = build_hazard_features(df_train, train_text)
test_hazards = build_hazard_features(df_test, test_text)


# 3. Text Complexity & Metadata Features
def extract_text_stats(text_series):
    stats = pd.DataFrame(index=text_series.index)
    stats["text_char_len"] = text_series.str.len().astype(np.float32)
    stats["text_word_count"] = (
        text_series.str.split().str.len().fillna(0).astype(np.float32)
    )
    stats["text_digit_count"] = text_series.str.count(r"\d").astype(np.float32)
    stats["text_upper_ratio"] = (
        text_series.str.count(r"[A-Z]") / (stats["text_char_len"] + 1e-5)
    ).astype(np.float32)
    return stats


train_text_stats = extract_text_stats(train_text)
test_text_stats = extract_text_stats(test_text)

# 4. Field-Specific Text Features & Pre-trained Transformer Embeddings
def get_transformer_embeddings(
    text_series, model_name="sentence-transformers/all-MiniLM-L6-v2", batch_size=64
):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    text_list = text_series.tolist()
    all_embeddings = []
    with torch.no_grad():
        for i in range(0, len(text_list), batch_size):
            batch_texts = text_list[i : i + batch_size]
            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            ).to(device)
            outputs = model(**inputs)
            attention_mask = inputs["attention_mask"].unsqueeze(-1)
            token_embeddings = outputs.last_hidden_state
            sum_embeddings = torch.sum(token_embeddings * attention_mask, dim=1)
            sum_mask = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
            mean_pooled = (sum_embeddings / sum_mask).cpu().numpy()
            all_embeddings.append(mean_pooled)

    return np.vstack(all_embeddings)


reason_train = df_train["reason_for_recall"].fillna("").astype(str)
reason_test = df_test["reason_for_recall"].fillna("").astype(str)
product_train = df_train["product_description"].fillna("").astype(str)
product_test = df_test["product_description"].fillna("").astype(str)
code_train = (
    df_train["code_info"].fillna("").astype(str)
    if "code_info" in df_train.columns
    else pd.Series([""] * len(df_train))
)
code_test = (
    df_test["code_info"].fillna("").astype(str)
    if "code_info" in df_test.columns
    else pd.Series([""] * len(df_test))
)
more_code_train = (
    df_train["more_code_info"].fillna("").astype(str)
    if "more_code_info" in df_train.columns
    else pd.Series([""] * len(df_train))
)
more_code_test = (
    df_test["more_code_info"].fillna("").astype(str)
    if "more_code_info" in df_test.columns
    else pd.Series([""] * len(df_test))
)

def fit_tfidf_svd(train_series, test_series, n_components=32, max_features=10000):
    vec = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        stop_words="english",
        min_df=1,
    )
    try:
        train_tfidf = vec.fit_transform(train_series)
        test_tfidf = vec.transform(test_series)
        if train_tfidf.shape[1] == 0:
            raise ValueError("Empty vocabulary")
        n_comp = min(n_components, train_tfidf.shape[1] - 1) if train_tfidf.shape[1] > 1 else 1
        svd = TruncatedSVD(n_components=n_comp, random_state=42)
        train_svd = svd.fit_transform(train_tfidf)
        test_svd = svd.transform(test_tfidf)
        if train_svd.shape[1] < n_components:
            pad_len = n_components - train_svd.shape[1]
            train_svd = np.hstack([train_svd, np.zeros((len(train_series), pad_len), dtype=np.float32)])
            test_svd = np.hstack([test_svd, np.zeros((len(test_series), pad_len), dtype=np.float32)])
    except Exception:
        train_svd = np.zeros((len(train_series), n_components), dtype=np.float32)
        test_svd = np.zeros((len(test_series), n_components), dtype=np.float32)
    return train_svd, test_svd

train_svd_reason, test_svd_reason = fit_tfidf_svd(reason_train, reason_test, n_components=32)
train_svd_prod, test_svd_prod = fit_tfidf_svd(product_train, product_test, n_components=32)
train_svd_code, test_svd_code = fit_tfidf_svd(code_train, code_test, n_components=32)
train_svd_more_code, test_svd_more_code = fit_tfidf_svd(more_code_train, more_code_test, n_components=32)

# Transformer Embeddings for 'reason_for_recall' and 'product_description'
train_tf_emb_reason = get_transformer_embeddings(reason_train)
test_tf_emb_reason = get_transformer_embeddings(reason_test)

svd_tf_reason = TruncatedSVD(n_components=128, random_state=42)
train_svd_tf_reason = svd_tf_reason.fit_transform(train_tf_emb_reason)
test_svd_tf_reason = svd_tf_reason.transform(test_tf_emb_reason)

train_tf_emb_prod = get_transformer_embeddings(product_train)
test_tf_emb_prod = get_transformer_embeddings(product_test)

svd_tf_prod = TruncatedSVD(n_components=128, random_state=42)
train_svd_tf_prod = svd_tf_prod.fit_transform(train_tf_emb_prod)
test_svd_tf_prod = svd_tf_prod.transform(test_tf_emb_prod)

reason_cols = [f"svd_reason_{i}" for i in range(32)]
prod_cols = [f"svd_prod_{i}" for i in range(32)]
code_cols = [f"svd_code_{i}" for i in range(32)]
more_code_cols = [f"svd_more_code_{i}" for i in range(32)]
tf_reason_cols = [f"svd_tf_reason_{i}" for i in range(128)]
tf_prod_cols = [f"svd_tf_prod_{i}" for i in range(128)]

train_svd_df = pd.DataFrame(
    np.hstack(
        [
            train_svd_reason,
            train_svd_prod,
            train_svd_code,
            train_svd_more_code,
            train_svd_tf_reason,
            train_svd_tf_prod,
        ]
    ),
    columns=reason_cols
    + prod_cols
    + code_cols
    + more_code_cols
    + tf_reason_cols
    + tf_prod_cols,
    index=df_train.index,
)
test_svd_df = pd.DataFrame(
    np.hstack(
        [
            test_svd_reason,
            test_svd_prod,
            test_svd_code,
            test_svd_more_code,
            test_svd_tf_reason,
            test_svd_tf_prod,
        ]
    ),
    columns=reason_cols
    + prod_cols
    + code_cols
    + more_code_cols
    + tf_reason_cols
    + tf_prod_cols,
    index=df_test.index,
)


# 5. Temporal Features
def process_dates(df, date_columns):
    date_df = pd.DataFrame(index=df.index)
    parsed_dates = {}
    for col in date_columns:
        if col in df.columns:
            parsed = pd.to_datetime(df[col], errors="coerce")
            parsed_dates[col] = parsed
            date_df[f"{col}_year"] = parsed.dt.year.fillna(-1).astype(np.float32)
            date_df[f"{col}_month"] = parsed.dt.month.fillna(-1).astype(np.float32)
            date_df[f"{col}_dayofweek"] = parsed.dt.dayofweek.fillna(-1).astype(np.float32)
        else:
            date_df[f"{col}_year"] = -1.0
            date_df[f"{col}_month"] = -1.0
            date_df[f"{col}_dayofweek"] = -1.0

    if "report_date" in parsed_dates and "initiation_date" in parsed_dates:
        date_df["days_initiation_to_report"] = (
            (parsed_dates["report_date"] - parsed_dates["initiation_date"])
            .dt.days.fillna(-999)
            .astype(np.float32)
        )
    elif "report_date" in date_columns and "initiation_date" in date_columns:
        date_df["days_initiation_to_report"] = -999.0

    if "termination_date" in parsed_dates and "initiation_date" in parsed_dates:
        date_df["days_initiation_to_termination"] = (
            (parsed_dates["termination_date"] - parsed_dates["initiation_date"])
            .dt.days.fillna(-999)
            .astype(np.float32)
        )
    elif "termination_date" in date_columns and "initiation_date" in date_columns:
        date_df["days_initiation_to_termination"] = -999.0

    return date_df


train_dates = process_dates(df_train, date_cols)
test_dates = process_dates(df_test, date_cols)

# 6. Categorical Frequency & OOF Target Encoding
cat_freq_df_train = pd.DataFrame(index=df_train.index)
cat_freq_df_test = pd.DataFrame(index=df_test.index)

for col in cat_cols:
    freq_map = df_train[col].value_counts(normalize=True).to_dict()
    cat_freq_df_train[f"{col}_freq"] = (
        df_train[col].map(freq_map).fillna(0).astype(np.float32)
    )
    cat_freq_df_test[f"{col}_freq"] = (
        df_test[col].map(freq_map).fillna(0).astype(np.float32)
    )

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
target_enc_cols = [c for c in ["recalling_firm", "state", "city"] if c in cat_cols]

oof_target_enc_train = {
    f"{col}_te_class_{k}": np.zeros(len(df_train), dtype=np.float32)
    for col in target_enc_cols
    for k in range(num_classes)
}
te_test_dict = {
    f"{col}_te_class_{k}": np.zeros(len(df_test), dtype=np.float32)
    for col in target_enc_cols
    for k in range(num_classes)
}

global_priors = [np.mean(df_train["target"] == k) for k in range(num_classes)]
smoothing = 10.0

for fold, (tr_idx, val_idx) in enumerate(skf.split(df_train, df_train["target"])):
    tr_df = df_train.iloc[tr_idx]
    val_df = df_train.iloc[val_idx]

    for col in target_enc_cols:
        for k in range(num_classes):
            prior = global_priors[k]
            counts = tr_df[col].value_counts()
            pos_counts = tr_df[tr_df["target"] == k][col].value_counts()

            smooth_map = (
                (pos_counts.reindex(counts.index, fill_value=0) + smoothing * prior)
                / (counts + smoothing)
            ).to_dict()

            col_name = f"{col}_te_class_{k}"
            oof_target_enc_train[col_name][val_idx] = (
                val_df[col].map(smooth_map).fillna(prior).values
            )

for col in target_enc_cols:
    for k in range(num_classes):
        prior = global_priors[k]
        counts = df_train[col].value_counts()
        pos_counts = df_train[df_train["target"] == k][col].value_counts()
        smooth_map = (
            (pos_counts.reindex(counts.index, fill_value=0) + smoothing * prior)
            / (counts + smoothing)
        ).to_dict()
        col_name = f"{col}_te_class_{k}"
        te_test_dict[col_name] = (
            df_test[col].map(smooth_map).fillna(prior).values.astype(np.float32)
        )

oof_te_train_df = pd.DataFrame(oof_target_enc_train, index=df_train.index)
oof_te_test_df = pd.DataFrame(te_test_dict, index=df_test.index)

# Assemble Complete Feature Sets
X_train_raw = pd.concat(
    [
        train_hazards,
        train_text_stats,
        train_svd_df,
        train_dates,
        cat_freq_df_train,
        oof_te_train_df,
    ],
    axis=1,
).astype(np.float32)

X_test_raw = pd.concat(
    [
        test_hazards,
        test_text_stats,
        test_svd_df,
        test_dates,
        cat_freq_df_test,
        oof_te_test_df,
    ],
    axis=1,
).astype(np.float32)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled = scaler.transform(X_test_raw)

df_train["fold"] = -1
for fold, (_, val_idx) in enumerate(skf.split(df_train, df_train["target"])):
    df_train.loc[val_idx, "fold"] = fold

y_train = df_train["target"].values
folds = df_train["fold"].values
input_dim = X_train_scaled.shape[1]
num_folds = 5


# Model Definitions
class GatedFeatureAttention(nn.Module):
    def __init__(self, in_features: int, reduction: int = 4):
        super().__init__()
        reduced_dim = max(8, in_features // reduction)
        self.fc1 = nn.Linear(in_features, reduced_dim)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(reduced_dim, in_features)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.sigmoid(self.fc2(self.act(self.fc1(x))))
        return x * weights


class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout_rate: float = 0.2):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.linear1 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.norm2(
            self.linear2(self.dropout(self.act(self.linear1(self.norm1(x)))))
        )
        return residual + out


class TabularGatedResNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 256,
        num_blocks: int = 3,
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        self.feature_attention = GatedFeatureAttention(input_dim)
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU()
        )
        self.blocks = nn.ModuleList(
            [
                ResidualBlock(hidden_dim, dropout_rate=dropout_rate)
                for _ in range(num_blocks)
            ]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout_rate / 2.0),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_attention(x)
        x = self.input_layer(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x)


def get_loss_criterion(label_smoothing: float = 0.05) -> nn.Module:
    return nn.CrossEntropyLoss(label_smoothing=label_smoothing)


def build_pytorch_model_and_optimizer(
    input_dim: int,
    num_classes: int,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
):
    model = TabularGatedResNet(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dim=256,
        num_blocks=3,
        dropout_rate=0.2,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=50, eta_min=1e-5
    )
    return model, optimizer, scheduler


def get_lgb_params(num_classes: int, random_state: int = 42) -> dict:
    return {
        "objective": "multiclass",
        "num_class": num_classes,
        "metric": "multi_logloss",
        "boosting_type": "gbdt",
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": 6,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l1": 0.5,
        "lambda_l2": 1.0,
        "min_child_samples": 20,
        "random_state": random_state,
        "n_jobs": -1,
        "verbose": -1,
    }


def get_xgb_params(num_classes: int, random_state: int = 42) -> dict:
    return {
        "objective": "multi:softprob",
        "num_class": num_classes,
        "eval_metric": "mlogloss",
        "learning_rate": 0.03,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.5,
        "reg_lambda": 1.0,
        "random_state": random_state,
        "n_jobs": -1,
    }


def get_catboost_params(num_classes: int, random_state: int = 42) -> dict:
    return {
        "loss_function": "MultiClass",
        "eval_metric": "MultiClass",
        "learning_rate": 0.04,
        "depth": 6,
        "l2_leaf_reg": 3.0,
        "random_seed": random_state,
        "verbose": False,
        "thread_count": -1,
    }


# Out-Of-Fold and Test Prediction Setup for Stacking
oof_lgb = np.zeros((len(y_train), num_classes), dtype=np.float32)
oof_xgb = np.zeros((len(y_train), num_classes), dtype=np.float32)
oof_cb = np.zeros((len(y_train), num_classes), dtype=np.float32)
oof_nn = np.zeros((len(y_train), num_classes), dtype=np.float32)

test_p_lgb_total = np.zeros((len(X_test_raw), num_classes), dtype=np.float32)
test_p_xgb_total = np.zeros((len(X_test_raw), num_classes), dtype=np.float32)
test_p_cb_total = np.zeros((len(X_test_raw), num_classes), dtype=np.float32)
test_p_nn_total = np.zeros((len(X_test_raw), num_classes), dtype=np.float32)

lgb_base_params = get_lgb_params(num_classes=num_classes, random_state=42)
xgb_base_params = get_xgb_params(num_classes=num_classes, random_state=42)
cb_base_params = get_catboost_params(num_classes=num_classes, random_state=42)

# Cross-Validation Loop
for fold in range(num_folds):
    tr_idx = np.where(folds != fold)[0]
    val_idx = np.where(folds == fold)[0]

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train_raw.iloc[tr_idx])
    X_va = scaler.transform(X_train_raw.iloc[val_idx])
    X_test_scaled = scaler.transform(X_test_raw)
    y_tr, y_va = y_train[tr_idx], y_train[val_idx]

    # LightGBM
    lgb_tr = lgb.Dataset(X_tr, label=y_tr)
    lgb_va = lgb.Dataset(X_va, label=y_va, reference=lgb_tr)
    model_lgb = lgb.train(
        lgb_base_params,
        lgb_tr,
        num_boost_round=1000,
        valid_sets=[lgb_va],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )
    val_p_lgb = model_lgb.predict(X_va)
    test_p_lgb = model_lgb.predict(X_test_scaled)

    oof_lgb[val_idx] = val_p_lgb
    test_p_lgb_total += test_p_lgb / num_folds

    # XGBoost
    model_xgb = XGBClassifier(
        **xgb_base_params, n_estimators=1000, early_stopping_rounds=50
    )
    model_xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    val_p_xgb = model_xgb.predict_proba(X_va)
    test_p_xgb = model_xgb.predict_proba(X_test_scaled)

    oof_xgb[val_idx] = val_p_xgb
    test_p_xgb_total += test_p_xgb / num_folds

    # CatBoost
    model_cb = cb.CatBoostClassifier(
        **cb_base_params, iterations=1000, early_stopping_rounds=50
    )
    model_cb.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=False)
    val_p_cb = model_cb.predict_proba(X_va)
    test_p_cb = model_cb.predict_proba(X_test_scaled)

    oof_cb[val_idx] = val_p_cb
    test_p_cb_total += test_p_cb / num_folds

    # PyTorch Neural Net
    nn_model, nn_opt, nn_sched = build_pytorch_model_and_optimizer(
        input_dim, num_classes, lr=1e-3, weight_decay=1e-4
    )
    nn_model = nn_model.to(device)
    criterion = get_loss_criterion(label_smoothing=0.05)

    tr_dataset = TensorDataset(
        torch.tensor(X_tr, dtype=torch.float32),
        torch.tensor(y_tr, dtype=torch.long),
    )
    va_dataset = TensorDataset(
        torch.tensor(X_va, dtype=torch.float32),
        torch.tensor(y_va, dtype=torch.long),
    )

    tr_loader = DataLoader(tr_dataset, batch_size=128, shuffle=True, drop_last=False)
    va_loader = DataLoader(va_dataset, batch_size=256, shuffle=False)

    best_val_loss = float("inf")
    best_nn_weights = None

    for epoch in range(35):
        nn_model.train()
        for bx, by in tr_loader:
            bx, by = bx.to(device), by.to(device)
            nn_opt.zero_grad()
            logits = nn_model(bx)
            loss = criterion(logits, by)
            loss.backward()
            nn_opt.step()
        nn_sched.step()

        nn_model.eval()
        val_loss = 0.0
        val_count = 0
        with torch.no_grad():
            for bx, by in va_loader:
                bx, by = bx.to(device), by.to(device)
                logits = nn_model(bx)
                loss = criterion(logits, by)
                val_loss += loss.item() * len(by)
                val_count += len(by)
        val_loss /= val_count

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_nn_weights = {
                k: v.cpu().clone() for k, v in nn_model.state_dict().items()
            }

    nn_model.load_state_dict({k: v.to(device) for k, v in best_nn_weights.items()})
    nn_model.eval()

    with torch.no_grad():
        va_tensor = torch.tensor(X_va, dtype=torch.float32).to(device)
        test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
        val_p_nn = (
            torch.softmax(nn_model(va_tensor), dim=1).cpu().numpy().astype(np.float32)
        )
        test_p_nn = (
            torch.softmax(nn_model(test_tensor), dim=1).cpu().numpy().astype(np.float32)
        )

    oof_nn[val_idx] = val_p_nn
    test_p_nn_total += test_p_nn / num_folds

# Stacking Meta-Learner
oof_meta = np.hstack([oof_lgb, oof_xgb, oof_cb, oof_nn])
test_meta = np.hstack(
    [test_p_lgb_total, test_p_xgb_total, test_p_cb_total, test_p_nn_total]
)

meta_model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
meta_model.fit(oof_meta, y_train)

oof_probs = meta_model.predict_proba(oof_meta)
test_probs = meta_model.predict_proba(test_meta)

oof_predictions = np.argmax(oof_probs, axis=1)
overall_accuracy = accuracy_score(y_train, oof_predictions)

# Submission Generation
test_predictions = np.argmax(test_probs, axis=1)
test_labels = [target_mapping[idx] for idx in test_predictions]

submission_df = pd.DataFrame(
    {"id": np.arange(len(test_labels)), "classification": test_labels}
)

submission_path = os.path.join(submission_dir, "submission.csv")
submission_df.to_csv(submission_path, index=False)

print(f"Final Validation Score: {overall_accuracy}")