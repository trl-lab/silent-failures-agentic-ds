import os
import re
from typing import Any, Dict, List, Tuple
from catboost import CatBoostRegressor
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb

# Ensure output directories exist
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# =========================================================================
# 1. LOAD DATA & DOMAIN FEATURE ENGINEERING
# =========================================================================
print("Loading datasets...")
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "salary_in_usd"
assert target_col in train_df.columns, f"Target '{target_col}' not found in train data."


TIER_1_COUNTRIES = {"US", "CH", "PR", "IL"}
TIER_2_COUNTRIES = {
    "CA", "GB", "DE", "NL", "AU", "NZ", "IE", "SG", "DK", "SE", "NO", "FR",
    "BE", "LU", "AT", "JP", "AE", "SA", "HK", "KR",
}
TIER_3_COUNTRIES = {
    "ES", "IT", "PT", "PL", "CZ", "RO", "EE", "LT", "LV", "HU", "GR", "MX",
    "BR", "AR", "CL", "CO", "HR", "SI", "SK", "BG", "CY", "MT", "RU", "UA",
    "TR", "ZA", "CR", "MY", "TH",
}


def get_country_tier(country_code: str) -> int:
    code = str(country_code).upper().strip()
    if code in TIER_1_COUNTRIES:
        return 1
    elif code in TIER_2_COUNTRIES:
        return 2
    elif code in TIER_3_COUNTRIES:
        return 3
    else:
        return 4


def extract_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract domain-specific hierarchical, geographical, and taxonomy features."""
    df = df.copy()

    # 1. Experience Level Ordinal Mapping
    exp_map = {"EN": 1, "MI": 2, "SE": 3, "EX": 4}
    if "experience_level" in df.columns:
        df["exp_level_num"] = df["experience_level"].map(exp_map).fillna(2).astype(int)
    else:
        df["exp_level_num"] = 2

    # 2. Company Size Ordinal Mapping
    size_map = {"S": 1, "M": 2, "L": 3}
    if "company_size" in df.columns:
        df["company_size_num"] = df["company_size"].map(size_map).fillna(2).astype(int)
    else:
        df["company_size_num"] = 2

    # 3. Remote Ratio Numerical & Categorical
    if "remote_ratio" in df.columns:
        df["remote_ratio_num"] = (
            pd.to_numeric(df["remote_ratio"], errors="coerce").fillna(50) / 100.0
        )
        df["is_fully_remote"] = (df["remote_ratio"] == 100).astype(int)
        df["is_onsite"] = (df["remote_ratio"] == 0).astype(int)
        df["is_hybrid"] = (
            (df["remote_ratio"] > 0) & (df["remote_ratio"] < 100)
        ).astype(int)
    else:
        df["remote_ratio_num"] = 0.5
        df["is_fully_remote"] = 0
        df["is_onsite"] = 0
        df["is_hybrid"] = 1

    # 4. Temporal Features
    if "work_year" in df.columns:
        df["work_year_num"] = (
            pd.to_numeric(df["work_year"], errors="coerce").fillna(2023).astype(int)
        )
        df["years_since_2020"] = df["work_year_num"] - 2020
    else:
        df["work_year_num"] = 2023
        df["years_since_2020"] = 3

    # 5. Job Title Taxonomy & NLP Keyword Extraction
    if "job_title" in df.columns:
        titles = df["job_title"].astype(str).str.lower()

        # Seniority / Leadership indicators
        df["title_is_lead_mgr"] = titles.apply(
            lambda x: int(
                any(
                    w in x
                    for w in [
                        "lead",
                        "manager",
                        "head",
                        "director",
                        "vp",
                        "chief",
                        "principal",
                        "officer",
                    ]
                )
            )
        ).astype(int)
        df["title_is_senior"] = titles.apply(
            lambda x: int(
                any(w in x for w in ["sr", "senior", "staff", "principal", "expert"])
            )
        ).astype(int)
        df["title_is_junior"] = titles.apply(
            lambda x: int(
                any(w in x for w in ["jr", "junior", "entry", "associate", "intern"])
            )
        ).astype(int)

        # Role Domains
        df["title_is_ml_engineer"] = titles.apply(
            lambda x: int(
                any(
                    w in x
                    for w in [
                        "machine learning",
                        "ml",
                        "deep learning",
                        "computer vision",
                        "nlp",
                        "ai",
                        "artificial intelligence",
                    ]
                )
            )
        ).astype(int)
        df["title_is_data_scientist"] = titles.apply(
            lambda x: int(
                "data scientist" in x or "data science" in x or "applied scientist" in x
            )
        ).astype(int)
        df["title_is_data_engineer"] = titles.apply(
            lambda x: int(
                any(
                    w in x
                    for w in [
                        "data engineer",
                        "big data",
                        "etl",
                        "pipeline",
                        "database",
                        "data infra",
                    ]
                )
            )
        ).astype(int)
        df["title_is_analyst"] = titles.apply(
            lambda x: int(
                any(
                    w in x
                    for w in [
                        "analyst",
                        "analytics",
                        "bi",
                        "business intelligence",
                        "visualization",
                    ]
                )
            )
        ).astype(int)
        df["title_is_researcher"] = titles.apply(
            lambda x: int(
                any(w in x for w in ["research", "scientist", "phd", "algorithm"])
            )
        ).astype(int)
        df["title_is_architect"] = titles.apply(lambda x: int("architect" in x)).astype(
            int
        )
        df["title_is_genai_llm"] = titles.apply(
            lambda x: int(
                any(
                    w in x
                    for w in [
                        "prompt",
                        "llm",
                        "generative",
                        "gen ai",
                        "large language",
                        "foundation",
                    ]
                )
            )
        ).astype(int)

        # Text Complexity Metrics
        df["title_word_count"] = titles.apply(lambda x: len(x.split())).astype(int)
        df["title_char_length"] = titles.apply(len).astype(int)
    else:
        for col in [
            "title_is_lead_mgr",
            "title_is_senior",
            "title_is_junior",
            "title_is_ml_engineer",
            "title_is_data_scientist",
            "title_is_data_engineer",
            "title_is_analyst",
            "title_is_researcher",
            "title_is_architect",
            "title_is_genai_llm",
            "title_word_count",
            "title_char_length",
        ]:
            df[col] = 0

    # 6. Geography & Mobility Relations with Macro Wage Tiers
    if "company_location" in df.columns:
        df["company_location_tier"] = df["company_location"].apply(get_country_tier).astype(int)
    else:
        df["company_location_tier"] = 4

    if "employee_residence" in df.columns:
        df["employee_residence_tier"] = df["employee_residence"].apply(get_country_tier).astype(int)
    else:
        df["employee_residence_tier"] = 4

    df["tier_diff"] = df["company_location_tier"] - df["employee_residence_tier"]
    df["tier_exp_combo"] = df["company_location_tier"].astype(str) + "_" + df["exp_level_num"].astype(str)

    if "employee_residence" in df.columns and "company_location" in df.columns:
        df["same_residence_location"] = (
            df["employee_residence"] == df["company_location"]
        ).astype(int)
        df["is_us_company"] = (df["company_location"] == "US").astype(int)
        df["is_us_residence"] = (df["employee_residence"] == "US").astype(int)
        df["is_cross_border_remote"] = (
            (df["same_residence_location"] == 0) & (df["remote_ratio_num"] > 0)
        ).astype(int)
    else:
        df["same_residence_location"] = 1
        df["is_us_company"] = 0
        df["is_us_residence"] = 0
        df["is_cross_border_remote"] = 0

    # 7. Interaction Terms
    df["exp_x_company_size"] = df["exp_level_num"] * df["company_size_num"]
    df["exp_x_remote"] = df["exp_level_num"] * df["remote_ratio_num"]
    df["exp_x_year"] = df["exp_level_num"] * df["years_since_2020"]
    df["is_us_x_senior"] = df["is_us_company"] * df["exp_level_num"]

    # 8. High-Level Categorical Combinations
    if "experience_level" in df.columns and "company_size" in df.columns:
        df["exp_size_combo"] = (
            df["experience_level"].astype(str) + "_" + df["company_size"].astype(str)
        )
    if "employee_residence" in df.columns and "company_location" in df.columns:
        df["residence_location_combo"] = (
            df["employee_residence"].astype(str)
            + "_"
            + df["company_location"].astype(str)
        )
    if "job_title" in df.columns and "experience_level" in df.columns:
        df["title_exp_combo"] = (
            df["job_title"].astype(str) + "_" + df["experience_level"].astype(str)
        )
    if "company_location" in df.columns and "experience_level" in df.columns:
        df["location_exp_combo"] = (
            df["company_location"].astype(str)
            + "_"
            + df["experience_level"].astype(str)
        )

    return df


print("Extracting domain features...")
train_fe = extract_domain_features(train_df)
test_fe = extract_domain_features(test_df)

# Relative Non-Target Deviation Features
role_exp_map = train_fe.groupby("job_title")["exp_level_num"].mean().to_dict()
global_exp_mean = float(train_fe["exp_level_num"].mean())
train_fe["exp_diff_role_mean"] = (
    train_fe["exp_level_num"] - train_fe["job_title"].map(role_exp_map).fillna(global_exp_mean)
).astype(np.float32)
test_fe["exp_diff_role_mean"] = (
    test_fe["exp_level_num"] - test_fe["job_title"].map(role_exp_map).fillna(global_exp_mean)
).astype(np.float32)

loc_rem_map = train_fe.groupby("company_location")["remote_ratio_num"].mean().to_dict()
global_rem_mean = float(train_fe["remote_ratio_num"].mean())
train_fe["remote_diff_loc_mean"] = (
    train_fe["remote_ratio_num"] - train_fe["company_location"].map(loc_rem_map).fillna(global_rem_mean)
).astype(np.float32)
test_fe["remote_diff_loc_mean"] = (
    test_fe["remote_ratio_num"] - test_fe["company_location"].map(loc_rem_map).fillna(global_rem_mean)
).astype(np.float32)

role_rem_map = train_fe.groupby("job_title")["remote_ratio_num"].mean().to_dict()
train_fe["remote_diff_role_mean"] = (
    train_fe["remote_ratio_num"] - train_fe["job_title"].map(role_rem_map).fillna(global_rem_mean)
).astype(np.float32)
test_fe["remote_diff_role_mean"] = (
    test_fe["remote_ratio_num"] - test_fe["job_title"].map(role_rem_map).fillna(global_rem_mean)
).astype(np.float32)

# Job Title Semantic Transformer Dense Embeddings + PCA Reduction
if "job_title" in train_fe.columns and "job_title" in test_fe.columns:
    all_unique_titles = list(set(train_fe["job_title"].dropna().unique()).union(set(test_fe["job_title"].dropna().unique())))
    title_to_emb = {}
    try:
        from sentence_transformers import SentenceTransformer
        sbert_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        title_embs = sbert_model.encode(all_unique_titles, show_progress_bar=False, convert_to_numpy=True)
        title_to_emb = {t: title_embs[i] for i, t in enumerate(all_unique_titles)}
    except Exception:
        try:
            from transformers import AutoModel, AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
            hf_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
            hf_model.eval()
            inputs = tokenizer(all_unique_titles, padding=True, truncation=True, return_tensors="pt")
            with torch.no_grad():
                out = hf_model(**inputs)
                mask = inputs["attention_mask"].unsqueeze(-1).expand(out.last_hidden_state.size()).float()
                sum_emb = torch.sum(out.last_hidden_state * mask, 1)
                sum_mask = torch.clamp(mask.sum(1), min=1e-9)
                title_embs = (sum_emb / sum_mask).cpu().numpy()
            title_to_emb = {t: title_embs[i] for i, t in enumerate(all_unique_titles)}
        except Exception:
            tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=1, token_pattern=r"(?u)\b\w+\b")
            title_embs = tfidf.fit_transform(all_unique_titles).toarray()
            title_to_emb = {t: title_embs[i] for i, t in enumerate(all_unique_titles)}

    emb_dim = len(next(iter(title_to_emb.values())))
    default_emb = np.zeros(emb_dim, dtype=np.float32)

    tr_raw_mat = np.array([title_to_emb.get(t, default_emb) for t in train_fe["job_title"].fillna("")])
    te_raw_mat = np.array([title_to_emb.get(t, default_emb) for t in test_fe["job_title"].fillna("")])

    n_components = min(16, tr_raw_mat.shape[1])
    pca = PCA(n_components=n_components, random_state=42)
    tr_pca = pca.fit_transform(tr_raw_mat)
    te_pca = pca.transform(te_raw_mat)

    for i in range(n_components):
        train_fe[f"title_dense_pca_{i}"] = tr_pca[:, i].astype(np.float32)
        test_fe[f"title_dense_pca_{i}"] = te_pca[:, i].astype(np.float32)

y = train_df[target_col].copy()
train_fe[target_col] = y

# =========================================================================
# 2. STRATIFIED CROSS-VALIDATION SPLITS & OUT-OF-FOLD ENCODINGS
# =========================================================================
N_SPLITS = 10
RANDOM_STATE = 42
train_fe["fold"] = -1

num_bins = min(10, max(2, len(train_fe) // 20))
binned_target = pd.qcut(y, q=num_bins, labels=False, duplicates="drop")

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
for fold, (train_idx, val_idx) in enumerate(skf.split(train_fe, binned_target)):
    train_fe.loc[val_idx, "fold"] = fold

# Frequency Encoding (Strictly computed out-of-fold)
cat_cols_for_freq = [
    c
    for c in [
        "job_title",
        "employee_residence",
        "company_location",
        "salary_currency",
        "experience_level",
        "employment_type",
        "company_size",
        "exp_size_combo",
        "residence_location_combo",
        "title_exp_combo",
        "location_exp_combo",
    ]
    if c in train_fe.columns
]

for col in cat_cols_for_freq:
    train_fe[f"{col}_freq"] = 0.0
    test_fe[f"{col}_freq"] = 0.0

    global_freq = train_fe[col].value_counts(normalize=True).to_dict()
    test_fe[f"{col}_freq"] = test_fe[col].map(global_freq).fillna(0.0).astype(float)

    for fold in range(N_SPLITS):
        tr_mask = train_fe["fold"] != fold
        val_mask = train_fe["fold"] == fold
        fold_freq = train_fe.loc[tr_mask, col].value_counts(normalize=True).to_dict()
        train_fe.loc[val_mask, f"{col}_freq"] = (
            train_fe.loc[val_mask, col].map(fold_freq).fillna(0.0).astype(float)
        )

# Target Encoding & Multi-Granularity Aggregation Statistics (Strictly out-of-fold)
target_enc_cols = [
    c
    for c in [
        "job_title",
        "employee_residence",
        "company_location",
        "experience_level",
        "employment_type",
        "company_size",
        "exp_size_combo",
        "title_exp_combo",
        "location_exp_combo",
        "residence_location_combo",
    ]
    if c in train_fe.columns
]

stat_group_cols = [
    c
    for c in [
        "job_title",
        "company_location",
        "experience_level",
        "employment_type",
        "title_exp_combo",
    ]
    if c in train_fe.columns
]

global_mean = float(y.mean())
global_std = float(y.std()) if float(y.std()) > 0 else 1.0
global_min = float(y.min())
global_max = float(y.max())
global_median = float(y.median())
smoothing = 10

for col in target_enc_cols:
    train_fe[f"{col}_target_enc"] = np.nan
    test_fe[f"{col}_target_enc"] = np.nan

    stats = train_fe.groupby(col)[target_col].agg(["count", "mean"])
    smoothed_enc = (stats["count"] * stats["mean"] + smoothing * global_mean) / (
        stats["count"] + smoothing
    )
    test_fe[f"{col}_target_enc"] = (
        test_fe[col].map(smoothed_enc).fillna(global_mean).astype(np.float32)
    )

    for fold in range(N_SPLITS):
        tr_mask = train_fe["fold"] != fold
        val_mask = train_fe["fold"] == fold

        fold_mean = float(train_fe.loc[tr_mask, target_col].mean())
        fold_stats = (
            train_fe.loc[tr_mask].groupby(col)[target_col].agg(["count", "mean"])
        )
        fold_smoothed = (
            fold_stats["count"] * fold_stats["mean"] + smoothing * fold_mean
        ) / (fold_stats["count"] + smoothing)

        train_fe.loc[val_mask, f"{col}_target_enc"] = (
            train_fe.loc[val_mask, col]
            .map(fold_smoothed)
            .fillna(fold_mean)
            .astype(np.float32)
        )

# Multi-granularity aggregation statistics (std, min, max, median) across key categorical groups
for col in stat_group_cols:
    for stat_name, global_val, agg_func in [
        ("std", global_std, lambda s: s.std(ddof=0)),
        ("min", global_min, "min"),
        ("max", global_max, "max"),
        ("median", global_median, "median"),
    ]:
        col_feat = f"{col}_target_{stat_name}"
        train_fe[col_feat] = np.nan
        test_fe[col_feat] = np.nan

        full_agg = train_fe.groupby(col)[target_col].agg(agg_func).to_dict()
        test_fe[col_feat] = (
            test_fe[col].map(full_agg).fillna(global_val).astype(np.float32)
        )

        for fold in range(N_SPLITS):
            tr_mask = train_fe["fold"] != fold
            val_mask = train_fe["fold"] == fold

            fold_agg = (
                train_fe.loc[tr_mask]
                .groupby(col)[target_col]
                .agg(agg_func)
                .to_dict()
            )
            train_fe.loc[val_mask, col_feat] = (
                train_fe.loc[val_mask, col]
                .map(fold_agg)
                .fillna(global_val)
                .astype(np.float32)
            )

# Relative Ratio Features Comparing Title Target Encodings Against Geographic & Experience Baselines
if "job_title_target_enc" in train_fe.columns and "company_location_target_enc" in train_fe.columns:
    train_fe["title_to_loc_te_ratio"] = (
        train_fe["job_title_target_enc"] / (train_fe["company_location_target_enc"] + 1e-5)
    ).astype(np.float32)
    test_fe["title_to_loc_te_ratio"] = (
        test_fe["job_title_target_enc"] / (test_fe["company_location_target_enc"] + 1e-5)
    ).astype(np.float32)

if "title_exp_combo_target_enc" in train_fe.columns and "company_location_target_enc" in train_fe.columns:
    train_fe["title_exp_to_loc_te_ratio"] = (
        train_fe["title_exp_combo_target_enc"] / (train_fe["company_location_target_enc"] + 1e-5)
    ).astype(np.float32)
    test_fe["title_exp_to_loc_te_ratio"] = (
        test_fe["title_exp_combo_target_enc"] / (test_fe["company_location_target_enc"] + 1e-5)
    ).astype(np.float32)

if "job_title_target_enc" in train_fe.columns and "experience_level_target_enc" in train_fe.columns:
    train_fe["title_to_exp_te_ratio"] = (
        train_fe["job_title_target_enc"] / (train_fe["experience_level_target_enc"] + 1e-5)
    ).astype(np.float32)
    test_fe["title_to_exp_te_ratio"] = (
        test_fe["job_title_target_enc"] / (test_fe["experience_level_target_enc"] + 1e-5)
    ).astype(np.float32)

# Label Encoding for Categorical columns
categorical_columns = train_fe.select_dtypes(
    include=["object", "category"]
).columns.tolist()
if target_col in categorical_columns:
    categorical_columns.remove(target_col)

for col in categorical_columns:
    le = LabelEncoder()
    train_fe[col] = train_fe[col].astype(str).fillna("Unknown")
    test_fe[col] = test_fe[col].astype(str).fillna("Unknown")

    all_vals = list(set(train_fe[col].unique()).union(set(test_fe[col].unique())))
    le.fit(all_vals)

    train_fe[f"{col}_cat"] = le.transform(train_fe[col])
    test_fe[f"{col}_cat"] = le.transform(test_fe[col])


# =========================================================================
# 3. PYTORCH TABULAR RESIDUAL NETWORK ARCHITECTURE
# =========================================================================
class SwiGLU(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear_gate = nn.Linear(in_features, out_features)
        self.linear_val = nn.Linear(in_features, out_features)
        self.out_proj = nn.Linear(out_features, in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_proj(F.silu(self.linear_gate(x)) * self.linear_val(x))


class TabularResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.15):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.swiglu1 = SwiGLU(hidden_dim, hidden_dim * 2)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.swiglu2 = SwiGLU(hidden_dim, hidden_dim * 2)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout1(self.swiglu1(self.norm1(x)))
        x = x + self.dropout2(self.swiglu2(self.norm2(x)))
        return x


class TabularResNet(nn.Module):
    def __init__(
        self,
        num_continuous_features: int,
        cat_cardinalities: List[int],
        embedding_dims: List[int] = None,
        hidden_dim: int = 128,
        num_blocks: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.embeddings = nn.ModuleList()
        total_emb_dim = 0
        if cat_cardinalities:
            if embedding_dims is None:
                embedding_dims = [
                    min(50, max(4, (card + 1) // 2)) for card in cat_cardinalities
                ]

            for card, emb_dim in zip(cat_cardinalities, embedding_dims):
                self.embeddings.append(nn.Embedding(card, emb_dim))
                total_emb_dim += emb_dim

        self.num_continuous = num_continuous_features
        if self.num_continuous > 0:
            self.cont_norm = nn.BatchNorm1d(num_continuous_features)
        else:
            self.cont_norm = nn.Identity()

        in_dim = total_emb_dim + num_continuous_features
        self.input_layer = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU()
        )

        self.res_blocks = nn.ModuleList(
            [
                TabularResidualBlock(hidden_dim, dropout=dropout)
                for _ in range(num_blocks)
            ]
        )

        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout / 2.0),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x_cont: torch.Tensor, x_cat: torch.Tensor = None) -> torch.Tensor:
        inputs = []
        if len(self.embeddings) > 0 and x_cat is not None:
            emb_outs = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
            inputs.append(torch.cat(emb_outs, dim=1))

        if self.num_continuous > 0:
            normed_cont = self.cont_norm(x_cont)
            inputs.append(normed_cont)

        x = torch.cat(inputs, dim=1) if len(inputs) > 1 else inputs[0]
        x = self.input_layer(x)

        for block in self.res_blocks:
            x = block(x)

        out = self.head(x)
        return out.squeeze(-1)


# =========================================================================
# 4. CROSS-VALIDATION TRAINING & MULTI-MODEL ENSEMBLE
# =========================================================================
cat_feature_cols = [c for c in train_fe.columns if c.endswith("_cat")]
exclude_cols = {
    target_col,
    "fold",
    "id",
    "salary",
    *train_fe.select_dtypes(include=["object"]).columns,
}
cont_feature_cols = [
    c for c in train_fe.columns if c not in exclude_cols and not c.endswith("_cat")
]
all_model_features = cont_feature_cols + cat_feature_cols

n_train = len(train_fe)
n_test = len(test_fe)

model_keys = [
    "lgb",
    "lgb_shallow",
    "xgb",
    "cat_deep",
    "cat_shallow",
    "rf",
    "extra_trees",
    "ridge",
    "nn",
]
oof_preds = {k: np.zeros(n_train) for k in model_keys}
test_preds = {k: np.zeros(n_test) for k in model_keys}

cat_cardinalities = []
for c in cat_feature_cols:
    max_val = max(train_fe[c].max(), test_fe[c].max())
    cat_cardinalities.append(int(max_val) + 1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Beginning {N_SPLITS}-fold training with {len(all_model_features)} features...")

for fold in range(N_SPLITS):
    tr_idx = train_fe["fold"] != fold
    val_idx = train_fe["fold"] == fold

    X_tr = train_fe.loc[tr_idx, all_model_features].values.astype(np.float32)
    y_tr = train_fe.loc[tr_idx, target_col].values.astype(np.float32)
    X_val = train_fe.loc[val_idx, all_model_features].values.astype(np.float32)
    y_val = train_fe.loc[val_idx, target_col].values.astype(np.float32)
    X_test_all = test_fe[all_model_features].values.astype(np.float32)

    # --- 1. LightGBM (Multi-seed standard) ---
    lgb_val_fold = np.zeros(len(val_idx[val_idx]))
    lgb_test_fold = np.zeros(n_test)
    for seed_offset in [0, 100]:
        lgb_model = lgb.LGBMRegressor(
            objective="regression",
            metric="rmse",
            n_estimators=3000,
            learning_rate=0.03,
            num_leaves=31,
            max_depth=6,
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42 + fold + seed_offset,
            n_jobs=-1,
            verbose=-1,
        )
        lgb_model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
        )
        lgb_val_fold += lgb_model.predict(X_val) / 2.0
        lgb_test_fold += lgb_model.predict(X_test_all) / 2.0
    oof_preds["lgb"][val_idx] = lgb_val_fold
    test_preds["lgb"] += lgb_test_fold / N_SPLITS

    # --- 2. LightGBM Shallow Architecture ---
    lgb_shallow = lgb.LGBMRegressor(
        objective="regression",
        metric="rmse",
        n_estimators=3000,
        learning_rate=0.03,
        num_leaves=15,
        max_depth=4,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.7,
        reg_alpha=0.5,
        reg_lambda=2.0,
        random_state=42 + fold,
        n_jobs=-1,
        verbose=-1,
    )
    lgb_shallow.fit(
        X_tr,
        y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )
    oof_preds["lgb_shallow"][val_idx] = lgb_shallow.predict(X_val)
    test_preds["lgb_shallow"] += lgb_shallow.predict(X_test_all) / N_SPLITS

    # --- 3. XGBoost (Multi-seed) ---
    xgb_val_fold = np.zeros(len(val_idx[val_idx]))
    xgb_test_fold = np.zeros(n_test)
    for seed_offset in [0, 100]:
        xgb_model = xgb.XGBRegressor(
            objective="reg:squarederror",
            eval_metric="rmse",
            n_estimators=3000,
            learning_rate=0.03,
            max_depth=5,
            min_child_weight=3,
            subsample=0.85,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=2.0,
            early_stopping_rounds=50,
            random_state=42 + fold + seed_offset,
            n_jobs=-1,
            tree_method="hist",
        )
        xgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        xgb_val_fold += xgb_model.predict(X_val) / 2.0
        xgb_test_fold += xgb_model.predict(X_test_all) / 2.0
    oof_preds["xgb"][val_idx] = xgb_val_fold
    test_preds["xgb"] += xgb_test_fold / N_SPLITS

    # --- 4. CatBoost Deep & Shallow Variants ---
    X_tr_df = train_fe.loc[tr_idx, all_model_features].copy()
    X_val_df = train_fe.loc[val_idx, all_model_features].copy()
    X_test_df = test_fe[all_model_features].copy()

    for col in cat_feature_cols:
        X_tr_df[col] = X_tr_df[col].astype(int)
        X_val_df[col] = X_val_df[col].astype(int)
        X_test_df[col] = X_test_df[col].astype(int)

    cat_deep = CatBoostRegressor(
        loss_function="RMSE",
        eval_metric="RMSE",
        iterations=2500,
        learning_rate=0.03,
        depth=7,
        l2_leaf_reg=5.0,
        early_stopping_rounds=50,
        random_seed=42 + fold,
        cat_features=cat_feature_cols,
        verbose=False,
        thread_count=-1,
    )
    cat_deep.fit(X_tr_df, y_tr, eval_set=(X_val_df, y_val), verbose=False)
    oof_preds["cat_deep"][val_idx] = cat_deep.predict(X_val_df)
    test_preds["cat_deep"] += cat_deep.predict(X_test_df) / N_SPLITS

    cat_shallow = CatBoostRegressor(
        loss_function="RMSE",
        eval_metric="RMSE",
        iterations=2500,
        learning_rate=0.03,
        depth=4,
        l2_leaf_reg=2.0,
        early_stopping_rounds=50,
        random_seed=142 + fold,
        cat_features=cat_feature_cols,
        verbose=False,
        thread_count=-1,
    )
    cat_shallow.fit(X_tr_df, y_tr, eval_set=(X_val_df, y_val), verbose=False)
    oof_preds["cat_shallow"][val_idx] = cat_shallow.predict(X_val_df)
    test_preds["cat_shallow"] += cat_shallow.predict(X_test_df) / N_SPLITS

    # --- 5. Random Forest Regressor ---
    rf_model = RandomForestRegressor(
        n_estimators=150,
        max_depth=12,
        min_samples_split=4,
        random_state=42 + fold,
        n_jobs=-1,
    )
    rf_model.fit(np.nan_to_num(X_tr, nan=0.0), y_tr)
    oof_preds["rf"][val_idx] = rf_model.predict(np.nan_to_num(X_val, nan=0.0))
    test_preds["rf"] += rf_model.predict(np.nan_to_num(X_test_all, nan=0.0)) / N_SPLITS

    # --- 6. Extra Trees Regressor ---
    et_model = ExtraTreesRegressor(
        n_estimators=150,
        max_depth=12,
        min_samples_split=4,
        random_state=42 + fold,
        n_jobs=-1,
    )
    et_model.fit(np.nan_to_num(X_tr, nan=0.0), y_tr)
    oof_preds["extra_trees"][val_idx] = et_model.predict(np.nan_to_num(X_val, nan=0.0))
    test_preds["extra_trees"] += et_model.predict(np.nan_to_num(X_test_all, nan=0.0)) / N_SPLITS

    # --- 7. Regularized Linear Ridge ---
    ridge_scaler = StandardScaler()
    X_ridge_tr = ridge_scaler.fit_transform(np.nan_to_num(X_tr, nan=0.0))
    X_ridge_val = ridge_scaler.transform(np.nan_to_num(X_val, nan=0.0))
    X_ridge_te = ridge_scaler.transform(np.nan_to_num(X_test_all, nan=0.0))

    ridge_model = Ridge(alpha=10.0, random_state=42 + fold)
    ridge_model.fit(X_ridge_tr, y_tr)
    oof_preds["ridge"][val_idx] = ridge_model.predict(X_ridge_val)
    test_preds["ridge"] += ridge_model.predict(X_ridge_te) / N_SPLITS

    # --- 5. PyTorch TabularResNet with SwiGLU ---
    scaler = StandardScaler()
    X_cont_tr = scaler.fit_transform(
        train_fe.loc[tr_idx, cont_feature_cols].values
    ).astype(np.float32)
    X_cont_val = scaler.transform(
        train_fe.loc[val_idx, cont_feature_cols].values
    ).astype(np.float32)
    X_cont_te = scaler.transform(test_fe[cont_feature_cols].values).astype(np.float32)

    X_cat_tr = train_fe.loc[tr_idx, cat_feature_cols].values.astype(np.int64)
    X_cat_val = train_fe.loc[val_idx, cat_feature_cols].values.astype(np.int64)
    X_cat_te = test_fe[cat_feature_cols].values.astype(np.int64)

    y_mean, y_std = float(y_tr.mean()), float(y_tr.std())
    y_tr_norm = ((y_tr - y_mean) / y_std).astype(np.float32)
    y_val_norm = ((y_val - y_mean) / y_std).astype(np.float32)

    train_ds = TensorDataset(
        torch.tensor(X_cont_tr),
        torch.tensor(X_cat_tr),
        torch.tensor(y_tr_norm).unsqueeze(1),
    )
    val_ds = TensorDataset(
        torch.tensor(X_cont_val),
        torch.tensor(X_cat_val),
        torch.tensor(y_val_norm).unsqueeze(1),
    )
    test_ds = TensorDataset(torch.tensor(X_cont_te), torch.tensor(X_cat_te))

    batch_size = 128
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=False
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size * 2, shuffle=False)

    nn_model = TabularResNet(
        num_continuous_features=len(cont_feature_cols),
        cat_cardinalities=cat_cardinalities,
        hidden_dim=128,
        num_blocks=2,
        dropout=0.15,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(nn_model.parameters(), lr=2e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=40, eta_min=1e-5
    )

    best_val_loss = float("inf")
    best_nn_weights = None
    patience, patience_counter = 12, 0

    for epoch in range(50):
        nn_model.train()
        for b_cont, b_cat, b_y in train_loader:
            b_cont, b_cat, b_y = (
                b_cont.to(device),
                b_cat.to(device),
                b_y.to(device),
            )
            optimizer.zero_grad()
            out = nn_model(b_cont, b_cat).unsqueeze(1)
            loss = criterion(out, b_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(nn_model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        nn_model.eval()
        val_losses = []
        with torch.no_grad():
            for b_cont, b_cat, b_y in val_loader:
                b_cont, b_cat, b_y = (
                    b_cont.to(device),
                    b_cat.to(device),
                    b_y.to(device),
                )
                pred = nn_model(b_cont, b_cat).unsqueeze(1)
                val_losses.append(criterion(pred, b_y).item())
        avg_val_loss = float(np.mean(val_losses))

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_nn_weights = {k: v.cpu() for k, v in nn_model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    nn_model.load_state_dict({k: v.to(device) for k, v in best_nn_weights.items()})
    nn_model.eval()

    val_preds_nn = []
    with torch.no_grad():
        for b_cont, b_cat, _ in val_loader:
            pred = (
                nn_model(b_cont.to(device), b_cat.to(device)).cpu().numpy() * y_std
                + y_mean
            )
            val_preds_nn.extend(pred)
    oof_preds["nn"][val_idx] = np.array(val_preds_nn)

    test_preds_nn = []
    with torch.no_grad():
        for b_cont, b_cat in test_loader:
            pred = (
                nn_model(b_cont.to(device), b_cat.to(device)).cpu().numpy() * y_std
                + y_mean
            )
            test_preds_nn.extend(pred)
    test_preds["nn"] += np.array(test_preds_nn) / N_SPLITS

    y_val_true = train_fe.loc[val_idx, target_col].values
    rmse_lgb = root_mean_squared_error(y_val_true, oof_preds["lgb"][val_idx])
    rmse_xgb = root_mean_squared_error(y_val_true, oof_preds["xgb"][val_idx])
    rmse_cat_deep = root_mean_squared_error(y_val_true, oof_preds["cat_deep"][val_idx])
    rmse_nn = root_mean_squared_error(y_val_true, oof_preds["nn"][val_idx])
    rmse_ridge = root_mean_squared_error(y_val_true, oof_preds["ridge"][val_idx])
    print(
        f"Fold {fold+1}/{N_SPLITS} RMSE -> LGB: {rmse_lgb:.2f} | XGB: {rmse_xgb:.2f} | CAT_DEEP: {rmse_cat_deep:.2f} | NN: {rmse_nn:.2f} | Ridge: {rmse_ridge:.2f}"
    )

# =========================================================================
# 5. ENSEMBLE OPTIMIZATION & SUBMISSION GENERATION
# =========================================================================
y_true_all = train_fe[target_col].values
model_keys = [
    "lgb",
    "lgb_shallow",
    "xgb",
    "cat_deep",
    "cat_shallow",
    "rf",
    "extra_trees",
    "ridge",
    "nn",
]
oof_matrix = np.column_stack([oof_preds[k] for k in model_keys])
test_matrix = np.column_stack([test_preds[k] for k in model_keys])


def regularized_ensemble_loss(weights: np.ndarray, alpha_l2: float = 1e-4) -> float:
    blend = oof_matrix @ weights
    rmse = root_mean_squared_error(y_true_all, blend)
    l2_penalty = alpha_l2 * np.sum(weights ** 2)
    return float(rmse + l2_penalty)


init_weights = np.ones(len(model_keys)) / len(model_keys)
bounds = [(0.0, 1.0) for _ in range(len(model_keys))]
constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

opt_res = minimize(
    regularized_ensemble_loss,
    init_weights,
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
)
best_weights = np.maximum(0.0, opt_res.x)
best_weights = best_weights / np.sum(best_weights)

# Calculate Final Out-of-Fold Validation RMSE
final_oof_blend = oof_matrix @ best_weights
final_validation_score = float(root_mean_squared_error(y_true_all, final_oof_blend))

# Compute Test Predictions and Save Submission
final_test_predictions = test_matrix @ best_weights
y_min, y_max = float(y.min()), float(y.max())
final_test_predictions = np.clip(final_test_predictions, a_min=y_min, a_max=y_max)

sub_df = pd.DataFrame(
    {"id": np.arange(len(final_test_predictions)), target_col: final_test_predictions}
)
submission_path = "./submission/submission.csv"
sub_df.to_csv(submission_path, index=False)

print(f"Final Validation Score: {final_validation_score}")