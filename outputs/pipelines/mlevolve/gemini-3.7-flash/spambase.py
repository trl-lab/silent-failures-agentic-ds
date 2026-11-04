import os
import random
from typing import Any, Dict, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from scipy.stats import rankdata
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import SVC
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import lightgbm as lgb
from xgboost import XGBClassifier

try:
    from catboost import CatBoostClassifier

    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False


# =========================================================================
# 0. Reproducibility & Environment Setup
# =========================================================================
def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


seed_everything(42)

INPUT_DIR = "./input"
SUBMISSION_DIR = "./submission"
WORKING_DIR = "./working"
os.makedirs(SUBMISSION_DIR, exist_ok=True)
os.makedirs(WORKING_DIR, exist_ok=True)

# =========================================================================
# 1. Load Data & Feature Engineering
# =========================================================================
train_df = pd.read_parquet(os.path.join(INPUT_DIR, "train.parquet"))
test_df = pd.read_parquet(os.path.join(INPUT_DIR, "test.parquet"))

target_col = "class"
train_df[target_col] = train_df[target_col].astype(int)
feature_cols_raw = [c for c in train_df.columns if c != target_col]

word_cols = [c for c in feature_cols_raw if c.startswith("word_freq_")]
char_cols = [c for c in feature_cols_raw if c.startswith("char_freq_")]
run_cols = [c for c in feature_cols_raw if c.startswith("capital_run_length_")]

spam_keyword_sub = [
    c
    for c in word_cols
    if any(
        k in c
        for k in [
            "free",
            "business",
            "money",
            "credit",
            "order",
            "remove",
            "your",
            "000",
            "font",
            "receive",
            "email",
            "all",
            "our",
            "over",
            "internet",
            "mail",
            "people",
            "make",
            "address",
        ]
    )
]

ham_keyword_sub = [
    c
    for c in word_cols
    if any(
        k in c
        for k in [
            "george",
            "650",
            "hp",
            "hpl",
            "lab",
            "data",
            "telnet",
            "857",
            "415",
            "85",
            "technology",
            "1999",
            "parts",
            "pm",
            "cs",
            "meeting",
            "original",
            "project",
            "re",
            "edu",
            "table",
            "conference",
        ]
    )
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer domain, aggregation, and non-linear interaction features."""
    new_cols = {}

    # 1. Binary presence indicators
    for col in word_cols:
        new_cols[f"{col}_present"] = (df[col] > 0).astype(np.float32)
    for col in char_cols:
        new_cols[f"{col}_present"] = (df[col] > 0).astype(np.float32)

    # 2. Log1p transformations for skewed continuous features
    for col in word_cols + char_cols + run_cols:
        new_cols[f"{col}_log1p"] = np.log1p(df[col].clip(lower=0)).astype(np.float32)

    # 3. Row-level statistical aggregations across word frequencies
    new_cols["word_freq_sum"] = df[word_cols].sum(axis=1)
    new_cols["word_freq_mean"] = df[word_cols].mean(axis=1)
    new_cols["word_freq_std"] = df[word_cols].std(axis=1)
    new_cols["word_freq_max"] = df[word_cols].max(axis=1)
    new_cols["word_freq_count_nonzero"] = (df[word_cols] > 0).sum(axis=1)
    new_cols["word_freq_sparsity"] = (df[word_cols] == 0).mean(axis=1)

    # 4. Row-level statistical aggregations across char frequencies
    new_cols["char_freq_sum"] = df[char_cols].sum(axis=1)
    new_cols["char_freq_count_nonzero"] = (df[char_cols] > 0).sum(axis=1)

    # 5. Semantic Spam vs Ham aggregate signals
    if spam_keyword_sub:
        new_cols["spam_keywords_sum"] = df[spam_keyword_sub].sum(axis=1)
        new_cols["spam_keywords_count"] = (df[spam_keyword_sub] > 0).sum(axis=1)
        new_cols["spam_keywords_max"] = df[spam_keyword_sub].max(axis=1)
    else:
        new_cols["spam_keywords_sum"] = 0.0
        new_cols["spam_keywords_count"] = 0.0
        new_cols["spam_keywords_max"] = 0.0

    if ham_keyword_sub:
        new_cols["ham_keywords_sum"] = df[ham_keyword_sub].sum(axis=1)
        new_cols["ham_keywords_count"] = (df[ham_keyword_sub] > 0).sum(axis=1)
        new_cols["ham_keywords_max"] = df[ham_keyword_sub].max(axis=1)
    else:
        new_cols["ham_keywords_sum"] = 0.0
        new_cols["ham_keywords_count"] = 0.0
        new_cols["ham_keywords_max"] = 0.0

    # Ratios and differences between spam and ham signals
    new_cols["spam_to_ham_sum_ratio"] = (new_cols["spam_keywords_sum"] + 0.001) / (
        new_cols["ham_keywords_sum"] + 0.001
    )
    new_cols["spam_to_ham_count_diff"] = (
        new_cols["spam_keywords_count"] - new_cols["ham_keywords_count"]
    )
    new_cols["spam_to_total_word_ratio"] = new_cols["spam_keywords_sum"] / (
        new_cols["word_freq_sum"] + 0.001
    )
    new_cols["ham_to_total_word_ratio"] = new_cols["ham_keywords_sum"] / (
        new_cols["word_freq_sum"] + 0.001
    )

    # Log-transformed ratios of spam-to-ham signals
    new_cols["log_spam_to_ham_ratio"] = np.log1p(
        new_cols["spam_keywords_sum"]
    ) - np.log1p(new_cols["ham_keywords_sum"])
    new_cols["log_spam_to_ham_count_ratio"] = np.log1p(
        new_cols["spam_keywords_count"]
    ) - np.log1p(new_cols["ham_keywords_count"])

    # 6. Specific character ratio metrics
    excl_col = [
        c for c in char_cols if "!" in c or "char_freq_5" in c or "exclamation" in c
    ]
    dollar_col = [
        c for c in char_cols if "$" in c or "char_freq_6" in c or "dollar" in c
    ]

    if excl_col:
        new_cols["char_excl_val"] = df[excl_col[0]]
        new_cols["char_excl_to_char_sum"] = df[excl_col[0]] / (
            new_cols["char_freq_sum"] + 1e-5
        )
    if dollar_col:
        new_cols["char_dollar_val"] = df[dollar_col[0]]
        new_cols["char_dollar_to_char_sum"] = df[dollar_col[0]] / (
            new_cols["char_freq_sum"] + 1e-5
        )
    if excl_col and dollar_col:
        new_cols["char_spam_punct_sum"] = df[excl_col[0]] + df[dollar_col[0]]

    # 7. Capital run length dynamic features and non-linear interactions
    if (
        "capital_run_length_total" in df.columns
        and "capital_run_length_average" in df.columns
    ):
        new_cols["capital_run_estimated_count"] = df["capital_run_length_total"] / (
            df["capital_run_length_average"] + 1e-5
        )
        new_cols["capital_total_pow05"] = np.power(
            df["capital_run_length_total"].clip(lower=0), 0.5
        )
        new_cols["capital_avg_pow05"] = np.power(
            df["capital_run_length_average"].clip(lower=0), 0.5
        )
        new_cols["capital_log_total_x_avg"] = np.log1p(
            df["capital_run_length_total"].clip(lower=0)
        ) * np.log1p(df["capital_run_length_average"].clip(lower=0))

    if (
        "capital_run_length_longest" in df.columns
        and "capital_run_length_total" in df.columns
    ):
        new_cols["capital_longest_pow05"] = np.power(
            df["capital_run_length_longest"].clip(lower=0), 0.5
        )
        if "capital_run_length_average" in df.columns:
            new_cols["capital_log_longest_to_avg"] = np.log1p(
                df["capital_run_length_longest"].clip(lower=0)
            ) / (np.log1p(df["capital_run_length_average"].clip(lower=0)) + 1e-4)
        new_cols["capital_log_longest_x_total"] = np.log1p(
            df["capital_run_length_longest"].clip(lower=0)
        ) * np.log1p(df["capital_run_length_total"].clip(lower=0))
        new_cols["capital_longest_to_total_ratio"] = df[
            "capital_run_length_longest"
        ] / (df["capital_run_length_total"] + 1.0)
        new_cols["capital_longest_to_avg_ratio"] = df["capital_run_length_longest"] / (
            df["capital_run_length_average"] + 1e-5
        )
        new_cols["capital_longest_minus_avg"] = (
            df["capital_run_length_longest"] - df["capital_run_length_average"]
        )

    # 8. Punctuation multiplied by capital run lengths
    if excl_col and "capital_run_length_total" in df.columns:
        new_cols["char_excl_x_cap_total"] = df[excl_col[0]] * np.log1p(
            df["capital_run_length_total"]
        )
    if dollar_col and "capital_run_length_longest" in df.columns:
        new_cols["char_dollar_x_cap_longest"] = df[dollar_col[0]] * np.log1p(
            df["capital_run_length_longest"]
        )
    if dollar_col and "capital_run_length_total" in df.columns:
        new_cols["char_dollar_x_cap_total"] = df[dollar_col[0]] * np.log1p(
            df["capital_run_length_total"]
        )

    new_feats_df = pd.DataFrame(new_cols, index=df.index)
    return pd.concat([df, new_feats_df], axis=1)


train_engineered = engineer_features(train_df)
test_engineered = engineer_features(test_df)

X_train_raw = train_engineered.drop(columns=[target_col])
y_train = train_engineered[target_col].values
X_test_raw = test_engineered.copy()

common_cols = [c for c in X_train_raw.columns if c in X_test_raw.columns]
X_train_raw = X_train_raw[common_cols]
X_test_raw = X_test_raw[common_cols]

# Stratified 10-Fold CV Setup
N_SPLITS = 10
RANDOM_STATE = 42
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

folds_series = np.zeros(len(train_engineered), dtype=int)
for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_raw, y_train)):
    folds_series[val_idx] = fold

train_engineered["fold"] = folds_series

# Leak-free fold-wise Naive Bayes log-odds, kNN manifold geometry & SVD
svd_components = 6
svd_train_feats = np.zeros((len(train_engineered), svd_components), dtype=np.float32)
svd_test_folds = np.zeros((len(test_engineered), svd_components), dtype=np.float32)

nb_cols = word_cols + char_cols
nb_train_feats = np.zeros((len(train_engineered), 8), dtype=np.float32)
nb_test_folds = np.zeros((len(test_engineered), 8), dtype=np.float32)

knn_num_feats = 13
knn_train_feats = np.zeros((len(train_engineered), knn_num_feats), dtype=np.float32)
knn_test_folds = np.zeros((len(test_engineered), knn_num_feats), dtype=np.float32)

for fold in range(N_SPLITS):
    val_mask = folds_series == fold
    trn_mask = ~val_mask

    trn_y = y_train[trn_mask]
    trn_nb_X = (train_engineered.loc[trn_mask, nb_cols].values > 0).astype(np.float32)
    val_nb_X = (train_engineered.loc[val_mask, nb_cols].values > 0).astype(np.float32)
    tst_nb_X = (test_engineered[nb_cols].values > 0).astype(np.float32)

    # 1. Fold-Safe Dual Naive Bayes (Bernoulli + Gaussian log-likelihood ratios)
    n_spam = np.sum(trn_y == 1)
    n_ham = np.sum(trn_y == 0)
    alpha = 1.0

    p_w_spam = (np.sum(trn_nb_X[trn_y == 1], axis=0) + alpha) / (n_spam + 2 * alpha)
    p_w_ham = (np.sum(trn_nb_X[trn_y == 0], axis=0) + alpha) / (n_ham + 2 * alpha)

    llr_pos = np.log(p_w_spam) - np.log(p_w_ham)
    llr_neg = np.log(1.0 - p_w_spam) - np.log(1.0 - p_w_ham)

    def compute_nb_scores(X_mat: np.ndarray) -> np.ndarray:
        tot_score = np.dot(X_mat, llr_pos) + np.dot(1.0 - X_mat, llr_neg)
        pos_evidence = np.dot(X_mat, np.maximum(llr_pos, 0.0))
        neg_evidence = np.dot(X_mat, np.maximum(-llr_pos, 0.0))
        diff_evidence = pos_evidence - neg_evidence
        ratio_evidence = (pos_evidence + 1e-4) / (neg_evidence + 1e-4)
        return np.column_stack(
            [tot_score, pos_evidence, neg_evidence, diff_evidence, ratio_evidence]
        )

    val_base_nb = compute_nb_scores(val_nb_X)
    tst_base_nb = compute_nb_scores(tst_nb_X)

    bnb = BernoulliNB(alpha=1.0)
    bnb.fit(trn_nb_X, trn_y)
    val_bnb_llr = bnb.predict_log_proba(val_nb_X)[:, 1] - bnb.predict_log_proba(val_nb_X)[:, 0]
    tst_bnb_llr = bnb.predict_log_proba(tst_nb_X)[:, 1] - bnb.predict_log_proba(tst_nb_X)[:, 0]

    scaler_geom = RobustScaler()
    X_trn_scaled = scaler_geom.fit_transform(X_train_raw.iloc[trn_mask])
    X_val_scaled = scaler_geom.transform(X_train_raw.iloc[val_mask])
    X_tst_scaled = scaler_geom.transform(X_test_raw)

    gnb = GaussianNB(var_smoothing=1e-2)
    gnb.fit(X_trn_scaled, trn_y)
    val_gnb_llr = gnb.predict_log_proba(X_val_scaled)[:, 1] - gnb.predict_log_proba(X_val_scaled)[:, 0]
    val_gnb_prob = gnb.predict_proba(X_val_scaled)[:, 1]
    tst_gnb_llr = gnb.predict_log_proba(X_tst_scaled)[:, 1] - gnb.predict_log_proba(X_tst_scaled)[:, 0]
    tst_gnb_prob = gnb.predict_proba(X_tst_scaled)[:, 1]

    nb_train_feats[val_mask] = np.column_stack(
        [val_base_nb, val_bnb_llr, val_gnb_llr, val_gnb_prob]
    )
    nb_test_folds += np.column_stack(
        [tst_base_nb, tst_bnb_llr, tst_gnb_llr, tst_gnb_prob]
    ) / N_SPLITS

    # 2. Fold-Safe k-Nearest Neighbors Manifold Geometry (k=15, k=35)
    nn_spam = NearestNeighbors(n_neighbors=35, metric="euclidean")
    nn_ham = NearestNeighbors(n_neighbors=35, metric="euclidean")
    nn_spam.fit(X_trn_scaled[trn_y == 1])
    nn_ham.fit(X_trn_scaled[trn_y == 0])
    nn_all = NearestNeighbors(n_neighbors=25, metric="euclidean")
    nn_all.fit(X_trn_scaled)

    def extract_knn_features(X_mat: np.ndarray) -> np.ndarray:
        dist_s, _ = nn_spam.kneighbors(X_mat, n_neighbors=35)
        dist_h, _ = nn_ham.kneighbors(X_mat, n_neighbors=35)

        d_s_15 = dist_s[:, :15].mean(axis=1)
        d_h_15 = dist_h[:, :15].mean(axis=1)
        d_s_35 = dist_s.mean(axis=1)
        d_h_35 = dist_h.mean(axis=1)
        d_s_min = dist_s[:, 0]
        d_h_min = dist_h[:, 0]

        ratio_15 = (d_s_15 + 1e-4) / (d_h_15 + 1e-4)
        ratio_35 = (d_s_35 + 1e-4) / (d_h_35 + 1e-4)
        diff_15 = d_h_15 - d_s_15
        diff_35 = d_h_35 - d_s_35
        min_diff = d_h_min - d_s_min
        min_ratio = (d_s_min + 1e-4) / (d_h_min + 1e-4)

        _, idx_all = nn_all.kneighbors(X_mat, n_neighbors=25)
        local_spam_density = trn_y[idx_all].mean(axis=1)

        return np.column_stack(
            [
                d_s_15,
                d_h_15,
                d_s_35,
                d_h_35,
                d_s_min,
                d_h_min,
                ratio_15,
                ratio_35,
                diff_15,
                diff_35,
                min_diff,
                min_ratio,
                local_spam_density,
            ]
        )

    knn_train_feats[val_mask] = extract_knn_features(X_val_scaled)
    knn_test_folds += extract_knn_features(X_tst_scaled) / N_SPLITS

    # 3. Fold-Safe SVD Dimensionality Reduction
    svd = TruncatedSVD(n_components=svd_components, random_state=RANDOM_STATE + fold)
    svd.fit(X_trn_scaled)
    svd_train_feats[val_mask] = svd.transform(X_val_scaled)
    svd_test_folds += svd.transform(X_tst_scaled) / N_SPLITS

nb_feat_names = [
    "nb_log_odds_score",
    "nb_pos_evidence",
    "nb_neg_evidence",
    "nb_evidence_diff",
    "nb_evidence_ratio",
    "nb_bnb_llr",
    "nb_gnb_llr",
    "nb_gnb_prob",
]
knn_feat_names = [
    "knn_dist_spam_15",
    "knn_dist_ham_15",
    "knn_dist_spam_35",
    "knn_dist_ham_35",
    "knn_dist_spam_min",
    "knn_dist_ham_min",
    "knn_dist_ratio_15",
    "knn_dist_ratio_35",
    "knn_dist_diff_15",
    "knn_dist_diff_35",
    "knn_dist_min_diff",
    "knn_dist_min_ratio",
    "knn_local_spam_density",
]

nb_train_df = pd.DataFrame(nb_train_feats, columns=nb_feat_names, index=train_engineered.index)
nb_test_df = pd.DataFrame(nb_test_folds, columns=nb_feat_names, index=test_engineered.index)

knn_train_df = pd.DataFrame(knn_train_feats, columns=knn_feat_names, index=train_engineered.index)
knn_test_df = pd.DataFrame(knn_test_folds, columns=knn_feat_names, index=test_engineered.index)

svd_train_df = pd.DataFrame(
    {f"svd_comp_{i}": svd_train_feats[:, i] for i in range(svd_components)},
    index=train_engineered.index,
)
svd_test_df = pd.DataFrame(
    {f"svd_comp_{i}": svd_test_folds[:, i] for i in range(svd_components)},
    index=test_engineered.index,
)

train_engineered = pd.concat([train_engineered, nb_train_df, knn_train_df, svd_train_df], axis=1)
test_engineered = pd.concat([test_engineered, nb_test_df, knn_test_df, svd_test_df], axis=1)

feature_cols = [c for c in train_engineered.columns if c not in [target_col, "fold"]]
X_test_all = test_engineered[feature_cols].values
y_true = train_engineered[target_col].values
n_samples = len(train_engineered)
n_test = len(test_engineered)
input_dim = len(feature_cols)


# =========================================================================
# 2. PyTorch Deep Tabular Residual Network (TabularResNet)
# =========================================================================
class TabularResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout_rate: float = 0.2):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.act1 = nn.Mish()
        self.dropout1 = nn.Dropout(dropout_rate)

        self.norm2 = nn.LayerNorm(hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.act2 = nn.Mish()
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.norm1(x)
        out = self.linear1(out)
        out = self.act1(out)
        out = self.dropout1(out)

        out = self.norm2(out)
        out = self.linear2(out)
        out = self.act2(out)
        out = self.dropout2(out)
        return residual + out


class TabularResNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Tuple[int, ...] = (256, 128, 64),
        dropout_rate: float = 0.2,
        num_res_blocks: int = 2,
    ):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.Mish(),
            nn.Dropout(dropout_rate),
        )
        self.res_backbone = nn.Sequential(
            *[
                TabularResidualBlock(hidden_dims[0], dropout_rate=dropout_rate)
                for _ in range(num_res_blocks)
            ]
        )

        layers = []
        in_dim = hidden_dims[0]
        for out_dim in hidden_dims[1:]:
            layers.extend(
                [
                    nn.Linear(in_dim, out_dim),
                    nn.LayerNorm(out_dim),
                    nn.Mish(),
                    nn.Dropout(dropout_rate),
                ]
            )
            in_dim = out_dim
        self.compression_head = nn.Sequential(*layers)
        self.classifier = nn.Linear(hidden_dims[-1], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_norm(x)
        x = self.input_proj(x)
        x = self.res_backbone(x)
        x = self.compression_head(x)
        logits = self.classifier(x)
        return logits.squeeze(-1)


# =========================================================================
# 3. Model Training & Cross Validation Across 6 Diverse Model Families
# =========================================================================
oof_lgb = np.zeros(n_samples, dtype=np.float32)
oof_xgb = np.zeros(n_samples, dtype=np.float32)
oof_cat = np.zeros(n_samples, dtype=np.float32) if HAS_CATBOOST else None
oof_et = np.zeros(n_samples, dtype=np.float32)
oof_rf = np.zeros(n_samples, dtype=np.float32)
oof_nn = np.zeros(n_samples, dtype=np.float32)

test_preds_lgb = np.zeros(n_test, dtype=np.float32)
test_preds_xgb = np.zeros(n_test, dtype=np.float32)
test_preds_cat = np.zeros(n_test, dtype=np.float32) if HAS_CATBOOST else None
test_preds_et = np.zeros(n_test, dtype=np.float32)
test_preds_rf = np.zeros(n_test, dtype=np.float32)
test_preds_nn = np.zeros(n_test, dtype=np.float32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for fold in range(N_SPLITS):
    trn_mask = train_engineered["fold"] != fold
    val_mask = train_engineered["fold"] == fold

    X_trn = train_engineered.loc[trn_mask, feature_cols].values
    y_trn = train_engineered.loc[trn_mask, target_col].values
    X_val = train_engineered.loc[val_mask, feature_cols].values
    y_val = train_engineered.loc[val_mask, target_col].values

    # Model 1: LightGBM
    lgb_clf = lgb.LGBMClassifier(
        n_estimators=1500,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=20,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective="binary",
        random_state=RANDOM_STATE + fold,
        n_jobs=-1,
        verbosity=-1,
    )
    lgb_clf.fit(
        X_trn,
        y_trn,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )
    oof_lgb[val_mask] = lgb_clf.predict_proba(X_val)[:, 1]
    test_preds_lgb += lgb_clf.predict_proba(X_test_all)[:, 1] / N_SPLITS

    # Model 2: XGBoost
    xgb_clf = XGBClassifier(
        n_estimators=1500,
        learning_rate=0.03,
        max_depth=6,
        min_child_weight=2,
        subsample=0.85,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        early_stopping_rounds=50,
        random_state=RANDOM_STATE + fold,
        n_jobs=-1,
    )
    xgb_clf.fit(X_trn, y_trn, eval_set=[(X_val, y_val)], verbose=False)
    oof_xgb[val_mask] = xgb_clf.predict_proba(X_val)[:, 1]
    test_preds_xgb += xgb_clf.predict_proba(X_test_all)[:, 1] / N_SPLITS

    # Model 3: CatBoost
    if HAS_CATBOOST:
        cat_clf = CatBoostClassifier(
            iterations=1500,
            learning_rate=0.03,
            depth=6,
            l2_leaf_reg=3.0,
            loss_function="Logloss",
            eval_metric="AUC",
            early_stopping_rounds=50,
            random_seed=RANDOM_STATE + fold,
            verbose=False,
            thread_count=-1,
        )
        cat_clf.fit(X_trn, y_trn, eval_set=(X_val, y_val), verbose=False)
        oof_cat[val_mask] = cat_clf.predict_proba(X_val)[:, 1]
        test_preds_cat += cat_clf.predict_proba(X_test_all)[:, 1] / N_SPLITS

    # Model 4: ExtraTreesClassifier
    et_clf = ExtraTreesClassifier(
        n_estimators=500,
        max_features="sqrt",
        min_samples_split=4,
        min_samples_leaf=2,
        max_depth=28,
        random_state=RANDOM_STATE + fold,
        n_jobs=-1,
    )
    et_clf.fit(X_trn, y_trn)
    oof_et[val_mask] = et_clf.predict_proba(X_val)[:, 1]
    test_preds_et += et_clf.predict_proba(X_test_all)[:, 1] / N_SPLITS

    # Model 5: RandomForestClassifier
    rf_clf = RandomForestClassifier(
        n_estimators=500,
        max_features="sqrt",
        min_samples_split=4,
        min_samples_leaf=2,
        max_depth=28,
        random_state=RANDOM_STATE + fold,
        n_jobs=-1,
    )
    rf_clf.fit(X_trn, y_trn)
    oof_rf[val_mask] = rf_clf.predict_proba(X_val)[:, 1]
    test_preds_rf += rf_clf.predict_proba(X_test_all)[:, 1] / N_SPLITS

    # Model 6: Multi-Seed PyTorch TabularResNet
    scaler_nn = StandardScaler()
    X_trn_nn = scaler_nn.fit_transform(X_trn)
    X_val_nn = scaler_nn.transform(X_val)
    X_tst_nn = scaler_nn.transform(X_test_all)

    train_dataset = TensorDataset(
        torch.tensor(X_trn_nn, dtype=torch.float32),
        torch.tensor(y_trn, dtype=torch.float32),
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val_nn, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32),
    )
    test_tensor = torch.tensor(X_tst_nn, dtype=torch.float32).to(device)

    nn_val_seed_preds = []
    nn_test_seed_preds = []

    nn_seeds = [RANDOM_STATE + fold, RANDOM_STATE + fold + 777]
    for nn_seed in nn_seeds:
        seed_everything(nn_seed)
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

        nn_model = TabularResNet(
            input_dim=input_dim,
            hidden_dims=(256, 128, 64),
            dropout_rate=0.2,
            num_res_blocks=2,
        ).to(device)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(
            nn_model.parameters(), lr=1e-3, weight_decay=1e-4, eps=1e-8
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=35, eta_min=1e-6
        )

        best_val_auc = 0.0
        best_val_preds = None
        best_weights = None

        for epoch in range(35):
            nn_model.train()
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                logits = nn_model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
            scheduler.step()

            nn_model.eval()
            val_preds_list = []
            with torch.no_grad():
                for batch_x, _ in val_loader:
                    batch_x = batch_x.to(device)
                    logits = nn_model(batch_x)
                    probs = torch.sigmoid(logits)
                    val_preds_list.append(probs.cpu().numpy())
            val_probs = np.concatenate(val_preds_list)
            epoch_auc = roc_auc_score(y_val, val_probs)

            if epoch_auc > best_val_auc:
                best_val_auc = epoch_auc
                best_val_preds = val_probs
                best_weights = {k: v.cpu() for k, v in nn_model.state_dict().items()}

        nn_val_seed_preds.append(best_val_preds)
        nn_model.load_state_dict({k: v.to(device) for k, v in best_weights.items()})
        nn_model.eval()
        with torch.no_grad():
            test_probs_seed = torch.sigmoid(nn_model(test_tensor)).cpu().numpy()
        nn_test_seed_preds.append(test_probs_seed)

    oof_nn[val_mask] = np.mean(nn_val_seed_preds, axis=0)
    test_preds_nn += np.mean(nn_test_seed_preds, axis=0) / N_SPLITS

# =========================================================================
# 4. SLSQP Constrained Meta-Optimization & Rank-and-Logit Blending
# =========================================================================
models_oof = [oof_lgb, oof_xgb, oof_et, oof_rf, oof_nn]
models_test = [test_preds_lgb, test_preds_xgb, test_preds_et, test_preds_rf, test_preds_nn]

if HAS_CATBOOST and oof_cat is not None:
    models_oof.insert(2, oof_cat)
    models_test.insert(2, test_preds_cat)

oof_matrix = np.column_stack(models_oof)
test_matrix = np.column_stack(models_test)
n_models = oof_matrix.shape[1]

clip_eps = 1e-6
oof_logits = logit(np.clip(oof_matrix, clip_eps, 1.0 - clip_eps))
test_logits = logit(np.clip(test_matrix, clip_eps, 1.0 - clip_eps))

oof_ranks = np.column_stack(
    [rankdata(oof_matrix[:, i]) / n_samples for i in range(n_models)]
)
test_ranks = np.column_stack(
    [rankdata(test_matrix[:, i]) / n_test for i in range(n_models)]
)


def objective(weights: np.ndarray) -> float:
    w = weights / np.sum(weights)
    blend_logits = np.dot(oof_logits, w)
    return -roc_auc_score(y_true, blend_logits)


init_weights = np.ones(n_models) / n_models
bounds = [(0.0, 1.0) for _ in range(n_models)]
constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

opt_res = minimize(
    objective,
    init_weights,
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
    options={"maxiter": 500, "ftol": 1e-7},
)

opt_weights = opt_res.x / np.sum(opt_res.x)

oof_logit_blend = expit(np.dot(oof_logits, opt_weights))
test_logit_blend = expit(np.dot(test_logits, opt_weights))

oof_rank_blend = np.dot(oof_ranks, opt_weights)
test_rank_blend = np.dot(test_ranks, opt_weights)

# Harmonized rank-and-logit blending
final_oof_preds = 0.5 * oof_logit_blend + 0.5 * oof_rank_blend
final_test_preds = 0.5 * test_logit_blend + 0.5 * test_rank_blend
final_score = roc_auc_score(y_true, final_oof_preds)

# =========================================================================
# 5. Submission Output & Integrity Checks
# =========================================================================
submission_df = pd.DataFrame(
    {"id": np.arange(len(final_test_preds)), "class": final_test_preds}
)

submission_path = os.path.join(SUBMISSION_DIR, "submission.csv")
submission_df.to_csv(submission_path, index=False)

assert len(submission_df) == len(
    test_df
), "Mismatch in submission row count vs test set."
assert list(submission_df.columns) == [
    "id",
    "class",
], "Invalid submission column structure."
assert not submission_df.isnull().values.any(), "Submission contains null/NaN values."

print(f"Final Validation Score: {final_score}")
