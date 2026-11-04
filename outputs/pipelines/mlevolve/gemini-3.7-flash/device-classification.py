import json
import os
import re
from typing import Any, Dict, Optional, Tuple
import warnings
from catboost import CatBoostClassifier
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.sparse import hstack
from scipy.special import softmax
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb

warnings.filterwarnings("ignore")

# -------------------------------------------------------------------------
# 1. Directories & Data Ingestion
# -------------------------------------------------------------------------
input_dir = "./input"
working_dir = "./working"
submission_dir = "./submission"

os.makedirs(working_dir, exist_ok=True)
os.makedirs(submission_dir, exist_ok=True)

train_path = os.path.join(input_dir, "train.parquet")
test_path = os.path.join(input_dir, "test.parquet")

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = "device_class"
id_col = "id" if "id" in test_df.columns else None

y_raw = train_df[target_col].copy()
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_raw)
num_classes = len(label_encoder.classes_)
target_mapping = {
    int(cls_idx): cls_name for cls_idx, cls_name in enumerate(label_encoder.classes_)
}

with open(os.path.join(working_dir, "target_mapping.json"), "w") as f:
    json.dump({int(k): str(v) for k, v in target_mapping.items()}, f, indent=2)

train_features = train_df.drop(columns=[target_col])
if id_col and id_col in train_features.columns:
    train_features = train_features.drop(columns=[id_col])

test_ids = test_df[id_col].values if id_col else np.arange(len(test_df))
test_features = (
    test_df.drop(columns=[id_col])
    if id_col and id_col in test_df.columns
    else test_df.copy()
)

common_cols = [col for col in train_features.columns if col in test_features.columns]
train_features = train_features[common_cols]
test_features = test_features[common_cols]

# -------------------------------------------------------------------------
# 2. Modality Detection & Feature Extraction
# -------------------------------------------------------------------------
num_cols = []
cat_cols = []
text_cols = []

for col in common_cols:
    train_series = train_features[col].dropna()
    if pd.api.types.is_numeric_dtype(train_features[col]):
        n_unique = train_series.nunique()
        if n_unique <= 2:
            cat_cols.append(col)
        else:
            num_cols.append(col)
    else:
        avg_word_count = (
            train_series.astype(str).apply(lambda x: len(x.split())).mean()
            if len(train_series) > 0
            else 0
        )
        n_unique = train_series.nunique()
        if avg_word_count > 3.0 or (n_unique > 50 and avg_word_count > 1.8):
            text_cols.append(col)
        else:
            cat_cols.append(col)


def extract_text_meta_features(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    meta_df = pd.DataFrame(index=df.index)
    for col in cols:
        s = df[col].fillna("").astype(str)
        meta_df[f"{col}_char_len"] = s.apply(len)
        meta_df[f"{col}_word_count"] = s.apply(lambda x: len(x.split()))
        meta_df[f"{col}_digit_count"] = s.apply(lambda x: sum(c.isdigit() for c in x))
        meta_df[f"{col}_upper_count"] = s.apply(lambda x: sum(c.isupper() for c in x))
        meta_df[f"{col}_has_punct"] = s.apply(
            lambda x: 1 if bool(re.search(r"[^\w\s]", x)) else 0
        )
        meta_df[f"{col}_avg_word_len"] = meta_df[f"{col}_char_len"] / (
            meta_df[f"{col}_word_count"] + 1e-5
        )
        meta_df[f"{col}_upper_ratio"] = meta_df[f"{col}_upper_count"] / (
            meta_df[f"{col}_char_len"] + 1e-5
        )
        meta_df[f"{col}_digit_ratio"] = meta_df[f"{col}_digit_count"] / (
            meta_df[f"{col}_char_len"] + 1e-5
        )
    return meta_df


def create_missingness_flags(
    train_f: pd.DataFrame, test_f: pd.DataFrame, cols: list
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_nulls = pd.DataFrame(index=train_f.index)
    test_nulls = pd.DataFrame(index=test_f.index)
    for col in cols:
        if train_f[col].isnull().sum() > 0 or test_f[col].isnull().sum() > 0:
            train_nulls[f"{col}_isna"] = train_f[col].isnull().astype(np.float32)
            test_nulls[f"{col}_isna"] = test_f[col].isnull().astype(np.float32)
    train_nulls["missing_count_row"] = train_f[cols].isnull().sum(axis=1).astype(np.float32)
    test_nulls["missing_count_row"] = test_f[cols].isnull().sum(axis=1).astype(np.float32)
    train_nulls["missing_ratio_row"] = (train_f[cols].isnull().sum(axis=1) / len(cols)).astype(np.float32)
    test_nulls["missing_ratio_row"] = (test_f[cols].isnull().sum(axis=1) / len(cols)).astype(np.float32)
    return train_nulls, test_nulls


train_null_flags, test_null_flags = create_missingness_flags(
    train_features, test_features, common_cols
)
train_text_meta = extract_text_meta_features(train_features, text_cols)
test_text_meta = extract_text_meta_features(test_features, text_cols)

# -------------------------------------------------------------------------
# 3. Stratified Folds Setup
# -------------------------------------------------------------------------
N_SPLITS = 5
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
fold_indices = np.zeros(len(train_features), dtype=int)
for fold_idx, (train_idx, val_idx) in enumerate(skf.split(train_features, y_encoded)):
    fold_indices[val_idx] = fold_idx

# -------------------------------------------------------------------------
# 4. Text Vectorization, Truncated SVD, NMF & OOF Multi-Class Stacking
# -------------------------------------------------------------------------
text_train_components = []
text_test_components = []

# Build target text items list including unified cross-column text
text_items = list(text_cols)
if len(text_cols) > 0:
    combined_train_text = train_features[text_cols].fillna("").astype(str).agg(" ".join, axis=1).values
    combined_test_text = test_features[text_cols].fillna("").astype(str).agg(" ".join, axis=1).values
    text_items.append("__combined_text__")

for col in text_items:
    if col == "__combined_text__":
        train_corpus = combined_train_text
        test_corpus = combined_test_text
        max_feat = 8000
        target_svd_comps = 64
        target_nmf_comps = 24
    else:
        train_corpus = train_features[col].fillna("").astype(str).values
        test_corpus = test_features[col].fillna("").astype(str).values
        max_feat = 5000
        target_svd_comps = 35
        target_nmf_comps = 15

    word_vec = TfidfVectorizer(
        ngram_range=(1, 3), max_features=max_feat, sublinear_tf=True, token_pattern=r"(?u)\b\w+\b"
    )
    char_vec = TfidfVectorizer(
        ngram_range=(2, 5), analyzer="char_wb", max_features=max_feat, sublinear_tf=True
    )

    tfidf_train_w = word_vec.fit_transform(train_corpus)
    tfidf_test_w = word_vec.transform(test_corpus)

    tfidf_train_c = char_vec.fit_transform(train_corpus)
    tfidf_test_c = char_vec.transform(test_corpus)

    tfidf_train = hstack([tfidf_train_w, tfidf_train_c]).tocsr()
    tfidf_test = hstack([tfidf_test_w, tfidf_test_c]).tocsr()

    n_svd_comps = min(target_svd_comps, max(2, tfidf_train.shape[1] - 1))
    if n_svd_comps > 0 and tfidf_train.shape[1] > 1:
        svd = TruncatedSVD(n_components=n_svd_comps, random_state=42)
        svd_train = svd.fit_transform(tfidf_train)
        svd_test = svd.transform(tfidf_test)
        svd_cols = [f"{col}_svd_{i}" for i in range(n_svd_comps)]
        text_train_components.append(
            pd.DataFrame(svd_train, columns=svd_cols, index=train_features.index)
        )
        text_test_components.append(
            pd.DataFrame(svd_test, columns=svd_cols, index=test_features.index)
        )

    n_nmf_comps = min(target_nmf_comps, max(2, tfidf_train.shape[1] - 1))
    if n_nmf_comps > 0 and tfidf_train.shape[1] > 1:
        nmf = NMF(n_components=n_nmf_comps, random_state=42, max_iter=200)
        nmf_train = nmf.fit_transform(tfidf_train)
        nmf_test = nmf.transform(tfidf_test)
        nmf_cols = [f"{col}_nmf_{i}" for i in range(n_nmf_comps)]
        text_train_components.append(
            pd.DataFrame(nmf_train, columns=nmf_cols, index=train_features.index)
        )
        text_test_components.append(
            pd.DataFrame(nmf_test, columns=nmf_cols, index=test_features.index)
        )

    oof_lr_probs = np.zeros((len(train_features), num_classes), dtype=np.float32)
    test_lr_probs = np.zeros((len(test_features), num_classes), dtype=np.float32)
    oof_nb_probs = np.zeros((len(train_features), num_classes), dtype=np.float32)
    test_nb_probs = np.zeros((len(test_features), num_classes), dtype=np.float32)
    oof_ridge_probs = np.zeros((len(train_features), num_classes), dtype=np.float32)
    test_ridge_probs = np.zeros((len(test_features), num_classes), dtype=np.float32)

    for fold_idx in range(N_SPLITS):
        val_mask = fold_indices == fold_idx
        tr_mask = ~val_mask

        # Logistic Regression
        lr = LogisticRegression(C=1.5, max_iter=350, random_state=42)
        lr.fit(tfidf_train[tr_mask], y_encoded[tr_mask])
        oof_lr_probs[val_mask] = lr.predict_proba(tfidf_train[val_mask])
        test_lr_probs += lr.predict_proba(tfidf_test) / N_SPLITS

        # Multinomial Naive Bayes
        nb = MultinomialNB(alpha=0.5)
        nb.fit(tfidf_train[tr_mask], y_encoded[tr_mask])
        oof_nb_probs[val_mask] = nb.predict_proba(tfidf_train[val_mask])
        test_nb_probs += nb.predict_proba(tfidf_test) / N_SPLITS

        # Calibrated Ridge Classifier
        ridge = CalibratedClassifierCV(RidgeClassifier(alpha=1.0, random_state=42), cv=3)
        ridge.fit(tfidf_train[tr_mask], y_encoded[tr_mask])
        oof_ridge_probs[val_mask] = ridge.predict_proba(tfidf_train[val_mask])
        test_ridge_probs += ridge.predict_proba(tfidf_test) / N_SPLITS

    lr_cols = [f"{col}_tfidf_lr_prob_{c}" for c in range(num_classes)]
    nb_cols = [f"{col}_tfidf_nb_prob_{c}" for c in range(num_classes)]
    ridge_cols = [f"{col}_tfidf_ridge_prob_{c}" for c in range(num_classes)]

    text_train_components.append(
        pd.DataFrame(oof_lr_probs, columns=lr_cols, index=train_features.index)
    )
    text_test_components.append(
        pd.DataFrame(test_lr_probs, columns=lr_cols, index=test_features.index)
    )
    text_train_components.append(
        pd.DataFrame(oof_nb_probs, columns=nb_cols, index=train_features.index)
    )
    text_test_components.append(
        pd.DataFrame(test_nb_probs, columns=nb_cols, index=test_features.index)
    )
    text_train_components.append(
        pd.DataFrame(oof_ridge_probs, columns=ridge_cols, index=train_features.index)
    )
    text_test_components.append(
        pd.DataFrame(test_ridge_probs, columns=ridge_cols, index=test_features.index)
    )

# -------------------------------------------------------------------------
# 5. Categorical, Frequency, Interaction & Target Encoding
# -------------------------------------------------------------------------
freq_encoded_train = pd.DataFrame(index=train_features.index)
freq_encoded_test = pd.DataFrame(index=test_features.index)
ordinal_encoded_train = pd.DataFrame(index=train_features.index)
ordinal_encoded_test = pd.DataFrame(index=test_features.index)

train_cat_full = train_features[cat_cols].astype(str).copy()
test_cat_full = test_features[cat_cols].astype(str).copy()
interaction_cols = []

if len(cat_cols) >= 2:
    for i in range(min(3, len(cat_cols))):
        for j in range(i + 1, min(3, len(cat_cols))):
            c1, c2 = cat_cols[i], cat_cols[j]
            pair_name = f"{c1}_x_{c2}"
            train_cat_full[pair_name] = train_cat_full[c1] + "__" + train_cat_full[c2]
            test_cat_full[pair_name] = test_cat_full[c1] + "__" + test_cat_full[c2]
            interaction_cols.append(pair_name)

all_categorical_cols = cat_cols + interaction_cols

for col in all_categorical_cols:
    freq_map = train_cat_full[col].value_counts(normalize=True).to_dict()
    freq_encoded_train[f"{col}_freq"] = (
        train_cat_full[col].map(freq_map).fillna(0.0).astype(np.float32)
    )
    freq_encoded_test[f"{col}_freq"] = (
        test_cat_full[col].map(freq_map).fillna(0.0).astype(np.float32)
    )

    unique_vals = list(train_cat_full[col].unique())
    val_to_cat = {val: i + 1 for i, val in enumerate(unique_vals)}
    ordinal_encoded_train[f"{col}_ord"] = (
        train_cat_full[col].map(val_to_cat).fillna(0).astype(np.int32)
    )
    ordinal_encoded_test[f"{col}_ord"] = (
        test_cat_full[col].map(val_to_cat).fillna(0).astype(np.int32)
    )

num_classes = len(np.unique(y_encoded))
oof_target_train = pd.DataFrame(index=train_features.index)
oof_target_test = pd.DataFrame(index=test_features.index)

for col in all_categorical_cols:
    for c in range(num_classes):
        oof_col_name = f"{col}_te_class_{c}"
        oof_target_train[oof_col_name] = 0.0
        global_prior = float((y_encoded == c).mean())

        for fold in range(N_SPLITS):
            val_mask = fold_indices == fold
            tr_mask = ~val_mask

            group_stats = (
                pd.DataFrame(
                    {
                        "cat": train_cat_full.loc[tr_mask, col],
                        "target": (y_encoded[tr_mask] == c).astype(float),
                    }
                )
                .groupby("cat")["target"]
                .agg(["count", "mean"])
            )

            smoothing_weight = 10.0
            smoothed = (
                group_stats["count"] * group_stats["mean"]
                + smoothing_weight * global_prior
            ) / (group_stats["count"] + smoothing_weight)
            smooth_dict = smoothed.to_dict()

            oof_target_train.loc[val_mask, oof_col_name] = (
                train_cat_full.loc[val_mask, col].map(smooth_dict).fillna(global_prior)
            )

        full_group_stats = (
            pd.DataFrame(
                {"cat": train_cat_full[col], "target": (y_encoded == c).astype(float)}
            )
            .groupby("cat")["target"]
            .agg(["count", "mean"])
        )
        full_smoothed = (
            full_group_stats["count"] * full_group_stats["mean"] + 10.0 * global_prior
        ) / (full_group_stats["count"] + 10.0)
        oof_target_test[oof_col_name] = (
            test_cat_full[col].map(full_smoothed.to_dict()).fillna(global_prior)
        )

# -------------------------------------------------------------------------
# 6. Numerical Transformations & Aggregations
# -------------------------------------------------------------------------
num_train = pd.DataFrame(index=train_features.index)
num_test = pd.DataFrame(index=test_features.index)

if len(num_cols) > 0:
    num_imputer = SimpleImputer(strategy="median")
    imputed_train = pd.DataFrame(
        num_imputer.fit_transform(train_features[num_cols]),
        columns=num_cols,
        index=train_features.index,
    )
    imputed_test = pd.DataFrame(
        num_imputer.transform(test_features[num_cols]),
        columns=num_cols,
        index=test_features.index,
    )

    for col in num_cols:
        num_train[f"{col}_raw"] = imputed_train[col]
        num_test[f"{col}_raw"] = imputed_test[col]
        if (imputed_train[col] >= 0).all():
            num_train[f"{col}_log1p"] = np.log1p(imputed_train[col])
            num_test[f"{col}_log1p"] = np.log1p(np.maximum(0, imputed_test[col]))

    if len(num_cols) >= 2:
        for i in range(min(4, len(num_cols))):
            for j in range(i + 1, min(4, len(num_cols))):
                c1, c2 = num_cols[i], num_cols[j]
                num_train[f"{c1}_div_{c2}"] = imputed_train[c1] / (
                    imputed_train[c2].abs() + 1e-5
                )
                num_test[f"{c1}_div_{c2}"] = imputed_test[c1] / (
                    imputed_test[c2].abs() + 1e-5
                )
                num_train[f"{c1}_sub_{c2}"] = imputed_train[c1] - imputed_train[c2]
                num_test[f"{c1}_sub_{c2}"] = imputed_test[c1] - imputed_test[c2]

    if len(cat_cols) > 0:
        top_cats = cat_cols[:2]
        for c_cat in top_cats:
            for n_col in num_cols[:3]:
                grouped_mean = train_features.groupby(c_cat)[n_col].mean().to_dict()
                grouped_std = (
                    train_features.groupby(c_cat)[n_col].std().fillna(0.0).to_dict()
                )

                num_train[f"{n_col}_grpby_{c_cat}_mean"] = (
                    train_features[c_cat].map(grouped_mean).fillna(0.0)
                )
                num_test[f"{n_col}_grpby_{c_cat}_mean"] = (
                    test_features[c_cat].map(grouped_mean).fillna(0.0)
                )
                num_train[f"{n_col}_grpby_{c_cat}_std"] = (
                    train_features[c_cat].map(grouped_std).fillna(0.0)
                )
                num_test[f"{n_col}_grpby_{c_cat}_std"] = (
                    test_features[c_cat].map(grouped_std).fillna(0.0)
                )

# -------------------------------------------------------------------------
# 7. Assemble Full Feature Matrices
# -------------------------------------------------------------------------
dfs_train_to_concat = [
    train_null_flags,
    train_text_meta,
    freq_encoded_train,
    ordinal_encoded_train,
    oof_target_train,
    num_train,
] + text_train_components

dfs_test_to_concat = [
    test_null_flags,
    test_text_meta,
    freq_encoded_test,
    ordinal_encoded_test,
    oof_target_test,
    num_test,
] + text_test_components

X_train_engineered = pd.concat(
    [df for df in dfs_train_to_concat if not df.empty], axis=1
)
X_test_engineered = pd.concat([df for df in dfs_test_to_concat if not df.empty], axis=1)

non_constant_cols = [
    c for c in X_train_engineered.columns if X_train_engineered[c].nunique() > 1
]
X_train_clean = (
    X_train_engineered[non_constant_cols].fillna(0.0).values.astype(np.float32)
)
X_test_clean = (
    X_test_engineered[non_constant_cols].fillna(0.0).values.astype(np.float32)
)
y_train_clean = y_encoded.astype(np.int64)

n_samples, n_features = X_train_clean.shape


# -------------------------------------------------------------------------
# 8. Neural Architecture & Loss Definitions
# -------------------------------------------------------------------------
class ResNetBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout_rate: float = 0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class TabularResNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 256,
        num_blocks: int = 2,
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )
        self.res_blocks = nn.ModuleList(
            [
                ResNetBlock(hidden_dim=hidden_dim, dropout_rate=dropout_rate)
                for _ in range(num_blocks)
            ]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_projection(x)
        for block in self.res_blocks:
            x = block(x)
        return self.head(x)


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing: float = 0.05):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_preds = torch.log_softmax(preds, dim=-1)
        loss_ce = -log_preds.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
        loss_smooth = -log_preds.mean(dim=-1)
        return ((1.0 - self.smoothing) * loss_ce + self.smoothing * loss_smooth).mean()


def build_neural_model(
    input_dim: int, num_classes: int, hidden_dim: int = 256, num_blocks: int = 2
) -> TabularResNet:
    return TabularResNet(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        num_blocks=num_blocks,
        dropout_rate=0.2,
    )


def get_optimizer_and_scheduler(
    model: nn.Module, lr: float = 1.5e-3, epochs: int = 35, steps_per_epoch: int = 50
) -> Tuple[optim.Optimizer, optim.lr_scheduler._LRScheduler]:
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs * max(1, steps_per_epoch), eta_min=1e-6
    )
    return optimizer, scheduler


# -------------------------------------------------------------------------
# 9. GBDT Configurations
# -------------------------------------------------------------------------
def get_lgbm_params(n_cls: int, seed: int = 42) -> Dict[str, Any]:
    return {
        "objective": "multiclass",
        "num_class": n_cls,
        "metric": "multi_logloss",
        "boosting_type": "gbdt",
        "n_estimators": 1000,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": 6,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "min_child_samples": 20,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": seed,
        "n_jobs": -1,
        "verbose": -1,
    }


def get_xgboost_params(n_cls: int, seed: int = 42) -> Dict[str, Any]:
    return {
        "objective": "multi:softprob",
        "num_class": n_cls,
        "eval_metric": "mlogloss",
        "n_estimators": 1000,
        "learning_rate": 0.03,
        "max_depth": 5,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "min_child_weight": 3,
        "gamma": 0.1,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": seed,
        "n_jobs": -1,
        "tree_method": "hist",
    }


def get_catboost_params(n_cls: int, seed: int = 42) -> Dict[str, Any]:
    return {
        "loss_function": "MultiClass",
        "iterations": 1000,
        "learning_rate": 0.03,
        "depth": 6,
        "l2_leaf_reg": 3.0,
        "subsample": 0.85,
        "bootstrap_type": "Bernoulli",
        "random_seed": seed,
        "verbose": False,
        "thread_count": -1,
    }


# -------------------------------------------------------------------------
# 10. Neural Training Function
# -------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_neural_net(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    X_te: np.ndarray,
    input_dim: int,
    n_cls: int,
    epochs: int = 35,
    batch_size: int = 64,
) -> Tuple[np.ndarray, np.ndarray]:
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_va_s = scaler.transform(X_va)
    X_te_s = scaler.transform(X_te)

    tr_dataset = TensorDataset(
        torch.tensor(X_tr_s, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.long)
    )
    va_dataset = TensorDataset(
        torch.tensor(X_va_s, dtype=torch.float32), torch.tensor(y_va, dtype=torch.long)
    )

    tr_loader = DataLoader(
        tr_dataset, batch_size=batch_size, shuffle=True, drop_last=False
    )
    va_loader = DataLoader(
        va_dataset, batch_size=batch_size * 2, shuffle=False, drop_last=False
    )

    model = build_neural_model(
        input_dim=input_dim, num_classes=n_cls, hidden_dim=256, num_blocks=2
    ).to(device)
    optimizer, scheduler = get_optimizer_and_scheduler(
        model,
        lr=1.5e-3,
        epochs=epochs,
        steps_per_epoch=len(tr_loader),
    )
    criterion = LabelSmoothingCrossEntropy(smoothing=0.05)

    best_va_loss = float("inf")
    best_weights = None

    for _ in range(epochs):
        model.train()
        for batch_x, batch_y in tr_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in va_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                va_loss += loss.item() * len(batch_y)
        va_loss /= len(y_va)

        if va_loss < best_va_loss:
            best_va_loss = va_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_weights is not None:
        model.load_state_dict(best_weights)
    model.eval()

    with torch.no_grad():
        va_t = torch.tensor(X_va_s, dtype=torch.float32).to(device)
        va_probs = torch.softmax(model(va_t), dim=-1).cpu().numpy()
        te_t = torch.tensor(X_te_s, dtype=torch.float32).to(device)
        te_probs = torch.softmax(model(te_t), dim=-1).cpu().numpy()

    return va_probs, te_probs


# -------------------------------------------------------------------------
# 11. Cross-Validation Model Ensemble Training
# -------------------------------------------------------------------------
oof_preds_lgb = np.zeros((n_samples, num_classes), dtype=np.float32)
oof_preds_xgb = np.zeros((n_samples, num_classes), dtype=np.float32)
oof_preds_cat = np.zeros((n_samples, num_classes), dtype=np.float32)
oof_preds_et = np.zeros((n_samples, num_classes), dtype=np.float32)
oof_preds_rf = np.zeros((n_samples, num_classes), dtype=np.float32)
oof_preds_nn = np.zeros((n_samples, num_classes), dtype=np.float32)

test_preds_lgb = np.zeros((len(test_ids), num_classes), dtype=np.float32)
test_preds_xgb = np.zeros((len(test_ids), num_classes), dtype=np.float32)
test_preds_cat = np.zeros((len(test_ids), num_classes), dtype=np.float32)
test_preds_et = np.zeros((len(test_ids), num_classes), dtype=np.float32)
test_preds_rf = np.zeros((len(test_ids), num_classes), dtype=np.float32)
test_preds_nn = np.zeros((len(test_ids), num_classes), dtype=np.float32)

for fold_idx in range(N_SPLITS):
    val_mask = fold_indices == fold_idx
    tr_mask = ~val_mask

    X_train_fold, y_train_fold = X_train_clean[tr_mask], y_train_clean[tr_mask]
    X_val_fold, y_val_fold = X_train_clean[val_mask], y_train_clean[val_mask]

    # 1. LightGBM
    lgb_params = get_lgbm_params(n_cls=num_classes, seed=42 + fold_idx)
    lgb_clf = lgb.LGBMClassifier(**lgb_params)
    lgb_clf.fit(
        X_train_fold,
        y_train_fold,
        eval_set=[(X_val_fold, y_val_fold)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=40, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    oof_preds_lgb[val_mask] = lgb_clf.predict_proba(X_val_fold)
    test_preds_lgb += lgb_clf.predict_proba(X_test_clean) / N_SPLITS

    # 2. XGBoost
    xgb_params = get_xgboost_params(n_cls=num_classes, seed=42 + fold_idx)
    xgb_clf = xgb.XGBClassifier(**xgb_params, early_stopping_rounds=40)
    xgb_clf.fit(
        X_train_fold, y_train_fold, eval_set=[(X_val_fold, y_val_fold)], verbose=False
    )
    oof_preds_xgb[val_mask] = xgb_clf.predict_proba(X_val_fold)
    test_preds_xgb += xgb_clf.predict_proba(X_test_clean) / N_SPLITS

    # 3. CatBoost
    cat_params = get_catboost_params(n_cls=num_classes, seed=42 + fold_idx)
    cat_clf = CatBoostClassifier(**cat_params, early_stopping_rounds=40)
    cat_clf.fit(
        X_train_fold,
        y_train_fold,
        eval_set=(X_val_fold, y_val_fold),
        verbose=False,
    )
    oof_preds_cat[val_mask] = cat_clf.predict_proba(X_val_fold)
    test_preds_cat += cat_clf.predict_proba(X_test_clean) / N_SPLITS

    # 4. ExtraTreesClassifier
    et_clf = ExtraTreesClassifier(
        n_estimators=300,
        max_depth=16,
        min_samples_split=4,
        random_state=42 + fold_idx,
        n_jobs=-1,
    )
    et_clf.fit(X_train_fold, y_train_fold)
    oof_preds_et[val_mask] = et_clf.predict_proba(X_val_fold)
    test_preds_et += et_clf.predict_proba(X_test_clean) / N_SPLITS

    # 5. RandomForestClassifier
    rf_clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=16,
        min_samples_split=4,
        random_state=42 + fold_idx,
        n_jobs=-1,
    )
    rf_clf.fit(X_train_fold, y_train_fold)
    oof_preds_rf[val_mask] = rf_clf.predict_proba(X_val_fold)
    test_preds_rf += rf_clf.predict_proba(X_test_clean) / N_SPLITS

    # 6. TabularResNet
    val_prob_nn, te_prob_nn = train_neural_net(
        X_train_fold,
        y_train_fold,
        X_val_fold,
        y_val_fold,
        X_test_clean,
        input_dim=n_features,
        n_cls=num_classes,
        epochs=35,
        batch_size=64,
    )
    oof_preds_nn[val_mask] = val_prob_nn
    test_preds_nn += te_prob_nn / N_SPLITS

# -------------------------------------------------------------------------
# 12. Blending, Verification & Submission Creation
# -------------------------------------------------------------------------
model_oofs = [
    oof_preds_lgb,
    oof_preds_xgb,
    oof_preds_cat,
    oof_preds_et,
    oof_preds_rf,
    oof_preds_nn,
]
model_tests = [
    test_preds_lgb,
    test_preds_xgb,
    test_preds_cat,
    test_preds_et,
    test_preds_rf,
    test_preds_nn,
]
n_models = len(model_oofs)


def objective(weights_raw: np.ndarray) -> float:
    w = softmax(weights_raw)
    blended = np.zeros_like(model_oofs[0])
    for i in range(n_models):
        blended += w[i] * model_oofs[i]
    eps = 1e-15
    blended = np.clip(blended, eps, 1.0 - eps)
    blended = blended / blended.sum(axis=1, keepdims=True)
    log_losses = -np.log(blended[np.arange(n_samples), y_train_clean])
    return float(np.mean(log_losses))


init_w = np.zeros(n_models, dtype=np.float64)
opt_res = minimize(objective, init_w, method="L-BFGS-B", options={"maxiter": 500})
optimal_weights = softmax(opt_res.x)

final_oof_probs = np.zeros_like(model_oofs[0])
final_test_probs = np.zeros_like(model_tests[0])

for i in range(n_models):
    final_oof_probs += optimal_weights[i] * model_oofs[i]
    final_test_probs += optimal_weights[i] * model_tests[i]

final_validation_score = accuracy_score(
    y_train_clean, np.argmax(final_oof_probs, axis=1)
)

final_test_pred_indices = np.argmax(final_test_probs, axis=1)
final_test_labels = [target_mapping[idx] for idx in final_test_pred_indices]

submission_df = pd.DataFrame({"id": test_ids, "device_class": final_test_labels})
submission_file_path = os.path.join(submission_dir, "submission.csv")
submission_df.to_csv(submission_file_path, index=False)

print(f"Final Validation Score: {final_validation_score}")
