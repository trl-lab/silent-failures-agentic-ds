import os
import random
import re
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from scipy.optimize import minimize
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")

# Ensure global reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ==============================================================================
# 1. Feature Engineering & Preprocessing Functions
# ==============================================================================
def clean_text(series):
    """Normalize text series by filling NaNs and converting to lowercase string."""
    return series.fillna("").astype(str).str.strip().str.lower()


def extract_date_features(df, date_cols):
    """Extract temporal components and pairwise date differences."""
    df_feat = pd.DataFrame(index=df.index)
    parsed_dates = {}

    for col in date_cols:
        parsed = pd.to_datetime(df[col], errors="coerce")
        parsed_dates[col] = parsed
        df_feat[f"{col}_year"] = parsed.dt.year.fillna(-1).astype(int)
        df_feat[f"{col}_month"] = parsed.dt.month.fillna(-1).astype(int)
        df_feat[f"{col}_day"] = parsed.dt.day.fillna(-1).astype(int)
        df_feat[f"{col}_dayofweek"] = parsed.dt.dayofweek.fillna(-1).astype(int)
        df_feat[f"{col}_quarter"] = parsed.dt.quarter.fillna(-1).astype(int)
        df_feat[f"{col}_dayofyear"] = parsed.dt.dayofyear.fillna(-1).astype(int)
        df_feat[f"{col}_is_null"] = df[col].isnull().astype(int)

    date_col_list = list(parsed_dates.keys())
    for i in range(len(date_col_list)):
        for j in range(i + 1, len(date_col_list)):
            col_a, col_b = date_col_list[i], date_col_list[j]
            diff = (parsed_dates[col_a] - parsed_dates[col_b]).dt.total_seconds() / (
                24 * 3600
            )
            df_feat[f"diff_days_{col_a}_{col_b}"] = diff.fillna(-999).astype(float)
            df_feat[f"log_abs_diff_{col_a}_{col_b}"] = np.log1p(np.abs(diff.fillna(0))).astype(float)

    # FDA regulatory urgency review metrics
    if "center_classification_date" in parsed_dates and "recall_initiation_date" in parsed_dates:
        review_days = (parsed_dates["center_classification_date"] - parsed_dates["recall_initiation_date"]).dt.total_seconds() / (24 * 3600)
        df_feat["regulatory_review_urgency_days"] = review_days.fillna(-999).astype(float)
        df_feat["is_urgent_review_under_7d"] = ((review_days >= 0) & (review_days <= 7)).astype(int)

    if "report_date" in parsed_dates and "recall_initiation_date" in parsed_dates:
        reporting_lag = (parsed_dates["report_date"] - parsed_dates["recall_initiation_date"]).dt.total_seconds() / (24 * 3600)
        df_feat["reporting_lag_days"] = reporting_lag.fillna(-999).astype(float)

    return df_feat


def extract_text_statistics(text_series, prefix):
    """Extract character, word, and lexical statistics from a text series."""
    stats = pd.DataFrame(index=text_series.index)
    raw_str = text_series.fillna("").astype(str)

    stats[f"{prefix}_char_len"] = raw_str.str.len()
    stats[f"{prefix}_word_count"] = raw_str.apply(lambda x: len(x.split()) if x else 0)
    stats[f"{prefix}_upper_count"] = raw_str.apply(
        lambda x: sum(1 for c in x if c.isupper())
    )
    stats[f"{prefix}_digit_count"] = raw_str.apply(
        lambda x: sum(1 for c in x if c.isdigit())
    )
    stats[f"{prefix}_punct_count"] = raw_str.apply(
        lambda x: sum(1 for c in x if c in '.,;:!?/-()[]{}""\'')
    )
    stats[f"{prefix}_avg_word_len"] = stats[f"{prefix}_char_len"] / (
        stats[f"{prefix}_word_count"] + 1e-5
    )
    stats[f"{prefix}_is_empty"] = (stats[f"{prefix}_char_len"] == 0).astype(int)
    return stats


def extract_domain_keywords(text_series, prefix):
    """Extract domain-specific keyword presence and counts for FDA drug recalls."""
    keywords = [
        "contamination",
        "microbial",
        "sterility",
        "sterile",
        "aseptic",
        "burkholderia",
        "ralstonia",
        "pseudomonas",
        "fungal",
        "mold",
        "endotoxin",
        "pyrogen",
        "subpotent",
        "superpotent",
        "potency",
        "particulate",
        "glass",
        "metal",
        "foreign",
        "defect",
        "mislabel",
        "mislabeled",
        "labeling",
        "mix-up",
        "wrong strength",
        "wrong product",
        "cgmp",
        "adulterat",
        "death",
        "injury",
        "adverse",
        "hazard",
        "life-threatening",
        "fatal",
        "severe",
        "temporary",
        "remote",
        "recall",
        "fda",
        "rx",
        "otc",
        "tablet",
        "capsule",
        "injection",
        "injectable",
        "intravenous",
        "parenteral",
        "ophthalmic",
        "eye drop",
        "inhalation",
        "solution",
        "suspension",
        "undeclared",
        "cross-contamination",
        "leak",
        "seal",
        "dissolution",
        "out-of-specification",
        "oos",
        "impurity",
        "degradation",
        "nitrosamine",
        "ndma",
        "temperature",
        "storage",
        "expiration",
    ]

    clean_str = clean_text(text_series)
    kw_df = pd.DataFrame(index=text_series.index)

    total_kw_matches = pd.Series(0, index=text_series.index)
    for kw in keywords:
        has_kw = clean_str.str.contains(r"\b" + re.escape(kw), regex=True).astype(int)
        kw_df[f"{prefix}_has_{kw}"] = has_kw
        total_kw_matches += has_kw

    kw_df[f"{prefix}_total_domain_keywords"] = total_kw_matches
    return kw_df


def extract_quantity_features(series):
    """Extract numeric quantity, units, and packaging indicators from product_quantity."""
    clean_s = clean_text(series)
    feats = pd.DataFrame(index=series.index)

    def parse_numeric(text):
        nums = re.findall(r"[\d,]+(?:\.\d+)?", text.replace(",", ""))
        if nums:
            try:
                return float(nums[0])
            except ValueError:
                return -1.0
        return -1.0

    feats["qty_numeric_val"] = clean_s.apply(parse_numeric)
    feats["qty_has_bottles"] = clean_s.str.contains("bottle", regex=False).astype(int)
    feats["qty_has_vials"] = clean_s.str.contains("vial", regex=False).astype(int)
    feats["qty_has_tablets"] = clean_s.str.contains(
        "tablet|capsule|cap|tab", regex=True
    ).astype(int)
    feats["qty_has_cartons"] = clean_s.str.contains(
        "carton|box|pack|case", regex=True
    ).astype(int)
    feats["qty_has_units"] = clean_s.str.contains("unit|lot|batch", regex=True).astype(
        int
    )
    feats["qty_is_nationwide"] = clean_s.str.contains(
        "nationwide|us|all states", regex=True
    ).astype(int)
    return feats


# ==============================================================================
# 2. Deep Tabular Neural Network Architecture
# ==============================================================================
class ResidualTabularBlock(nn.Module):
    """Residual Tabular Block with LayerNorm, Dropout, and GELU activation."""

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


class DrugRecallTabularNet(nn.Module):
    """Multi-Layer Residual Tabular Neural Network for FDA Recall Classification."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 3,
        hidden_dims: list = [256, 128, 64],
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, hidden_dims[0]),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )

        layers = []
        for i in range(len(hidden_dims) - 1):
            layers.append(
                ResidualTabularBlock(
                    hidden_dim=hidden_dims[i], dropout_rate=dropout_rate
                )
            )
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout_rate))

        layers.append(
            ResidualTabularBlock(hidden_dim=hidden_dims[-1], dropout_rate=dropout_rate)
        )
        self.hidden_layers = nn.Sequential(*layers)

        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dims[-1]), nn.Linear(hidden_dims[-1], num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(x)
        x = self.hidden_layers(x)
        return self.head(x)


def create_nn_criterion(
    num_classes: int = 3, label_smoothing: float = 0.05
) -> nn.Module:
    return nn.CrossEntropyLoss(label_smoothing=label_smoothing)


def create_nn_optimizer_and_scheduler(
    model: nn.Module,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    epochs: int = 40,
    steps_per_epoch: int = 10,
):
    optimizer = optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay, eps=1e-8
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs * steps_per_epoch), eta_min=1e-6
    )
    return optimizer, scheduler


# ==============================================================================
# 3. GBDT Model Definitions
# ==============================================================================
def get_lightgbm_model(
    num_classes: int = 3, random_state: int = 42
) -> lgb.LGBMClassifier:
    params = {
        "objective": "multiclass",
        "num_class": num_classes,
        "metric": "multi_logloss",
        "boosting_type": "gbdt",
        "n_estimators": 1200,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": 6,
        "min_child_samples": 25,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.7,
        "reg_alpha": 0.5,
        "reg_lambda": 2.0,
        "random_state": random_state,
        "n_jobs": -1,
        "verbosity": -1,
    }
    return lgb.LGBMClassifier(**params)


def get_xgboost_model(
    num_classes: int = 3, random_state: int = 42
) -> xgb.XGBClassifier:
    params = {
        "objective": "multi:softprob",
        "num_class": num_classes,
        "eval_metric": "mlogloss",
        "n_estimators": 1200,
        "learning_rate": 0.03,
        "max_depth": 5,
        "min_child_weight": 4,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "reg_alpha": 0.5,
        "reg_lambda": 2.5,
        "random_state": random_state,
        "n_jobs": -1,
        "tree_method": "hist",
    }
    return xgb.XGBClassifier(**params)


def get_catboost_model(
    num_classes: int = 3, random_state: int = 42
) -> CatBoostClassifier:
    params = {
        "loss_function": "MultiClass",
        "iterations": 1000,
        "learning_rate": 0.03,
        "depth": 6,
        "l2_leaf_reg": 3.0,
        "random_seed": random_state,
        "verbose": False,
        "thread_count": -1,
    }
    return CatBoostClassifier(**params)


# ==============================================================================
# 4. Pipeline Execution
# ==============================================================================
def main():
    print("Loading raw parquet data from ./input/...")
    train_df = pd.read_parquet("./input/train.parquet")
    test_df = pd.read_parquet("./input/test.parquet")

    target_col = "classification"
    if target_col not in train_df.columns:
        raise ValueError(f"Target column '{target_col}' not found in train.parquet")

    y_raw = train_df[target_col].copy()
    X_train_raw = train_df.drop(columns=[target_col]).copy()
    X_test_raw = test_df.copy()

    # Target encoding
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_raw.astype(str))
    target_classes = le.classes_
    num_classes = len(target_classes)
    print(f"Target classes mapped: {dict(zip(target_classes, range(num_classes)))}")

    all_cols = list(X_train_raw.columns)
    date_cols = [
        c
        for c in all_cols
        if "date" in c.lower() or pd.api.types.is_datetime64_any_dtype(X_train_raw[c])
    ]
    text_cols_priority = [
        "reason_for_recall",
        "product_description",
        "code_info",
        "distribution_pattern",
        "recalling_firm",
    ]
    text_cols = [c for c in text_cols_priority if c in all_cols]

    other_str_cols = [
        c
        for c in all_cols
        if c not in date_cols
        and c not in text_cols
        and (
            X_train_raw[c].dtype == "object"
            or isinstance(X_train_raw[c].dtype, pd.CategoricalDtype)
        )
    ]

    numeric_cols = [
        c
        for c in all_cols
        if c not in date_cols
        and c not in text_cols
        and c not in other_str_cols
        and pd.api.types.is_numeric_dtype(X_train_raw[c])
    ]

    train_feature_blocks = []
    test_feature_blocks = []

    # 1. Date Features
    if date_cols:
        train_date_feat = extract_date_features(X_train_raw, date_cols)
        test_date_feat = extract_date_features(X_test_raw, date_cols)
        train_feature_blocks.append(train_date_feat)
        test_feature_blocks.append(test_date_feat)

    # 2. Text Statistics & Domain Keywords
    for t_col in text_cols:
        train_stats = extract_text_statistics(X_train_raw[t_col], prefix=t_col)
        test_stats = extract_text_statistics(X_test_raw[t_col], prefix=t_col)
        train_feature_blocks.append(train_stats)
        test_feature_blocks.append(test_stats)

        train_kw = extract_domain_keywords(X_train_raw[t_col], prefix=t_col)
        test_kw = extract_domain_keywords(X_test_raw[t_col], prefix=t_col)
        train_feature_blocks.append(train_kw)
        test_feature_blocks.append(test_kw)

    # 3. Product Quantity Features
    if "product_quantity" in all_cols:
        train_qty_feat = extract_quantity_features(X_train_raw["product_quantity"])
        test_qty_feat = extract_quantity_features(X_test_raw["product_quantity"])
        train_feature_blocks.append(train_qty_feat)
        test_feature_blocks.append(test_qty_feat)

    # 4. TF-IDF + SVD Semantics (Strictly fit on train)
    for t_col in ["reason_for_recall", "product_description"]:
        if t_col in all_cols:
            tfidf = TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                stop_words="english",
                sublinear_tf=True,
                min_df=2,
            )
            train_text = clean_text(X_train_raw[t_col])
            test_text = clean_text(X_test_raw[t_col])

            tfidf_train = tfidf.fit_transform(train_text)
            tfidf_test = tfidf.transform(test_text)

            n_components = (
                min(25, tfidf_train.shape[1] - 1)
                if tfidf_train.shape[1] > 25
                else max(1, tfidf_train.shape[1] - 1)
            )
            svd = TruncatedSVD(n_components=n_components, random_state=42)
            svd_train = svd.fit_transform(tfidf_train)
            svd_test = svd.transform(tfidf_test)

            svd_cols = [f"{t_col}_svd_{k}" for k in range(n_components)]
            train_feature_blocks.append(
                pd.DataFrame(svd_train, columns=svd_cols, index=X_train_raw.index)
            )
            test_feature_blocks.append(
                pd.DataFrame(svd_test, columns=svd_cols, index=X_test_raw.index)
            )

    # 5. Categorical Frequency & Binary Missing Indicators
    cat_cols_to_encode = [c for c in other_str_cols + text_cols if c in all_cols]
    for c in cat_cols_to_encode:
        train_counts = X_train_raw[c].fillna("__MISSING__").value_counts()
        train_freq = (
            X_train_raw[c]
            .fillna("__MISSING__")
            .map(train_counts)
            .fillna(0)
            .astype(float)
        )
        test_freq = (
            X_test_raw[c]
            .fillna("__MISSING__")
            .map(train_counts)
            .fillna(0)
            .astype(float)
        )

        train_feature_blocks.append(
            pd.DataFrame({f"{c}_freq_count": train_freq}, index=X_train_raw.index)
        )
        test_feature_blocks.append(
            pd.DataFrame({f"{c}_freq_count": test_freq}, index=X_test_raw.index)
        )

        train_feature_blocks.append(
            pd.DataFrame(
                {f"{c}_is_na": X_train_raw[c].isnull().astype(int)},
                index=X_train_raw.index,
            )
        )
        test_feature_blocks.append(
            pd.DataFrame(
                {f"{c}_is_na": X_test_raw[c].isnull().astype(int)},
                index=X_test_raw.index,
            )
        )

    # 6. Numeric features
    if numeric_cols:
        num_train = X_train_raw[numeric_cols].fillna(-999).astype(float)
        num_test = X_test_raw[numeric_cols].fillna(-999).astype(float)
        train_feature_blocks.append(num_train)
        test_feature_blocks.append(num_test)

    # Combine all engineered blocks
    X_train_df = pd.concat(train_feature_blocks, axis=1)
    X_test_df = pd.concat(test_feature_blocks, axis=1)

    common_cols = [col for col in X_train_df.columns if col in X_test_df.columns]
    X_train_df = X_train_df[common_cols]
    X_test_df = X_test_df[common_cols]

    cleaned_col_names = [re.sub(r"[\[\]<>,:\s]+", "_", col) for col in common_cols]
    X_train_df.columns = cleaned_col_names
    X_test_df.columns = cleaned_col_names

    X_train_df = X_train_df.loc[:, ~X_train_df.columns.duplicated()]
    X_test_df = X_test_df.loc[:, ~X_test_df.columns.duplicated()]

    X_train = np.nan_to_num(X_train_df.values, nan=0.0, posinf=1e5, neginf=-1e5)
    X_test = np.nan_to_num(X_test_df.values, nan=0.0, posinf=1e5, neginf=-1e5)
    n_features = X_train.shape[1]

    # Stratified K-Fold setup
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    fold_splits = list(skf.split(X_train, y_encoded))

    # -------------------------------------------------------------------------
    # Leak-Free Out-of-Fold Linear Text Meta-Features & Target Encoding
    # -------------------------------------------------------------------------
    text_meta_sources = ["reason_for_recall", "product_description"]
    combined_train_text = clean_text(X_train_raw["reason_for_recall"] + " " + X_train_raw["product_description"] + " " + X_train_raw.get("code_info", ""))
    combined_test_text = clean_text(X_test_raw["reason_for_recall"] + " " + X_test_raw["product_description"] + " " + X_test_raw.get("code_info", ""))

    text_sources_dict = {
        "reason_for_recall": (clean_text(X_train_raw["reason_for_recall"]), clean_text(X_test_raw["reason_for_recall"])),
        "product_description": (clean_text(X_train_raw["product_description"]), clean_text(X_test_raw["product_description"])),
        "combined_text": (combined_train_text, combined_test_text)
    }

    te_cats = [c for c in ["recalling_firm", "initial_firm_notification", "voluntary_mandated", "status", "distribution_pattern"] if c in all_cols]

    oof_meta_features = np.zeros((len(X_train), len(text_sources_dict) * num_classes + len(te_cats) * num_classes), dtype=float)
    test_meta_features = np.zeros((len(X_test), len(text_sources_dict) * num_classes + len(te_cats) * num_classes), dtype=float)

    for fold, (train_idx, val_idx) in enumerate(fold_splits):
        col_offset = 0
        y_tr_fold = y_encoded[train_idx]

        # 1. High-dim TF-IDF + LogisticRegression linear meta-features
        for src_name, (tr_full_s, te_full_s) in text_sources_dict.items():
            tfidf_meta = TfidfVectorizer(
                ngram_range=(1, 3),
                max_features=20000,
                sublinear_tf=True,
                min_df=2
            )
            X_tr_tfidf = tfidf_meta.fit_transform(tr_full_s.iloc[train_idx])
            X_va_tfidf = tfidf_meta.transform(tr_full_s.iloc[val_idx])
            X_te_tfidf = tfidf_meta.transform(te_full_s)

            linear_clf = LogisticRegression(C=1.5, max_iter=500, solver="lbfgs", multi_class="multinomial", random_state=SEED)
            linear_clf.fit(X_tr_tfidf, y_tr_fold)

            oof_meta_features[val_idx, col_offset:col_offset + num_classes] = linear_clf.predict_proba(X_va_tfidf)
            test_meta_features[:, col_offset:col_offset + num_classes] += linear_clf.predict_proba(X_te_tfidf) / n_splits
            col_offset += num_classes

        # 2. Smoothed out-of-fold target encoding for categoricals
        m_smooth = 10.0
        global_priors = np.array([np.mean(y_tr_fold == c) for c in range(num_classes)])

        for cat_col in te_cats:
            tr_cat = X_train_raw[cat_col].fillna("__MISSING__").iloc[train_idx]
            va_cat = X_train_raw[cat_col].fillna("__MISSING__").iloc[val_idx]
            te_cat = X_test_raw[cat_col].fillna("__MISSING__")

            cat_counts = tr_cat.value_counts()
            for cls_idx in range(num_classes):
                cls_counts = tr_cat[y_tr_fold == cls_idx].value_counts()
                smooth_map = (cls_counts + m_smooth * global_priors[cls_idx]) / (cat_counts + m_smooth)

                oof_meta_features[val_idx, col_offset + cls_idx] = va_cat.map(smooth_map).fillna(global_priors[cls_idx]).values
                test_meta_features[:, col_offset + cls_idx] += te_cat.map(smooth_map).fillna(global_priors[cls_idx]).values / n_splits
            col_offset += num_classes

    # Concatenate base features with engineered meta-features
    X_train_full = np.hstack([X_train, oof_meta_features])
    X_test_full = np.hstack([X_test, test_meta_features])
    n_features_full = X_train_full.shape[1]

    oof_lgb = np.zeros((len(X_train), num_classes), dtype=float)
    oof_xgb = np.zeros((len(X_train), num_classes), dtype=float)
    oof_cb = np.zeros((len(X_train), num_classes), dtype=float)
    oof_nn = np.zeros((len(X_train), num_classes), dtype=float)

    test_preds_lgb = np.zeros((len(X_test), num_classes), dtype=float)
    test_preds_xgb = np.zeros((len(X_test), num_classes), dtype=float)
    test_preds_cb = np.zeros((len(X_test), num_classes), dtype=float)
    test_preds_nn = np.zeros((len(X_test), num_classes), dtype=float)

    # -------------------------------------------------------------------------
    # Train Models across folds
    # -------------------------------------------------------------------------
    for fold, (train_idx, val_idx) in enumerate(fold_splits):
        X_tr, y_tr = X_train_full[train_idx], y_encoded[train_idx]
        X_va, y_va = X_train_full[val_idx], y_encoded[val_idx]

        # 1. LightGBM
        model_lgb = get_lightgbm_model(num_classes=num_classes, random_state=SEED + fold)
        callbacks = [
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0),
        ]
        model_lgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_names=["val"], callbacks=callbacks)
        oof_lgb[val_idx] = model_lgb.predict_proba(X_va)
        test_preds_lgb += model_lgb.predict_proba(X_test_full) / n_splits

        # 2. XGBoost
        model_xgb = get_xgboost_model(num_classes=num_classes, random_state=SEED + fold)
        model_xgb.set_params(early_stopping_rounds=50)
        model_xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        oof_xgb[val_idx] = model_xgb.predict_proba(X_va)
        test_preds_xgb += model_xgb.predict_proba(X_test_full) / n_splits

        # 3. CatBoost
        model_cb = get_catboost_model(num_classes=num_classes, random_state=SEED + fold)
        model_cb.fit(X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=50, verbose=False)
        oof_cb[val_idx] = model_cb.predict_proba(X_va)
        test_preds_cb += model_cb.predict_proba(X_test_full) / n_splits

        # 4. Tabular Residual Deep Neural Network
        set_seed(SEED + fold)
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_va_s = scaler.transform(X_va)
        X_te_s = scaler.transform(X_test_full)

        train_dataset = TensorDataset(torch.tensor(X_tr_s, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.long))
        val_dataset = TensorDataset(torch.tensor(X_va_s, dtype=torch.float32), torch.tensor(y_va, dtype=torch.long))
        test_tensor = torch.tensor(X_te_s, dtype=torch.float32).to(device)

        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, drop_last=False)
        val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

        model_nn = DrugRecallTabularNet(
            input_dim=n_features_full,
            num_classes=num_classes,
            hidden_dims=[256, 128, 64],
            dropout_rate=0.25,
        ).to(device)

        criterion = create_nn_criterion(num_classes=num_classes, label_smoothing=0.05)
        optimizer, scheduler = create_nn_optimizer_and_scheduler(
            model_nn, lr=1e-3, weight_decay=1e-4, epochs=30, steps_per_epoch=len(train_loader)
        )

        best_val_loss = float("inf")
        best_val_probs = None
        best_model_state = None

        for epoch in range(30):
            model_nn.train()
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                logits = model_nn(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                nn.utils.clip_grad_norm_(model_nn.parameters(), max_norm=2.0)
                optimizer.step()
                scheduler.step()

            model_nn.eval()
            val_loss = 0.0
            val_preds_list = []
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    logits = model_nn(batch_x)
                    loss = criterion(logits, batch_y)
                    val_loss += loss.item() * batch_x.size(0)
                    probs = torch.softmax(logits, dim=-1).cpu().numpy()
                    val_preds_list.append(probs)

            val_loss /= len(val_dataset)
            current_val_probs = np.vstack(val_preds_list)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_probs = current_val_probs
                best_model_state = model_nn.state_dict().copy()

        oof_nn[val_idx] = best_val_probs
        model_nn.load_state_dict(best_model_state)
        model_nn.eval()
        with torch.no_grad():
            test_logits = model_nn(test_tensor)
            test_probs = torch.softmax(test_logits, dim=-1).cpu().numpy()
            test_preds_nn += test_probs / n_splits

    # -------------------------------------------------------------------------
    # Continuous Simplex Optimization for Multi-Model Ensemble Blend
    # -------------------------------------------------------------------------
    def objective_func(weights):
        w = np.maximum(0.0, np.array(weights))
        if w.sum() == 0:
            return 1e5
        w = w / w.sum()
        blend = w[0] * oof_lgb + w[1] * oof_xgb + w[2] * oof_cb + w[3] * oof_nn
        blend = np.clip(blend, 1e-15, 1.0 - 1e-15)
        blend = blend / blend.sum(axis=1, keepdims=True)
        return log_loss(y_encoded, blend)

    bounds = [(0.0, 1.0) for _ in range(4)]
    constraints = {"type": "eq", "fun": lambda w: 1.0 - np.sum(w)}

    best_weights = np.array([0.25, 0.25, 0.25, 0.25])
    best_val_accuracy = accuracy_score(y_encoded, np.argmax(0.25 * oof_lgb + 0.25 * oof_xgb + 0.25 * oof_cb + 0.25 * oof_nn, axis=1))

    for init_w in [
        [0.25, 0.25, 0.25, 0.25],
        [0.35, 0.35, 0.20, 0.10],
        [0.30, 0.30, 0.30, 0.10],
        [0.40, 0.25, 0.25, 0.10],
        [0.20, 0.40, 0.30, 0.10],
    ]:
        res = minimize(objective_func, init_w, method="SLSQP", bounds=bounds, constraints=constraints)
        if res.success:
            w = np.maximum(0.0, res.x)
            w = w / w.sum()
            blend = w[0] * oof_lgb + w[1] * oof_xgb + w[2] * oof_cb + w[3] * oof_nn
            acc = accuracy_score(y_encoded, np.argmax(blend, axis=1))
            if acc > best_val_accuracy:
                best_val_accuracy = acc
                best_weights = w

    # Direct accuracy fine search around optimal neighborhood
    np.random.seed(SEED)
    for _ in range(300):
        pert = np.random.dirichlet([best_weights[0] * 20 + 1, best_weights[1] * 20 + 1, best_weights[2] * 20 + 1, best_weights[3] * 20 + 1])
        blend = pert[0] * oof_lgb + pert[1] * oof_xgb + pert[2] * oof_cb + pert[3] * oof_nn
        acc = accuracy_score(y_encoded, np.argmax(blend, axis=1))
        if acc > best_val_accuracy:
            best_val_accuracy = acc
            best_weights = pert

    w_lgb, w_xgb, w_cb, w_nn = best_weights

    # -------------------------------------------------------------------------
    # Test Inference & Submission Generation
    # -------------------------------------------------------------------------
    final_test_probs = (
        w_lgb * test_preds_lgb + w_xgb * test_preds_xgb + w_cb * test_preds_cb + w_nn * test_preds_nn
    )
    final_test_classes_idx = np.argmax(final_test_probs, axis=1)
    final_test_predictions = target_classes[final_test_classes_idx]

    os.makedirs("./submission", exist_ok=True)
    submission_df = pd.DataFrame(
        {
            "id": np.arange(len(final_test_predictions)),
            "classification": final_test_predictions,
        }
    )
    submission_df.to_csv("./submission/submission.csv", index=False)

    print(f"Final Validation Score: {best_val_accuracy}")


if __name__ == "__main__":
    main()