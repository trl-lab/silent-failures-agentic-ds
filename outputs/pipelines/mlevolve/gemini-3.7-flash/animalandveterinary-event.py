import gc
import json
import math
import os
import warnings
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from scipy.optimize import minimize
from scipy.special import softmax
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

from catboost import CatBoostClassifier
import lightgbm as lgb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb

warnings.filterwarnings("ignore")


# =========================================================================
# 1. Utilities and Target Normalization
# =========================================================================


def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_directory(path: str):
    os.makedirs(path, exist_ok=True)


def normalize_target(df: pd.DataFrame, target_col: str = "serious_ae") -> pd.DataFrame:
    """
    Standardizes the target column into binary {0, 1} format.
    Handles Boolean, string ('Y'/'N', '1'/'2', 'true'/'false'), and integer mappings.
    """
    if target_col not in df.columns:
        return df

    target = df[target_col]
    if pd.api.types.is_bool_dtype(target):
        df[target_col] = target.astype(int)
    elif pd.api.types.is_numeric_dtype(target):
        unique_vals = set(target.dropna().unique())
        if unique_vals == {1, 2}:
            df[target_col] = target.map({1: 1, 2: 0}).astype(int)
        else:
            min_val = target.min()
            df[target_col] = (target > min_val).astype(int)
    else:
        str_map = {
            "1": 1,
            "0": 0,
            "1.0": 1,
            "0.0": 0,
            "y": 1,
            "n": 0,
            "yes": 1,
            "no": 0,
            "true": 1,
            "false": 0,
            "t": 1,
            "f": 0,
            "serious": 1,
            "non-serious": 0,
            "nonserious": 0,
        }
        df[target_col] = (
            target.astype(str)
            .str.strip()
            .str.lower()
            .map(str_map)
            .fillna(0)
            .astype(int)
        )

    return df


# =========================================================================
# 2. Feature Engineering Pipeline
# =========================================================================


def identify_column_types(df: pd.DataFrame, target_col: str = "serious_ae"):
    """
    Dynamically identifies numerical, categorical, datetime, and text column types.
    """
    feature_cols = [c for c in df.columns if c != target_col]
    numeric_cols = []
    categorical_cols = []
    datetime_cols = []
    text_cols = []

    for col in feature_cols:
        col_series = df[col]
        if pd.api.types.is_datetime64_any_dtype(col_series):
            datetime_cols.append(col)
            continue

        if pd.api.types.is_numeric_dtype(col_series):
            numeric_cols.append(col)
            continue

        non_null_samples = col_series.dropna().astype(str)
        if len(non_null_samples) == 0:
            categorical_cols.append(col)
            continue

        sample_subset = non_null_samples.iloc[: min(50, len(non_null_samples))]
        date_parse_success = False
        try:
            parsed = pd.to_datetime(sample_subset, errors="coerce")
            if parsed.notnull().mean() > 0.85:
                date_parse_success = True
        except Exception:
            pass

        if date_parse_success:
            datetime_cols.append(col)
            continue

        avg_word_count = non_null_samples.apply(lambda x: len(x.split())).mean()
        avg_char_length = non_null_samples.apply(len).mean()

        if avg_word_count > 4.0 or avg_char_length > 40.0:
            text_cols.append(col)
        else:
            categorical_cols.append(col)

    return {
        "numeric": numeric_cols,
        "categorical": categorical_cols,
        "datetime": datetime_cols,
        "text": text_cols,
    }


def process_datetime_features(
    train_df: pd.DataFrame, test_df: pd.DataFrame, dt_cols: list
):
    for col in dt_cols:
        tr_dt = pd.to_datetime(train_df[col], errors="coerce")
        te_dt = pd.to_datetime(test_df[col], errors="coerce")

        ref_date = tr_dt.min()
        if pd.isnull(ref_date):
            ref_date = pd.Timestamp("2000-01-01")

        for name, dt in [("train", tr_dt), ("test", te_dt)]:
            target_df = train_df if name == "train" else test_df
            target_df[f"{col}_year"] = dt.dt.year.fillna(-1).astype(float)
            target_df[f"{col}_month"] = dt.dt.month.fillna(-1).astype(float)
            target_df[f"{col}_day"] = dt.dt.day.fillna(-1).astype(float)
            target_df[f"{col}_dayofweek"] = dt.dt.dayofweek.fillna(-1).astype(float)
            target_df[f"{col}_dayofyear"] = dt.dt.dayofyear.fillna(-1).astype(float)
            target_df[f"{col}_quarter"] = dt.dt.quarter.fillna(-1).astype(float)
            target_df[f"{col}_is_weekend"] = dt.dt.dayofweek.isin([5, 6]).astype(float)
            target_df[f"{col}_elapsed_days"] = (
                (dt - ref_date).dt.total_seconds() / 86400.0
            ).fillna(-1)

        train_df.drop(columns=[col], inplace=True, errors="ignore")
        test_df.drop(columns=[col], inplace=True, errors="ignore")

    return train_df, test_df


def process_text_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    text_cols: list,
    target_col: str = "serious_ae",
    n_components: int = 20,
):
    serious_keywords = [
        "death", "died", "fatal", "euthanasia", "severe", "hospital", "critical",
        "emergency", "acute", "shock", "failure", "bleed", "hemorrhage", "seizure",
        "collapse", "toxicity", "paralysis", "coma", "arrest", "necrosis",
        "cardiac", "respiratory", "dead", "unresponsive", "killed"
    ]
    mild_keywords = [
        "mild", "recovered", "resolved", "normal", "slight", "transient",
        "minor", "uneventful", "improved", "healthy"
    ]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for col in text_cols:
        tr_series = train_df[col].fillna("").astype(str)
        te_series = test_df[col].fillna("").astype(str)

        for name, series in [("train", tr_series), ("test", te_series)]:
            target_df = train_df if name == "train" else test_df
            lower_series = series.str.lower()
            target_df[f"{col}_char_len"] = series.apply(len).astype(float)
            target_df[f"{col}_word_count"] = series.apply(
                lambda x: len(x.split())
            ).astype(float)
            target_df[f"{col}_digit_count"] = series.apply(
                lambda x: sum(c.isdigit() for c in x)
            ).astype(float)
            target_df[f"{col}_upper_count"] = series.apply(
                lambda x: sum(c.isupper() for c in x)
            ).astype(float)
            target_df[f"{col}_punct_count"] = series.apply(
                lambda x: sum(not c.isalnum() and not c.isspace() for c in x)
            ).astype(float)

            serious_hits = lower_series.apply(
                lambda text: sum(kw in text for kw in serious_keywords)
            ).astype(float)
            mild_hits = lower_series.apply(
                lambda text: sum(kw in text for kw in mild_keywords)
            ).astype(float)
            target_df[f"{col}_serious_kw_count"] = serious_hits
            target_df[f"{col}_mild_kw_count"] = mild_hits
            target_df[f"{col}_kw_ratio"] = (serious_hits + 0.1) / (mild_hits + 0.1)

        tfidf = TfidfVectorizer(
            max_features=5000, ngram_range=(1, 3), sublinear_tf=True, min_df=2
        )
        tr_tfidf = tfidf.fit_transform(tr_series)
        te_tfidf = tfidf.transform(te_series)

        n_comp = min(n_components, max(2, tr_tfidf.shape[1] - 1))
        svd = TruncatedSVD(n_components=n_comp, random_state=42)
        tr_svd = svd.fit_transform(tr_tfidf)
        te_svd = svd.transform(te_tfidf)

        for comp_idx in range(n_comp):
            train_df[f"{col}_svd_{comp_idx}"] = tr_svd[:, comp_idx]
            test_df[f"{col}_svd_{comp_idx}"] = te_svd[:, comp_idx]

        if target_col in train_df.columns:
            y = train_df[target_col].values
            oof_text_ridge = np.zeros(len(train_df))
            test_text_ridge = np.zeros(len(test_df))

            for tr_idx, val_idx in skf.split(train_df, y):
                clf = RidgeClassifier(alpha=1.0, random_state=42)
                clf.fit(tr_tfidf[tr_idx], y[tr_idx])
                oof_text_ridge[val_idx] = clf.decision_function(tr_tfidf[val_idx])
                test_text_ridge += clf.decision_function(te_tfidf) / 5.0

            train_df[f"{col}_ridge_serious_score"] = oof_text_ridge
            test_df[f"{col}_ridge_serious_score"] = test_text_ridge

        train_df.drop(columns=[col], inplace=True, errors="ignore")
        test_df.drop(columns=[col], inplace=True, errors="ignore")

    return train_df, test_df


def apply_out_of_fold_target_encoding(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cat_cols: list,
    target_col: str,
    n_splits: int = 5,
    smoothing: float = 15.0,
):
    global_mean = train_df[target_col].mean()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    for col in cat_cols:
        oof_te = np.zeros(len(train_df))
        test_te = np.zeros(len(test_df))

        for train_idx, val_idx in skf.split(train_df, train_df[target_col]):
            tr_fold = train_df.iloc[train_idx]
            val_fold = train_df.iloc[val_idx]

            stats = tr_fold.groupby(col)[target_col].agg(["count", "mean"])
            counts = stats["count"]
            means = stats["mean"]
            smoothed = (counts * means + smoothing * global_mean) / (counts + smoothing)

            oof_te[val_idx] = val_fold[col].map(smoothed).fillna(global_mean).values
            test_te += test_df[col].map(smoothed).fillna(global_mean).values / n_splits

        train_df[f"{col}_target_enc"] = oof_te
        test_df[f"{col}_target_enc"] = test_te

    return train_df, test_df


def compute_group_aggregations(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cat_cols: list,
    num_cols: list,
    top_k_cats: int = 3,
):
    selected_cats = cat_cols[: min(top_k_cats, len(cat_cols))]
    selected_nums = num_cols[: min(4, len(num_cols))]

    for cat in selected_cats:
        for num in selected_nums:
            group_stats = (
                train_df.groupby(cat)[num]
                .agg(["mean", "std", "min", "max", "median"])
                .reset_index()
            )
            group_stats.columns = [
                cat,
                f"{cat}_{num}_grp_mean",
                f"{cat}_{num}_grp_std",
                f"{cat}_{num}_grp_min",
                f"{cat}_{num}_grp_max",
                f"{cat}_{num}_grp_median",
            ]

            train_df = train_df.merge(group_stats, on=cat, how="left")
            test_df = test_df.merge(group_stats, on=cat, how="left")

            train_df[f"{num}_diff_from_{cat}_mean"] = (
                train_df[num] - train_df[f"{cat}_{num}_grp_mean"]
            )
            test_df[f"{num}_diff_from_{cat}_mean"] = (
                test_df[num] - test_df[f"{cat}_{num}_grp_mean"]
            )
            train_df[f"{num}_diff_from_{cat}_median"] = (
                train_df[num] - train_df[f"{cat}_{num}_grp_median"]
            )
            test_df[f"{num}_diff_from_{cat}_median"] = (
                test_df[num] - test_df[f"{cat}_{num}_grp_median"]
            )

    return train_df, test_df


def build_features(train_path: str, test_path: str, target_col: str = "serious_ae"):
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    if target_col in train_df.columns:
        train_df = normalize_target(train_df, target_col=target_col)

    col_types = identify_column_types(train_df, target_col=target_col)

    if col_types["datetime"]:
        train_df, test_df = process_datetime_features(
            train_df, test_df, col_types["datetime"]
        )

    if col_types["text"]:
        train_df, test_df = process_text_features(
            train_df, test_df, col_types["text"], target_col=target_col, n_components=20
        )

    base_cat_cols = [
        c
        for c in col_types["categorical"]
        if c in train_df.columns and c in test_df.columns
    ]

    cross_cat_cols = []
    top_cats_for_cross = base_cat_cols[: min(4, len(base_cat_cols))]
    for i in range(len(top_cats_for_cross)):
        for j in range(i + 1, len(top_cats_for_cross)):
            c1, c2 = top_cats_for_cross[i], top_cats_for_cross[j]
            cross_col = f"{c1}_x_{c2}"
            train_df[cross_col] = (
                train_df[c1].astype(str) + "_" + train_df[c2].astype(str)
            )
            test_df[cross_col] = (
                test_df[c1].astype(str) + "_" + test_df[c2].astype(str)
            )
            cross_cat_cols.append(cross_col)

    cat_cols_to_encode = base_cat_cols + cross_cat_cols

    if target_col in train_df.columns:
        train_df, test_df = apply_out_of_fold_target_encoding(
            train_df,
            test_df,
            cat_cols=cat_cols_to_encode,
            target_col=target_col,
            n_splits=5,
            smoothing=15.0,
        )

    for col in cat_cols_to_encode:
        freq_map = train_df[col].value_counts(normalize=True).to_dict()
        train_df[f"{col}_freq"] = train_df[col].map(freq_map).fillna(0.0).astype(float)
        test_df[f"{col}_freq"] = test_df[col].map(freq_map).fillna(0.0).astype(float)

        le = LabelEncoder()
        train_df[col] = train_df[col].fillna("__MISSING__").astype(str)
        test_df[col] = test_df[col].fillna("__MISSING__").astype(str)

        train_df[f"{col}_encoded"] = le.fit_transform(train_df[col])
        test_mapping = {val: idx for idx, val in enumerate(le.classes_)}
        test_df[f"{col}_encoded"] = (
            test_df[col].map(test_mapping).fillna(-1).astype(int)
        )

    numeric_cols_current = [
        c for c in col_types["numeric"] if c in train_df.columns and c != target_col
    ]
    for col in numeric_cols_current:
        train_df[f"{col}_isnan"] = train_df[col].isnull().astype(float)
        test_df[f"{col}_isnan"] = test_df[col].isnull().astype(float)

        median_val = train_df[col].median()
        train_df[col] = train_df[col].fillna(median_val)
        test_df[col] = test_df[col].fillna(median_val)

        if (train_df[col] >= 0).all() and (train_df[col].skew() > 1.5):
            train_df[f"{col}_log1p"] = np.log1p(train_df[col])
            test_df[f"{col}_log1p"] = np.log1p(test_df[col])

    if len(numeric_cols_current) >= 2:
        top_nums = numeric_cols_current[: min(4, len(numeric_cols_current))]
        for i in range(len(top_nums)):
            for j in range(i + 1, len(top_nums)):
                c1, c2 = top_nums[i], top_nums[j]
                train_df[f"{c1}_mult_{c2}"] = train_df[c1] * train_df[c2]
                test_df[f"{c1}_mult_{c2}"] = test_df[c1] * test_df[c2]
                train_df[f"{c1}_div_{c2}"] = train_df[c1] / (train_df[c2].abs() + 1e-5)
                test_df[f"{c1}_div_{c2}"] = test_df[c1] / (test_df[c2].abs() + 1e-5)

    if cat_cols_to_encode and numeric_cols_current:
        train_df, test_df = compute_group_aggregations(
            train_df,
            test_df,
            cat_cols=cat_cols_to_encode,
            num_cols=numeric_cols_current,
            top_k_cats=2,
        )

    cols_to_drop = [c for c in cat_cols_to_encode if c in train_df.columns]
    train_df.drop(columns=cols_to_drop, inplace=True, errors="ignore")
    test_df.drop(columns=cols_to_drop, inplace=True, errors="ignore")

    feature_cols = [
        c for c in train_df.columns if c != target_col and c in test_df.columns
    ]

    for col in feature_cols:
        med_val = train_df[col].replace([np.inf, -np.inf], np.nan).median()
        train_df[col] = train_df[col].replace([np.inf, -np.inf], np.nan).fillna(med_val)
        test_df[col] = test_df[col].replace([np.inf, -np.inf], np.nan).fillna(med_val)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_assignments = np.zeros(len(train_df), dtype=int)
    for fold, (_, val_idx) in enumerate(skf.split(train_df, train_df[target_col])):
        fold_assignments[val_idx] = fold

    train_df["cv_fold"] = fold_assignments

    return train_df, test_df, feature_cols, target_col


# =========================================================================
# 3. Model Definitions
# =========================================================================


class TabularResNet(nn.Module):

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_blocks: int = 3,
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
        )

        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            block = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout_rate),
            )
            self.blocks.append(block)

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout_rate / 2.0),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_projection(x)
        for block in self.blocks:
            h = h + block(h)
        return self.head(h).squeeze(-1)


# =========================================================================
# 4. Fold Training Functions
# =========================================================================


def train_lightgbm_fold(X_tr, y_tr, X_val, y_val, X_test, fold: int, seed: int = 42):
    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 20,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.75,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": seed + fold,
        "n_jobs": -1,
        "verbose": -1,
    }

    train_data = lgb.Dataset(X_tr, label=y_tr)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    callbacks = [
        lgb.early_stopping(stopping_rounds=50, verbose=False),
        lgb.log_evaluation(-1),
    ]
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        callbacks=callbacks,
    )

    val_preds = model.predict(X_val, num_iteration=model.best_iteration)
    test_preds = model.predict(X_test, num_iteration=model.best_iteration)

    return val_preds, test_preds


def train_xgboost_fold(X_tr, y_tr, X_val, y_val, X_test, fold: int, seed: int = 42):
    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        learning_rate=0.03,
        max_depth=5,
        min_child_weight=2,
        subsample=0.8,
        colsample_bytree=0.75,
        reg_alpha=0.1,
        reg_lambda=1.0,
        n_estimators=1000,
        early_stopping_rounds=50,
        random_state=seed + fold,
        n_jobs=-1,
        verbosity=0,
    )

    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

    val_preds = model.predict_proba(X_val)[:, 1]
    test_preds = model.predict_proba(X_test)[:, 1]

    return val_preds, test_preds


def train_catboost_fold(X_tr, y_tr, X_val, y_val, X_test, fold: int, seed: int = 42):
    model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=3.0,
        eval_metric="AUC",
        loss_function="Logloss",
        random_seed=seed + fold,
        early_stopping_rounds=50,
        verbose=False,
    )

    model.fit(
        X_tr,
        y_tr,
        eval_set=(X_val, y_val),
        use_best_model=True,
        verbose=False,
    )

    val_preds = model.predict_proba(X_val)[:, 1]
    test_preds = model.predict_proba(X_test)[:, 1]

    return val_preds, test_preds


def train_extra_trees_fold(X_tr, y_tr, X_val, y_val, X_test, fold: int, seed: int = 42):
    model = ExtraTreesClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=seed + fold,
        n_jobs=-1,
    )
    model.fit(X_tr, y_tr)
    val_preds = model.predict_proba(X_val)[:, 1]
    test_preds = model.predict_proba(X_test)[:, 1]
    return val_preds, test_preds


def build_stacking_ensemble(oof_preds_list, test_preds_list, y_true):
    model_names = ["LightGBM", "XGBoost", "CatBoost", "ExtraTrees", "TabularResNet"]
    for i, (name, oof) in enumerate(zip(model_names, oof_preds_list)):
        auc = roc_auc_score(y_true, oof)
        print(f"Model {i+1} ({name}) OOF ROC-AUC: {auc:.5f}")

    oof_matrix = np.column_stack(oof_preds_list)
    test_matrix = np.column_stack(test_preds_list)
    n_models = len(oof_preds_list)

    # 1. Probability Blend Optimization (Nelder-Mead)
    def prob_objective(weights):
        w = softmax(weights)
        blend = np.dot(oof_matrix, w)
        return -roc_auc_score(y_true, blend)

    res_prob = minimize(
        prob_objective,
        np.zeros(n_models),
        method="Nelder-Mead",
        options={"maxiter": 1000, "xatol": 1e-4, "fatol": 1e-4},
    )
    opt_w_prob = softmax(res_prob.x)
    oof_prob_blend = np.dot(oof_matrix, opt_w_prob)
    test_prob_blend = np.dot(test_matrix, opt_w_prob)
    auc_prob = roc_auc_score(y_true, oof_prob_blend)
    print(f"Probability Blend OOF ROC-AUC: {auc_prob:.5f} (Weights: {np.round(opt_w_prob, 4)})")

    # 2. Rank Blend Optimization (Nelder-Mead)
    oof_rank_matrix = np.zeros_like(oof_matrix)
    test_rank_matrix = np.zeros_like(test_matrix)
    for j in range(n_models):
        oof_rank_matrix[:, j] = rankdata(oof_matrix[:, j]) / len(oof_matrix)
        test_rank_matrix[:, j] = rankdata(test_matrix[:, j]) / len(test_matrix)

    def rank_objective(weights):
        w = softmax(weights)
        blend = np.dot(oof_rank_matrix, w)
        return -roc_auc_score(y_true, blend)

    res_rank = minimize(
        rank_objective,
        np.zeros(n_models),
        method="Nelder-Mead",
        options={"maxiter": 1000, "xatol": 1e-4, "fatol": 1e-4},
    )
    opt_w_rank = softmax(res_rank.x)
    oof_rank_blend = np.dot(oof_rank_matrix, opt_w_rank)
    test_rank_blend = np.dot(test_rank_matrix, opt_w_rank)
    auc_rank = roc_auc_score(y_true, oof_rank_blend)
    print(f"Rank Blend OOF ROC-AUC: {auc_rank:.5f} (Weights: {np.round(opt_w_rank, 4)})")

    # 3. Logistic Regression Stacking Meta-Learner
    meta_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    meta_oof = np.zeros(len(y_true))
    meta_test = np.zeros(test_matrix.shape[0])
    for tr_idx, val_idx in meta_skf.split(oof_matrix, y_true):
        meta_lr = LogisticRegression(C=0.5, penalty="l2", solver="lbfgs", max_iter=500, random_state=42)
        meta_lr.fit(oof_matrix[tr_idx], y_true[tr_idx])
        meta_oof[val_idx] = meta_lr.predict_proba(oof_matrix[val_idx])[:, 1]
        meta_test += meta_lr.predict_proba(test_matrix)[:, 1] / 5.0

    auc_meta = roc_auc_score(y_true, meta_oof)
    print(f"Meta-Learner (Logistic Regression) OOF ROC-AUC: {auc_meta:.5f}")

    # 4. Multi-Strategy Meta Blend
    candidate_oofs = np.column_stack([oof_prob_blend, oof_rank_blend, meta_oof])
    candidate_tests = np.column_stack([test_prob_blend, test_rank_blend, meta_test])

    def meta_blend_objective(weights):
        w = softmax(weights)
        blend = np.dot(candidate_oofs, w)
        return -roc_auc_score(y_true, blend)

    res_meta = minimize(
        meta_blend_objective,
        np.zeros(candidate_oofs.shape[1]),
        method="Nelder-Mead",
        options={"maxiter": 1000, "xatol": 1e-4, "fatol": 1e-4},
    )
    meta_weights = softmax(res_meta.x)
    oof_final = np.dot(candidate_oofs, meta_weights)
    test_final = np.dot(candidate_tests, meta_weights)
    final_score = roc_auc_score(y_true, oof_final)
    print(f"Final Combined Ensemble OOF ROC-AUC: {final_score:.5f} (Strategy Weights: {np.round(meta_weights, 4)})")

    return oof_final, test_final, final_score


def train_tabular_resnet_fold(
    X_tr,
    y_tr,
    X_val,
    y_val,
    X_test,
    fold: int,
    device: torch.device,
    seed: int = 42,
    epochs: int = 35,
    batch_size: int = 64,
):
    set_seed(seed + fold)

    input_dim = X_tr.shape[1]
    model = TabularResNet(
        input_dim=input_dim, hidden_dim=128, num_blocks=3, dropout_rate=0.2
    ).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    mean_vec = X_tr.mean(axis=0)
    std_vec = X_tr.std(axis=0) + 1e-6

    X_tr_norm = (X_tr - mean_vec) / std_vec
    X_val_norm = (X_val - mean_vec) / std_vec
    X_test_norm = (X_test - mean_vec) / std_vec

    tr_tensor = torch.tensor(X_tr_norm, dtype=torch.float32)
    y_tr_tensor = torch.tensor(y_tr, dtype=torch.float32)
    val_tensor = torch.tensor(X_val_norm, dtype=torch.float32).to(device)
    test_tensor = torch.tensor(X_test_norm, dtype=torch.float32).to(device)

    dataset = TensorDataset(tr_tensor, y_tr_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    best_val_auc = -1.0
    best_val_preds = np.zeros(len(y_val))
    best_test_preds = np.zeros(len(X_test))

    for epoch in range(epochs):
        model.train()
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(val_tensor)
            val_probs = torch.sigmoid(val_logits).cpu().numpy()
            current_auc = roc_auc_score(y_val, val_probs)

            if current_auc > best_val_auc:
                best_val_auc = current_auc
                best_val_preds = val_probs
                test_logits = model(test_tensor)
                best_test_preds = torch.sigmoid(test_logits).cpu().numpy()

    return best_val_preds, best_test_preds


# =========================================================================
# 5. Main Execution Pipeline
# =========================================================================


def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_dir = "./input"
    working_dir = "./working"
    submission_dir = "./submission"

    ensure_directory(working_dir)
    ensure_directory(submission_dir)

    train_path = os.path.join(input_dir, "train.parquet")
    test_path = os.path.join(input_dir, "test.parquet")
    target_col = "serious_ae"

    # Feature Engineering
    train_df, test_df, feature_cols, target_col = build_features(
        train_path, test_path, target_col=target_col
    )

    # Align text ridge feature out-of-fold indices with actual cv_fold to eliminate target leakage across CV folds
    n_folds = train_df["cv_fold"].nunique()
    oof_lgb = np.zeros(len(train_df))
    oof_xgb = np.zeros(len(train_df))
    oof_cat = np.zeros(len(train_df))
    oof_et = np.zeros(len(train_df))
    oof_nn = np.zeros(len(train_df))

    test_preds_lgb = np.zeros(len(test_df))
    test_preds_xgb = np.zeros(len(test_df))
    test_preds_cat = np.zeros(len(test_df))
    test_preds_et = np.zeros(len(test_df))
    test_preds_nn = np.zeros(len(test_df))

    X_test_mat = test_df[feature_cols].values

    # Stratified K-Fold Cross-Validation Training
    for fold in range(n_folds):
        tr_mask = train_df["cv_fold"] != fold
        val_mask = train_df["cv_fold"] == fold

        cur_feature_cols = list(feature_cols)
        X_tr_df = train_df.loc[tr_mask, cur_feature_cols].copy()
        y_tr = train_df.loc[tr_mask, target_col].values
        X_val_df = train_df.loc[val_mask, cur_feature_cols].copy()
        y_val = train_df.loc[val_mask, target_col].values
        X_te_df = test_df[cur_feature_cols].copy()

        X_tr = X_tr_df.values
        X_val = X_val_df.values
        X_test_mat = X_te_df.values

        val_p_lgb, test_p_lgb = train_lightgbm_fold(
            X_tr, y_tr, X_val, y_val, X_test_mat, fold=fold
        )
        oof_lgb[val_mask] = val_p_lgb
        test_preds_lgb += test_p_lgb / n_folds

        val_p_xgb, test_p_xgb = train_xgboost_fold(
            X_tr, y_tr, X_val, y_val, X_test_mat, fold=fold
        )
        oof_xgb[val_mask] = val_p_xgb
        test_preds_xgb += test_p_xgb / n_folds

        val_p_cat, test_p_cat = train_catboost_fold(
            X_tr, y_tr, X_val, y_val, X_test_mat, fold=fold
        )
        oof_cat[val_mask] = val_p_cat
        test_preds_cat += test_p_cat / n_folds

        val_p_et, test_p_et = train_extra_trees_fold(
            X_tr, y_tr, X_val, y_val, X_test_mat, fold=fold
        )
        oof_et[val_mask] = val_p_et
        test_preds_et += test_p_et / n_folds

        val_p_nn, test_p_nn = train_tabular_resnet_fold(
            X_tr, y_tr, X_val, y_val, X_test_mat, fold=fold, device=device
        )
        oof_nn[val_mask] = val_p_nn
        test_preds_nn += test_p_nn / n_folds

    y_true = train_df[target_col].values

    # Model Ensembling & Stacking
    oof_list = [oof_lgb, oof_xgb, oof_cat, oof_et, oof_nn]
    test_list = [test_preds_lgb, test_preds_xgb, test_preds_cat, test_preds_et, test_preds_nn]

    oof_ensemble, test_ensemble, final_score = build_stacking_ensemble(
        oof_list, test_list, y_true
    )

    submission_path = os.path.join(submission_dir, "submission.csv")
    submission_df = pd.DataFrame(
        {"id": np.arange(len(test_ensemble)), "serious_ae": test_ensemble}
    )
    submission_df.to_csv(submission_path, index=False)

    print(f"Final Validation Score: {final_score}")


if __name__ == "__main__":
    main()