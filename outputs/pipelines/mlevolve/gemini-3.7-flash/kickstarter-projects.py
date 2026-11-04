import os
import copy
import string
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import rankdata
from scipy.sparse import hstack
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

# ==============================================================================
# 1. Directories & Data Ingestion
# ==============================================================================
input_dir = "./input"
submission_dir = "./submission"
working_dir = "./working"

os.makedirs(submission_dir, exist_ok=True)
os.makedirs(working_dir, exist_ok=True)

train_raw = pd.read_parquet(os.path.join(input_dir, "train.parquet"))
test_raw = pd.read_parquet(os.path.join(input_dir, "test.parquet"))

target_col = (
    "status"
    if "status" in train_raw.columns
    else [c for c in train_raw.columns if c.lower() in ["status", "state", "target"]][0]
)
y_raw = train_raw[target_col]

# Map target to binary 0/1 integers
if y_raw.dtype == object or isinstance(y_raw.iloc[0], str):
    y = (y_raw.str.lower().str.strip() == "successful").astype(int).values
elif y_raw.dtype == bool:
    y = y_raw.astype(int).values
else:
    y = (y_raw.values == 1).astype(int)

feature_cols = [
    col for col in test_raw.columns if col in train_raw.columns and col != target_col
]
train_df = train_raw[feature_cols].copy().reset_index(drop=True)
test_df = test_raw[feature_cols].copy().reset_index(drop=True)

# Stratified 5-Fold Split
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
train_df["fold"] = -1
for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, y)):
    train_df.loc[val_idx, "fold"] = fold


# ==============================================================================
# 2. Feature Engineering Functions
# ==============================================================================
def detect_column_types(df):
    text_cols, date_cols, cat_cols, num_cols = [], [], [], []
    for c in df.columns:
        if c == "fold":
            continue
        c_lower = c.lower()
        if any(
            kw in c_lower for kw in ["deadline", "launch", "created", "date", "time"]
        ):
            date_cols.append(c)
        elif any(kw in c_lower for kw in ["name", "blurb", "desc", "title"]):
            text_cols.append(c)
        elif df[c].dtype in [object, "category"] or c_lower in [
            "category",
            "main_category",
            "currency",
            "country",
        ]:
            cat_cols.append(c)
        elif np.issubdtype(df[c].dtype, np.number):
            num_cols.append(c)
    return text_cols, date_cols, cat_cols, num_cols


text_cols, date_cols, cat_cols, num_cols = detect_column_types(train_df)


def extract_text_features(df, col_name):
    s = df[col_name].fillna("").astype(str)
    feat = pd.DataFrame(index=df.index)
    feat[f"{col_name}_char_count"] = s.str.len()
    feat[f"{col_name}_word_count"] = s.str.split().str.len()
    feat[f"{col_name}_avg_word_len"] = feat[f"{col_name}_char_count"] / (
        feat[f"{col_name}_word_count"] + 1e-5
    )
    feat[f"{col_name}_upper_count"] = s.apply(
        lambda x: sum(1 for ch in x if ch.isupper())
    )
    feat[f"{col_name}_upper_ratio"] = feat[f"{col_name}_upper_count"] / (
        feat[f"{col_name}_char_count"] + 1e-5
    )
    feat[f"{col_name}_digit_count"] = s.apply(
        lambda x: sum(1 for ch in x if ch.isdigit())
    )
    feat[f"{col_name}_punct_count"] = s.apply(
        lambda x: sum(1 for ch in x if ch in string.punctuation)
    )
    feat[f"{col_name}_excl_count"] = s.apply(lambda x: x.count("!"))
    feat[f"{col_name}_quest_count"] = s.apply(lambda x: x.count("?"))
    feat[f"{col_name}_is_all_caps"] = (feat[f"{col_name}_upper_ratio"] > 0.85).astype(
        int
    )
    feat[f"{col_name}_has_number"] = (feat[f"{col_name}_digit_count"] > 0).astype(int)
    return feat


def extract_datetime_features(df, date_cols):
    feat = pd.DataFrame(index=df.index)
    dt_series = {}
    for c in date_cols:
        parsed = pd.to_datetime(df[c], errors="coerce")
        if parsed.isna().mean() > 0.5:
            parsed = pd.to_datetime(
                pd.to_numeric(df[c], errors="coerce"), unit="s", errors="coerce"
            )
        dt_series[c] = parsed

        prefix = c.replace("_at", "").replace("_date", "")
        feat[f"{prefix}_year"] = parsed.dt.year.fillna(-1).astype(int)
        feat[f"{prefix}_month"] = parsed.dt.month.fillna(-1).astype(int)
        feat[f"{prefix}_day"] = parsed.dt.day.fillna(-1).astype(int)
        feat[f"{prefix}_dayofweek"] = parsed.dt.dayofweek.fillna(-1).astype(int)
        feat[f"{prefix}_hour"] = parsed.dt.hour.fillna(-1).astype(int)
        feat[f"{prefix}_is_weekend"] = parsed.dt.dayofweek.isin([5, 6]).astype(int)
        feat[f"{prefix}_quarter"] = parsed.dt.quarter.fillna(-1).astype(int)

        feat[f"{prefix}_month_sin"] = np.sin(2 * np.pi * parsed.dt.month / 12.0).fillna(
            0
        )
        feat[f"{prefix}_month_cos"] = np.cos(2 * np.pi * parsed.dt.month / 12.0).fillna(
            0
        )
        feat[f"{prefix}_dow_sin"] = np.sin(
            2 * np.pi * parsed.dt.dayofweek / 7.0
        ).fillna(0)
        feat[f"{prefix}_dow_cos"] = np.cos(
            2 * np.pi * parsed.dt.dayofweek / 7.0
        ).fillna(0)
        feat[f"{prefix}_hour_sin"] = np.sin(2 * np.pi * parsed.dt.hour / 24.0).fillna(0)
        feat[f"{prefix}_hour_cos"] = np.cos(2 * np.pi * parsed.dt.hour / 24.0).fillna(0)

    launch_keys = [k for k in dt_series.keys() if "launch" in k.lower()]
    deadline_keys = [k for k in dt_series.keys() if "deadline" in k.lower()]
    created_keys = [k for k in dt_series.keys() if "created" in k.lower()]

    if launch_keys and deadline_keys:
        l_col = dt_series[launch_keys[0]]
        d_col = dt_series[deadline_keys[0]]
        duration_days = (d_col - l_col).dt.total_seconds() / 86400.0
        feat["duration_days"] = duration_days.fillna(duration_days.median())
        feat["duration_weeks"] = feat["duration_days"] / 7.0
        feat["is_standard_duration_30"] = (feat["duration_days"].round() == 30).astype(
            int
        )
        feat["is_standard_duration_60"] = (feat["duration_days"].round() == 60).astype(
            int
        )

    if launch_keys and created_keys:
        l_col = dt_series[launch_keys[0]]
        c_col = dt_series[created_keys[0]]
        prep_days = (l_col - c_col).dt.total_seconds() / 86400.0
        feat["prep_time_days"] = prep_days.fillna(prep_days.median())
        feat["log1p_prep_time"] = np.log1p(np.maximum(feat["prep_time_days"], 0))

    return feat


def extract_financial_features(df, num_cols, duration_series=None, prep_series=None):
    feat = pd.DataFrame(index=df.index)
    goal_cols = [c for c in num_cols if "goal" in c.lower()]
    if not goal_cols and len(num_cols) > 0:
        goal_cols = [num_cols[0]]

    for gc in goal_cols:
        val = pd.to_numeric(df[gc], errors="coerce").fillna(0)
        feat[f"{gc}_log1p"] = np.log1p(np.maximum(val, 0))
        feat[f"{gc}_is_round_10"] = (val % 10 == 0).astype(int)
        feat[f"{gc}_is_round_100"] = (val % 100 == 0).astype(int)
        feat[f"{gc}_is_round_1000"] = (val % 1000 == 0).astype(int)
        feat[f"{gc}_is_round_5000"] = (val % 5000 == 0).astype(int)
        feat[f"{gc}_is_round_10000"] = (val % 10000 == 0).astype(int)

        if duration_series is not None:
            dur_safe = np.maximum(duration_series, 0.5)
            daily_goal = val / dur_safe
            feat[f"{gc}_per_day"] = daily_goal
            feat[f"{gc}_per_day_log1p"] = np.log1p(np.maximum(daily_goal, 0))

        if prep_series is not None:
            prep_safe = np.maximum(prep_series, 0.5)
            prep_goal = val / prep_safe
            feat[f"{gc}_per_prep_day"] = prep_goal
            feat[f"{gc}_per_prep_day_log1p"] = np.log1p(np.maximum(prep_goal, 0))

    return feat


train_feat_blocks = []
test_feat_blocks = []

train_dt_feat = extract_datetime_features(train_df, date_cols)
test_dt_feat = extract_datetime_features(test_df, date_cols)
train_feat_blocks.append(train_dt_feat)
test_feat_blocks.append(test_dt_feat)

dur_train = train_dt_feat["duration_days"] if "duration_days" in train_dt_feat else None
dur_test = test_dt_feat["duration_days"] if "duration_days" in test_dt_feat else None
prep_train = train_dt_feat["prep_time_days"] if "prep_time_days" in train_dt_feat else None
prep_test = test_dt_feat["prep_time_days"] if "prep_time_days" in test_dt_feat else None

train_fin_feat = extract_financial_features(train_df, num_cols, dur_train, prep_train)
test_fin_feat = extract_financial_features(test_df, num_cols, dur_test, prep_test)
train_feat_blocks.append(train_fin_feat)
test_feat_blocks.append(test_fin_feat)

for tc in text_cols:
    train_feat_blocks.append(extract_text_features(train_df, tc))
    test_feat_blocks.append(extract_text_features(test_df, tc))

# Multi-text synergy (combine name, blurb, and categorical text tokens)
train_combined_text = pd.Series("", index=train_df.index)
test_combined_text = pd.Series("", index=test_df.index)

text_sources = [c for c in text_cols] + [c for c in ["category", "main_category"] if c in train_df.columns]
for c in text_sources:
    train_combined_text += train_df[c].fillna("").astype(str) + " "
    test_combined_text += test_df[c].fillna("").astype(str) + " "

train_combined_text = train_combined_text.str.strip()
test_combined_text = test_combined_text.str.strip()

# Dual-Granularity Text Feature Extraction (Word + Sub-word Char N-grams)
word_tfidf_pipe = TfidfVectorizer(
    max_features=25000,
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True,
    stop_words="english",
)
word_tfidf_train = word_tfidf_pipe.fit_transform(train_combined_text)
word_tfidf_test = word_tfidf_pipe.transform(test_combined_text)

char_tfidf_pipe = TfidfVectorizer(
    max_features=25000,
    analyzer="char_wb",
    ngram_range=(3, 5),
    min_df=3,
    sublinear_tf=True,
)
char_tfidf_train = char_tfidf_pipe.fit_transform(train_combined_text)
char_tfidf_test = char_tfidf_pipe.transform(test_combined_text)

# Dual TruncatedSVD components for word and character representations
n_svd_word = 24
n_svd_char = 24

svd_word = TruncatedSVD(n_components=n_svd_word, random_state=42)
svd_train_word = svd_word.fit_transform(word_tfidf_train)
svd_test_word = svd_word.transform(word_tfidf_test)

svd_char = TruncatedSVD(n_components=n_svd_char, random_state=42)
svd_train_char = svd_char.fit_transform(char_tfidf_train)
svd_test_char = svd_char.transform(char_tfidf_test)

word_svd_cols = [f"word_text_svd_{i}" for i in range(n_svd_word)]
char_svd_cols = [f"char_text_svd_{i}" for i in range(n_svd_char)]

train_feat_blocks.append(pd.DataFrame(svd_train_word, columns=word_svd_cols, index=train_df.index))
test_feat_blocks.append(pd.DataFrame(svd_test_word, columns=word_svd_cols, index=test_df.index))
train_feat_blocks.append(pd.DataFrame(svd_train_char, columns=char_svd_cols, index=train_df.index))
test_feat_blocks.append(pd.DataFrame(svd_test_char, columns=char_svd_cols, index=test_df.index))

# Out-of-fold Multi-granularity Ridge Text Classifier Probability Meta-Features
oof_ridge_word = np.zeros(len(train_df), dtype=float)
test_ridge_word = np.zeros(len(test_df), dtype=float)
oof_ridge_char = np.zeros(len(train_df), dtype=float)
test_ridge_char = np.zeros(len(test_df), dtype=float)
oof_ridge_joint = np.zeros(len(train_df), dtype=float)
test_ridge_joint = np.zeros(len(test_df), dtype=float)

for fold in range(n_splits):
    tr_idx = train_df["fold"] != fold
    va_idx = train_df["fold"] == fold

    fold_word_tfidf = TfidfVectorizer(
        max_features=25000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
        stop_words="english",
    )
    w_tr = fold_word_tfidf.fit_transform(train_combined_text.iloc[tr_idx.values])
    w_va = fold_word_tfidf.transform(train_combined_text.iloc[va_idx.values])
    w_te = fold_word_tfidf.transform(test_combined_text)

    fold_char_tfidf = TfidfVectorizer(
        max_features=25000,
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=3,
        sublinear_tf=True,
    )
    c_tr = fold_char_tfidf.fit_transform(train_combined_text.iloc[tr_idx.values])
    c_va = fold_char_tfidf.transform(train_combined_text.iloc[va_idx.values])
    c_te = fold_char_tfidf.transform(test_combined_text)

    # Word Ridge
    ridge_word = RidgeClassifier(alpha=1.0, random_state=42 + fold)
    ridge_word.fit(w_tr, y[tr_idx])
    dec_w_va = ridge_word.decision_function(w_va)
    oof_ridge_word[va_idx] = 1.0 / (1.0 + np.exp(-dec_w_va))
    test_ridge_word += (1.0 / (1.0 + np.exp(-ridge_word.decision_function(w_te)))) / n_splits

    # Char Ridge
    ridge_char = RidgeClassifier(alpha=1.0, random_state=42 + fold)
    ridge_char.fit(c_tr, y[tr_idx])
    dec_c_va = ridge_char.decision_function(c_va)
    oof_ridge_char[va_idx] = 1.0 / (1.0 + np.exp(-dec_c_va))
    test_ridge_char += (1.0 / (1.0 + np.exp(-ridge_char.decision_function(c_te)))) / n_splits

    # Joint Ridge
    j_tr = hstack([w_tr, c_tr]).tocsr()
    j_va = hstack([w_va, c_va]).tocsr()
    j_te = hstack([w_te, c_te]).tocsr()

    ridge_joint = RidgeClassifier(alpha=1.0, random_state=42 + fold)
    ridge_joint.fit(j_tr, y[tr_idx])
    dec_j_va = ridge_joint.decision_function(j_va)
    oof_ridge_joint[va_idx] = 1.0 / (1.0 + np.exp(-dec_j_va))
    test_ridge_joint += (1.0 / (1.0 + np.exp(-ridge_joint.decision_function(j_te)))) / n_splits

train_feat_blocks.append(
    pd.DataFrame(
        {
            "text_ridge_word_oof_prob": oof_ridge_word,
            "text_ridge_char_oof_prob": oof_ridge_char,
            "text_ridge_joint_oof_prob": oof_ridge_joint,
        },
        index=train_df.index,
    )
)
test_feat_blocks.append(
    pd.DataFrame(
        {
            "text_ridge_word_oof_prob": test_ridge_word,
            "text_ridge_char_oof_prob": test_ridge_char,
            "text_ridge_joint_oof_prob": test_ridge_joint,
        },
        index=test_df.index,
    )
)

X_train_base = pd.concat(train_feat_blocks, axis=1)
X_test_base = pd.concat(test_feat_blocks, axis=1)

# Categorical Interactions and Encodings
for c in cat_cols:
    train_df[c] = train_df[c].fillna("missing").astype(str).str.lower().str.strip()
    test_df[c] = test_df[c].fillna("missing").astype(str).str.lower().str.strip()

cat_interactions = []
interaction_pairs = [
    ("main_category", "currency"),
    ("category", "country"),
    ("main_category", "country"),
    ("currency", "country"),
    ("category", "currency"),
    ("category", "main_category"),
]

for col1, col2 in interaction_pairs:
    if col1 in cat_cols and col2 in cat_cols:
        inter_name = f"inter_{col1}_{col2}"
        train_df[inter_name] = train_df[col1] + "__" + train_df[col2]
        test_df[inter_name] = test_df[col1] + "__" + test_df[col2]
        cat_interactions.append(inter_name)

all_cats = cat_cols + cat_interactions

# Frequency Encodings
for c in all_cats:
    freq_map = train_df[c].value_counts(normalize=True).to_dict()
    X_train_base[f"{c}_freq"] = train_df[c].map(freq_map).fillna(0.0).astype(float)
    X_test_base[f"{c}_freq"] = test_df[c].map(freq_map).fillna(0.0).astype(float)


# Category-Conditional Percentile Rank Function
def compute_cat_pct_rank(train_cat, train_vals, test_cat, test_vals):
    train_pct = pd.Series(0.5, index=train_cat.index, dtype=float)
    test_pct = pd.Series(0.5, index=test_cat.index, dtype=float)
    for cat_val in train_cat.unique():
        tr_mask = train_cat == cat_val
        te_mask = test_cat == cat_val
        vals_tr = train_vals[tr_mask].values
        if len(vals_tr) == 0:
            continue
        vals_sorted = np.sort(vals_tr)
        train_pct.loc[tr_mask] = (
            np.searchsorted(vals_sorted, train_vals[tr_mask].values, side="right")
        ) / float(len(vals_sorted))
        if te_mask.any():
            test_pct.loc[te_mask] = (
                np.searchsorted(vals_sorted, test_vals[te_mask].values, side="right")
            ) / float(len(vals_sorted))
    return train_pct, test_pct


# Multi-level Financial Ratios, Percentile Ranks & Velocity Metrics
primary_goal_col = [c for c in num_cols if "goal" in c.lower()]
goal_name = (
    primary_goal_col[0] if primary_goal_col else (num_cols[0] if num_cols else None)
)

if goal_name is not None:
    goal_train_numeric = pd.to_numeric(train_df[goal_name], errors="coerce").fillna(0.0)
    goal_test_numeric = pd.to_numeric(test_df[goal_name], errors="coerce").fillna(0.0)
    goal_train_log = np.log1p(np.maximum(goal_train_numeric, 0))
    goal_test_log = np.log1p(np.maximum(goal_test_numeric, 0))

    for cat_c in all_cats:
        # Category-conditional goal percentile rank
        tr_pct, te_pct = compute_cat_pct_rank(
            train_df[cat_c], goal_train_numeric, test_df[cat_c], goal_test_numeric
        )
        X_train_base[f"{goal_name}_pct_rank_{cat_c}"] = tr_pct
        X_test_base[f"{goal_name}_pct_rank_{cat_c}"] = te_pct

        temp_tr = pd.DataFrame(
            {
                cat_c: train_df[cat_c],
                "goal": goal_train_numeric,
                "log_goal": goal_train_log,
            }
        )
        group_stats = (
            temp_tr.groupby(cat_c)
            .agg(
                goal_mean=("goal", "mean"),
                goal_median=("goal", "median"),
                goal_std=("goal", "std"),
                goal_q25=("goal", lambda x: np.percentile(x, 25)),
                goal_q75=("goal", lambda x: np.percentile(x, 75)),
                log_mean=("log_goal", "mean"),
                log_std=("log_goal", "std"),
            )
            .reset_index()
        )

        group_stats["goal_std"] = group_stats["goal_std"].fillna(1.0)
        group_stats["log_std"] = group_stats["log_std"].fillna(1.0)

        merged_tr = pd.DataFrame({cat_c: train_df[cat_c]}).merge(
            group_stats, on=cat_c, how="left"
        )
        merged_te = pd.DataFrame({cat_c: test_df[cat_c]}).merge(
            group_stats, on=cat_c, how="left"
        )

        global_median = float(goal_train_numeric.median())
        global_mean = float(goal_train_numeric.mean())
        global_std = (
            float(goal_train_numeric.std())
            if goal_train_numeric.std() > 0
            else 1.0
        )
        global_q25 = float(np.percentile(goal_train_numeric, 25))
        global_q75 = float(np.percentile(goal_train_numeric, 75))
        global_log_mean = float(goal_train_log.mean())
        global_log_std = (
            float(goal_train_log.std()) if goal_train_log.std() > 0 else 1.0
        )

        for m_df in [merged_tr, merged_te]:
            m_df["goal_median"] = m_df["goal_median"].fillna(global_median)
            m_df["goal_mean"] = m_df["goal_mean"].fillna(global_mean)
            m_df["goal_std"] = m_df["goal_std"].fillna(global_std)
            m_df["goal_q25"] = m_df["goal_q25"].fillna(global_q25)
            m_df["goal_q75"] = m_df["goal_q75"].fillna(global_q75)
            m_df["log_mean"] = m_df["log_mean"].fillna(global_log_mean)
            m_df["log_std"] = m_df["log_std"].fillna(global_log_std)

        X_train_base[f"{goal_name}_ratio_{cat_c}_median"] = goal_train_numeric / (
            merged_tr["goal_median"] + 1.0
        )
        X_train_base[f"{goal_name}_ratio_{cat_c}_q25"] = goal_train_numeric / (
            merged_tr["goal_q25"] + 1.0
        )
        X_train_base[f"{goal_name}_ratio_{cat_c}_q75"] = goal_train_numeric / (
            merged_tr["goal_q75"] + 1.0
        )
        X_train_base[f"{goal_name}_diff_{cat_c}_mean"] = (
            goal_train_numeric - merged_tr["goal_mean"]
        ) / (merged_tr["goal_std"] + 1e-5)
        X_train_base[f"{goal_name}_log_diff_{cat_c}_z"] = (
            goal_train_log - merged_tr["log_mean"]
        ) / (merged_tr["log_std"] + 1e-5)

        X_test_base[f"{goal_name}_ratio_{cat_c}_median"] = goal_test_numeric / (
            merged_te["goal_median"] + 1.0
        )
        X_test_base[f"{goal_name}_ratio_{cat_c}_q25"] = goal_test_numeric / (
            merged_te["goal_q25"] + 1.0
        )
        X_test_base[f"{goal_name}_ratio_{cat_c}_q75"] = goal_test_numeric / (
            merged_te["goal_q75"] + 1.0
        )
        X_test_base[f"{goal_name}_diff_{cat_c}_mean"] = (
            goal_test_numeric - merged_te["goal_mean"]
        ) / (merged_te["goal_std"] + 1e-5)
        X_test_base[f"{goal_name}_log_diff_{cat_c}_z"] = (
            goal_test_log - merged_te["log_mean"]
        ) / (merged_te["log_std"] + 1e-5)

    # Duration-scaled Financial Velocity Metrics
    if dur_train is not None and dur_test is not None:
        tr_dur_safe = np.maximum(dur_train, 0.5)
        te_dur_safe = np.maximum(dur_test, 0.5)

        tr_vel = goal_train_numeric / tr_dur_safe
        te_vel = goal_test_numeric / te_dur_safe

        X_train_base["goal_daily_velocity"] = tr_vel
        X_test_base["goal_daily_velocity"] = te_vel

        X_train_base["goal_weekly_velocity"] = goal_train_numeric / (
            tr_dur_safe / 7.0 + 1e-5
        )
        X_test_base["goal_weekly_velocity"] = goal_test_numeric / (
            te_dur_safe / 7.0 + 1e-5
        )

        X_train_base["log_goal_daily_velocity"] = np.log1p(np.maximum(tr_vel, 0))
        X_test_base["log_goal_daily_velocity"] = np.log1p(np.maximum(te_vel, 0))

        if "category" in train_df.columns:
            tr_vel_pct, te_vel_pct = compute_cat_pct_rank(
                train_df["category"], tr_vel, test_df["category"], te_vel
            )
            X_train_base["goal_velocity_pct_rank_category"] = tr_vel_pct
            X_test_base["goal_velocity_pct_rank_category"] = te_vel_pct

# Out-of-fold Target Encoding
global_target_mean = float(np.mean(y))
smoothing_weight = 10.0

for c in all_cats:
    te_train = np.zeros(len(train_df), dtype=float)
    for fold in range(n_splits):
        tr_idx = train_df["fold"] != fold
        val_idx = train_df["fold"] == fold

        fold_train_df = pd.DataFrame(
            {"cat": train_df.loc[tr_idx, c], "target": y[tr_idx]}
        )
        stats = fold_train_df.groupby("cat")["target"].agg(["count", "mean"])
        smoothed = (
            stats["count"] * stats["mean"] + smoothing_weight * global_target_mean
        ) / (stats["count"] + smoothing_weight)
        te_map = smoothed.to_dict()

        te_train[val_idx] = (
            train_df.loc[val_idx, c].map(te_map).fillna(global_target_mean).values
        )

    X_train_base[f"{c}_target_enc"] = te_train

    full_stats = (
        pd.DataFrame({"cat": train_df[c], "target": y})
        .groupby("cat")["target"]
        .agg(["count", "mean"])
    )
    full_smoothed = (
        full_stats["count"] * full_stats["mean"] + smoothing_weight * global_target_mean
    ) / (full_stats["count"] + smoothing_weight)
    full_te_map = full_smoothed.to_dict()
    X_test_base[f"{c}_target_enc"] = (
        test_df[c].map(full_te_map).fillna(global_target_mean).values
    )

for c in all_cats:
    unique_vals = sorted(train_df[c].unique().tolist())
    val2idx = {val: idx for idx, val in enumerate(unique_vals)}
    X_train_base[f"{c}_cat_idx"] = train_df[c].map(val2idx).fillna(-1).astype(int)
    X_test_base[f"{c}_cat_idx"] = test_df[c].map(val2idx).fillna(-1).astype(int)

common_features = [
    col for col in X_train_base.columns if col in X_test_base.columns and col != "fold"
]
X_train_final = X_train_base[common_features].copy()
X_test_final = X_test_base[common_features].copy()

X_train_final = X_train_final.replace([np.inf, -np.inf], np.nan)
X_test_final = X_test_final.replace([np.inf, -np.inf], np.nan)

for col in common_features:
    if X_train_final[col].isna().any() or X_test_final[col].isna().any():
        fill_val = X_train_final[col].median()
        if pd.isna(fill_val):
            fill_val = 0.0
        X_train_final[col] = X_train_final[col].fillna(fill_val)
        X_test_final[col] = X_test_final[col].fillna(fill_val)

folds = train_df["fold"].values
n_features = len(common_features)


# ==============================================================================
# 3. Model Definitions
# ==============================================================================
def create_lgbm_classifier(random_state=42):
    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "learning_rate": 0.02,
        "num_leaves": 47,
        "max_depth": -1,
        "min_child_samples": 25,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.65,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": random_state,
        "n_estimators": 3500,
        "n_jobs": -1,
        "verbose": -1,
    }
    return lgb.LGBMClassifier(**params)


def create_xgb_classifier(random_state=42):
    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "booster": "gbtree",
        "learning_rate": 0.02,
        "max_depth": 7,
        "min_child_weight": 3,
        "subsample": 0.8,
        "colsample_bytree": 0.65,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": random_state,
        "n_estimators": 3500,
        "n_jobs": -1,
        "tree_method": "hist",
        "early_stopping_rounds": 50,
    }
    return xgb.XGBClassifier(**params)


def create_catboost_classifier(random_state=42):
    params = {
        "iterations": 3000,
        "learning_rate": 0.02,
        "depth": 7,
        "rsm": 0.65,
        "l2_leaf_reg": 3.0,
        "eval_metric": "AUC",
        "random_seed": random_state,
        "verbose": False,
        "early_stopping_rounds": 50,
        "thread_count": -1,
    }
    return CatBoostClassifier(**params)


class TabularResidualBlock(nn.Module):

    def __init__(self, in_features, out_features, dropout_rate=0.2):
        super().__init__()
        self.norm = nn.LayerNorm(in_features)
        self.linear1 = nn.Linear(in_features, out_features)
        self.act = nn.Mish()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear2 = nn.Linear(out_features, out_features)
        self.shortcut = (
            nn.Linear(in_features, out_features)
            if in_features != out_features
            else nn.Identity()
        )

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.norm(x)
        out = self.linear1(out)
        out = self.act(out)
        out = self.dropout(out)
        out = self.linear2(out)
        return out + residual


class TabularResidualMLP(nn.Module):

    def __init__(
        self,
        input_dim,
        hidden_dims=[256, 128, 64],
        dropout_rate=0.25,
        output_dim=1,
    ):
        super().__init__()
        self.input_norm = nn.BatchNorm1d(input_dim)
        self.input_proj = nn.Linear(input_dim, hidden_dims[0])
        self.input_act = nn.Mish()

        blocks = []
        curr_dim = hidden_dims[0]
        for h_dim in hidden_dims[1:]:
            blocks.append(
                TabularResidualBlock(curr_dim, h_dim, dropout_rate=dropout_rate)
            )
            curr_dim = h_dim

        self.res_blocks = nn.Sequential(*blocks)
        self.head_norm = nn.LayerNorm(curr_dim)
        self.head_dropout = nn.Dropout(dropout_rate)
        self.head = nn.Linear(curr_dim, output_dim)

    def forward(self, x):
        x = self.input_norm(x)
        x = self.input_proj(x)
        x = self.input_act(x)
        x = self.res_blocks(x)
        x = self.head_norm(x)
        x = self.head_dropout(x)
        logits = self.head(x)
        return logits.squeeze(-1)


# ==============================================================================
# 4. Model Training & Cross-Validation
# ==============================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

oof_lgb = np.zeros(len(X_train_final), dtype=np.float64)
oof_xgb = np.zeros(len(X_train_final), dtype=np.float64)
oof_cb = np.zeros(len(X_train_final), dtype=np.float64)
oof_nn = np.zeros(len(X_train_final), dtype=np.float64)

test_preds_lgb = np.zeros(len(X_test_final), dtype=np.float64)
test_preds_xgb = np.zeros(len(X_test_final), dtype=np.float64)
test_preds_cb = np.zeros(len(X_test_final), dtype=np.float64)
test_preds_nn = np.zeros(len(X_test_final), dtype=np.float64)

for fold in range(n_splits):
    tr_mask = folds != fold
    va_mask = folds == fold

    X_tr = X_train_final.loc[tr_mask, common_features].copy()
    y_tr = y[tr_mask]
    X_va = X_train_final.loc[va_mask, common_features].copy()
    y_va = y[va_mask]
    X_te = X_test_final[common_features].copy()

    # LightGBM
    lgb_clf = create_lgbm_classifier(random_state=42 + fold)
    lgb_clf.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    val_preds_lgb = lgb_clf.predict_proba(X_va)[:, 1]
    oof_lgb[va_mask] = val_preds_lgb
    test_preds_lgb += lgb_clf.predict_proba(X_te)[:, 1] / n_splits
    auc_lgb_f = roc_auc_score(y_va, val_preds_lgb)

    # XGBoost
    xgb_clf = create_xgb_classifier(random_state=42 + fold)
    xgb_clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    val_preds_xgb = xgb_clf.predict_proba(X_va)[:, 1]
    oof_xgb[va_mask] = val_preds_xgb
    test_preds_xgb += xgb_clf.predict_proba(X_te)[:, 1] / n_splits
    auc_xgb_f = roc_auc_score(y_va, val_preds_xgb)

    # CatBoost
    cb_clf = create_catboost_classifier(random_state=42 + fold)
    cb_clf.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=False)
    val_preds_cb = cb_clf.predict_proba(X_va)[:, 1]
    oof_cb[va_mask] = val_preds_cb
    test_preds_cb += cb_clf.predict_proba(X_te)[:, 1] / n_splits
    auc_cb_f = roc_auc_score(y_va, val_preds_cb)

    # Tabular Residual MLP
    scaler = StandardScaler()
    X_tr_sc = np.nan_to_num(
        scaler.fit_transform(X_tr.values), nan=0.0, posinf=0.0, neginf=0.0
    )
    X_va_sc = np.nan_to_num(
        scaler.transform(X_va.values), nan=0.0, posinf=0.0, neginf=0.0
    )
    X_te_sc = np.nan_to_num(
        scaler.transform(X_te.values), nan=0.0, posinf=0.0, neginf=0.0
    )

    tr_dataset = TensorDataset(
        torch.tensor(X_tr_sc, dtype=torch.float32),
        torch.tensor(y_tr, dtype=torch.float32),
    )
    va_dataset = TensorDataset(
        torch.tensor(X_va_sc, dtype=torch.float32),
        torch.tensor(y_va, dtype=torch.float32),
    )
    tr_loader = DataLoader(
        tr_dataset, batch_size=512, shuffle=True, drop_last=False, num_workers=0
    )
    va_loader = DataLoader(va_dataset, batch_size=1024, shuffle=False, num_workers=0)

    nn_model = TabularResidualMLP(
        input_dim=n_features, hidden_dims=[256, 128, 64], dropout_rate=0.25
    ).to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        nn_model.parameters(),
        lr=2e-3,
        weight_decay=1e-4,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-5)

    best_va_loss = float("inf")
    best_weights = None

    for epoch in range(20):
        nn_model.train()
        for bx, by in tr_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = nn_model(bx)
            loss = loss_fn(logits, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(nn_model.parameters(), max_norm=2.0)
            optimizer.step()
        scheduler.step()

        nn_model.eval()
        va_loss_accum = 0.0
        va_count = 0
        with torch.no_grad():
            for bx, by in va_loader:
                bx, by = bx.to(device), by.to(device)
                logits = nn_model(bx)
                va_loss_accum += loss_fn(logits, by).item() * len(by)
                va_count += len(by)
        avg_va_loss = va_loss_accum / max(va_count, 1)

        if avg_va_loss < best_va_loss:
            best_va_loss = avg_va_loss
            best_weights = copy.deepcopy(nn_model.state_dict())

    nn_model.load_state_dict(best_weights)
    nn_model.eval()

    with torch.no_grad():
        va_logits = nn_model(torch.tensor(X_va_sc, dtype=torch.float32).to(device))
        val_preds_nn = torch.sigmoid(va_logits).cpu().numpy()
        oof_nn[va_mask] = val_preds_nn
        auc_nn_f = roc_auc_score(y_va, val_preds_nn)

        te_logits = nn_model(torch.tensor(X_te_sc, dtype=torch.float32).to(device))
        test_preds_nn += torch.sigmoid(te_logits).cpu().numpy() / n_splits

    print(
        f"Fold {fold + 1}/{n_splits} - LGB AUC: {auc_lgb_f:.5f} | XGB AUC: {auc_xgb_f:.5f} | CB AUC: {auc_cb_f:.5f} | NN AUC: {auc_nn_f:.5f}"
    )


# ==============================================================================
# 5. Ensemble Optimization & Submission
# ==============================================================================
oof_lgb_rank = rankdata(oof_lgb) / len(oof_lgb)
oof_xgb_rank = rankdata(oof_xgb) / len(oof_xgb)
oof_cb_rank = rankdata(oof_cb) / len(oof_cb)
oof_nn_rank = rankdata(oof_nn) / len(oof_nn)

test_lgb_rank = rankdata(test_preds_lgb) / len(test_preds_lgb)
test_xgb_rank = rankdata(test_preds_xgb) / len(test_preds_xgb)
test_cb_rank = rankdata(test_preds_cb) / len(test_preds_cb)
test_nn_rank = rankdata(test_preds_nn) / len(test_preds_nn)


def ensemble_loss_fn(weights):
    w = np.maximum(weights, 0.0)
    w = w / (np.sum(w) + 1e-8)
    blend_oof = (
        w[0] * oof_lgb_rank
        + w[1] * oof_xgb_rank
        + w[2] * oof_cb_rank
        + w[3] * oof_nn_rank
    )
    return -roc_auc_score(y, blend_oof)


init_weights = [0.35, 0.30, 0.25, 0.10]
bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]
opt_res = minimize(
    ensemble_loss_fn,
    init_weights,
    method="Nelder-Mead",
    bounds=bounds,
    options={"maxiter": 200},
)

best_w = np.maximum(opt_res.x, 0.0)
best_w = best_w / np.sum(best_w)

final_oof_preds = (
    best_w[0] * oof_lgb_rank
    + best_w[1] * oof_xgb_rank
    + best_w[2] * oof_cb_rank
    + best_w[3] * oof_nn_rank
)
final_score = roc_auc_score(y, final_oof_preds)

final_test_preds = (
    best_w[0] * test_lgb_rank
    + best_w[1] * test_xgb_rank
    + best_w[2] * test_cb_rank
    + best_w[3] * test_nn_rank
)

id_vals = np.arange(len(final_test_preds), dtype=int)

submission_df = pd.DataFrame({"id": id_vals, "status": final_test_preds})
submission_df.to_csv(os.path.join(submission_dir, "submission.csv"), index=False)

print(f"Final Validation Score: {final_score}")