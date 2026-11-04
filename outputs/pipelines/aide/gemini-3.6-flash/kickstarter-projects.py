import os
import re
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import lightgbm as lgb
import warnings

warnings.filterwarnings("ignore")

# 1. Load Data
train_path = "./input/train.parquet"
test_path = "./input/test.parquet"

train = pd.read_parquet(train_path)
test = pd.read_parquet(test_path)

target_col = "status"
if target_col in train.columns:
    y_raw = train[target_col]
    if y_raw.dtype == object or str(y_raw.dtype) in ["string", "category"]:
        unique_vals = [str(v).lower() for v in y_raw.dropna().unique()]
        if "successful" in unique_vals:
            y = (y_raw.astype(str).str.lower() == "successful").astype(int)
        else:
            y = pd.factorize(y_raw)[0]
    else:
        y = y_raw.astype(int)
else:
    raise ValueError("Target column not found in train data")

common_cols = [
    c for c in train.columns if c in test.columns and c not in [target_col, "id"]
]

n_train = len(train)
n_test = len(test)

df = pd.concat([train[common_cols], test[common_cols]], axis=0, ignore_index=True)


# 2. Extract JSON Categorical Fields, Hierarchies & Location Data
def parse_json_column(series, key):
    pattern = rf'"{key}"\s*:\s*"([^"]+)"'
    return series.astype(str).str.extract(pattern, expand=False).fillna("missing")


for col in df.columns:
    if df[col].dtype == object:
        sample_str = str(df[col].dropna().iloc[0]) if not df[col].dropna().empty else ""
        if '{"id"' in sample_str or '"name":' in sample_str or '"slug":' in sample_str:
            df[f"{col}_name"] = parse_json_column(df[col], "name")
            df[f"{col}_slug"] = parse_json_column(df[col], "slug")
            df[f"{col}_state"] = parse_json_column(df[col], "state")
            df[f"{col}_display_type"] = parse_json_column(df[col], "display_type")

# Split category slugs into explicit parent and subcategories
slug_cols = [c for c in df.columns if "slug" in c.lower()]
for col in slug_cols:
    df[f"{col}_parent"] = df[col].apply(
        lambda x: str(x).split("/")[0] if "/" in str(x) else str(x)
    )
    df[f"{col}_sub"] = df[col].apply(
        lambda x: (
            str(x).split("/")[1]
            if "/" in str(x) and len(str(x).split("/")) > 1
            else str(x)
        )
    )

# 3. Datetime, Duration & Preparation Engineering
date_cols = [
    c
    for c in df.columns
    if "at" in c.lower()
    or "deadline" in c.lower()
    or "date" in c.lower()
    or "time" in c.lower()
]
timestamp_cols = []

for c in list(df.columns):
    if c in date_cols or "created" in c or "launched" in c or "deadline" in c:
        try:
            sample_val = df[c].dropna().iloc[0]
            if isinstance(sample_val, (int, float, np.number)) and sample_val > 1e8:
                df[f"{c}_dt"] = pd.to_datetime(df[c], unit="s", errors="coerce")
                timestamp_cols.append(f"{c}_dt")
            elif isinstance(sample_val, str) or pd.api.types.is_datetime64_any_dtype(
                df[c]
            ):
                df[f"{c}_dt"] = pd.to_datetime(df[c], errors="coerce")
                timestamp_cols.append(f"{c}_dt")
        except Exception:
            pass

launch_cols = [c for c in timestamp_cols if "launch" in c]
deadline_cols = [c for c in timestamp_cols if "deadline" in c]
created_cols = [c for c in timestamp_cols if "create" in c]

if launch_cols and deadline_cols:
    df["duration_days"] = np.maximum(
        0.1, (df[deadline_cols[0]] - df[launch_cols[0]]).dt.total_seconds() / 86400.0
    )

if launch_cols and created_cols:
    df["prep_days"] = np.maximum(
        0.0, (df[launch_cols[0]] - df[created_cols[0]]).dt.total_seconds() / 86400.0
    )

if "duration_days" in df.columns and "prep_days" in df.columns:
    df["prep_duration_ratio"] = df["prep_days"] / (df["duration_days"] + 1e-5)

for ts_col in timestamp_cols:
    base = ts_col.replace("_dt", "")
    df[f"{base}_year"] = df[ts_col].dt.year
    df[f"{base}_month"] = df[ts_col].dt.month
    df[f"{base}_day"] = df[ts_col].dt.day
    df[f"{base}_dayofweek"] = df[ts_col].dt.dayofweek
    df[f"{base}_hour"] = df[ts_col].dt.hour
    df[f"{base}_is_weekend"] = (df[ts_col].dt.dayofweek >= 5).astype(int)

df = df.drop(columns=timestamp_cols, errors="ignore")

# 4. Standardized Goal Calculation & Daily Funding Rate
goal_cols = [
    c
    for c in df.columns
    if "goal" in c.lower() and pd.api.types.is_numeric_dtype(df[c])
]
fx_cols = [
    c
    for c in df.columns
    if ("rate" in c.lower() or "fx" in c.lower() or "static_usd" in c.lower())
    and pd.api.types.is_numeric_dtype(df[c])
]

df["usd_goal"] = np.nan
if goal_cols:
    primary_goal = goal_cols[0]
    if fx_cols:
        primary_fx = fx_cols[0]
        df["usd_goal"] = df[primary_goal] * df[primary_fx]
    else:
        df["usd_goal"] = df[primary_goal]

df["usd_goal"] = df["usd_goal"].fillna(df[goal_cols[0]] if goal_cols else 0)
df["log_usd_goal"] = np.log1p(np.maximum(0, df["usd_goal"]))

if "duration_days" in df.columns:
    df["usd_goal_per_day"] = df["usd_goal"] / df["duration_days"]
    df["log_usd_goal_per_day"] = np.log1p(np.maximum(0, df["usd_goal_per_day"]))

for g_col in goal_cols:
    df[f"log_{g_col}"] = np.log1p(np.maximum(0, df[g_col].fillna(0)))

# 5. Group Aggregations on USD Goal, Goal Per Day, and Duration
cat_group_cols = [
    c
    for c in df.columns
    if any(
        k in c.lower()
        for k in ["cat", "parent", "sub", "country", "currency", "slug", "state"]
    )
    and (df[c].dtype == object or str(df[c].dtype) == "category")
]

agg_targets = [
    col
    for col in ["usd_goal", "usd_goal_per_day", "duration_days"]
    if col in df.columns
]

for cat in cat_group_cols[:10]:
    for target in agg_targets:
        g_mean = df.groupby(cat)[target].transform("mean")
        g_std = df.groupby(cat)[target].transform("std").fillna(0)
        df[f"{target}_mean_by_{cat}"] = g_mean
        df[f"{target}_ratio_by_{cat}"] = df[target] / (g_mean + 1e-5)
        df[f"{target}_zscore_by_{cat}"] = (df[target] - g_mean) / (g_std + 1e-5)

# 6. Text Processing (Counts & Enhanced SVD Features)
text_cols = [c for c in df.columns if c in ["name", "blurb", "description", "title"]]
combined_text = pd.Series("", index=df.index)

for t_col in text_cols:
    if df[t_col].dtype == object:
        s = df[t_col].fillna("")
        df[f"{t_col}_char_len"] = s.str.len()
        df[f"{t_col}_word_count"] = s.str.split().str.len()
        df[f"{t_col}_caps_ratio"] = s.apply(
            lambda v: sum(1 for ch in v if ch.isupper()) / (len(v) + 1)
        )
        df[f"{t_col}_excl_count"] = s.apply(lambda v: v.count("!"))
        df[f"{t_col}_quest_count"] = s.apply(lambda v: v.count("?"))
        combined_text = combined_text + " " + s

if combined_text.str.strip().str.len().sum() > 0:
    tfidf = TfidfVectorizer(
        max_features=20000, stop_words="english", ngram_range=(1, 2)
    )
    tfidf_mat = tfidf.fit_transform(combined_text)
    svd = TruncatedSVD(n_components=40, random_state=42)
    svd_feats = svd.fit_transform(tfidf_mat)
    for i in range(40):
        df[f"tfidf_svd_{i}"] = svd_feats[:, i]

# 7. Frequency & Categorical Encoding
cat_cols = [
    c for c in df.columns if df[c].dtype == object or str(df[c].dtype) == "category"
]
for c in cat_cols:
    freq = df[c].value_counts(normalize=True).to_dict()
    df[f"{c}_freq"] = df[c].map(freq)
    df[c] = df[c].astype("category")

# 8. Model Training (Stratified 5-Fold LightGBM with Optimized Hyperparameters)
X_train = df.iloc[:n_train].copy()
X_test = df.iloc[n_train:].copy()
y_train = y.values if isinstance(y, (pd.Series, pd.DataFrame)) else y

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(n_train)
test_preds = np.zeros(n_test)

lgb_params = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "learning_rate": 0.03,
    "num_leaves": 63,
    "max_depth": -1,
    "feature_fraction": 0.75,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "min_child_samples": 25,
    "random_state": 42,
    "n_estimators": 2500,
    "verbose": -1,
    "n_jobs": -1,
}

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
    X_tr, y_tr = X_train.iloc[train_idx], y_train[train_idx]
    X_va, y_va = X_train.iloc[val_idx], y_train[val_idx]

    model = lgb.LGBMClassifier(**lgb_params)

    callbacks = [
        lgb.early_stopping(stopping_rounds=50, verbose=False),
        lgb.log_evaluation(period=0),
    ]

    model.fit(
        X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric="auc", callbacks=callbacks
    )

    val_pred = model.predict_proba(X_va)[:, 1]
    oof_preds[val_idx] = val_pred
    test_preds += model.predict_proba(X_test)[:, 1] / kf.n_splits

final_auc = roc_auc_score(y_train, oof_preds)
print(f"Validation ROC AUC Score: {final_auc:.6f}")

# 9. Save Test Predictions
os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame({"id": np.arange(n_test), "status": test_preds})
submission.to_csv("./working/submission.csv", index=False)
