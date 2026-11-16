import os
import re
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import warnings

warnings.filterwarnings("ignore")

# 1. Load Data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "Medicine status"
unique_targets = train_df[target_col].dropna().unique()

if set(unique_targets) == {0, 1} or set(unique_targets) == {0.0, 1.0}:
    y = train_df[target_col].astype(int).values
else:
    sorted_targets = sorted(unique_targets)
    val_strs = [str(v).lower() for v in sorted_targets]
    pos_keywords = [
        "authorised",
        "authorized",
        "active",
        "positive",
        "true",
        "yes",
        "approved",
        "1",
    ]

    if any(k in val_strs[0] for k in pos_keywords) and not any(
        k in val_strs[1] for k in pos_keywords
    ):
        target_map = {sorted_targets[0]: 1, sorted_targets[1]: 0}
    elif any(k in val_strs[1] for k in pos_keywords) and not any(
        k in val_strs[0] for k in pos_keywords
    ):
        target_map = {sorted_targets[0]: 0, sorted_targets[1]: 1}
    else:
        target_map = {val: i for i, val in enumerate(sorted_targets)}
    y = train_df[target_col].map(target_map).values

# Identify feature columns
drop_cols = [target_col]
for c in train_df.columns:
    if c.lower() in ["id", "unnamed: 0"]:
        drop_cols.append(c)

feature_cols = [c for c in train_df.columns if c not in drop_cols]

# Concatenate datasets for consistent feature processing
df_all = pd.concat([train_df[feature_cols], test_df[feature_cols]], ignore_index=True)
n_train = len(train_df)

X_all = pd.DataFrame(index=df_all.index)

# 2. Feature Engineering Pipeline
for col in feature_cols:
    s = df_all[col]

    # Datetime parsing
    if s.dtype == "object" or "datetime" in str(s.dtype):
        parsed_dt = pd.to_datetime(s, errors="coerce", format="mixed")
        valid_dt_ratio = parsed_dt.notna().mean()
        if valid_dt_ratio > 0.5:
            X_all[f"{col}_year"] = parsed_dt.dt.year.fillna(-1)
            X_all[f"{col}_month"] = parsed_dt.dt.month.fillna(-1)
            X_all[f"{col}_day"] = parsed_dt.dt.day.fillna(-1)
            X_all[f"{col}_dayofweek"] = parsed_dt.dt.dayofweek.fillna(-1)
            X_all[f"{col}_quarter"] = parsed_dt.dt.quarter.fillna(-1)
            continue

    # Numeric columns
    if pd.api.types.is_numeric_dtype(s):
        X_all[col] = s.values
        X_all[f"{col}_isnan"] = s.isna().astype(int).values
        continue

    # Categorical and text columns
    if s.dtype == "object" or pd.api.types.is_categorical_dtype(s):
        s_str = s.fillna("").astype(str)
        avg_len = s_str.str.len().mean()

        freq = s_str.value_counts(normalize=True)
        X_all[f"{col}_freq"] = s_str.map(freq).fillna(0).values
        X_all[f"{col}_len"] = s_str.str.len().values
        X_all[f"{col}_words"] = s_str.str.split().str.len().fillna(0).values

        if avg_len > 30:
            try:
                tfidf = TfidfVectorizer(max_features=500, stop_words="english")
                text_vecs = tfidf.fit_transform(s_str)
                n_comp = min(10, text_vecs.shape[1] - 1)
                if n_comp > 0:
                    svd = TruncatedSVD(n_components=n_comp, random_state=42)
                    svd_feats = svd.fit_transform(text_vecs)
                    for comp_idx in range(svd_feats.shape[1]):
                        X_all[f"{col}_svd_{comp_idx}"] = svd_feats[:, comp_idx]
            except Exception:
                cat_col = s_str.astype("category")
                X_all[f"{col}_cat"] = cat_col.cat.codes
        else:
            cat_col = s_str.astype("category")
            X_all[f"{col}_cat"] = cat_col.cat.codes

# Sanitize feature column names for LightGBM compatibility
clean_columns = []
seen_cols = {}
for col in X_all.columns:
    clean_c = re.sub(r"[^\w]+", "_", str(col)).strip("_")
    if not clean_c:
        clean_c = "feature"
    if clean_c in seen_cols:
        seen_cols[clean_c] += 1
        clean_columns.append(f"{clean_c}_{seen_cols[clean_c]}")
    else:
        seen_cols[clean_c] = 0
        clean_columns.append(clean_c)

X_all.columns = clean_columns

# Separate processed train and test features
X_train = X_all.iloc[:n_train].copy()
X_test = X_all.iloc[n_train:].copy()

# 3. Model Training and Cross-Validation
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(n_train)
test_preds = np.zeros(len(test_df))

lgb_params = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "learning_rate": 0.03,
    "num_leaves": 31,
    "random_state": 42,
    "n_estimators": 1000,
    "verbose": -1,
}

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y)):
    X_tr, y_tr = X_train.iloc[train_idx], y[train_idx]
    X_va, y_va = X_train.iloc[val_idx], y[val_idx]

    model = lgb.LGBMClassifier(**lgb_params)
    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
    )

    oof_preds[val_idx] = model.predict_proba(X_va)[:, 1]
    test_preds += model.predict_proba(X_test)[:, 1] / kf.n_splits

cv_auc = roc_auc_score(y, oof_preds)
print(f"5-Fold CV ROC AUC Score: {cv_auc:.5f}")

# 4. Generate Submission
os.makedirs("./working", exist_ok=True)
sub = pd.DataFrame({"id": np.arange(len(test_preds)), "Medicine status": test_preds})
sub.to_csv("./working/submission.csv", index=False)
