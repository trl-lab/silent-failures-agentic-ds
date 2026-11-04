import os
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import lightgbm as lgb
from catboost import CatBoostClassifier
import warnings

warnings.filterwarnings("ignore")

# 1. Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "Medicine status"

# 2. Process target
y = train_df[target_col]
if (
    y.dtype == "object"
    or y.dtype.name == "category"
    or not pd.api.types.is_numeric_dtype(y)
):
    unique_vals = sorted(y.dropna().unique())
    val_map = {val: i for i, val in enumerate(unique_vals)}
    y = y.map(val_map).astype(int)
else:
    y = y.astype(int)

train_df = train_df.drop(columns=[target_col])

# Combine for uniform preprocessing
n_train = len(train_df)
combined_df = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)

# 3. Feature engineering
features_df = pd.DataFrame(index=combined_df.index)
parsed_date_series = {}

for col in combined_df.columns:
    s = combined_df[col]

    # Try date parsing if string/object or date-related column name
    if s.dtype == "object" or "date" in col.lower() or "time" in col.lower():
        parsed_dates = pd.to_datetime(s, errors="coerce")
        if parsed_dates.notna().sum() > 0.2 * len(s):
            features_df[f"{col}_year"] = parsed_dates.dt.year
            features_df[f"{col}_month"] = parsed_dates.dt.month
            features_df[f"{col}_day"] = parsed_dates.dt.day
            features_df[f"{col}_dayofweek"] = parsed_dates.dt.dayofweek
            features_df[f"{col}_is_na"] = parsed_dates.isna().astype(int)
            parsed_date_series[col] = parsed_dates
            continue

    # Numeric features
    if pd.api.types.is_numeric_dtype(s):
        features_df[col] = s.fillna(s.median())
    elif s.dtype == "object" or s.dtype.name == "category":
        str_s = s.fillna("").astype(str)
        unique_count = str_s.nunique()
        avg_len = str_s.apply(len).mean()

        # Text features via TF-IDF + SVD
        if avg_len > 30 or unique_count > 300:
            tfidf = TfidfVectorizer(
                max_features=10000,
                stop_words="english",
                ngram_range=(1, 2),
                sublinear_tf=True,
            )
            tfidf_mat = tfidf.fit_transform(str_s)
            n_components = (
                min(24, tfidf_mat.shape[1] - 1) if tfidf_mat.shape[1] > 1 else 0
            )
            if n_components > 0:
                svd = TruncatedSVD(n_components=n_components, random_state=42)
                svd_feats = svd.fit_transform(tfidf_mat)
                for i in range(n_components):
                    features_df[f"{col}_svd_{i}"] = svd_feats[:, i]
            features_df[f"{col}_len"] = str_s.apply(len)
            features_df[f"{col}_word_count"] = str_s.apply(lambda x: len(x.split()))
        else:
            freq = str_s.value_counts(normalize=True)
            features_df[f"{col}_freq"] = str_s.map(freq)
            features_df[col] = str_s.astype("category").cat.codes

# Compute pairwise date difference intervals
date_col_names = list(parsed_date_series.keys())
for i in range(len(date_col_names)):
    d1_name = date_col_names[i]
    d1 = parsed_date_series[d1_name]
    for j in range(i + 1, len(date_col_names)):
        d2_name = date_col_names[j]
        d2 = parsed_date_series[d2_name]
        diff_days = (d1 - d2).dt.total_seconds() / 86400.0
        features_df[f"diff_days_{d1_name}_vs_{d2_name}"] = diff_days

# Sanitize feature column names
clean_cols = []
seen = {}
for c in features_df.columns:
    clean_c = re.sub(r"[^A-Za-z0-9_]", "_", str(c))
    if clean_c in seen:
        seen[clean_c] += 1
        clean_c = f"{clean_c}_{seen[clean_c]}"
    else:
        seen[clean_c] = 0
    clean_cols.append(clean_c)
features_df.columns = clean_cols

# Train / Test split
X = features_df.iloc[:n_train].copy().fillna(-999)
X_test = features_df.iloc[n_train:].copy().fillna(-999)

# 4. Stratified 5-Fold Ensemble Modeling (LightGBM + CatBoost)
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

lgb_oof = np.zeros(len(X))
lgb_test = np.zeros(len(X_test))

cb_oof = np.zeros(len(X))
cb_test = np.zeros(len(X_test))

lgb_params = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "learning_rate": 0.03,
    "num_leaves": 31,
    "max_depth": -1,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "verbosity": -1,
    "random_state": 42,
    "n_jobs": -1,
}

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

    # LightGBM
    trn_data = lgb.Dataset(X_tr, label=y_tr)
    val_data = lgb.Dataset(X_val, label=y_val, reference=trn_data)

    model_lgb = lgb.train(
        lgb_params,
        trn_data,
        num_boost_round=1500,
        valid_sets=[trn_data, val_data],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )

    lgb_oof[val_idx] = model_lgb.predict(X_val, num_iteration=model_lgb.best_iteration)
    lgb_test += (
        model_lgb.predict(X_test, num_iteration=model_lgb.best_iteration) / n_splits
    )

    # CatBoost
    model_cb = CatBoostClassifier(
        iterations=1500,
        learning_rate=0.04,
        depth=6,
        eval_metric="AUC",
        random_seed=42 + fold,
        verbose=False,
        early_stopping_rounds=50,
    )
    model_cb.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=False)

    cb_oof[val_idx] = model_cb.predict_proba(X_val)[:, 1]
    cb_test += model_cb.predict_proba(X_test)[:, 1] / n_splits

# 5. Ensemble Blend & Evaluation
oof_preds = 0.5 * lgb_oof + 0.5 * cb_oof
test_preds = 0.5 * lgb_test + 0.5 * cb_test

cv_auc = roc_auc_score(y, oof_preds)
print(f"5-Fold CV Out-of-Fold ROC AUC (Ensemble): {cv_auc:.5f}")

# 6. Create Submission
os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame(
    {"id": np.arange(len(test_preds)), "Medicine status": test_preds}
)
submission.to_csv("./working/submission.csv", index=False)
print("Submission successfully saved to ./working/submission.csv")
