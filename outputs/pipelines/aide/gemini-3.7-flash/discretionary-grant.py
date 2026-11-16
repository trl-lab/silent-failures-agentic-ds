import os
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from catboost import CatBoostRegressor, Pool

# 1. Load Data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "AwardTotal"
y = train_df[target_col].values
X_train_raw = train_df.drop(columns=[target_col]).copy()
X_test_raw = test_df.copy()

n_train = len(X_train_raw)
df_all = pd.concat([X_train_raw, X_test_raw], axis=0).reset_index(drop=True)

# 2. Identify Column Types
date_cols = []
text_cols = []
cat_cols = []
num_cols = []

for col in df_all.columns:
    if np.issubdtype(df_all[col].dtype, np.datetime64):
        date_cols.append(col)
    elif df_all[col].dtype == "object" or df_all[col].dtype.name == "category":
        sample_lens = df_all[col].dropna().astype(str).str.len()
        avg_len = sample_lens.mean() if len(sample_lens) > 0 else 0
        n_unique = df_all[col].nunique()
        if avg_len > 60 or (avg_len > 25 and n_unique > 500):
            text_cols.append(col)
        else:
            cat_cols.append(col)
    else:
        num_cols.append(col)

# 3. Process Datetime Features
date_features = pd.DataFrame(index=df_all.index)
for col in date_cols:
    dt_series = pd.to_datetime(df_all[col], errors="coerce")
    date_features[f"{col}_year"] = dt_series.dt.year.fillna(-1).astype(float)
    date_features[f"{col}_month"] = dt_series.dt.month.fillna(-1).astype(float)
    date_features[f"{col}_day"] = dt_series.dt.day.fillna(-1).astype(float)
    date_features[f"{col}_dayofweek"] = dt_series.dt.dayofweek.fillna(-1).astype(float)
    date_features[f"{col}_quarter"] = dt_series.dt.quarter.fillna(-1).astype(float)

    # Month cyclicity
    month_norm = 2 * np.pi * dt_series.dt.month / 12.0
    date_features[f"{col}_sin_month"] = np.sin(month_norm).fillna(0)
    date_features[f"{col}_cos_month"] = np.cos(month_norm).fillna(0)

# Date differences between pairs of date columns
if len(date_cols) >= 2:
    for i in range(len(date_cols)):
        for j in range(i + 1, len(date_cols)):
            col1, col2 = date_cols[i], date_cols[j]
            dt1 = pd.to_datetime(df_all[col1], errors="coerce")
            dt2 = pd.to_datetime(df_all[col2], errors="coerce")
            diff_days = (dt2 - dt1).dt.total_seconds() / 86400.0
            date_features[f"{col1}_{col2}_diff_days"] = diff_days

# 4. Process Numeric Features
num_features = df_all[num_cols].copy()

# 5. Process Categorical Features (Native strings for CatBoost & Frequency features)
native_cat_features = pd.DataFrame(index=df_all.index)
cat_freq_features = pd.DataFrame(index=df_all.index)
cat_col_names = []

for col in cat_cols:
    val_series = df_all[col].fillna("__MISSING__").astype(str)
    # Frequency encoding
    freq = val_series.value_counts(normalize=True)
    cat_freq_features[f"{col}_freq"] = val_series.map(freq).astype(float)

    # Native categorical feature (string representation for CatBoost)
    cat_col_name = f"{col}_native"
    native_cat_features[cat_col_name] = val_series
    cat_col_names.append(cat_col_name)

# Group Aggregations: Compute summary stats of key numeric/duration columns grouped by top categoricals
all_num_and_date = pd.concat([num_features, date_features], axis=1)
group_agg_features = pd.DataFrame(index=df_all.index)

if len(all_num_and_date.columns) > 0 and len(cat_cols) > 0:
    candidate_num_cols = all_num_and_date.columns[
        :5
    ]  # Select primary numerical/duration signals
    for c_col in cat_cols[:4]:
        c_series = df_all[c_col].fillna("__MISSING__").astype(str)
        for n_col in candidate_num_cols:
            grouped = all_num_and_date.groupby(c_series)[n_col]
            group_mean = c_series.map(grouped.mean()).astype(float)
            group_std = c_series.map(grouped.std()).fillna(0).astype(float)
            group_agg_features[f"{c_col}_{n_col}_grp_mean"] = group_mean
            group_agg_features[f"{c_col}_{n_col}_grp_std"] = group_std

# 6. Process Text Features: TF-IDF + TruncatedSVD + Lexical stats
text_features = pd.DataFrame(index=df_all.index)
for col in text_cols:
    corpus = df_all[col].fillna("").astype(str)

    # Lexical features
    text_features[f"{col}_char_len"] = corpus.str.len()
    text_features[f"{col}_word_count"] = corpus.apply(lambda s: len(s.split()))
    text_features[f"{col}_digit_count"] = corpus.apply(
        lambda s: sum(c.isdigit() for c in s)
    )
    text_features[f"{col}_upper_count"] = corpus.apply(
        lambda s: sum(c.isupper() for c in s)
    )

    # TF-IDF + SVD
    tfidf = TfidfVectorizer(
        max_features=35000,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
    )
    tfidf_mat = tfidf.fit_transform(corpus)
    n_components = (
        min(40, tfidf_mat.shape[1] - 1)
        if tfidf_mat.shape[1] > 40
        else max(1, tfidf_mat.shape[1] - 1)
    )
    if n_components > 1:
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        svd_mat = svd.fit_transform(tfidf_mat)
        for comp_idx in range(n_components):
            text_features[f"{col}_svd_{comp_idx}"] = svd_mat[:, comp_idx]

# Combine all engineered features
X_all = pd.concat(
    [
        num_features,
        date_features,
        cat_freq_features,
        group_agg_features,
        text_features,
        native_cat_features,
    ],
    axis=1,
)

X_train = X_all.iloc[:n_train].reset_index(drop=True)
X_test = X_all.iloc[n_train:].reset_index(drop=True)

# 7. 5-Fold Cross Validation Modeling with Native CatBoost categorical support
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(n_train)
test_preds = np.zeros(len(X_test))

cb_params = {
    "iterations": 2000,
    "learning_rate": 0.035,
    "depth": 6,
    "l2_leaf_reg": 3.0,
    "loss_function": "RMSE",
    "eval_metric": "RMSE",
    "random_seed": 42,
    "verbose": False,
}

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y)):
    X_tr, y_tr = X_train.iloc[train_idx], y[train_idx]
    X_va, y_va = X_train.iloc[val_idx], y[val_idx]

    train_pool = Pool(X_tr, y_tr, cat_features=cat_col_names)
    val_pool = Pool(X_va, y_va, cat_features=cat_col_names)
    test_pool = Pool(X_test, cat_features=cat_col_names)

    model = CatBoostRegressor(**cb_params)
    model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=75, verbose=False)

    oof_preds[val_idx] = model.predict(val_pool)
    test_preds += model.predict(test_pool) / kf.n_splits

# 8. Evaluation Metric
cv_rmse = root_mean_squared_error(y, oof_preds)
print(f"5-Fold CV Out-Of-Fold RMSE: {cv_rmse:.5f}")

# 9. Save Submission
os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame({"id": np.arange(len(test_preds)), "AwardTotal": test_preds})
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved successfully to ./working/submission.csv")
