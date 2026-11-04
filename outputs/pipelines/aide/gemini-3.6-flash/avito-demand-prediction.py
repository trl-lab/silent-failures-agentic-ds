import os
import gc
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import root_mean_squared_error
import lightgbm as lgb

os.makedirs("./working", exist_ok=True)

# 1. Load Data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

y_train = train_df["deal_probability"].values.astype(np.float32)
n_train = len(train_df)
n_test = len(test_df)

# 2. Text Features: Separate TF-IDF + Ridge Meta Feature for Title
title_train = train_df["title"].fillna("").astype(str)
title_test = test_df["title"].fillna("").astype(str)
all_title = pd.concat([title_train, title_test], ignore_index=True)

tfidf_title = TfidfVectorizer(max_features=25000, ngram_range=(1, 2), min_df=3)
X_tfidf_title = tfidf_title.fit_transform(all_title)

X_tfidf_title_train = X_tfidf_title[:n_train]
X_tfidf_title_test = X_tfidf_title[n_train:]
del all_title, tfidf_title, X_tfidf_title
gc.collect()

kf_text = KFold(n_splits=5, shuffle=True, random_state=42)
oof_ridge_title = np.zeros(n_train, dtype=np.float32)
test_ridge_title = np.zeros(n_test, dtype=np.float32)

for tr_idx, va_idx in kf_text.split(X_tfidf_title_train):
    ridge = Ridge(alpha=5.0, solver="sag", random_state=42)
    ridge.fit(X_tfidf_title_train[tr_idx], y_train[tr_idx])
    oof_ridge_title[va_idx] = ridge.predict(X_tfidf_title_train[va_idx])
    test_ridge_title += ridge.predict(X_tfidf_title_test) / 5.0

del X_tfidf_title_train, X_tfidf_title_test
gc.collect()

# Text Features: Separate TF-IDF + Ridge Meta Feature for Description
desc_train = train_df["description"].fillna("").astype(str)
desc_test = test_df["description"].fillna("").astype(str)
all_desc = pd.concat([desc_train, desc_test], ignore_index=True)

tfidf_desc = TfidfVectorizer(max_features=35000, ngram_range=(1, 2), min_df=3)
X_tfidf_desc = tfidf_desc.fit_transform(all_desc)

X_tfidf_desc_train = X_tfidf_desc[:n_train]
X_tfidf_desc_test = X_tfidf_desc[n_train:]
del all_desc, tfidf_desc, X_tfidf_desc
gc.collect()

oof_ridge_desc = np.zeros(n_train, dtype=np.float32)
test_ridge_desc = np.zeros(n_test, dtype=np.float32)

for tr_idx, va_idx in kf_text.split(X_tfidf_desc_train):
    ridge = Ridge(alpha=5.0, solver="sag", random_state=42)
    ridge.fit(X_tfidf_desc_train[tr_idx], y_train[tr_idx])
    oof_ridge_desc[va_idx] = ridge.predict(X_tfidf_desc_train[va_idx])
    test_ridge_desc += ridge.predict(X_tfidf_desc_test) / 5.0

del X_tfidf_desc_train, X_tfidf_desc_test
gc.collect()

# 3. Out-of-Fold Target Encoding for High-Cardinality Categoricals
te_cols = ["city", "param_1", "param_2", "param_3", "category_name"]
m_smooth = 10.0
global_mean = float(y_train.mean())

te_oof_dict = {}
te_test_dict = {}

for col in te_cols:
    oof_te = np.zeros(n_train, dtype=np.float32)
    s_train = train_df[col].fillna("missing").astype(str)
    s_test = test_df[col].fillna("missing").astype(str)

    for tr_idx, va_idx in kf_text.split(s_train):
        tr_s = s_train.iloc[tr_idx]
        tr_y = y_train[tr_idx]

        df_tr_col = pd.DataFrame({"cat": tr_s, "target": tr_y})
        stats = df_tr_col.groupby("cat")["target"].agg(["count", "mean"])
        smooth = (stats["count"] * stats["mean"] + m_smooth * global_mean) / (
            stats["count"] + m_smooth
        )

        va_s = s_train.iloc[va_idx]
        oof_te[va_idx] = va_s.map(smooth).fillna(global_mean).values.astype(np.float32)

    df_full_col = pd.DataFrame({"cat": s_train, "target": y_train})
    stats_full = df_full_col.groupby("cat")["target"].agg(["count", "mean"])
    smooth_full = (
        stats_full["count"] * stats_full["mean"] + m_smooth * global_mean
    ) / (stats_full["count"] + m_smooth)

    test_te = s_test.map(smooth_full).fillna(global_mean).values.astype(np.float32)

    te_oof_dict[f"{col}_te"] = oof_te
    te_test_dict[f"{col}_te"] = test_te

# 4. Tabular & Aggregated Feature Engineering
full_df = pd.concat(
    [train_df.drop(columns=["deal_probability"], errors="ignore"), test_df],
    ignore_index=True,
)

# Text length statistics
full_df["title_len"] = full_df["title"].fillna("").str.len()
full_df["desc_len"] = full_df["description"].fillna("").str.len()
full_df["title_word_count"] = full_df["title"].fillna("").str.split().str.len()
full_df["desc_word_count"] = full_df["description"].fillna("").str.split().str.len()

# Numerical & Missing features
full_df["price_isna"] = full_df["price"].isna().astype(np.int8)
full_df["price_log"] = (
    np.log1p(full_df["price"].clip(lower=0)).fillna(-1).astype(np.float32)
)
full_df["item_seq_number_log"] = np.log1p(full_df["item_seq_number"]).astype(np.float32)
full_df["image_top_1"] = full_df["image_top_1"].fillna(-1).astype(np.float32)

# Group price aggregations and relative price features
cat_mean_price = full_df.groupby("category_name")["price_log"].transform("mean")
cat_std_price = full_df.groupby("category_name")["price_log"].transform("std").fillna(0)
full_df["price_diff_cat_mean"] = (full_df["price_log"] - cat_mean_price).astype(
    np.float32
)
full_df["price_zscore_cat"] = (
    (full_df["price_log"] - cat_mean_price) / (cat_std_price + 1e-5)
).astype(np.float32)

param1_mean_price = full_df.groupby("param_1")["price_log"].transform("mean")
param1_std_price = full_df.groupby("param_1")["price_log"].transform("std").fillna(0)
full_df["price_diff_param1_mean"] = (full_df["price_log"] - param1_mean_price).astype(
    np.float32
)
full_df["price_zscore_param1"] = (
    (full_df["price_log"] - param1_mean_price) / (param1_std_price + 1e-5)
).astype(np.float32)

# Activation date features
activation_date = pd.to_datetime(full_df["activation_date"])
full_df["dayofweek"] = activation_date.dt.dayofweek.astype(np.int8)
full_df["day"] = activation_date.dt.day.astype(np.int8)

# Add Target Encoded features
for col_te in te_oof_dict:
    full_df[col_te] = np.concatenate([te_oof_dict[col_te], te_test_dict[col_te]])

# Categorical features
cat_cols = [
    "region",
    "city",
    "parent_category_name",
    "category_name",
    "param_1",
    "param_2",
    "param_3",
    "user_type",
]

for col in cat_cols:
    full_df[col] = full_df[col].astype("category")

features_to_use = (
    [
        "price_isna",
        "price_log",
        "item_seq_number_log",
        "image_top_1",
        "title_len",
        "desc_len",
        "title_word_count",
        "desc_word_count",
        "price_diff_cat_mean",
        "price_zscore_cat",
        "price_diff_param1_mean",
        "price_zscore_param1",
        "dayofweek",
        "day",
    ]
    + list(te_oof_dict.keys())
    + cat_cols
)

X_all = full_df[features_to_use].copy()
X_all["tfidf_ridge_title"] = np.concatenate([oof_ridge_title, test_ridge_title])
X_all["tfidf_ridge_desc"] = np.concatenate([oof_ridge_desc, test_ridge_desc])

X_train = X_all.iloc[:n_train].copy()
X_test = X_all.iloc[n_train:].copy()

del full_df, train_df, test_df, X_all
gc.collect()

# 5. LightGBM 5-Fold Cross Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(n_train, dtype=np.float32)
test_preds = np.zeros(n_test, dtype=np.float32)

params = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.08,
    "num_leaves": 63,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_tr, y_tr = X_train.iloc[train_idx], y_train[train_idx]
    X_va, y_va = X_train.iloc[val_idx], y_train[val_idx]

    dtrain = lgb.Dataset(X_tr, label=y_tr)
    dval = lgb.Dataset(X_va, label=y_va, reference=dtrain)

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        valid_sets=[dtrain, dval],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
    )

    oof_preds[val_idx] = model.predict(X_va)
    test_preds += model.predict(X_test) / kf.n_splits

# Post-processing predictions
oof_preds = np.clip(oof_preds, 0.0, 1.0)
test_preds = np.clip(test_preds, 0.0, 1.0)

# Evaluate metric
cv_rmse = root_mean_squared_error(y_train, oof_preds)
print(f"5-Fold CV RMSE: {cv_rmse:.5f}")

# Save Test Predictions
submission = pd.DataFrame({"id": np.arange(n_test), "deal_probability": test_preds})
submission.to_csv("./working/submission.csv", index=False)
