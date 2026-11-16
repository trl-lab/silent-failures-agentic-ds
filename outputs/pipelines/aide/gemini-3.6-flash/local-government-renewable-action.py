import os
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Ridge
from sklearn.metrics import root_mean_squared_error
import lightgbm as lgb
from catboost import CatBoostRegressor

warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)

os.makedirs("./working", exist_ok=True)

# Load datasets
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

target_col = "Size of Purchase (MW)"
y = train[target_col].values
train_df = train.drop(columns=[target_col]).copy()
test_df = test.copy()

n_train = len(train_df)
combined = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)

# Classify column types
date_cols = []
cat_cols = []
num_cols = []
text_cols = []

for col in combined.columns:
    if pd.api.types.is_datetime64_any_dtype(combined[col]):
        date_cols.append(col)
    elif combined[col].dtype == "object" or combined[col].dtype == "category":
        sample_vals = combined[col].dropna().astype(str)
        if (
            len(sample_vals) > 0
            and sample_vals.str.contains(r"^\d{4}-\d{2}-\d{2}").all()
        ):
            try:
                combined[col] = pd.to_datetime(combined[col])
                date_cols.append(col)
                continue
            except Exception:
                pass

        avg_len = sample_vals.str.len().mean() if len(sample_vals) > 0 else 0
        nunique = combined[col].nunique()
        if avg_len > 30 or nunique > 100:
            text_cols.append(col)
        else:
            cat_cols.append(col)
    else:
        num_cols.append(col)

# Datetime feature extraction
for col in date_cols:
    combined[f"{col}_year"] = combined[col].dt.year
    combined[f"{col}_month"] = combined[col].dt.month
    combined[f"{col}_day"] = combined[col].dt.day
    combined[f"{col}_dayofweek"] = combined[col].dt.dayofweek
    combined[f"{col}_quarter"] = combined[col].dt.quarter
    combined[f"{col}_is_month_end"] = combined[col].dt.is_month_end.astype(int)
    combined.drop(columns=[col], inplace=True)

# Frequency encoding for categoricals
for col in cat_cols:
    combined[col] = combined[col].fillna("missing").astype(str)
    freq = combined[col].value_counts().to_dict()
    combined[f"{col}_freq"] = combined[col].map(freq)

# Text TF-IDF processing
tfidf_mat = None
if text_cols:
    combined_text = combined[text_cols].astype(str).agg(" ".join, axis=1)
    combined["text_len"] = combined_text.str.len()
    combined["text_word_count"] = combined_text.str.split().str.len()
    combined["text_digit_count"] = combined_text.apply(
        lambda s: sum(c.isdigit() for c in s)
    )

    tfidf = TfidfVectorizer(
        max_features=10000, ngram_range=(1, 2), stop_words="english"
    )
    tfidf_mat = tfidf.fit_transform(combined_text)

    n_components = min(16, tfidf_mat.shape[1] - 1) if tfidf_mat.shape[1] > 1 else 1
    if n_components > 0:
        svd = TruncatedSVD(n_components=n_components, random_state=SEED)
        svd_feats = svd.fit_transform(tfidf_mat)
        for i in range(n_components):
            combined[f"tfidf_svd_{i}"] = svd_feats[:, i]

    combined.drop(columns=text_cols, inplace=True)

combined["num_missing"] = combined.isnull().sum(axis=1)

X_train = combined.iloc[:n_train].copy()
X_test = combined.iloc[n_train:].copy()

# 5-fold CV setup
kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

oof_preds_lgb = np.zeros(n_train)
oof_preds_cat = np.zeros(n_train)
test_preds_lgb = np.zeros(len(X_test))
test_preds_cat = np.zeros(len(X_test))

# Setup target encoding and text ridge prediction columns
for c in cat_cols:
    X_train[f"{c}_te"] = 0.0
    X_test[f"{c}_te"] = 0.0

X_train["text_ridge_pred"] = 0.0
X_test["text_ridge_pred"] = 0.0

X_train_lgb = X_train.copy()
X_test_lgb = X_test.copy()
for col in cat_cols:
    X_train_lgb[col] = X_train_lgb[col].astype("category")
    X_test_lgb[col] = X_test_lgb[col].astype("category")

X_train_cb = X_train.copy()
X_test_cb = X_test.copy()

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y)):
    y_tr, y_val = y[train_idx], y[val_idx]
    global_mean = y_tr.mean()

    # Smoothed Out-of-fold target encoding (m-estimate, m=10.0)
    for c in cat_cols:
        tr_cat = X_train.iloc[train_idx][c]
        stats = (
            pd.DataFrame({"cat": tr_cat, "target": y_tr})
            .groupby("cat")["target"]
            .agg(["count", "mean"])
        )
        te_map = (
            (stats["count"] * stats["mean"] + 10.0 * global_mean)
            / (stats["count"] + 10.0)
        ).to_dict()

        te_tr = tr_cat.map(te_map).fillna(global_mean).astype(float).values
        te_val = (
            X_train.iloc[val_idx][c]
            .map(te_map)
            .fillna(global_mean)
            .astype(float)
            .values
        )
        te_test = X_test[c].map(te_map).fillna(global_mean).astype(float).values

        X_train_lgb.iloc[train_idx, X_train_lgb.columns.get_loc(f"{c}_te")] = te_tr
        X_train_lgb.iloc[val_idx, X_train_lgb.columns.get_loc(f"{c}_te")] = te_val
        X_test_lgb.iloc[:, X_test_lgb.columns.get_loc(f"{c}_te")] += te_test / 5.0

        X_train_cb.iloc[train_idx, X_train_cb.columns.get_loc(f"{c}_te")] = te_tr
        X_train_cb.iloc[val_idx, X_train_cb.columns.get_loc(f"{c}_te")] = te_val
        X_test_cb.iloc[:, X_test_cb.columns.get_loc(f"{c}_te")] += te_test / 5.0

    # Leakage-free TF-IDF Ridge text meta-feature generation
    if tfidf_mat is not None:
        tfidf_tr_fold = tfidf_mat[:n_train][train_idx]
        tfidf_val_fold = tfidf_mat[:n_train][val_idx]
        tfidf_test_fold = tfidf_mat[n_train:]

        inner_kf = KFold(n_splits=4, shuffle=True, random_state=SEED + fold)
        ridge_tr_feature = np.zeros(len(train_idx))

        for in_tr_idx, in_val_idx in inner_kf.split(train_idx):
            r_inner = Ridge(alpha=10.0, random_state=SEED)
            r_inner.fit(tfidf_tr_fold[in_tr_idx], y_tr[in_tr_idx])
            ridge_tr_feature[in_val_idx] = r_inner.predict(tfidf_tr_fold[in_val_idx])

        r_full = Ridge(alpha=10.0, random_state=SEED)
        r_full.fit(tfidf_tr_fold, y_tr)
        ridge_val_feature = r_full.predict(tfidf_val_fold)
        ridge_test_feature = r_full.predict(tfidf_test_fold)

        X_train_lgb.iloc[train_idx, X_train_lgb.columns.get_loc("text_ridge_pred")] = (
            ridge_tr_feature
        )
        X_train_lgb.iloc[val_idx, X_train_lgb.columns.get_loc("text_ridge_pred")] = (
            ridge_val_feature
        )
        X_test_lgb.iloc[:, X_test_lgb.columns.get_loc("text_ridge_pred")] += (
            ridge_test_feature / 5.0
        )

        X_train_cb.iloc[train_idx, X_train_cb.columns.get_loc("text_ridge_pred")] = (
            ridge_tr_feature
        )
        X_train_cb.iloc[val_idx, X_train_cb.columns.get_loc("text_ridge_pred")] = (
            ridge_val_feature
        )
        X_test_cb.iloc[:, X_test_cb.columns.get_loc("text_ridge_pred")] += (
            ridge_test_feature / 5.0
        )

    # LightGBM model training
    X_tr_lgb, X_val_lgb = X_train_lgb.iloc[train_idx], X_train_lgb.iloc[val_idx]
    lgb_model = lgb.LGBMRegressor(
        objective="regression",
        metric="rmse",
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=SEED + fold,
        verbose=-1,
    )
    lgb_model.fit(
        X_tr_lgb,
        y_tr,
        eval_set=[(X_val_lgb, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
    )
    oof_preds_lgb[val_idx] = lgb_model.predict(X_val_lgb)
    test_preds_lgb += lgb_model.predict(X_test_lgb) / 5.0

    # CatBoost model training
    X_tr_cb, X_val_cb = X_train_cb.iloc[train_idx], X_train_cb.iloc[val_idx]
    cat_model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.04,
        depth=6,
        loss_function="RMSE",
        random_seed=SEED + fold,
        verbose=0,
        cat_features=cat_cols,
    )
    cat_model.fit(X_tr_cb, y_tr, eval_set=(X_val_cb, y_val), early_stopping_rounds=50)
    oof_preds_cat[val_idx] = cat_model.predict(X_val_cb)
    test_preds_cat += cat_model.predict(X_test_cb) / 5.0

# Ensemble predictions and post-processing
oof_preds = np.clip(0.5 * oof_preds_lgb + 0.5 * oof_preds_cat, 0.0, None)
test_preds = np.clip(0.5 * test_preds_lgb + 0.5 * test_preds_cat, 0.0, None)

cv_rmse = root_mean_squared_error(y, oof_preds)
print(f"OOF RMSE: {cv_rmse:.5f}")

# Save output submission file
submission = pd.DataFrame({"id": np.arange(len(test_preds)), target_col: test_preds})
submission.to_csv("./working/submission.csv", index=False)
