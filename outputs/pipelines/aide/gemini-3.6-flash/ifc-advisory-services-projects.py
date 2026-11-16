import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import root_mean_squared_error
import lightgbm as lgb
import xgboost as xgb

# Load dataset
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "Estimated Total Budget ($)"
y = train_df[target_col].values
y_log = np.log1p(np.maximum(0, y))

n_train = len(train_df)
df = pd.concat(
    [train_df.drop(columns=[target_col], errors="ignore"), test_df], ignore_index=True
)

# Identify column types
date_cols = []
text_cols = []
cat_cols = []
num_cols = []

for col in df.columns:
    if col.lower() == "id":
        continue
    if "date" in col.lower() or "time" in col.lower():
        date_cols.append(col)
    elif df[col].dtype == "object" or df[col].dtype.name == "category":
        sample_vals = df[col].dropna().astype(str)
        if len(sample_vals) > 0 and sample_vals.str.split().str.len().mean() > 3:
            text_cols.append(col)
        else:
            cat_cols.append(col)
    else:
        num_cols.append(col)

# Date feature extraction & pairwise differences
parsed_dates = {}
day_cols = []
for col in date_cols:
    parsed = pd.to_datetime(df[col], errors="coerce")
    parsed_dates[col] = parsed
    df[f"{col}_year"] = parsed.dt.year
    df[f"{col}_month"] = parsed.dt.month
    df[f"{col}_quarter"] = parsed.dt.quarter
    df[f"{col}_dayofweek"] = parsed.dt.dayofweek
    days_feat = f"{col}_days"
    df[days_feat] = (parsed - pd.Timestamp("2000-01-01")).dt.days
    day_cols.append(days_feat)

date_col_list = list(parsed_dates.keys())
for i in range(len(date_col_list)):
    for j in range(i + 1, len(date_col_list)):
        c1, c2 = date_col_list[i], date_col_list[j]
        df[f"{c1}_diff_{c2}"] = (parsed_dates[c1] - parsed_dates[c2]).dt.days

# Group statistics: mean and std of temporal milestone days per category
for c_cat in cat_cols:
    for c_day in day_cols:
        grp_mean = df.groupby(c_cat)[c_day].transform("mean")
        grp_std = df.groupby(c_cat)[c_day].transform("std").fillna(0)
        df[f"{c_cat}_{c_day}_grp_mean"] = grp_mean
        df[f"{c_cat}_{c_day}_grp_std"] = grp_std

# Text surface statistics
for col in text_cols:
    s = df[col].fillna("").astype(str)
    df[f"{col}_char_len"] = s.str.len()
    df[f"{col}_word_count"] = s.str.split().str.len()
    df[f"{col}_digit_count"] = s.str.count(r"\d")

# Unified text combining all text columns
if text_cols:
    combined_text = df[text_cols].fillna("").astype(str).agg(" ".join, axis=1)
else:
    combined_text = pd.Series([""] * len(df))

# Text TF-IDF with sublinear TF scaling, SVD, and OOF Ridge meta-features
svd_features_list = []
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Individual text column embeddings
text_meta_train = (
    np.zeros((n_train, len(text_cols))) if text_cols else np.zeros((n_train, 0))
)
text_meta_test = (
    np.zeros((len(test_df), len(text_cols)))
    if text_cols
    else np.zeros((len(test_df), 0))
)

for i, col in enumerate(text_cols):
    text_data = df[col].fillna("").astype(str)
    tfidf = TfidfVectorizer(
        max_features=5000, stop_words="english", ngram_range=(1, 2), sublinear_tf=True
    )
    X_tfidf = tfidf.fit_transform(text_data)

    svd = TruncatedSVD(n_components=15, random_state=42)
    svd_feats = svd.fit_transform(X_tfidf)
    svd_df = pd.DataFrame(svd_feats, columns=[f"{col}_svd_{j}" for j in range(15)])
    svd_features_list.append(svd_df)

    X_tfidf_train = X_tfidf[:n_train]
    X_tfidf_test = X_tfidf[n_train:]

    for tr_idx, va_idx in kf.split(X_tfidf_train):
        ridge = RidgeCV(alphas=np.logspace(-2, 3, 6))
        ridge.fit(X_tfidf_train[tr_idx], y_log[tr_idx])
        text_meta_train[va_idx, i] = ridge.predict(X_tfidf_train[va_idx])

    ridge = RidgeCV(alphas=np.logspace(-2, 3, 6))
    ridge.fit(X_tfidf_train, y_log)
    text_meta_test[:, i] = ridge.predict(X_tfidf_test)

# Combined text processing
if text_cols:
    tfidf_comb = TfidfVectorizer(
        max_features=10000, stop_words="english", ngram_range=(1, 2), sublinear_tf=True
    )
    X_comb = tfidf_comb.fit_transform(combined_text)
    svd_comb = TruncatedSVD(n_components=20, random_state=42)
    comb_svd_feats = svd_comb.fit_transform(X_comb)
    comb_svd_df = pd.DataFrame(
        comb_svd_feats, columns=[f"comb_text_svd_{j}" for j in range(20)]
    )
    svd_features_list.append(comb_svd_df)

    comb_train = X_comb[:n_train]
    comb_test = X_comb[n_train:]
    comb_meta_train = np.zeros(n_train)
    comb_meta_test = np.zeros(len(test_df))

    for tr_idx, va_idx in kf.split(comb_train):
        ridge = RidgeCV(alphas=np.logspace(-2, 3, 6))
        ridge.fit(comb_train[tr_idx], y_log[tr_idx])
        comb_meta_train[va_idx] = ridge.predict(comb_train[va_idx])

    ridge = RidgeCV(alphas=np.logspace(-2, 3, 6))
    ridge.fit(comb_train, y_log)
    comb_meta_test = ridge.predict(comb_test)

    df["comb_text_ridge_pred"] = np.concatenate([comb_meta_train, comb_meta_test])

if svd_features_list:
    df = pd.concat([df] + svd_features_list, axis=1)

for i, col in enumerate(text_cols):
    df[f"{col}_ridge_pred"] = np.concatenate(
        [text_meta_train[:, i], text_meta_test[:, i]]
    )

df.drop(columns=text_cols + date_cols, errors="ignore", inplace=True)

# Frequency encoding & Label encoding
for col in cat_cols:
    freq = df[col].value_counts(normalize=True).to_dict()
    df[f"{col}_freq"] = df[col].map(freq)
    df[col] = df[col].astype(str).astype("category").cat.codes

X_train = df.iloc[:n_train].copy()
X_test = df.iloc[n_train:].copy()

if "id" in X_train.columns:
    X_train.drop(columns=["id"], inplace=True)
    X_test.drop(columns=["id"], inplace=True)

# Multi-scale Out-of-Fold Target Encoding
global_mean = np.mean(y_log)
smooth_factors = [5, 20]

for col in cat_cols:
    for sm in smooth_factors:
        X_train[f"{col}_te_sm{sm}"] = np.nan
        X_test[f"{col}_te_sm{sm}"] = np.nan

        for tr_idx, va_idx in kf.split(X_train):
            tr_data = pd.DataFrame(
                {"cat": X_train.iloc[tr_idx][col], "target": y_log[tr_idx]}
            )
            stats = tr_data.groupby("cat")["target"].agg(["count", "mean"])
            smooth = (stats["count"] * stats["mean"] + sm * global_mean) / (
                stats["count"] + sm
            )
            X_train.iloc[va_idx, X_train.columns.get_loc(f"{col}_te_sm{sm}")] = (
                X_train.iloc[va_idx][col].map(smooth).fillna(global_mean)
            )

        stats = (
            pd.DataFrame({"cat": X_train[col], "target": y_log})
            .groupby("cat")["target"]
            .agg(["count", "mean"])
        )
        smooth = (stats["count"] * stats["mean"] + sm * global_mean) / (
            stats["count"] + sm
        )
        X_test[f"{col}_te_sm{sm}"] = X_test[col].map(smooth).fillna(global_mean)

# Formatting for models
X_train_lgb = X_train.copy()
X_test_lgb = X_test.copy()
for col in cat_cols:
    X_train_lgb[col] = X_train_lgb[col].astype("category")
    X_test_lgb[col] = X_test_lgb[col].astype("category")

X_train_num = X_train.copy()
X_test_num = X_test.copy()
for col in cat_cols:
    X_train_num[col] = X_train_num[col].astype("int32")
    X_test_num[col] = X_test_num[col].astype("int32")

# Training models: LightGBM, XGBoost, HistGradientBoosting
oof_lgb = np.zeros(n_train)
oof_xgb = np.zeros(n_train)
oof_hgb = np.zeros(n_train)

test_lgb = np.zeros(len(test_df))
test_xgb = np.zeros(len(test_df))
test_hgb = np.zeros(len(test_df))

for tr_idx, va_idx in kf.split(X_train):
    # LightGBM Regressor
    X_tr_lgb, y_tr = X_train_lgb.iloc[tr_idx], y_log[tr_idx]
    X_va_lgb, y_va = X_train_lgb.iloc[va_idx], y_log[va_idx]

    model_lgb = lgb.LGBMRegressor(
        n_estimators=1200,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=-1,
    )
    model_lgb.fit(
        X_tr_lgb,
        y_tr,
        eval_set=[(X_va_lgb, y_va)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    oof_lgb[va_idx] = model_lgb.predict(X_va_lgb)
    test_lgb += model_lgb.predict(X_test_lgb) / kf.n_splits

    # XGBoost Regressor
    X_tr_num = X_train_num.iloc[tr_idx]
    X_va_num = X_train_num.iloc[va_idx]

    model_xgb = xgb.XGBRegressor(
        n_estimators=1200,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        early_stopping_rounds=50,
        tree_method="hist",
    )
    model_xgb.fit(X_tr_num, y_tr, eval_set=[(X_va_num, y_va)], verbose=False)
    oof_xgb[va_idx] = model_xgb.predict(X_va_num)
    test_xgb += model_xgb.predict(X_test_num) / kf.n_splits

    # HistGradientBoostingRegressor
    model_hgb = HistGradientBoostingRegressor(
        max_iter=1000,
        learning_rate=0.03,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        random_state=42,
        early_stopping=True,
        n_iter_no_change=50,
    )
    model_hgb.fit(X_tr_num, y_tr)
    oof_hgb[va_idx] = model_hgb.predict(X_va_num)
    test_hgb += model_hgb.predict(X_test_num) / kf.n_splits

# Convert raw log predictions to linear dollar scale before blending
oof_lgb_raw = np.clip(np.expm1(oof_lgb), 0, None)
oof_xgb_raw = np.clip(np.expm1(oof_xgb), 0, None)
oof_hgb_raw = np.clip(np.expm1(oof_hgb), 0, None)

test_lgb_raw = np.clip(np.expm1(test_lgb), 0, None)
test_xgb_raw = np.clip(np.expm1(test_xgb), 0, None)
test_hgb_raw = np.clip(np.expm1(test_hgb), 0, None)


# Raw Scale Linear Blending with Bias Correction
def loss_func(params):
    w1, w2, w3, bias = params
    pred = w1 * oof_lgb_raw + w2 * oof_xgb_raw + w3 * oof_hgb_raw + bias
    pred = np.clip(pred, 0, None)
    return root_mean_squared_error(y, pred)


bounds = [(0.0, None), (0.0, None), (0.0, None), (None, None)]
res = minimize(
    loss_func,
    [0.33, 0.33, 0.33, 0.0],
    method="SLSQP",
    bounds=bounds,
)
best_params = res.x

oof_preds = np.clip(
    best_params[0] * oof_lgb_raw
    + best_params[1] * oof_xgb_raw
    + best_params[2] * oof_hgb_raw
    + best_params[3],
    0,
    None,
)

cv_rmse = root_mean_squared_error(y, oof_preds)
print(f"Validation Metric: {cv_rmse:.4f}")

# Save final test predictions
test_preds = np.clip(
    best_params[0] * test_lgb_raw
    + best_params[1] * test_xgb_raw
    + best_params[2] * test_hgb_raw
    + best_params[3],
    0,
    None,
)

os.makedirs("./working", exist_ok=True)
sub = pd.DataFrame({"id": np.arange(len(test_preds)), target_col: test_preds})
sub.to_csv("./working/submission.csv", index=False)
