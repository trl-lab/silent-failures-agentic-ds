import gc
import os
import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from scipy.optimize import minimize
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold
import xgboost as xgb

# 1. Load Data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "Federal Funds"
train_df = train_df.dropna(subset=[target_col]).reset_index(drop=True)

y_raw = train_df[target_col].values
use_log = (y_raw.min() >= 0) and (np.max(y_raw) > 100 or pd.Series(y_raw).skew() > 1.0)

if use_log:
    y = np.log1p(y_raw)
else:
    y = y_raw

n_train = len(train_df)
n_test = len(test_df)
combined = pd.concat(
    [
        train_df.drop(columns=[target_col], errors="ignore"),
        test_df.drop(columns=[target_col], errors="ignore"),
    ],
    ignore_index=True,
)

# 2. Identify Feature Types
text_cols = []
cat_cols = []
num_cols = []

for col in combined.columns:
    if col.lower() in ["id", "index"]:
        continue
    dtype = combined[col].dtype
    if dtype == "object" or str(dtype) == "string" or str(dtype) == "category":
        avg_len = combined[col].dropna().astype(str).str.len().mean()
        unique_cnt = combined[col].nunique()
        if avg_len > 40 or unique_cnt > 500:
            text_cols.append(col)
        else:
            cat_cols.append(col)
    else:
        num_cols.append(col)

# 3. Text Feature Extraction (Word + Character n-grams)
text_combined = (
    combined[text_cols].fillna("").astype(str).agg(" ".join, axis=1)
    if text_cols
    else pd.Series([""] * len(combined))
)

# Word TF-IDF
tfidf_word = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), sublinear_tf=True)
X_tfidf_word = tfidf_word.fit_transform(text_combined)

svd_w = TruncatedSVD(n_components=25, random_state=42)
svd_word_feats = svd_w.fit_transform(X_tfidf_word)
svd_word_cols = [f"svd_w_{i}" for i in range(25)]
svd_word_df = pd.DataFrame(svd_word_feats, columns=svd_word_cols)

# Character TF-IDF
tfidf_char = TfidfVectorizer(
    max_features=15000, ngram_range=(3, 5), analyzer="char_wb", sublinear_tf=True
)
X_tfidf_char = tfidf_char.fit_transform(text_combined)

svd_c = TruncatedSVD(n_components=15, random_state=42)
svd_char_feats = svd_c.fit_transform(X_tfidf_char)
svd_char_cols = [f"svd_c_{i}" for i in range(15)]
svd_char_df = pd.DataFrame(svd_char_feats, columns=svd_char_cols)

# Text statistics
text_stats = pd.DataFrame()
text_stats["char_len"] = text_combined.str.len().astype(np.float32)
text_stats["word_len"] = text_combined.str.split().str.len().astype(np.float32)
text_stats["num_digits"] = text_combined.str.count(r"\d").astype(np.float32)
text_stats["num_caps"] = text_combined.str.count(r"[A-Z]").astype(np.float32)

# 4. Categorical and Numerical Feature Processing
features_df = pd.DataFrame()

# Frequency encoding
for col in cat_cols:
    freq = combined[col].value_counts(normalize=True).to_dict()
    features_df[f"{col}_freq"] = combined[col].map(freq).fillna(0).astype(np.float32)

# Numerical Imputation & Scaling
for col in num_cols:
    med = combined[col].median()
    features_df[col] = combined[col].fillna(med).astype(np.float32)
    features_df[f"{col}_is_na"] = combined[col].isna().astype(np.float32)

features_df = pd.concat([features_df, svd_word_df, svd_char_df, text_stats], axis=1)

# Split features back into Train and Test
X_train_full = features_df.iloc[:n_train].copy().reset_index(drop=True)
X_test_full = features_df.iloc[n_train:].copy().reset_index(drop=True)

X_tfidf_w_train = X_tfidf_word[:n_train]
X_tfidf_w_test = X_tfidf_word[n_train:]
X_tfidf_c_train = X_tfidf_char[:n_train]
X_tfidf_c_test = X_tfidf_char[n_train:]

# 5. Cross-Validation Setup & Out-of-fold Meta-features
kf = KFold(n_splits=5, shuffle=True, random_state=42)

oof_ridge_w = np.zeros(n_train)
test_ridge_w = np.zeros(n_test)
oof_ridge_c = np.zeros(n_train)
test_ridge_c = np.zeros(n_test)

for fold, (trn_idx, val_idx) in enumerate(kf.split(X_train_full, y)):
    # Word Ridge meta feature
    ridge_w = Ridge(alpha=1.0)
    ridge_w.fit(X_tfidf_w_train[trn_idx], y[trn_idx])
    oof_ridge_w[val_idx] = ridge_w.predict(X_tfidf_w_train[val_idx])
    test_ridge_w += ridge_w.predict(X_tfidf_w_test) / 5.0

    # Char Ridge meta feature
    ridge_c = Ridge(alpha=1.0)
    ridge_c.fit(X_tfidf_c_train[trn_idx], y[trn_idx])
    oof_ridge_c[val_idx] = ridge_c.predict(X_tfidf_c_train[val_idx])
    test_ridge_c += ridge_c.predict(X_tfidf_c_test) / 5.0

X_train_full["ridge_word"] = oof_ridge_w
X_test_full["ridge_word"] = test_ridge_w
X_train_full["ridge_char"] = oof_ridge_c
X_test_full["ridge_char"] = test_ridge_c

# Smoothed Target Encoding for Categorical Variables
m_smooth = 10.0
global_mean = y.mean()

for col in cat_cols:
    X_train_full[f"{col}_te"] = np.nan
    X_test_full[f"{col}_te"] = np.nan

    for trn_idx, val_idx in kf.split(X_train_full, y):
        cat_trn = combined.iloc[:n_train].iloc[trn_idx][col]
        df_trn = pd.DataFrame({"cat": cat_trn, "target": y[trn_idx]})
        stats = df_trn.groupby("cat")["target"].agg(["count", "mean"])
        smoothed = (stats["count"] * stats["mean"] + m_smooth * global_mean) / (
            stats["count"] + m_smooth
        )

        cat_val = combined.iloc[:n_train].iloc[val_idx][col]
        X_train_full.loc[val_idx, f"{col}_te"] = cat_val.map(smoothed).fillna(
            global_mean
        )

    df_full = pd.DataFrame({"cat": combined.iloc[:n_train][col], "target": y})
    stats_full = df_full.groupby("cat")["target"].agg(["count", "mean"])
    smoothed_full = (
        stats_full["count"] * stats_full["mean"] + m_smooth * global_mean
    ) / (stats_full["count"] + m_smooth)
    X_test_full[f"{col}_te"] = (
        combined.iloc[n_train:][col].map(smoothed_full).fillna(global_mean)
    )

# Sanitize feature names for compatibility with XGBoost
clean_cols = [
    str(c).replace("[", "_").replace("]", "_").replace("<", "_")
    for c in X_train_full.columns
]
X_train_full.columns = clean_cols
X_test_full.columns = clean_cols

# 6. Tri-Model Ensemble Training
oof_preds_lgb = np.zeros(n_train)
oof_preds_cat = np.zeros(n_train)
oof_preds_xgb = np.zeros(n_train)

test_preds_lgb = np.zeros(n_test)
test_preds_cat = np.zeros(n_test)
test_preds_xgb = np.zeros(n_test)

for fold, (trn_idx, val_idx) in enumerate(kf.split(X_train_full, y)):
    X_tr, y_tr = X_train_full.iloc[trn_idx], y[trn_idx]
    X_va, y_va = X_train_full.iloc[val_idx], y[val_idx]

    # LightGBM
    lgb_train = lgb.Dataset(X_tr, y_tr)
    lgb_val = lgb.Dataset(X_va, y_va, reference=lgb_train)

    params_lgb = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "random_state": 42 + fold,
        "verbosity": -1,
        "n_estimators": 1000,
    }

    model_lgb = lgb.train(
        params_lgb,
        lgb_train,
        valid_sets=[lgb_val],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )

    oof_preds_lgb[val_idx] = model_lgb.predict(X_va)
    test_preds_lgb += model_lgb.predict(X_test_full) / 5.0

    # CatBoost
    model_cat = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        eval_metric="RMSE",
        random_seed=42 + fold,
        verbose=0,
    )
    model_cat.fit(X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=50)

    oof_preds_cat[val_idx] = model_cat.predict(X_va)
    test_preds_cat += model_cat.predict(X_test_full) / 5.0

    # XGBoost
    model_xgb = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        early_stopping_rounds=50,
        tree_method="hist",
        n_jobs=-1,
    )
    model_xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

    oof_preds_xgb[val_idx] = model_xgb.predict(X_va)
    test_preds_xgb += model_xgb.predict(X_test_full) / 5.0

# 7. Optimal Ensemble Weight Optimization via SLSQP
init_weights = np.ones(3) / 3.0
bounds = [(0.0, 1.0)] * 3
constraints = {"type": "eq", "fun": lambda w: 1.0 - np.sum(w)}


def loss_func(w):
    w_norm = w / np.sum(w)
    blend_oof = (
        w_norm[0] * oof_preds_lgb
        + w_norm[1] * oof_preds_cat
        + w_norm[2] * oof_preds_xgb
    )
    if use_log:
        pred_raw = np.clip(np.expm1(blend_oof), 0, None)
    else:
        pred_raw = np.clip(blend_oof, 0, None)
    return root_mean_squared_error(y_raw, pred_raw)


res = minimize(
    loss_func, init_weights, method="SLSQP", bounds=bounds, constraints=constraints
)
best_weights = res.x / np.sum(res.x)

oof_preds_blend = (
    best_weights[0] * oof_preds_lgb
    + best_weights[1] * oof_preds_cat
    + best_weights[2] * oof_preds_xgb
)
test_preds_blend = (
    best_weights[0] * test_preds_lgb
    + best_weights[1] * test_preds_cat
    + best_weights[2] * test_preds_xgb
)

if use_log:
    oof_final = np.expm1(oof_preds_blend)
    test_final = np.expm1(test_preds_blend)
else:
    oof_final = oof_preds_blend
    test_final = test_preds_blend

oof_final = np.clip(oof_final, 0, None)
test_final = np.clip(test_final, 0, None)

cv_rmse = root_mean_squared_error(y_raw, oof_final)
print(f"Validation RMSE: {cv_rmse:.4f}")

# 8. Save Submission
submission = pd.DataFrame(
    {"id": np.arange(len(test_final)), "Federal Funds": test_final}
)
os.makedirs("./working", exist_ok=True)
submission.to_csv("./working/submission.csv", index=False)
