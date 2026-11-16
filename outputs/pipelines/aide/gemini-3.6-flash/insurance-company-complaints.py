import os
import warnings
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")

# 1. Load Data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "Recovery"
y = train_df[target_col].values.astype(float)
X_train_raw = train_df.drop(columns=[target_col])
X_test_raw = test_df.copy()

n_train = len(train_df)
combined = pd.concat([X_train_raw, X_test_raw], axis=0).reset_index(drop=True)

# 2. Feature Engineering
feature_df = pd.DataFrame(index=combined.index)

# Detect Column Types
date_cols = []
text_cols = []
cat_cols = []
num_cols = []

for col in combined.columns:
    if pd.api.types.is_datetime64_any_dtype(combined[col]):
        date_cols.append(col)
    elif combined[col].dtype == "object" or isinstance(
        combined[col].dtype, pd.CategoricalDtype
    ):
        non_null = combined[col].dropna()
        if len(non_null) > 0 and non_null.astype(str).str.len().mean() > 30:
            text_cols.append(col)
        else:
            cat_cols.append(col)
    else:
        try:
            parsed = pd.to_datetime(combined[col], errors="coerce")
            if parsed.notnull().sum() > 0.8 * len(combined):
                combined[col] = parsed
                date_cols.append(col)
                continue
        except Exception:
            pass
        num_cols.append(col)

# Date Features
for col in date_cols:
    dt = pd.to_datetime(combined[col])
    feature_df[f"{col}_year"] = dt.dt.year
    feature_df[f"{col}_month"] = dt.dt.month
    feature_df[f"{col}_day"] = dt.dt.day
    feature_df[f"{col}_dayofweek"] = dt.dt.dayofweek
    feature_df[f"{col}_dayofyear"] = dt.dt.dayofyear
    feature_df[f"{col}_quarter"] = dt.dt.quarter
    feature_df[f"{col}_is_month_end"] = dt.dt.is_month_end.astype(float)
    min_date = dt.min()
    if pd.notnull(min_date):
        feature_df[f"{col}_elapsed_days"] = (dt - min_date).dt.total_seconds() / 86400.0

if len(date_cols) >= 2:
    for i in range(len(date_cols)):
        for j in range(i + 1, len(date_cols)):
            dt1 = pd.to_datetime(combined[date_cols[i]])
            dt2 = pd.to_datetime(combined[date_cols[j]])
            feature_df[f"{date_cols[i]}_{date_cols[j]}_diff"] = (
                dt1 - dt2
            ).dt.total_seconds() / 86400.0

# Text Features: Multi-granularity TF-IDF + RidgeCV OOF Meta-features
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for col in text_cols:
    text_data = combined[col].fillna("").astype(str)

    # Surface statistics
    feature_df[f"{col}_char_len"] = text_data.str.len()
    feature_df[f"{col}_word_count"] = text_data.str.split().str.len()
    feature_df[f"{col}_digit_count"] = text_data.str.count(r"\d")
    feature_df[f"{col}_upper_count"] = text_data.str.count(r"[A-Z]")

    try:
        # Word-level TF-IDF
        tfidf_word = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 3),
            stop_words="english",
            sublinear_tf=True,
        )
        word_mat = tfidf_word.fit_transform(text_data)

        # Character-level TF-IDF
        tfidf_char = TfidfVectorizer(
            max_features=10000,
            analyzer="char",
            ngram_range=(3, 5),
            sublinear_tf=True,
        )
        char_mat = tfidf_char.fit_transform(text_data)

        # Combine word and char TF-IDF matrices
        combined_tfidf = hstack([word_mat, char_mat]).tocsr()

        # Truncated SVD dimensionality reduction
        svd = TruncatedSVD(n_components=15, random_state=42)
        svd_feats = svd.fit_transform(combined_tfidf)
        for k in range(15):
            feature_df[f"{col}_svd_{k}"] = svd_feats[:, k]

        tfidf_train = combined_tfidf[:n_train]
        tfidf_test = combined_tfidf[n_train:]

        oof_ridge = np.zeros(n_train)
        test_ridge = np.zeros(len(X_test_raw))

        alphas = np.logspace(-2, 3, 6)

        for tr_idx, val_idx in kf.split(X_train_raw):
            ridge = RidgeCV(alphas=alphas)
            ridge.fit(tfidf_train[tr_idx], y[tr_idx])
            oof_ridge[val_idx] = ridge.predict(tfidf_train[val_idx])
            test_ridge += ridge.predict(tfidf_test) / kf.n_splits

        feature_df[f"{col}_ridge_oof"] = np.concatenate([oof_ridge, test_ridge])
    except Exception:
        pass

# Categorical Features
for col in cat_cols:
    cat_series = combined[col].fillna("__MISSING__").astype(str)
    freqs = cat_series.value_counts(normalize=True).to_dict()
    feature_df[f"{col}_freq"] = cat_series.map(freqs)
    feature_df[f"{col}_cat"] = cat_series

    cat_train = cat_series.iloc[:n_train]
    cat_test = cat_series.iloc[n_train:]
    global_mean = y.mean()

    oof_te = np.zeros(n_train)
    test_te = np.zeros(len(X_test_raw))

    for tr_idx, val_idx in kf.split(X_train_raw):
        tr_cat = cat_train.iloc[tr_idx]
        tr_y = y[tr_idx]
        stats = (
            pd.DataFrame({"cat": tr_cat, "target": tr_y})
            .groupby("cat")["target"]
            .agg(["count", "mean"])
        )
        smooth = 10
        stats["smooth_mean"] = (
            stats["count"] * stats["mean"] + smooth * global_mean
        ) / (stats["count"] + smooth)
        te_map = stats["smooth_mean"].to_dict()

        oof_te[val_idx] = cat_train.iloc[val_idx].map(te_map).fillna(global_mean)
        test_te += cat_test.map(te_map).fillna(global_mean) / kf.n_splits

    feature_df[f"{col}_te_oof"] = np.concatenate([oof_te, test_te])

# Numerical Features
for col in num_cols:
    feature_df[col] = pd.to_numeric(combined[col], errors="coerce")
    feature_df[f"{col}_is_null"] = combined[col].isnull().astype(float)

# Separate Train and Test
X_train = feature_df.iloc[:n_train].copy()
X_test = feature_df.iloc[n_train:].copy()

cat_features = [col for col in X_train.columns if col.endswith("_cat")]

# Model Training with CatBoost
oof_preds = np.zeros(n_train)
test_preds = np.zeros(len(X_test))

try:
    from catboost import CatBoostRegressor, Pool

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train)):
        X_tr, y_tr = X_train.iloc[tr_idx], y[tr_idx]
        X_va, y_va = X_train.iloc[val_idx], y[val_idx]

        train_pool = Pool(X_tr, y_tr, cat_features=cat_features)
        val_pool = Pool(X_va, y_va, cat_features=cat_features)
        test_pool = Pool(X_test, cat_features=cat_features)

        model = CatBoostRegressor(
            iterations=1500,
            learning_rate=0.03,
            depth=6,
            loss_function="RMSE",
            eval_metric="RMSE",
            random_seed=42 + fold,
            verbose=0,
        )
        model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=100)

        oof_preds[val_idx] = model.predict(val_pool)
        test_preds += model.predict(test_pool) / kf.n_splits

except ImportError:
    import lightgbm as lgb

    for c in cat_features:
        X_train[c] = X_train[c].astype("category")
        X_test[c] = X_test[c].astype("category")

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train)):
        X_tr, y_tr = X_train.iloc[tr_idx], y[tr_idx]
        X_va, y_va = X_train.iloc[val_idx], y[val_idx]

        model = lgb.LGBMRegressor(
            n_estimators=1500,
            learning_rate=0.03,
            max_depth=6,
            num_leaves=31,
            random_state=42 + fold,
            n_jobs=-1,
        )
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )

        oof_preds[val_idx] = model.predict(X_va)
        test_preds += model.predict(X_test) / kf.n_splits

# Post-processing
if (y >= 0).all():
    oof_preds = np.clip(oof_preds, a_min=0, a_max=None)
    test_preds = np.clip(test_preds, a_min=0, a_max=None)

cv_rmse = np.sqrt(mean_squared_error(y, oof_preds))
print(f"Validation RMSE: {cv_rmse:.4f}")

# Save test predictions
os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame({"id": np.arange(len(test_preds)), "Recovery": test_preds})
submission.to_csv("./working/submission.csv", index=False)
