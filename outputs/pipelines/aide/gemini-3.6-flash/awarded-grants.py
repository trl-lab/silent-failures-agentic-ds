import os
import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

# Load datasets
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

target_col = "Financial Assistance"
y_train = np.log1p(np.maximum(0, train[target_col].values))

df_train = train.drop(columns=[target_col]).copy()
df_test = test.copy()

n_train = len(df_train)
df_all = pd.concat([df_train, df_test], axis=0, ignore_index=True)

# Detect date columns
date_cols = [
    col
    for col in df_all.columns
    if "date" in col.lower() or "dt" in col.lower() or "year" in col.lower()
]
for col in df_all.columns:
    if col not in date_cols and df_all[col].dtype in ["object", "string"]:
        sample = df_all[col].dropna().head(100)
        if len(sample) > 0:
            try:
                pd.to_datetime(sample, errors="raise")
                date_cols.append(col)
            except (ValueError, TypeError):
                pass

date_cols = list(set(date_cols))
dt_series_dict = {}
valid_date_cols = []

for col in date_cols:
    dt_s = pd.to_datetime(df_all[col], errors="coerce")
    if dt_s.notna().sum() > 0:
        dt_series_dict[col] = dt_s
        valid_date_cols.append(col)
        df_all[f"{col}_year"] = dt_s.dt.year.fillna(-1)
        df_all[f"{col}_month"] = dt_s.dt.month.fillna(-1)
        df_all[f"{col}_day"] = dt_s.dt.day.fillna(-1)
        df_all[f"{col}_dayofweek"] = dt_s.dt.dayofweek.fillna(-1)
        df_all[f"{col}_quarter"] = dt_s.dt.quarter.fillna(-1)

# Compute pairwise date difference features
for i in range(len(valid_date_cols)):
    for j in range(i + 1, len(valid_date_cols)):
        col1, col2 = valid_date_cols[i], valid_date_cols[j]
        diff_days = (dt_series_dict[col1] - dt_series_dict[col2]).dt.days.fillna(-999)
        df_all[f"{col1}_{col2}_diff_days"] = diff_days

# Drop raw date columns
df_all.drop(columns=date_cols, inplace=True, errors="ignore")

# Categorize remaining columns
str_cols = df_all.select_dtypes(
    include=["object", "string", "category"]
).columns.tolist()
num_cols = df_all.select_dtypes(
    include=["int64", "int32", "float64", "float32"]
).columns.tolist()

text_cols = []
cat_cols = []
for col in str_cols:
    avg_len = df_all[col].astype(str).str.len().mean()
    if avg_len > 50:
        text_cols.append(col)
    else:
        cat_cols.append(col)

# Text TF-IDF + SVD feature extraction
if text_cols:
    combined_text = df_all[text_cols].fillna("").agg(" ".join, axis=1)
    tfidf = TfidfVectorizer(max_features=5000, stop_words="english")
    tfidf_mat = tfidf.fit_transform(combined_text)
    svd = TruncatedSVD(n_components=10, random_state=42)
    svd_feat = svd.fit_transform(tfidf_mat)
    for i in range(10):
        df_all[f"tfidf_svd_{i}"] = svd_feat[:, i]
    df_all.drop(columns=text_cols, inplace=True)

# Categorical Frequency Encoding & String Conversion
for col in cat_cols:
    freq = df_all[col].value_counts(normalize=True).to_dict()
    df_all[f"{col}_freq"] = df_all[col].map(freq).fillna(0)
    df_all[col] = df_all[col].astype(str).fillna("missing")

# Grouped Aggregation Features
top_cats = [col for col in cat_cols if df_all[col].nunique() < 200][:3]
numeric_feats = [col for col in num_cols if col not in ["id", "index"]]
if top_cats and numeric_feats:
    for cat in top_cats:
        for num in numeric_feats[:3]:
            grouped = df_all.groupby(cat)[num].agg(["mean", "std"]).reset_index()
            grouped.columns = [cat, f"{cat}_{num}_mean", f"{cat}_{num}_std"]
            df_all = df_all.merge(grouped, on=cat, how="left")

# Train/Test Split
X_train = df_all.iloc[:n_train].copy()
X_test = df_all.iloc[n_train:].copy()

num_cols_final = X_train.select_dtypes(include=[np.number]).columns.tolist()
for col in num_cols_final:
    med = X_train[col].median()
    X_train[col] = X_train[col].fillna(med)
    X_test[col] = X_test[col].fillna(med)

cat_cols_final = [col for col in cat_cols if col in X_train.columns]

# Prepare LightGBM formatted inputs
X_train_lgb = X_train.copy()
X_test_lgb = X_test.copy()
for col in cat_cols_final:
    X_train_lgb[col] = X_train_lgb[col].astype("category")
    X_test_lgb[col] = X_test_lgb[col].astype("category")

# 5-Fold Cross Validation Strategy with Blended Ensemble
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(n_train)
test_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_tr, y_tr = X_train.iloc[train_idx], y_train[train_idx]
    X_va, y_va = X_train.iloc[val_idx], y_train[val_idx]

    X_tr_lgb, X_va_lgb = X_train_lgb.iloc[train_idx], X_train_lgb.iloc[val_idx]

    # Model 1: CatBoost
    cb_model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        cat_features=cat_cols_final,
        random_seed=42 + fold,
        verbose=0,
        early_stopping_rounds=50,
    )
    cb_model.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=False)

    val_cb_preds = cb_model.predict(X_va)
    test_cb_preds = cb_model.predict(X_test)

    # Model 2: LightGBM
    lgb_model = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        random_state=42 + fold,
        n_jobs=-1,
        verbose=-1,
    )
    lgb_model.fit(
        X_tr_lgb,
        y_tr,
        eval_set=[(X_va_lgb, y_va)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )

    val_lgb_preds = lgb_model.predict(X_va_lgb)
    test_lgb_preds = lgb_model.predict(X_test_lgb)

    # Equal weight blending
    fold_val_preds = 0.5 * val_cb_preds + 0.5 * val_lgb_preds
    fold_test_preds = 0.5 * test_cb_preds + 0.5 * test_lgb_preds

    oof_preds[val_idx] = fold_val_preds
    test_preds += fold_test_preds / 5.0

# Compute Metric
oof_rmse = np.sqrt(mean_squared_error(y_train, oof_preds))
print(f"Validation RMSE: {oof_rmse:.4f}")

# Output predictions format
final_test_preds = np.expm1(np.maximum(0, test_preds))

os.makedirs("./working", exist_ok=True)
sub = pd.DataFrame()
if "id" in test.columns:
    sub["id"] = test["id"]
else:
    sub["id"] = np.arange(len(test))

sub[target_col] = final_test_preds
sub.to_csv("./working/submission.csv", index=False)
