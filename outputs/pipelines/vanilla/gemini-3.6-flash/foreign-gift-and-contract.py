import os
import re
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

train_path = "/path/to/user/workspace/ds_failures/datasets/strable/foreign-gift-and-contract/data/train.parquet"
test_path = "/path/to/user/workspace/ds_failures/datasets/strable/foreign-gift-and-contract/data/test.parquet"

train = pd.read_parquet(train_path)
test = pd.read_parquet(test_path)

target_col = "Amount"
y_train = train[target_col].values

X_train_df = train.drop(columns=[target_col]).copy()
X_test_df = test.copy()

use_log = (y_train >= 0).all()
if use_log:
    y_train_trans = np.log1p(y_train)
else:
    y_train_trans = y_train

n_train = len(train)
df_all = pd.concat([X_train_df, X_test_df], axis=0, ignore_index=True)

numeric_cols = []
cat_cols = []
text_cols = []
date_cols = []

for col in df_all.columns:
    if col.lower() in ['id', 'index', 'unnamed: 0']:
        continue
    
    if pd.api.types.is_datetime64_any_dtype(df_all[col]):
        date_cols.append(col)
        continue
    
    if df_all[col].dtype == 'object' or isinstance(df_all[col].dtype, pd.CategoricalDtype):
        if 'date' in col.lower() or 'year' in col.lower():
            try:
                parsed_date = pd.to_datetime(df_all[col], errors='coerce')
                if parsed_date.notna().sum() > 0.5 * len(df_all):
                    df_all[col] = parsed_date
                    date_cols.append(col)
                    continue
            except Exception:
                pass
        
        col_str = df_all[col].astype(str)
        avg_len = col_str.str.len().mean()
        n_unique = df_all[col].nunique()
        
        if avg_len > 30 or (n_unique > 500 and avg_len > 15):
            text_cols.append(col)
        else:
            cat_cols.append(col)
    else:
        numeric_cols.append(col)

features = pd.DataFrame(index=df_all.index)

for col in date_cols:
    dt = pd.to_datetime(df_all[col], errors='coerce')
    features[f"{col}_year"] = dt.dt.year.fillna(-1).astype(np.float32)
    features[f"{col}_month"] = dt.dt.month.fillna(-1).astype(np.float32)
    features[f"{col}_day"] = dt.dt.day.fillna(-1).astype(np.float32)
    features[f"{col}_dayofweek"] = dt.dt.dayofweek.fillna(-1).astype(np.float32)
    features[f"{col}_quarter"] = dt.dt.quarter.fillna(-1).astype(np.float32)

for col in numeric_cols:
    num_series = pd.to_numeric(df_all[col], errors='coerce')
    med = num_series.median()
    if pd.isna(med):
        med = 0.0
    features[col] = num_series.fillna(med).astype(np.float32)

for col in cat_cols:
    col_str = df_all[col].astype(str).fillna('missing')
    encoded, _ = pd.factorize(col_str)
    features[f"{col}_cat"] = encoded.astype(np.int32)
    freq = col_str.map(col_str.value_counts(normalize=True)).astype(np.float32)
    features[f"{col}_freq"] = freq

for col in text_cols:
    col_str = df_all[col].fillna("").astype(str)
    try:
        vec = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))
        tfidf_mat = vec.fit_transform(col_str)
        n_comp = min(16, tfidf_mat.shape[1] - 1)
        if n_comp > 1:
            svd = TruncatedSVD(n_components=n_comp, random_state=42)
            svd_mat = svd.fit_transform(tfidf_mat)
            for i in range(n_comp):
                features[f"{col}_svd_{i}"] = svd_mat[:, i].astype(np.float32)
    except Exception:
        pass

X_train = features.iloc[:n_train].values
X_test = features.iloc[n_train:].values

kf = KFold(n_splits=5, shuffle=True, random_state=42)

test_preds_lgb = np.zeros(len(X_test))
test_preds_cat = np.zeros(len(X_test))
test_preds_xgb = np.zeros(len(X_test))

for fold, (trn_idx, val_idx) in enumerate(kf.split(X_train, y_train_trans)):
    X_tr, y_tr = X_train[trn_idx], y_train_trans[trn_idx]
    X_va, y_va = X_train[val_idx], y_train_trans[val_idx]
    
    model_lgb = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        n_jobs=-1
    )
    model_lgb.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    test_preds_lgb += model_lgb.predict(X_test) / kf.n_splits
    
    model_cat = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.04,
        depth=6,
        random_seed=42 + fold,
        verbose=0
    )
    model_cat.fit(
        X_tr, y_tr,
        eval_set=(X_va, y_va),
        early_stopping_rounds=50
    )
    test_preds_cat += model_cat.predict(X_test) / kf.n_splits

    model_xgb = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        n_jobs=-1,
        early_stopping_rounds=50
    )
    model_xgb.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        verbose=False
    )
    test_preds_xgb += model_xgb.predict(X_test) / kf.n_splits

test_preds_trans = (test_preds_lgb + test_preds_cat + test_preds_xgb) / 3.0

if use_log:
    test_preds = np.expm1(np.maximum(0, test_preds_trans))
else:
    test_preds = test_preds_trans

test_preds = np.maximum(0, test_preds)

sub = pd.DataFrame({
    'id': np.arange(len(test_preds)),
    'Amount': test_preds
})

sub.to_csv("predictions.csv", index=False)