import os
import gc
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
import lightgbm as lgb

def main():
    train_path = '/path/to/user/workspace/ds_failures/datasets/kaggle/goodreads-books-reviews-290312/data/train.parquet'
    test_path = '/path/to/user/workspace/ds_failures/datasets/kaggle/goodreads-books-reviews-290312/data/test.parquet'
    
    train = pd.read_parquet(train_path)
    test = pd.read_parquet(test_path)
    
    target_col = 'rating'
    y_train = train[target_col].values.astype(np.float32)
    global_mean = float(np.mean(y_train))
    
    X_train = pd.DataFrame(index=train.index)
    X_test = pd.DataFrame(index=test.index)
    
    cols = [c for c in train.columns if c != target_col]
    
    for col in cols:
        if 'date' in col.lower() or 'at' in col.lower() or pd.api.types.is_datetime64_any_dtype(train[col]):
            try:
                tr_dt = pd.to_datetime(train[col], errors='coerce')
                te_dt = pd.to_datetime(test[col], errors='coerce')
                if tr_dt.notna().sum() > 0:
                    X_train[f'{col}_year'] = tr_dt.dt.year.fillna(-1).astype(np.float32)
                    X_train[f'{col}_month'] = tr_dt.dt.month.fillna(-1).astype(np.float32)
                    X_train[f'{col}_day'] = tr_dt.dt.day.fillna(-1).astype(np.float32)
                    X_train[f'{col}_dayofweek'] = tr_dt.dt.dayofweek.fillna(-1).astype(np.float32)
                    
                    X_test[f'{col}_year'] = te_dt.dt.year.fillna(-1).astype(np.float32)
                    X_test[f'{col}_month'] = te_dt.dt.month.fillna(-1).astype(np.float32)
                    X_test[f'{col}_day'] = te_dt.dt.day.fillna(-1).astype(np.float32)
                    X_test[f'{col}_dayofweek'] = te_dt.dt.dayofweek.fillna(-1).astype(np.float32)
            except Exception:
                pass

    text_cols = []
    cat_cols = []
    num_cols = []
    
    for col in cols:
        if col in X_train.columns:
            continue
        dtype = train[col].dtype
        if pd.api.types.is_numeric_dtype(dtype) and not ('id' in col.lower()):
            num_cols.append(col)
        else:
            sample_s = train[col].dropna().astype(str)
            if len(sample_s) > 0 and sample_s.str.len().mean() > 20:
                text_cols.append(col)
            else:
                cat_cols.append(col)
                
    for col in num_cols:
        X_train[col] = train[col].astype(np.float32)
        X_test[col] = test[col].astype(np.float32)

    main_text_col = None
    if len(text_cols) > 0:
        lens = {c: train[c].dropna().astype(str).str.len().mean() for c in text_cols}
        main_text_col = max(lens, key=lens.get)
        
    if main_text_col is not None:
        tr_str = train[main_text_col].fillna("").astype(str)
        te_str = test[main_text_col].fillna("").astype(str)
        
        X_train['text_char_len'] = tr_str.str.len().astype(np.float32)
        X_test['text_char_len'] = te_str.str.len().astype(np.float32)
        
        X_train['text_word_cnt'] = tr_str.str.split().str.len().astype(np.float32)
        X_test['text_word_cnt'] = te_str.str.split().str.len().astype(np.float32)
        
        X_train['text_excl_cnt'] = tr_str.str.count('!').astype(np.float32)
        X_test['text_excl_cnt'] = te_str.str.count('!').astype(np.float32)
        
        X_train['text_quest_cnt'] = tr_str.str.count(r'\?').astype(np.float32)
        X_test['text_quest_cnt'] = te_str.str.count(r'\?').astype(np.float32)
        
        tfidf = TfidfVectorizer(max_features=40000, ngram_range=(1, 2), sublinear_tf=True, min_df=3)
        X_tr_tfidf = tfidf.fit_transform(tr_str)
        X_te_tfidf = tfidf.transform(te_str)
        
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        oof_ridge = np.zeros(len(train), dtype=np.float32)
        test_ridge = np.zeros(len(test), dtype=np.float32)
        
        for tr_idx, val_idx in kf.split(train):
            ridge = Ridge(alpha=3.0, solver='auto')
            ridge.fit(X_tr_tfidf[tr_idx], y_train[tr_idx])
            oof_ridge[val_idx] = ridge.predict(X_tr_tfidf[val_idx])
            test_ridge += ridge.predict(X_te_tfidf) / 5.0
            
        X_train['tfidf_ridge'] = oof_ridge
        X_test['tfidf_ridge'] = test_ridge
        
        del tfidf, X_tr_tfidf, X_te_tfidf
        gc.collect()

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    for col in cat_cols:
        full_s = pd.concat([train[col], test[col]], axis=0)
        freq_map = full_s.value_counts(normalize=True).to_dict()
        X_train[f'{col}_freq'] = train[col].map(freq_map).fillna(0).astype(np.float32)
        X_test[f'{col}_freq'] = test[col].map(freq_map).fillna(0).astype(np.float32)
        
        m = 10.0
        oof_te = np.full(len(train), global_mean, dtype=np.float32)
        
        for tr_idx, val_idx in kf.split(train):
            tr_sub = train.iloc[tr_idx]
            stats = tr_sub.groupby(col)[target_col].agg(['count', 'mean'])
            smoothed = (stats['count'] * stats['mean'] + m * global_mean) / (stats['count'] + m)
            val_cats = train.iloc[val_idx][col]
            oof_te