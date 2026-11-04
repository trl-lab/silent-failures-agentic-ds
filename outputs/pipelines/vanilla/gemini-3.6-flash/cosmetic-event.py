import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, KFold
import lightgbm as lgb
from catboost import CatBoostClassifier
import xgboost as xgb

train_path = '/path/to/user/workspace/ds_failures/datasets/strable/cosmetic-event/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/strable/cosmetic-event/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'outcomes'
y_raw = train_df[target_col].astype(str)
le = LabelEncoder()
y = le.fit_transform(y_raw)

n_train = len(train_df)
n_test = len(test_df)

X_train_raw = train_df.drop(columns=[target_col], errors='ignore')
X_test_raw = test_df.drop(columns=[target_col], errors='ignore')

all_df = pd.concat([X_train_raw, X_test_raw], axis=0, ignore_index=True)

features_list = []

for col in all_df.columns:
    col_data = all_df[col]
    
    if col_data.dtype == 'object' or 'date' in col.lower() or 'time' in col.lower():
        try:
            dt = pd.to_datetime(col_data, errors='coerce')
            if dt.notna().sum() > 0.2 * len(dt):
                f_yr = dt.dt.year.fillna(-1).values.reshape(-1, 1)
                f_mo = dt.dt.month.fillna(-1).values.reshape(-1, 1)
                f_dy = dt.dt.day.fillna(-1).values.reshape(-1, 1)
                f_dw = dt.dt.dayofweek.fillna(-1).values.reshape(-1, 1)
                f_na = dt.isna().astype(int).values.reshape(-1, 1)
                features_list.extend([f_yr, f_mo, f_dy, f_dw, f_na])
                continue
        except Exception:
            pass

    if pd.api.types.is_numeric_dtype(col_data):
        vals = col_data.values.astype(np.float32)
        has_na = pd.isna(vals)
        med = np.nanmedian(vals) if not np.all(has_na) else 0.0
        vals[has_na] = med
        features_list.append(vals.reshape(-1, 1))
        if np.any(has_na):
            features_list.append(has_na.astype(np.float32).reshape(-1, 1))
    else:
        s = col_data.fillna("missing").astype(str)
        
        freq = s.map(s.value_counts(normalize=True)).fillna(0).values.astype(np.float32).reshape(-1, 1)
        features_list.append(freq)
        
        codes, _ = pd.factorize(s)
        features_list.append(codes.astype(np.float32).reshape(-1, 1))
        
        avg_len = s.str.len().mean()
        n_uniq = s.nunique()
        if avg_len > 10 or n_uniq > 20:
            str_len = s.str.len().values.astype(np.float32).reshape(-1, 1)
            word_cnt = s.str.split().str.len().values.astype(np.float32).reshape(-1, 1)
            features_list.extend([str_len, word_cnt])
            
            try:
                tfidf = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2), sublinear_tf=True)
                tfidf_mat = tfidf.fit_transform(s)
                if tfidf_mat.shape[1] > 2:
                    n_comp = min(30, tfidf_mat.shape[1] - 1)
                    svd = TruncatedSVD(n_components=n_comp, random_state=42)
                    svd_mat = svd.fit_transform(tfidf_mat).astype(np.float32)
                    features_list.append(svd_mat)
            except Exception:
                pass

X_all = np.hstack(features_list)
X_all = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)

X_train = X_all[:n_train]
X_test = X_all[n_train:]

num_classes = len(le.classes_)
test_preds = np.zeros((n_test, num_classes))

try:
    min_class_count = np.min(np.bincount(y))
    if min_class_count >= 5:
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        splits = list(kf.split(X_train, y))
    else:
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        splits = list(kf.split(X_train))
except Exception:
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    splits = list(kf.split(X_train))

for train_idx, val_idx in splits:
    X_tr, y_tr = X_train[train_idx], y[train_idx]
    
    models = []
    
    try:
        clf_lgb = lgb.LGBMClassifier(
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        clf_lgb.fit(X_