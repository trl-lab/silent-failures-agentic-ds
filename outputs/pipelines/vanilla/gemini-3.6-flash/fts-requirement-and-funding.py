import os
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Ridge
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

warnings.filterwarnings('ignore')

train_path = '/path/to/user/workspace/ds_failures/datasets/strable/fts-requirement-and-funding/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/strable/fts-requirement-and-funding/data/test.parquet'

train = pd.read_parquet(train_path)
test = pd.read_parquet(test_path)

target_col = 'funding'

train = train.dropna(subset=[target_col]).reset_index(drop=True)
y = train[target_col].values.astype(np.float64)

X_train_raw = train.drop(columns=[target_col])
X_test_raw = test.drop(columns=[target_col], errors='ignore')

n_train = len(train)
df_all = pd.concat([X_train_raw, X_test_raw], axis=0).reset_index(drop=True)

id_cols = [c for c in df_all.columns if c.lower() in ['id', 'unnamed: 0']]
if id_cols:
    df_all = df_all.drop(columns=id_cols)

features_df = pd.DataFrame(index=df_all.index)

for col in df_all.columns:
    if pd.api.types.is_datetime64_any_dtype(df_all[col]):
        features_df[f'{col}_year'] = df_all[col].dt.year
        features_df[f'{col}_month'] = df_all[col].dt.month
        features_df[f'{col}_day'] = df_all[col].dt.day
        features_df[f'{col}_dayofweek'] = df_all[col].dt.dayofweek
        continue
    
    if df_all[col].dtype == 'object':
        sample = df_all[col].dropna().head(10).astype(str)
        if len(sample) > 0 and sample.str.contains(r'^\d{4}-\d{2}-\d{2}').all():
            dt_col = pd.to_datetime(df_all[col], errors='coerce')
            features_df[f'{col}_year'] = dt_col.dt.year
            features_df[f'{col}_month'] = dt_col.dt.month
            features_df[f'{col}_day'] = dt_col.dt.day
            features_df[f'{col}_dayofweek'] = dt_col.dt.dayofweek
            continue

    if pd.api.types.is_numeric_dtype(df_all[col]):
        val = df_all[col].replace([np.inf, -np.inf], np.nan)
        features_df[col] = val
        val_clean = val.dropna()
        if len(val_clean) > 0 and (val_clean >= 0).all():
            features_df[f'{col}_log1p'] = np.log1p(val.clip(lower=0))
    else:
        str_col = df_all[col].fillna('Missing').astype(str)
        cat_type = str_col.astype('category')
        features_df[f'{col}_cat'] = cat_type.cat.codes
        
        freq = str_col.map(str_col.value_counts())
        features_df[f'{col}_freq'] = freq
        
        avg_len = str_col.str.len().mean()
        if avg_len > 25 and str_col.nunique() > 50:
            try:
                tfidf = TfidfVectorizer(max_features=500, stop_words='english')
                tfidf_mat = tfidf.fit_transform(str_col)
                n_comp = min(10, tfidf_mat.shape[1] - 1)
                if n_comp > 1:
                    svd = TruncatedSVD(n_components=n_comp, random_state=42)
                    svd_mat = svd.fit_transform(tfidf_mat)
                    for i in range(n_comp):
                        features_df[f'{col}_tfidf_svd_{i}'] = svd_mat[:, i]
            except Exception:
                pass

num_cols = [c for c in features_df.columns if features_df[c].dtype in [np.float64, np.float32, np.int64, np.int32] and not c.endswith('_cat') and not c.endswith('_code') and not 'svd' in c]
cat_cols = [c for c in features_df.columns if c.endswith('_cat')]

for c_cat in cat_cols[:5]:
    for c_num in num_cols[:5]:
        if features_df[c_cat].nunique() > 1 and features_df[c_cat].nunique() < 1000:
            grp = features_df.groupby(c_cat)[c_num]
            features_df[f'{c_num}_mean_by_{c_cat}'] = grp.transform('mean')
            features_df[f'{c_num}_std_by_{c_cat}'] = grp.transform('std')

X_train = features_df.iloc[:n_train].copy()
X_test = features_df.iloc[n_train:].copy()

X_train = X_train.replace([np.inf, -np.inf], np.nan)
X_test = X_test.replace([np.inf, -np.inf], np.nan)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

lgb_oof = np.zeros(n_train)
lgb_preds = np.zeros(len(X_test))

cb_oof = np.zeros(n_train)
cb_preds = np.zeros(len(X_test))

xgb_oof = np.zeros(n_train)
xgb_preds = np.zeros(len(X_test))

for fold, (trn_idx, val_idx) in enumerate(kf.split(X_train, y)):
    X_tr, y_tr = X_train.iloc[trn_idx], y[trn_idx]
    X_va, y_va = X_train.iloc[val_idx], y[val_idx]
    
    model_lgb = lgb.LGBMRegressor(
        objective='regression',
        metric='rmse',
        n_estimators=1500,
        learning_rate=0.03,
        num_leaves=63,
        random_state=42 + fold,
        n_jobs=-1,
        verbose=-1
    )
    model_lgb.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )
    lgb_oof[val_idx] = model_lgb.predict(X_va)
    lgb_preds += model_lgb.predict(X_test) / 5.0
    
    model_cb = CatBoostRegressor(
        loss_function='RMSE',
        eval_metric='RMSE',
        iterations=1500,
        learning_rate=0.03,
        depth=6,
        random_seed=42 + fold,
        verbose=0,
        early_stopping_rounds=5