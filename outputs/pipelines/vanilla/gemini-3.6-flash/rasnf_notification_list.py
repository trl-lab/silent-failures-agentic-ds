import os
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import LabelEncoder

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

warnings.filterwarnings('ignore')

train_path = '/path/to/user/workspace/ds_failures/datasets/strable/rasnf_notification_list/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/strable/rasnf_notification_list/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'num_countries_affected'

y_raw = train_df[target_col]
if y_raw.dtype == object or y_raw.dtype.name == 'category' or len(np.unique(y_raw)) == 2:
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
else:
    y = y_raw.values.astype(int)

X_train = train_df.drop(columns=[target_col])
X_test = test_df.drop(columns=[target_col], errors='ignore')

X_train = X_train.loc[:, ~X_train.columns.duplicated()]
X_test = X_test.loc[:, ~X_test.columns.duplicated()]

common_cols = [c for c in X_train.columns if c in X_test.columns]
X_train = X_train[common_cols].copy()
X_test = X_test[common_cols].copy()

for col in common_cols:
    if pd.api.types.is_datetime64_any_dtype(X_train[col]) or 'date' in col.lower() or 'time' in col.lower():
        try:
            dt_tr = pd.to_datetime(X_train[col], errors='coerce')
            dt_te = pd.to_datetime(X_test[col], errors='coerce')
            if dt_tr.notna().sum() > 0.3 * len(dt_tr):
                X_train[f'{col}_year'] = dt_tr.dt.year
                X_train[f'{col}_month'] = dt_tr.dt.month
                X_train[f'{col}_day'] = dt_tr.dt.day
                X_train[f'{col}_dayofweek'] = dt_tr.dt.dayofweek
                X_train[f'{col}_dayofyear'] = dt_tr.dt.dayofyear

                X_test[f'{col}_year'] = dt_te.dt.year
                X_test[f'{col}_month'] = dt_te.dt.month
                X_test[f'{col}_day'] = dt_te.dt.day
                X_test[f'{col}_dayofweek'] = dt_te.dt.dayofweek
                X_test[f'{col}_dayofyear'] = dt_te.dt.dayofyear

                X_train.drop(columns=[col], inplace=True)
                X_test.drop(columns=[col], inplace=True)
        except Exception:
            pass

num_cols = []
cat_cols = []
text_cols = []

for col in X_train.columns:
    if pd.api.types.is_numeric_dtype(X_train[col]):
        num_cols.append(col)
    else:
        s_tr = X_train[col].astype(str).fillna('')
        avg_len = s_tr.str.len().mean()
        nunique = s_tr.nunique()
        if avg_len > 30 or nunique > 100:
            text_cols.append(col)
        else:
            cat_cols.append(col)

processed_train_list = []
processed_test_list = []

if num_cols:
    processed_train_list.append(X_train[num_cols].astype(np.float32).fillna(-999))
    processed_test_list.append(X_test[num_cols].astype(np.float32).fillna(-999))

for col in cat_cols:
    s_tr = X_train[col].astype(str).fillna('missing')
    s_te = X_test[col].astype(str).fillna('missing')

    categories = sorted(list(set(s_tr.unique()).union(set(s_te.unique()))))
    cat_map = {c: i for i, c in enumerate(categories)}
    tr_ord = s_tr.map(cat_map).astype(np.float32)
    te_ord = s_te.map(cat_map).astype(np.float32)

    freq = s_tr.value_counts(normalize=True).to_dict()
    tr_freq = s_tr.map(freq).fillna(0).astype(np.float32)
    te_freq = s_te.map(freq).fillna(0).astype(np.float32)

    df_tr_cat = pd.DataFrame({f'{col}_ord': tr_ord, f'{col}_freq': tr_freq}, index=X_train.index)
    df_te_cat = pd.DataFrame({f'{col}_ord': te_ord, f'{col}_freq': te_freq}, index=X_test.index)

    processed_train_list.append(df_tr_cat)
    processed_test_list.append(df_te_cat)

if text_cols:
    all_text_tr = X_train[text_cols].fillna('').astype(str).agg(' '.join, axis=1)
    all_text_te = X_test[text_cols].fillna('').astype(str).agg(' '.join, axis=1)

    tfidf = TfidfVectorizer(max_features=10000, stop_words='english', ngram_range=(1, 2))
    tr_tfidf = tfidf.fit_transform(all_text_tr)
    te_tfidf = tfidf.transform(all_text_te)

    n_comp = min(50, max(2, tr_tfidf.shape[1] - 1))
    svd = TruncatedSVD(n_components=n_comp, random_state=42)
    tr_svd = svd.fit_transform(tr_tfidf)
    te_svd = svd.transform(te_tfidf)

    svd_cols = [f'text_svd_{i}' for i in range(n_comp)]
    df_tr_svd = pd.DataFrame(tr_svd, columns=svd_cols, index=X_train.index).astype(np.float32)
    df_te_svd = pd.DataFrame(te_svd, columns=svd_cols, index=X_test.index).astype(np.float32)

    processed_train_list.append(df_tr_svd)
    processed_test_list.append(df_te_svd)

X_tr = pd.concat(processed_train_list, axis=1)
X_te = pd.concat(processed_test_list, axis=1)

col_names = [f'feature_{i}' for i in range(X_tr.shape[1])]
X_tr.columns = col_names
X_te.columns = col_names

n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

lgb_preds = np.zeros(len(X_te))
cat_preds = np.zeros(len(X_te))
xgb_preds = np.zeros(len(X_te))

lgb_params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'random_state': 42,
    'verbose': -1,
    'n_estimators': 1000
}

xgb_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'learning_rate': 0.05,
    'max_depth': 6,
    'random_state': 42,
    'n_estimators': 1000,
    'tree_method': 'hist'
}

for fold, (train_idx, val_idx) in enumerate(skf.split(X_tr, y)):
    X_train_fold, y_train_fold = X_tr.iloc[train_idx], y[train_idx]
    X_val_fold, y_val_fold = X_tr.iloc[val_idx], y[val_idx]

    model_lgb = lgb.LGBMClassifier(**lgb_params)
    model_lgb.fit(
        X_train_fold, y_train_fold,
        eval_set=[(X_val_fold, y_val_fold)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )
    lgb_preds += model_lgb.predict_proba(X_te)[:, 1] / n_splits

    model_cat = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        eval_metric='AUC',
        random_seed=42,
        verbose=False,
        early_stopping_rounds=50
    )
    model_cat.fit(X_train_fold, y_train_fold, eval_set=(X_val_fold, y_val_fold))
    cat_preds += model_cat.predict_proba(X_te)[:, 1] / n_splits

    model_xgb = xgb.XGBClassifier(**xgb_params, early_stopping_rounds=50)
    model_xgb.fit(
        X_train_fold, y_train_fold,
        eval_set=[(X_val_fold, y_val_fold)],
        verbose=False
    )
    xgb_preds += model_xgb.predict_proba(X_te)[:, 1] / n_splits

final_preds = lgb_preds * 0.4 + cat_preds * 0.3 + xgb_preds * 0.3

submission = pd.DataFrame({
    'id': np.arange(len(final_preds)),
    'num_countries_affected': final_preds
})

submission.to_csv('predictions.csv', index=False)