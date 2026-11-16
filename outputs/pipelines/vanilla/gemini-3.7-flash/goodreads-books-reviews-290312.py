import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
import lightgbm as lgb

# 1. Load data
train_path = '/path/to/user/workspace/ds_failures/datasets/kaggle/goodreads-books-reviews-290312/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/kaggle/goodreads-books-reviews-290312/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

y_train = train_df['rating'].values.astype(np.float32)
global_mean = float(y_train.mean())

# 2. Text Features with Fast TF-IDF + Ridge
train_text = train_df['review_text'].fillna('').astype(str)
test_text = test_df['review_text'].fillna('').astype(str)

tfidf = TfidfVectorizer(max_features=15000, ngram_range=(1, 1), sublinear_tf=True, min_df=5)
X_train_tfidf = tfidf.fit_transform(train_text)
X_test_tfidf = tfidf.transform(test_text)

# 3-Fold CV for Ridge OOF
kf = KFold(n_splits=3, shuffle=True, random_state=42)
oof_ridge = np.zeros(len(train_df), dtype=np.float32)
test_preds_ridge = np.zeros(len(test_df), dtype=np.float32)

for train_idx, val_idx in kf.split(train_df):
    X_tr, y_tr = X_train_tfidf[train_idx], y_train[train_idx]
    X_va = X_train_tfidf[val_idx]
    
    ridge = Ridge(alpha=3.0, solver='sag', random_state=42)
    ridge.fit(X_tr, y_tr)
    
    oof_ridge[val_idx] = ridge.predict(X_va)
    test_preds_ridge += ridge.predict(X_test_tfidf).astype(np.float32) / 3.0

# 3. Fast Tabular Features
def extract_tabular_features(df, text_series):
    feats = pd.DataFrame(index=df.index)
    
    # Fast text metrics
    feats['char_len'] = text_series.str.len().astype(np.float32)
    feats['n_votes'] = df['n_votes'].fillna(0).astype(np.float32)
    feats['n_comments'] = df['n_comments'].fillna(0).astype(np.float32)
    
    # Fast year extraction from string (e.g. '... 2012')
    for col in ['date_added', 'date_updated', 'read_at', 'started_at']:
        if col in df.columns:
            s = df[col].astype(str)
            year = s.str[-4:]
            feats[f'{col}_year'] = pd.to_numeric(year, errors='coerce').fillna(-1).astype(np.float32)
            
    return feats

X_train_tab = extract_tabular_features(train_df, train_text)
X_test_tab = extract_tabular_features(test_df, test_text)

# Frequency encoding
for col in ['user_id', 'book_id']:
    counts = train_df[col].value_counts()
    X_train_tab[f'{col}_count'] = train_df[col].map(counts).fillna(1).astype(np.float32)
    X_test_tab[f'{col}_count'] = test_df[col].map(counts).fillna(1).astype(np.float32)

# Smoothed target encoding for user_id and book_id
smoothing = 10.0
for col in ['user_id', 'book_id']:
    X_train_tab[f'{col}_te'] = np.nan
    for train_idx, val_idx in kf.split(train_df):
        tr_df = train_df.iloc[train_idx]
        stats = tr_df.groupby(col)['rating'].agg(['count', 'mean'])
        smoothed = (stats['count'] * stats['mean'] + smoothing * global_mean) / (stats['count'] + smoothing)
        X_train_tab.loc[val_idx, f'{col}_te'] = train_df.iloc[val_idx][col].map(smoothed).fillna(global_mean).astype(np.float32)
        
    full_stats = train_df.groupby(col)['rating'].agg(['count', 'mean'])
    full_smoothed = (full_stats['count'] * full_stats['mean'] + smoothing * global_mean) / (full_stats['count'] + smoothing)
    X_test_tab[f'{col}_te'] = test_df[col].map(full_smoothed).fillna(global_mean).astype(np.float32)

# Include Ridge meta-feature
X_train_tab['ridge_pred'] = oof_ridge
X_test_tab['ridge_pred'] = test_preds_ridge

# 4. Fast LightGBM Training
trn_idx, val_idx = next(kf.split(X_train_tab))
X_tr, y_tr = X_train_tab.iloc[trn_idx], y_train[trn_idx]
X_va, y_va = X_train_tab.iloc[val_idx], y_train[val_idx]

trn_data = lgb.Dataset(X_tr, label=y_tr)
val_data = lgb.Dataset(X_va, label=y_va, reference=trn_data)

lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'learning_rate': 0.1,
    'num_leaves': 63,
    'n_jobs': -1,
    'random_state': 42,
    'verbose': -1
}

model = lgb.train(
    lgb_params,
    trn_data,
    num_boost_round=400,
    valid_sets=[trn_data, val_data],
    callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]
)

test_preds_lgb = model.predict(X_test_tab, num_iteration=model.best_iteration)

# 5. Ensemble and Output
final_preds = 0.8 * test_preds_lgb + 0.2 * test_preds_ridge
final_preds = np.clip(final_preds, 0, 5)

submission = pd.DataFrame({
    'id': np.arange(len(test_df)),
    'rating': final_preds
})
submission.to_csv('predictions.csv', index=False)