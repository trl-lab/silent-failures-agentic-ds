import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import lightgbm as lgb

train_path = '/path/to/user/workspace/ds_failures/datasets/strable/national-average-drug-acquisition-cost/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/strable/national-average-drug-acquisition-cost/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'NADAC Per Unit'
y_train = train_df[target_col].values
train_df = train_df.drop(columns=[target_col])

n_train = len(train_df)
df = pd.concat([train_df, test_df], ignore_index=True)

date_cols = ['Effective Date', 'Corresponding Generic Drug Effective Date', 'As of Date']
for col in date_cols:
    if col in df.columns:
        dt = pd.to_datetime(df[col], errors='coerce')
        df[f'{col}_year'] = dt.dt.year.fillna(-1).astype(int)
        df[f'{col}_month'] = dt.dt.month.fillna(-1).astype(int)
        df[f'{col}_day'] = dt.dt.day.fillna(-1).astype(int)
        df[f'{col}_dayofweek'] = dt.dt.dayofweek.fillna(-1).astype(int)
        df[f'{col}_timestamp'] = dt.astype('int64') // 10**9
        df[f'{col}_timestamp'] = df[f'{col}_timestamp'].where(dt.notnull(), np.nan)
        df = df.drop(columns=[col])

if 'Effective Date_timestamp' in df.columns and 'As of Date_timestamp' in df.columns:
    df['diff_effective_asof'] = df['As of Date_timestamp'] - df['Effective Date_timestamp']

if 'NDC' in df.columns:
    ndc_str = df['NDC'].astype(str)
    df['NDC_len'] = ndc_str.str.len()
    df['NDC_prefix_5'] = ndc_str.str[:5].astype('category').cat.codes
    df['NDC_prefix_4'] = ndc_str.str[:4].astype('category').cat.codes

if 'NDC Description' in df.columns:
    desc = df['NDC Description'].fillna('').astype(str)
    df['desc_char_len'] = desc.str.len()
    df['desc_word_count'] = desc.str.split().str.len()
    
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), token_pattern=r'(?u)\b\w+\b')
    tfidf_matrix = tfidf.fit_transform(desc)
    svd = TruncatedSVD(n_components=25, random_state=42)
    svd_feats = svd.fit_transform(tfidf_matrix)
    for i in range(25):
        df[f'desc_svd_{i}'] = svd_feats[:, i]
    df = df.drop(columns=['NDC Description'])

cat_cols = ['Pricing Unit', 'OTC', 'Explanation Code', 'Classification for Rate Setting']
for col in cat_cols:
    if col in df.columns:
        df[col] = df[col].astype('category')

X_train = df.iloc[:n_train].copy()
X_test = df.iloc[n_train:].copy()

kf = KFold(n_splits=5, shuffle=True, random_state=42)
test_preds = np.zeros(len(X_test))

lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 63,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'random_state': 42,
    'n_estimators': 1500,
    'n_jobs': -1,
    'verbose': -1
}

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
    X_tr, y_tr = X_train.iloc[train_idx], y_train[train_idx]
    X_va, y_va = X_train.iloc[val_idx], y_train[val_idx]
    
    model = lgb.LGBMRegressor(**lgb_params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )
    test_preds += model.predict(X_test) / kf.n_splits

submission_df = pd.DataFrame({
    'id': np.arange(len(test_preds)),
    'NADAC Per Unit': test_preds
})

submission_df.to_csv('predictions.csv', index=False)