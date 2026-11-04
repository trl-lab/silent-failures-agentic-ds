import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import TruncatedSVD
import lightgbm as lgb
from catboost import CatBoostClassifier

train_path = '/path/to/user/workspace/ds_failures/datasets/strable/osha-accidents/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/strable/osha-accidents/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'Task Assigned'
y_raw = train_df[target_col]
le_target = LabelEncoder()
y = le_target.fit_transform(y_raw)

text_cols = ['Abstract Text', 'Event Description', 'Event Keywords']
for col in text_cols:
    if col in train_df.columns:
        train_df[col] = train_df[col].fillna('').astype(str)
        test_df[col] = test_df[col].fillna('').astype(str)
    else:
        train_df[col] = ''
        test_df[col] = ''

train_text = train_df['Abstract Text'] + ' ' + train_df['Event Description'] + ' ' + train_df['Event Keywords']
test_text = test_df['Abstract Text'] + ' ' + test_df['Event Description'] + ' ' + test_df['Event Keywords']

tfidf = TfidfVectorizer(max_features=25000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
X_train_tfidf = tfidf.fit_transform(train_text)
X_test_tfidf = tfidf.transform(test_text)

svd = TruncatedSVD(n_components=30, random_state=42)
X_train_svd = svd.fit_transform(X_train_tfidf)
X_test_svd = svd.transform(X_test_tfidf)

svd_cols = [f'svd_{i}' for i in range(30)]
svd_train_df = pd.DataFrame(X_train_svd, columns=svd_cols, index=train_df.index)
svd_test_df = pd.DataFrame(X_test_svd, columns=svd_cols, index=test_df.index)

def extract_date_features(df):
    date_col = 'Event Date'
    feats = pd.DataFrame(index=df.index)
    if date_col in df.columns:
        dt = pd.to_datetime(df[date_col], errors='coerce')
        feats['event_year'] = dt.dt.year.fillna(-1)
        feats['event_month'] = dt.dt.month.fillna(-1)
        feats['event_day'] = dt.dt.day.fillna(-1)
        feats['event_dayofweek'] = dt.dt.dayofweek.fillna(-1)
    return feats

train_date = extract_date_features(train_df)
test_date = extract_date_features(test_df)

ignore_cols = [target_col, 'summary_nr', 'Event Date'] + text_cols
feature_cols = [c for c in train_df.columns if c not in ignore_cols]

train_tab = train_df[feature_cols].copy()
test_tab = test_df[feature_cols].copy()

for col in feature_cols:
    train_tab[col] = train_tab[col].astype(str).fillna('missing')
    test_tab[col] = test_tab[col].astype(str).fillna('missing')
    le = LabelEncoder()
    train_tab[col] = le.fit_transform(train_tab[col])
    test_tab[col] = test_tab[col].map(lambda s: le.transform([s])[0] if s in le.classes_ else -1)

train_tab['text_len'] = train_text.apply(len)
test_tab['text_len'] = test_text.apply(len)
train_tab['word_count'] = train_text.apply(lambda x: len(x.split()))
test_tab['word_count'] = test_text.apply(lambda x: len(x.split()))

X_train_all = pd.concat([train_tab, train_date, svd_train_df], axis=1)
X_test_all = pd.concat([test_tab, test_date, svd_test_df], axis=1)

n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_lr = np.zeros(len(train_df))
test_preds_lr = np.zeros(len(test_df))

oof_lgb = np.zeros(len(train_df))
test_preds_lgb = np.zeros(len(test_df))

oof_cb = np.zeros(len(train_df))
test_preds_cb = np.zeros(len(test_df))

for fold, (trn_idx, val_idx) in enumerate(skf.split(train_df, y)):
    # 1. Logistic Regression on TF-IDF
    lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    lr.fit(X_train_tfidf[trn_idx], y[trn_idx])
    oof_lr[val_idx] = lr.predict_proba(X_train_tfidf[val_idx])[:, 1]
    test_preds_lr += lr.predict_proba(X_test_tfidf)[:, 1] / n_splits

    # 2. LightGBM on tabular + SVD + metadata
    X_tr_fold = X_train_all.iloc[trn_idx]
    y_tr_fold = y[trn_idx]
    X_va_fold = X_train_all.iloc[val_idx]
    y_va_fold = y[val_idx]

    lgbm = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42 + fold,
        verbose=-1
    )
    lgbm.fit(X_tr_fold, y_tr_fold)
    oof_lgb[val_idx] = lgbm.predict_proba(X_va_fold)[:, 1]
    test_preds_lgb += lgbm.predict_proba(X_test_all)[:, 1] / n_splits

    # 3. CatBoost
    cb = CatBoostClassifier(
        iterations=350,
        learning_rate=0.05,
        depth=6,
        random_seed=42 + fold,
        verbose=0
    )
    cb.fit(X_tr_fold, y_tr_fold)
    oof_cb[val_idx] = cb.predict_proba(X_va_fold)[:, 1]
    test_preds_cb += cb.predict_proba(X_test_all)[:, 1] / n_splits

final_test_preds = 0.45 * test_preds_lr + 0.30 * test_preds_lgb + 0.25 * test_preds_cb

submission_df = pd.DataFrame({
    'id': np.arange(len(test_df)),
    'Task Assigned': final_test_preds
})

submission_df.to_csv('predictions.csv', index=False)