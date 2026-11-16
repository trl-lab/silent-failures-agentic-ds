import pandas as pd
import numpy as np
import re
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Load data
train_path = '/path/to/user/workspace/ds_failures/datasets/strable/orphan-designations/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/strable/orphan-designations/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

# Target encoding
target_col = 'Status'
le = LabelEncoder()
y_train = le.fit_transform(train_df[target_col].astype(str))

# Combine datasets for consistent feature engineering
train_len = len(train_df)
df = pd.concat([train_df.drop(columns=[target_col]), test_df], ignore_index=True)

# Feature Engineering
features_df = pd.DataFrame(index=df.index)

# 1. Missing value indicators
for col in df.columns:
    features_df[f'{col}_isna'] = df[col].isna().astype(int)

# 2. Text length and character counts
for col in df.columns:
    s = df[col].fillna('').astype(str)
    features_df[f'{col}_len'] = s.str.len()
    features_df[f'{col}_word_count'] = s.apply(lambda x: len(x.split()))

# 3. Date features
date_cols = ['Date of designation / refusal', 'First published date', 'Last updated date']
parsed_dates = {}
for col in date_cols:
    dt = pd.to_datetime(df[col], format='%d/%m/%Y', errors='coerce')
    parsed_dates[col] = dt
    features_df[f'{col}_year'] = dt.dt.year.fillna(-1)
    features_df[f'{col}_month'] = dt.dt.month.fillna(-1)
    features_df[f'{col}_day'] = dt.dt.day.fillna(-1)
    features_df[f'{col}_dayofweek'] = dt.dt.dayofweek.fillna(-1)
    features_df[f'{col}_ts'] = dt.astype('int64').replace(-9223372036854775808, np.nan).fillna(-1)

# Date differences (in days)
features_df['diff_pub_desig'] = (parsed_dates['First published date'] - parsed_dates['Date of designation / refusal']).dt.days.fillna(-999)
features_df['diff_last_pub'] = (parsed_dates['Last updated date'] - parsed_dates['First published date']).dt.days.fillna(-999)
features_df['diff_last_desig'] = (parsed_dates['Last updated date'] - parsed_dates['Date of designation / refusal']).dt.days.fillna(-999)

# 4. Parse EU designation number: format EU/3/YY/NNNN
eu_pattern = r'EU/(\d+)/(\d+)/(\d+)'
eu_extracted = df['EU designation number'].fillna('').str.extract(eu_pattern)
features_df['eu_part1'] = pd.to_numeric(eu_extracted[0], errors='coerce').fillna(-1)
features_df['eu_year'] = pd.to_numeric(eu_extracted[1], errors='coerce').fillna(-1)
features_df['eu_seq'] = pd.to_numeric(eu_extracted[2], errors='coerce').fillna(-1)

# 5. EMA product number features
ema_extracted = df['Related EMA product number'].fillna('').str.extract(r'EMEA/([A-Z]+)/([A-Z]+)/(\d+)')
features_df['ema_num'] = pd.to_numeric(ema_extracted[2], errors='coerce').fillna(-1)

# 6. Text TF-IDF + SVD
text_corpus = (
    df['Medicine name'].fillna('') + ' ' +
    df['Active substance'].fillna('') + ' ' +
    df['Intended use'].fillna('')
).tolist()

tfidf = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))
tfidf_matrix = tfidf.fit_transform(text_corpus)

n_components = min(30, tfidf_matrix.shape[1] - 1)
svd = TruncatedSVD(n_components=n_components, random_state=42)
svd_features = svd.fit_transform(tfidf_matrix)

for i in range(n_components):
    features_df[f'svd_{i}'] = svd_features[:, i]

# Split back into train and test
X_train = features_df.iloc[:train_len].copy()
X_test = features_df.iloc[train_len:].copy()

# Cross-validated training and ensembling
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

lgb_preds = np.zeros(len(X_test))
cat_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    X_tr, y_tr = X_train.iloc[train_idx], y_train[train_idx]
    X_va, y_va = X_train.iloc[val_idx], y_train[val_idx]

    # LightGBM
    lgb_model = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        random_state=42 + fold,
        verbose=-1
    )
    lgb_model.fit(X_tr, y_tr)
    lgb_preds += lgb_model.predict_proba(X_test)[:, 1] / skf.n_splits

    # CatBoost
    cat_model = CatBoostClassifier(
        iterations=300,
        learning_rate=0.05,
        random_state=42 + fold,
        verbose=0
    )
    cat_model.fit(X_tr, y_tr)
    cat_preds += cat_model.predict_proba(X_test)[:, 1] / skf.n_splits

# Ensemble average
final_preds = 0.5 * lgb_preds + 0.5 * cat_preds

# Create submission
submission_df = pd.DataFrame({
    'id': np.arange(len(test_df)),
    'Status': final_preds
})

submission_df.to_csv('predictions.csv', index=False)