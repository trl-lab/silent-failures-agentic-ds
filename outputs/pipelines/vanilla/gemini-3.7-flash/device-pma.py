import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import warnings

warnings.filterwarnings('ignore')

# 1. Load data
train_path = '/path/to/user/workspace/ds_failures/datasets/strable/device-pma/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/strable/device-pma/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'decision_code'
le = LabelEncoder()
y_train = le.fit_transform(train_df[target_col].astype(str))

n_train = len(train_df)
df_all = pd.concat([train_df.drop(columns=[target_col]), test_df], ignore_index=True)

# 2. Feature Engineering
features_df = pd.DataFrame(index=df_all.index)

# Date features
date_cols = ['date_received', 'decision_date', 'fed_reg_notice_date']
for col in date_cols:
    dt = pd.to_datetime(df_all[col], errors='coerce')
    features_df[f'{col}_year'] = dt.dt.year
    features_df[f'{col}_month'] = dt.dt.month
    features_df[f'{col}_day'] = dt.dt.day
    features_df[f'{col}_dayofweek'] = dt.dt.dayofweek
    features_df[f'{col}_isna'] = dt.isna().astype(int)

# Date differences in days
dt_received = pd.to_datetime(df_all['date_received'], errors='coerce')
dt_decision = pd.to_datetime(df_all['decision_date'], errors='coerce')
dt_fed_reg = pd.to_datetime(df_all['fed_reg_notice_date'], errors='coerce')

features_df['days_to_decision'] = (dt_decision - dt_received).dt.days
features_df['days_to_fed_reg'] = (dt_fed_reg - dt_decision).dt.days
features_df['days_received_to_fed_reg'] = (dt_fed_reg - dt_received).dt.days

# PMA number features
pma_str = df_all['pma_number'].astype(str)
df_all['pma_prefix'] = pma_str.str[0]
features_df['pma_year_code'] = pd.to_numeric(pma_str.str[1:3], errors='coerce')
features_df['pma_len'] = pma_str.str.len()

# Missing value flags & string lengths
text_cols = ['applicant', 'street_1', 'street_2', 'city', 'state', 'zip', 'zip_ext',
             'generic_name', 'trade_name', 'product_code', 'advisory_committee',
             'advisory_committee_description', 'expedited_review_flag', 'docket_number', 'openfda']

for col in text_cols:
    features_df[f'{col}_isna'] = df_all[col].isna().astype(int)
    features_df[f'{col}_len'] = df_all[col].fillna('').astype(str).str.len()

# Frequency encoding & Label encoding for categorical columns
cat_cols = ['pma_prefix', 'applicant', 'city', 'state', 'zip', 'product_code',
            'advisory_committee', 'advisory_committee_description', 'expedited_review_flag']

for col in cat_cols:
    val_counts = df_all[col].fillna('__MISSING__').value_counts()
    features_df[f'{col}_freq'] = df_all[col].fillna('__MISSING__').map(val_counts)
    
    col_le = LabelEncoder()
    features_df[f'{col}_encoded'] = col_le.fit_transform(df_all[col].fillna('__MISSING__').astype(str))

# TF-IDF + SVD for rich text fields
rich_text = (
    df_all['generic_name'].fillna('') + ' ' +
    df_all['trade_name'].fillna('') + ' ' +
    df_all['openfda'].fillna('') + ' ' +
    df_all['applicant'].fillna('')
).astype(str)

tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), stop_words='english')
tfidf_mat = tfidf.fit_transform(rich_text)

svd = TruncatedSVD(n_components=25, random_state=42)
svd_feats = svd.fit_transform(tfidf_mat)
for i in range(25):
    features_df[f'svd_{i}'] = svd_feats[:, i]

# Split features back into train and test
X_train = features_df.iloc[:n_train].copy()
X_test = features_df.iloc[n_train:].copy()

# 3. Model Training with Stratified K-Fold CV
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

lgb_preds = np.zeros(len(X_test))
cb_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    X_tr, y_tr = X_train.iloc[train_idx], y_train[train_idx]
    X_va, y_va = X_train.iloc[val_idx], y_train[val_idx]
    
    # LightGBM
    lgb_model = LGBMClassifier(
        n_estimators=500,
        learning_rate=0.03,
        num_leaves=31,
        random_state=42 + fold,
        verbose=-1,
        n_jobs=-1
    )
    lgb_model.fit(X_tr, y_tr)
    lgb_preds += lgb_model.predict_proba(X_test)[:, 1] / n_splits
    
    # CatBoost
    cb_model = CatBoostClassifier(
        iterations=500,
        learning_rate=0.04,
        depth=6,
        random_state=42 + fold,
        verbose=0,
        thread_count=-1
    )
    cb_model.fit(X_tr, y_tr)
    cb_preds += cb_model.predict_proba(X_test)[:, 1] / n_splits

final_preds = 0.5 * lgb_preds + 0.5 * cb_preds

# 4. Save predictions
submission_df = pd.DataFrame({
    'id': np.arange(len(test_df)),
    'decision_code': final_preds
})
submission_df.to_csv('predictions.csv', index=False)