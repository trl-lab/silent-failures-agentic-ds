import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import lightgbm as lgb

# 1. Load data
train_path = '/path/to/user/workspace/ds_failures/datasets/strable/michelin-ratings/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/strable/michelin-ratings/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'awardFor'
y_raw = train_df[target_col].astype(str)

le = LabelEncoder()
y = le.fit_transform(y_raw)

n_train = len(train_df)
combined_df = pd.concat([train_df.drop(columns=[target_col]), test_df], ignore_index=True)

# 2. Feature Engineering
features_df = pd.DataFrame()

# Numerical features
features_df['latitude'] = combined_df['latitude'].astype(float)
features_df['longitude'] = combined_df['longitude'].astype(float)

# Date features
dates = pd.to_datetime(combined_df['datePublished'], errors='coerce')
features_df['pub_year'] = dates.dt.year.fillna(-1).astype(int)
features_df['pub_month'] = dates.dt.month.fillna(-1).astype(int)
features_df['pub_day'] = dates.dt.day.fillna(-1).astype(int)
features_df['pub_dayofweek'] = dates.dt.dayofweek.fillna(-1).astype(int)

# Categorical features
cat_cols = [
    'addressCountry', 'addressRegion', 'addressLocality', 
    'knowsLanguage', 'acceptsReservations', 'servesCuisine', 
    'currenciesAccepted', 'priceRange'
]

for col in cat_cols:
    s = combined_df[col].astype(str).fillna('missing')
    features_df[col] = pd.factorize(s)[0]
    
    # Frequency encoding
    freq = s.value_counts()
    features_df[f'{col}_freq'] = s.map(freq)

# Text length features
for col in ['description', 'name', 'streetAddress', 'paymentAccepted']:
    text_s = combined_df[col].astype(str).fillna('')
    features_df[f'{col}_len'] = text_s.str.len()
    features_df[f'{col}_word_count'] = text_s.str.split().apply(len)

# TF-IDF + TruncatedSVD on description
desc_text = combined_df['description'].fillna('').astype(str)
tfidf_desc = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), sublinear_tf=True)
desc_mat = tfidf_desc.fit_transform(desc_text)

svd_desc = TruncatedSVD(n_components=64, random_state=42)
desc_svd_feats = svd_desc.fit_transform(desc_mat)

for i in range(64):
    features_df[f'desc_svd_{i}'] = desc_svd_feats[:, i]

# TF-IDF + TruncatedSVD on servesCuisine and name
cuisine_name_text = (combined_df['servesCuisine'].fillna('').astype(str) + " " + 
                     combined_df['name'].fillna('').astype(str))
tfidf_cn = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
cn_mat = tfidf_cn.fit_transform(cuisine_name_text)

svd_cn = TruncatedSVD(n_components=32, random_state=42)
cn_svd_feats = svd_cn.fit_transform(cn_mat)

for i in range(32):
    features_df[f'cn_svd_{i}'] = cn_svd_feats[:, i]

# Split back to train and test
X_train = features_df.iloc[:n_train].copy()
X_test = features_df.iloc[n_train:].copy()

# 3. Model Training with Stratified K-Fold Cross-Validation
num_classes = len(le.classes_)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

test_preds_proba = np.zeros((len(test_df), num_classes))

lgb_params = {
    'objective': 'multiclass',
    'num_class': num_classes,
    'metric': 'multi_logloss',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'random_state': 42,
    'n_estimators': 400,
    'verbose': -1
}

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y)):
    X_tr, y_tr = X_train.iloc[train_idx], y[train_idx]
    X_va, y_va = X_train.iloc[val_idx], y[val_idx]
    
    clf = lgb.LGBMClassifier(**lgb_params)
    clf.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(stopping_rounds=40, verbose=False)]
    )
    
    test_preds_proba += clf.predict_proba(X_test) / skf.n_splits

# 4. Generate Final Predictions
final_preds_indices = np.argmax(test_preds_proba, axis=1)
final_preds_labels = le.inverse_transform(final_preds_indices)

# 5. Save Output
output_df = pd.DataFrame({
    'id': np.arange(len(test_df)),
    'awardFor': final_preds_labels
})
output_df.to_csv('predictions.csv', index=False)