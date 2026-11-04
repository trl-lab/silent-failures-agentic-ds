import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# 1. Load data
train_path = '/path/to/user/workspace/ds_failures/datasets/strable/rasnf_notification_list/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/strable/rasnf_notification_list/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'num_countries_affected'
y_raw = train_df[target_col]

# Encode target
le_target = LabelEncoder()
y = le_target.fit_transform(y_raw)

n_train = len(train_df)
n_test = len(test_df)

# Drop target from train
X_train_raw = train_df.drop(columns=[target_col]).copy()
X_test_raw = test_df.copy()

combined_raw = pd.concat([X_train_raw, X_test_raw], axis=0).reset_index(drop=True)

# 2. Feature Engineering
features_df = pd.DataFrame(index=combined_raw.index)

# Presence / Missingness features
for col in combined_raw.columns:
    features_df[f'{col}_is_null'] = combined_raw[col].isna().astype(int)

# String processing
for col in combined_raw.columns:
    s = combined_raw[col].fillna('').astype(str)
    features_df[f'{col}_len'] = s.str.len()
    features_df[f'{col}_word_count'] = s.str.split().str.len()

# Categorical Frequency and Label Encoding
for col in combined_raw.columns:
    s = combined_raw[col].fillna('__MISSING__').astype(str)
    freq = s.value_counts(normalize=True)
    features_df[f'{col}_freq'] = s.map(freq).astype(float)
    
    le = LabelEncoder()
    features_df[f'{col}_encoded'] = le.fit_transform(s)

# Text TF-IDF + SVD features on combined text
text_cols = [
    'Risk type', 'Risk legal provision', 'Product', 'Name', 'Description', 
    'Brand', 'Category', 'Risk', 'Compulsory measures', 'Voluntary measures',
    'Alert submitted by', 'Country of origin'
]
available_text_cols = [c for c in text_cols if c in combined_raw.columns]

combined_text = combined_raw[available_text_cols].fillna('').astype(str).agg(' '.join, axis=1)

tfidf = TfidfVectorizer(max_features=25000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
tfidf_mat = tfidf.fit_transform(combined_text)

svd = TruncatedSVD(n_components=64, random_state=42)
svd_features = svd.fit_transform(tfidf_mat)

svd_cols = [f'svd_{i}' for i in range(svd_features.shape[1])]
svd_df = pd.DataFrame(svd_features, columns=svd_cols, index=combined_raw.index)

X_all = pd.concat([features_df, svd_df], axis=1)

X_train = X_all.iloc[:n_train].reset_index(drop=True)
X_test = X_all.iloc[n_train:].reset_index(drop=True)

# 3. Model Training & Cross-Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

lgb_test_preds = np.zeros(n_test)
cb_test_preds = np.zeros(n_test)

for train_idx, val_idx in skf.split(X_train, y):
    X_tr, y_tr = X_train.iloc[train_idx], y[train_idx]
    X_va, y_va = X_train.iloc[val_idx], y[val_idx]
    
    # LightGBM
    lgb_model = LGBMClassifier(
        n_estimators=600,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    lgb_model.fit(X_tr, y_tr)
    lgb_test_preds += lgb_model.predict_proba(X_test)[:, 1] / skf.n_splits
    
    # CatBoost
    cb_model = CatBoostClassifier(
        iterations=600,
        learning_rate=0.04,
        depth=6,
        random_seed=42,
        verbose=0
    )
    cb_model.fit(X_tr, y_tr)
    cb_test_preds += cb_model.predict_proba(X_test)[:, 1] / skf.n_splits

final_preds = 0.5 * lgb_test_preds + 0.5 * cb_test_preds

# 4. Save Predictions
submission = pd.DataFrame({
    'id': np.arange(n_test),
    'num_countries_affected': final_preds
})

submission.to_csv('predictions.csv', index=False)