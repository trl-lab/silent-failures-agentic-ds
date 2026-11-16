import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# Load data
train_path = '/path/to/user/workspace/ds_failures/datasets/strable/corporate-procurement-contract-awards/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/strable/corporate-procurement-contract-awards/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'Contract Award Amount (USD)'
y_train = train_df[target_col].values

# Combine for uniform preprocessing
n_train = len(train_df)
combined_df = pd.concat([train_df.drop(columns=[target_col]), test_df], ignore_index=True)

# Feature Engineering
# 1. Date features
if 'Award Date' in combined_df.columns:
    dates = pd.to_datetime(combined_df['Award Date'], errors='coerce')
    combined_df['award_year'] = dates.dt.year.fillna(-1).astype(int)
    combined_df['award_month'] = dates.dt.month.fillna(-1).astype(int)
    combined_df['award_day'] = dates.dt.day.fillna(-1).astype(int)
    combined_df['award_dayofweek'] = dates.dt.dayofweek.fillna(-1).astype(int)
    combined_df['award_timestamp'] = dates.astype('int64') // 10**9
    combined_df.loc[dates.isna(), 'award_timestamp'] = -1

# 2. Text features from Contract Description
text_col = 'Contract Description'
if text_col in combined_df.columns:
    combined_df[text_col] = combined_df[text_col].fillna('').astype(str)
    tfidf = TfidfVectorizer(max_features=10000, stop_words='english', ngram_range=(1, 2))
    tfidf_matrix = tfidf.fit_transform(combined_df[text_col])
    svd = TruncatedSVD(n_components=30, random_state=42)
    svd_feats = svd.fit_transform(tfidf_matrix)
    for i in range(30):
        combined_df[f'desc_svd_{i}'] = svd_feats[:, i]
    combined_df['desc_char_len'] = combined_df[text_col].apply(len)
    combined_df['desc_word_count'] = combined_df[text_col].apply(lambda x: len(x.split()))

# 3. Categorical encoding (Frequency encoding & Categorical dtype for GBDTs)
cat_cols = [
    'Commodity Category', 'WBG Organization', 'Selection Number',
    'Supplier', 'Supplier Country / Economy', 'Supplier Country / Economy Code',
    'Fund Source', 'VPU description', 'Quarter and Fiscal Year'
]

for col in cat_cols:
    if col in combined_df.columns:
        combined_df[col] = combined_df[col].fillna('Missing').astype(str)
        # Frequency encoding
        freq = combined_df[col].value_counts()
        combined_df[f'{col}_freq'] = combined_df[col].map(freq)
        combined_df[col] = combined_df[col].astype('category')

# Drop raw date string and description if needed, or keep for CatBoost
if 'Award Date' in combined_df.columns:
    combined_df.drop(columns=['Award Date'], inplace=True)
if 'Contract Description' in combined_df.columns:
    combined_df.drop(columns=['Contract Description'], inplace=True)

X_train = combined_df.iloc[:n_train].copy()
X_test = combined_df.iloc[n_train:].copy()

# K-Fold CV Ensembling
kf = KFold(n_splits=5, shuffle=True, random_state=42)

lgb_preds = np.zeros(len(X_test))
cb_preds = np.zeros(len(X_test))

# Prepare categorical column indices for CatBoost
cat_features_cb = [col for col in cat_cols if col in X_train.columns]
for col in cat_features_cb:
    X_train[col] = X_train[col].astype(str)
    X_test[col] = X_test[col].astype(str)

for train_idx, val_idx in kf.split(X_train):
    X_tr, y_tr = X_train.iloc[train_idx], y_train[train_idx]
    X_va, y_va = X_train.iloc[val_idx], y_train[val_idx]
    
    # LightGBM (convert categorical back to category dtype for LGBM)
    X_tr_lgb = X_tr.copy()
    X_va_lgb = X_va.copy()
    X_te_lgb = X_test.copy()
    for col in cat_features_cb:
        X_tr_lgb[col] = X_tr_lgb[col].astype('category')
        X_va_lgb[col] = X_va_lgb[col].astype('category')
        X_te_lgb[col] = X_te_lgb[col].astype('category')
        
    lgb_model = LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=63,
        random_state=42,
        verbosity=-1
    )
    lgb_model.fit(
        X_tr_lgb, y_tr,
        eval_set=[(X_va_lgb, y_va)],
        callbacks=[]
    )
    lgb_preds += lgb_model.predict(X_te_lgb) / kf.n_splits
    
    # CatBoost
    cb_model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        cat_features=cat_features_cb,
        verbose=0,
        random_seed=42
    )
    cb_model.fit(X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=50, verbose=False)
    cb_preds += cb_model.predict(X_test) / kf.n_splits

# Ensemble average
final_predictions = 0.5 * lgb_preds + 0.5 * cb_preds

# Save predictions
output_df = pd.DataFrame({
    'id': np.arange(len(final_predictions)),
    'Contract Award Amount (USD)': final_predictions
})
output_df.to_csv('predictions.csv', index=False)