import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from catboost import CatBoostRegressor

# Load datasets
train_path = '/path/to/user/workspace/ds_failures/datasets/strable/grant/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/strable/grant/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'Federal Funds'
y_train = train_df[target_col].values
train_df = train_df.drop(columns=[target_col])

n_train = len(train_df)
n_test = len(test_df)
combined_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)

# Feature Engineering
# Text features on Description
combined_df['Description'] = combined_df['Description'].fillna('')
combined_df['desc_char_len'] = combined_df['Description'].apply(len)
combined_df['desc_word_len'] = combined_df['Description'].apply(lambda x: len(x.split()))
combined_df['desc_is_empty'] = (combined_df['desc_char_len'] == 0).astype(int)

tfidf = TfidfVectorizer(max_features=10000, stop_words='english', ngram_range=(1, 2))
tfidf_train = tfidf.fit_transform(combined_df['Description'].iloc[:n_train])
tfidf_all = tfidf.transform(combined_df['Description'])

svd = TruncatedSVD(n_components=30, random_state=42)
svd_feats = svd.fit_transform(tfidf_all)
for i in range(30):
    combined_df[f'svd_{i}'] = svd_feats[:, i]

# Log Number features
log_series = combined_df['Log Number'].fillna('UNKNOWN').astype(str)
combined_df['log_prefix'] = log_series.apply(lambda x: x.split('-')[0] if '-' in x else 'UNK')
combined_df['log_suffix'] = log_series.apply(lambda x: x.split('-')[-1] if '-' in x else 'UNK')

# Categorical features
cat_cols = ['Institution', 'Program', 'Funding Office', 'City', 'State', 'log_prefix', 'log_suffix']

for col in cat_cols:
    combined_df[col] = combined_df[col].fillna('MISSING').astype(str)
    # Frequency encoding
    freq = combined_df[col].value_counts()
    combined_df[f'{col}_freq'] = combined_df[col].map(freq)
    
    # Label encoding
    le = LabelEncoder()
    combined_df[col] = le.fit_transform(combined_df[col])

# Drop raw text / unused columns
drop_cols = ['Description', 'Log Number']
combined_df = combined_df.drop(columns=drop_cols)

# Ensure numeric columns are properly typed
for col in combined_df.columns:
    if combined_df[col].dtype == 'object':
        combined_df[col] = combined_df[col].astype('category')

X_train = combined_df.iloc[:n_train].copy()
X_test = combined_df.iloc[n_train:].copy()

# Cross-validation and Ensembling
kf = KFold(n_splits=5, shuffle=True, random_state=42)

lgb_preds = np.zeros(n_test)
cb_preds = np.zeros(n_test)

lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'learning_rate': 0.05,
    'num_leaves': 63,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'random_state': 42,
    'verbose': -1,
    'n_estimators': 1000
}

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
    X_tr, y_tr = X_train.iloc[train_idx], y_train[train_idx]
    X_va, y_va = X_train.iloc[val_idx], y_train[val_idx]
    
    # LightGBM
    model_lgb = lgb.LGBMRegressor(**lgb_params)
    model_lgb.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )
    lgb_preds += model_lgb.predict(X_test) / kf.n_splits
    
    # CatBoost
    model_cb = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        loss_function='RMSE',
        random_seed=42 + fold,
        verbose=False,
        early_stopping_rounds=50
    )
    model_cb.fit(X_tr, y_tr, eval_set=(X_va, y_va))
    cb_preds += model_cb.predict(X_test) / kf.n_splits

# Blend predictions
final_preds = 0.5 * lgb_preds + 0.5 * cb_preds

# Save predictions
output_df = pd.DataFrame({
    'id': np.arange(len(final_preds)),
    'Federal Funds': final_preds
})
output_df.to_csv('predictions.csv', index=False)