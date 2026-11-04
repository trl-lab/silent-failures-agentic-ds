import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import lightgbm as lgb
from catboost import CatBoostRegressor

# Load data
train_path = '/path/to/user/workspace/ds_failures/datasets/strable/covid-clinical-trials/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/strable/covid-clinical-trials/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target = 'days_to_complete'
y = train_df[target].values

n_train = len(train_df)
combined_df = pd.concat([train_df.drop(columns=[target]), test_df], axis=0).reset_index(drop=True)

# Feature Engineering
features = pd.DataFrame(index=combined_df.index)

# 1. Numerical features
features['Rank'] = combined_df['Rank'].astype(float)
features['Enrollment'] = combined_df['Enrollment'].astype(float)
features['Enrollment_log'] = np.log1p(np.maximum(0, combined_df['Enrollment'].fillna(0)))

# 2. Date features
date_cols = ['Primary Completion Date', 'First Posted', 'Last Update Posted', 'Results First Posted']
parsed_dates = {}
for col in date_cols:
    if col in combined_df.columns:
        parsed_dates[col] = pd.to_datetime(combined_df[col], errors='coerce')
        features[f'{col}_year'] = parsed_dates[col].dt.year.astype(float)
        features[f'{col}_month'] = parsed_dates[col].dt.month.astype(float)
        features[f'{col}_day'] = parsed_dates[col].dt.day.astype(float)
        features[f'{col}_dayofweek'] = parsed_dates[col].dt.dayofweek.astype(float)
        features[f'{col}_epoch_days'] = (parsed_dates[col] - pd.Timestamp('2020-01-01')).dt.total_seconds() / 86400.0

if 'Primary Completion Date' in parsed_dates and 'First Posted' in parsed_dates:
    diff_days = (parsed_dates['Primary Completion Date'] - parsed_dates['First Posted']).dt.total_seconds() / 86400.0
    features['diff_primary_first_posted'] = diff_days
    features['diff_primary_first_posted_log'] = np.log1p(np.maximum(0, diff_days))

if 'Last Update Posted' in parsed_dates and 'First Posted' in parsed_dates:
    diff_update = (parsed_dates['Last Update Posted'] - parsed_dates['First Posted']).dt.total_seconds() / 86400.0
    features['diff_update_first_posted'] = diff_update

# 3. Categorical counts & string lengths
text_and_cat_cols = [
    'Status', 'Study Results', 'Conditions', 'Interventions', 'Outcome Measures',
    'Sponsor/Collaborators', 'Gender', 'Age', 'Phases', 'Funded Bys',
    'Study Type', 'Study Designs', 'Locations'
]

for col in text_and_cat_cols:
    if col in combined_df.columns:
        col_str = combined_df[col].astype(str)
        features[f'{col}_len'] = col_str.str.len()
        features[f'{col}_word_count'] = col_str.str.split().apply(lambda x: len(x) if isinstance(x, list) else 0)
        features[f'{col}_pipe_count'] = col_str.apply(lambda x: x.count('|'))
        features[f'{col}_comma_count'] = col_str.apply(lambda x: x.count(','))
        
        # Frequency encoding
        freq = col_str.value_counts()
        features[f'{col}_freq'] = col_str.map(freq).astype(float)
        
        # Categorical type for LightGBM
        features[f'{col}_cat'] = combined_df[col].astype('category').cat.codes

# 4. Text embeddings via TF-IDF + SVD on aggregated text
combined_text = (
    combined_df['Title'].fillna('') + ' ' +
    combined_df['Conditions'].fillna('') + ' ' +
    combined_df['Interventions'].fillna('') + ' ' +
    combined_df['Outcome Measures'].fillna('')
)

tfidf = TfidfVectorizer(max_features=5000, stop_words='english', token_pattern=r'(?u)\b\w+\b')
tfidf_mat = tfidf.fit_transform(combined_text)
svd = TruncatedSVD(n_components=20, random_state=42)
svd_feats = svd.fit_transform(tfidf_mat)

for i in range(20):
    features[f'svd_{i}'] = svd_feats[:, i]

# Split back to train and test
X_train = features.iloc[:n_train].copy()
X_test = features.iloc[n_train:].copy()

# Cross-validated modeling
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

lgb_preds = np.zeros(len(X_test))
cat_preds = np.zeros(len(X_test))

cat_features_idx = [c for c in X_train.columns if c.endswith('_cat')]

for train_idx, val_idx in kf.split(X_train):
    X_tr, y_tr = X_train.iloc[train_idx], y[train_idx]
    X_va, y_va = X_train.iloc[val_idx], y[val_idx]

    # LightGBM
    lgb_model = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=-1
    )
    lgb_model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    lgb_preds += lgb_model.predict(X_test) / n_splits

    # CatBoost
    cat_model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.03,
        depth=6,
        random_seed=42,
        verbose=False
    )
    cat_model.fit(
        X_tr, y_tr,
        eval_set=(X_va, y_va),
        early_stopping_rounds=50
    )
    cat_preds += cat_model.predict(X_test) / n_splits

# Ensemble predictions
final_predictions = 0.5 * lgb_preds + 0.5 * cat_preds

# Save predictions
output_df = pd.DataFrame({
    'id': np.arange(len(test_df)),
    'days_to_complete': final_predictions
})
output_df.to_csv('predictions.csv', index=False)