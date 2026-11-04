import pandas as pd
import numpy as np
import re
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from scipy.sparse import hstack

# Load data
train_path = '/path/to/user/workspace/ds_failures/datasets/strable/journal-ranking_wide/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/strable/journal-ranking_wide/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'H index'
y_train = train_df[target_col].values
X_train_raw = train_df.drop(columns=[target_col])
X_test_raw = test_df.copy()

# Feature extraction function
def extract_features(df):
    df_feat = pd.DataFrame(index=df.index)
    
    # Numeric features
    num_cols = ['Total Refs.', 'Overton', 'SDG']
    for col in num_cols:
        if col in df.columns:
            df_feat[col] = df[col].astype(float)
            df_feat[f'{col}_log1p'] = np.log1p(np.maximum(0, df[col].astype(float)))
            
    # Extract years from Coverage
    def parse_coverage(val):
        if not isinstance(val, str) or not val.strip():
            return np.nan, np.nan, np.nan
        years = [int(y) for y in re.findall(r'\b(19\d\d|20\d\d)\b', val)]
        if not years:
            return np.nan, np.nan, np.nan
        min_y = min(years)
        max_y = max(years)
        span = max_y - min_y + 1
        return min_y, max_y, span

    coverage_parsed = df['Coverage'].apply(parse_coverage)
    df_feat['coverage_min_year'] = [p[0] for p in coverage_parsed]
    df_feat['coverage_max_year'] = [p[1] for p in coverage_parsed]
    df_feat['coverage_span'] = [p[2] for p in coverage_parsed]
    
    # String length and counts
    for col in ['Categories', 'Areas', 'Title', 'Publisher']:
        if col in df.columns:
            s = df[col].fillna('').astype(str)
            df_feat[f'{col}_len'] = s.apply(len)
            df_feat[f'{col}_word_count'] = s.apply(lambda x: len(x.split()))
            if col in ['Categories', 'Areas']:
                df_feat[f'{col}_item_count'] = s.apply(lambda x: len(x.split(';')) if x else 0)

    # Q-rankings counts in Categories
    if 'Categories' in df.columns:
        cat_str = df['Categories'].fillna('').astype(str)
        df_feat['q1_count'] = cat_str.apply(lambda x: x.count('(Q1)'))
        df_feat['q2_count'] = cat_str.apply(lambda x: x.count('(Q2)'))
        df_feat['q3_count'] = cat_str.apply(lambda x: x.count('(Q3)'))
        df_feat['q4_count'] = cat_str.apply(lambda x: x.count('(Q4)'))

    # Categorical columns
    cat_cols = ['Type', 'Country', 'Region', 'Publisher']
    for col in cat_cols:
        if col in df.columns:
            df_feat[col] = df[col].fillna('Missing').astype('category')
            
    return df_feat

X_train_df = extract_features(X_train_raw)
X_test_df = extract_features(X_test_raw)

# Align categorical types
cat_cols = ['Type', 'Country', 'Region', 'Publisher']
for col in cat_cols:
    categories = list(set(X_train_df[col].astype(str)).union(set(X_test_df[col].astype(str))))
    X_train_df[col] = pd.Categorical(X_train_df[col].astype(str), categories=categories)
    X_test_df[col] = pd.Categorical(X_test_df[col].astype(str), categories=categories)

# TF-IDF on combined text
train_text = (X_train_raw['Title'].fillna('') + ' ' + 
              X_train_raw['Categories'].fillna('') + ' ' + 
              X_train_raw['Areas'].fillna('') + ' ' + 
              X_train_raw['Publisher'].fillna('')).astype(str)

test_text = (X_test_raw['Title'].fillna('') + ' ' + 
             X_test_raw['Categories'].fillna('') + ' ' + 
             X_test_raw['Areas'].fillna('') + ' ' + 
             X_test_raw['Publisher'].fillna('')).astype(str)

tfidf = TfidfVectorizer(max_features=100, ngram_range=(1, 2), stop_words='english')
X_train_tfidf = tfidf.fit_transform(train_text).toarray()
X_test_tfidf = tfidf.transform(test_text).toarray()

tfidf_cols = [f'tfidf_{i}' for i in range(X_train_tfidf.shape[1])]
for i, col_name in enumerate(tfidf_cols):
    X_train_df[col_name] = X_train_tfidf[:, i]
    X_test_df[col_name] = X_test_tfidf[:, i]

# 5-fold cross-validation with LightGBM and CatBoost
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

lgb_preds = np.zeros(len(X_test_df))
cb_preds = np.zeros(len(X_test_df))

# Prepare for CatBoost
cb_cat_cols = cat_cols
X_train_cb = X_train_df.copy()
X_test_cb = X_test_df.copy()
for col in cb_cat_cols:
    X_train_cb[col] = X_train_cb[col].astype(str)
    X_test_cb[col] = X_test_cb[col].astype(str)

for train_idx, val_idx in kf.split(X_train_df):
    X_tr_lgb, y_tr = X_train_df.iloc[train_idx], y_train[train_idx]
    X_va_lgb, y_va = X_train_df.iloc[val_idx], y_train[val_idx]
    
    lgb_model = LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=31,
        random_state=42,
        verbosity=-1
    )
    lgb_model.fit(
        X_tr_lgb, y_tr,
        eval_set=[(X_va_lgb, y_va)],
        callbacks=[]
    )
    lgb_preds += lgb_model.predict(X_test_df) / n_splits

    X_tr_cb = X_train_cb.iloc[train_idx]
    X_va_cb = X_train_cb.iloc[val_idx]
    
    cb_model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.04,
        depth=6,
        cat_features=cb_cat_cols,
        verbose=0,
        random_seed=42
    )
    cb_model.fit(
        X_tr_cb, y_tr,
        eval_set=(X_va_cb, y_va),
        early_stopping_rounds=50,
        verbose=False
    )
    cb_preds += cb_model.predict(X_test_cb) / n_splits

final_predictions = 0.5 * lgb_preds + 0.5 * cb_preds

# Save predictions
output_df = pd.DataFrame({
    'id': np.arange(len(final_predictions)),
    'H index': final_predictions
})
output_df.to_csv('predictions.csv', index=False)