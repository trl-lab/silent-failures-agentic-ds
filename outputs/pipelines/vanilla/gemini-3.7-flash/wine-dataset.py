import re
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# Load datasets
train_df = pd.read_parquet('/path/to/user/workspace/ds_failures/datasets/strable/wine-dataset/data/train.parquet')
test_df = pd.read_parquet('/path/to/user/workspace/ds_failures/datasets/strable/wine-dataset/data/test.parquet')

y = train_df['Price'].values
n_train = len(train_df)
df = pd.concat([train_df.drop(columns=['Price']), test_df], ignore_index=True)

# Feature extraction: clean and parse structured columns
def extract_number(series):
    return series.astype(str).str.extract(r'(\d+\.?\d*)')[0].astype(float)

def extract_vintage(series):
    return series.astype(str).str.extract(r'(\d{4})')[0].astype(float)

df['ABV_num'] = extract_number(df['ABV'])
df['Capacity_num'] = extract_number(df['Capacity'])
df['Vintage_num'] = extract_vintage(df['Vintage'])
df['Unit_num'] = pd.to_numeric(df['Unit'], errors='coerce')

# Text concatenation for TF-IDF
text_combined = (
    df['Title'].fillna('') + ' ' +
    df['Description'].fillna('') + ' ' +
    df['Characteristics'].fillna('') + ' ' +
    df['Grape'].fillna('') + ' ' +
    df['Region'].fillna('') + ' ' +
    df['Appellation'].fillna('')
)

# Text TF-IDF + SVD features
tfidf = TfidfVectorizer(max_features=25000, stop_words='english', ngram_range=(1, 2))
tfidf_mat = tfidf.fit_transform(text_combined)

svd = TruncatedSVD(n_components=64, random_state=42)
svd_feats = svd.fit_transform(tfidf_mat)
svd_cols = [f'svd_{i}' for i in range(64)]
svd_df = pd.DataFrame(svd_feats, columns=svd_cols)

# Categorical columns encoding
cat_cols = ['Grape', 'Secondary Grape Varieties', 'Closure', 'Country', 
            'Per bottle / case / each', 'Type', 'Region', 'Style', 'Appellation']

for col in cat_cols:
    df[col] = df[col].astype('category')

# Frequency encoding
for col in cat_cols:
    freq = df[col].value_counts(normalize=True)
    df[f'{col}_freq'] = df[col].map(freq).astype(float)

num_cols = ['ABV_num', 'Capacity_num', 'Vintage_num', 'Unit_num'] + [f'{col}_freq' for col in cat_cols]

# Assemble tabular features
X_all = pd.concat([df[num_cols + cat_cols], svd_df], axis=1)

X_train = X_all.iloc[:n_train].copy()
X_test = X_all.iloc[n_train:].copy()
tfidf_train = tfidf_mat[:n_train]
tfidf_test = tfidf_mat[n_train:]

# K-Fold Out-of-fold Ridge on TF-IDF
kf = KFold(n_splits=5, shuffle=True, random_state=42)
ridge_oof = np.zeros(n_train)
ridge_test_preds = np.zeros(len(test_df))

for train_idx, val_idx in kf.split(X_train):
    ridge = Ridge(alpha=1.0)
    ridge.fit(tfidf_train[train_idx], y[train_idx])
    ridge_oof[val_idx] = ridge.predict(tfidf_train[val_idx])
    ridge_test_preds += ridge.predict(tfidf_test) / kf.n_splits

X_train['ridge_tfidf'] = ridge_oof
X_test['ridge_tfidf'] = ridge_test_preds

# Convert categories for LightGBM and CatBoost
for col in cat_cols:
    X_train[col] = X_train[col].astype('category')
    X_test[col] = X_test[col].astype('category')

# Train LightGBM & CatBoost models using 5-fold CV
lgb_preds = np.zeros(len(test_df))
cb_preds = np.zeros(len(test_df))

# Prepare numeric/encoded versions for CatBoost
cb_cat_cols = [c for c in cat_cols]
X_train_cb = X_train.copy()
X_test_cb = X_test.copy()
for col in cb_cat_cols:
    X_train_cb[col] = X_train_cb[col].astype(str).fillna('missing')
    X_test_cb[col] = X_test_cb[col].astype(str).fillna('missing')

for train_idx, val_idx in kf.split(X_train):
    # LightGBM
    lgb = LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1
    )
    lgb.fit(
        X_train.iloc[train_idx], y[train_idx],
        eval_set=[(X_train.iloc[val_idx], y[val_idx])],
        callbacks=[]
    )
    lgb_preds += lgb.predict(X_test) / kf.n_splits

    # CatBoost
    cb = CatBoostRegressor(
        iterations=800,
        learning_rate=0.05,
        depth=6,
        cat_features=cb_cat_cols,
        verbose=False,
        random_seed=42
    )
    cb.fit(
        X_train_cb.iloc[train_idx], y[train_idx],
        eval_set=(X_train_cb.iloc[val_idx], y[val_idx]),
        early_stopping_rounds=50,
        verbose=False
    )
    cb_preds += cb.predict(X_test_cb) / kf.n_splits

# Ensemble predictions
final_predictions = 0.5 * lgb_preds + 0.4 * cb_preds + 0.1 * ridge_test_preds

# Save predictions
output_df = pd.DataFrame({
    'id': np.arange(len(final_predictions)),
    'Price': final_predictions
})
output_df.to_csv('predictions.csv', index=False)