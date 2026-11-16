import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import root_mean_squared_error
import lightgbm as lgb
from catboost import CatBoostRegressor

train_path = '/path/to/user/workspace/ds_failures/datasets/strable/clear-corpus/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/strable/clear-corpus/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'BT Easiness'
y_train = train_df[target_col].values

num_cols = [
    'Pub Year', 'MPAA #Max', 'MPAA #Avg', 'Google WC', 'Joon WC v1', 'British WC',
    'Sentence Count v1', 'Sentence Count v2', 'Paragraphs',
    'Flesch-Reading-Ease', 'Flesch-Kincaid-Grade-Level', 'Automated Readability Index',
    'SMOG Readability', 'New Dale-Chall Readability Formula', 'CAREC', 'CAREC_M', 'CARES', 'CML2RI'
]

cat_cols = ['Category', 'Location', 'License', 'MPAA Max', 'Source']

for col in num_cols:
    train_df[col] = train_df[col].astype(float)
    test_df[col] = test_df[col].astype(float)
    median_val = train_df[col].median()
    train_df[col] = train_df[col].fillna(median_val)
    test_df[col] = test_df[col].fillna(median_val)

for col in cat_cols:
    train_df[col] = train_df[col].astype(str).fillna('missing')
    test_df[col] = test_df[col].astype(str).fillna('missing')

train_excerpts = train_df['Excerpt'].fillna('').values
test_excerpts = test_df['Excerpt'].fillna('').values

n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

ridge_oof = np.zeros(len(train_df))
ridge_test_preds = np.zeros(len(test_df))

tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=10000, sublinear_tf=True)
X_tfidf_train = tfidf.fit_transform(train_excerpts)
X_tfidf_test = tfidf.transform(test_excerpts)

for train_idx, val_idx in kf.split(train_df):
    X_tr_tfidf, y_tr = X_tfidf_train[train_idx], y_train[train_idx]
    X_va_tfidf = X_tfidf_train[val_idx]
    
    ridge = Ridge(alpha=1.0, random_state=42)
    ridge.fit(X_tr_tfidf, y_tr)
    
    ridge_oof[val_idx] = ridge.predict(X_va_tfidf)
    ridge_test_preds += ridge.predict(X_tfidf_test) / n_splits

train_df['ridge_pred'] = ridge_oof
test_df['ridge_pred'] = ridge_test_preds

feature_cols = num_cols + ['ridge_pred']

for col in cat_cols:
    train_df[col] = train_df[col].astype('category')
    test_df[col] = test_df[col].astype('category')
    feature_cols.append(col)

X_train = train_df[feature_cols]
X_test = test_df[feature_cols]

lgb_test_preds = np.zeros(len(test_df))
cb_test_preds = np.zeros(len(test_df))

for train_idx, val_idx in kf.split(X_train):
    X_tr, y_tr = X_train.iloc[train_idx], y_train[train_idx]
    X_va, y_va = X_train.iloc[val_idx], y_train[val_idx]
    
    lgb_model = lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.03,
        num_leaves=31,
        random_state=42,
        verbosity=-1
    )
    lgb_model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    lgb_test_preds += lgb_model.predict(X_test) / n_splits

    cat_indices = [i for i, c in enumerate(feature_cols) if c in cat_cols]
    cb_model = CatBoostRegressor(
        iterations=600,
        learning_rate=0.03,
        depth=5,
        cat_features=cat_indices,
        random_seed=42,
        verbose=False
    )
    cb_model.fit(
        X_tr, y_tr,
        eval_set=(X_va, y_va),
        early_stopping_rounds=50,
        verbose=False
    )
    cb_test_preds += cb_model.predict(X_test) / n_splits

final_predictions = 0.5 * lgb_test_preds + 0.5 * cb_test_preds

submission = pd.DataFrame({
    'id': np.arange(len(final_predictions)),
    'BT Easiness': final_predictions
})
submission.to_csv('predictions.csv', index=False)