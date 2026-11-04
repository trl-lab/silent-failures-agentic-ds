import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Ridge
import lightgbm as lgb
from catboost import CatBoostRegressor

# Load datasets
train_df = pd.read_parquet('/path/to/user/workspace/ds_failures/datasets/strable/beer-ratings/data/train.parquet')
test_df = pd.read_parquet('/path/to/user/workspace/ds_failures/datasets/strable/beer-ratings/data/test.parquet')

target_col = 'review_overall'
y_train = train_df[target_col].values
X_train = train_df.drop(columns=[target_col]).copy()
X_test = test_df.copy()
if target_col in X_test.columns:
    X_test = X_test.drop(columns=[target_col])

# Combine text columns
text_cols = ['Name', 'Style', 'Brewery', 'Beer Name (Full)', 'Description']
for col in text_cols:
    X_train[col] = X_train[col].fillna('').astype(str)
    X_test[col] = X_test[col].fillna('').astype(str)

train_text = X_train[text_cols].agg(' '.join, axis=1)
test_text = X_test[text_cols].agg(' '.join, axis=1)

# TF-IDF + Ridge out-of-fold feature
tfidf = TfidfVectorizer(max_features=25000, stop_words='english', ngram_range=(1, 2))
X_train_tfidf = tfidf.fit_transform(train_text)
X_test_tfidf = tfidf.transform(test_text)

svd = TruncatedSVD(n_components=30, random_state=42)
train_svd = svd.fit_transform(X_train_tfidf)
test_svd = svd.transform(X_test_tfidf)

for i in range(30):
    X_train[f'svd_{i}'] = train_svd[:, i]
    X_test[f'svd_{i}'] = test_svd[:, i]

# Numerical feature engineering
num_cols = [
    'ABV', 'Min IBU', 'Max IBU', 'Astringency', 'Body', 'Alcohol',
    'Bitter', 'Sweet', 'Sour', 'Salty', 'Fruits', 'Hoppy', 'Spices',
    'Malty', 'number_of_reviews'
]

for df in [X_train, X_test]:
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    df['IBU_diff'] = df['Max IBU'] - df['Min IBU']
    df['IBU_ratio'] = (df['Max IBU'] + 1) / (df['Min IBU'] + 1)
    df['Bitter_Sweet_ratio'] = (df['Bitter'] + 1) / (df['Sweet'] + 1)
    df['Total_flavor'] = df[['Astringency', 'Body', 'Alcohol', 'Bitter', 'Sweet', 'Sour', 'Salty', 'Fruits', 'Hoppy', 'Spices', 'Malty']].sum(axis=1)
    df['Log_reviews'] = np.log1p(np.maximum(0, df['number_of_reviews']))

# Frequency encoding for high-cardinality categoricals
cat_cols = ['Style', 'Brewery']
for col in cat_cols:
    freq = X_train[col].value_counts()
    X_train[f'{col}_freq'] = X_train[col].map(freq).fillna(0)
    X_test[f'{col}_freq'] = X_test[col].map(freq).fillna(0)

# Drop raw text columns for tree-based models
feature_cols = [c for c in X_train.columns if c not in text_cols]
X_tr = X_train[feature_cols].copy()
X_te = X_test[feature_cols].copy()

# K-Fold Ensemble
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_preds_lgb = np.zeros(len(X_tr))
oof_preds_ridge = np.zeros(len(X_tr))
oof_preds_cat = np.zeros(len(X_tr))

test_preds_lgb = np.zeros(len(X_te))
test_preds_ridge = np.zeros(len(X_te))
test_preds_cat = np.zeros(len(X_te))

for train_idx, val_idx in kf.split(X_tr):
    # Ridge on TF-IDF
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train_tfidf[train_idx], y_train[train_idx])
    oof_preds_ridge[val_idx] = ridge.predict(X_train_tfidf[val_idx])
    test_preds_ridge += ridge.predict(X_test_tfidf) / n_splits

    # LightGBM
    X_train_fold, y_train_fold = X_tr.iloc[train_idx].copy(), y_train[train_idx]
    X_val_fold, y_val_fold = X_tr.iloc[val_idx].copy(), y_train[val_idx]
    
    # Add Ridge prediction as feature
    X_train_fold['ridge_pred'] = ridge.predict(X_train_tfidf[train_idx])
    X_val_fold['ridge_pred'] = oof_preds_ridge[val_idx]
    X_te_fold = X_te.copy()
    X_te_fold['ridge_pred'] = ridge.predict(X_test_tfidf)

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
        X_train_fold, y_train_fold,
        eval_set=[(X_val_fold, y_val_fold)],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    oof_preds_lgb[val_idx] = lgb_model.predict(X_val_fold)
    test_preds_lgb += lgb_model.predict(X_te_fold) / n_splits

    # CatBoost
    cat_model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.04,
        depth=6,
        random_seed=42,
        verbose=0
    )
    cat_model.fit(
        X_train_fold, y_train_fold,
        eval_set=(X_val_fold, y_val_fold),
        early_stopping_rounds=50
    )
    oof_preds_cat[val_idx] = cat_model.predict(X_val_fold)
    test_preds_cat += cat_model.predict(X_te_fold) / n_splits

final_predictions = 0.5 * test_preds_lgb + 0.4 * test_preds_cat + 0.1 * test_preds_ridge

# Create submission file
submission = pd.DataFrame({
    'id': range(len(test_df)),
    'review_overall': final_predictions
})
submission.to_csv('predictions.csv', index=False)