import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import lightgbm as lgb
from catboost import CatBoostRegressor

# Load data
train_path = '/path/to/user/workspace/ds_failures/datasets/strable/industry-payments-project/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/strable/industry-payments-project/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'projectPayment'
y = train_df[target_col].values
n_train = len(train_df)
n_test = len(test_df)

# Combine for consistent feature engineering
combined_df = pd.concat([train_df.drop(columns=[target_col]), test_df], ignore_index=True)

# Handle missing values and convert object columns to string
for col in combined_df.columns:
    if combined_df[col].dtype == 'object' or combined_df[col].dtype == 'string':
        combined_df[col] = combined_df[col].fillna('__MISSING__').astype(str)
    else:
        combined_df[col] = combined_df[col].fillna(-1)

# Text features engineering
text_cols = ['nrgiNotes', 'notes', 'projectName', 'reportingCompany']
for col in text_cols:
    combined_df[f'{col}_len'] = combined_df[col].astype(str).apply(len)
    combined_df[f'{col}_word_count'] = combined_df[col].astype(str).apply(lambda x: len(x.split()))

# Combined text for TF-IDF + SVD
combined_text = (
    combined_df['projectName'].astype(str) + ' ' +
    combined_df['nrgiNotes'].astype(str) + ' ' +
    combined_df['notes'].astype(str) + ' ' +
    combined_df['reportingCompany'].astype(str)
)

tfidf = TfidfVectorizer(max_features=5000, stop_words='english', token_pattern=r'(?u)\b\w+\b')
tfidf_mat = tfidf.fit_transform(combined_text)

svd = TruncatedSVD(n_components=20, random_state=42)
svd_features = svd.fit_transform(tfidf_mat)
for i in range(20):
    combined_df[f'svd_{i}'] = svd_features[:, i]

# Categorical columns
cat_cols = ['reportingCompany', 'projectCountry', 'paymentType', 'projectName', 'reportYear']
for col in cat_cols:
    combined_df[f'{col}_freq'] = combined_df[col].map(combined_df[col].value_counts())
    combined_df[col] = combined_df[col].astype('category')

# Drop unused raw text columns
combined_df = combined_df.drop(columns=['sourceID', 'nrgiNotes', 'notes'])

# Split back into train and test
X_train = combined_df.iloc[:n_train].copy()
X_test = combined_df.iloc[n_train:].copy()

# Setup K-Fold Cross Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
lgb_preds = np.zeros(n_test)
cat_preds = np.zeros(n_test)

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y)):
    X_tr, y_tr = X_train.iloc[train_idx], y[train_idx]
    X_va, y_va = X_train.iloc[val_idx], y[val_idx]

    # LightGBM
    lgb_train = lgb.Dataset(X_tr, label=y_tr)
    lgb_val = lgb.Dataset(X_va, label=y_va, reference=lgb_train)

    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 63,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'random_state': 42 + fold,
        'verbose': -1,
        'n_jobs': -1
    }

    lgb_model = lgb.train(
        params,
        lgb_train,
        num_boost_round=1000,
        valid_sets=[lgb_train, lgb_val],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )

    lgb_preds += lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration) / kf.n_splits

    # CatBoost
    cat_cols_cb = [col for col in cat_cols if col in X_train.columns]
    X_tr_cb = X_tr.copy()
    X_va_cb = X_va.copy()
    X_test_cb = X_test.copy()
    for col in cat_cols_cb:
        X_tr_cb[col] = X_tr_cb[col].astype(str)
        X_va_cb[col] = X_va_cb[col].astype(str)
        X_test_cb[col] = X_test_cb[col].astype(str)

    cb_model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        cat_features=cat_cols_cb,
        random_seed=42 + fold,
        verbose=False
    )
    cb_model.fit(X_tr_cb, y_tr, eval_set=(X_va_cb, y_va), early_stopping_rounds=50, verbose=False)
    cat_preds += cb_model.predict(X_test_cb) / kf.n_splits

# Ensemble predictions
final_predictions = 0.5 * lgb_preds + 0.5 * cat_preds

# Save submission
submission = pd.DataFrame({
    'id': np.arange(n_test),
    'projectPayment': final_predictions
})
submission.to_csv('predictions.csv', index=False)