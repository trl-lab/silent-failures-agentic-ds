import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
from catboost import CatBoostRegressor

# Load data
train_path = '/path/to/user/workspace/ds_failures/datasets/strable/first-time-nadac-rates/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/strable/first-time-nadac-rates/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'NADAC Rate'
y_train = train_df[target_col].values

# Feature Engineering
def preprocess_df(df):
    df = df.copy()
    
    # Clean Package Size
    if 'Package Size' in df.columns:
        df['Package_Size_num'] = pd.to_numeric(
            df['Package Size'].astype(str).str.replace(r'[^\d.]', '', regex=True), 
            errors='coerce'
        )
    else:
        df['Package_Size_num'] = np.nan

    # Parse As of Date
    if 'As of Date' in df.columns:
        dt = pd.to_datetime(df['As of Date'], errors='coerce')
        df['date_year'] = dt.dt.year
        df['date_month'] = dt.dt.month
        df['date_day'] = dt.dt.day
        df['date_dayofweek'] = dt.dt.dayofweek
        df['date_dayofyear'] = dt.dt.dayofyear
        df['date_timestamp'] = dt.astype('int64') // 10**9
    else:
        for c in ['date_year', 'date_month', 'date_day', 'date_dayofweek', 'date_dayofyear', 'date_timestamp']:
            df[c] = np.nan

    # Text length features
    for col in ['Drug Product', 'Brief Indication/Description']:
        if col in df.columns:
            df[f'{col}_len'] = df[col].astype(str).str.len()
            df[f'{col}_word_count'] = df[col].astype(str).apply(lambda x: len(x.split()))

    return df

train_processed = preprocess_df(train_df)
test_processed = preprocess_df(test_df)

# Text features via TF-IDF + SVD on Drug Product
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), token_pattern=r'(?u)\b\w+\b')
svd = TruncatedSVD(n_components=16, random_state=42)

train_drug_text = train_df['Drug Product'].fillna('').astype(str)
test_drug_text = test_df['Drug Product'].fillna('').astype(str)

train_tfidf = tfidf.fit_transform(train_drug_text)
test_tfidf = tfidf.transform(test_drug_text)

train_svd = svd.fit_transform(train_tfidf)
test_svd = svd.transform(test_tfidf)

for i in range(16):
    train_processed[f'drug_svd_{i}'] = train_svd[:, i]
    test_processed[f'drug_svd_{i}'] = test_svd[:, i]

# Categorical columns handling
cat_cols = [
    'Drug Class',
    'NCPDP Billing Unit Standard',
    'Primary Reason Code',
    'Classification for Rate Setting',
    'Brief Indication/Description'
]

drop_cols = [target_col, 'Drug Product', 'Package Size', 'As of Date']
feature_cols = [c for c in train_processed.columns if c not in drop_cols]

for col in cat_cols:
    if col in feature_cols:
        train_processed[col] = train_processed[col].astype(str).fillna('missing').astype('category')
        test_processed[col] = test_processed[col].astype(str).fillna('missing').astype('category')

X_train = train_processed[feature_cols].copy()
X_test = test_processed[feature_cols].copy()

# K-Fold Ensemble
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

lgb_preds = np.zeros(len(test_df))
cat_preds = np.zeros(len(test_df))

lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'random_state': 42,
    'verbose': -1,
    'n_estimators': 1000
}

# String type categoricals for CatBoost
X_train_cat = X_train.copy()
X_test_cat = X_test.copy()
for col in cat_cols:
    if col in X_train_cat.columns:
        X_train_cat[col] = X_train_cat[col].astype(str)
        X_test_cat[col] = X_test_cat[col].astype(str)

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_tr, y_tr = X_train.iloc[train_idx], y_train[train_idx]
    X_va, y_va = X_train.iloc[val_idx], y_train[val_idx]
    
    # LightGBM
    lgb_model = lgb.LGBMRegressor(**lgb_params)
    lgb_model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )
    lgb_preds += lgb_model.predict(X_test) / n_splits
    
    # CatBoost
    cat_model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        eval_metric='RMSE',
        random_seed=42 + fold,
        verbose=False,
        cat_features=cat_cols
    )
    cat_model.fit(
        X_train_cat.iloc[train_idx], y_tr,
        eval_set=(X_train_cat.iloc[val_idx], y_va),
        early_stopping_rounds=50,
        verbose=False
    )
    cat_preds += cat_model.predict(X_test_cat) / n_splits

# Ensemble prediction
final_preds = 0.5 * lgb_preds + 0.5 * cat_preds

# Save predictions
output_df = pd.DataFrame({
    'id': np.arange(len(final_preds)),
    'NADAC Rate': final_preds
})
output_df.to_csv('predictions.csv', index=False)