import re
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import lightgbm as lgb
from catboost import CatBoostClassifier

train_path = '/path/to/user/workspace/ds_failures/datasets/strable/drug-drugsfda/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/strable/drug-drugsfda/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'marketing_status'
le_target = LabelEncoder()
y_train = le_target.fit_transform(train_df[target_col].astype(str))

def extract_features(df):
    feats = pd.DataFrame(index=df.index)
    
    # application_number features
    app_num = df['application_number'].fillna('').astype(str)
    feats['app_type'] = app_num.apply(lambda x: re.match(r'^([A-Za-z]+)', x).group(1) if re.match(r'^([A-Za-z]+)', x) else 'OTHER')
    feats['app_num_val'] = app_num.apply(lambda x: float(re.search(r'(\d+)', x).group(1)) if re.search(r'(\d+)', x) else np.nan)
    
    # submissions features
    subs = df['submissions'].fillna('').astype(str)
    feats['sub_len'] = subs.str.len()
    feats['sub_suppl_count'] = subs.apply(lambda x: x.count('submission_type: SUPPL'))
    feats['sub_orig_count'] = subs.apply(lambda x: x.count('submission_type: ORIG'))
    feats['sub_ap_count'] = subs.apply(lambda x: x.count('submission_status: AP'))
    feats['sub_ta_count'] = subs.apply(lambda x: x.count('submission_status: TA'))
    feats['sub_labeling_count'] = subs.apply(lambda x: x.count('LABELING'))
    feats['sub_cmc_count'] = subs.apply(lambda x: x.count('MANUF (CMC)'))
    
    # dates extraction from submissions
    def extract_dates(text):
        dates = re.findall(r'submission_status_date:\s*(\d{8})', text)
        if dates:
            dates = [int(d) for d in dates]
            return min(dates), max(dates), len(dates)
        return np.nan, np.nan, 0
    
    date_info = subs.apply(extract_dates)
    feats['min_date'] = [d[0] for d in date_info]
    feats['max_date'] = [d[1] for d in date_info]
    feats['date_diff'] = feats['max_date'] - feats['min_date']
    feats['total_submissions'] = [d[2] for d in date_info]
    
    # Text length features
    for col in ['sponsor_name', 'strengths', 'brand_name', 'dosage_form']:
        col_series = df[col].fillna('').astype(str)
        feats[f'{col}_len'] = col_series.str.len()
        feats[f'{col}_word_count'] = col_series.apply(lambda x: len(x.split()))
        
    return feats

X_train_num = extract_features(train_df)
X_test_num = extract_features(test_df)

cat_cols = ['sponsor_name', 'reference_drug', 'brand_name', 'dosage_form', 'app_type']
for col in ['sponsor_name', 'reference_drug', 'brand_name', 'dosage_form']:
    X_train_num[col] = train_df[col].fillna('MISSING').astype(str)
    X_test_num[col] = test_df[col].fillna('MISSING').astype(str)

# Text TF-IDF + SVD on submissions and combined text
train_text = (train_df['submissions'].fillna('') + ' ' + 
              train_df['brand_name'].fillna('') + ' ' + 
              train_df['dosage_form'].fillna('') + ' ' + 
              train_df['sponsor_name'].fillna('')).astype(str)

test_text = (test_df['submissions'].fillna('') + ' ' + 
             test_df['brand_name'].fillna('') + ' ' + 
             test_df['dosage_form'].fillna('') + ' ' + 
             test_df['sponsor_name'].fillna('')).astype(str)

tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_tfidf = tfidf.fit_transform(train_text)
X_test_tfidf = tfidf.transform(test_text)

svd = TruncatedSVD(n_components=30, random_state=42)
X_train_svd = svd.fit_transform(X_train_tfidf)
X_test_svd = svd.transform(X_test_tfidf)

for i in range(30):
    X_train_num[f'svd_{i}'] = X_train_svd[:, i]
    X_test_num[f'svd_{i}'] = X_test_svd[:, i]

for c in cat_cols:
    le = LabelEncoder()
    full_vals = pd.concat([X_train_num[c], X_test_num[c]], axis=0).astype(str)
    le.fit(full_vals)
    X_train_num[c] = le.transform(X_train_num[c].astype(str))
    X_test_num[c] = le.transform(X_test_num[c].astype(str))

for c in cat_cols:
    X_train_num[c] = X_train_num[c].astype('category')
    X_test_num[c] = X_test_num[c].astype('category')

# Modeling with K-Fold cross validation
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

lgb_preds = np.zeros(len(test_df))
cat_preds = np.zeros(len(test_df))

cat_cols_idx = [X_train_num.columns.get_loc(c) for c in cat_cols]

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_num, y_train)):
    X_tr, y_tr = X_train_num.iloc[train_idx], y_train[train_idx]
    X_va, y_va = X_train_num.iloc[val_idx], y_train[val_idx]
    
    # LightGBM
    lgb_model = lgb.LGBMClassifier(
        n_estimators=600,
        learning_rate=0.03,
        num_leaves=31,
        random_state=42 + fold,
        verbose=-1
    )
    lgb_model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )
    lgb_preds += lgb_model.predict_proba(X_test_num)[:, 1] / n_splits
    
    # CatBoost
    X_tr_cat = X_tr.copy()
    X_va_cat = X_va.copy()
    X_test_cat = X_test_num.copy()
    for c in cat_cols:
        X_tr_cat[c] = X_tr_cat[c].astype(int)
        X_va_cat[c] = X_va_cat[c].astype(int)
        X_test_cat[c] = X_test_cat[c].astype(int)
        
    cb_model = CatBoostClassifier(
        iterations=600,
        learning_rate=0.03,
        depth=6,
        cat_features=cat_cols_idx,
        random_seed=42 + fold,
        verbose=0
    )
    cb_model.fit(
        X_tr_cat, y_tr,
        eval_set=(X_va_cat, y_va),
        early_stopping_rounds=50,
        verbose=False
    )
    cat_preds += cb_model.predict_proba(X_test_cat)[:, 1] / n_splits

final_preds = 0.5 * lgb_preds + 0.5 * cat_preds

submission = pd.DataFrame({
    'id': np.arange(len(test_df)),
    'marketing_status': final_preds
})

submission.to_csv('predictions.csv', index=False)