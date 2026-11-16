import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# Load data
train_path = '/path/to/user/workspace/ds_failures/datasets/strable/financial-product-complaint/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/strable/financial-product-complaint/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'Company response to consumer'
y_raw = train_df[target_col].astype(str)

le_target = LabelEncoder()
y = le_target.fit_transform(y_raw)
num_classes = len(le_target.classes_)

# Feature Engineering function
def extract_features(df):
    feats = pd.DataFrame(index=df.index)
    
    # Dates
    date_recv = pd.to_datetime(df['Date received'], errors='coerce')
    date_sent = pd.to_datetime(df['Date sent to company'], errors='coerce')
    
    feats['recv_year'] = date_recv.dt.year.fillna(-1).astype(int)
    feats['recv_month'] = date_recv.dt.month.fillna(-1).astype(int)
    feats['recv_day'] = date_recv.dt.day.fillna(-1).astype(int)
    feats['recv_dayofweek'] = date_recv.dt.dayofweek.fillna(-1).astype(int)
    
    feats['sent_year'] = date_sent.dt.year.fillna(-1).astype(int)
    feats['sent_month'] = date_sent.dt.month.fillna(-1).astype(int)
    feats['sent_day'] = date_sent.dt.day.fillna(-1).astype(int)
    feats['sent_dayofweek'] = date_sent.dt.dayofweek.fillna(-1).astype(int)
    
    feats['days_diff'] = (date_sent - date_recv).dt.days.fillna(-999).astype(float)
    
    # Text length stats
    narrative = df['Consumer complaint narrative'].fillna('')
    feats['narrative_is_missing'] = (df['Consumer complaint narrative'].isna()).astype(int)
    feats['narrative_len'] = narrative.str.len()
    feats['narrative_words'] = narrative.str.split().str.len()
    
    # Categoricals
    cat_cols = [
        'Product', 'Sub-product', 'Issue', 'Sub-issue', 
        'Company public response', 'Company', 'State', 
        'ZIP code', 'Tags', 'Consumer consent provided?', 'Submitted via'
    ]
    for col in cat_cols:
        feats[col] = df[col].fillna('Missing').astype(str)
        
    return feats

X_train_df = extract_features(train_df)
X_test_df = extract_features(test_df)

# Text TF-IDF + SVD on Consumer complaint narrative
tfidf = TfidfVectorizer(max_features=5000, stop_words='english', min_df=3)
narrative_train = train_df['Consumer complaint narrative'].fillna('')
narrative_test = test_df['Consumer complaint narrative'].fillna('')

tfidf_train = tfidf.fit_transform(narrative_train)
tfidf_test = tfidf.transform(narrative_test)

svd = TruncatedSVD(n_components=30, random_state=42)
svd_train = svd.fit_transform(tfidf_train)
svd_test = svd.transform(tfidf_test)

for i in range(30):
    X_train_df[f'svd_{i}'] = svd_train[:, i]
    X_test_df[f'svd_{i}'] = svd_test[:, i]

# Categorical encoding
cat_cols = [
    'Product', 'Sub-product', 'Issue', 'Sub-issue', 
    'Company public response', 'Company', 'State', 
    'ZIP code', 'Tags', 'Consumer consent provided?', 'Submitted via'
]

for col in cat_cols:
    le = LabelEncoder()
    # Combine unique values from train and test to handle unseen categories safely
    unique_vals = list(set(X_train_df[col].unique()).union(set(X_test_df[col].unique())))
    le.fit(unique_vals)
    X_train_df[col] = le.transform(X_train_df[col])
    X_test_df[col] = le.transform(X_test_df[col])
    X_train_df[col] = X_train_df[col].astype('category')
    X_test_df[col] = X_test_df[col].astype('category')

# K-Fold Training & Prediction
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
test_probs = np.zeros((len(test_df), num_classes))

lgb_params = {
    'objective': 'multiclass',
    'num_class': num_classes,
    'metric': 'multi_logloss',
    'boosting_type': 'gbdt',
    'n_estimators': 300,
    'learning_rate': 0.08,
    'num_leaves': 63,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1
}

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_df, y)):
    X_tr, y_tr = X_train_df.iloc[train_idx], y[train_idx]
    X_va, y_va = X_train_df.iloc[val_idx], y[val_idx]
    
    clf = lgb.LGBMClassifier(**lgb_params)
    clf.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]
    )
    
    test_probs += clf.predict_proba(X_test_df) / skf.n_splits

final_preds = np.argmax(test_probs, axis=1)
final_labels = le_target.inverse_transform(final_preds)

# Save predictions
output_df = pd.DataFrame({
    'id': np.arange(len(test_df)),
    'Company response to consumer': final_labels
})
output_df.to_csv('predictions.csv', index=False)