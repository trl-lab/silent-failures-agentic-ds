import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from scipy.sparse import hstack, csr_matrix

# 1. Load Data
train_df = pd.read_parquet('/path/to/user/workspace/ds_failures/datasets/strable/drug-shortages/data/train.parquet')
test_df = pd.read_parquet('/path/to/user/workspace/ds_failures/datasets/strable/drug-shortages/data/test.parquet')

# 2. Target Encoding
le_target = LabelEncoder()
y_train = le_target.fit_transform(train_df['availability'].astype(str))
n_classes = len(le_target.classes_)

# 3. Feature Engineering - Dates & Tabular
def process_tabular(df, full_df_ref=None):
    features = pd.DataFrame(index=df.index)
    
    # Date features
    dates = pd.to_datetime(df['initial_posting_date'], errors='coerce')
    features['year'] = dates.dt.year.fillna(-1)
    features['month'] = dates.dt.month.fillna(-1)
    features['day'] = dates.dt.day.fillna(-1)
    features['dayofweek'] = dates.dt.dayofweek.fillna(-1)
    features['dayofyear'] = dates.dt.dayofyear.fillna(-1)
    
    # Numeric timestamp
    timestamps = dates.astype('int64') // 10**9
    features['timestamp'] = timestamps.where(dates.notnull(), -1)
    
    # Text length features
    text_cols = ['generic_name', 'openfda', 'therapeutic_category', 'dosage_form', 'presentation', 'company_name']
    for col in text_cols:
        col_str = df[col].fillna('').astype(str)
        features[f'{col}_len'] = col_str.str.len()
        features[f'{col}_word_count'] = col_str.str.split().str.len()
    
    # Frequency encoding
    ref = full_df_ref if full_df_ref is not None else df
    for col in ['company_name', 'dosage_form', 'therapeutic_category', 'generic_name', 'contact_info']:
        freq = ref[col].value_counts(normalize=True).to_dict()
        features[f'{col}_freq'] = df[col].map(freq).fillna(0)
        
    return features

# Combined reference for frequency encoding
full_data = pd.concat([train_df.drop(columns=['availability']), test_df], axis=0, ignore_index=True)

X_train_tab = process_tabular(train_df, full_df_ref=full_data)
X_test_tab = process_tabular(test_df, full_df_ref=full_data)

scaler = StandardScaler()
X_train_num_scaled = scaler.fit_transform(X_train_tab.fillna(0))
X_test_num_scaled = scaler.transform(X_test_tab.fillna(0))

# 4. Text Representation
text_cols = ['generic_name', 'contact_info', 'openfda', 'therapeutic_category', 'dosage_form', 'presentation', 'company_name']
train_text = train_df[text_cols].fillna('').astype(str).agg(' '.join, axis=1)
test_text = test_df[text_cols].fillna('').astype(str).agg(' '.join, axis=1)

# 5. Cross-Validation & Modeling
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

test_preds_lr = np.zeros((len(test_df), n_classes))
test_preds_lgb = np.zeros((len(test_df), n_classes))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_tab, y_train)):
    # Text vectorization per fold
    tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), sublinear_tf=True, min_df=2)
    X_train_text_fold = tfidf.fit_transform(train_text.iloc[train_idx])
    X_val_text_fold = tfidf.transform(train_text.iloc[val_idx])
    X_test_text_fold = tfidf.transform(test_text)
    
    # Combine sparse text features and scaled dense features
    X_train_fold = hstack([csr_matrix(X_train_num_scaled[train_idx]), X_train_text_fold]).tocsr()
    X_val_fold = hstack([csr_matrix(X_train_num_scaled[val_idx]), X_val_text_fold]).tocsr()
    X_test_fold = hstack([csr_matrix(X_test_num_scaled), X_test_text_fold]).tocsr()
    
    y_train_fold = y_train[train_idx]
    y_val_fold = y_train[val_idx]
    
    # Model 1: Logistic Regression
    clf_lr = LogisticRegression(max_iter=1000, C=2.0, random_state=42)
    clf_lr.fit(X_train_fold, y_train_fold)
    test_preds_lr += clf_lr.predict_proba(X_test_fold) / skf.n_splits
    
    # Model 2: LightGBM Classifier
    clf_lgb = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        verbose=-1,
        n_jobs=-1
    )
    clf_lgb.fit(
        X_train_fold,
        y_train_fold,
        eval_set=[(X_val_fold, y_val_fold)]
    )
    test_preds_lgb += clf_lgb.predict_proba(X_test_fold) / skf.n_splits

# 6. Ensemble Predictions
final_probs = 0.5 * test_preds_lr + 0.5 * test_preds_lgb
final_pred_indices = np.argmax(final_probs, axis=1)
final_labels = le_target.inverse_transform(final_pred_indices)

# 7. Save Predictions
submission_df = pd.DataFrame({
    'id': np.arange(len(test_df)),
    'availability': final_labels
})
submission_df.to_csv('predictions.csv', index=False)