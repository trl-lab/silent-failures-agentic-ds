import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import hstack, csr_matrix

# Load datasets
train_path = '/path/to/user/workspace/ds_failures/datasets/strable/paediatric-investigation-plan/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/strable/paediatric-investigation-plan/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'Decision type'
feature_cols = [c for c in train_df.columns if c != target_col]

# Encode target
le = LabelEncoder()
y_train = le.fit_transform(train_df[target_col].astype(str))

# Feature engineering
def build_features(df):
    text_data = []
    num_features = []
    
    date_cols = ['Decision date', 'Compliance opinion date', 'First published date', 'Last updated date']
    
    for idx, row in df.iterrows():
        # Text concatenation with field markers
        row_text_parts = []
        for col in feature_cols:
            val = row[col]
            if pd.notna(val) and str(val).strip() != '' and str(val).strip().lower() != 'none':
                row_text_parts.append(f"{col}: {val}")
        text_data.append(" | ".join(row_text_parts))
    
    # Numeric / structural features
    num_df = pd.DataFrame(index=df.index)
    for col in feature_cols:
        num_df[f"{col}_isna"] = df[col].isna().astype(float)
        num_df[f"{col}_len"] = df[col].fillna('').astype(str).str.len()
    
    # Domain specific flags
    pip_str = df['PIP number'].fillna('').astype(str)
    num_df['has_m_mod'] = pip_str.str.contains(r'-M\d+', regex=True).astype(float)
    num_df['pip_is_pip'] = pip_str.str.contains('PIP', case=False).astype(float)
    
    dec_num_str = df['Decision number'].fillna('').astype(str)
    num_df['dec_starts_p'] = dec_num_str.str.startswith('P/').astype(float)
    num_df['dec_starts_w'] = dec_num_str.str.startswith('W/').astype(float)
    
    # Date parsing
    for col in date_cols:
        if col in df.columns:
            d = pd.to_datetime(df[col], dayfirst=True, errors='coerce')
            num_df[f"{col}_year"] = d.dt.year.fillna(-1)
            num_df[f"{col}_month"] = d.dt.month.fillna(-1)
            num_df[f"{col}_day"] = d.dt.day.fillna(-1)
            num_df[f"{col}_is_not_null"] = d.notna().astype(float)
            
    return text_data, num_df.fillna(0).values

train_text, train_num = build_features(train_df)
test_text, test_num = build_features(test_df)

# Text vectorizers
word_vec = TfidfVectorizer(ngram_range=(1, 3), max_features=40000, sublinear_tf=True, token_pattern=r'(?u)\b\w+\b')
char_vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5), max_features=40000, sublinear_tf=True)

X_train_word = word_vec.fit_transform(train_text)
X_test_word = word_vec.transform(test_text)

X_train_char = char_vec.fit_transform(train_text)
X_test_char = char_vec.transform(test_text)

X_train = hstack([X_train_word, X_train_char, csr_matrix(train_num)]).tocsr()
X_test = hstack([X_test_word, X_test_char, csr_matrix(test_num)]).tocsr()

# Modeling with K-Fold cross validation and ensembling
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
test_preds_proba = np.zeros((X_test.shape[0], len(le.classes_)))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    X_tr, y_tr = X_train[train_idx], y_train[train_idx]
    
    clf1 = LogisticRegression(C=5.0, max_iter=1000, random_state=42 + fold)
    clf2 = SGDClassifier(loss='log_loss', alpha=1e-5, max_iter=1000, random_state=42 + fold)
    
    clf1.fit(X_tr, y_tr)
    clf2.fit(X_tr, y_tr)
    
    pred1 = clf1.predict_proba(X_test)
    pred2 = clf2.predict_proba(X_test)
    
    test_preds_proba += (0.6 * pred1 + 0.4 * pred2) / skf.n_splits

# Final predicted class labels
final_preds = np.argmax(test_preds_proba, axis=1)
predicted_labels = le.inverse_transform(final_preds)

# Save predictions
output_df = pd.DataFrame({
    'id': np.arange(len(predicted_labels)),
    'Decision type': predicted_labels
})
output_df.to_csv('predictions.csv', index=False)