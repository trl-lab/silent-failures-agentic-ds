import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, VotingClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
import scipy.sparse as sp

# Load datasets
train_path = '/path/to/user/workspace/ds_failures/datasets/strable/drug-enforcement/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/strable/drug-enforcement/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'classification'
y_train_raw = train_df[target_col].astype(str)

le = LabelEncoder()
y_train = le.fit_transform(y_train_raw)

# Drop target from train
X_train = train_df.drop(columns=[target_col])
X_test = test_df.copy()

# Identify key text columns and categorical columns
text_cols = ['reason_for_recall', 'product_description', 'distribution_pattern', 'code_info']
cat_cols = ['status', 'state', 'country', 'product_type', 'voluntary_mandated', 
            'initial_firm_notification', 'recalling_firm']

# Fill NA in text columns and combine them
def extract_text(df, col):
    return df[col].fillna('').astype(str)

# Build TF-IDF representations
tfidf_reason = TfidfVectorizer(max_features=25000, ngram_range=(1, 2), sublinear_tf=True)
X_train_reason = tfidf_reason.fit_transform(extract_text(X_train, 'reason_for_recall'))
X_test_reason = tfidf_reason.transform(extract_text(X_test, 'reason_for_recall'))

tfidf_prod = TfidfVectorizer(max_features=25000, ngram_range=(1, 2), sublinear_tf=True)
X_train_prod = tfidf_prod.fit_transform(extract_text(X_train, 'product_description'))
X_test_prod = tfidf_prod.transform(extract_text(X_test, 'product_description'))

# Combined other text fields
def combine_other_text(df):
    return (extract_text(df, 'distribution_pattern') + ' ' +
            extract_text(df, 'code_info') + ' ' +
            extract_text(df, 'recalling_firm') + ' ' +
            extract_text(df, 'brand_name') + ' ' +
            extract_text(df, 'generic_name') + ' ' +
            extract_text(df, 'substance_name'))

tfidf_other = TfidfVectorizer(max_features=15000, ngram_range=(1, 2), sublinear_tf=True)
X_train_other = tfidf_other.fit_transform(combine_other_text(X_train))
X_test_other = tfidf_other.transform(combine_other_text(X_test))

# Categorical features encoding
ohe = OneHotEncoder(handle_unknown='ignore', max_categories=50)
X_train_cat = ohe.fit_transform(X_train[cat_cols].fillna('Missing').astype(str))
X_test_cat = ohe.transform(X_test[cat_cols].fillna('Missing').astype(str))

# Combine all features into sparse matrices
X_train_all = sp.hstack([X_train_reason, X_train_prod, X_train_other, X_train_cat]).tocsr()
X_test_all = sp.hstack([X_test_reason, X_test_prod, X_test_other, X_test_cat]).tocsr()

# Train Logistic Regression model
clf = LogisticRegression(C=2.0, max_iter=1000, random_state=42, n_jobs=-1)
clf.fit(X_train_all, y_train)

# Predict
test_preds = clf.predict(X_test_all)
predicted_classes = le.inverse_transform(test_preds)

# Save predictions
output = pd.DataFrame({
    'id': np.arange(len(test_df)),
    'classification': predicted_classes
})

output.to_csv('predictions.csv', index=False)