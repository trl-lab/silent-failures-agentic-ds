import pandas as pd
import numpy as np
from scipy.sparse import hstack
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC
from sklearn.ensemble import VotingClassifier

# 1. Load data
train_path = '/path/to/user/workspace/ds_failures/datasets/strable/device-classification/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/strable/device-classification/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'device_class'
y_train_raw = train_df[target_col].astype(str)

# Encode target
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train_raw)

# Separate features
categorical_cols = [
    'third_party_flag',
    'life_sustain_support_flag',
    'gmp_exempt_flag',
    'summary_malfunction_reporting',
    'product_code',
    'review_panel',
    'implant_flag'
]

# Fill missing values
for col in categorical_cols:
    train_df[col] = train_df[col].fillna('missing').astype(str)
    test_df[col] = test_df[col].fillna('missing').astype(str)

train_df['device_name'] = train_df['device_name'].fillna('').astype(str)
test_df['device_name'] = test_df['device_name'].fillna('').astype(str)

train_df['definition'] = train_df['definition'].fillna('').astype(str)
test_df['definition'] = test_df['definition'].fillna('').astype(str)

# One-hot encode categorical features
ohe = OneHotEncoder(handle_unknown='ignore')
X_train_cat = ohe.fit_transform(train_df[categorical_cols])
X_test_cat = ohe.transform(test_df[categorical_cols])

# TF-IDF on device_name
tfidf_name = TfidfVectorizer(ngram_range=(1, 2), max_features=25000, sublinear_tf=True)
X_train_name = tfidf_name.fit_transform(train_df['device_name'])
X_test_name = tfidf_name.transform(test_df['device_name'])

# TF-IDF on definition
tfidf_def = TfidfVectorizer(ngram_range=(1, 2), max_features=25000, sublinear_tf=True)
X_train_def = tfidf_def.fit_transform(train_df['definition'])
X_test_def = tfidf_def.transform(test_df['definition'])

# Combine all features
X_train = hstack([X_train_cat, X_train_name, X_train_def]).tocsr()
X_test = hstack([X_test_cat, X_test_name, X_test_def]).tocsr()

# Train models and ensemble
lr = LogisticRegression(C=2.0, max_iter=1000, random_state=42)
svc = CalibratedClassifierCV(LinearSVC(C=1.0, random_state=42))

ensemble = VotingClassifier(
    estimators=[
        ('lr', lr),
        ('svc', svc)
    ],
    voting='soft'
)

ensemble.fit(X_train, y_train)

# Predict
y_pred_idx = ensemble.predict(X_test)
y_pred = label_encoder.inverse_transform(y_pred_idx)

# Save predictions
submission_df = pd.DataFrame({
    'id': np.arange(len(test_df)),
    'device_class': y_pred
})
submission_df.to_csv('predictions.csv', index=False)