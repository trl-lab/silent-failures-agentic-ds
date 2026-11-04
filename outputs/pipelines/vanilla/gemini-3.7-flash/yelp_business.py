import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix
from lightgbm import LGBMClassifier

train_path = '/path/to/user/workspace/ds_failures/datasets/strable/yelp_business/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/strable/yelp_business/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'stars'
y_raw = train_df[target_col]

le_target = LabelEncoder()
y_train = le_target.fit_transform(y_raw)

n_train = len(train_df)
df = pd.concat([train_df.drop(columns=[target_col], errors='ignore'), test_df], ignore_index=True)

lat = df['latitude'].fillna(df['latitude'].median()).values
lon = df['longitude'].fillna(df['longitude'].median()).values
rc = df['review_count'].fillna(0).values
log_rc = np.log1p(np.maximum(0, rc))
is_open = df['is_open'].fillna(0).astype(int).values
name_len = df['name'].fillna('').astype(str).str.len().values
has_hours = df['hours'].notnull().astype(int).values
has_attr = df['attributes'].notnull().astype(int).values

city_le = LabelEncoder().fit_transform(df['city'].fillna('Missing').astype(str))
state_le = LabelEncoder().fit_transform(df['state'].fillna('Missing').astype(str))
postal_le = LabelEncoder().fit_transform(df['postal_code'].fillna('Missing').astype(str))

dense_feats = np.column_stack([
    lat, lon, rc, log_rc, is_open, name_len, has_hours, has_attr,
    city_le, state_le, postal_le
])

tfidf_cat = TfidfVectorizer(max_features=100, stop_words='english')
cat_feats = tfidf_cat.fit_transform(df['categories'].fillna('').astype(str))

tfidf_attr = TfidfVectorizer(max_features=80, stop_words='english')
attr_feats = tfidf_attr.fit_transform(df['attributes'].fillna('').astype(str))

X_all = hstack([csr_matrix(dense_feats), cat_feats, attr_feats]).tocsr()
X_train = X_all[:n_train]
X_test = X_all[n_train:]

model = LGBMClassifier(
    n_estimators=200,
    learning_rate=0.1,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)
model.fit(X_train, y_train)

y_pred_enc = model.predict(X_test)
y_pred = le_target.inverse_transform(y_pred_enc)

submission = pd.DataFrame({
    'id': np.arange(len(test_df), dtype=int),
    'stars': y_pred
})
submission.to_csv('predictions.csv', index=False)