import pandas as pd
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import Ridge
import gc

# 1. Load data
train_df = pd.read_parquet('/path/to/user/workspace/ds_failures/datasets/strable/mercari/data/train.parquet')
test_df = pd.read_parquet('/path/to/user/workspace/ds_failures/datasets/strable/mercari/data/test.parquet')

y_train = train_df['price'].values

# 2. Preprocess text and categorical missing values
for df in [train_df, test_df]:
    df['name'] = df['name'].fillna('').astype(str)
    df['item_description'] = df['item_description'].fillna('').astype(str)
    df['brand_name'] = df['brand_name'].fillna('missing').astype(str)
    df['category_name'] = df['category_name'].fillna('missing').astype(str)
    df['item_condition_id'] = df['item_condition_id'].fillna(-1).astype(str)
    df['shipping'] = df['shipping'].fillna(-1).astype(str)

# 3. Feature extraction
# Text features for item name
vec_name = TfidfVectorizer(max_features=100000, ngram_range=(1, 2), token_pattern=r'(?u)\b\w+\b')
X_train_name = vec_name.fit_transform(train_df['name'])
X_test_name = vec_name.transform(test_df['name'])

# Text features for item description
vec_desc = TfidfVectorizer(max_features=150000, ngram_range=(1, 2), min_df=3, token_pattern=r'(?u)\b\w+\b')
X_train_desc = vec_desc.fit_transform(train_df['item_description'])
X_test_desc = vec_desc.transform(test_df['item_description'])

# Category features
vec_cat = CountVectorizer(token_pattern=r'(?u)[^/]+')
X_train_cat = vec_cat.fit_transform(train_df['category_name'])
X_test_cat = vec_cat.transform(test_df['category_name'])

# Brand features
vec_brand = CountVectorizer(min_df=2, token_pattern=r'(?u)\b[\w-]+\b')
X_train_brand = vec_brand.fit_transform(train_df['brand_name'])
X_test_brand = vec_brand.transform(test_df['brand_name'])

# Categorical features: item_condition_id and shipping
ohe = OneHotEncoder(handle_unknown='ignore')
X_train_cat_meta = ohe.fit_transform(train_df[['item_condition_id', 'shipping']])
X_test_cat_meta = ohe.transform(test_df[['item_condition_id', 'shipping']])

# Stack features
X_train = hstack([X_train_name, X_train_desc, X_train_cat, X_train_brand, X_train_cat_meta]).tocsr()
X_test = hstack([X_test_name, X_test_desc, X_test_cat, X_test_brand, X_test_cat_meta]).tocsr()

del X_train_name, X_train_desc, X_train_cat, X_train_brand, X_train_cat_meta
del X_test_name, X_test_desc, X_test_cat, X_test_brand, X_test_cat_meta
del train_df
gc.collect()

# 4. Train Ridge regression model
model = Ridge(solver='auto', alpha=4.0, random_state=42)
model.fit(X_train, y_train)

# 5. Predict and postprocess
preds = model.predict(X_test)
preds = np.clip(preds, a_min=0, a_max=None)

# 6. Save predictions
submission_df = pd.DataFrame({
    'id': np.arange(len(preds)),
    'price': preds
})
submission_df.to_csv('predictions.csv', index=False)