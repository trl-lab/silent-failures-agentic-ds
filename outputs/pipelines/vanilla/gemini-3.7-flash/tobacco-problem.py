import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix
import lightgbm as lgb

# Load data
train_path = '/path/to/user/workspace/ds_failures/datasets/strable/tobacco-problem/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/strable/tobacco-problem/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'number_health_problems'
y_raw = train_df[target_col].astype(str)

le_target = LabelEncoder()
y = le_target.fit_transform(y_raw)

n_train = len(train_df)
combined_df = pd.concat([train_df.drop(columns=[target_col]), test_df], ignore_index=True)

# Date features
dates = pd.to_datetime(combined_df['date_submitted'], errors='coerce')
combined_df['year'] = dates.dt.year.fillna(-1).astype(int)
combined_df['month'] = dates.dt.month.fillna(-1).astype(int)
combined_df['day'] = dates.dt.day.fillna(-1).astype(int)
combined_df['dayofweek'] = dates.dt.dayofweek.fillna(-1).astype(int)

# Numeric & Length features
combined_df['num_tobacco_products'] = combined_df['number_tobacco_products'].fillna(0).astype(float)
combined_df['num_product_problems'] = combined_df['number_product_problems'].fillna(0).astype(float)

text_cols = ['reported_product_problems', 'tobacco_products']
for col in text_cols:
    combined_df[col] = combined_df[col].fillna('missing').astype(str)
    combined_df[f'{col}_len'] = combined_df[col].apply(len)
    combined_df[f'{col}_comma_count'] = combined_df[col].apply(lambda x: x.count(','))
    combined_df[f'{col}_word_count'] = combined_df[col].apply(lambda x: len(x.split()))

# Categorical features
nonuser = pd.get_dummies(combined_df['nonuser_affected'].fillna('Unknown'), prefix='nonuser', drop_first=False)

numeric_cols = [
    'year', 'month', 'day', 'dayofweek',
    'num_tobacco_products', 'num_product_problems',
    'reported_product_problems_len', 'reported_product_problems_comma_count', 'reported_product_problems_word_count',
    'tobacco_products_len', 'tobacco_products_comma_count', 'tobacco_products_word_count'
]
num_features = combined_df[numeric_cols].values
cat_features = nonuser.values

# TF-IDF on text
tfidf_problems = TfidfVectorizer(max_features=250, ngram_range=(1, 2), stop_words='english')
tfidf_problems_mat = tfidf_problems.fit_transform(combined_df['reported_product_problems'])

tfidf_tobacco = TfidfVectorizer(max_features=250, ngram_range=(1, 2), stop_words='english')
tfidf_tobacco_mat = tfidf_tobacco.fit_transform(combined_df['tobacco_products'])

# Combine all features into sparse matrix
X_sparse = hstack([
    csr_matrix(num_features),
    csr_matrix(cat_features),
    tfidf_problems_mat,
    tfidf_tobacco_mat
]).tocsr().astype(np.float32)

X_train = X_sparse[:n_train]
X_test = X_sparse[n_train:]

# K-Fold LightGBM
num_classes = len(le_target.classes_)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
test_preds = np.zeros((X_test.shape[0], num_classes))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y)):
    X_tr, y_tr = X_train[train_idx], y[train_idx]
    X_va, y_va = X_train[val_idx], y[val_idx]
    
    clf = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        n_jobs=-1,
        verbose=-1
    )
    
    clf.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    
    test_preds += clf.predict_proba(X_test) / skf.n_splits

final_classes_idx = np.argmax(test_preds, axis=1)
final_predictions = le_target.inverse_transform(final_classes_idx)

# Save predictions
output_df = pd.DataFrame({
    'id': range(len(final_predictions)),
    'number_health_problems': final_predictions
})
output_df.to_csv('predictions.csv', index=False)