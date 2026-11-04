import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from lightgbm import LGBMClassifier
from scipy.sparse import hstack, csr_matrix

# Load data
train_path = '/path/to/user/workspace/ds_failures/datasets/strable/food-event/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/strable/food-event/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

# Target encoding
target_col = 'outcomes'
le = LabelEncoder()
y_train = le.fit_transform(train_df[target_col].astype(str))

# Feature engineering function
def preprocess_features(df):
    df = df.copy()
    
    # Text features
    df['reactions'] = df['reactions'].fillna('').astype(str)
    df['products'] = df['products'].fillna('').astype(str)
    
    # Age cleaning & normalization
    age_num = pd.to_numeric(df['age'], errors='coerce')
    age_unit = df['age_unit'].astype(str).str.lower()
    
    age_in_years = age_num.copy()
    age_in_years[age_unit.str.contains('month', na=False)] = age_num / 12.0
    age_in_years[age_unit.str.contains('week', na=False)] = age_num / 52.0
    age_in_years[age_unit.str.contains('day', na=False)] = age_num / 365.0
    age_in_years[age_unit.str.contains('decade', na=False)] = age_num * 10.0
    
    df['age_years'] = age_in_years.fillna(-1.0)
    df['age_is_missing'] = age_num.isna().astype(float)
    
    # Gender
    df['gender'] = df['gender'].fillna('Unknown').astype(str)
    
    # Dates
    df['date_created_num'] = pd.to_numeric(df['date_created'], errors='coerce').fillna(0)
    df['date_started_num'] = pd.to_numeric(df['date_started'], errors='coerce').fillna(0)
    
    return df

train_processed = preprocess_features(train_df)
test_processed = preprocess_features(test_df)

# Text TF-IDF vectorizers
tfidf_reactions = TfidfVectorizer(max_features=15000, ngram_range=(1, 2), sublinear_tf=True)
tfidf_products = TfidfVectorizer(max_features=15000, ngram_range=(1, 2), sublinear_tf=True)

X_train_rx = tfidf_reactions.fit_transform(train_processed['reactions'])
X_test_rx = tfidf_reactions.transform(test_processed['reactions'])

X_train_prod = tfidf_products.fit_transform(train_processed['products'])
X_test_prod = tfidf_products.transform(test_processed['products'])

# Categorical & Numerical features
cat_cols = ['gender']
ohe = OneHotEncoder(handle_unknown='ignore')
X_train_cat = ohe.fit_transform(train_processed[cat_cols])
X_test_cat = ohe.transform(test_processed[cat_cols])

num_cols = ['age_years', 'age_is_missing', 'date_created_num', 'date_started_num']
scaler = StandardScaler()
X_train_num = csr_matrix(scaler.fit_transform(train_processed[num_cols]))
X_test_num = csr_matrix(scaler.transform(test_processed[num_cols]))

# Stack all features into sparse matrix
X_train_all = hstack([X_train_rx, X_train_prod, X_train_cat, X_train_num]).tocsr()
X_test_all = hstack([X_test_rx, X_test_prod, X_test_cat, X_test_num]).tocsr()

# Train Model
model = LogisticRegression(C=1.5, max_iter=1000, solver='saga', random_state=42, n_jobs=-1)
model.fit(X_train_all, y_train)

# Predict
preds_encoded = model.predict(X_test_all)
preds_labels = le.inverse_transform(preds_encoded)

# Save predictions
output_df = pd.DataFrame({
    'id': np.arange(len(preds_labels)),
    'outcomes': preds_labels
})
output_df.to_csv('predictions.csv', index=False)