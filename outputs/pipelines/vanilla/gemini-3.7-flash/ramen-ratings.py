import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from lightgbm import LGBMClassifier

# Load data
train_path = '/path/to/user/workspace/ds_failures/datasets/strable/ramen-ratings/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/strable/ramen-ratings/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

# Handle missing values
cat_cols = ['Brand', 'Variety', 'Style', 'Country']
for col in cat_cols:
    if col in train_df.columns:
        train_df[col] = train_df[col].fillna('Missing').astype(str)
        test_df[col] = test_df[col].fillna('Missing').astype(str)

X_train = train_df.drop(columns=['Stars'])
y_train = train_df['Stars'].astype(str)
X_test = test_df.copy()

# Encode target
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)

# Feature extraction and preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('variety_tfidf', TfidfVectorizer(max_features=2000, ngram_range=(1, 2)), 'Variety'),
        ('brand_tfidf', TfidfVectorizer(max_features=500, analyzer='char_wb', ngram_range=(2, 4)), 'Brand'),
        ('cats', OneHotEncoder(handle_unknown='ignore'), ['Style', 'Country'])
    ]
)

# Define models and ensemble
lr = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
lgbm = LGBMClassifier(n_estimators=150, learning_rate=0.05, random_state=42, verbose=-1)

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', VotingClassifier(estimators=[('lr', lr), ('lgbm', lgbm)], voting='soft'))
])

# Fit model
model.fit(X_train, y_train_encoded)

# Predict on test set
y_test_pred_encoded = model.predict(X_test)
y_test_pred = label_encoder.inverse_transform(y_test_pred_encoded)

# Save predictions
output_df = pd.DataFrame({
    'id': range(len(y_test_pred)),
    'Stars': y_test_pred
})
output_df.to_csv('predictions.csv', index=False)