import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import MultinomialNB

# Load data
train_path = '/path/to/user/workspace/ds_failures/datasets/strable/drug-ndc/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/strable/drug-ndc/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'dea_schedule'

# Encode target
le = LabelEncoder()
y_train = le.fit_transform(train_df[target_col].astype(str))
X_train_raw = train_df.drop(columns=[target_col])
X_test_raw = test_df.copy()

# Identify useful text / categorical columns
feature_cols = [
    'generic_name', 'active_ingredients', 'pharm_class', 'brand_name',
    'brand_name_base', 'dosage_form', 'route', 'marketing_category',
    'labeler_name', 'product_type', 'openfda', 'packaging'
]
feature_cols = [col for col in feature_cols if col in X_train_raw.columns]

def create_text_features(df, cols):
    text_series = []
    for col in cols:
        val = df[col].fillna('').astype(str)
        text_series.append(col + ': ' + val)
    return pd.Series([' '.join(row) for row in zip(*text_series)], index=df.index)

X_train_text = create_text_features(X_train_raw, feature_cols)
X_test_text = create_text_features(X_test_raw, feature_cols)

# Feature extraction: Word and Character TF-IDF
vectorizer = FeatureUnion([
    ('word_tfidf', TfidfVectorizer(ngram_range=(1, 3), min_df=2, max_features=30000, sublinear_tf=True)),
    ('char_tfidf', TfidfVectorizer(ngram_range=(3, 5), analyzer='char', min_df=3, max_features=30000, sublinear_tf=True))
])

X_train_vec = vectorizer.fit_transform(X_train_text)
X_test_vec = vectorizer.transform(X_test_text)

# Build an ensemble of strong text classifiers
clf1 = LogisticRegression(C=5.0, max_iter=1000, random_state=42)
clf2 = SGDClassifier(loss='log_loss', alpha=1e-5, max_iter=1000, random_state=42)
clf3 = MultinomialNB(alpha=0.1)

ensemble = VotingClassifier(
    estimators=[
        ('lr', clf1),
        ('sgd', clf2),
        ('nb', clf3)
    ],
    voting='soft'
)

ensemble.fit(X_train_vec, y_train)
y_pred = ensemble.predict(X_test_vec)

# Map back to original labels
y_pred_labels = le.inverse_transform(y_pred)

# Save predictions
output_df = pd.DataFrame({
    'id': np.arange(len(y_pred_labels)),
    'dea_schedule': y_pred_labels
})
output_df.to_csv('predictions.csv', index=False)