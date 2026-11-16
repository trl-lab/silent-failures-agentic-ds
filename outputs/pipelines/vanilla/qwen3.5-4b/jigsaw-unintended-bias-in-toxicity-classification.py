import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from lightgbm import LGBMClassifier
import os

# Configuration
TRAIN_PATH = '/path/to/user/workspace/ds_failures/datasets/mlebench/jigsaw-unintended-bias-in-toxicity-classification/data/train.parquet'
TEST_PATH = '/path/to/user/workspace/ds_failures/datasets/mlebench/jigsaw-unintended-bias-in-toxicity-classification/data/test.parquet'
OUTPUT_PATH = 'predictions.csv'
TARGET_COLUMN = 'target'

# Load Data
print(f"Loading train data from {TRAIN_PATH}...")
train_df = pd.read_parquet(TRAIN_PATH)
print(f"Loading test data from {TEST_PATH}...")
test_df = pd.read_parquet(TEST_PATH)

# Determine Text Column
col_candidates = [c for c in train_df.columns if isinstance(c, str) and ('text' in c.lower())]
if not col_candidates:
    col_candidates = ['comment_text'] # Fallback for standard dataset schema

selected_text_col = col_candidates[0] if col_candidates else 'comment_text'
target_col = TARGET_COLUMN

# Extract Features and Labels
X_text_train = train_df[selected_text_col].copy()
y_train = train_df[target_col].values

X_text_test = test_df[selected_text_col].copy()

# Handle Missing Values in Text
X_text_train = X_text_train.fillna('')
X_text_test = X_text_test.fillna('')

# Vectorize
print("Vectorizing text features...")
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=10000, stop_words='english')
X_train_vec = vectorizer.fit_transform(X_text_train)
X_test_vec = vectorizer.transform(X_text_test)

# Train Model
print("Training LightGBM model...")
model = LGBMClassifier(
    objective='binary',
    num_leaves=31,
    learning_rate=0.1,
    n_estimators=500,
    verbose=-1,
    random_state=42
)
model.fit(X_train_vec, y_train, eval_set=None)

# Predictions
print("Generating predictions...")
predictions = model.predict_proba(X_test_vec)[:, 1]

# Prepare Output DataFrame
output_ids = list(range(len(predictions)))
results_df = pd.DataFrame({
    'id': output_ids,
    'target': predictions
})

# Save to CSV
results_df.to_csv(OUTPUT_PATH, index=False)
print(f"Predictions saved to {OUTPUT_PATH}")