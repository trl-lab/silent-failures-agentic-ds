import os
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Set seed for reproducibility
np.random.seed(42)

# File paths
train_path = "./input/train.parquet"
test_path = "./input/test.parquet"
output_dir = "./working"
os.makedirs(output_dir, exist_ok=True)

# Load datasets
train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = "classification"
y_raw = train_df[target_col]

# Encode target values
le = LabelEncoder()
y = le.fit_transform(y_raw)

# Identify text feature columns
feature_cols = [c for c in train_df.columns if c != target_col]


# Function to aggregate text across columns
def aggregate_text(df, cols):
    return df[cols].fillna("").astype(str).agg(" ".join, axis=1)


full_train_text = aggregate_text(train_df, feature_cols)
full_test_text = aggregate_text(test_df, feature_cols)

# Vectorizer 1: Global Word TF-IDF (unigrams to trigrams)
word_vec = TfidfVectorizer(
    ngram_range=(1, 3),
    max_features=50000,
    sublinear_tf=True,
    min_df=2,
    strip_accents="unicode",
)

# Vectorizer 2: Global Character-boundary TF-IDF
char_vec = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(3, 5),
    max_features=40000,
    sublinear_tf=True,
    min_df=2,
)

X_train_word = word_vec.fit_transform(full_train_text)
X_test_word = word_vec.transform(full_test_text)

X_train_char = char_vec.fit_transform(full_train_text)
X_test_char = char_vec.transform(full_test_text)

sparse_train_list = [X_train_word, X_train_char]
sparse_test_list = [X_test_word, X_test_char]

# Vectorize individual key domain fields if present
key_fields = [
    "reason_for_recall",
    "product_description",
    "distribution_pattern",
    "code_info",
]
for col in key_fields:
    if col in train_df.columns:
        col_train = train_df[col].fillna("").astype(str)
        col_test = test_df[col].fillna("").astype(str)
        col_vec = TfidfVectorizer(
            ngram_range=(1, 2), max_features=20000, sublinear_tf=True, min_df=2
        )
        X_tr_col = col_vec.fit_transform(col_train)
        X_te_col = col_vec.transform(col_test)
        sparse_train_list.append(X_tr_col)
        sparse_test_list.append(X_te_col)

# Stack all sparse feature representations
X_train = hstack(sparse_train_list).tocsr()
X_test = hstack(sparse_test_list).tocsr()

# 5-Fold Stratified Cross Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

n_classes = len(le.classes_)
oof_probs = np.zeros((len(train_df), n_classes))
test_probs = np.zeros((len(test_df), n_classes))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y)):
    X_tr, y_tr = X_train[train_idx], y[train_idx]
    X_val, y_val = X_train[val_idx], y[val_idx]

    model = LogisticRegression(
        C=2.0, max_iter=1000, solver="lbfgs", random_state=42 + fold
    )

    model.fit(X_tr, y_tr)

    oof_probs[val_idx] = model.predict_proba(X_val)
    test_probs += model.predict_proba(X_test) / skf.n_splits

# Compute Out-of-Fold Accuracy
oof_preds = np.argmax(oof_probs, axis=1)
acc = accuracy_score(y, oof_preds)
print(f"Validation Accuracy: {acc:.5f}")

# Map test set predictions to original labels
test_preds = np.argmax(test_probs, axis=1)
test_pred_labels = le.inverse_transform(test_preds)

# Save submission file
submission = pd.DataFrame(
    {"id": np.arange(len(test_df)), "classification": test_pred_labels}
)

sub_path = os.path.join(output_dir, "submission.csv")
submission.to_csv(sub_path, index=False)
print(f"Submission saved to {sub_path} with shape {submission.shape}")
