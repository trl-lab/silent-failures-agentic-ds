import os
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Ridge
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack

# 1. Load Data
train_path = "./input/train.parquet"
test_path = "./input/test.parquet"

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = "BT Easiness"

# Identify the text column
text_candidates = ["Excerpt", "excerpt", "text", "Text", "content", "passage"]
text_col = None
for col in text_candidates:
    if col in train_df.columns:
        text_col = col
        break

if text_col is None:
    # Find the object/string column with longest average string length
    obj_cols = train_df.select_dtypes(include=["object", "string"]).columns
    if len(obj_cols) > 0:
        text_col = max(obj_cols, key=lambda c: train_df[c].astype(str).str.len().mean())
    else:
        raise ValueError("Could not find text column in dataset.")

train_texts = train_df[text_col].fillna("").astype(str)
test_texts = test_df[text_col].fillna("").astype(str)
y = train_df[target_col].values


# 2. Feature Engineering: Linguistic & Readability Statistics
def count_syllables(word):
    word = word.lower().strip(".:;?!")
    if not word:
        return 0
    if len(word) <= 3:
        return 1
    word = re.sub(r"(?:[^laeiouy]|ed|es|e)$", "", word)
    word = re.sub(r"^y", "", word)
    syllables = len(re.findall(r"[aeiouy]{1,2}", word))
    return max(1, syllables)


def extract_readability_features(texts):
    features = []
    for text in texts:
        char_count = len(text)
        words = re.findall(r"\b\w+\b", text)
        word_count = len(words)
        sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
        sentence_count = max(1, len(sentences))

        if word_count == 0:
            features.append([0.0] * 18)
            continue

        word_lens = [len(w) for w in words]
        avg_word_len = np.mean(word_lens)
        max_word_len = np.max(word_lens)
        avg_sent_len = word_count / sentence_count

        syllable_counts = [count_syllables(w) for w in words]
        total_syllables = sum(syllable_counts)
        avg_syllables_per_word = total_syllables / word_count
        complex_words = sum(1 for s in syllable_counts if s >= 3)
        pct_complex_words = complex_words / word_count

        # Readability metrics
        flesch_reading_ease = (
            206.835 - (1.015 * avg_sent_len) - (84.6 * avg_syllables_per_word)
        )
        flesch_kincaid_grade = (
            (0.39 * avg_sent_len) + (11.8 * avg_syllables_per_word) - 15.59
        )
        gunning_fog = 0.4 * (avg_sent_len + 100 * pct_complex_words)
        coleman_liau = (
            0.0588 * (char_count / word_count * 100)
            - 0.296 * (sentence_count / word_count * 100)
            - 15.8
        )
        ari = (
            4.71 * (char_count / word_count)
            + 0.5 * (word_count / sentence_count)
            - 21.43
        )

        # Lexical diversity
        ttr = len(set(w.lower() for w in words)) / word_count
        num_digits = sum(c.isdigit() for c in text) / max(1, char_count)
        num_punct = sum(c in ".,;:!?\"'-()" for c in text) / max(1, char_count)
        num_caps = sum(c.isupper() for c in text) / max(1, char_count)

        features.append(
            [
                char_count,
                word_count,
                sentence_count,
                avg_word_len,
                max_word_len,
                avg_sent_len,
                avg_syllables_per_word,
                pct_complex_words,
                flesch_reading_ease,
                flesch_kincaid_grade,
                gunning_fog,
                coleman_liau,
                ari,
                ttr,
                num_digits,
                num_punct,
                num_caps,
                total_syllables,
            ]
        )
    return np.array(features, dtype=np.float32)


X_num_train = extract_readability_features(train_texts)
X_num_test = extract_readability_features(test_texts)

# 3. Text Representations (Word & Char TF-IDF + SVD)
word_vec = TfidfVectorizer(ngram_range=(1, 2), max_features=15000, sublinear_tf=True)
char_vec = TfidfVectorizer(
    ngram_range=(2, 5), analyzer="char", max_features=25000, sublinear_tf=True
)

X_word_train = word_vec.fit_transform(train_texts)
X_word_test = word_vec.transform(test_texts)

X_char_train = char_vec.fit_transform(train_texts)
X_char_test = char_vec.transform(test_texts)

svd_word = TruncatedSVD(n_components=120, random_state=42)
svd_char = TruncatedSVD(n_components=120, random_state=42)

X_word_svd_train = svd_word.fit_transform(X_word_train)
X_word_svd_test = svd_word.transform(X_word_test)

X_char_svd_train = svd_char.fit_transform(X_char_train)
X_char_svd_test = svd_char.transform(X_char_test)

# Combine dense and sparse representations
scaler = StandardScaler()
X_dense_train = scaler.fit_transform(
    np.hstack(
        [X_num_train, X_word_svd_train, X_char_svd_test_train := X_char_svd_train]
    )
)
X_dense_test = scaler.transform(
    np.hstack([X_num_test, X_word_svd_test, X_char_svd_test])
)

X_full_train = hstack([X_word_train, X_char_train, X_dense_train]).tocsr()
X_full_test = hstack([X_word_test, X_char_test, X_dense_test]).tocsr()

# 4. 5-Fold Cross-Validation Modeling
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(train_df))
test_preds = np.zeros(len(test_df))

for fold, (train_idx, val_idx) in enumerate(kf.split(X_full_train, y)):
    X_tr, y_tr = X_full_train[train_idx], y[train_idx]
    X_va, y_va = X_full_train[val_idx], y[val_idx]

    model = Ridge(alpha=3.0, random_state=42, solver="auto")
    model.fit(X_tr, y_tr)

    oof_preds[val_idx] = model.predict(X_va)
    test_preds += model.predict(X_full_test) / kf.n_splits

cv_rmse = root_mean_squared_error(y, oof_preds)
print(f"5-Fold CV RMSE: {cv_rmse:.5f}")

# 5. Save Submission
os.makedirs("./working", exist_ok=True)
sub = pd.DataFrame({"id": np.arange(len(test_df)), target_col: test_preds})
sub.to_csv("./working/submission.csv", index=False)
print("Submission saved successfully. Shape:", sub.shape)
