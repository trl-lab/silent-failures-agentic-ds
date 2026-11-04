import os
import re
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.sparse import csr_matrix, hstack
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

# 1. Load Data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "BT Easiness"
y = train_df[target_col].values

# 2. Identify Primary Text Column & Metadata Numerical Columns
object_cols = [
    col for col in train_df.columns if train_df[col].dtype in ["object", "string"]
]
if object_cols:
    text_col = max(object_cols, key=lambda c: train_df[c].astype(str).str.len().mean())
else:
    text_col = "text"

num_cols = [
    col
    for col in train_df.columns
    if col not in object_cols and col != target_col and not col.lower().startswith("id")
]


def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


train_texts = np.array([clean_text(t) for t in train_df[text_col]])
test_texts = np.array([clean_text(t) for t in test_df[text_col]])


# 3. Enhanced Readability & Structural Feature Extraction
def count_syllables(word):
    word = word.lower()
    if len(word) <= 3:
        return 1
    word = re.sub(r"(?:[^laeiouy]|ed|es|e)$", "", word)
    word = re.sub(r"^y", "", word)
    syllables = len(re.findall(r"[aeiouy]{1,2}", word))
    return max(1, syllables)


def extract_text_features(texts):
    features = []
    for t in texts:
        char_count = len(t)
        words = t.split()
        word_count = max(len(words), 1)
        sentences = [s for s in re.split(r"[.!?]+", t) if s.strip()]
        sentence_count = max(len(sentences), 1)

        syllable_count = sum(count_syllables(w) for w in words)
        complex_words = sum(1 for w in words if count_syllables(w) >= 3)
        long_words = sum(1 for w in words if len(w) > 6)

        avg_word_len = sum(len(w) for w in words) / word_count
        avg_sent_len = word_count / sentence_count
        unique_word_ratio = len(set(w.lower() for w in words)) / word_count
        upper_word_ratio = sum(1 for w in words if w.isupper()) / word_count
        long_word_ratio = long_words / word_count

        # Classical & Extended Readability Metrics
        flesch_reading_ease = (
            206.835 - (1.015 * avg_sent_len) - (84.6 * (syllable_count / word_count))
        )
        coleman_liau = (
            0.0588 * (char_count / word_count * 100)
            - 0.296 * (sentence_count / word_count * 100)
            - 15.8
        )
        ari = 4.71 * (char_count / word_count) + 0.5 * avg_sent_len - 21.43
        gunning_fog = 0.4 * (avg_sent_len + 100.0 * (complex_words / word_count))
        smog = 1.0430 * np.sqrt(complex_words * (30.0 / sentence_count)) + 3.1291
        lix = avg_sent_len + (long_word_ratio * 100.0)

        # Variance/Statistical Descriptors
        sent_lengths = [len(s.split()) for s in sentences]
        sent_len_std = float(np.std(sent_lengths)) if sent_lengths else 0.0
        word_lengths = [len(w) for w in words]
        word_len_std = float(np.std(word_lengths)) if word_lengths else 0.0

        # Punctuation and Structural Metrics
        comma_count = t.count(",")
        semicolon_count = t.count(";")
        colon_count = t.count(":")
        hyphen_count = t.count("-")
        quote_count = t.count('"') + t.count("'")
        excl_count = t.count("!")
        quest_count = t.count("?")
        digit_ratio = sum(1 for c in t if c.isdigit()) / max(char_count, 1)

        features.append(
            [
                char_count,
                word_count,
                sentence_count,
                syllable_count,
                complex_words,
                avg_word_len,
                avg_sent_len,
                unique_word_ratio,
                upper_word_ratio,
                long_word_ratio,
                flesch_reading_ease,
                coleman_liau,
                ari,
                gunning_fog,
                smog,
                lix,
                sent_len_std,
                word_len_std,
                comma_count,
                semicolon_count,
                colon_count,
                hyphen_count,
                quote_count,
                excl_count,
                quest_count,
                digit_ratio,
            ]
        )
    return np.array(features)


X_num_train_text = extract_text_features(train_texts)
X_num_test_text = extract_text_features(test_texts)

if num_cols:
    train_meta = train_df[num_cols].fillna(train_df[num_cols].median()).values
    test_meta = test_df[num_cols].fillna(train_df[num_cols].median()).values
    X_num_train = np.hstack([X_num_train_text, train_meta])
    X_num_test = np.hstack([X_num_test_text, test_meta])
else:
    X_num_train = X_num_train_text
    X_num_test = X_num_test_text

# 4. Cross-Validation Setup
kf = KFold(n_splits=5, shuffle=True, random_state=42)

oof_preds_ridge = np.zeros(len(train_df))
test_preds_ridge = np.zeros(len(test_df))

oof_preds_lgb = np.zeros(len(train_df))
test_preds_lgb = np.zeros(len(test_df))

alphas = np.logspace(-1, 3, 25)

for fold, (train_idx, val_idx) in enumerate(kf.split(train_df)):
    tr_texts, va_texts = train_texts[train_idx], train_texts[val_idx]

    scaler = StandardScaler()
    X_num_tr_scaled = scaler.fit_transform(X_num_train[train_idx])
    X_num_va_scaled = scaler.transform(X_num_train[val_idx])
    X_num_te_scaled = scaler.transform(X_num_test)

    tfidf_word = TfidfVectorizer(
        ngram_range=(1, 3), max_features=30000, min_df=2, sublinear_tf=True
    )
    tfidf_char = TfidfVectorizer(
        ngram_range=(2, 6),
        analyzer="char_wb",
        max_features=30000,
        min_df=2,
        sublinear_tf=True,
    )

    X_tr_word = tfidf_word.fit_transform(tr_texts)
    X_va_word = tfidf_word.transform(va_texts)
    X_te_word = tfidf_word.transform(test_texts)

    X_tr_char = tfidf_char.fit_transform(tr_texts)
    X_va_char = tfidf_char.transform(va_texts)
    X_te_char = tfidf_char.transform(test_texts)

    X_tr_tfidf = hstack([X_tr_word, X_tr_char]).tocsr()
    X_va_tfidf = hstack([X_va_word, X_va_char]).tocsr()
    X_te_tfidf = hstack([X_te_word, X_te_char]).tocsr()

    X_tr_ridge = hstack([csr_matrix(X_num_tr_scaled), X_tr_tfidf]).tocsr()
    X_va_ridge = hstack([csr_matrix(X_num_va_scaled), X_va_tfidf]).tocsr()
    X_te_ridge = hstack([csr_matrix(X_num_te_scaled), X_te_tfidf]).tocsr()

    y_tr, y_va = y[train_idx], y[val_idx]

    # RidgeCV Model
    ridge = RidgeCV(alphas=alphas)
    ridge.fit(X_tr_ridge, y_tr)
    oof_preds_ridge[val_idx] = ridge.predict(X_va_ridge)
    test_preds_ridge += ridge.predict(X_te_ridge) / kf.n_splits

    # LightGBM Model on SVD components + text metadata
    svd = TruncatedSVD(n_components=64, random_state=42 + fold)
    X_tr_svd = svd.fit_transform(X_tr_tfidf)
    X_va_svd = svd.transform(X_va_tfidf)
    X_te_svd = svd.transform(X_te_tfidf)

    X_tr_lgb = np.hstack([X_num_tr_scaled, X_tr_svd])
    X_va_lgb = np.hstack([X_num_va_scaled, X_va_svd])
    X_te_lgb = np.hstack([X_num_te_scaled, X_te_svd])

    lgb_train = lgb.Dataset(X_tr_lgb, label=y_tr)
    lgb_val = lgb.Dataset(X_va_lgb, label=y_va, reference=lgb_train)

    params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.03,
        "num_leaves": 31,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "seed": 42 + fold,
        "verbose": -1,
    }

    gbm = lgb.train(
        params,
        lgb_train,
        num_boost_round=1000,
        valid_sets=[lgb_val],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
    )

    oof_preds_lgb[val_idx] = gbm.predict(X_va_lgb)
    test_preds_lgb += gbm.predict(X_te_lgb) / kf.n_splits


# 5. Optimize Ensemble Weights via Out-of-Fold Cross-Validation
def rmse_obj(weights):
    w1, w2 = weights
    pred = w1 * oof_preds_ridge + w2 * oof_preds_lgb
    return root_mean_squared_error(y, pred)


res = minimize(
    rmse_obj,
    [0.80, 0.20],
    method="SLSQP",
    bounds=[(0, 1), (0, 1)],
    constraints={"type": "eq", "fun": lambda w: 1.0 - sum(w)},
)

opt_w1, opt_w2 = res.x
oof_preds_ensemble = opt_w1 * oof_preds_ridge + opt_w2 * oof_preds_lgb
test_preds_ensemble = opt_w1 * test_preds_ridge + opt_w2 * test_preds_lgb

cv_rmse = root_mean_squared_error(y, oof_preds_ensemble)
print(f"Validation RMSE: {cv_rmse:.5f}")

# 6. Save Predictions to Submission File
os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame(
    {"id": np.arange(len(test_df)), "BT Easiness": test_preds_ensemble}
)
submission.to_csv("./working/submission.csv", index=False)
