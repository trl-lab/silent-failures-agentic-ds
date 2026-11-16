import os
import gc
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

os.makedirs("./working", exist_ok=True)

train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

train["comment_text"] = train["comment_text"].fillna("").astype(str)
test["comment_text"] = test["comment_text"].fillna("").astype(str)
y = train["target"].astype(int).to_numpy()

SVS_COMPONENTS = 60
MAX_FEATURES = 20_000


def build_text_features(df: pd.DataFrame) -> np.ndarray:
    text = df["comment_text"].astype(str)
    length = text.str.len().to_numpy(dtype=np.float32)
    word_count = text.str.split().str.len().fillna(0).to_numpy(dtype=np.float32)
    avg_word_len = np.divide(
        length, word_count, out=np.zeros_like(length), where=word_count > 0
    )

    uppercase_count = text.str.count(r"[A-Z]").to_numpy(dtype=np.float32)
    uppercase_ratio = np.divide(
        uppercase_count,
        length,
        out=np.zeros_like(length),
        where=length > 0,
    )

    features = pd.DataFrame(
        {
            "length": length,
            "word_count": word_count,
            "avg_word_len": avg_word_len,
            "uppercase_count": uppercase_count,
            "uppercase_ratio": uppercase_ratio,
            "exclamation_count": text.str.count("!").to_numpy(dtype=np.float32),
            "question_count": text.str.count("?").to_numpy(dtype=np.float32),
            "digit_count": text.str.count(r"\d").to_numpy(dtype=np.float32),
            "repeated_char_count": text.str.count(r"(.)\1{2,}").to_numpy(
                dtype=np.float32
            ),
            "has_url": text.str.contains(r"http|www", regex=True)
            .astype(np.float32)
            .to_numpy(),
            "has_email": text.str.contains("@", regex=False)
            .astype(np.float32)
            .to_numpy(),
            "has_newline": text.str.contains("\n", regex=False)
            .astype(np.float32)
            .to_numpy(),
            "has_many_caps": (uppercase_ratio > 0.5).astype(np.float32).to_numpy(),
        },
        index=df.index,
    )
    return features.to_numpy(dtype=np.float32)


def make_vectorizers():
    word_vec = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        max_features=MAX_FEATURES,
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        dtype=np.float32,
    )
    char_vec = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        analyzer="char_wb",
        ngram_range=(2, 4),
        max_features=MAX_FEATURES,
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        dtype=np.float32,
    )
    return word_vec, char_vec


def fit_tfidf_pipeline(text: pd.Series):
    word_vec, char_vec = make_vectorizers()
    word_vec.fit(text)
    char_vec.fit(text)
    X = sparse.hstack(
        [word_vec.transform(text), char_vec.transform(text)],
        format="csr",
    )
    svd = TruncatedSVD(n_components=SVS_COMPONENTS, random_state=42)
    svd.fit(X)
    return word_vec, char_vec, svd


def transform_tfidf_pipeline(text: pd.Series, word_vec, char_vec, svd) -> np.ndarray:
    X = sparse.hstack(
        [word_vec.transform(text), char_vec.transform(text)],
        format="csr",
    )
    return svd.transform(X).astype(np.float32)


text_feats_train = build_text_features(train)
text_feats_test = build_text_features(test)

params = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.06,
    "num_leaves": 63,
    "max_bin": 255,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 1,
    "min_data_in_leaf": 50,
    "lambda_l1": 0.1,
    "lambda_l2": 1.0,
    "is_unbalance": True,
    "seed": 42,
    "num_threads": -1,
    "verbosity": -1,
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(train), dtype=np.float32)
best_iterations = []

for fold, (train_idx, valid_idx) in enumerate(skf.split(train, y), start=1):
    train_text = train.loc[train_idx, "comment_text"]
    valid_text = train.loc[valid_idx, "comment_text"]

    word_vec, char_vec, svd = fit_tfidf_pipeline(train_text)

    X_train_svd = transform_tfidf_pipeline(train_text, word_vec, char_vec, svd)
    X_valid_svd = transform_tfidf_pipeline(valid_text, word_vec, char_vec, svd)

    X_train = np.hstack([X_train_svd, text_feats_train[train_idx]])
    X_valid = np.hstack([X_valid_svd, text_feats_train[valid_idx]])

    train_ds = lgb.Dataset(X_train, label=y[train_idx])
    valid_ds = lgb.Dataset(X_valid, label=y[valid_idx])

    model = lgb.train(
        params,
        train_ds,
        num_boost_round=1500,
        valid_sets=[valid_ds],
        valid_names=["valid"],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
    )

    best_iter = int(getattr(model, "best_iteration", 0) or 0)
    best_iter = max(1, best_iter)
    best_iterations.append(best_iter)

    oof_preds[valid_idx] = model.predict(X_valid, num_iteration=best_iter)

    fold_auc = roc_auc_score(y[valid_idx], oof_preds[valid_idx])
    print(f"Fold {fold} ROC AUC: {fold_auc:.6f}, best_iteration: {best_iter}")

    del X_train, X_valid, X_train_svd, X_valid_svd, train_ds, valid_ds, model
    gc.collect()

cv_auc = roc_auc_score(y, oof_preds)
print(f"CV ROC AUC: {cv_auc:.6f}")

final_best_iter = int(np.clip(np.mean(best_iterations), 50, 1200))

word_vec, char_vec, svd = fit_tfidf_pipeline(train["comment_text"])
X_train_full_svd = transform_tfidf_pipeline(
    train["comment_text"], word_vec, char_vec, svd
)
X_train_full = np.hstack([X_train_full_svd, text_feats_train])

full_ds = lgb.Dataset(X_train_full, label=y)
final_model = lgb.train(
    params,
    full_ds,
    num_boost_round=final_best_iter,
    callbacks=[lgb.log_evaluation(-1)],
)

X_test_svd = transform_tfidf_pipeline(test["comment_text"], word_vec, char_vec, svd)
X_test = np.hstack([X_test_svd, text_feats_test])

test_preds = final_model.predict(X_test)
test_preds = np.clip(test_preds, 0.0, 1.0)

submission = pd.DataFrame(
    {
        "id": np.arange(len(test), dtype=int),
        "target": test_preds.astype(np.float64),
    }
)
submission.to_csv("./working/submission.csv", index=False)
print(f"Saved submission with shape {submission.shape}")
