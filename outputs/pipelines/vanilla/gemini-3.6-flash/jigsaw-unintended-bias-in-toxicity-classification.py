```python
import numpy as np
import pandas as pd
from scipy.sparse import hstack, diags
from scipy.stats import rankdata
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier, SGDRegressor

def main():
    train_path = "/path/to/user/workspace/ds_failures/datasets/mlebench/jigsaw-unintended-bias-in-toxicity-classification/data/train.parquet"
    test_path = "/path/to/user/workspace/ds_failures/datasets/mlebench/jigsaw-unintended-bias-in-toxicity-classification/data/test.parquet"

    train = pd.read_parquet(train_path)
    test = pd.read_parquet(test_path)

    train_text = train["comment_text"].fillna("").astype(str)
    test_text = test["comment_text"].fillna("").astype(str)

    y_train_cont = train["target"].values.astype(np.float32)
    y_train_bin = (y_train_cont >= 0.5).astype(np.int32)

    aux_cols = ["severe_toxicity", "obscene", "identity_attack", "insult", "threat"]
    aux_sum = np.zeros(len(train), dtype=np.float32)
    for col in aux_cols:
        if col in train.columns:
            aux_sum += train[col].fillna(0).values.astype(np.float32)
    y_composite = y_train_cont + 0.25 * aux_sum

    word_vec = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=150000,
        sublinear_tf=True,
        strip_accents="unicode",
        lower=True,
        min_df=3,
        dtype=np.float32,
    )
    X_train_word = word_vec.fit_transform(train_text)
    X_test_word = word_vec.transform(test_text)

    char_vec = TfidfVectorizer(
        ngram_range=(2, 5),
        analyzer="char_wb",
        max_features=100000,
        sublinear_tf=True,
        strip_accents="unicode",
        lower=True,
        min_df=5,
        dtype=np.float32,
    )
    X_train_char = char_vec.fit_transform(train_text)
    X_test_char = char_vec.transform(test_text)

    X_train = hstack([X_train_word, X_train_char], format="csr")
    X_test = hstack([X_test_word, X_test_char], format="csr")

    model1 = SGDClassifier(
        loss="log_loss", alpha=1e-5, max_iter=30, random_state=42
    )
    model1.fit(X_train, y_train_bin)
    preds1 = model1.predict_proba(X_test)[:, 1]

    model2 = SGDClassifier(
        loss="modified_huber", alpha=1e-5, max_iter=30, random_state=42
    )
    model2.fit(X_train, y_train_bin)
    preds2 = model2.predict_proba(X_test)[:, 1]

    model3 = SGDRegressor(
        loss="squared_error", alpha=1e-6, max_iter=30, random_state=42
    )
    model3.fit(X_train, y_train_cont)
    preds3 = model3.predict(X_test)

    pos_sum = np.asarray(X_train[y_train_bin == 1].sum(axis=0)).squeeze() + 1.0
    neg_sum = np.asarray(X_train[y_train_bin == 0].sum(axis=0)).squeeze() + 1.0
    pos_sum = pos_sum / np.sum(pos_sum)
    neg_sum = neg_sum / np.sum(neg_sum)
    r = np.log(pos_sum / neg_sum)

    r_mat = diags(r)
    X_train_nb = X_train @ r_mat
    X_test_nb = X_test @ r_mat

    model4 = SGDClassifier(
        loss="log_loss", alpha=1e-5, max_iter=30, random_state=42
    )
    model4.fit(X_train_nb, y_train_bin)
    preds4 = model4.predict_proba(X_test_nb)[:, 1]

    model5 = SGDRegressor(
        loss="squared_error", alpha=1e-6, max_iter=30, random_state=42
    )
    model5.fit(X_train, y_composite)
    preds5 = model5.predict(X_test)

    rank1 = rankdata(preds1) / len(preds1)
    rank2 = rankdata(preds2) / len(preds2)
    rank3 = rankdata(preds3) / len(preds3)
    rank4 = rankdata(preds4) / len(preds4)
    rank5 = rankdata(preds5) / len(preds5)

    final_preds = (rank1 + rank2 + rank3 + rank4 + rank5) / 5.0

    sub = pd.DataFrame({"id": np.arange(len(test)), "target": final_preds})
    sub.to_csv("predictions.csv", index=False)

if __name__ ==