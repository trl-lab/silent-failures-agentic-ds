import warnings
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from lightgbm import LGBMClassifier

warnings.filterwarnings('ignore')


def main():
    train_path = "/path/to/user/workspace/ds_failures/datasets/strable/financial-product-complaint/data/train.parquet"
    test_path = "/path/to/user/workspace/ds_failures/datasets/strable/financial-product-complaint/data/test.parquet"

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    target_col = "Company response to consumer"

    y_raw = train_df[target_col]
    le = LabelEncoder()
    y_train = le.fit_transform(y_raw)

    X_train_df = train_df.drop(columns=[target_col])
    X_test_df = test_df.copy()

    id_cols = [
        c
        for c in X_train_df.columns
        if c.lower() in ['id', 'row_id', 'index', 'unnamed: 0']
    ]
    if id_cols:
        X_train_df = X_train_df.drop(columns=id_cols)
        X_test_df = X_test_df.drop(columns=id_cols)

    num_cols = []
    str_cols = []

    for c in X_train_df.columns:
        if pd.api.types.is_numeric_dtype(X_train_df[c]):
            num_cols.append(c)
        else:
            str_cols.append(c)

    num_train_list = []
    num_test_list = []

    for c in num_cols:
        median_val = (
            X_train_df[c].median() if not pd.isna(X_train_df[c].median()) else 0
        )
        tr_val = X_train_df[c].fillna(median_val).values
        te_val = X_test_df[c].fillna(median_val).values
        num_train_list.append(tr_val.reshape(-1, 1))
        num_test_list.append(te_val.reshape(-1, 1))

    if num_cols:
        num_train_mat = np.hstack(num_train_list)
        num_test_mat = np.hstack(num_test_list)
        scaler = StandardScaler()
        num_train_sparse = sp.csr_matrix(scaler.fit_transform(num_train_mat))
        num_test_sparse = sp.csr_matrix(scaler.transform(num_test_mat))
    else:
        num_train_sparse = None
        num_test_sparse = None

    def build_text_series(df):
        text_series = pd.Series("", index=df.index)
        for c in str_cols:
            text_series = (
                text_series + " " + c + ": " + df[c].fillna("").astype(str)
            )
        return text_series

    train_text = build_text_series(X_train_df)
    test_text = build_text_series(X_test_df)

    word_tfidf = TfidfVectorizer(
        ngram_range=(1, 2), max_features=60000, sublinear_tf=True, min_df=2
    )
    word_tr = word_tfidf.fit_transform(train_text)
    word_te = word_tfidf.transform(test_text)

    char_tfidf = TfidfVectorizer(
        ngram_range=(3, 5),
        analyzer='char_wb',
        max_features=30000,
        sublinear_tf=True,
        min_df=5,
    )
    char_tr = char_tfidf.fit_transform(train_text)
    char_te = char_tfidf.transform(test_text)

    to_hstack_train = [word_tr, char_tr]
    to_hstack_test = [word_te, char_te]

    if num_train_sparse is not None:
        to_hstack_train.append(num_train_sparse)
        to_hstack_test.append(num_test_sparse)

    X_tr = sp.hstack(to_hstack_train).tocsr()
    X_te = sp.hstack(to_hstack_test).tocsr()

    clf_lr = LogisticRegression(
        C=2.0, max_iter=1000, solver='lbfgs', random_state=42, n_jobs=-1
    )
    clf_lr.fit(X_tr, y_train)
    prob_lr = clf_lr.predict_proba(X_te)

    try:
        clf_sgd = SGDClassifier(
            loss='log_loss', alpha=1e-5, max_iter=1000, random_state=42
        )
        clf_sgd.fit(X_tr, y_train)
    except Exception:
        clf_sgd = SGDClassifier(
            loss='log', alpha=1e-5, max_iter=1000, random_state=42
        )
        clf_sgd.fit(X_tr, y_train)

    prob_sgd = clf_sgd.predict_proba(X_te)

    clf_lgb = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.08,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    clf_lgb.fit(X_tr, y_train)
    prob_lgb = clf_lgb.predict_proba(X_te)

    prob_ensemble = 0.45 * prob_lr + 0.25 * prob_sgd + 0.30 * prob_lgb
    preds_encoded = np.argmax(prob_ensemble, axis=1)

    preds = le.inverse_transform(preds_encoded)

    sub = pd.DataFrame({"id": np.arange(len(preds)), target_col: preds})
    sub.to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    main()