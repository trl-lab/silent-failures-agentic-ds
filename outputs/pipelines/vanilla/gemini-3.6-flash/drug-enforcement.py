import os
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
from catboost import CatBoostClassifier
import xgboost as xgb

train_path = "/path/to/user/workspace/ds_failures/datasets/strable/drug-enforcement/data/train.parquet"
test_path = "/path/to/user/workspace/ds_failures/datasets/strable/drug-enforcement/data/test.parquet"

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = "classification"
y_raw = train_df[target_col]
X_train_raw = train_df.drop(columns=[target_col]).copy()
X_test_raw = test_df.copy()

le = LabelEncoder()
y = le.fit_transform(y_raw)
num_classes = len(le.classes_)

string_cols = [
    col for col in X_train_raw.columns 
    if X_train_raw[col].dtype == "object" or X_train_raw[col].dtype.name == "category"
]
numeric_cols = [col for col in X_train_raw.columns if col not in string_cols]

for col in string_cols:
    X_train_raw[col] = X_train_raw[col].fillna("").astype(str)
    X_test_raw[col] = X_test_raw[col].fillna("").astype(str)

for col in numeric_cols:
    med = X_train_raw[col].median() if len(X_train_raw[col].dropna()) > 0 else 0
    X_train_raw[col] = X_train_raw[col].fillna(med)
    X_test_raw[col] = X_test_raw[col].fillna(med)

text_train = X_train_raw[string_cols].apply(
    lambda r: " ".join([f"{c}: {val}" for c, val in r.items() if val]), axis=1
)
text_test = X_test_raw[string_cols].apply(
    lambda r: " ".join([f"{c}: {val}" for c, val in r.items() if val]), axis=1
)

vec_word = TfidfVectorizer(ngram_range=(1, 3), max_features=25000, sublinear_tf=True)
vec_char = TfidfVectorizer(ngram_range=(2, 5), analyzer="char", max_features=25000, sublinear_tf=True)

X_word_tr = vec_word.fit_transform(text_train)
X_word_te = vec_word.transform(text_test)

X_char_tr = vec_char.fit_transform(text_train)
X_char_te = vec_char.transform(text_test)

X_sparse_tr = hstack([X_word_tr, X_char_tr]).tocsr()
X_sparse_te = hstack([X_word_te, X_char_te]).tocsr()

if len(numeric_cols) > 0:
    scaler = StandardScaler()
    X_num_tr = scaler.fit_transform(X_train_raw[numeric_cols])
    X_num_te = scaler.transform(X_test_raw[numeric_cols])
    X_sparse_full_tr = hstack([X_sparse_tr, csr_matrix(X_num_tr)]).tocsr()
    X_sparse_full_te = hstack([X_sparse_te, csr_matrix(X_num_te)]).tocsr()
else:
    X_sparse_full_tr = X_sparse_tr
    X_sparse_full_te = X_sparse_te

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

linear_models = [
    ("lr_c1", LogisticRegression(C=1.0, max_iter=1000, random_state=42, n_jobs=-1)),
    ("lr_c5", LogisticRegression(C=5.0, max_iter=1000, random_state=42, n_jobs=-1)),
    ("sgd", SGDClassifier(loss="log_loss", alpha=1e-5, max_iter=1000, random_state=42)),
    ("nb", MultinomialNB(alpha=0.1)),
]

oof_preds_list = []
test_preds_list = []

for name, model in linear_models:
    oof_prob = np.zeros((len(y), num_classes))
    test_prob = np.zeros((X_sparse_te.shape[0], num_classes))

    for train_idx, val_idx in skf.split(X_sparse_full_tr, y):
        if name == "nb":
            model.fit(X_sparse_tr[train_idx], y[train_idx])
            oof_prob[val_idx] = model.predict_proba(X_sparse_tr[val_idx])
            test_prob += model.predict_proba(X_sparse_te) / 5.0
        else:
            model.fit(X_sparse_full_tr[train_idx], y[train_idx])
            oof_prob[val_idx] = model.predict_proba(X_sparse_full_tr[val_idx])
            test_prob += model.predict_proba(X_sparse_full_te) / 5.0

    oof_preds_list.append(oof_prob)
    test_preds_list.append(test_prob)

svd_comp = min(100, X_sparse_tr.shape[1] - 1)
svd = TruncatedSVD(n_components=svd_comp, random_state=42)
X_svd_tr = svd.fit_transform(X_sparse_tr)
X_svd_te = svd.transform(X_sparse_te)

dense_tr_components = [X_svd_tr] + oof_preds_list
dense_te_components = [X_svd_te] + test_preds_list

if len(numeric_cols) > 0:
    dense_tr_components.append(X_num_tr)
    dense_te_components.append(X_num_te)

X_dense_tr = np.hstack(dense_tr_components)
X_dense_te = np.hstack(dense_te_components)

lgb_oof = np.zeros((len(y), num_classes))
lgb_test = np.zeros((X_dense_te.shape[0], num_classes))

cat_oof = np