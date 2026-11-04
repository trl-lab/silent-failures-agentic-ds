import os
import warnings
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.naive_bayes import ComplementNB
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
import lightgbm as lgb

warnings.filterwarnings("ignore")

# Load datasets
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "classification"
le_target = LabelEncoder()
y = le_target.fit_transform(train_df[target_col])

train_len = len(train_df)
df_all = pd.concat(
    [train_df.drop(columns=[target_col]), test_df], axis=0, ignore_index=True
)

# Identify feature types
date_cols = [col for col in df_all.columns if "date" in col.lower()]
text_cols = [
    col
    for col in df_all.columns
    if col not in date_cols and df_all[col].dtype == "object"
]

# 1. Temporal feature engineering
for col in date_cols:
    df_all[col] = pd.to_datetime(df_all[col], errors="coerce")
    df_all[f"{col}_year"] = df_all[col].dt.year
    df_all[f"{col}_month"] = df_all[col].dt.month
    df_all[f"{col}_day"] = df_all[col].dt.day
    df_all[f"{col}_dayofweek"] = df_all[col].dt.dayofweek
    df_all[f"{col}_dayofyear"] = df_all[col].dt.dayofyear

for i in range(len(date_cols)):
    for j in range(i + 1, len(date_cols)):
        col1, col2 = date_cols[i], date_cols[j]
        df_all[f"date_diff_{col1}_{col2}"] = (df_all[col1] - df_all[col2]).dt.days

df_all.drop(columns=date_cols, inplace=True)

# 2. Text preprocessing & tabular features
df_all["combined_text"] = ""
for col in text_cols:
    df_all["combined_text"] += " " + df_all[col].fillna("").astype(str)

df_all["text_len"] = df_all["combined_text"].apply(len)
df_all["word_count"] = df_all["combined_text"].apply(lambda x: len(x.split()))
df_all["upper_ratio"] = df_all["combined_text"].apply(
    lambda x: sum(1 for c in x if c.isupper()) / (len(x) + 1)
)
df_all["digit_ratio"] = df_all["combined_text"].apply(
    lambda x: sum(1 for c in x if c.isdigit()) / (len(x) + 1)
)

for col in text_cols:
    freq = df_all[col].value_counts(normalize=True).to_dict()
    df_all[f"{col}_freq"] = df_all[col].map(freq).fillna(0)
    df_all[f"{col}_cat"] = pd.factorize(df_all[col])[0]

X_train_df = df_all.iloc[:train_len].copy()
X_test_df = df_all.iloc[train_len:].copy()

# Combined text TF-IDF & TruncatedSVD
word_tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=10000, sublinear_tf=True)
char_tfidf = TfidfVectorizer(
    ngram_range=(3, 5), analyzer="char_wb", max_features=10000, sublinear_tf=True
)

X_train_text_w = word_tfidf.fit_transform(X_train_df["combined_text"])
X_test_text_w = word_tfidf.transform(X_test_df["combined_text"])

X_train_text_c = char_tfidf.fit_transform(X_train_df["combined_text"])
X_test_text_c = char_tfidf.transform(X_test_df["combined_text"])

X_train_text_comb = hstack([X_train_text_w, X_train_text_c]).tocsr()
X_test_text_comb = hstack([X_test_text_w, X_test_text_c]).tocsr()

svd = TruncatedSVD(n_components=50, random_state=42)
X_train_svd = svd.fit_transform(X_train_text_comb)
X_test_svd = svd.transform(X_test_text_comb)

svd_cols = [f"svd_{i}" for i in range(50)]
svd_train_df = pd.DataFrame(X_train_svd, columns=svd_cols, index=X_train_df.index)
svd_test_df = pd.DataFrame(X_test_svd, columns=svd_cols, index=X_test_df.index)

X_train_df.drop(columns=text_cols + ["combined_text"], inplace=True)
X_test_df.drop(columns=text_cols + ["combined_text"], inplace=True)

X_train_full = pd.concat([X_train_df, svd_train_df], axis=1)
X_test_full = pd.concat([X_test_df, svd_test_df], axis=1)

imputer = SimpleImputer(strategy="median")
X_train_mat = imputer.fit_transform(X_train_full)
X_test_mat = imputer.transform(X_test_full)

num_classes = len(np.unique(y))
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Identify individual text columns with rich textual content
rich_text_cols = [
    col for col in text_cols if df_all[col].fillna("").astype(str).str.len().mean() > 5
]
if not rich_text_cols:
    rich_text_cols = text_cols

text_sources = [("combined", X_train_text_comb, X_test_text_comb)]

for col in rich_text_cols:
    col_str_train = train_df[col].fillna("").astype(str)
    col_str_test = test_df[col].fillna("").astype(str)
    col_tfidf = TfidfVectorizer(
        ngram_range=(1, 2), max_features=5000, sublinear_tf=True
    )
    tr_col_vec = col_tfidf.fit_transform(col_str_train)
    te_col_vec = col_tfidf.transform(col_str_test)
    text_sources.append((col, tr_col_vec, te_col_vec))

meta_train_list = [X_train_mat]
meta_test_list = [X_test_mat]

for src_name, tr_vec, te_vec in text_sources:
    oof_nb = np.zeros((train_len, num_classes))
    test_nb = np.zeros((len(test_df), num_classes))
    oof_lr = np.zeros((train_len, num_classes))
    test_lr = np.zeros((len(test_df), num_classes))

    for train_idx, val_idx in skf.split(tr_vec, y):
        cnb = ComplementNB(alpha=0.1)
        cnb.fit(tr_vec[train_idx], y[train_idx])
        oof_nb[val_idx] = cnb.predict_proba(tr_vec[val_idx])
        test_nb += cnb.predict_proba(te_vec) / 5.0

        lr = LogisticRegression(
            C=1.0, max_iter=500, solver="lbfgs", multi_class="multinomial"
        )
        lr.fit(tr_vec[train_idx], y[train_idx])
        oof_lr[val_idx] = lr.predict_proba(tr_vec[val_idx])
        test_lr += lr.predict_proba(te_vec) / 5.0

    meta_train_list.extend([oof_nb, oof_lr])
    meta_test_list.extend([test_nb, test_lr])

X_train_final = np.hstack(meta_train_list)
X_test_final = np.hstack(meta_test_list)

# 3. Final Gradient Boosted Classifier Training
oof_preds_prob = np.zeros((train_len, num_classes))
test_preds_prob = np.zeros((len(test_df), num_classes))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_final, y)):
    X_tr, y_tr = X_train_final[train_idx], y[train_idx]
    X_va, y_va = X_train_final[val_idx], y[val_idx]

    model = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        objective="multiclass",
        num_class=num_classes,
        verbosity=-1,
    )

    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )

    oof_preds_prob[val_idx] = model.predict_proba(X_va)
    test_preds_prob += model.predict_proba(X_test_final) / 5.0

oof_preds = np.argmax(oof_preds_prob, axis=1)
accuracy = accuracy_score(y, oof_preds)
print(f"5-Fold CV Accuracy: {accuracy:.5f}")

# Generate and save test submission
test_preds = np.argmax(test_preds_prob, axis=1)
submission = pd.DataFrame(
    {
        "id": np.arange(len(test_preds)),
        "classification": le_target.inverse_transform(test_preds),
    }
)

os.makedirs("./working", exist_ok=True)
submission.to_csv("./working/submission.csv", index=False)
