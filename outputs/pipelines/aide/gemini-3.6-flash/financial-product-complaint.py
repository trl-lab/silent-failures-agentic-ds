import os
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier

# 1. Load Data
train_path = "./input/train.parquet"
test_path = "./input/test.parquet"

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = "Company response to consumer"

# Target encoding
le_target = LabelEncoder()
y_train = le_target.fit_transform(train_df[target_col].astype(str))
n_classes = len(le_target.classes_)

# Combine datasets for unified feature engineering
n_train = len(train_df)
df_all = pd.concat(
    [train_df.drop(columns=[target_col], errors="ignore"), test_df], ignore_index=True
)

# 2. Text Feature Construction
text_cols = [
    c
    for c in [
        "Consumer complaint narrative",
        "Issue",
        "Sub-issue",
        "Product",
        "Sub-product",
        "Company public response",
        "Company",
        "Tags",
    ]
    if c in df_all.columns
]
for col in text_cols:
    df_all[col] = df_all[col].fillna("").astype(str)

df_all["full_text"] = df_all[text_cols].apply(lambda row: " ".join(row), axis=1)

narrative_col = (
    "Consumer complaint narrative"
    if "Consumer complaint narrative" in df_all.columns
    else text_cols[0]
)
df_all["narrative_len"] = df_all[narrative_col].str.len()
df_all["narrative_word_count"] = df_all[narrative_col].apply(lambda x: len(x.split()))
df_all["is_narrative_missing"] = (df_all[narrative_col] == "").astype(int)

# 3. Date Feature Construction
date_cols = [
    c for c in ["Date received", "Date sent to company"] if c in df_all.columns
]
for c in date_cols:
    df_all[c] = pd.to_datetime(df_all[c], errors="coerce")

if len(date_cols) == 2:
    df_all["days_to_send"] = (
        df_all["Date sent to company"] - df_all["Date received"]
    ).dt.days.fillna(-1)

if "Date received" in df_all.columns:
    df_all["receive_year"] = df_all["Date received"].dt.year.fillna(-1)
    df_all["receive_month"] = df_all["Date received"].dt.month.fillna(-1)
    df_all["receive_day"] = df_all["Date received"].dt.day.fillna(-1)
    df_all["receive_dow"] = df_all["Date received"].dt.dayofweek.fillna(-1)

df_all = df_all.drop(columns=date_cols, errors="ignore")

# 4. Categorical Frequency & Numerical Encoding
cat_cols = [
    c
    for c in [
        "Product",
        "Sub-product",
        "Issue",
        "Sub-issue",
        "State",
        "Submitted via",
        "Company",
        "Tags",
    ]
    if c in df_all.columns
]
tabular_num_df = pd.DataFrame(index=df_all.index)

for c in cat_cols:
    freq = df_all[c].value_counts()
    tabular_num_df[f"{c}_freq"] = df_all[c].map(freq).astype(float)
    le = LabelEncoder()
    tabular_num_df[f"{c}_code"] = le.fit_transform(df_all[c].astype(str))

# Add extracted numerical features
num_feats = [
    "narrative_len",
    "narrative_word_count",
    "is_narrative_missing",
    "days_to_send",
    "receive_year",
    "receive_month",
    "receive_day",
    "receive_dow",
]
for nf in num_feats:
    if nf in df_all.columns:
        tabular_num_df[nf] = df_all[nf].astype(float)

# Separate train and test tabular numerical features
train_num = tabular_num_df.iloc[:n_train].reset_index(drop=True)
test_num = tabular_num_df.iloc[n_train:].reset_index(drop=True)

train_text = df_all["full_text"].iloc[:n_train].reset_index(drop=True)
test_text = df_all["full_text"].iloc[n_train:].reset_index(drop=True)

# 5. Cross-Validation and Stacking Setup
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_nb_probs = np.zeros((n_train, n_classes))
test_nb_probs = np.zeros((len(test_num), n_classes))

oof_lr_probs = np.zeros((n_train, n_classes))
test_lr_probs = np.zeros((len(test_num), n_classes))

oof_svd = np.zeros((n_train, 25))
test_svd = np.zeros((len(test_num), 25))

oof_final_probs = np.zeros((n_train, n_classes))
test_cb_probs = np.zeros((len(test_num), n_classes))

print("Starting 5-Fold Stratified Cross-Validation...")

for fold, (train_idx, val_idx) in enumerate(skf.split(train_num, y_train)):
    X_tr_text = train_text.iloc[train_idx]
    X_val_text = train_text.iloc[val_idx]
    X_te_text = test_text

    y_tr, y_val = y_train[train_idx], y_train[val_idx]

    # Fit TF-IDF on fold text
    vec = TfidfVectorizer(
        max_features=35000, ngram_range=(1, 2), sublinear_tf=True, min_df=3
    )
    X_tr_tfidf = vec.fit_transform(X_tr_text)
    X_val_tfidf = vec.transform(X_val_text)
    X_te_tfidf = vec.transform(X_te_text)

    # Base Model 1: Complement Naive Bayes
    cnb = ComplementNB(alpha=0.5)
    cnb.fit(X_tr_tfidf, y_tr)
    oof_nb_probs[val_idx] = cnb.predict_proba(X_val_tfidf)
    test_nb_probs += cnb.predict_proba(X_te_tfidf) / 5.0

    # Base Model 2: Logistic Regression
    lr = LogisticRegression(
        C=2.0,
        max_iter=400,
        solver="lbfgs",
        multi_class="multinomial",
        random_state=42,
        n_jobs=-1,
    )
    lr.fit(X_tr_tfidf, y_tr)
    oof_lr_probs[val_idx] = lr.predict_proba(X_val_tfidf)
    test_lr_probs += lr.predict_proba(X_te_tfidf) / 5.0

    # SVD Component
    svd = TruncatedSVD(n_components=25, random_state=42)
    oof_svd[val_idx] = svd.fit_transform(X_val_tfidf)
    test_svd += svd.transform(X_te_tfidf) / 5.0

# Assemble clean numerical meta-features
meta_nb_train = pd.DataFrame(
    oof_nb_probs, columns=[f"nb_p_{i}" for i in range(n_classes)]
)
meta_lr_train = pd.DataFrame(
    oof_lr_probs, columns=[f"lr_p_{i}" for i in range(n_classes)]
)
meta_svd_train = pd.DataFrame(oof_svd, columns=[f"svd_{i}" for i in range(25)])

meta_nb_test = pd.DataFrame(
    test_nb_probs, columns=[f"nb_p_{i}" for i in range(n_classes)]
)
meta_lr_test = pd.DataFrame(
    test_lr_probs, columns=[f"lr_p_{i}" for i in range(n_classes)]
)
meta_svd_test = pd.DataFrame(test_svd, columns=[f"svd_{i}" for i in range(25)])

X_train_full = pd.concat(
    [train_num, meta_nb_train, meta_lr_train, meta_svd_train], axis=1
)
X_test_full = pd.concat([test_num, meta_nb_test, meta_lr_test, meta_svd_test], axis=1)

# 6. Second Stage Model: CatBoost
for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_full, y_train)):
    X_tr, y_tr = X_train_full.iloc[train_idx], y_train[train_idx]
    X_val, y_val = X_train_full.iloc[val_idx], y_train[val_idx]

    cb = CatBoostClassifier(
        iterations=650,
        learning_rate=0.07,
        depth=6,
        loss_function="MultiClass",
        eval_metric="Accuracy",
        random_seed=42 + fold,
        verbose=0,
    )
    cb.fit(X_tr, y_tr, eval_set=(X_val, y_val), early_stopping_rounds=50)

    oof_final_probs[val_idx] = cb.predict_proba(X_val)
    test_cb_probs += cb.predict_proba(X_test_full) / 5.0

oof_preds = np.argmax(oof_final_probs, axis=1)
acc = accuracy_score(y_train, oof_preds)
print(f"Validation Accuracy: {acc:.5f}")

# 7. Generate Submission File
os.makedirs("./working", exist_ok=True)
test_preds = np.argmax(test_cb_probs, axis=1)
test_pred_labels = le_target.inverse_transform(test_preds)

submission = pd.DataFrame(
    {"id": np.arange(len(test_num)), target_col: test_pred_labels}
)
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved successfully to ./working/submission.csv.")
