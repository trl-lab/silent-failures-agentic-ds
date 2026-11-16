import os
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

os.makedirs("./working", exist_ok=True)

# 1. Load Data
train_path = "./input/train.parquet"
test_path = "./input/test.parquet"

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = "Status"

if (
    train_df[target_col].dtype == "object"
    or train_df[target_col].dtype.name == "category"
):
    le = LabelEncoder()
    y = le.fit_transform(train_df[target_col].astype(str))
else:
    y = train_df[target_col].values.astype(int)

X_train = train_df.drop(columns=[target_col], errors="ignore").copy()
X_test = test_df.copy()

common_cols = [c for c in X_train.columns if c in X_test.columns]
X_train = X_train[common_cols]
X_test = X_test[common_cols]

# 2. Detect Column Types
date_cols = []
text_cols = []
cat_cols = []
num_cols = []

for col in X_train.columns:
    train_series = X_train[col]
    test_series = X_test[col]

    if train_series.dtype == "object" or train_series.dtype.name == "category":
        sample = train_series.dropna().astype(str).head(100)
        is_date = False
        try:
            parsed = pd.to_datetime(sample, errors="coerce")
            if parsed.notna().sum() / max(len(sample), 1) > 0.6:
                is_date = True
        except Exception:
            is_date = False

        if is_date:
            date_cols.append(col)
        else:
            avg_len = sample.apply(len).mean() if len(sample) > 0 else 0
            avg_words = (
                sample.apply(lambda x: len(x.split())).mean() if len(sample) > 0 else 0
            )
            if avg_words > 2.5 or avg_len > 30:
                text_cols.append(col)
            else:
                cat_cols.append(col)
    elif np.issubdtype(train_series.dtype, np.datetime64):
        date_cols.append(col)
    elif np.issubdtype(train_series.dtype, np.number):
        if train_series.nunique() <= 15:
            cat_cols.append(col)
        else:
            num_cols.append(col)
    else:
        cat_cols.append(col)

df_tr = pd.DataFrame(index=X_train.index)
df_te = pd.DataFrame(index=X_test.index)

# 3. Numerical Features
for col in num_cols:
    df_tr[col] = X_train[col].astype(float)
    df_te[col] = X_test[col].astype(float)

# 4. Temporal Features
for col in date_cols:
    tr_dt = pd.to_datetime(X_train[col], errors="coerce")
    te_dt = pd.to_datetime(X_test[col], errors="coerce")

    df_tr[f"{col}_year"] = tr_dt.dt.year
    df_te[f"{col}_year"] = te_dt.dt.year
    df_tr[f"{col}_month"] = tr_dt.dt.month
    df_te[f"{col}_month"] = te_dt.dt.month
    df_tr[f"{col}_day"] = tr_dt.dt.day
    df_te[f"{col}_day"] = te_dt.dt.day
    df_tr[f"{col}_dayofweek"] = tr_dt.dt.dayofweek
    df_te[f"{col}_dayofweek"] = te_dt.dt.dayofweek
    df_tr[f"{col}_dayofyear"] = tr_dt.dt.dayofyear
    df_te[f"{col}_dayofyear"] = te_dt.dt.dayofyear
    df_tr[f"{col}_quarter"] = tr_dt.dt.quarter
    df_te[f"{col}_quarter"] = te_dt.dt.quarter

    ts_tr = tr_dt.astype("int64") // 10**9
    ts_te = te_dt.astype("int64") // 10**9
    df_tr[f"{col}_timestamp"] = ts_tr.where(tr_dt.notna(), np.nan)
    df_te[f"{col}_timestamp"] = ts_te.where(te_dt.notna(), np.nan)

# 5. Categorical Features (Frequency Encoding + Ordinal)
for col in cat_cols:
    tr_cat = X_train[col].astype(str).fillna("missing")
    te_cat = X_test[col].astype(str).fillna("missing")

    freq_map = tr_cat.value_counts(normalize=True).to_dict()
    df_tr[f"{col}_freq"] = tr_cat.map(freq_map).fillna(0.0)
    df_te[f"{col}_freq"] = te_cat.map(freq_map).fillna(0.0)

    cat_type = pd.CategoricalDtype(categories=tr_cat.unique())
    df_tr[f"{col}_cat"] = tr_cat.astype(cat_type).cat.codes
    df_te[f"{col}_cat"] = te_cat.astype(cat_type).cat.codes

# 6. Text Features
if text_cols:
    for col in text_cols:
        tr_str = X_train[col].fillna("").astype(str)
        te_str = X_test[col].fillna("").astype(str)

        df_tr[f"{col}_len"] = tr_str.apply(len)
        df_te[f"{col}_len"] = te_str.apply(len)
        df_tr[f"{col}_words"] = tr_str.apply(lambda s: len(s.split()))
        df_te[f"{col}_words"] = te_str.apply(lambda s: len(s.split()))
        df_tr[f"{col}_digits"] = tr_str.apply(lambda s: sum(c.isdigit() for c in s))
        df_te[f"{col}_digits"] = te_str.apply(lambda s: sum(c.isdigit() for c in s))

    combined_tr_text = X_train[text_cols].fillna("").astype(str).agg(" ".join, axis=1)
    combined_te_text = X_test[text_cols].fillna("").astype(str).agg(" ".join, axis=1)

    tfidf_word = TfidfVectorizer(
        ngram_range=(1, 2), max_features=10000, sublinear_tf=True
    )
    tr_word_mat = tfidf_word.fit_transform(combined_tr_text)
    te_word_mat = tfidf_word.transform(combined_te_text)

    tfidf_char = TfidfVectorizer(
        ngram_range=(3, 5), analyzer="char", max_features=10000, sublinear_tf=True
    )
    tr_char_mat = tfidf_char.fit_transform(combined_tr_text)
    te_char_mat = tfidf_char.transform(combined_te_text)

    svd_word = TruncatedSVD(n_components=16, random_state=42)
    tr_word_svd = svd_word.fit_transform(tr_word_mat)
    te_word_svd = svd_word.transform(te_word_mat)

    svd_char = TruncatedSVD(n_components=16, random_state=42)
    tr_char_svd = svd_char.fit_transform(tr_char_mat)
    te_char_svd = svd_char.transform(te_char_mat)

    for i in range(16):
        df_tr[f"text_word_svd_{i}"] = tr_word_svd[:, i]
        df_te[f"text_word_svd_{i}"] = te_word_svd[:, i]
        df_tr[f"text_char_svd_{i}"] = tr_char_svd[:, i]
        df_te[f"text_char_svd_{i}"] = te_char_svd[:, i]

# 7. Out-of-fold Target Encoding
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for col in cat_cols:
    df_tr[f"{col}_te"] = np.nan
    df_te[f"{col}_te"] = 0.0

    tr_cat = X_train[col].astype(str).fillna("missing")
    te_cat = X_test[col].astype(str).fillna("missing")
    global_mean = float(y.mean())

    te_acc = np.zeros(len(df_te))

    for train_idx, val_idx in skf.split(df_tr, y):
        cat_fold_tr = tr_cat.iloc[train_idx]
        y_fold_tr = y[train_idx]

        means = pd.Series(y_fold_tr).groupby(cat_fold_tr.values).mean()

        val_mapped = tr_cat.iloc[val_idx].map(means).fillna(global_mean).values
        df_tr.iloc[val_idx, df_tr.columns.get_loc(f"{col}_te")] = val_mapped

        te_acc += te_cat.map(means).fillna(global_mean).values / 5.0

    df_te[f"{col}_te"] = te_acc

# 8. Cross-Validation and Modeling
oof_preds = np.zeros(len(df_tr))
test_preds = np.zeros(len(df_te))

for fold, (train_idx, val_idx) in enumerate(skf.split(df_tr, y)):
    X_tr, y_tr = df_tr.iloc[train_idx], y[train_idx]
    X_va, y_va = df_tr.iloc[val_idx], y[val_idx]

    model = HistGradientBoostingClassifier(
        max_iter=400,
        learning_rate=0.03,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=30,
        random_state=42 + fold,
    )

    model.fit(X_tr, y_tr)

    oof_preds[val_idx] = model.predict_proba(X_va)[:, 1]
    test_preds += model.predict_proba(df_te)[:, 1] / skf.n_splits

# 9. Compute and Print Metric
cv_auc = roc_auc_score(y, oof_preds)
print(f"Validation Metric: {cv_auc:.5f}")

# 10. Save Submission
sub = pd.DataFrame({"id": np.arange(len(test_preds)), "Status": test_preds})
sub.to_csv("./working/submission.csv", index=False)
