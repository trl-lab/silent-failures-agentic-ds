import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score
import lightgbm as lgb
import warnings

warnings.filterwarnings("ignore")

# 1. Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "classification"
y_raw = train_df[target_col]
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw)
num_classes = len(label_encoder.classes_)
n_train = len(train_df)
n_test = len(test_df)

combined_df = pd.concat(
    [train_df.drop(columns=[target_col]), test_df], ignore_index=True
)

# 2. Identify and parse feature types
date_cols = []
text_cols = []
cat_cols = []
num_cols = []

for col in combined_df.columns:
    if col.lower() in ["id", "index", "unnamed: 0"]:
        continue

    # Try date parsing if column contains 'date' in name or is datetime
    if "date" in col.lower() or pd.api.types.is_datetime64_any_dtype(combined_df[col]):
        try:
            combined_df[col] = pd.to_datetime(combined_df[col], errors="coerce")
            date_cols.append(col)
            continue
        except Exception:
            pass

    if pd.api.types.is_numeric_dtype(combined_df[col]):
        num_cols.append(col)
    else:
        sample_vals = combined_df[col].dropna().astype(str)
        if len(sample_vals) > 0:
            avg_len = sample_vals.apply(len).mean()
            if avg_len > 30 or sample_vals.nunique() > 500:
                text_cols.append(col)
            else:
                cat_cols.append(col)

# 3. Extract Date Features and Date Differences
for col in date_cols:
    combined_df[f"{col}_year"] = combined_df[col].dt.year
    combined_df[f"{col}_month"] = combined_df[col].dt.month
    combined_df[f"{col}_day"] = combined_df[col].dt.day
    combined_df[f"{col}_dayofweek"] = combined_df[col].dt.dayofweek
    num_cols.extend([f"{col}_year", f"{col}_month", f"{col}_day", f"{col}_dayofweek"])

for i in range(len(date_cols)):
    for j in range(i + 1, len(date_cols)):
        d1, d2 = date_cols[i], date_cols[j]
        diff_col = f"diff_days_{d1}_{d2}"
        combined_df[diff_col] = (
            combined_df[d1] - combined_df[d2]
        ).dt.total_seconds() / 86400.0
        num_cols.append(diff_col)

# 4. Categorical Encoding & Frequency Features
for col in cat_cols:
    freq = combined_df[col].value_counts(normalize=True).to_dict()
    combined_df[f"{col}_freq"] = combined_df[col].map(freq).fillna(0)
    num_cols.append(f"{col}_freq")
    combined_df[col] = combined_df[col].astype("category").cat.codes

# 5. Text Surface Features (length, word count, uppercase count)
for col in text_cols:
    s = combined_df[col].fillna("").astype(str)
    combined_df[f"{col}_len"] = s.apply(len)
    combined_df[f"{col}_word_count"] = s.apply(lambda x: len(x.split()))
    combined_df[f"{col}_caps_count"] = s.apply(
        lambda x: sum(1 for c in x if c.isupper())
    )
    num_cols.extend([f"{col}_len", f"{col}_word_count", f"{col}_caps_count"])

# 6. Combined full text field
if text_cols:
    combined_df["_all_text_combined"] = (
        combined_df[text_cols].fillna("").astype(str).agg(" ".join, axis=1)
    )
    eval_text_cols = text_cols + ["_all_text_combined"]
else:
    eval_text_cols = []

# TruncatedSVD text features
svd_dfs = []
for col in text_cols:
    filled_text = combined_df[col].fillna("").astype(str)
    tfidf = TfidfVectorizer(max_features=15000, ngram_range=(1, 2), sublinear_tf=True)
    tfidf_mat = tfidf.fit_transform(filled_text)
    n_components = min(48, tfidf_mat.shape[1] - 1)
    if n_components > 2:
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        svd_feats = svd.fit_transform(tfidf_mat)
        svd_col_names = [f"{col}_svd_{i}" for i in range(n_components)]
        svd_dfs.append(pd.DataFrame(svd_feats, columns=svd_col_names))

# Assemble non-OOF tabular features
dense_feature_cols = [c for c in cat_cols + num_cols if c in combined_df.columns]
X_base = combined_df[dense_feature_cols].copy()
if svd_dfs:
    X_base = pd.concat([X_base] + svd_dfs, axis=1)

# 7. Out-of-Fold Linear Text Classification Meta-Features
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_text_features = {
    f"{col}_oof_p{c}": np.zeros(n_train)
    for col in eval_text_cols
    for c in range(num_classes)
}
test_text_features = {
    f"{col}_oof_p{c}": np.zeros(n_test)
    for col in eval_text_cols
    for c in range(num_classes)
}

for col in eval_text_cols:
    text_data = combined_df[col].fillna("").astype(str).values
    train_text = text_data[:n_train]
    test_text = text_data[n_train:]

    for fold, (train_idx, val_idx) in enumerate(skf.split(train_text, y)):
        tr_txt, val_txt = train_text[train_idx], train_text[val_idx]
        y_tr = y[train_idx]

        tfidf_vec = TfidfVectorizer(
            max_features=25000, ngram_range=(1, 2), sublinear_tf=True, min_df=2
        )
        X_tr_tfidf = tfidf_vec.fit_transform(tr_txt)
        X_va_tfidf = tfidf_vec.transform(val_txt)
        X_te_tfidf = tfidf_vec.transform(test_text)

        clf = LogisticRegression(C=2.0, max_iter=1000, random_state=42)
        clf.fit(X_tr_tfidf, y_tr)

        val_probs = clf.predict_proba(X_va_tfidf)
        test_probs = clf.predict_proba(X_te_tfidf)

        for c in range(num_classes):
            oof_text_features[f"{col}_oof_p{c}"][val_idx] = val_probs[:, c]
            test_text_features[f"{col}_oof_p{c}"] += test_probs[:, c] / n_splits

train_oof_df = pd.DataFrame(oof_text_features)
test_oof_df = pd.DataFrame(test_text_features)

X_train = pd.concat(
    [X_base.iloc[:n_train].reset_index(drop=True), train_oof_df], axis=1
)
X_test = pd.concat([X_base.iloc[n_train:].reset_index(drop=True), test_oof_df], axis=1)

# Ensure column names are unique and clean
X_train.columns = [f"f_{i}_{c}" for i, c in enumerate(X_train.columns)]
X_test.columns = [f"f_{i}_{c}" for i, c in enumerate(X_test.columns)]

# 8. Train Stratified 5-Fold LightGBM Model
oof_preds = np.zeros((n_train, num_classes))
test_preds = np.zeros((n_test, num_classes))

lgb_params = {
    "objective": "multiclass",
    "num_class": num_classes,
    "metric": "multi_logloss",
    "boosting_type": "gbdt",
    "learning_rate": 0.04,
    "num_leaves": 31,
    "max_depth": -1,
    "feature_fraction": 0.75,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "min_child_samples": 20,
    "verbose": -1,
    "n_jobs": -1,
    "random_state": 42,
}

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y)):
    X_tr, y_tr = X_train.iloc[train_idx], y[train_idx]
    X_va, y_va = X_train.iloc[val_idx], y[val_idx]

    trn_data = lgb.Dataset(X_tr, label=y_tr)
    val_data = lgb.Dataset(X_va, label=y_va, reference=trn_data)

    model = lgb.train(
        lgb_params,
        trn_data,
        num_boost_round=1200,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )

    val_pred_probs = model.predict(X_va, num_iteration=model.best_iteration)
    oof_preds[val_idx] = val_pred_probs
    test_preds += model.predict(X_test, num_iteration=model.best_iteration) / n_splits

oof_pred_labels = np.argmax(oof_preds, axis=1)
cv_accuracy = accuracy_score(y, oof_pred_labels)
print(f"5-Fold CV Accuracy: {cv_accuracy:.5f}")

# 9. Generate and save test submission
test_pred_labels = np.argmax(test_preds, axis=1)
test_pred_classes = label_encoder.inverse_transform(test_pred_labels)

os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame(
    {"id": np.arange(len(test_df)), "classification": test_pred_classes}
)
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved successfully to ./working/submission.csv")
