import os
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import lightgbm as lgb
import warnings

warnings.filterwarnings("ignore")

# 1. Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "classification"
y_raw = train_df[target_col].astype(str)
le = LabelEncoder()
y = le.fit_transform(y_raw)
num_classes = len(le.classes_)

train_len = len(train_df)
combined_df = pd.concat(
    [train_df.drop(columns=[target_col]), test_df], ignore_index=True
)

# 2. Identify column types
date_cols = []
text_cols = []
cat_cols = []
num_cols = []

for col in combined_df.columns:
    if combined_df[col].dtype == "object" or combined_df[col].dtype.name == "category":
        # Check if date-like string
        sample_val = combined_df[col].dropna().astype(str).head(20)
        is_date = False
        if len(sample_val) > 0 and any(
            "date" in col.lower() or "time" in col.lower() for _ in [1]
        ):
            try:
                pd.to_datetime(sample_val, errors="raise")
                is_date = True
            except Exception:
                pass

        if is_date or "date" in col.lower():
            date_cols.append(col)
        else:
            # Check text length vs unique count
            avg_len = sample_val.map(len).mean() if len(sample_val) > 0 else 0
            if avg_len > 30 or combined_df[col].nunique() > 200:
                text_cols.append(col)
            else:
                cat_cols.append(col)
    else:
        if np.issubdtype(combined_df[col].dtype, np.datetime64):
            date_cols.append(col)
        else:
            num_cols.append(col)

# 3. Feature Engineering
features_train = []
features_test = []

# Date features
date_feat_df = pd.DataFrame(index=combined_df.index)
for col in date_cols:
    dt_series = pd.to_datetime(combined_df[col], errors="coerce")
    date_feat_df[f"{col}_year"] = dt_series.dt.year.fillna(-1)
    date_feat_df[f"{col}_month"] = dt_series.dt.month.fillna(-1)
    date_feat_df[f"{col}_day"] = dt_series.dt.day.fillna(-1)
    date_feat_df[f"{col}_dayofweek"] = dt_series.dt.dayofweek.fillna(-1)
    date_feat_df[f"{col}_timestamp"] = dt_series.astype("int64") // 10**9
    date_feat_df[f"{col}_timestamp"] = date_feat_df[f"{col}_timestamp"].replace(
        date_feat_df[f"{col}_timestamp"].min(), np.nan
    )

# Numeric features
num_feat_df = pd.DataFrame(index=combined_df.index)
for col in num_cols:
    num_feat_df[col] = pd.to_numeric(combined_df[col], errors="coerce")

# Categorical features - frequency encoding & integer encoding
cat_feat_df = pd.DataFrame(index=combined_df.index)
for col in cat_cols:
    s = combined_df[col].astype(str).fillna("missing")
    freq = s.value_counts(normalize=True)
    cat_feat_df[f"{col}_freq"] = s.map(freq).astype(float)
    cat_feat_df[f"{col}_cat"] = pd.factorize(s)[0]

# Text TF-IDF + SVD features
text_components = []
for col in text_cols:
    text_data = combined_df[col].fillna("").astype(str)
    tfidf = TfidfVectorizer(
        max_features=25000, ngram_range=(1, 2), min_df=2, sublinear_tf=True
    )
    tfidf_mat = tfidf.fit_transform(text_data)

    n_comp = min(30, max(5, tfidf_mat.shape[1] // 10))
    svd = TruncatedSVD(n_components=n_comp, random_state=42)
    svd_feats = svd.fit_transform(tfidf_mat)

    svd_cols = [f"{col}_svd_{i}" for i in range(n_comp)]
    text_components.append(
        pd.DataFrame(svd_feats, columns=svd_cols, index=combined_df.index)
    )

# Combine tabular features
all_features = pd.concat(
    [num_feat_df, date_feat_df, cat_feat_df] + text_components, axis=1
)

X_train = all_features.iloc[:train_len].reset_index(drop=True)
X_test = all_features.iloc[train_len:].reset_index(drop=True)

# 4. Text Out-Of-Fold Logistic Regression Meta-Features
if len(text_cols) > 0:
    combined_text = combined_df[text_cols].fillna("").astype(str).agg(" ".join, axis=1)
    tfidf_full = TfidfVectorizer(
        max_features=30000, ngram_range=(1, 2), min_df=2, sublinear_tf=True
    )
    X_tfidf_train = tfidf_full.fit_transform(combined_text.iloc[:train_len])
    X_tfidf_test = tfidf_full.transform(combined_text.iloc[train_len:])

    skf_text = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_text_preds = np.zeros((train_len, num_classes))
    test_text_preds = np.zeros((len(test_df), num_classes))

    for tr_idx, val_idx in skf_text.split(X_tfidf_train, y):
        clf = LogisticRegression(C=1.0, max_iter=200, solver="lbfgs", n_jobs=-1)
        clf.fit(X_tfidf_train[tr_idx], y[tr_idx])
        oof_text_preds[val_idx] = clf.predict_proba(X_tfidf_train[val_idx])
        test_text_preds += clf.predict_proba(X_tfidf_test) / 5.0

    for c in range(num_classes):
        X_train[f"text_meta_prob_{c}"] = oof_text_preds[:, c]
        X_test[f"text_meta_prob_{c}"] = test_text_preds[:, c]

# 5. 5-Fold Stratified LightGBM Model
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_preds = np.zeros((train_len, num_classes))
test_preds = np.zeros((len(test_df), num_classes))

lgb_params = {
    "objective": "multiclass",
    "num_class": num_classes,
    "metric": "multi_logloss",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": 6,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "min_child_samples": 20,
    "random_state": 42,
    "verbosity": -1,
    "n_jobs": -1,
}

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y)):
    X_tr, y_tr = X_train.iloc[train_idx], y[train_idx]
    X_val, y_val = X_train.iloc[val_idx], y[val_idx]

    trn_data = lgb.Dataset(X_tr, label=y_tr)
    val_data = lgb.Dataset(X_val, label=y_val, reference=trn_data)

    model = lgb.train(
        lgb_params,
        trn_data,
        num_boost_round=1000,
        valid_sets=[trn_data, val_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(-1),
        ],
    )

    oof_preds[val_idx] = model.predict(X_val, num_iteration=model.best_iteration)
    test_preds += model.predict(X_test, num_iteration=model.best_iteration) / n_splits

oof_pred_labels = np.argmax(oof_preds, axis=1)
cv_accuracy = accuracy_score(y, oof_pred_labels)
print(f"5-Fold CV Accuracy: {cv_accuracy:.5f}")

# 6. Generate Submission
final_test_labels = le.inverse_transform(np.argmax(test_preds, axis=1))
os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame(
    {"id": np.arange(len(test_df)), "classification": final_test_labels}
)
submission.to_csv("./working/submission.csv", index=False)
print("Saved submission to ./working/submission.csv successfully.")
