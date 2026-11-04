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

# Set paths
INPUT_DIR = "./input"
WORKING_DIR = "./working"
os.makedirs(WORKING_DIR, exist_ok=True)

# Load data
train_df = pd.read_parquet(os.path.join(INPUT_DIR, "train.parquet"))
test_df = pd.read_parquet(os.path.join(INPUT_DIR, "test.parquet"))

target_col = "risk_decision"
le_target = LabelEncoder()
y_train = le_target.fit_transform(train_df[target_col].astype(str))
num_classes = len(le_target.classes_)

# Combine for uniform preprocessing
train_df = train_df.drop(columns=[target_col])
n_train = len(train_df)
df = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)

# Identify column types
date_cols = []
text_cols = []
cat_cols = []
num_cols = []

for col in df.columns:
    if col.lower() in ["id", "index"]:
        continue
    if np.issubdtype(df[col].dtype, np.datetime64):
        date_cols.append(col)
    elif df[col].dtype == "object" or df[col].dtype.name == "category":
        sample_non_null = df[col].dropna().astype(str).head(50)
        is_date = False
        if len(sample_non_null) > 0:
            sample_str = sample_non_null.iloc[0]
            if re.match(r"^\d{4}[-/]\d{2}[-/]\d{2}", sample_str) or re.match(
                r"^\d{2}[-/]\d{2}[-/]\d{4}", sample_str
            ):
                is_date = True
        if is_date:
            date_cols.append(col)
        else:
            avg_words = (
                sample_non_null.apply(lambda x: len(x.split())).mean()
                if len(sample_non_null) > 0
                else 0
            )
            n_unique = df[col].nunique()
            if avg_words > 4 or (avg_words > 2 and n_unique > 200):
                text_cols.append(col)
            else:
                cat_cols.append(col)
    elif np.issubdtype(df[col].dtype, np.number):
        num_cols.append(col)

feature_dfs = []

# Process Date features
for col in date_cols:
    dt = pd.to_datetime(df[col], errors="coerce")
    date_features = pd.DataFrame(
        {
            f"{col}_year": dt.dt.year.fillna(-1),
            f"{col}_month": dt.dt.month.fillna(-1),
            f"{col}_day": dt.dt.day.fillna(-1),
            f"{col}_dayofweek": dt.dt.dayofweek.fillna(-1),
            f"{col}_dayofyear": dt.dt.dayofyear.fillna(-1),
            f"{col}_quarter": dt.dt.quarter.fillna(-1),
            f"{col}_is_month_end": dt.dt.is_month_end.astype(int),
        }
    )
    feature_dfs.append(date_features)

# Add key categorical interaction features
if len(cat_cols) >= 2:
    for i in range(min(4, len(cat_cols))):
        for j in range(i + 1, min(4, len(cat_cols))):
            col_a, col_b = cat_cols[i], cat_cols[j]
            comb_name = f"{col_a}_X_{col_b}"
            df[comb_name] = df[col_a].astype(str) + "___" + df[col_b].astype(str)
            cat_cols.append(comb_name)

# Process Categorical features: Frequency encoding + Label encoding
cat_encoded_dfs = []
for col in cat_cols:
    s = df[col].astype(str).fillna("__MISSING__")
    freq = s.value_counts(normalize=True)
    freq_feat = s.map(freq).rename(f"{col}_freq")
    le = LabelEncoder()
    label_feat = pd.Series(le.fit_transform(s), name=f"{col}_le")
    cat_encoded_dfs.append(pd.concat([freq_feat, label_feat], axis=1))

if cat_encoded_dfs:
    feature_dfs.append(pd.concat(cat_encoded_dfs, axis=1))

# Process Numerical features
if num_cols:
    feature_dfs.append(df[num_cols].fillna(-999))

# Text TruncatedSVD Features
text_features_list = []
combined_text = pd.Series("", index=df.index)
for col in text_cols:
    col_str = df[col].astype(str).fillna("")
    combined_text = combined_text + " " + col_str

    tfidf = TfidfVectorizer(
        max_features=4000, stop_words="english", ngram_range=(1, 2), sublinear_tf=True
    )
    tfidf_mat = tfidf.fit_transform(col_str)
    n_svd = min(25, tfidf_mat.shape[1] - 1)
    if n_svd > 1:
        svd = TruncatedSVD(n_components=n_svd, random_state=42)
        svd_feats = svd.fit_transform(tfidf_mat)
        svd_df = pd.DataFrame(
            svd_feats, columns=[f"{col}_svd_{i}" for i in range(n_svd)], index=df.index
        )
        text_features_list.append(svd_df)

# Global combined text TF-IDF representation
tfidf_all = TfidfVectorizer(
    max_features=12000, stop_words="english", ngram_range=(1, 2), sublinear_tf=True
)
tfidf_all_mat = tfidf_all.fit_transform(combined_text)
n_svd = min(35, tfidf_all_mat.shape[1] - 1)
if n_svd > 1:
    svd = TruncatedSVD(n_components=n_svd, random_state=42)
    svd_all_feats = svd.fit_transform(tfidf_all_mat)
    svd_all_df = pd.DataFrame(
        svd_all_feats,
        columns=[f"all_text_svd_{i}" for i in range(n_svd)],
        index=df.index,
    )
    text_features_list.append(svd_all_df)

if text_features_list:
    feature_dfs.extend(text_features_list)

X_tab = pd.concat(feature_dfs, axis=1)

# Generate Out-Of-Fold linear model text probability features
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

tfidf_train_mat = tfidf_all_mat[:n_train]
tfidf_test_mat = tfidf_all_mat[n_train:]

oof_lr_probs = np.zeros((n_train, num_classes))
test_lr_probs = np.zeros((len(test_df), num_classes))

for fold, (train_idx, val_idx) in enumerate(skf.split(tfidf_train_mat, y_train)):
    clf = LogisticRegression(C=2.0, max_iter=250, random_state=42)
    clf.fit(tfidf_train_mat[train_idx], y_train[train_idx])
    oof_lr_probs[val_idx] = clf.predict_proba(tfidf_train_mat[val_idx])
    test_lr_probs += clf.predict_proba(tfidf_test_mat) / n_splits

lr_prob_cols = [f"lr_tfidf_prob_{c}" for c in range(num_classes)]
lr_oof_df = pd.DataFrame(oof_lr_probs, columns=lr_prob_cols)
lr_test_df = pd.DataFrame(test_lr_probs, columns=lr_prob_cols)
lr_all_df = pd.concat([lr_oof_df, lr_test_df], axis=0).reset_index(drop=True)

X_all = pd.concat([X_tab.reset_index(drop=True), lr_all_df], axis=1)
X_all.columns = [re.sub(r"[^\w\s]", "_", str(c)) for c in X_all.columns]

X_train = X_all.iloc[:n_train].reset_index(drop=True)
X_test = X_all.iloc[n_train:].reset_index(drop=True)

# 5-Fold Stratified Cross Validation with LightGBM
oof_preds = np.zeros((n_train, num_classes))
test_preds = np.zeros((len(X_test), num_classes))

lgb_params = {
    "objective": "multiclass",
    "num_class": num_classes,
    "metric": "multi_logloss",
    "boosting_type": "gbdt",
    "learning_rate": 0.04,
    "num_leaves": 47,
    "max_depth": -1,
    "feature_fraction": 0.75,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "min_child_samples": 15,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    X_tr, y_tr = X_train.iloc[train_idx], y_train[train_idx]
    X_va, y_va = X_train.iloc[val_idx], y_train[val_idx]

    trn_data = lgb.Dataset(X_tr, label=y_tr)
    val_data = lgb.Dataset(X_va, label=y_va, reference=trn_data)

    model = lgb.train(
        lgb_params,
        trn_data,
        num_boost_round=1500,
        valid_sets=[trn_data, val_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=60, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )

    val_preds = model.predict(X_va, num_iteration=model.best_iteration)
    oof_preds[val_idx] = val_preds
    test_preds += model.predict(X_test, num_iteration=model.best_iteration) / n_splits

oof_pred_labels = np.argmax(oof_preds, axis=1)
cv_accuracy = accuracy_score(y_train, oof_pred_labels)
print(f"Validation Accuracy: {cv_accuracy:.5f}")

# Generate test predictions and save submission
test_pred_labels = np.argmax(test_preds, axis=1)
test_pred_original = le_target.inverse_transform(test_pred_labels)

submission = pd.DataFrame(
    {"id": np.arange(len(test_df)), "risk_decision": test_pred_original}
)

submission_path = os.path.join(WORKING_DIR, "submission.csv")
submission.to_csv(submission_path, index=False)
print(
    f"Submission successfully saved to {submission_path} with {len(submission)} rows."
)
