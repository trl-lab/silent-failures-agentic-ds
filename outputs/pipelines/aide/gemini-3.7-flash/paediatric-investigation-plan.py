import os
import re
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")

# 1. Load Data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "Decision type"

# Encode target variable
le_target = LabelEncoder()
y_train = le_target.fit_transform(train_df[target_col].astype(str))
num_classes = len(le_target.classes_)

n_train = len(train_df)
combined_df = pd.concat(
    [train_df.drop(columns=[target_col]), test_df], axis=0
).reset_index(drop=True)

# 2. Automated Feature Engineering - Base Tabular & Temporal Features
engineered_features = pd.DataFrame(index=combined_df.index)

# Detect date columns
date_cols = []
for col in combined_df.columns:
    if combined_df[col].dtype == "object" or pd.api.types.is_datetime64_any_dtype(
        combined_df[col]
    ):
        sample_non_null = combined_df[col].dropna()
        if not sample_non_null.empty:
            try:
                parsed_dates = pd.to_datetime(
                    sample_non_null.iloc[:50], errors="coerce", format="mixed"
                )
                if parsed_dates.notnull().mean() > 0.8:
                    date_cols.append(col)
            except Exception:
                pass

for col in date_cols:
    dt_series = pd.to_datetime(combined_df[col], errors="coerce", format="mixed")
    engineered_features[f"{col}_year"] = dt_series.dt.year.fillna(-1).astype(float)
    engineered_features[f"{col}_month"] = dt_series.dt.month.fillna(-1).astype(float)
    engineered_features[f"{col}_day"] = dt_series.dt.day.fillna(-1).astype(float)
    engineered_features[f"{col}_dayofweek"] = dt_series.dt.dayofweek.fillna(-1).astype(
        float
    )
    engineered_features[f"{col}_dayofyear"] = dt_series.dt.dayofyear.fillna(-1).astype(
        float
    )
    engineered_features[f"{col}_timestamp"] = dt_series.astype("int64") // 10**9
    engineered_features.loc[dt_series.isnull(), f"{col}_timestamp"] = -1.0

if len(date_cols) > 1:
    for i in range(len(date_cols)):
        for j in range(i + 1, len(date_cols)):
            col1, col2 = date_cols[i], date_cols[j]
            d1 = pd.to_datetime(combined_df[col1], errors="coerce", format="mixed")
            d2 = pd.to_datetime(combined_df[col2], errors="coerce", format="mixed")
            diff_days = (d1 - d2).dt.days
            engineered_features[f"diff_days_{col1}_{col2}"] = diff_days.fillna(
                -9999
            ).astype(float)

remaining_cols = [c for c in combined_df.columns if c not in date_cols]
text_cols = []
cat_cols = []
num_cols = []

for col in remaining_cols:
    if pd.api.types.is_numeric_dtype(combined_df[col]):
        num_cols.append(col)
        engineered_features[col] = combined_df[col].fillna(-1).astype(float)
    else:
        unique_ratio = combined_df[col].nunique() / max(
            len(combined_df[col].dropna()), 1
        )
        avg_len = combined_df[col].astype(str).str.len().mean()
        if unique_ratio > 0.05 or avg_len > 30:
            text_cols.append(col)
        else:
            cat_cols.append(col)

# Text length statistics & SVD dense features
for col in text_cols:
    text_data = combined_df[col].fillna("").astype(str)
    tfidf_word = TfidfVectorizer(
        max_features=5000, ngram_range=(1, 2), token_pattern=r"(?u)\b\w+\b"
    )
    word_mat = tfidf_word.fit_transform(text_data)

    n_components = min(15, word_mat.shape[1] - 1) if word_mat.shape[1] > 15 else 1
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    word_svd = svd.fit_transform(word_mat)

    for i in range(n_components):
        engineered_features[f"{col}_word_svd_{i}"] = word_svd[:, i]

    engineered_features[f"{col}_char_len"] = text_data.str.len().astype(float)
    engineered_features[f"{col}_word_count"] = (
        text_data.str.split().str.len().astype(float)
    )

# Categorical Frequency and Label Encoding
for col in cat_cols:
    cat_series = combined_df[col].fillna("__MISSING__").astype(str)
    freq_map = cat_series.value_counts(normalize=True).to_dict()
    engineered_features[f"{col}_freq"] = cat_series.map(freq_map).astype(float)
    le_cat = LabelEncoder()
    engineered_features[f"{col}_encoded"] = le_cat.fit_transform(cat_series).astype(
        float
    )

# Build combined text corpus for out-of-fold high-dimensional text modeling
combined_text = pd.Series("", index=combined_df.index)
for col in text_cols:
    combined_text = combined_text + " " + combined_df[col].fillna("").astype(str)

train_text = combined_text.iloc[:n_train].values
test_text = combined_text.iloc[n_train:].values

X_train_base = engineered_features.iloc[:n_train].copy().reset_index(drop=True)
X_test_base = engineered_features.iloc[n_train:].copy().reset_index(drop=True)

# 3. Stratified 5-Fold Cross Validation with Text Meta-features and Model Ensembling
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_text_prob = np.zeros((n_train, num_classes))
test_text_prob = np.zeros((len(test_df), num_classes))

if len(text_cols) > 0:
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_base, y_train)):
        text_vec = TfidfVectorizer(
            ngram_range=(1, 3), max_features=25000, sublinear_tf=True
        )
        X_tr_tfidf = text_vec.fit_transform(train_text[train_idx])
        X_va_tfidf = text_vec.transform(train_text[val_idx])
        X_te_tfidf = text_vec.transform(test_text)

        clf = LogisticRegression(C=2.0, max_iter=500, random_state=42)
        clf.fit(X_tr_tfidf, y_train[train_idx])

        oof_text_prob[val_idx] = clf.predict_proba(X_va_tfidf)
        test_text_prob += clf.predict_proba(X_te_tfidf) / skf.n_splits

    X_train_full = X_train_base.copy()
    X_test_full = X_test_base.copy()

    for k in range(num_classes):
        X_train_full[f"text_meta_prob_{k}"] = oof_text_prob[:, k]
        X_test_full[f"text_meta_prob_{k}"] = test_text_prob[:, k]
else:
    X_train_full = X_train_base.copy()
    X_test_full = X_test_base.copy()

# Sanitize column names
clean_col_map = {
    col: re.sub(r"[^A-Za-z0-9_]+", "_", col) for col in X_train_full.columns
}
X_train_full = X_train_full.rename(columns=clean_col_map)
X_test_full = X_test_full.rename(columns=clean_col_map)

oof_ensemble_prob = np.zeros((n_train, num_classes))
test_ensemble_prob = np.zeros((len(test_df), num_classes))

xgb_params = {
    "objective": "multi:softprob",
    "num_class": num_classes,
    "eval_metric": "mlogloss",
    "learning_rate": 0.04,
    "max_depth": 6,
    "subsample": 0.85,
    "colsample_bytree": 0.8,
    "min_child_weight": 2,
    "n_estimators": 600,
    "random_state": 42,
    "n_jobs": -1,
    "early_stopping_rounds": 40,
}

lgb_params = {
    "objective": "multiclass",
    "num_class": num_classes,
    "metric": "multi_logloss",
    "learning_rate": 0.04,
    "num_leaves": 31,
    "subsample": 0.85,
    "colsample_bytree": 0.8,
    "n_estimators": 600,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}

cb_params = {
    "loss_function": "MultiClass",
    "eval_metric": "MultiClass",
    "learning_rate": 0.05,
    "depth": 6,
    "iterations": 600,
    "random_seed": 42,
    "verbose": False,
    "early_stopping_rounds": 40,
}

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_full, y_train)):
    X_tr, y_tr = X_train_full.iloc[train_idx], y_train[train_idx]
    X_va, y_va = X_train_full.iloc[val_idx], y_train[val_idx]

    # XGBoost
    model_xgb = xgb.XGBClassifier(**xgb_params)
    model_xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    preds_xgb_val = model_xgb.predict_proba(X_va)
    preds_xgb_test = model_xgb.predict_proba(X_test_full)

    # LightGBM
    model_lgb = lgb.LGBMClassifier(**lgb_params)
    model_lgb.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(0)],
    )
    preds_lgb_val = model_lgb.predict_proba(X_va)
    preds_lgb_test = model_lgb.predict_proba(X_test_full)

    # CatBoost
    model_cb = CatBoostClassifier(**cb_params)
    model_cb.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=False)
    preds_cb_val = model_cb.predict_proba(X_va)
    preds_cb_test = model_cb.predict_proba(X_test_full)

    # Blend fold predictions
    fold_val_blend = 0.40 * preds_xgb_val + 0.35 * preds_lgb_val + 0.25 * preds_cb_val
    fold_test_blend = (
        0.40 * preds_xgb_test + 0.35 * preds_lgb_test + 0.25 * preds_cb_test
    )

    oof_ensemble_prob[val_idx] = fold_val_blend
    test_ensemble_prob += fold_test_blend / skf.n_splits

# 4. Evaluation and Submission Generation
oof_preds = np.argmax(oof_ensemble_prob, axis=1)
oof_accuracy = accuracy_score(y_train, oof_preds)
print(f"OOF Validation Accuracy: {oof_accuracy:.5f}")

test_preds = np.argmax(test_ensemble_prob, axis=1)
test_pred_labels = le_target.inverse_transform(test_preds)

os.makedirs("./working", exist_ok=True)
submission_df = pd.DataFrame(
    {"id": np.arange(len(test_df)), target_col: test_pred_labels}
)
submission_df.to_csv("./working/submission.csv", index=False)
print("Submission saved successfully to ./working/submission.csv")
