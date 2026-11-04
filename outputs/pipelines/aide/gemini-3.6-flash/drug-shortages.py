import os
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

import lightgbm as lgb
from xgboost import XGBClassifier

try:
    from catboost import CatBoostClassifier

    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

# Directories
train_path = "./input/train.parquet"
test_path = "./input/test.parquet"
output_dir = "./working"
os.makedirs(output_dir, exist_ok=True)

# Load data
train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = "availability"
y_raw = train_df[target_col].astype(str)
le = LabelEncoder()
y = le.fit_transform(y_raw)
num_classes = len(le.classes_)

# Handle ID
if "id" in test_df.columns:
    test_ids = test_df["id"].values
else:
    test_ids = np.arange(len(test_df))

X_train = train_df.drop(columns=[target_col], errors="ignore")
X_test = test_df.copy()

id_cols = [c for c in X_train.columns if c.lower() in ["id", "index"]]
X_train = X_train.drop(columns=id_cols, errors="ignore")
X_test = X_test.drop(columns=id_cols, errors="ignore")

n_train = len(X_train)
df_all = pd.concat([X_train, X_test], ignore_index=True)

# Parse datetimes
date_cols = []
for col in df_all.columns:
    if pd.api.types.is_datetime64_any_dtype(df_all[col]):
        date_cols.append(col)
    elif df_all[col].dtype in ["object", "string"]:
        sample_non_null = df_all[col].dropna().astype(str).head(20)
        if (
            len(sample_non_null) > 0
            and sample_non_null.str.contains(r"\d{4}-\d{2}-\d{2}").all()
        ):
            try:
                df_all[col] = pd.to_datetime(df_all[col], errors="coerce")
                date_cols.append(col)
            except Exception:
                pass

feature_df = pd.DataFrame(index=df_all.index)

# Datetime feature extraction
for col in date_cols:
    dt_s = df_all[col]
    feature_df[f"{col}_year"] = dt_s.dt.year.fillna(-1)
    feature_df[f"{col}_month"] = dt_s.dt.month.fillna(-1)
    feature_df[f"{col}_day"] = dt_s.dt.day.fillna(-1)
    feature_df[f"{col}_dayofweek"] = dt_s.dt.dayofweek.fillna(-1)
    feature_df[f"{col}_dayofyear"] = dt_s.dt.dayofyear.fillna(-1)
    feature_df[f"{col}_quarter"] = dt_s.dt.quarter.fillna(-1)
    feature_df[f"{col}_timestamp"] = np.where(
        dt_s.notna(), dt_s.astype("int64") // 10**9, -1
    )

if len(date_cols) >= 2:
    for i in range(len(date_cols)):
        for j in range(i + 1, len(date_cols)):
            c1, c2 = date_cols[i], date_cols[j]
            feature_df[f"diff_{c1}_{c2}_days"] = (
                df_all[c1] - df_all[c2]
            ).dt.total_seconds().fillna(0) / (24 * 3600)

# String and numerical features
str_cols = [
    c
    for c in df_all.columns
    if c not in date_cols and df_all[c].dtype in ["object", "string", "category"]
]
num_cols = [c for c in df_all.columns if c not in date_cols and c not in str_cols]

for col in num_cols:
    med_val = df_all[col].median() if not df_all[col].isna().all() else 0
    feature_df[col] = df_all[col].fillna(med_val)
    feature_df[f"{col}_isna"] = df_all[col].isna().astype(int)

for col in str_cols:
    str_series = df_all[col].fillna("").astype(str)
    feature_df[f"{col}_len"] = str_series.str.len()
    feature_df[f"{col}_word_count"] = str_series.str.split().str.len()

    freq_map = str_series.value_counts(normalize=True).to_dict()
    feature_df[f"{col}_freq"] = str_series.map(freq_map).fillna(0)

    cat_le = LabelEncoder()
    feature_df[f"{col}_label"] = cat_le.fit_transform(str_series)

    # Column-specific TF-IDF + SVD text representations
    non_empty_count = (str_series.str.strip() != "").sum()
    if str_series.nunique() > 2 and non_empty_count > 10:
        try:
            tfidf = TfidfVectorizer(
                max_features=500, stop_words="english", min_df=2, ngram_range=(1, 2)
            )
            tfidf_mat = tfidf.fit_transform(str_series)
            if tfidf_mat.shape[1] > 2:
                n_comp = min(15, tfidf_mat.shape[1] - 1)
                svd = TruncatedSVD(n_components=n_comp, random_state=42)
                svd_feats = svd.fit_transform(tfidf_mat)
                for i in range(n_comp):
                    feature_df[f"svd_{col}_{i}"] = svd_feats[:, i]
        except Exception:
            pass

# Prepare training and test feature sets
X_tr_all = feature_df.iloc[:n_train].copy()
X_te_all = feature_df.iloc[n_train:].copy()

for c in X_tr_all.columns:
    X_tr_all[c] = pd.to_numeric(X_tr_all[c], errors="coerce").fillna(0)
    X_te_all[c] = pd.to_numeric(X_te_all[c], errors="coerce").fillna(0)

# Clean column names for LightGBM/XGBoost compatibility
clean_cols = [re.sub(r"[^\w]", "_", str(c)) for c in X_tr_all.columns]
X_tr_all.columns = clean_cols
X_te_all.columns = clean_cols

# Cross-validation scheme
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_probs = np.zeros((n_train, num_classes))
test_probs = np.zeros((len(X_te_all), num_classes))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_tr_all, y)):
    X_train_f, y_train_f = X_tr_all.iloc[train_idx], y[train_idx]
    X_val_f, y_val_f = X_tr_all.iloc[val_idx], y[val_idx]

    fold_val_probs = []
    fold_test_probs = []

    # 1. LightGBM
    lgb_clf = lgb.LGBMClassifier(
        n_estimators=700,
        learning_rate=0.05,
        max_depth=6,
        random_state=42 + fold,
        n_jobs=-1,
        verbose=-1,
        objective="multiclass" if num_classes > 2 else "binary",
    )
    lgb_clf.fit(
        X_train_f,
        y_train_f,
        eval_set=[(X_val_f, y_val_f)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )
    fold_val_probs.append(lgb_clf.predict_proba(X_val_f))
    fold_test_probs.append(lgb_clf.predict_proba(X_te_all))

    # 2. XGBoost
    xgb_clf = XGBClassifier(
        n_estimators=700,
        learning_rate=0.05,
        max_depth=6,
        random_state=42 + fold,
        eval_metric="mlogloss" if num_classes > 2 else "logloss",
        early_stopping_rounds=50,
        tree_method="hist",
        n_jobs=-1,
    )
    xgb_clf.fit(X_train_f, y_train_f, eval_set=[(X_val_f, y_val_f)], verbose=False)
    fold_val_probs.append(xgb_clf.predict_proba(X_val_f))
    fold_test_probs.append(xgb_clf.predict_proba(X_te_all))

    # 3. CatBoost (if available)
    if HAS_CATBOOST:
        cb_clf = CatBoostClassifier(
            iterations=700,
            learning_rate=0.05,
            depth=6,
            random_seed=42 + fold,
            verbose=0,
            loss_function="MultiClass" if num_classes > 2 else "Logloss",
        )
        cb_clf.fit(
            X_train_f,
            y_train_f,
            eval_set=(X_val_f, y_val_f),
            early_stopping_rounds=50,
            verbose=False,
        )
        fold_val_probs.append(cb_clf.predict_proba(X_val_f))
        fold_test_probs.append(cb_clf.predict_proba(X_te_all))

    # Blend fold predictions
    val_blend = np.mean(fold_val_probs, axis=0)
    test_blend = np.mean(fold_test_probs, axis=0)

    oof_probs[val_idx] = val_blend
    test_probs += test_blend / skf.n_splits

oof_preds = np.argmax(oof_probs, axis=1)
cv_accuracy = accuracy_score(y, oof_preds)
print(f"Validation Accuracy: {cv_accuracy:.5f}")

# Final predictions and submission file output
final_test_preds = np.argmax(test_probs, axis=1)
final_test_labels = le.inverse_transform(final_test_preds)

sub = pd.DataFrame({"id": test_ids, "availability": final_test_labels})
sub.to_csv(os.path.join(output_dir, "submission.csv"), index=False)
