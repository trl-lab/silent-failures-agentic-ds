import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import accuracy_score
import lightgbm as lgb

os.makedirs("./working", exist_ok=True)

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Target processing
target_col = "outcomes"
le = LabelEncoder()
y = le.fit_transform(train_df[target_col].astype(str))
num_classes = len(le.classes_)

# Exclude target and ID columns from feature set
ignore_cols = {target_col, "id", "ID"}
feature_cols = [c for c in train_df.columns if c not in ignore_cols]

# Combine for feature engineering
n_train = len(train_df)
df_all = pd.concat(
    [
        train_df[feature_cols],
        test_df[[c for c in feature_cols if c in test_df.columns]],
    ],
    axis=0,
).reset_index(drop=True)

processed_dfs = []

for col in feature_cols:
    if col not in df_all.columns:
        continue
    s = df_all[col]

    # Datetime features
    if (
        pd.api.types.is_datetime64_any_dtype(s)
        or "date" in col.lower()
        or "time" in col.lower()
    ):
        try:
            dt_s = pd.to_datetime(s, errors="coerce")
            if dt_s.notna().sum() > 0.5 * len(dt_s):
                dt_df = pd.DataFrame(
                    {
                        f"{col}_year": dt_s.dt.year,
                        f"{col}_month": dt_s.dt.month,
                        f"{col}_day": dt_s.dt.day,
                        f"{col}_dayofweek": dt_s.dt.dayofweek,
                    }
                )
                processed_dfs.append(dt_df)
                continue
        except Exception:
            pass

    # Numerical features
    if pd.api.types.is_numeric_dtype(s):
        processed_dfs.append(pd.DataFrame({col: s}))
        continue

    # Categorical and text features
    s_str = s.fillna("").astype(str)
    n_unique = s_str.nunique()
    avg_len = s_str.str.len().mean()

    if n_unique <= 200 or avg_len < 30:
        cat_codes = s_str.astype("category").cat.codes
        freq_map = s_str.value_counts(normalize=True).to_dict()
        freq_encoded = s_str.map(freq_map).fillna(0)
        cat_df = pd.DataFrame({f"{col}_code": cat_codes, f"{col}_freq": freq_encoded})
        processed_dfs.append(cat_df)
    else:
        tfidf = TfidfVectorizer(
            max_features=5000, stop_words="english", token_pattern=r"(?u)\b\w+\b"
        )
        tfidf_mat = tfidf.fit_transform(s_str)
        n_comp = min(30, tfidf_mat.shape[1] - 1) if tfidf_mat.shape[1] > 1 else 0
        if n_comp > 0:
            svd = TruncatedSVD(n_components=n_comp, random_state=42)
            svd_mat = svd.fit_transform(tfidf_mat)
            svd_df = pd.DataFrame(
                svd_mat, columns=[f"{col}_svd_{i}" for i in range(n_comp)]
            )
            processed_dfs.append(svd_df)

X_all = pd.concat(processed_dfs, axis=1)
X_train = X_all.iloc[:n_train].reset_index(drop=True)
X_test = X_all.iloc[n_train:].reset_index(drop=True)

# 5-fold Stratified K-Fold CV
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros((n_train, num_classes)) if num_classes > 2 else np.zeros(n_train)
test_preds = (
    np.zeros((len(X_test), num_classes)) if num_classes > 2 else np.zeros(len(X_test))
)

params = {
    "boosting_type": "gbdt",
    "random_state": 42,
    "n_estimators": 1000,
    "learning_rate": 0.05,
    "verbose": -1,
    "n_jobs": -1,
}

if num_classes == 2:
    params["objective"] = "binary"
    params["metric"] = "binary_logloss"
else:
    params["objective"] = "multiclass"
    params["num_class"] = num_classes
    params["metric"] = "multi_logloss"

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y)):
    X_tr, y_tr = X_train.iloc[train_idx], y[train_idx]
    X_va, y_va = X_train.iloc[val_idx], y[val_idx]

    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )

    if num_classes == 2:
        oof_preds[val_idx] = model.predict_proba(X_va)[:, 1]
        test_preds += model.predict_proba(X_test)[:, 1] / skf.n_splits
    else:
        oof_preds[val_idx] = model.predict_proba(X_va)
        test_preds += model.predict_proba(X_test) / skf.n_splits

if num_classes == 2:
    oof_pred_labels = (oof_preds >= 0.5).astype(int)
    test_pred_labels = (test_preds >= 0.5).astype(int)
else:
    oof_pred_labels = np.argmax(oof_preds, axis=1)
    test_pred_labels = np.argmax(test_preds, axis=1)

cv_accuracy = accuracy_score(y, oof_pred_labels)
print(f"Validation Accuracy: {cv_accuracy:.5f}")

# Inverse transform labels and create submission
predicted_outcomes = le.inverse_transform(test_pred_labels)

sub = pd.DataFrame({"id": np.arange(len(test_df)), "outcomes": predicted_outcomes})
sub.to_csv("./working/submission.csv", index=False)
