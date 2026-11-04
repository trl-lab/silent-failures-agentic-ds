import os
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import lightgbm as lgb

warnings.filterwarnings("ignore")

# Load data
train_path = "./input/train.parquet"
test_path = "./input/test.parquet"

train = pd.read_parquet(train_path)
test = pd.read_parquet(test_path)

# Separate target
target_col = "risk_decision"
le = LabelEncoder()
y = le.fit_transform(train[target_col])
train_df = train.drop(columns=[target_col])
test_df = test.copy()

n_train = len(train_df)
df = pd.concat([train_df, test_df], ignore_index=True)

# Datetime extractions
for col in list(df.columns):
    if pd.api.types.is_datetime64_any_dtype(df[col]):
        df[f"{col}_year"] = df[col].dt.year
        df[f"{col}_month"] = df[col].dt.month
        df[f"{col}_day"] = df[col].dt.day
        df[f"{col}_dayofweek"] = df[col].dt.dayofweek
        df[f"{col}_dayofyear"] = df[col].dt.dayofyear
        df[f"{col}_quarter"] = df[col].dt.quarter
        df = df.drop(columns=[col])
    elif df[col].dtype == "object":
        try:
            converted = pd.to_datetime(df[col], errors="coerce")
            if converted.notna().sum() > 0.5 * len(df):
                df[f"{col}_year"] = converted.dt.year
                df[f"{col}_month"] = converted.dt.month
                df[f"{col}_day"] = converted.dt.day
                df[f"{col}_dayofweek"] = converted.dt.dayofweek
                df[f"{col}_dayofyear"] = converted.dt.dayofyear
                df[f"{col}_quarter"] = converted.dt.quarter
                df = df.drop(columns=[col])
        except Exception:
            pass

# Per-column frequency encoding, length metrics, and field-specific TF-IDF + SVD
cat_cols = []
for col in list(df.columns):
    if df[col].dtype == "object" or df[col].dtype.name == "category":
        df[col] = df[col].astype(str).fillna("")
        cat_cols.append(col)

        # Frequency encoding
        freq = df[col].value_counts(normalize=True)
        df[f"{col}_freq"] = df[col].map(freq)

        # Text length metrics
        df[f"{col}_len"] = df[col].str.len()
        df[f"{col}_word_count"] = df[col].str.split().str.len()

        # Per-column TF-IDF and SVD extraction
        unique_count = df[col].nunique()
        if unique_count > 2:
            n_comp = min(16, unique_count - 1)
            tfidf = TfidfVectorizer(
                max_features=5000, ngram_range=(1, 2), sublinear_tf=True
            )
            tfidf_mat = tfidf.fit_transform(df[col])
            if tfidf_mat.shape[1] > 1:
                actual_comp = min(n_comp, tfidf_mat.shape[1] - 1)
                if actual_comp > 0:
                    svd = TruncatedSVD(n_components=actual_comp, random_state=42)
                    svd_feats = svd.fit_transform(tfidf_mat)
                    for i in range(actual_comp):
                        df[f"{col}_svd_{i}"] = svd_feats[:, i]

for col in cat_cols:
    df[col] = df[col].astype("category")

train_feats = df.iloc[:n_train].copy()
test_feats = df.iloc[n_train:].copy()

# Cross-Validation setup
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
num_classes = len(le.classes_)
oof_preds = np.zeros((n_train, num_classes))
test_preds = np.zeros((len(test_feats), num_classes))

params = {
    "objective": "multiclass" if num_classes > 2 else "binary",
    "boosting_type": "gbdt",
    "learning_rate": 0.03,
    "num_leaves": 63,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "random_state": 42,
    "verbose": -1,
    "n_estimators": 1500,
}

if num_classes > 2:
    params["num_class"] = num_classes

for fold, (train_idx, val_idx) in enumerate(skf.split(train_feats, y)):
    X_train, y_train = train_feats.iloc[train_idx], y[train_idx]
    X_val, y_val = train_feats.iloc[val_idx], y[val_idx]

    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )

    oof_preds[val_idx] = model.predict_proba(X_val)
    test_preds += model.predict_proba(test_feats) / skf.n_splits

oof_pred_labels = oof_preds.argmax(axis=1)
acc = accuracy_score(y, oof_pred_labels)
print(f"Validation Metric (Accuracy): {acc:.5f}")

# Generate submission
test_pred_labels = test_preds.argmax(axis=1)
test_pred_classes = le.inverse_transform(test_pred_labels)

os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame(
    {"id": np.arange(len(test_feats)), "risk_decision": test_pred_classes}
)
submission.to_csv("./working/submission.csv", index=False)
