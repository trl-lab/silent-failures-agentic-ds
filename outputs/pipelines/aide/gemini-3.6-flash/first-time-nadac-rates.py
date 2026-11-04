import os
import re
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

# 1. Load Data
train_path = "./input/train.parquet"
test_path = "./input/test.parquet"

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = "NADAC Rate"
y_train = train_df[target_col].values

# Concatenate for uniform preprocessing
n_train = len(train_df)
combined_df = pd.concat(
    [
        train_df.drop(columns=[target_col], errors="ignore"),
        test_df.drop(columns=[target_col], errors="ignore"),
    ],
    axis=0,
    ignore_index=True,
)

# 2. Feature Engineering
features = pd.DataFrame(index=combined_df.index)

# Process Datetime columns
for col in combined_df.columns:
    if pd.api.types.is_datetime64_any_dtype(combined_df[col]) or "date" in col.lower():
        dt_col = pd.to_datetime(combined_df[col], errors="coerce")
        features[f"{col}_year"] = dt_col.dt.year.fillna(-1)
        features[f"{col}_month"] = dt_col.dt.month.fillna(-1)
        features[f"{col}_day"] = dt_col.dt.day.fillna(-1)
        features[f"{col}_dayofweek"] = dt_col.dt.dayofweek.fillna(-1)
        features[f"{col}_dayofyear"] = dt_col.dt.dayofyear.fillna(-1)
        min_date = dt_col.min()
        if pd.notna(min_date):
            features[f"{col}_elapsed_days"] = (dt_col - min_date).dt.days.fillna(-1)

# Process Categorical & Text columns
text_cols = []
cat_cols = []
num_cols = []

for col in combined_df.columns:
    if col in features.columns:
        continue
    if pd.api.types.is_numeric_dtype(combined_df[col]):
        features[col] = combined_df[col].fillna(
            combined_df[col].median() if not combined_df[col].dropna().empty else 0
        )
        num_cols.append(col)
    else:
        combined_df[col] = combined_df[col].astype(str).fillna("missing")
        avg_len = combined_df[col].str.len().mean()
        if avg_len > 15 or combined_df[col].nunique() > 200:
            text_cols.append(col)
        else:
            cat_cols.append(col)

# Extract numbers from text fields (e.g. drug strength/dosage)
for col in combined_df.columns:
    if not pd.api.types.is_numeric_dtype(combined_df[col]):
        str_series = combined_df[col].astype(str)
        extracted_num = str_series.str.extract(r"(\d+\.?\d*)")[0].astype(float)
        if extracted_num.notna().sum() > 0.1 * len(combined_df):
            features[f"{col}_extracted_num"] = extracted_num.fillna(0)

# Frequency Encoding for categorical columns
for col in cat_cols + text_cols:
    str_series = combined_df[col].astype(str)
    freq = str_series.value_counts(normalize=True).to_dict()
    features[f"{col}_freq"] = str_series.map(freq).fillna(0)

    # Label encoding as categorical string for CatBoost
    features[f"{col}_cat"] = str_series

# TF-IDF + Truncated SVD for text columns
for col in text_cols:
    corpus = combined_df[col].astype(str).tolist()
    vectorizer = TfidfVectorizer(
        max_features=1000, stop_words="english", token_pattern=r"(?u)\b\w+\b"
    )
    tfidf_mat = vectorizer.fit_transform(corpus)
    if tfidf_mat.shape[1] > 0:
        n_components = min(10, tfidf_mat.shape[1])
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        svd_mat = svd.fit_transform(tfidf_mat)
        for i in range(n_components):
            features[f"{col}_svd_{i}"] = svd_mat[:, i]

# Separate train and test features
X_train = features.iloc[:n_train].copy()
X_test = features.iloc[n_train:].copy()

# Identify categorical features for CatBoost
cat_feature_names = [c for c in X_train.columns if c.endswith("_cat")]

# 3. Model Training & Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(n_train)
test_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
    X_tr, y_tr = X_train.iloc[train_idx], y_train[train_idx]
    X_va, y_va = X_train.iloc[val_idx], y_train[val_idx]

    model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        eval_metric="RMSE",
        random_seed=42 + fold,
        cat_features=cat_feature_names,
        verbose=0,
    )

    model.fit(
        X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=50, verbose=False
    )

    oof_preds[val_idx] = model.predict(X_va)
    test_preds += model.predict(X_test) / kf.n_splits

# 4. Evaluation Metric
cv_rmse = root_mean_squared_error(y_train, oof_preds)
print(f"Validation RMSE: {cv_rmse:.5f}")

# 5. Save Submission File
os.makedirs("./working", exist_ok=True)
sub = pd.DataFrame(
    {
        "id": np.arange(len(test_preds)),
        "NADAC Rate": np.clip(test_preds, a_min=0, a_max=None),
    }
)
sub.to_csv("./working/submission.csv", index=False)
