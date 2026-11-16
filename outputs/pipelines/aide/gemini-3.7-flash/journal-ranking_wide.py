import os
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import lightgbm as lgb
import warnings

warnings.filterwarnings("ignore")

# 1. Load data
train_path = "./input/train.parquet"
test_path = "./input/test.parquet"

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = "H index"
y = train_df[target_col].values
X_train_raw = train_df.drop(columns=[target_col]).copy()
X_test_raw = test_df.copy()

n_train = len(X_train_raw)
combined = pd.concat([X_train_raw, X_test_raw], axis=0, ignore_index=True)


# 2. Preprocess features
def try_parse_numeric(series):
    if series.dtype in [np.float64, np.int64, np.float32, np.int32]:
        return series, True
    if series.dtype == "object":
        sample = series.dropna().astype(str)
        if len(sample) == 0:
            return series, False
        # Check if values look like numbers with comma decimals or percentages
        cleaned = sample.str.replace(",", ".").str.replace("%", "").str.strip()
        # Test conversion rate
        converted = pd.to_numeric(cleaned, errors="coerce")
        if converted.notnull().mean() > 0.8:
            full_cleaned = (
                series.astype(str)
                .str.replace(",", ".")
                .str.replace("%", "")
                .str.strip()
            )
            return pd.to_numeric(full_cleaned, errors="coerce"), True
    return series, False


processed_dfs = []
text_cols = []
cat_cols = []

for col in combined.columns:
    col_series = combined[col]
    parsed_num, is_num = try_parse_numeric(col_series)
    if is_num:
        processed_dfs.append(parsed_num.rename(col))
    else:
        # Check if it's text or categorical
        n_unique = col_series.nunique()
        if n_unique > 200 and col_series.astype(str).str.len().mean() > 15:
            text_cols.append(col)
        else:
            cat_cols.append(col)

feature_df = pd.concat(processed_dfs, axis=1) if processed_dfs else pd.DataFrame()

# Categorical columns
for col in cat_cols:
    feature_df[col] = combined[col].astype("category")

# Text columns: TF-IDF + SVD
for col in text_cols:
    texts = combined[col].fillna("").astype(str)
    tfidf = TfidfVectorizer(
        max_features=5000, stop_words="english", token_pattern=r"(?u)\b\w+\b"
    )
    tfidf_mat = tfidf.fit_transform(texts)
    n_components = min(16, max(2, tfidf_mat.shape[1] - 1))
    if n_components >= 2:
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        svd_feats = svd.fit_transform(tfidf_mat)
        for i in range(n_components):
            feature_df[f"{col}_svd_{i}"] = svd_feats[:, i]

X_train = feature_df.iloc[:n_train].reset_index(drop=True)
X_test = feature_df.iloc[n_train:].reset_index(drop=True)

# 3. Model Training with 5-Fold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

oof_preds = np.zeros(n_train)
test_preds = np.zeros(len(X_test))

lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "max_depth": -1,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "min_child_samples": 20,
    "random_state": 42,
    "verbosity": -1,
    "n_jobs": -1,
}

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y)):
    X_tr, y_tr = X_train.iloc[train_idx], y[train_idx]
    X_va, y_va = X_train.iloc[val_idx], y[val_idx]

    trn_data = lgb.Dataset(X_tr, label=y_tr)
    val_data = lgb.Dataset(X_va, label=y_va, reference=trn_data)

    model = lgb.train(
        lgb_params,
        trn_data,
        num_boost_round=2000,
        valid_sets=[trn_data, val_data],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
    )

    oof_preds[val_idx] = model.predict(X_va, num_iteration=model.best_iteration)
    test_preds += (
        model.predict(X_test, num_iteration=model.best_iteration) / kf.n_splits
    )

cv_rmse = root_mean_squared_error(y, oof_preds)
print(f"5-Fold CV RMSE: {cv_rmse:.4f}")

# 4. Generate Submission
os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame({"id": np.arange(len(test_preds)), "H index": test_preds})
submission.to_csv("./working/submission.csv", index=False)
print("Saved submission to ./working/submission.csv successfully.")
