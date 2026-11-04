import os
import itertools
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import lightgbm as lgb

# 1. Load Data
train_path = "./input/train.parquet"
test_path = "./input/test.parquet"

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = "days_to_complete"
y = train_df[target_col].values
train_df = train_df.drop(columns=[target_col])

# Combine for uniform preprocessing
n_train = len(train_df)
combined_df = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)

# 2. Feature Engineering
processed_dfs = []

# Detect Date/Datetime columns
date_cols = []
for col in combined_df.columns:
    if "date" in col.lower() or combined_df[col].dtype == "datetime64[ns]":
        date_cols.append(col)
    elif combined_df[col].dtype == "object":
        sample = combined_df[col].dropna().astype(str).head(50)
        if len(sample) > 0 and sample.str.match(r"^\d{4}-\d{2}-\d{2}").mean() > 0.8:
            date_cols.append(col)

# Parse date series and extract standard components
parsed_dates = {}
for col in date_cols:
    dt_series = pd.to_datetime(combined_df[col], errors="coerce")
    parsed_dates[col] = dt_series
    date_features = pd.DataFrame(
        {
            f"{col}_year": dt_series.dt.year,
            f"{col}_month": dt_series.dt.month,
            f"{col}_day": dt_series.dt.day,
            f"{col}_dayofweek": dt_series.dt.dayofweek,
            f"{col}_dayofyear": dt_series.dt.dayofyear,
            f"{col}_timestamp": dt_series.astype("int64") // 10**9,
        }
    )
    date_features.loc[dt_series.isna(), f"{col}_timestamp"] = np.nan
    processed_dfs.append(date_features)

# Pairwise Date Differences (Intervals in Days)
if len(date_cols) > 1:
    date_diff_dict = {}
    for col1, col2 in itertools.combinations(date_cols, 2):
        diff_days = (
            parsed_dates[col1] - parsed_dates[col2]
        ).dt.total_seconds() / 86400.0
        date_diff_dict[f"diff_days_{col1}_minus_{col2}"] = diff_days
    processed_dfs.append(pd.DataFrame(date_diff_dict))

remaining_cols = [c for c in combined_df.columns if c not in date_cols]

# Separate Text, Categorical, and Numeric features
text_cols = []
cat_cols = []
num_cols = []

for col in remaining_cols:
    if pd.api.types.is_numeric_dtype(combined_df[col]):
        num_cols.append(col)
    else:
        avg_len = (
            combined_df[col].dropna().astype(str).str.len().mean()
            if len(combined_df[col].dropna()) > 0
            else 0
        )
        n_unique = combined_df[col].nunique()
        if avg_len > 40 or (n_unique > 200 and avg_len > 20):
            text_cols.append(col)
        else:
            cat_cols.append(col)

# Process Numeric columns
if num_cols:
    processed_dfs.append(combined_df[num_cols].copy())

# Process Categorical columns (frequency & label encoding)
if cat_cols:
    cat_df = pd.DataFrame()
    for col in cat_cols:
        series_str = combined_df[col].astype(str).fillna("missing")
        cat_df[f"{col}_cat"] = series_str.astype("category").cat.codes
        freq = series_str.value_counts(normalize=True)
        cat_df[f"{col}_freq"] = series_str.map(freq)
    processed_dfs.append(cat_df)

# Process Text columns with TF-IDF + TruncatedSVD
for col in text_cols:
    text_data = combined_df[col].fillna("").astype(str)
    tfidf = TfidfVectorizer(max_features=5000, stop_words="english", ngram_range=(1, 2))
    tfidf_mat = tfidf.fit_transform(text_data)

    n_components = min(20, tfidf_mat.shape[1] - 1) if tfidf_mat.shape[1] > 20 else 2
    if n_components > 0:
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        svd_mat = svd.fit_transform(tfidf_mat)
        svd_df = pd.DataFrame(
            svd_mat, columns=[f"{col}_svd_{i}" for i in range(n_components)]
        )
        processed_dfs.append(svd_df)

    len_df = pd.DataFrame(
        {
            f"{col}_char_len": text_data.str.len(),
            f"{col}_word_count": text_data.apply(lambda x: len(x.split())),
        }
    )
    processed_dfs.append(len_df)

# Assemble feature matrix
X_all = pd.concat(processed_dfs, axis=1)
X_train = X_all.iloc[:n_train].reset_index(drop=True)
X_test = X_all.iloc[n_train:].reset_index(drop=True)

# 3. Model Training & 5-Fold Cross Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(n_train)
test_preds = np.zeros(len(X_test))

lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.03,
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
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
    )

    oof_preds[val_idx] = model.predict(X_va, num_iteration=model.best_iteration)
    test_preds += (
        model.predict(X_test, num_iteration=model.best_iteration) / kf.n_splits
    )

cv_rmse = root_mean_squared_error(y, oof_preds)
print(f"5-Fold CV RMSE: {cv_rmse:.4f}")

# 4. Save Submission
os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame(
    {"id": np.arange(len(test_preds)), "days_to_complete": test_preds}
)
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved successfully to ./working/submission.csv")
