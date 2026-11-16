import os
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
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "NADAC Per Unit"
y = train_df[target_col].values
train_df = train_df.drop(columns=[target_col])

# Combine for uniform preprocessing
n_train = len(train_df)
combined = pd.concat([train_df, test_df], axis=0, ignore_index=True)

# 2. Feature Engineering & Preprocessing
# Identify column types
date_cols = []
text_cols = []
cat_cols = []
num_cols = []

for col in combined.columns:
    if pd.api.types.is_datetime64_any_dtype(combined[col]):
        date_cols.append(col)
    elif combined[col].dtype == "object":
        # Check if it could be a date
        sample_vals = combined[col].dropna().astype(str).head(20)
        parsed = False
        try:
            pd.to_datetime(sample_vals)
            # If successful and looks like date
            combined[col] = pd.to_datetime(combined[col], errors="coerce")
            date_cols.append(col)
            parsed = True
        except Exception:
            pass

        if not parsed:
            # Check cardinality to classify as text vs categorical
            unique_count = combined[col].nunique()
            avg_len = (
                combined[col].dropna().astype(str).map(len).mean()
                if len(combined[col].dropna()) > 0
                else 0
            )
            if unique_count > 500 and avg_len > 15:
                text_cols.append(col)
            else:
                cat_cols.append(col)
    else:
        num_cols.append(col)

# Process date columns
for col in date_cols:
    combined[f"{col}_year"] = combined[col].dt.year
    combined[f"{col}_month"] = combined[col].dt.month
    combined[f"{col}_day"] = combined[col].dt.day
    combined[f"{col}_dayofweek"] = combined[col].dt.dayofweek
    combined[f"{col}_dayofyear"] = combined[col].dt.dayofyear
    combined[f"{col}_is_month_start"] = combined[col].dt.is_month_start.astype(float)
    combined[f"{col}_is_month_end"] = combined[col].dt.is_month_end.astype(float)
    combined[f"{col}_timestamp"] = combined[col].astype(np.int64) // 10**9
    combined.drop(columns=[col], inplace=True)

# Process text columns with TF-IDF + SVD
for col in text_cols:
    text_data = combined[col].fillna("").astype(str)
    tfidf = TfidfVectorizer(max_features=2000, ngram_range=(1, 2), stop_words="english")
    tfidf_mat = tfidf.fit_transform(text_data)
    svd_dims = min(15, tfidf_mat.shape[1] - 1) if tfidf_mat.shape[1] > 1 else 1
    if svd_dims > 0:
        svd = TruncatedSVD(n_components=svd_dims, random_state=42)
        svd_feats = svd.fit_transform(tfidf_mat)
        for d in range(svd_dims):
            combined[f"{col}_svd_{d}"] = svd_feats[:, d]
    combined.drop(columns=[col], inplace=True)

# Process categorical columns
for col in cat_cols:
    combined[col] = combined[col].astype(str).fillna("MISSING")
    # Frequency encoding
    freq = combined[col].value_counts(normalize=True)
    combined[f"{col}_freq"] = combined[col].map(freq).astype(float)
    # Convert to category for LightGBM
    combined[col] = combined[col].astype("category")

# Separate back into train and test
X = combined.iloc[:n_train].copy()
X_test = combined.iloc[n_train:].copy()

# 3. Model Training & Cross-Validation
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_preds = np.zeros(n_train)
test_preds = np.zeros(len(X_test))

lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "min_child_samples": 20,
    "n_estimators": 2000,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_train_f, y_train_f = X.iloc[train_idx], y[train_idx]
    X_val_f, y_val_f = X.iloc[val_idx], y[val_idx]

    model = lgb.LGBMRegressor(**lgb_params)
    model.fit(
        X_train_f,
        y_train_f,
        eval_set=[(X_val_f, y_val_f)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )

    oof_preds[val_idx] = model.predict(X_val_f)
    test_preds += model.predict(X_test) / n_splits

cv_rmse = root_mean_squared_error(y, oof_preds)
print(f"5-Fold CV RMSE: {cv_rmse:.5f}")

# 4. Save test predictions
os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame(
    {"id": np.arange(len(test_preds)), "NADAC Per Unit": test_preds}
)
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved successfully with shape:", submission.shape)
