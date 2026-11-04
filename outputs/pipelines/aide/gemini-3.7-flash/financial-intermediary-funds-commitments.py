import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import warnings

warnings.filterwarnings("ignore")

# Load datasets
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "Amount in USD"
y = train_df[target_col].values
train_df = train_df.drop(columns=[target_col])

# Combine for uniform preprocessing
n_train = len(train_df)
combined = pd.concat([train_df, test_df], ignore_index=True)

# Feature engineering
features = pd.DataFrame(index=combined.index)

for col in combined.columns:
    # Check if datetime
    if pd.api.types.is_datetime64_any_dtype(combined[col]) or (
        combined[col].dtype == "object"
        and combined[col].astype(str).str.match(r"^\d{4}-\d{2}-\d{2}").sum()
        > 0.5 * len(combined)
    ):
        dt_col = pd.to_datetime(combined[col], errors="coerce")
        features[f"{col}_year"] = dt_col.dt.year.fillna(-1).astype(int)
        features[f"{col}_month"] = dt_col.dt.month.fillna(-1).astype(int)
        features[f"{col}_day"] = dt_col.dt.day.fillna(-1).astype(int)
        features[f"{col}_dayofweek"] = dt_col.dt.dayofweek.fillna(-1).astype(int)
        features[f"{col}_dayofyear"] = dt_col.dt.dayofyear.fillna(-1).astype(int)
        features[f"{col}_ts"] = dt_col.astype("int64") // 10**9
        features[f"{col}_ts"] = features[f"{col}_ts"].replace(-9223372037, np.nan)
    elif pd.api.types.is_numeric_dtype(combined[col]):
        features[col] = combined[col].fillna(combined[col].median())
    elif pd.api.types.is_object_dtype(combined[col]) or pd.api.types.is_string_dtype(
        combined[col]
    ):
        s = combined[col].fillna("Missing").astype(str)
        # Check text length / unique ratio
        unique_cnt = s.nunique()
        avg_len = s.str.len().mean()

        # Categorical label encoding & frequency encoding
        features[f"{col}_freq"] = s.map(s.value_counts(normalize=True))
        features[f"{col}_code"] = pd.factorize(s)[0]

        # If long text descriptions, apply TF-IDF + SVD
        if avg_len > 25 and unique_cnt > 50:
            tfidf = TfidfVectorizer(
                max_features=500, stop_words="english", ngram_range=(1, 2)
            )
            tfidf_mat = tfidf.fit_transform(s)
            svd_dim = min(10, tfidf_mat.shape[1] - 1)
            if svd_dim > 1:
                svd = TruncatedSVD(n_components=svd_dim, random_state=42)
                svd_feats = svd.fit_transform(tfidf_mat)
                for i in range(svd_dim):
                    features[f"{col}_svd_{i}"] = svd_feats[:, i]

X_train = features.iloc[:n_train].reset_index(drop=True)
X_test = features.iloc[n_train:].reset_index(drop=True)

# 5-Fold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(n_train)
test_preds = np.zeros(len(X_test))

lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "n_estimators": 2000,
    "learning_rate": 0.03,
    "num_leaves": 31,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y)):
    X_tr, y_tr = X_train.iloc[train_idx], y[train_idx]
    X_va, y_val = X_train.iloc[val_idx], y[val_idx]

    model = lgb.LGBMRegressor(**lgb_params)
    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )

    oof_preds[val_idx] = model.predict(X_va)
    test_preds += model.predict(X_test) / kf.n_splits

cv_rmse = np.sqrt(mean_squared_error(y, oof_preds))
print(f"5-Fold CV RMSE: {cv_rmse:.4f}")

# Generate and save submission file
os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame({"id": np.arange(len(test_preds)), target_col: test_preds})
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved successfully.")
