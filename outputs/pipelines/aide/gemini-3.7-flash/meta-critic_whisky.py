import os
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from catboost import CatBoostRegressor

# 1. Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "Meta Critic"
y = train_df[target_col].values
X_train_raw = train_df.drop(columns=[target_col]).copy()
X_test_raw = test_df.copy()

# Ensure working directory exists
os.makedirs("./working", exist_ok=True)

# 2. Feature Engineering & Preprocessing
combined = pd.concat([X_train_raw, X_test_raw], axis=0, ignore_index=True)


# Helper function to extract numerical values from text columns (e.g., Age, ABV)
def extract_numeric(val):
    if pd.isna(val):
        return np.nan
    val_str = str(val)
    match = re.search(r"(\d+(?:\.\d+)?)", val_str)
    return float(match.group(1)) if match else np.nan


features_df = pd.DataFrame(index=combined.index)

# Process columns based on data types and content
text_cols = []
cat_cols = []

for col in combined.columns:
    col_data = combined[col]
    if pd.api.types.is_numeric_dtype(col_data):
        features_df[col] = col_data.fillna(col_data.median())
    else:
        # Check if the column is primarily numeric strings
        num_extracted = col_data.apply(extract_numeric)
        valid_ratio = num_extracted.notna().mean()
        if valid_ratio > 0.6:
            features_df[f"{col}_num"] = num_extracted.fillna(num_extracted.median())

        # Categorical / text handling
        n_unique = col_data.nunique(dropna=True)
        if n_unique <= 50:
            cat_cols.append(col)
            # Frequency encoding
            freq_map = col_data.value_counts(normalize=True).to_dict()
            features_df[f"{col}_freq"] = col_data.map(freq_map).fillna(0)
            # Factorize categorical
            features_df[f"{col}_cat"] = pd.factorize(col_data.astype(str))[0]
        else:
            text_cols.append(col)

# Text Feature Extraction using TF-IDF + TruncatedSVD
for col in text_cols:
    text_series = combined[col].fillna("").astype(str)
    tfidf = TfidfVectorizer(max_features=1000, ngram_range=(1, 2), stop_words="english")
    tfidf_mat = tfidf.fit_transform(text_series)
    n_components = min(10, tfidf_mat.shape[1] - 1) if tfidf_mat.shape[1] > 1 else 1
    if n_components > 0:
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        svd_mat = svd.fit_transform(tfidf_mat)
        for i in range(n_components):
            features_df[f"{col}_svd_{i}"] = svd_mat[:, i]

# Split processed features back to train and test
n_train = len(train_df)
X_train = features_df.iloc[:n_train].reset_index(drop=True)
X_test = features_df.iloc[n_train:].reset_index(drop=True)

# 3. Model Training & Evaluation with 5-Fold Cross Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_predictions = np.zeros(n_train)
test_predictions = np.zeros(len(X_test))

cb_params = {
    "iterations": 1000,
    "learning_rate": 0.03,
    "depth": 5,
    "loss_function": "RMSE",
    "eval_metric": "RMSE",
    "random_seed": 42,
    "verbose": 0,
    "early_stopping_rounds": 50,
}

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y)):
    X_tr, y_tr = X_train.iloc[train_idx], y[train_idx]
    X_va, y_val = X_train.iloc[val_idx], y[val_idx]

    model = CatBoostRegressor(**cb_params)
    model.fit(X_tr, y_tr, eval_set=(X_va, y_val), verbose=False)

    oof_preds = model.predict(X_va)
    oof_predictions[val_idx] = oof_preds
    test_predictions += model.predict(X_test) / kf.n_splits

cv_rmse = root_mean_squared_error(y, oof_predictions)
print(f"5-Fold Cross-Validation RMSE: {cv_rmse:.5f}")

# 4. Save test submission
submission = pd.DataFrame(
    {"id": np.arange(len(test_predictions)), "Meta Critic": test_predictions}
)
submission.to_csv("./working/submission.csv", index=False)
print("Submission successfully saved to ./working/submission.csv")
