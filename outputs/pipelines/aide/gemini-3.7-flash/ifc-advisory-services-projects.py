import os
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import lightgbm as lgb

# 1. Load Data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "Estimated Total Budget ($)"
y = train_df[target_col].values
X_train_raw = train_df.drop(columns=[target_col]).copy()
X_test_raw = test_df.copy()

combined = pd.concat([X_train_raw, X_test_raw], axis=0).reset_index(drop=True)
n_train = len(train_df)

# 2. Feature Engineering
# Identify potential date columns and parse them
date_cols = [
    col for col in combined.columns if "date" in col.lower() or "year" in col.lower()
]
parsed_dates = {}
for col in date_cols:
    try:
        parsed = pd.to_datetime(combined[col], errors="coerce")
        if parsed.notna().sum() > 0.1 * len(combined):
            combined[f"{col}_year"] = parsed.dt.year
            combined[f"{col}_month"] = parsed.dt.month
            combined[f"{col}_day"] = parsed.dt.day
            parsed_dates[col] = parsed
    except Exception:
        pass

# Compute duration between date pairs if available
date_keys = list(parsed_dates.keys())
for i in range(len(date_keys)):
    for j in range(i + 1, len(date_keys)):
        d1, d2 = parsed_dates[date_keys[i]], parsed_dates[date_keys[j]]
        diff = (d2 - d1).dt.days
        if diff.notna().sum() > 0.1 * len(combined):
            combined[f"duration_{date_keys[i]}_{date_keys[j]}"] = diff

# Identify text columns vs categorical vs numeric
text_cols = []
cat_cols = []
num_cols = []

for col in combined.columns:
    if col in date_cols:
        continue
    if pd.api.types.is_numeric_dtype(combined[col]):
        num_cols.append(col)
    else:
        avg_len = combined[col].dropna().astype(str).str.len().mean()
        n_unique = combined[col].nunique()
        if avg_len > 40 or (n_unique > 50 and avg_len > 25):
            text_cols.append(col)
        else:
            cat_cols.append(col)

# Text Feature Extraction via TF-IDF + SVD
svd_dfs = []
for col in text_cols:
    combined_text = combined[col].fillna("").astype(str)
    tfidf = TfidfVectorizer(max_features=5000, stop_words="english", ngram_range=(1, 2))
    tfidf_mat = tfidf.fit_transform(combined_text)

    n_comp = min(15, max(2, tfidf_mat.shape[1] - 1))
    svd = TruncatedSVD(n_components=n_comp, random_state=42)
    svd_feats = svd.fit_transform(tfidf_mat)
    svd_df = pd.DataFrame(svd_feats, columns=[f"svd_{col}_{k}" for k in range(n_comp)])
    svd_dfs.append(svd_df)

# Categorical Frequency Encoding & Label Encoding
cat_encoded_dfs = []
for col in cat_cols:
    s = combined[col].astype(str).fillna("missing")
    freq = s.map(s.value_counts(normalize=True)).rename(f"{col}_freq")
    le = pd.Categorical(s).codes
    le_s = pd.Series(le, name=f"{col}_cat")
    cat_encoded_dfs.append(freq)
    cat_encoded_dfs.append(le_s)

# Combine all engineered features
processed_parts = [combined[num_cols]]
if cat_encoded_dfs:
    processed_parts.append(pd.concat(cat_encoded_dfs, axis=1))
if svd_dfs:
    processed_parts.append(pd.concat(svd_dfs, axis=1))

X_all = pd.concat(processed_parts, axis=1)

# Clean column names for LightGBM
X_all.columns = [re.sub(r"[^\w_]", "_", str(c)) for c in X_all.columns]

X_train = X_all.iloc[:n_train].reset_index(drop=True)
X_test = X_all.iloc[n_train:].reset_index(drop=True)

# 3. K-Fold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(X_train))
test_preds = np.zeros(len(X_test))

lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "n_estimators": 1000,
    "random_state": 42,
    "verbose": -1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
}

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y)):
    X_tr, y_tr = X_train.iloc[train_idx], y[train_idx]
    X_val, y_val = X_train.iloc[val_idx], y[val_idx]

    model = lgb.LGBMRegressor(**lgb_params)
    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )

    val_pred = model.predict(X_val)
    oof_preds[val_idx] = val_pred
    test_preds += model.predict(X_test) / kf.n_splits

# 4. Evaluation
cv_rmse = root_mean_squared_error(y, oof_preds)
print(f"5-Fold CV RMSE: {cv_rmse:.4f}")

# 5. Generate Submission
os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame(
    {
        "id": np.arange(len(test_preds)),
        target_col: np.clip(test_preds, a_min=0, a_max=None),
    }
)
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved successfully to ./working/submission.csv")
