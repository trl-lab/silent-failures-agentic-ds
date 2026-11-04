import os
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import lightgbm as lgb

warnings.filterwarnings("ignore")

# 1. Load Data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "num_countries_affected"
y_raw = train_df[target_col]

# Handle target formatting for binary classification
if y_raw.dtype == "bool":
    y = y_raw.astype(int)
elif y_raw.nunique() > 2:
    if np.issubdtype(y_raw.dtype, np.number):
        if set(np.unique(y_raw.dropna())).issubset({0, 1, 0.0, 1.0}):
            y = y_raw.astype(int)
        else:
            y = (y_raw > y_raw.median()).astype(int)
    else:
        labels = y_raw.unique()
        y = (y_raw == labels[0]).astype(int)
else:
    if not np.issubdtype(y_raw.dtype, np.number):
        unique_vals = y_raw.dropna().unique()
        y = (y_raw == unique_vals[1]).astype(int)
    else:
        y = y_raw.astype(int)

train_df = train_df.drop(columns=[target_col])

# Combine for uniform processing
df_all = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)

features = []
categorical_cols = []

for col in df_all.columns:
    if col.lower() in ["id", "index"]:
        continue

    # Process datetime columns
    if "date" in col.lower() or "time" in col.lower():
        try:
            dt = pd.to_datetime(df_all[col], errors="coerce")
            if dt.notna().sum() > 0.5 * len(df_all):
                df_all[f"{col}_year"] = dt.dt.year
                df_all[f"{col}_month"] = dt.dt.month
                df_all[f"{col}_day"] = dt.dt.day
                df_all[f"{col}_dayofweek"] = dt.dt.dayofweek
                df_all[f"{col}_dayofyear"] = dt.dt.dayofyear
                features.extend(
                    [
                        f"{col}_year",
                        f"{col}_month",
                        f"{col}_day",
                        f"{col}_dayofweek",
                        f"{col}_dayofyear",
                    ]
                )
                continue
        except Exception:
            pass

    # Process text / categorical columns
    if (
        df_all[col].dtype == "object"
        or df_all[col].dtype.name == "category"
        or df_all[col].dtype == "bool"
    ):
        avg_len = df_all[col].fillna("").astype(str).str.len().mean()
        nunique = df_all[col].nunique()

        if avg_len > 30 and nunique > 100:
            text_data = df_all[col].fillna("").astype(str)
            tfidf = TfidfVectorizer(
                max_features=2000, stop_words="english", token_pattern=r"(?u)\b\w+\b"
            )
            tfidf_mat = tfidf.fit_transform(text_data)
            n_comps = min(25, tfidf_mat.shape[1] - 1)
            if n_comps > 1:
                svd = TruncatedSVD(n_components=n_comps, random_state=42)
                svd_feat = svd.fit_transform(tfidf_mat)
                for i in range(n_comps):
                    col_name = f"{col}_svd_{i}"
                    df_all[col_name] = svd_feat[:, i]
                    features.append(col_name)
        else:
            df_all[col] = df_all[col].astype("category")
            categorical_cols.append(col)
            features.append(col)
    else:
        features.append(col)

# Add frequency encoding for categoricals
for col in categorical_cols:
    freq_col = f"{col}_freq"
    df_all[freq_col] = df_all[col].map(df_all[col].value_counts(normalize=True))
    features.append(freq_col)

# Separate back to train and test
X = df_all.iloc[: len(train_df)][features].reset_index(drop=True)
X_test = df_all.iloc[len(train_df) :][features].reset_index(drop=True)

# 2. Cross-Validation and Model Training
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))

lgb_params = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "random_state": 42,
    "n_estimators": 1000,
    "verbose": -1,
}

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
    X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]

    model = lgb.LGBMClassifier(**lgb_params)
    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
    )

    oof_preds[val_idx] = model.predict_proba(X_va)[:, 1]
    test_preds += model.predict_proba(X_test)[:, 1] / skf.n_splits

cv_auc = roc_auc_score(y, oof_preds)
print(f"5-Fold Cross-Validation ROC AUC: {cv_auc:.5f}")

# 3. Save Submission
os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame(
    {"id": np.arange(len(test_preds)), "num_countries_affected": test_preds}
)
submission.to_csv("./working/submission.csv", index=False)
