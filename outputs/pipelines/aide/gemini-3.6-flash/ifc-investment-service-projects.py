import os
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import lightgbm as lgb

warnings.filterwarnings("ignore")

# 1. Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "Total IFC investment as approved by Board(Million - USD)"
y = train_df[target_col].values

# Extract features
X_train = train_df.drop(columns=[target_col])
X_test = test_df.copy()

if "id" in X_train.columns:
    X_train = X_train.drop(columns=["id"])
if "id" in X_test.columns:
    X_test = X_test.drop(columns=["id"])

# Combine train and test for consistent preprocessing
df_all = pd.concat([X_train, X_test], axis=0).reset_index(drop=True)
n_train = len(train_df)

processed_features = pd.DataFrame(index=df_all.index)

for col in df_all.columns:
    col_data = df_all[col]

    # Check for datetime parsing
    if pd.api.types.is_datetime64_any_dtype(col_data) or (
        "date" in col.lower() or "year" in col.lower()
    ):
        try:
            dt_col = pd.to_datetime(col_data, errors="coerce")
            if dt_col.notna().sum() > 0.3 * len(df_all):
                processed_features[f"{col}_year"] = dt_col.dt.year
                processed_features[f"{col}_month"] = dt_col.dt.month
                processed_features[f"{col}_day"] = dt_col.dt.day
                processed_features[f"{col}_quarter"] = dt_col.dt.quarter
                processed_features[f"{col}_dayofweek"] = dt_col.dt.dayofweek
                continue
        except Exception:
            pass

    # Numerical columns
    if pd.api.types.is_numeric_dtype(col_data):
        processed_features[col] = col_data
    else:
        # Categorical / Text columns
        col_str = col_data.astype(str).fillna("Missing")
        n_unique = col_str.nunique()

        if n_unique <= 200:
            # Categorical feature
            processed_features[col] = col_str.astype("category")
        else:
            # Free text field -> TF-IDF + SVD
            try:
                tfidf = TfidfVectorizer(max_features=5000, stop_words="english")
                tfidf_mat = tfidf.fit_transform(col_str)
                svd_components = min(10, max(1, n_unique - 1))
                svd = TruncatedSVD(n_components=svd_components, random_state=42)
                svd_feats = svd.fit_transform(tfidf_mat)
                for i in range(svd_feats.shape[1]):
                    processed_features[f"{col}_svd_{i}"] = svd_feats[:, i]
            except Exception:
                processed_features[col] = col_str.astype("category")

X_train_proc = processed_features.iloc[:n_train].reset_index(drop=True)
X_test_proc = processed_features.iloc[n_train:].reset_index(drop=True)

# Ensure categorical types for LightGBM
cat_cols = X_train_proc.select_dtypes(include=["category", "object"]).columns.tolist()
for c in cat_cols:
    X_train_proc[c] = X_train_proc[c].astype("category")
    X_test_proc[c] = X_test_proc[c].astype("category")

# K-Fold Cross Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(n_train)
test_preds = np.zeros(len(test_df))

params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "random_state": 42,
    "verbose": -1,
    "n_estimators": 1000,
}

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_proc, y)):
    X_tr, y_tr = X_train_proc.iloc[train_idx], y[train_idx]
    X_va, y_va = X_train_proc.iloc[val_idx], y[val_idx]

    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
    )

    val_pred = model.predict(X_va)
    oof_preds[val_idx] = val_pred
    test_preds += model.predict(X_test_proc) / kf.n_splits

# Lower bound predictions at 0
oof_preds = np.clip(oof_preds, 0, None)
test_preds = np.clip(test_preds, 0, None)

cv_rmse = np.sqrt(mean_squared_error(y, oof_preds))
print(f"Validation RMSE: {cv_rmse:.4f}")

# Save submission
os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame({"id": np.arange(len(test_df)), target_col: test_preds})
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved successfully. Shape:", submission.shape)
