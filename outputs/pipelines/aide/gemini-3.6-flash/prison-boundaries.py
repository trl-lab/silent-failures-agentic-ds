import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import lightgbm as lgb

# Load datasets
train_path = "./input/train.parquet"
test_path = "./input/test.parquet"

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

# Drop missing target rows if any
target_col = "CAPACITY"
train_df = train_df.dropna(subset=[target_col]).reset_index(drop=True)

y = train_df[target_col].values
X = train_df.drop(columns=[target_col]).copy()
X_test = test_df.copy()

# Ensure matching columns between train and test
common_cols = [col for col in X.columns if col in X_test.columns]
X = X[common_cols]
X_test = X_test[common_cols]

# Datetime component extraction
for col in common_cols:
    if pd.api.types.is_datetime64_any_dtype(X[col]):
        for df in [X, X_test]:
            df[col + "_year"] = df[col].dt.year
            df[col + "_month"] = df[col].dt.month
            df[col + "_day"] = df[col].dt.day
        X.drop(columns=[col], inplace=True)
        X_test.drop(columns=[col], inplace=True)

# Combine train and test for uniform feature engineering
n_train = len(X)
df_all = pd.concat([X, X_test], ignore_index=True)

# Extract text features using TF-IDF + TruncatedSVD
text_candidate_cols = [c for c in df_all.columns if df_all[c].dtype == "object"]
if len(text_candidate_cols) > 0:
    combined_text = (
        df_all[text_candidate_cols].fillna("").astype(str).agg(" ".join, axis=1)
    )
    tfidf = TfidfVectorizer(max_features=5000, stop_words="english", ngram_range=(1, 2))
    tfidf_mat = tfidf.fit_transform(combined_text)

    svd = TruncatedSVD(n_components=15, random_state=42)
    svd_feats = svd.fit_transform(tfidf_mat)

    for i in range(15):
        df_all[f"text_svd_{i}"] = svd_feats[:, i]

# Categorical Frequency Encoding & Spatial Group Aggregations
cat_cols = [c for c in df_all.columns if df_all[c].dtype == "object"]
num_cols = [
    c
    for c in df_all.columns
    if pd.api.types.is_numeric_dtype(df_all[c]) and not c.startswith("text_svd_")
]

for c in cat_cols:
    freq = df_all[c].value_counts()
    df_all[f"{c}_freq"] = df_all[c].map(freq)

    # Compute spatial coordinate stats per categorical group if available
    for num_c in num_cols[:2]:
        grp = df_all.groupby(c)[num_c].agg(["mean", "std"]).reset_index()
        grp.columns = [c, f"{c}_{num_c}_mean", f"{c}_{num_c}_std"]
        df_all = df_all.merge(grp, on=c, how="left")

# Convert non-numeric columns to categorical type for LightGBM
for col in df_all.columns:
    if not pd.api.types.is_numeric_dtype(df_all[col]):
        df_all[col] = df_all[col].astype(str).astype("category")

# Separate back into train and test
X = df_all.iloc[:n_train].copy().reset_index(drop=True)
X_test = df_all.iloc[n_train:].copy().reset_index(drop=True)

# Setup 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(X))
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
    "n_jobs": -1,
}

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_train_fold, y_train_fold = X.iloc[train_idx], y[train_idx]
    X_val_fold, y_val_fold = X.iloc[val_idx], y[val_idx]

    model = lgb.LGBMRegressor(**lgb_params)
    model.fit(
        X_train_fold,
        y_train_fold,
        eval_set=[(X_val_fold, y_val_fold)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )

    val_preds = model.predict(X_val_fold)
    oof_preds[val_idx] = val_preds
    test_preds += model.predict(X_test) / kf.n_splits

# Compute CV RMSE
overall_rmse = np.sqrt(mean_squared_error(y, oof_preds))
print(f"5-Fold CV RMSE: {overall_rmse:.4f}")

# Save test predictions in submission format
os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame(
    {"id": np.arange(len(test_preds)), "CAPACITY": np.clip(test_preds, 0, None)}
)
submission.to_csv("./working/submission.csv", index=False)
