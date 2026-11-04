import os
import warnings
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")

# 1. Load Data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "AwardTotal"
y = train_df[target_col].values
train_df = train_df.drop(columns=[target_col])

# Combine train and test for consistent feature processing
n_train = len(train_df)
df = pd.concat([train_df, test_df], axis=0, ignore_index=True)

# 2. Feature Engineering
feature_cols = []
parsed_dates = {}

# Process date/time columns & pairwise date differences
for col in df.columns:
    if (
        pd.api.types.is_datetime64_any_dtype(df[col])
        or "date" in col.lower()
        or "year" in col.lower()
    ):
        try:
            dt_series = pd.to_datetime(df[col], errors="coerce")
            if dt_series.notna().sum() > 0.3 * len(df):
                parsed_dates[col] = dt_series
                df[f"{col}_year"] = dt_series.dt.year
                df[f"{col}_month"] = dt_series.dt.month
                df[f"{col}_day"] = dt_series.dt.day
                df[f"{col}_dayofweek"] = dt_series.dt.dayofweek
                feature_cols.extend(
                    [f"{col}_year", f"{col}_month", f"{col}_day", f"{col}_dayofweek"]
                )
        except Exception:
            pass

# Compute pairwise date differences (e.g. project duration in days)
date_cols = list(parsed_dates.keys())
for i in range(len(date_cols)):
    for j in range(i + 1, len(date_cols)):
        col1, col2 = date_cols[i], date_cols[j]
        diff_col = f"diff_days_{col1}_vs_{col2}"
        df[diff_col] = (parsed_dates[col2] - parsed_dates[col1]).dt.days
        feature_cols.append(diff_col)

# Identify numeric and string columns
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols = [c for c in numeric_cols if c not in feature_cols]

string_cols = df.select_dtypes(
    include=["object", "category", "string"]
).columns.tolist()

for col in string_cols:
    non_nulls = df[col].dropna()
    if len(non_nulls) == 0:
        continue

    # Extract text statistics for all string columns
    str_series = df[col].fillna("").astype(str)
    df[f"{col}_char_len"] = str_series.str.len()
    df[f"{col}_word_count"] = str_series.str.split().str.len()
    feature_cols.extend([f"{col}_char_len", f"{col}_word_count"])

    avg_len = non_nulls.astype(str).str.len().mean()

    if avg_len > 30:
        # Text column: TF-IDF with unigrams + bigrams and SVD dimensionality reduction
        try:
            tfidf = TfidfVectorizer(
                max_features=10000, ngram_range=(1, 2), stop_words="english"
            )
            tfidf_mat = tfidf.fit_transform(str_series)
            n_comp = min(25, tfidf_mat.shape[1] - 1)
            if n_comp > 1:
                svd = TruncatedSVD(n_components=n_comp, random_state=42)
                svd_feats = svd.fit_transform(tfidf_mat)
                for k in range(n_comp):
                    feat_name = f"{col}_svd_{k}"
                    df[feat_name] = svd_feats[:, k]
                    feature_cols.append(feat_name)
        except Exception:
            pass

    # Frequency encoding + Categorical conversion
    freq = df[col].value_counts().to_dict()
    df[f"{col}_freq"] = df[col].map(freq).fillna(0)
    feature_cols.append(f"{col}_freq")

    df[col] = df[col].astype(str).astype("category")
    feature_cols.append(col)

# Add remaining numeric columns
feature_cols.extend([c for c in numeric_cols if c not in feature_cols])
feature_cols = list(dict.fromkeys(feature_cols))

# Split back to train and test
X = df.iloc[:n_train][feature_cols].copy()
X_test = df.iloc[n_train:][feature_cols].copy()

# 3. Model Training with 5-Fold Cross-Validation and Out-of-Fold Target Encoding
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(n_train)
test_preds = np.zeros(len(X_test))

lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.03,
    "num_leaves": 63,
    "max_depth": -1,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "min_child_samples": 20,
    "random_state": 42,
    "n_estimators": 2000,
    "verbose": -1,
}

smoothing = 10.0

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_tr = X.iloc[train_idx].copy()
    y_tr = y[train_idx]
    X_va = X.iloc[val_idx].copy()
    y_va = y[val_idx]
    X_te = X_test.copy()

    fold_global_mean = float(np.mean(y_tr))

    # Out-of-fold target encoding for categorical columns using nested CV
    sub_kf = KFold(n_splits=5, shuffle=True, random_state=42 + fold)

    for col in string_cols:
        te_col = f"{col}_te"

        # Initialize columns with global mean
        X_tr[te_col] = fold_global_mean
        X_va[te_col] = fold_global_mean
        X_te[te_col] = fold_global_mean

        # Inner OOF encoding for train set
        for sub_tr_idx, sub_val_idx in sub_kf.split(X_tr, y_tr):
            sub_cats = X_tr[col].iloc[sub_tr_idx].astype(str)
            sub_y = y_tr[sub_tr_idx]

            df_sub = pd.DataFrame({"cat": sub_cats, "y": sub_y})
            stats = df_sub.groupby("cat")["y"].agg(["count", "mean"])
            smoothed = (
                stats["count"] * stats["mean"] + smoothing * fold_global_mean
            ) / (stats["count"] + smoothing)

            val_cats = X_tr[col].iloc[sub_val_idx].astype(str)
            mapped_vals = val_cats.map(smoothed).fillna(fold_global_mean).to_numpy()
            X_tr.loc[X_tr.index[sub_val_idx], te_col] = mapped_vals

        # Full fold train encoding for validation and test sets
        full_cats = X_tr[col].astype(str)
        df_full = pd.DataFrame({"cat": full_cats, "y": y_tr})
        stats_full = df_full.groupby("cat")["y"].agg(["count", "mean"])
        smoothed_full = (
            stats_full["count"] * stats_full["mean"] + smoothing * fold_global_mean
        ) / (stats_full["count"] + smoothing)

        X_va[te_col] = (
            X_va[col].astype(str).map(smoothed_full).fillna(fold_global_mean).to_numpy()
        )
        X_te[te_col] = (
            X_te[col].astype(str).map(smoothed_full).fillna(fold_global_mean).to_numpy()
        )

    model = lgb.LGBMRegressor(**lgb_params)
    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )

    val_pred = model.predict(X_va)
    oof_preds[val_idx] = val_pred
    test_preds += model.predict(X_te) / kf.n_splits

# Ensure non-negative predictions
oof_preds = np.clip(oof_preds, 0, None)
test_preds = np.clip(test_preds, 0, None)

# Calculate and print validation score
cv_rmse = root_mean_squared_error(y, oof_preds)
print(f"5-Fold CV RMSE: {cv_rmse:.4f}")

# 4. Save Predictions
os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame({"id": np.arange(len(test_preds)), "AwardTotal": test_preds})
submission.to_csv("./working/submission.csv", index=False)
