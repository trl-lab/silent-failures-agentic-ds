import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

# 1. Load data
train_path = "./input/train.parquet"
test_path = "./input/test.parquet"
train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = "Federal Funds"
y = train_df[target_col].values
X = train_df.drop(columns=[target_col])
X_test = test_df.copy()


# 2. Enhanced Feature Engineering
def engineer_features(df_train, df_test):
    train_feat = df_train.copy()
    test_feat = df_test.copy()

    # Identify date columns and parse them
    date_cols = []
    parsed_train_dates = {}
    parsed_test_dates = {}

    for col in train_feat.columns:
        if (
            pd.api.types.is_datetime64_any_dtype(train_feat[col])
            or "date" in col.lower()
            or "year" in col.lower()
        ):
            try:
                dt_tr = pd.to_datetime(train_feat[col], errors="coerce")
                dt_te = pd.to_datetime(test_feat[col], errors="coerce")
                if dt_tr.notna().sum() > 0.1 * len(train_feat):
                    date_cols.append(col)
                    parsed_train_dates[col] = dt_tr
                    parsed_test_dates[col] = dt_te

                    train_feat[f"{col}_year"] = dt_tr.dt.year
                    train_feat[f"{col}_month"] = dt_tr.dt.month
                    train_feat[f"{col}_day"] = dt_tr.dt.day
                    train_feat[f"{col}_dayofweek"] = dt_tr.dt.dayofweek

                    test_feat[f"{col}_year"] = dt_te.dt.year
                    test_feat[f"{col}_month"] = dt_te.dt.month
                    test_feat[f"{col}_day"] = dt_te.dt.day
                    test_feat[f"{col}_dayofweek"] = dt_te.dt.dayofweek
                    train_feat.drop(columns=[col], inplace=True)
                    test_feat.drop(columns=[col], inplace=True)
            except Exception:
                pass

    # Date duration features between pairs of identified date columns
    for i in range(len(date_cols)):
        for j in range(i + 1, len(date_cols)):
            col1, col2 = date_cols[i], date_cols[j]
            diff_tr = (
                parsed_train_dates[col2] - parsed_train_dates[col1]
            ).dt.total_seconds() / 86400.0
            diff_te = (
                parsed_test_dates[col2] - parsed_test_dates[col1]
            ).dt.total_seconds() / 86400.0
            train_feat[f"duration_{col2}_minus_{col1}"] = diff_tr
            test_feat[f"duration_{col2}_minus_{col1}"] = diff_te

    # Process text and categorical columns
    obj_cols = [
        c
        for c in train_feat.columns
        if train_feat[c].dtype == "object"
        or isinstance(train_feat[c].dtype, pd.CategoricalDtype)
    ]

    for col in obj_cols:
        tr_series = train_feat[col].fillna("").astype(str)
        te_series = test_feat[col].fillna("").astype(str)

        # Frequency encoding
        freq = tr_series.value_counts(normalize=True).to_dict()
        train_feat[f"{col}_freq"] = tr_series.map(freq).fillna(0).astype(float)
        test_feat[f"{col}_freq"] = te_series.map(freq).fillna(0).astype(float)

        # Check if column is free-text (longer average string length)
        avg_len = tr_series.str.len().mean()
        if avg_len > 25:
            # Text length statistics
            train_feat[f"{col}_char_len"] = tr_series.str.len()
            test_feat[f"{col}_char_len"] = te_series.str.len()
            train_feat[f"{col}_word_count"] = tr_series.str.split().str.len()
            test_feat[f"{col}_word_count"] = te_series.str.split().str.len()

            # TF-IDF + SVD embeddings
            try:
                tfidf = TfidfVectorizer(
                    max_features=5000,
                    stop_words="english",
                    ngram_range=(1, 2),
                    min_df=3,
                )
                tfidf_tr = tfidf.fit_transform(tr_series)
                tfidf_te = tfidf.transform(te_series)

                n_comp = min(16, tfidf_tr.shape[1] - 1)
                if n_comp >= 2:
                    svd = TruncatedSVD(n_components=n_comp, random_state=42)
                    svd_tr = svd.fit_transform(tfidf_tr)
                    svd_te = svd.transform(tfidf_te)

                    for k in range(n_comp):
                        train_feat[f"{col}_svd_{k}"] = svd_tr[:, k]
                        test_feat[f"{col}_svd_{k}"] = svd_te[:, k]
            except Exception:
                pass

        # Native LightGBM category type
        train_feat[col] = (
            train_feat[col].astype(str).fillna("missing").astype("category")
        )
        test_feat[col] = test_feat[col].astype(str).fillna("missing").astype("category")

    # Handle boolean columns
    for col in train_feat.columns:
        if pd.api.types.is_bool_dtype(train_feat[col]):
            train_feat[col] = train_feat[col].astype(int)
            test_feat[col] = test_feat[col].astype(int)

    return train_feat, test_feat


X, X_test = engineer_features(X, X_test)

# Align feature columns
feature_cols = [c for c in X.columns if c in X_test.columns]
X = X[feature_cols]
X_test = X_test[feature_cols]

# 3. 5-Fold Cross Validation Setup
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))

lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.03,
    "num_leaves": 63,
    "max_depth": 8,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "verbose": -1,
    "random_state": 42,
    "n_jobs": -1,
}

# 4. Training Loop
for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_tr, y_tr = X.iloc[train_idx], y[train_idx]
    X_val, y_val = X.iloc[val_idx], y[val_idx]

    trn_data = lgb.Dataset(X_tr, label=y_tr)
    val_data = lgb.Dataset(X_val, label=y_val, reference=trn_data)

    model = lgb.train(
        lgb_params,
        trn_data,
        num_boost_round=3000,
        valid_sets=[trn_data, val_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=-1),
        ],
    )

    oof_preds[val_idx] = model.predict(X_val, num_iteration=model.best_iteration)
    test_preds += model.predict(X_test, num_iteration=model.best_iteration) / n_splits

# 5. Post-Processing & Evaluation
if (y >= 0).all():
    oof_preds = np.clip(oof_preds, a_min=0, a_max=None)
    test_preds = np.clip(test_preds, a_min=0, a_max=None)

cv_rmse = np.sqrt(mean_squared_error(y, oof_preds))
print(f"5-Fold CV RMSE: {cv_rmse:.4f}")

# 6. Save Submission
os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame(
    {"id": np.arange(len(test_preds)), "Federal Funds": test_preds}
)
submission.to_csv("./working/submission.csv", index=False)
print("Submission successfully saved to ./working/submission.csv")
