import os
import itertools
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

try:
    import xgboost as xgb

    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import catboost as cb

    HAS_CB = True
except ImportError:
    HAS_CB = False

# Ensure working directory exists
os.makedirs("./working", exist_ok=True)

# 1. Load Data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "Status"
# Binary target encoding
if (
    train_df[target_col].dtype == object
    or train_df[target_col].dtype.name == "category"
):
    unique_vals = sorted(train_df[target_col].dropna().unique())
    val_map = {val: i for i, val in enumerate(unique_vals)}
    y = train_df[target_col].map(val_map).values
else:
    unique_vals = np.sort(train_df[target_col].unique())
    if set(unique_vals).issubset({0, 1}):
        y = train_df[target_col].astype(int).values
    else:
        val_map = {val: i for i, val in enumerate(unique_vals)}
        y = train_df[target_col].map(val_map).values

n_train = len(train_df)
n_test = len(test_df)

X_train_raw = train_df.drop(columns=[target_col]).copy()
X_test_raw = test_df.copy()

combined_df = pd.concat([X_train_raw, X_test_raw], axis=0, ignore_index=True)

# 2. Advanced Feature Engineering
features_df = pd.DataFrame(index=combined_df.index)
parsed_dates = {}

for col in combined_df.columns:
    series = combined_df[col]

    # Handle numeric columns
    if pd.api.types.is_numeric_dtype(series):
        features_df[col] = series.fillna(series.median())
        features_df[f"{col}_isna"] = series.isna().astype(int)
        continue

    # Attempt date parsing
    date_parsed = pd.to_datetime(series, errors="coerce")
    non_null_date_ratio = date_parsed.notna().mean()
    if non_null_date_ratio > 0.4:
        parsed_dates[col] = date_parsed
        features_df[f"{col}_year"] = date_parsed.dt.year.fillna(-1)
        features_df[f"{col}_month"] = date_parsed.dt.month.fillna(-1)
        features_df[f"{col}_day"] = date_parsed.dt.day.fillna(-1)
        features_df[f"{col}_dayofweek"] = date_parsed.dt.dayofweek.fillna(-1)
        features_df[f"{col}_dayofyear"] = date_parsed.dt.dayofyear.fillna(-1)
        features_df[f"{col}_timestamp"] = np.where(
            date_parsed.notna(), date_parsed.astype("int64") // 10**9, -1
        )
        features_df[f"{col}_isna"] = date_parsed.isna().astype(int)
        continue

    # Categorical and Text Features
    str_series = (
        series.astype(str)
        .replace({"None": "missing", "nan": "missing"})
        .fillna("missing")
    )

    # Lexical / surface features
    features_df[f"{col}_len"] = str_series.apply(len)
    features_df[f"{col}_word_count"] = str_series.apply(lambda x: len(x.split()))
    features_df[f"{col}_num_digits"] = str_series.apply(
        lambda x: sum(c.isdigit() for c in x)
    )
    features_df[f"{col}_num_upper"] = str_series.apply(
        lambda x: sum(c.isupper() for c in x)
    )

    avg_len = features_df[f"{col}_len"].mean()
    n_unique = str_series.nunique()

    # TF-IDF + SVD for text columns
    if avg_len > 25 or n_unique > 150:
        tfidf = TfidfVectorizer(
            max_features=10000,
            stop_words="english",
            token_pattern=r"(?u)\b\w+\b",
            ngram_range=(1, 2),
            min_df=2,
        )
        try:
            tfidf_mat = tfidf.fit_transform(str_series)
            n_components = min(20, max(2, tfidf_mat.shape[1] - 1))
            if n_components > 1:
                svd = TruncatedSVD(n_components=n_components, random_state=42)
                svd_feats = svd.fit_transform(tfidf_mat)
                for i in range(n_components):
                    features_df[f"{col}_svd_{i}"] = svd_feats[:, i]
        except Exception:
            pass

    # Frequency encoding
    freq_map = str_series.value_counts(normalize=True).to_dict()
    features_df[f"{col}_freq"] = str_series.map(freq_map).fillna(0).astype(float)
    features_df[f"{col}_cat"] = str_series.astype("category").cat.codes

# Compute pairwise date differences (regulatory event durations)
date_cols = list(parsed_dates.keys())
for col1, col2 in itertools.combinations(date_cols, 2):
    diff_days = (parsed_dates[col1] - parsed_dates[col2]).dt.total_seconds() / 86400.0
    features_df[f"diff_days_{col1}_{col2}"] = diff_days.fillna(-9999)
    features_df[f"diff_days_{col1}_{col2}_isna"] = diff_days.isna().astype(int)

# Split back into train and test
X = features_df.iloc[:n_train].copy().fillna(0)
X_test = features_df.iloc[n_train:].copy().fillna(0)

# Replace infinite values if any
X.replace([np.inf, -np.inf], 0, inplace=True)
X_test.replace([np.inf, -np.inf], 0, inplace=True)

# 3. Model Training & Ensembling with 5-Fold Stratified CV
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_preds_lgb = np.zeros(n_train)
test_preds_lgb = np.zeros(n_test)

oof_preds_cb = np.zeros(n_train)
test_preds_cb = np.zeros(n_test)

oof_preds_xgb = np.zeros(n_train)
test_preds_xgb = np.zeros(n_test)

lgb_params = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "learning_rate": 0.03,
    "num_leaves": 31,
    "max_depth": 6,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "min_child_samples": 20,
    "random_state": 42,
    "verbosity": -1,
    "n_jobs": -1,
}

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_tr, y_tr = X.iloc[train_idx], y[train_idx]
    X_va, y_va = X.iloc[val_idx], y[val_idx]

    # LightGBM
    trn_data = lgb.Dataset(X_tr, label=y_tr)
    val_data = lgb.Dataset(X_va, label=y_va, reference=trn_data)
    model_lgb = lgb.train(
        lgb_params,
        trn_data,
        num_boost_round=1500,
        valid_sets=[trn_data, val_data],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )
    oof_preds_lgb[val_idx] = model_lgb.predict(
        X_va, num_iteration=model_lgb.best_iteration
    )
    test_preds_lgb += (
        model_lgb.predict(X_test, num_iteration=model_lgb.best_iteration) / n_splits
    )

    # CatBoost
    if HAS_CB:
        model_cb = cb.CatBoostClassifier(
            iterations=1500,
            learning_rate=0.04,
            depth=6,
            eval_metric="AUC",
            random_seed=42 + fold,
            verbose=False,
            early_stopping_rounds=50,
            thread_count=-1,
        )
        model_cb.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=False)
        oof_preds_cb[val_idx] = model_cb.predict_proba(X_va)[:, 1]
        test_preds_cb += model_cb.predict_proba(X_test)[:, 1] / n_splits

    # XGBoost
    if HAS_XGB:
        model_xgb = xgb.XGBClassifier(
            n_estimators=1500,
            learning_rate=0.03,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="auc",
            random_state=42 + fold,
            early_stopping_rounds=50,
            n_jobs=-1,
        )
        model_xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        oof_preds_xgb[val_idx] = model_xgb.predict_proba(X_va)[:, 1]
        test_preds_xgb += model_xgb.predict_proba(X_test)[:, 1] / n_splits

# Dynamic Ensembling Weights
models_oof = [oof_preds_lgb]
models_test = [test_preds_lgb]

if HAS_CB:
    models_oof.append(oof_preds_cb)
    models_test.append(test_preds_cb)

if HAS_XGB:
    models_oof.append(oof_preds_xgb)
    models_test.append(test_preds_xgb)

oof_ensemble = np.mean(models_oof, axis=0)
final_test_preds = np.mean(models_test, axis=0)

cv_roc_auc = roc_auc_score(y, oof_ensemble)
print(f"5-Fold CV ROC AUC: {cv_roc_auc:.5f}")

# 4. Save Submission
submission_df = pd.DataFrame({"id": np.arange(n_test), "Status": final_test_preds})
submission_df.to_csv("./working/submission.csv", index=False)
print("Submission successfully saved to ./working/submission.csv")
