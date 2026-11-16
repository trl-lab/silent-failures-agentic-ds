import os
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import lightgbm as lgb
from catboost import CatBoostClassifier

# 1. Load Data
train_path = "./input/train.parquet"
test_path = "./input/test.parquet"
output_dir = "./working"
os.makedirs(output_dir, exist_ok=True)

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = "Task Assigned"

# Process target
if (
    train_df[target_col].dtype == object
    or train_df[target_col].dtype.name == "category"
):
    unique_vals = sorted(train_df[target_col].dropna().unique())
    val_map = {val: i for i, val in enumerate(unique_vals)}
    y = train_df[target_col].map(val_map).astype(int).values
else:
    vals = np.sort(train_df[target_col].dropna().unique())
    if set(vals).issubset({1, 2}):
        y = (train_df[target_col].values == 2).astype(int)
    elif len(vals) == 2:
        y = (train_df[target_col].values == vals[1]).astype(int)
    else:
        y = (train_df[target_col].values > np.median(vals)).astype(int)

train_df = train_df.drop(columns=[target_col])
n_train = len(train_df)
combined = pd.concat([train_df, test_df], axis=0, ignore_index=True)

# 2. Feature Engineering
features_df = pd.DataFrame(index=combined.index)

# Detect and parse date columns
date_cols = []
for col in combined.columns:
    if "date" in col.lower() or "time" in col.lower():
        try:
            parsed = pd.to_datetime(combined[col], errors="coerce")
            if parsed.notna().sum() > 0.3 * len(combined):
                date_cols.append(col)
                features_df[f"{col}_year"] = parsed.dt.year.fillna(-1).astype(int)
                features_df[f"{col}_month"] = parsed.dt.month.fillna(-1).astype(int)
                features_df[f"{col}_day"] = parsed.dt.day.fillna(-1).astype(int)
                features_df[f"{col}_dayofweek"] = parsed.dt.dayofweek.fillna(-1).astype(
                    int
                )
                features_df[f"{col}_is_weekend"] = parsed.dt.dayofweek.isin(
                    [5, 6]
                ).astype(int)
        except Exception:
            pass

remaining_cols = [c for c in combined.columns if c not in date_cols]
text_cols = []
cat_cols = []
num_cols = []

for col in remaining_cols:
    if pd.api.types.is_numeric_dtype(combined[col]):
        num_cols.append(col)
    else:
        avg_len = combined[col].astype(str).str.len().mean()
        nunique = combined[col].nunique()
        if avg_len > 35 and nunique > 50:
            text_cols.append(col)
        else:
            cat_cols.append(col)

# Numeric features
for col in num_cols:
    features_df[f"{col}_isna"] = combined[col].isna().astype(int)
    median_val = combined[col].median()
    features_df[col] = combined[col].fillna(median_val if pd.notna(median_val) else 0)

# Categorical features: Frequency encoding + Label encoding
for col in cat_cols:
    val_counts = combined[col].value_counts(dropna=False)
    features_df[f"{col}_freq"] = combined[col].map(val_counts).fillna(0).astype(int)
    cat_series = combined[col].astype(str).fillna("__MISSING__")
    features_df[f"{col}_encoded"] = pd.factorize(cat_series)[0]

# Text features: Length statistics + TF-IDF with TruncatedSVD
for col in text_cols:
    text_data = combined[col].fillna("").astype(str)
    features_df[f"{col}_len"] = text_data.str.len()
    features_df[f"{col}_word_count"] = text_data.apply(lambda x: len(x.split()))

    tfidf = TfidfVectorizer(
        max_features=5000, ngram_range=(1, 2), stop_words="english", min_df=2
    )
    tfidf_mat = tfidf.fit_transform(text_data)

    n_components = min(16, tfidf_mat.shape[1] - 1) if tfidf_mat.shape[1] > 1 else 0
    if n_components > 0:
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        svd_mat = svd.fit_transform(tfidf_mat)
        for i in range(n_components):
            features_df[f"{col}_svd_{i}"] = svd_mat[:, i]

# Split back to train and test
X_train = features_df.iloc[:n_train].copy().reset_index(drop=True)
X_test = features_df.iloc[n_train:].copy().reset_index(drop=True)

# Out-of-fold Target Encoding for categoricals
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
global_mean = float(np.mean(y))

for col in cat_cols:
    enc_col = f"{col}_encoded"
    X_train[f"{col}_te"] = np.nan
    X_test[f"{col}_te"] = 0.0

    test_cat = X_test[enc_col]
    test_te_acc = np.zeros(len(X_test))

    for tr_idx, val_idx in skf.split(X_train, y):
        tr_df = pd.DataFrame(
            {"cat": X_train.iloc[tr_idx][enc_col].values, "target": y[tr_idx]}
        )
        stats = tr_df.groupby("cat")["target"].agg(["count", "mean"])

        smooth = 10.0
        te_map = (
            (stats["count"] * stats["mean"] + smooth * global_mean)
            / (stats["count"] + smooth)
        ).to_dict()

        val_cat = X_train.iloc[val_idx][enc_col]
        X_train.loc[val_idx, f"{col}_te"] = (
            val_cat.map(te_map).fillna(global_mean).values
        )
        test_te_acc += test_cat.map(te_map).fillna(global_mean).values / 5.0

    X_test[f"{col}_te"] = test_te_acc

# 3. Model Training & Cross-Validation with LightGBM and CatBoost
oof_preds_lgb = np.zeros(n_train)
test_preds_lgb = np.zeros(len(X_test))
oof_preds_cat = np.zeros(n_train)
test_preds_cat = np.zeros(len(X_test))

lgb_params = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.03,
    "num_leaves": 31,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "random_state": 42,
    "verbosity": -1,
    "n_jobs": -1,
}

cat_params = {
    "iterations": 800,
    "learning_rate": 0.04,
    "depth": 5,
    "loss_function": "Logloss",
    "eval_metric": "AUC",
    "random_seed": 42,
    "verbose": False,
    "early_stopping_rounds": 50,
}

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y)):
    X_tr, y_tr = X_train.iloc[train_idx], y[train_idx]
    X_va, y_va = X_train.iloc[val_idx], y[val_idx]

    # LightGBM
    dtrain = lgb.Dataset(X_tr, label=y_tr)
    dval = lgb.Dataset(X_va, label=y_va, reference=dtrain)
    lgb_model = lgb.train(
        lgb_params,
        dtrain,
        num_boost_round=1000,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )
    oof_preds_lgb[val_idx] = lgb_model.predict(
        X_va, num_iteration=lgb_model.best_iteration
    )
    test_preds_lgb += (
        lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration) / 5.0
    )

    # CatBoost
    cat_model = CatBoostClassifier(**cat_params)
    cat_model.fit(X_tr, y_tr, eval_set=(X_va, y_va), use_best_model=True)
    oof_preds_cat[val_idx] = cat_model.predict_proba(X_va)[:, 1]
    test_preds_cat += cat_model.predict_proba(X_test)[:, 1] / 5.0

# Ensemble blend
oof_preds = 0.5 * oof_preds_lgb + 0.5 * oof_preds_cat
test_preds = 0.5 * test_preds_lgb + 0.5 * test_preds_cat

cv_auc = roc_auc_score(y, oof_preds)
print(f"5-Fold CV ROC-AUC: {cv_auc:.6f}")

# 4. Save Submission
submission = pd.DataFrame(
    {"id": np.arange(len(test_preds)), "Task Assigned": test_preds}
)
submission.to_csv(os.path.join(output_dir, "submission.csv"), index=False)
print("Submission saved successfully.")
