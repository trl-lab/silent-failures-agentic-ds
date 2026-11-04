import os
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import lightgbm as lgb

try:
    import catboost as cb

    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

warnings.filterwarnings("ignore")

# Create working directory if it doesn't exist
os.makedirs("./working", exist_ok=True)

# 1. Load Datasets
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "current_status"
y_raw = train_df[target_col].copy()

# Target encoding (binary 0/1)
if y_raw.dtype == object or str(y_raw.dtype).startswith("cat"):
    le_target = LabelEncoder()
    y = le_target.fit_transform(y_raw.astype(str))
else:
    y = y_raw.values.astype(int)

n_train = len(train_df)
df_all = pd.concat(
    [
        train_df.drop(columns=[target_col]),
        test_df.drop(columns=[target_col], errors="ignore"),
    ],
    axis=0,
).reset_index(drop=True)

# Remove row ID or unnecessary index columns if present
id_cols = [
    col
    for col in df_all.columns
    if col.lower() in ["id", "index", "row_id", "unnamed: 0"]
]
features_all = df_all.drop(columns=id_cols, errors="ignore").copy()

# 2. Datetime Features Extraction
date_cols = []
for col in features_all.columns:
    if "date" in col.lower() or features_all[col].dtype == "datetime64[ns]":
        date_cols.append(col)
    else:
        sample = features_all[col].dropna().astype(str).head(50)
        if len(sample) > 0 and sample.str.match(r"^\d{4}-\d{2}-\d{2}").all():
            date_cols.append(col)

for col in date_cols:
    dt_series = pd.to_datetime(features_all[col], errors="coerce")
    features_all[f"{col}_year"] = dt_series.dt.year
    features_all[f"{col}_month"] = dt_series.dt.month
    features_all[f"{col}_day"] = dt_series.dt.day
    features_all[f"{col}_dayofweek"] = dt_series.dt.dayofweek
    features_all[f"{col}_dayofyear"] = dt_series.dt.dayofyear
    features_all[f"{col}_is_month_end"] = dt_series.dt.is_month_end.astype(float)

for i in range(len(date_cols)):
    for j in range(i + 1, len(date_cols)):
        col1, col2 = date_cols[i], date_cols[j]
        dt1 = pd.to_datetime(features_all[col1], errors="coerce")
        dt2 = pd.to_datetime(features_all[col2], errors="coerce")
        features_all[f"diff_{col1}_{col2}"] = (dt1 - dt2).dt.total_seconds() / 86400.0

features_all.drop(columns=date_cols, inplace=True, errors="ignore")

# 3. Categorical vs Text Feature Identification
cat_cols = []
text_cols = []

for col in features_all.columns:
    if features_all[col].dtype == object or str(features_all[col].dtype).startswith(
        "cat"
    ):
        non_nulls = features_all[col].dropna().astype(str)
        if len(non_nulls) > 0 and non_nulls.str.len().mean() > 50:
            text_cols.append(col)
        else:
            cat_cols.append(col)

# Text Feature Engineering with TF-IDF + SVD
for tcol in text_cols:
    text_data = features_all[tcol].fillna("").astype(str)
    tfidf = TfidfVectorizer(max_features=5000, stop_words="english", ngram_range=(1, 2))
    tfidf_mat = tfidf.fit_transform(text_data)
    svd = TruncatedSVD(n_components=10, random_state=42)
    svd_mat = svd.fit_transform(tfidf_mat)
    for k in range(svd_mat.shape[1]):
        features_all[f"tfidf_{tcol}_svd_{k}"] = svd_mat[:, k]
features_all.drop(columns=text_cols, inplace=True, errors="ignore")

# Frequency Encoding
for col in cat_cols:
    freq = features_all[col].value_counts(normalize=True).to_dict()
    features_all[f"{col}_freq"] = features_all[col].map(freq).fillna(0)

# Group Aggregation Features
num_cols = [
    c
    for c in features_all.columns
    if c not in cat_cols and features_all[c].dtype in [float, int, "float64", "int64"]
]
diff_cols = [c for c in num_cols if c.startswith("diff_")]
agg_target_cols = diff_cols if len(diff_cols) > 0 else num_cols[:5]

for c_col in cat_cols:
    for n_col in agg_target_cols:
        grouped = features_all.groupby(c_col)[n_col]
        features_all[f"{c_col}_{n_col}_mean"] = features_all[c_col].map(
            grouped.transform("mean")
        )
        features_all[f"{c_col}_{n_col}_std"] = (
            features_all[c_col].map(grouped.transform("std")).fillna(0)
        )

# Label Encoding
for col in cat_cols:
    features_all[col] = LabelEncoder().fit_transform(
        features_all[col].fillna("Missing").astype(str)
    )

# Split train and test
X_train = features_all.iloc[:n_train].copy().reset_index(drop=True)
X_test = features_all.iloc[n_train:].copy().reset_index(drop=True)

# 4. Out-of-Fold Bayesian Smoothed Target Encoding
m = 10.0  # Smoothing parameter
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
global_mean = y.mean()

for col in cat_cols:
    X_train[f"{col}_te"] = np.nan
    X_test[f"{col}_te"] = 0.0

    for tr_idx, val_idx in skf.split(X_train, y):
        tr_x, tr_y = X_train.iloc[tr_idx], pd.Series(
            y[tr_idx], index=X_train.index[tr_idx]
        )
        val_x = X_train.iloc[val_idx]

        stats = tr_y.groupby(tr_x[col]).agg(["count", "mean"])
        smoothed = (stats["count"] * stats["mean"] + m * global_mean) / (
            stats["count"] + m
        )
        te_map = smoothed.to_dict()

        X_train.loc[X_train.index[val_idx], f"{col}_te"] = (
            val_x[col].map(te_map).fillna(global_mean)
        )

    full_stats = (
        pd.Series(y, index=X_train.index).groupby(X_train[col]).agg(["count", "mean"])
    )
    smoothed_full = (full_stats["count"] * full_stats["mean"] + m * global_mean) / (
        full_stats["count"] + m
    )
    te_map_full = smoothed_full.to_dict()
    X_test[f"{col}_te"] = X_test[col].map(te_map_full).fillna(global_mean)

# Convert categorical columns to category dtype for LightGBM
for col in cat_cols:
    X_train[col] = X_train[col].astype("category")
    X_test[col] = X_test[col].astype("category")

# 5. Model Training & Cross-Validation
oof_preds_lgb = np.zeros(n_train)
test_preds_lgb = np.zeros(len(X_test))

oof_preds_cb = np.zeros(n_train)
test_preds_cb = np.zeros(len(X_test))

lgb_params = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "learning_rate": 0.03,
    "num_leaves": 63,
    "max_depth": -1,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "min_child_samples": 20,
    "verbose": -1,
    "random_state": 42,
    "n_estimators": 2000,
}

cb_params = (
    {
        "iterations": 2000,
        "learning_rate": 0.03,
        "depth": 6,
        "eval_metric": "AUC",
        "random_seed": 42,
        "verbose": 0,
    }
    if HAS_CATBOOST
    else None
)

for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y)):
    X_tr, y_tr = X_train.iloc[tr_idx], y[tr_idx]
    X_va, y_va = X_train.iloc[val_idx], y[val_idx]

    # LightGBM
    trn_data = lgb.Dataset(X_tr, label=y_tr)
    val_data = lgb.Dataset(X_va, label=y_va, reference=trn_data)

    model_lgb = lgb.train(
        lgb_params,
        trn_data,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
    )
    oof_preds_lgb[val_idx] = model_lgb.predict(
        X_va, num_iteration=model_lgb.best_iteration
    )
    test_preds_lgb += (
        model_lgb.predict(X_test, num_iteration=model_lgb.best_iteration) / 5.0
    )

    # CatBoost
    if HAS_CATBOOST:
        X_tr_cb = X_tr.copy()
        X_va_cb = X_va.copy()
        X_te_cb = X_test.copy()
        for col in cat_cols:
            X_tr_cb[col] = X_tr_cb[col].astype(int)
            X_va_cb[col] = X_va_cb[col].astype(int)
            X_te_cb[col] = X_te_cb[col].astype(int)

        model_cb = cb.CatBoostClassifier(**cb_params)
        model_cb.fit(
            X_tr_cb,
            y_tr,
            eval_set=(X_va_cb, y_va),
            cat_features=[col for col in cat_cols],
            early_stopping_rounds=50,
            verbose=False,
        )
        oof_preds_cb[val_idx] = model_cb.predict_proba(X_va_cb)[:, 1]
        test_preds_cb += model_cb.predict_proba(X_te_cb)[:, 1] / 5.0

# 6. Ensemble Weight Optimization & Prediction
if HAS_CATBOOST:

    def loss_func(weights):
        w1, w2 = weights
        blend = w1 * oof_preds_lgb + w2 * oof_preds_cb
        return -roc_auc_score(y, blend)

    res = minimize(loss_func, [0.5, 0.5], method="Nelder-Mead")
    w1_opt, w2_opt = res.x
    w_sum = w1_opt + w2_opt
    w1_opt, w2_opt = w1_opt / w_sum, w2_opt / w_sum

    oof_preds = w1_opt * oof_preds_lgb + w2_opt * oof_preds_cb
    test_preds = w1_opt * test_preds_lgb + w2_opt * test_preds_cb
else:
    oof_preds = oof_preds_lgb
    test_preds = test_preds_lgb

final_auc = roc_auc_score(y, oof_preds)
print(f"Validation ROC AUC: {final_auc:.5f}")

# 7. Save Submission File
submission = pd.DataFrame(
    {"id": np.arange(len(test_preds)), "current_status": test_preds}
)
submission.to_csv("./working/submission.csv", index=False)
