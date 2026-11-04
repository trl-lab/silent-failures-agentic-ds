import os
import warnings
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import lightgbm as lgb

try:
    import xgboost as xgb

    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from catboost import CatBoostClassifier

    HAS_CAT = True
except ImportError:
    HAS_CAT = False

warnings.filterwarnings("ignore")

# 1. Load Data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "marketing_status"
assert (
    target_col in train_df.columns
), f"Target column {target_col} not found in train data."

# Encode binary target as Pandas Series
le_target = LabelEncoder()
y = pd.Series(le_target.fit_transform(train_df[target_col]), index=train_df.index)

n_train = len(train_df)
n_test = len(test_df)

# Combine datasets for consistent feature engineering
combined = pd.concat(
    [
        train_df.drop(columns=[target_col], errors="ignore"),
        test_df.drop(columns=[target_col], errors="ignore"),
    ],
    ignore_index=True,
)

# Drop identifier columns if present
drop_cols = [c for c in ["id", "ID", "index"] if c in combined.columns]
if drop_cols:
    combined.drop(columns=drop_cols, inplace=True)

# 2. Categorize Columns
date_cols = []
cat_cols = []
num_cols = []

for col in combined.columns:
    if pd.api.types.is_datetime64_any_dtype(combined[col]):
        date_cols.append(col)
    elif combined[col].dtype == "object" or combined[col].dtype == "string":
        sample_vals = combined[col].dropna().astype(str).head(50)
        parsed_dates = pd.to_datetime(sample_vals, errors="coerce")
        if parsed_dates.notna().sum() > 30:
            combined[col] = pd.to_datetime(combined[col], errors="coerce")
            date_cols.append(col)
        else:
            cat_cols.append(col)
    elif pd.api.types.is_numeric_dtype(combined[col]):
        num_cols.append(col)
    else:
        cat_cols.append(col)

# Process Date Features
for col in date_cols:
    combined[f"{col}_year"] = combined[col].dt.year
    combined[f"{col}_month"] = combined[col].dt.month
    combined[f"{col}_day"] = combined[col].dt.day
    combined[f"{col}_dayofweek"] = combined[col].dt.dayofweek
    combined[f"{col}_quarter"] = combined[col].dt.quarter
    combined[f"{col}_dayofyear"] = combined[col].dt.dayofyear

    min_date = combined[col].min()
    if pd.notna(min_date):
        combined[f"{col}_days_since_start"] = (combined[col] - min_date).dt.days
    combined.drop(columns=[col], inplace=True)

# Process Categoricals & Text
for col in cat_cols:
    combined[col] = combined[col].fillna("Missing").astype(str)

    # Text length stats
    combined[f"{col}_len"] = combined[col].apply(len)
    combined[f"{col}_word_count"] = combined[col].apply(lambda x: len(x.split()))

    # Frequency encoding
    freq = combined[col].value_counts()
    combined[f"{col}_freq"] = combined[col].map(freq)

# Add TF-IDF + TruncatedSVD for textual features across categoricals
if cat_cols:
    text_data = combined[cat_cols].astype(str).agg(" ".join, axis=1)
    tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=15000, sublinear_tf=True)
    tfidf_mat = tfidf.fit_transform(text_data)

    n_components = min(25, tfidf_mat.shape[1] - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    svd_feats = svd.fit_transform(tfidf_mat)

    for i in range(n_components):
        combined[f"tfidf_svd_{i}"] = svd_feats[:, i]

# Label Encoding Categorical Variables
for col in cat_cols:
    le = LabelEncoder()
    combined[f"{col}_encoded"] = le.fit_transform(combined[col])

# Separate Train and Test Sets
X_train = combined.iloc[:n_train].copy().reset_index(drop=True)
X_test = combined.iloc[n_train:].copy().reset_index(drop=True)

# Out-of-fold target encoding for categorical columns
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
smoothing = 10
global_mean = y.mean()

for col in cat_cols:
    X_train[f"{col}_te"] = np.nan
    X_test[f"{col}_te"] = np.nan

    for train_idx, val_idx in skf.split(X_train, y):
        tr_cat = X_train[col].iloc[train_idx]
        tr_y = y.iloc[train_idx]

        df_tr = pd.DataFrame({"cat": tr_cat.values, "y": tr_y.values})
        stats = df_tr.groupby("cat")["y"].agg(["count", "mean"])

        smooth_te = (stats["count"] * stats["mean"] + smoothing * global_mean) / (
            stats["count"] + smoothing
        )

        val_cat = X_train[col].iloc[val_idx]
        X_train.loc[val_idx, f"{col}_te"] = (
            val_cat.map(smooth_te).fillna(global_mean).values
        )

    df_full = pd.DataFrame({"cat": X_train[col].values, "y": y.values})
    full_stats = df_full.groupby("cat")["y"].agg(["count", "mean"])
    full_smooth_te = (
        full_stats["count"] * full_stats["mean"] + smoothing * global_mean
    ) / (full_stats["count"] + smoothing)
    X_test[f"{col}_te"] = X_test[col].map(full_smooth_te).fillna(global_mean)

# Drop raw categorical text columns
X_train.drop(columns=cat_cols, inplace=True)
X_test.drop(columns=cat_cols, inplace=True)

# Ensure numeric dtypes
for col in X_train.columns:
    X_train[col] = pd.to_numeric(X_train[col], errors="coerce").fillna(0)
    X_test[col] = pd.to_numeric(X_test[col], errors="coerce").fillna(0)

# 3. Model Training & Cross-Validation
oof_lgb = np.zeros(n_train)
test_lgb = np.zeros(n_test)

oof_xgb = np.zeros(n_train)
test_xgb = np.zeros(n_test)

oof_cat = np.zeros(n_train)
test_cat = np.zeros(n_test)

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y)):
    X_tr, y_tr = X_train.iloc[train_idx], y.iloc[train_idx]
    X_va, y_va = X_train.iloc[val_idx], y.iloc[val_idx]

    # LightGBM Classifier
    model_lgb = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        verbosity=-1,
    )
    model_lgb.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )
    oof_lgb[val_idx] = model_lgb.predict_proba(X_va)[:, 1]
    test_lgb += model_lgb.predict_proba(X_test)[:, 1] / skf.n_splits

    # XGBoost Classifier
    if HAS_XGB:
        model_xgb = xgb.XGBClassifier(
            n_estimators=1000,
            learning_rate=0.03,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42 + fold,
            eval_metric="auc",
            early_stopping_rounds=50,
            tree_method="hist",
        )
        model_xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        oof_xgb[val_idx] = model_xgb.predict_proba(X_va)[:, 1]
        test_xgb += model_xgb.predict_proba(X_test)[:, 1] / skf.n_splits

    # CatBoost Classifier
    if HAS_CAT:
        model_cat = CatBoostClassifier(
            iterations=1000,
            learning_rate=0.03,
            depth=6,
            random_seed=42 + fold,
            eval_metric="AUC",
            early_stopping_rounds=50,
            verbose=False,
        )
        model_cat.fit(X_tr, y_tr, eval_set=(X_va, y_va), use_best_model=True)
        oof_cat[val_idx] = model_cat.predict_proba(X_va)[:, 1]
        test_cat += model_cat.predict_proba(X_test)[:, 1] / skf.n_splits

# Rank Average Ensembling
valid_models = [oof_lgb]
test_models = [test_lgb]

if HAS_XGB:
    valid_models.append(oof_xgb)
    test_models.append(test_xgb)

if HAS_CAT:
    valid_models.append(oof_cat)
    test_models.append(test_cat)

valid_ranks = [rankdata(m) / len(m) for m in valid_models]
test_ranks = [rankdata(m) / len(m) for m in test_models]

oof_final = np.mean(valid_ranks, axis=0)
test_final = np.mean(test_ranks, axis=0)

# Calculate Metric
cv_score = roc_auc_score(y, oof_final)
print(f"5-Fold Cross-Validation ROC-AUC Score: {cv_score:.6f}")

# 4. Generate Submission File
os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame({"id": np.arange(n_test), "marketing_status": test_final})
submission.to_csv("./working/submission.csv", index=False)
