import os
import re
import warnings
import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import xgboost as xgb

warnings.filterwarnings("ignore")

# 1. Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "NADAC Per Unit"
y_train = train_df[target_col].values
y_train_log = np.log1p(np.maximum(0, y_train))

train_len = len(train_df)
df = pd.concat(
    [train_df.drop(columns=[target_col], errors="ignore"), test_df],
    ignore_index=True,
)

# 2. Date and Text Feature Engineering
for col in list(df.columns):
    if df[col].dtype == "datetime64[ns]" or pd.api.types.is_datetime64_any_dtype(
        df[col]
    ):
        df[f"{col}_year"] = df[col].dt.year
        df[f"{col}_month"] = df[col].dt.month
        df[f"{col}_day"] = df[col].dt.day
        df[f"{col}_dayofweek"] = df[col].dt.dayofweek
        df[f"{col}_dayofyear"] = df[col].dt.dayofyear
        df.drop(columns=[col], inplace=True)
    elif df[col].dtype == "object":
        sample_vals = df[col].dropna().astype(str).head(100)
        if any(
            re.search(r"\d{4}-\d{2}-\d{2}", v) or re.search(r"\d{2}/\d{2}/\d{4}", v)
            for v in sample_vals
        ):
            parsed = pd.to_datetime(df[col], errors="coerce")
            if parsed.notna().sum() > 0.5 * len(df):
                df[f"{col}_year"] = parsed.dt.year
                df[f"{col}_month"] = parsed.dt.month
                df[f"{col}_day"] = parsed.dt.day
                df[f"{col}_dayofweek"] = parsed.dt.dayofweek
                df[f"{col}_dayofyear"] = parsed.dt.dayofyear
                df.drop(columns=[col], inplace=True)

text_cols = []
cat_cols = []
num_cols = []

for col in df.columns:
    if col in ["id", "ID"]:
        continue
    if df[col].dtype == "object" or df[col].dtype.name == "category":
        nunique = df[col].nunique()
        if nunique > 50 and df[col].astype(str).str.len().mean() > 15:
            text_cols.append(col)
        else:
            cat_cols.append(col)
    else:
        num_cols.append(col)

# Structural Text Features
for col in text_cols:
    str_series = df[col].fillna("").astype(str)
    df[f"{col}_len"] = str_series.str.len()
    df[f"{col}_words"] = str_series.str.split().str.len()
    df[f"{col}_digits"] = str_series.str.count(r"\d")

# TF-IDF + SVD text decomposition
svd_dfs = []
for col in text_cols:
    tfidf = TfidfVectorizer(max_features=5000, stop_words="english")
    text_data = df[col].fillna("").astype(str)
    tfidf_mat = tfidf.fit_transform(text_data)
    n_comp = min(15, tfidf_mat.shape[1] - 1)
    if n_comp > 1:
        svd = TruncatedSVD(n_components=n_comp, random_state=42)
        svd_mat = svd.fit_transform(tfidf_mat)
        svd_df = pd.DataFrame(
            svd_mat, columns=[f"{col}_svd_{i}" for i in range(n_comp)]
        )
        svd_dfs.append(svd_df)

if svd_dfs:
    df = pd.concat([df] + svd_dfs, axis=1)

# Categorical Interaction Features
interaction_cols = []
if len(cat_cols) >= 2:
    for i in range(min(4, len(cat_cols))):
        for j in range(i + 1, min(4, len(cat_cols))):
            col1, col2 = cat_cols[i], cat_cols[j]
            inter_col = f"{col1}_X_{col2}"
            df[inter_col] = df[col1].astype(str) + "_" + df[col2].astype(str)
            interaction_cols.append(inter_col)

# Expanded Group Aggregations
for c_col in cat_cols[:5]:
    for n_col in num_cols[:5]:
        if c_col != n_col:
            grp = df.groupby(c_col)[n_col].agg(["mean", "std", "median"]).reset_index()
            grp.columns = [
                c_col,
                f"{c_col}_{n_col}_grp_mean",
                f"{c_col}_{n_col}_grp_std",
                f"{c_col}_{n_col}_grp_med",
            ]
            df = df.merge(grp, on=c_col, how="left")

all_te_cols = cat_cols + interaction_cols

for col in cat_cols + text_cols + interaction_cols:
    freq = df[col].value_counts()
    df[f"{col}_freq"] = df[col].map(freq)
    df[col] = df[col].astype("category").cat.codes

df = df.apply(pd.to_numeric, errors="coerce").fillna(0)

X_train_full = df.iloc[:train_len].copy()
X_test_full = df.iloc[train_len:].copy()

# 3. 5-Fold Cross Validation with 3-Model Ensemble (LightGBM, CatBoost, XGBoost)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(train_len)
test_preds = np.zeros(len(test_df))

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_full, y_train_log)):
    X_tr = X_train_full.iloc[train_idx].copy()
    y_tr = y_train_log[train_idx]
    X_va = X_train_full.iloc[val_idx].copy()
    y_va = y_train_log[val_idx]
    X_te = X_test_full.copy()

    # Multi-level Smoothed Out-of-fold Target Encoding
    global_mean = y_tr.mean()
    global_std = y_tr.std()
    weight = 10.0

    for col in all_te_cols:
        stats = (
            pd.DataFrame({"cat": X_tr[col], "target": y_tr})
            .groupby("cat")["target"]
            .agg(["count", "mean", "std"])
        )
        smooth_mean = (stats["count"] * stats["mean"] + weight * global_mean) / (
            stats["count"] + weight
        )
        smooth_std = stats["std"].fillna(0)

        X_tr[f"{col}_te_mean"] = X_tr[col].map(smooth_mean).fillna(global_mean)
        X_va[f"{col}_te_mean"] = X_va[col].map(smooth_mean).fillna(global_mean)
        X_te[f"{col}_te_mean"] = X_te[col].map(smooth_mean).fillna(global_mean)

        X_tr[f"{col}_te_std"] = X_tr[col].map(smooth_std).fillna(global_std)
        X_va[f"{col}_te_std"] = X_va[col].map(smooth_std).fillna(global_std)
        X_te[f"{col}_te_std"] = X_te[col].map(smooth_std).fillna(global_std)

    # Model 1: LightGBM Regressor
    model_lgb = lgb.LGBMRegressor(
        n_estimators=1200,
        learning_rate=0.03,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        n_jobs=-1,
    )
    model_lgb.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )

    # Model 2: CatBoost Regressor
    model_cat = cb.CatBoostRegressor(
        iterations=1200,
        learning_rate=0.04,
        depth=6,
        random_seed=42 + fold,
        verbose=0,
    )
    model_cat.fit(X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=50)

    # Model 3: XGBoost Regressor
    model_xgb = xgb.XGBRegressor(
        n_estimators=1200,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        n_jobs=-1,
        tree_method="hist",
        early_stopping_rounds=50,
    )
    model_xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

    # 3-way Ensemble predictions
    val_pred = (
        0.4 * model_lgb.predict(X_va)
        + 0.35 * model_cat.predict(X_va)
        + 0.25 * model_xgb.predict(X_va)
    )
    oof_preds[val_idx] = val_pred

    test_pred_fold = (
        0.4 * model_lgb.predict(X_te)
        + 0.35 * model_cat.predict(X_te)
        + 0.25 * model_xgb.predict(X_te)
    )
    test_preds += test_pred_fold / kf.get_n_splits()

# 4. Evaluation and Submission
oof_preds_raw = np.maximum(0, np.expm1(oof_preds))
overall_rmse = np.sqrt(mean_squared_error(y_train, oof_preds_raw))
print(f"Validation RMSE: {overall_rmse:.5f}")

os.makedirs("./working", exist_ok=True)
sub = pd.DataFrame(
    {
        "id": np.arange(len(test_preds)),
        "NADAC Per Unit": np.maximum(0, np.expm1(test_preds)),
    }
)
sub.to_csv("./working/submission.csv", index=False)
