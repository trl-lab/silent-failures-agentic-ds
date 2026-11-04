import os
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error
import lightgbm as lgb

warnings.filterwarnings("ignore")

# 1. Load Data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "amountUSD"
y_train_raw = train_df[target_col].values.astype(float)

X_train = train_df.drop(columns=[target_col]).copy()
X_test = test_df.copy()

n_train = len(train_df)
df_all = pd.concat([X_train, X_test], axis=0, ignore_index=True)

# 2. Datetime Processing & Parsing
date_cols = []
for col in df_all.columns:
    if pd.api.types.is_datetime64_any_dtype(df_all[col]):
        date_cols.append(col)
    elif df_all[col].dtype == "object":
        try:
            parsed = pd.to_datetime(df_all[col], errors="coerce")
            if parsed.notna().sum() > 0.5 * len(df_all):
                df_all[col] = parsed
                date_cols.append(col)
        except Exception:
            pass

for col in date_cols:
    df_all[f"{col}_year"] = df_all[col].dt.year
    df_all[f"{col}_month"] = df_all[col].dt.month
    df_all[f"{col}_day"] = df_all[col].dt.day
    df_all[f"{col}_dayofweek"] = df_all[col].dt.dayofweek
    df_all[f"{col}_quarter"] = df_all[col].dt.quarter
    df_all[f"{col}_dayofyear"] = df_all[col].dt.dayofyear

    min_date = df_all[col].min()
    if pd.notna(min_date):
        df_all[f"{col}_elapsed_days"] = (df_all[col] - min_date).dt.total_seconds() / (
            24 * 3600
        )
    df_all.drop(columns=[col], inplace=True)

# Identify Categoricals & Numeric Columns
cat_cols = df_all.select_dtypes(include=["object", "category"]).columns.tolist()

drop_cols = [c for c in df_all.columns if df_all[c].nunique(dropna=False) <= 1]
df_all.drop(columns=drop_cols, inplace=True)

cat_cols = [c for c in cat_cols if c not in drop_cols]
num_cols = df_all.select_dtypes(include=["number"]).columns.tolist()

# Ensure categorical columns are object (string) format for feature engineering
for col in cat_cols:
    df_all[col] = df_all[col].fillna("missing").astype(str)

# Frequency Encoding
for col in cat_cols:
    freq = df_all[col].value_counts().to_dict()
    df_all[f"{col}_freq"] = df_all[col].map(freq).astype(float)

# Separate train and test
X_train_df = df_all.iloc[:n_train].copy().reset_index(drop=True)
X_test_df = df_all.iloc[n_train:].copy().reset_index(drop=True)

# 3. Out-Of-Fold Target Encoding
kf = KFold(n_splits=5, shuffle=True, random_state=42)
global_mean = float(y_train_raw.mean())

for col in cat_cols:
    X_train_df[f"{col}_te"] = global_mean
    X_test_df[f"{col}_te"] = global_mean

    for train_idx, val_idx in kf.split(X_train_df):
        tr_cat = X_train_df.iloc[train_idx][col]
        tr_y = y_train_raw[train_idx]

        stats = (
            pd.DataFrame({"cat": tr_cat, "y": tr_y})
            .groupby("cat")["y"]
            .agg(["count", "mean"])
        )
        smooth = 10.0
        smoothed = (stats["count"] * stats["mean"] + smooth * global_mean) / (
            stats["count"] + smooth
        )

        val_cat = X_train_df.iloc[val_idx][col]
        mapped_vals = val_cat.map(smoothed).fillna(global_mean).values.astype(float)
        X_train_df.loc[val_idx, f"{col}_te"] = mapped_vals

    full_stats = (
        pd.DataFrame({"cat": X_train_df[col], "y": y_train_raw})
        .groupby("cat")["y"]
        .agg(["count", "mean"])
    )
    smoothed_full = (
        full_stats["count"] * full_stats["mean"] + smooth * global_mean
    ) / (full_stats["count"] + smooth)
    X_test_df[f"{col}_te"] = (
        X_test_df[col].map(smoothed_full).fillna(global_mean).values.astype(float)
    )

# Group Aggregation Statistics
for c_col in cat_cols[:3]:
    for n_col in num_cols[:3]:
        if n_col in X_train_df.columns:
            grp = df_all.groupby(c_col)[n_col].agg(["mean", "std"]).reset_index()
            grp.columns = [
                c_col,
                f"{c_col}_{n_col}_grp_mean",
                f"{c_col}_{n_col}_grp_std",
            ]
            X_train_df = X_train_df.merge(grp, on=c_col, how="left")
            X_test_df = X_test_df.merge(grp, on=c_col, how="left")

# Convert string features to category dtype for LightGBM
cat_features = X_train_df.select_dtypes(include=["object"]).columns.tolist()
for c in cat_features:
    X_train_df[c] = X_train_df[c].astype("category")
    X_test_df[c] = X_test_df[c].astype("category")

# 4. Model Training & Evaluation
use_log = np.all(y_train_raw >= 0) and (
    y_train_raw.max() > 10 * np.median(y_train_raw)
    if np.median(y_train_raw) > 0
    else False
)
y_train_mod = np.log1p(y_train_raw) if use_log else y_train_raw

oof_preds = np.zeros(n_train)
test_preds = np.zeros(len(X_test_df))

lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.03,
    "num_leaves": 31,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "random_state": 42,
    "n_estimators": 2000,
    "verbose": -1,
}

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_df)):
    X_tr, y_tr = X_train_df.iloc[train_idx], y_train_mod[train_idx]
    X_va, y_va = X_train_df.iloc[val_idx], y_train_mod[val_idx]

    model = lgb.LGBMRegressor(**lgb_params)
    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )

    val_pred = model.predict(X_va)
    if use_log:
        val_pred = np.expm1(val_pred)
    oof_preds[val_idx] = val_pred

    t_pred = model.predict(X_test_df)
    if use_log:
        t_pred = np.expm1(t_pred)
    test_preds += t_pred / kf.n_splits

oof_preds = np.clip(oof_preds, 0, None)
test_preds = np.clip(test_preds, 0, None)

cv_rmse = root_mean_squared_error(y_train_raw, oof_preds)
print(f"Validation RMSE: {cv_rmse:.4f}")

# 5. Save Submission
os.makedirs("./working", exist_ok=True)
sub = pd.DataFrame({"id": np.arange(len(test_preds)), "amountUSD": test_preds})
sub.to_csv("./working/submission.csv", index=False)
