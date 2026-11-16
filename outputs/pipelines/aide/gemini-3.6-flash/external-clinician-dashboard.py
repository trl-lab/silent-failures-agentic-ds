import os
import warnings
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
import lightgbm as lgb
from sklearn.linear_model import Ridge
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

# 1. Load Data
train_path = "./input/train.parquet"
test_path = "./input/test.parquet"

train = pd.read_parquet(train_path)
test = pd.read_parquet(test_path)

target_col = "Score"
assert target_col in train.columns, f"Target {target_col} not found in train data."

# Prepare test ID column
if "id" in test.columns:
    test_ids = test["id"].values
else:
    test_ids = np.arange(len(test))

train_y = train[target_col].values
train_df = train.drop(columns=[target_col], errors="ignore").copy()
test_df = test.copy()

if "id" in train_df.columns:
    train_df = train_df.drop(columns=["id"])
if "id" in test_df.columns:
    test_df = test_df.drop(columns=["id"])

# 2. Preprocessing & Feature Engineering
cat_cols = []
num_cols = []
date_cols = []

for col in train_df.columns:
    if pd.api.types.is_datetime64_any_dtype(train_df[col]):
        date_cols.append(col)
    elif pd.api.types.is_numeric_dtype(train_df[col]):
        num_cols.append(col)
    else:
        parsed = pd.to_datetime(train_df[col], errors="coerce")
        if parsed.notna().sum() > 0.5 * len(train_df) and train_df[col].nunique() > 10:
            train_df[col] = parsed
            test_df[col] = pd.to_datetime(test_df[col], errors="coerce")
            date_cols.append(col)
        else:
            cat_cols.append(col)

# Datetime Feature Extraction
for col in date_cols:
    for df in [train_df, test_df]:
        df[f"{col}_year"] = df[col].dt.year
        df[f"{col}_month"] = df[col].dt.month
        df[f"{col}_day"] = df[col].dt.day
        df[f"{col}_dayofweek"] = df[col].dt.dayofweek
    train_df = train_df.drop(columns=[col])
    test_df = test_df.drop(columns=[col])

# Refresh column types
cat_cols = [
    col for col in train_df.columns if not pd.api.types.is_numeric_dtype(train_df[col])
]
num_cols = [
    col for col in train_df.columns if pd.api.types.is_numeric_dtype(train_df[col])
]

# Clean Infinite values
train_df[num_cols] = train_df[num_cols].replace([np.inf, -np.inf], np.nan)
test_df[num_cols] = test_df[num_cols].replace([np.inf, -np.inf], np.nan)

# Frequency Encoding
for col in cat_cols:
    freq = pd.concat([train_df[col], test_df[col]]).value_counts().to_dict()
    train_df[f"{col}_freq"] = train_df[col].map(freq).fillna(0)
    test_df[f"{col}_freq"] = test_df[col].map(freq).fillna(0)

# Out-Of-Fold Target Encoding for Categoricals
kf = KFold(n_splits=5, shuffle=True, random_state=42)
global_mean = train_y.mean()
smoothing = 10.0

for col in cat_cols:
    te_col = f"{col}_te"
    train_df[te_col] = np.nan
    test_df[te_col] = 0.0

    for tr_idx, va_idx in kf.split(train_df):
        tr_sub_y = train_y[tr_idx]
        tr_sub_cat = train_df.iloc[tr_idx][col]

        stats = (
            pd.DataFrame({"cat": tr_sub_cat, "y": tr_sub_y})
            .groupby("cat")["y"]
            .agg(["count", "mean"])
        )
        smooth = (stats["count"] * stats["mean"] + smoothing * global_mean) / (
            stats["count"] + smoothing
        )
        mapping = smooth.to_dict()

        train_df.iloc[va_idx, train_df.columns.get_loc(te_col)] = (
            train_df.iloc[va_idx][col].map(mapping).fillna(global_mean)
        )
        test_df[te_col] += test_df[col].map(mapping).fillna(global_mean).values / 5.0

# Extended Group Aggregations
top_cats = [c for c in cat_cols if 1 < train_df[c].nunique() < 500][:4]
top_nums = num_cols[:4]

full_df = pd.concat([train_df, test_df], axis=0)
for c_col in top_cats:
    for n_col in top_nums:
        grp_mean = full_df.groupby(c_col)[n_col].transform("mean")
        grp_std = full_df.groupby(c_col)[n_col].transform("std")
        grp_median = full_df.groupby(c_col)[n_col].transform("median")

        train_df[f"{c_col}_{n_col}_grp_mean"] = grp_mean.iloc[: len(train_df)]
        train_df[f"{c_col}_{n_col}_grp_std"] = grp_std.iloc[: len(train_df)]
        train_df[f"{c_col}_{n_col}_grp_median"] = grp_median.iloc[: len(train_df)]

        test_df[f"{c_col}_{n_col}_grp_mean"] = grp_mean.iloc[len(train_df) :]
        test_df[f"{c_col}_{n_col}_grp_std"] = grp_std.iloc[len(train_df) :]
        test_df[f"{c_col}_{n_col}_grp_median"] = grp_median.iloc[len(train_df) :]

# Dedicated datasets for native categorical handling
X_train_cat = train_df.copy()
X_test_cat = test_df.copy()

X_train_cb = train_df.copy()
X_test_cb = test_df.copy()

for col in cat_cols:
    # Unified categorical categories for LightGBM and XGBoost
    all_categories = pd.concat([train_df[col], test_df[col]]).fillna("missing").unique()
    X_train_cat[col] = pd.Categorical(
        X_train_cat[col].fillna("missing"), categories=all_categories
    )
    X_test_cat[col] = pd.Categorical(
        X_test_cat[col].fillna("missing"), categories=all_categories
    )

    # String format for CatBoost native cat_features
    X_train_cb[col] = X_train_cb[col].fillna("missing").astype(str)
    X_test_cb[col] = X_test_cb[col].fillna("missing").astype(str)

# 3. Model Training & Validation Setup
oof_lgb = np.zeros(len(train_df))
oof_cat = np.zeros(len(train_df))
oof_xgb = np.zeros(len(train_df))

test_lgb = np.zeros(len(test_df))
test_cat = np.zeros(len(test_df))
test_xgb = np.zeros(len(test_df))

for fold, (tr_idx, va_idx) in enumerate(kf.split(train_df)):
    y_tr, y_val = train_y[tr_idx], train_y[va_idx]

    # Model 1: LightGBM (Native Categorical)
    X_tr_lgb, X_va_lgb = X_train_cat.iloc[tr_idx], X_train_cat.iloc[va_idx]
    model_lgb = lgb.LGBMRegressor(
        n_estimators=1500,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        verbosity=-1,
    )
    model_lgb.fit(
        X_tr_lgb,
        y_tr,
        eval_set=[(X_va_lgb, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    oof_lgb[va_idx] = model_lgb.predict(X_va_lgb)
    test_lgb += model_lgb.predict(X_test_cat) / 5.0

    # Model 2: CatBoost (Native Categorical)
    X_tr_cb, X_va_cb = X_train_cb.iloc[tr_idx], X_train_cb.iloc[va_idx]
    model_cat = CatBoostRegressor(
        iterations=1500,
        learning_rate=0.03,
        depth=6,
        cat_features=cat_cols,
        random_seed=42 + fold,
        verbose=0,
    )
    model_cat.fit(
        X_tr_cb,
        y_tr,
        eval_set=(X_va_cb, y_val),
        early_stopping_rounds=50,
        verbose=False,
    )
    oof_cat[va_idx] = model_cat.predict(X_va_cb)
    test_cat += model_cat.predict(X_test_cb) / 5.0

    # Model 3: XGBoost (Native Categorical with Hist)
    X_tr_xgb, X_va_xgb = X_train_cat.iloc[tr_idx], X_train_cat.iloc[va_idx]
    model_xgb = XGBRegressor(
        n_estimators=1500,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        enable_categorical=True,
        tree_method="hist",
        random_state=42 + fold,
        early_stopping_rounds=50,
        n_jobs=-1,
    )
    model_xgb.fit(
        X_tr_xgb,
        y_tr,
        eval_set=[(X_va_xgb, y_val)],
        verbose=False,
    )
    oof_xgb[va_idx] = model_xgb.predict(X_va_xgb)
    test_xgb += model_xgb.predict(X_test_cat) / 5.0

# 4. Out-of-Fold Blending
OOF_matrix = np.column_stack([oof_lgb, oof_cat, oof_xgb])
Test_matrix = np.column_stack([test_lgb, test_cat, test_xgb])

meta_model = Ridge(alpha=10.0, positive=True)
meta_model.fit(OOF_matrix, train_y)

oof_blend = meta_model.predict(OOF_matrix)
test_blend = meta_model.predict(Test_matrix)

if (train_y >= 0).all():
    oof_blend = np.clip(oof_blend, 0, None)
    test_blend = np.clip(test_blend, 0, None)

final_rmse = root_mean_squared_error(train_y, oof_blend)
print(f"Validation Metric (RMSE): {final_rmse:.5f}")

# 5. Export Submission
os.makedirs("./working", exist_ok=True)
sub = pd.DataFrame({"id": test_ids, "Score": test_blend})
sub.to_csv("./working/submission.csv", index=False)
