import os
import warnings
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import rankdata
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb

warnings.filterwarnings("ignore")

# 1. Load Data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target = train_df["TARGET"].values
train_df = train_df.drop(columns=["TARGET"], errors="ignore")

# Drop SK_ID_CURR if present in features
train_df = train_df.drop(columns=["SK_ID_CURR"], errors="ignore")
test_df = test_df.drop(columns=["SK_ID_CURR"], errors="ignore")

train_len = len(train_df)
df = pd.concat([train_df, test_df], axis=0, ignore_index=True)

# 2. Enhanced Domain Feature Engineering
df["DAYS_EMPLOYED_ANOM"] = (df["DAYS_EMPLOYED"] == 365243).astype(int)
df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(365243, np.nan)

df["MISSING_COUNT"] = df.isnull().sum(axis=1)

# Key Financial Ratios
df["PAYMENT_RATE"] = df["AMT_ANNUITY"] / (df["AMT_CREDIT"] + 1e-5)
df["ANNUITY_INCOME_PERC"] = df["AMT_ANNUITY"] / (df["AMT_INCOME_TOTAL"] + 1e-5)
df["CREDIT_TO_INCOME"] = df["AMT_CREDIT"] / (df["AMT_INCOME_TOTAL"] + 1e-5)
df["DAYS_EMPLOYED_PERC"] = df["DAYS_EMPLOYED"] / (df["DAYS_BIRTH"] + 1e-5)
df["INCOME_PER_PERSON"] = df["AMT_INCOME_TOTAL"] / (
    df["CNT_FAM_MEMBERS"].fillna(1) + 1e-5
)
df["GOODS_PRICE_TO_CREDIT"] = df["AMT_GOODS_PRICE"] / (df["AMT_CREDIT"] + 1e-5)
df["GOODS_INCOME_PERC"] = df["AMT_GOODS_PRICE"] / (df["AMT_INCOME_TOTAL"] + 1e-5)
df["AGE_YEARS"] = -df["DAYS_BIRTH"] / 365.25
df["EMPLOYED_YEARS"] = -df["DAYS_EMPLOYED"] / 365.25

# Down-Payment & Credit-Goods Features
df["CREDIT_GOODS_DIFF"] = df["AMT_CREDIT"] - df["AMT_GOODS_PRICE"]
df["DOWN_PAYMENT_RATIO"] = df["CREDIT_GOODS_DIFF"] / (df["AMT_CREDIT"] + 1e-5)

# Time Duration Ratios relative to Age
df["REGISTRATION_TO_BIRTH"] = df["DAYS_REGISTRATION"] / (df["DAYS_BIRTH"] + 1e-5)
df["ID_PUBLISH_TO_BIRTH"] = df["DAYS_ID_PUBLISH"] / (df["DAYS_BIRTH"] + 1e-5)
if "DAYS_LAST_PHONE_CHANGE" in df.columns:
    df["PHONE_CHANGE_TO_BIRTH"] = df["DAYS_LAST_PHONE_CHANGE"] / (
        df["DAYS_BIRTH"] + 1e-5
    )

# Document & Bureau Request Aggregations
doc_cols = [c for c in df.columns if "FLAG_DOCUMENT" in c]
if doc_cols:
    df["DOCUMENT_COUNT"] = df[doc_cols].sum(axis=1)

bureau_req_cols = [c for c in df.columns if "AMT_REQ_CREDIT_BUREAU" in c]
if bureau_req_cols:
    df["BUREAU_REQ_SUM"] = df[bureau_req_cols].sum(axis=1)

# External Source Aggregations & Interactions
ext_cols = [
    c for c in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"] if c in df.columns
]
if ext_cols:
    df["EXT_SOURCE_COUNT"] = df[ext_cols].notnull().sum(axis=1)
    df["EXT_SOURCE_MEAN"] = df[ext_cols].mean(axis=1)
    df["EXT_SOURCE_STD"] = df[ext_cols].std(axis=1)
    df["EXT_SOURCE_MIN"] = df[ext_cols].min(axis=1)
    df["EXT_SOURCE_MAX"] = df[ext_cols].max(axis=1)
    if len(ext_cols) == 3:
        df["EXT_SOURCE_PROD"] = (
            df["EXT_SOURCE_1"] * df["EXT_SOURCE_2"] * df["EXT_SOURCE_3"]
        )
        df["EXT_SOURCE_1_2"] = df["EXT_SOURCE_1"] * df["EXT_SOURCE_2"]
        df["EXT_SOURCE_2_3"] = df["EXT_SOURCE_2"] * df["EXT_SOURCE_3"]
        df["EXT_SOURCE_1_3"] = df["EXT_SOURCE_1"] * df["EXT_SOURCE_3"]

    df["EXT_SOURCE_MEAN_PAYMENT_RATE"] = df["EXT_SOURCE_MEAN"] * df["PAYMENT_RATE"]
    df["EXT_SOURCE_MEAN_CREDIT_INCOME"] = df["EXT_SOURCE_MEAN"] * df["CREDIT_TO_INCOME"]
    df["EXT_SOURCE_MEAN_ANNUITY_INCOME"] = (
        df["EXT_SOURCE_MEAN"] * df["ANNUITY_INCOME_PERC"]
    )

# Group Aggregations: Mean, Diff, and Z-Score
for cat_col in [
    "ORGANIZATION_TYPE",
    "OCCUPATION_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_INCOME_TYPE",
]:
    if cat_col in df.columns:
        for num_col in [
            "AMT_INCOME_TOTAL",
            "AMT_CREDIT",
            "PAYMENT_RATE",
            "EXT_SOURCE_2",
        ]:
            if num_col in df.columns:
                mean_val = df.groupby(cat_col)[num_col].transform("mean")
                std_val = df.groupby(cat_col)[num_col].transform("std")
                df[f"{cat_col}_{num_col}_MEAN"] = mean_val
                df[f"{cat_col}_{num_col}_DIFF"] = df[num_col] - mean_val
                df[f"{cat_col}_{num_col}_ZSCORE"] = (df[num_col] - mean_val) / (
                    std_val + 1e-5
                )

# Frequency Encoding & Categorical Type Assignment
cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
for c in cat_cols:
    freq = df[c].value_counts()
    df[f"{c}_FREQ"] = df[c].map(freq)
    df[c] = df[c].astype("category")

# 3. Split Train / Test
X_train = df.iloc[:train_len].copy()
X_test = df.iloc[train_len:].copy()

# 4. Cross-Validation Setup & Tri-Model Training
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_lgb = np.zeros(len(X_train))
oof_xgb = np.zeros(len(X_train))
oof_hgb = np.zeros(len(X_train))

test_lgb = np.zeros(len(X_test))
test_xgb = np.zeros(len(X_test))
test_hgb = np.zeros(len(X_test))

lgb_params = {
    "n_estimators": 1500,
    "learning_rate": 0.03,
    "num_leaves": 31,
    "max_depth": -1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "n_jobs": -1,
    "objective": "binary",
    "metric": "auc",
    "verbose": -1,
}

xgb_params = {
    "n_estimators": 1200,
    "learning_rate": 0.03,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "tree_method": "hist",
    "enable_categorical": True,
    "eval_metric": "auc",
    "early_stopping_rounds": 50,
    "random_state": 42,
    "n_jobs": -1,
}

hgb_params = {
    "max_iter": 400,
    "learning_rate": 0.03,
    "max_leaf_nodes": 31,
    "categorical_features": "from_dtype",
    "early_stopping": True,
    "n_iter_no_change": 30,
    "random_state": 42,
}

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, target)):
    X_tr, y_tr = X_train.iloc[train_idx], target[train_idx]
    X_va, y_va = X_train.iloc[val_idx], target[val_idx]

    # Model 1: LightGBM
    lgb_model = lgb.LGBMClassifier(**lgb_params)
    lgb_model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    oof_lgb[val_idx] = lgb_model.predict_proba(X_va)[:, 1]
    test_lgb += lgb_model.predict_proba(X_test)[:, 1] / skf.n_splits

    # Model 2: XGBoost
    xgb_model = xgb.XGBClassifier(**xgb_params)
    xgb_model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    oof_xgb[val_idx] = xgb_model.predict_proba(X_va)[:, 1]
    test_xgb += xgb_model.predict_proba(X_test)[:, 1] / skf.n_splits

    # Model 3: HistGradientBoosting
    hgb_model = HistGradientBoostingClassifier(**hgb_params)
    hgb_model.fit(X_tr, y_tr)
    oof_hgb[val_idx] = hgb_model.predict_proba(X_va)[:, 1]
    test_hgb += hgb_model.predict_proba(X_test)[:, 1] / skf.n_splits

# 5. Rank Normalization & Optimization Across Three Diverse Model Families
oof_lgb_rank = rankdata(oof_lgb) / len(oof_lgb)
oof_xgb_rank = rankdata(oof_xgb) / len(oof_xgb)
oof_hgb_rank = rankdata(oof_hgb) / len(oof_hgb)

test_lgb_rank = rankdata(test_lgb) / len(test_lgb)
test_xgb_rank = rankdata(test_xgb) / len(test_xgb)
test_hgb_rank = rankdata(test_hgb) / len(test_hgb)


def loss_func(weights):
    w1, w2, w3 = weights
    blend = w1 * oof_lgb_rank + w2 * oof_xgb_rank + w3 * oof_hgb_rank
    return -roc_auc_score(target, blend)


opt_res = minimize(
    loss_func, [0.4, 0.4, 0.2], method="Nelder-Mead", options={"maxiter": 500}
)
w1, w2, w3 = opt_res.x
norm = w1 + w2 + w3
w1, w2, w3 = w1 / norm, w2 / norm, w3 / norm

oof_blend = w1 * oof_lgb_rank + w2 * oof_xgb_rank + w3 * oof_hgb_rank
test_blend = w1 * test_lgb_rank + w2 * test_xgb_rank + w3 * test_hgb_rank

# 6. Evaluate Metric
blend_auc = roc_auc_score(target, oof_blend)
print(f"Validation Metric (OOF ROC AUC): {blend_auc:.5f}")

# 7. Save Submission File
os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame({"id": np.arange(len(test_blend)), "TARGET": test_blend})
submission.to_csv("./working/submission.csv", index=False)
