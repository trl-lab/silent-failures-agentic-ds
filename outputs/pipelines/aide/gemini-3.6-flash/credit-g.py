import os
import numpy as np
import pandas as pd

from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

# Ensure output directory exists
os.makedirs("./working", exist_ok=True)

# 1. Load datasets
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "class"
y_raw = train_df[target_col].copy()

# Map target to binary integers (0 and 1)
if y_raw.dtype in ["object", "string", "category"]:
    unique_vals = sorted(y_raw.unique())
    target_map = {val: i for i, val in enumerate(unique_vals)}
    y = y_raw.map(target_map).astype(int)
else:
    y = (y_raw - y_raw.min()).astype(int)

train_df = train_df.drop(columns=[target_col])

if "id" in test_df.columns:
    test_ids = test_df["id"].values
    test_df = test_df.drop(columns=["id"])
else:
    test_ids = np.arange(len(test_df))

if "id" in train_df.columns:
    train_df = train_df.drop(columns=["id"])


# 2. Feature Engineering
def create_features(df_train, df_test):
    df_all = pd.concat([df_train, df_test], axis=0, ignore_index=True)

    # Identify initial categorical columns
    cat_cols = df_all.select_dtypes(include=["object", "category"]).columns.tolist()

    # Ensure categoricals are standard strings
    for c in cat_cols:
        df_all[c] = df_all[c].astype(str)

    # Financial interaction ratios
    if "credit_amount" in df_all.columns and "duration" in df_all.columns:
        df_all["monthly_payment_approx"] = (
            df_all["credit_amount"] / (df_all["duration"] + 1e-5)
        ).astype(float)

    if "credit_amount" in df_all.columns and "age" in df_all.columns:
        df_all["credit_per_age"] = (
            df_all["credit_amount"] / (df_all["age"] + 1e-5)
        ).astype(float)

    if "duration" in df_all.columns and "age" in df_all.columns:
        df_all["duration_per_age"] = (
            df_all["duration"] / (df_all["age"] + 1e-5)
        ).astype(float)

    # Proxy disposable income based on installment rate/commitment
    inst_col = [
        c for c in ["installment_commitment", "installment_rate"] if c in df_all.columns
    ]
    if inst_col and "monthly_payment_approx" in df_all.columns:
        df_all["est_disposable_income"] = (
            df_all["monthly_payment_approx"] / (df_all[inst_col[0]] + 1e-5)
        ).astype(float)

    # Categorical frequency encoding
    for c in cat_cols:
        freq_map = df_all[c].value_counts(normalize=True).to_dict()
        df_all[f"{c}_freq"] = df_all[c].map(freq_map).astype(float)

    # Group mean and standard deviation aggregations
    num_cols = ["credit_amount", "duration", "age"]
    num_cols = [c for c in num_cols if c in df_all.columns]

    for g_col in cat_cols:
        for n_col in num_cols:
            df_all[f"{n_col}_mean_by_{g_col}"] = (
                df_all.groupby(g_col)[n_col].transform("mean").astype(float)
            )
            df_all[f"{n_col}_std_by_{g_col}"] = (
                df_all.groupby(g_col)[n_col].transform("std").fillna(0).astype(float)
            )

    # Factorize categoricals for tree/numeric features
    for c in cat_cols:
        df_all[f"{c}_code"] = pd.factorize(df_all[c])[0].astype(int)

    X_tr = df_all.iloc[: len(df_train)].copy()
    X_te = df_all.iloc[len(df_train) :].copy()
    return X_tr, X_te, cat_cols


X, X_test, cat_cols = create_features(train_df, test_df)

# 3. Stratified 5-Fold Cross Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_lgb = np.zeros(len(X))
oof_cat = np.zeros(len(X))
oof_xgb = np.zeros(len(X))

test_lgb = np.zeros(len(X_test))
test_cat = np.zeros(len(X_test))
test_xgb = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train, y_train = X.iloc[train_idx].copy(), y.iloc[train_idx].copy()
    X_val, y_val = X.iloc[val_idx].copy(), y.iloc[val_idx].copy()
    X_te = X_test.copy()

    # Out-of-fold target encoding with smoothing
    smooth_m = 10.0
    global_mean = float(y_train.mean())

    for c in cat_cols:
        stats = y_train.groupby(X_train[c]).agg(["count", "mean"])
        smooth_enc = (stats["count"] * stats["mean"] + smooth_m * global_mean) / (
            stats["count"] + smooth_m
        )

        X_train[f"{c}_te"] = (
            X_train[c].map(smooth_enc).fillna(global_mean).astype(float)
        )
        X_val[f"{c}_te"] = X_val[c].map(smooth_enc).fillna(global_mean).astype(float)
        X_te[f"{c}_te"] = X_te[c].map(smooth_enc).fillna(global_mean).astype(float)

    # Model 1: LightGBM
    X_tr_lgb, X_va_lgb, X_te_lgb = (
        X_train.copy(),
        X_val.copy(),
        X_te.copy(),
    )
    for c in cat_cols:
        X_tr_lgb[c] = X_tr_lgb[c].astype("category")
        X_va_lgb[c] = X_va_lgb[c].astype("category")
        X_te_lgb[c] = X_te_lgb[c].astype("category")

    lgb_model = LGBMClassifier(
        n_estimators=600,
        learning_rate=0.025,
        num_leaves=15,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        verbose=-1,
    )
    lgb_model.fit(
        X_tr_lgb,
        y_train,
        eval_set=[(X_va_lgb, y_val)],
        eval_metric="auc",
        callbacks=[early_stopping(50, verbose=False), log_evaluation(-1)],
    )
    oof_lgb[val_idx] = lgb_model.predict_proba(X_va_lgb)[:, 1]
    test_lgb += lgb_model.predict_proba(X_te_lgb)[:, 1] / skf.n_splits

    # Model 2: CatBoost
    X_tr_cat, X_va_cat, X_te_cat = (
        X_train.copy(),
        X_val.copy(),
        X_te.copy(),
    )
    for c in cat_cols:
        X_tr_cat[c] = X_tr_cat[c].astype(str)
        X_va_cat[c] = X_va_cat[c].astype(str)
        X_te_cat[c] = X_te_cat[c].astype(str)

    cat_model = CatBoostClassifier(
        iterations=600,
        learning_rate=0.025,
        depth=4,
        random_seed=42 + fold,
        cat_features=cat_cols,
        verbose=False,
        eval_metric="AUC",
    )
    cat_model.fit(
        X_tr_cat,
        y_train,
        eval_set=(X_va_cat, y_val),
        early_stopping_rounds=50,
        verbose=False,
    )
    oof_cat[val_idx] = cat_model.predict_proba(X_va_cat)[:, 1]
    test_cat += cat_model.predict_proba(X_te_cat)[:, 1] / skf.n_splits

    # Model 3: XGBoost (drop string categoricals, use numeric features)
    xgb_cols = [c for c in X_train.columns if c not in cat_cols]
    X_tr_xgb = X_train[xgb_cols].astype(float)
    X_va_xgb = X_val[xgb_cols].astype(float)
    X_te_xgb = X_te[xgb_cols].astype(float)

    xgb_model = XGBClassifier(
        n_estimators=600,
        learning_rate=0.025,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        eval_metric="auc",
        early_stopping_rounds=50,
        n_jobs=-1,
    )
    xgb_model.fit(
        X_tr_xgb,
        y_train,
        eval_set=[(X_va_xgb, y_val)],
        verbose=False,
    )
    oof_xgb[val_idx] = xgb_model.predict_proba(X_va_xgb)[:, 1]
    test_xgb += xgb_model.predict_proba(X_te_xgb)[:, 1] / skf.n_splits

# Blend OOF predictions
oof_blend = 0.45 * oof_lgb + 0.35 * oof_cat + 0.20 * oof_xgb
test_preds = 0.45 * test_lgb + 0.35 * test_cat + 0.20 * test_xgb

auc_score = roc_auc_score(y, oof_blend)
if auc_score < 0.5:
    oof_blend = 1.0 - oof_blend
    test_preds = 1.0 - test_preds
    auc_score = roc_auc_score(y, oof_blend)

print(f"OOF ROC AUC Score: {auc_score:.5f}")

# 4. Save submission file
submission = pd.DataFrame({"id": test_ids, "class": test_preds})
submission.to_csv("./working/submission.csv", index=False)
