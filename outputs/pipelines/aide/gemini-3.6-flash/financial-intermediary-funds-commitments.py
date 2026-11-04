import os
import warnings
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
import lightgbm as lgb
from scipy.optimize import minimize
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold
import xgboost as xgb

warnings.filterwarnings("ignore")

# 1. Load Data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "Amount in USD"
y_train = train_df[target_col].values

n_train = len(train_df)
n_test = len(test_df)

# Combine for unified feature engineering
df_all = pd.concat(
    [train_df.drop(columns=[target_col], errors="ignore"), test_df],
    ignore_index=True,
)

# Identify and process datetime features
date_cols = []
for col in df_all.columns:
    if "date" in col.lower() or df_all[col].dtype == "datetime64[ns]":
        date_cols.append(col)
    elif df_all[col].dtype == "object":
        try:
            converted = pd.to_datetime(df_all[col], errors="coerce")
            if converted.notna().sum() > 0.5 * len(df_all):
                df_all[col] = converted
                date_cols.append(col)
        except Exception:
            pass

for col in date_cols:
    df_all[col] = pd.to_datetime(df_all[col], errors="coerce")
    df_all[f"{col}_year"] = df_all[col].dt.year
    df_all[f"{col}_month"] = df_all[col].dt.month
    df_all[f"{col}_quarter"] = df_all[col].dt.quarter
    df_all[f"{col}_day"] = df_all[col].dt.day
    df_all[f"{col}_dayofweek"] = df_all[col].dt.dayofweek
    df_all[f"{col}_dayofyear"] = df_all[col].dt.dayofyear
    df_all[f"{col}_days_since_2000"] = (
        df_all[col] - pd.Timestamp("2000-01-01")
    ).dt.days

for i in range(len(date_cols)):
    for j in range(i + 1, len(date_cols)):
        c1, c2 = date_cols[i], date_cols[j]
        df_all[f"{c1}_{c2}_diff_days"] = (df_all[c1] - df_all[c2]).dt.days

df_all = df_all.drop(columns=date_cols)

# Categorical and numerical columns identification
cat_cols = df_all.select_dtypes(include=["object", "category"]).columns.tolist()

# Frequency Encoding
for col in cat_cols:
    freq = df_all[col].value_counts().to_dict()
    df_all[f"{col}_freq"] = df_all[col].map(freq).fillna(0)

# Categorical Interaction Features
top_cats = sorted(cat_cols, key=lambda c: df_all[c].nunique(), reverse=True)[:4]
for i in range(len(top_cats)):
    for j in range(i + 1, len(top_cats)):
        c1, c2 = top_cats[i], top_cats[j]
        combo_col = f"{c1}_{c2}_combo"
        df_all[combo_col] = df_all[c1].astype(str) + "_" + df_all[c2].astype(str)
        if combo_col not in cat_cols:
            cat_cols.append(combo_col)

for col in cat_cols:
    df_all[col] = df_all[col].astype(str).fillna("missing")

# Split train / test
X_train_df = df_all.iloc[:n_train].copy()
X_test_df = df_all.iloc[n_train:].copy()

# Out-of-fold target encoding
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for col in cat_cols:
    te_train = np.zeros(n_train)
    te_test = np.zeros(n_test)

    for tr_idx, va_idx in kf.split(X_train_df):
        tr_sub = X_train_df.iloc[tr_idx]
        va_sub = X_train_df.iloc[va_idx]
        y_tr_sub = y_train[tr_idx]

        means = (
            pd.DataFrame({col: tr_sub[col], "target": y_tr_sub})
            .groupby(col)["target"]
            .mean()
            .to_dict()
        )
        global_mean = y_tr_sub.mean()
        te_train[va_idx] = va_sub[col].map(means).fillna(global_mean)

    global_mean_all = y_train.mean()
    means_all = (
        pd.DataFrame({col: X_train_df[col], "target": y_train})
        .groupby(col)["target"]
        .mean()
        .to_dict()
    )
    te_test = X_test_df[col].map(means_all).fillna(global_mean_all)

    X_train_df[f"{col}_te"] = te_train
    X_test_df[f"{col}_te"] = te_test

# Group aggregations
for cat_c in top_cats[:2]:
    for num_c in [c for c in X_train_df.columns if c.endswith("_te")][:3]:
        grp_mean = X_train_df.groupby(cat_c)[num_c].mean().to_dict()
        X_train_df[f"{cat_c}_{num_c}_grp_mean"] = (
            X_train_df[cat_c].map(grp_mean).fillna(0)
        )
        X_test_df[f"{cat_c}_{num_c}_grp_mean"] = (
            X_test_df[cat_c].map(grp_mean).fillna(0)
        )

# Prepare categorical dtype for LightGBM / XGBoost
X_train_lgb = X_train_df.copy()
X_test_lgb = X_test_df.copy()
for col in cat_cols:
    X_train_lgb[col] = X_train_lgb[col].astype("category")
    X_test_lgb[col] = X_test_lgb[col].astype("category")

# OOF predictions setup
oof_lgb = np.zeros(n_train)
preds_lgb = np.zeros(n_test)

oof_cat = np.zeros(n_train)
preds_cat = np.zeros(n_test)

oof_xgb = np.zeros(n_train)
preds_xgb = np.zeros(n_test)

for fold, (tr_idx, va_idx) in enumerate(kf.split(X_train_df)):
    X_tr, y_tr = X_train_lgb.iloc[tr_idx], y_train[tr_idx]
    X_va, y_va = X_train_lgb.iloc[va_idx], y_train[va_idx]
    X_te = X_test_lgb.copy()

    # 1. LightGBM Regressor
    model_lgb = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=31,
        random_state=42 + fold,
        subsample=0.8,
        colsample_bytree=0.8,
        verbose=-1,
    )
    model_lgb.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    oof_lgb[va_idx] = model_lgb.predict(X_va)
    preds_lgb += model_lgb.predict(X_te) / 5.0

    # 2. CatBoost Regressor
    X_tr_cat = X_train_df.iloc[tr_idx].copy()
    X_va_cat = X_train_df.iloc[va_idx].copy()
    X_te_cat = X_test_df.copy()

    cat_cols_cat = [c for c in cat_cols if c in X_tr_cat.columns]
    model_cat = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.03,
        depth=6,
        random_seed=42 + fold,
        cat_features=cat_cols_cat,
        verbose=0,
        early_stopping_rounds=50,
    )
    model_cat.fit(X_tr_cat, y_tr, eval_set=(X_va_cat, y_va))
    oof_cat[va_idx] = model_cat.predict(X_va_cat)
    preds_cat += model_cat.predict(X_te_cat) / 5.0

    # 3. XGBoost Regressor
    model_xgb = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=6,
        random_state=42 + fold,
        subsample=0.8,
        colsample_bytree=0.8,
        enable_categorical=True,
        early_stopping_rounds=50,
        tree_method="hist",
    )
    model_xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    oof_xgb[va_idx] = model_xgb.predict(X_va)
    preds_xgb += model_xgb.predict(X_te) / 5.0


# Optimize ensemble weights
def blend_loss(w):
    pred = w[0] * oof_lgb + w[1] * oof_cat + w[2] * oof_xgb
    return root_mean_squared_error(y_train, pred)


opt_res = minimize(
    blend_loss,
    [1 / 3, 1 / 3, 1 / 3],
    bounds=[(0, 1), (0, 1), (0, 1)],
    constraints={"type": "eq", "fun": lambda w: 1 - sum(w)},
)
weights = opt_res.x

oof_blend = np.clip(
    weights[0] * oof_lgb + weights[1] * oof_cat + weights[2] * oof_xgb,
    a_min=0,
    a_max=None,
)
final_rmse = root_mean_squared_error(y_train, oof_blend)

# Generate final test set predictions
test_preds = np.clip(
    weights[0] * preds_lgb + weights[1] * preds_cat + weights[2] * preds_xgb,
    a_min=0,
    a_max=None,
)

os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame(
    {"id": np.arange(len(test_preds)), "Amount in USD": test_preds}
)
submission.to_csv("./working/submission.csv", index=False)

print(f"Validation RMSE: {final_rmse:.4f}")
