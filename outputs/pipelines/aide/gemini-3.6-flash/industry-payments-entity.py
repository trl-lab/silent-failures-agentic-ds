import os
import warnings
import numpy as np
import pandas as pd

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

from scipy.optimize import minimize
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")

# 1. Load Data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "entityPayment"
y = train_df[target_col].values.astype(np.float64)

# Combine datasets for consistent feature engineering
df_all = pd.concat(
    [
        train_df.drop(columns=[target_col], errors="ignore"),
        test_df.drop(columns=[target_col], errors="ignore"),
    ],
    axis=0,
    ignore_index=True,
)
n_train = len(train_df)
n_test = len(test_df)

id_cols = [c for c in df_all.columns if c.lower() in ["id", "index"]]
feature_cols = [c for c in df_all.columns if c not in id_cols]

# Temporal feature extraction
for col in list(feature_cols):
    if df_all[col].dtype == "object" or "date" in col.lower() or "time" in col.lower():
        try:
            converted = pd.to_datetime(df_all[col], errors="coerce")
            if converted.notnull().sum() > 0.5 * len(df_all):
                df_all[f"{col}_year"] = converted.dt.year
                df_all[f"{col}_month"] = converted.dt.month
                df_all[f"{col}_day"] = converted.dt.day
                df_all[f"{col}_dayofweek"] = converted.dt.dayofweek
                df_all[f"{col}_dayofyear"] = converted.dt.dayofyear
                df_all[f"{col}_quarter"] = converted.dt.quarter
                df_all.drop(columns=[col], inplace=True)
        except Exception:
            pass

feature_cols = [c for c in df_all.columns if c not in id_cols and c != target_col]

cat_cols = []
num_cols = []
for c in feature_cols:
    if df_all[c].dtype == "object" or str(df_all[c].dtype) == "category":
        cat_cols.append(c)
    else:
        num_cols.append(c)

# Clean categorical values
for c in cat_cols:
    df_all[c] = df_all[c].fillna("Missing").astype(str)

# Pairwise Categorical Feature Interactions
top_cats = cat_cols[:6]
pair_cat_cols = []
for i in range(len(top_cats)):
    for j in range(i + 1, len(top_cats)):
        c1, c2 = top_cats[i], top_cats[j]
        pair_col = f"{c1}_x_{c2}"
        df_all[pair_col] = df_all[c1] + "_" + df_all[c2]
        pair_cat_cols.append(pair_col)

all_cat_cols = cat_cols + pair_cat_cols

# Frequency encodings
for c in all_cat_cols:
    freq = df_all[c].value_counts(normalize=True).to_dict()
    df_all[f"{c}_freq"] = df_all[c].map(freq).fillna(0)

# Improved Group Statistics on Moderate Cardinality Features
valid_group_cats = [c for c in cat_cols if 1 < df_all[c].nunique() < 5000][:5]
for cat_c in valid_group_cats:
    for num_c in num_cols[:5]:
        grouped = df_all.groupby(cat_c)[num_c].agg([("mean", "mean"), ("std", "std")])
        df_all[f"{cat_c}_{num_c}_mean"] = df_all[cat_c].map(grouped["mean"]).fillna(0)
        df_all[f"{cat_c}_{num_c}_std"] = df_all[cat_c].map(grouped["std"]).fillna(0)

# Prepare base matrices
X_all = df_all.copy()
for c in all_cat_cols:
    X_all[c] = X_all[c].astype("category").cat.codes

X_train = X_all.iloc[:n_train].copy().reset_index(drop=True)
X_test = X_all.iloc[n_train:].copy().reset_index(drop=True)

# Slice categorical columns for Out-of-Fold Target Encoding
train_cats_df = df_all.iloc[:n_train].reset_index(drop=True)
test_cats_df = df_all.iloc[n_train:].reset_index(drop=True)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
smoothing_m = 10.0

for c in all_cat_cols:
    X_train[f"{c}_te_mean"] = np.nan
    X_train[f"{c}_te_std"] = np.nan
    X_test[f"{c}_te_mean"] = 0.0
    X_test[f"{c}_te_std"] = 0.0

    cat_tr_series = train_cats_df[c]
    cat_te_series = test_cats_df[c]

    for tr_idx, val_idx in kf.split(X_train):
        sub_cat_tr = cat_tr_series.iloc[tr_idx]
        sub_val_cat = cat_tr_series.iloc[val_idx]
        sub_y_tr = y[tr_idx]

        g_mean = float(sub_y_tr.mean())
        g_std = float(sub_y_tr.std()) if len(sub_y_tr) > 1 else 0.0

        tmp_df = pd.DataFrame({"cat": sub_cat_tr, "target": sub_y_tr})
        grp = tmp_df.groupby("cat")["target"].agg(["count", "mean", "std"])
        smoothed_mean = (grp["count"] * grp["mean"] + smoothing_m * g_mean) / (
            grp["count"] + smoothing_m
        )
        std_enc = grp["std"].fillna(0.0)

        X_train.loc[val_idx, f"{c}_te_mean"] = sub_val_cat.map(smoothed_mean).fillna(
            g_mean
        )
        X_train.loc[val_idx, f"{c}_te_std"] = sub_val_cat.map(std_enc).fillna(g_std)

    g_mean_full = float(y.mean())
    g_std_full = float(y.std()) if len(y) > 1 else 0.0
    tmp_df_full = pd.DataFrame({"cat": cat_tr_series, "target": y})
    grp_full = tmp_df_full.groupby("cat")["target"].agg(["count", "mean", "std"])
    smoothed_mean_full = (
        grp_full["count"] * grp_full["mean"] + smoothing_m * g_mean_full
    ) / (grp_full["count"] + smoothing_m)
    std_enc_full = grp_full["std"].fillna(0.0)

    X_test[f"{c}_te_mean"] = cat_te_series.map(smoothed_mean_full).fillna(g_mean_full)
    X_test[f"{c}_te_std"] = cat_te_series.map(std_enc_full).fillna(g_std_full)

    X_train[f"{c}_te_mean"] = X_train[f"{c}_te_mean"].fillna(float(y.mean()))
    X_train[f"{c}_te_std"] = X_train[f"{c}_te_std"].fillna(0.0)
    X_test[f"{c}_te_mean"] = X_test[f"{c}_te_mean"].fillna(float(y.mean()))
    X_test[f"{c}_te_std"] = X_test[f"{c}_te_std"].fillna(0.0)

X_train = X_train.fillna(0)
X_test = X_test.fillna(0)

# Model Training Setup
oof_lgb = np.zeros(n_train)
oof_xgb = np.zeros(n_train)
oof_cb = np.zeros(n_train)

test_preds_lgb = np.zeros(n_test)
test_preds_xgb = np.zeros(n_test)
test_preds_cb = np.zeros(n_test)

min_val = max(0.0, float(y.min()))

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_tr, y_tr = X_train.iloc[train_idx], y[train_idx]
    X_va, y_va = X_train.iloc[val_idx], y[val_idx]

    # LightGBM Model
    model_lgb = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=31,
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
    oof_lgb[val_idx] = model_lgb.predict(X_va)
    test_preds_lgb += model_lgb.predict(X_test) / kf.n_splits

    # XGBoost Model
    model_xgb = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        n_jobs=-1,
        early_stopping_rounds=50,
    )
    model_xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    oof_xgb[val_idx] = model_xgb.predict(X_va)
    test_preds_xgb += model_xgb.predict(X_test) / kf.n_splits

    # CatBoost Model
    model_cb = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.03,
        depth=6,
        random_seed=42 + fold,
        verbose=0,
        early_stopping_rounds=50,
    )
    model_cb.fit(X_tr, y_tr, eval_set=(X_va, y_va))
    oof_cb[val_idx] = model_cb.predict(X_va)
    test_preds_cb += model_cb.predict(X_test) / kf.n_splits

# Minimum bound post-processing
oof_lgb = np.clip(oof_lgb, min_val, None)
oof_xgb = np.clip(oof_xgb, min_val, None)
oof_cb = np.clip(oof_cb, min_val, None)

test_preds_lgb = np.clip(test_preds_lgb, min_val, None)
test_preds_xgb = np.clip(test_preds_xgb, min_val, None)
test_preds_cb = np.clip(test_preds_cb, min_val, None)


# Optimize blending weights
def blend_loss(weights):
    w1, w2, w3 = weights
    pred = w1 * oof_lgb + w2 * oof_xgb + w3 * oof_cb
    return root_mean_squared_error(y, np.clip(pred, min_val, None))


res = minimize(
    blend_loss,
    [1 / 3, 1 / 3, 1 / 3],
    bounds=[(0, 1), (0, 1), (0, 1)],
    constraints={"type": "eq", "fun": lambda w: sum(w) - 1},
)

w1, w2, w3 = res.x
oof_blend = np.clip(w1 * oof_lgb + w2 * oof_xgb + w3 * oof_cb, min_val, None)
final_rmse = root_mean_squared_error(y, oof_blend)

print(f"Validation Metric: {final_rmse:.5f}")

# Final blended test predictions
test_preds = np.clip(
    w1 * test_preds_lgb + w2 * test_preds_xgb + w3 * test_preds_cb,
    min_val,
    None,
)

# Save submission
os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame({"id": np.arange(n_test), "entityPayment": test_preds})
submission.to_csv("./working/submission.csv", index=False)
