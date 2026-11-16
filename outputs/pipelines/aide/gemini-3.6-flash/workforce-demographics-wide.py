import os
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

warnings.filterwarnings("ignore")

# 1. Load data
train_path = "./input/train.parquet"
test_path = "./input/test.parquet"

train = pd.read_parquet(train_path)
test = pd.read_parquet(test_path)

target_col = "Female (%)"
y = train[target_col].values

# Extract id column or generate index
if "id" in test.columns:
    test_ids = test["id"].values
else:
    test_ids = np.arange(len(test))

train_feats = train.drop(columns=[target_col])
test_feats = test.copy()

n_train = len(train_feats)
df_full = pd.concat([train_feats, test_feats], axis=0, ignore_index=True)

# Identify categorical and numerical columns
cat_cols = [
    col
    for col in df_full.columns
    if df_full[col].dtype in ["object", "category"] or df_full[col].nunique() < 20
]
num_cols = [col for col in df_full.columns if col not in cat_cols and col != "id"]

# Handle missing values
for col in num_cols:
    med = df_full[col].median()
    df_full[col] = df_full[col].fillna(med if pd.notnull(med) else 0)

for col in cat_cols:
    df_full[col] = df_full[col].astype(str).fillna("Missing")

# Feature Engineering
# Frequency Encoding
for col in cat_cols:
    freq_map = df_full[col].value_counts(normalize=True)
    df_full[f"{col}_freq"] = df_full[col].map(freq_map)

# Pairwise Categorical Interactions
if len(cat_cols) >= 2:
    for i in range(min(len(cat_cols), 3)):
        for j in range(i + 1, min(len(cat_cols), 3)):
            col1, col2 = cat_cols[i], cat_cols[j]
            comb_name = f"{col1}__x__{col2}"
            df_full[comb_name] = df_full[col1] + "_" + df_full[col2]
            freq_map = df_full[comb_name].value_counts(normalize=True)
            df_full[f"{comb_name}_freq"] = df_full[comb_name].map(freq_map)

# Group aggregations
for c_col in cat_cols[:4]:
    for n_col in num_cols[:4]:
        grp = df_full.groupby(c_col)[n_col]
        df_full[f"{c_col}_{n_col}_mean"] = grp.transform("mean")
        df_full[f"{c_col}_{n_col}_std"] = grp.transform("std").fillna(0)

# Factorize object/categorical columns
for col in df_full.select_dtypes(include=["object", "category"]).columns:
    df_full[col] = pd.factorize(df_full[col])[0]

# Split back to train and test
X = df_full.iloc[:n_train].copy()
X_test = df_full.iloc[n_train:].copy()

if "id" in X.columns:
    X = X.drop(columns=["id"])
    X_test = X_test.drop(columns=["id"])

# Cross-Validation setup
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(train))
test_preds = np.zeros(len(test))

xgb_params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "n_estimators": 2000,
    "learning_rate": 0.02,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "random_state": 42,
    "tree_method": "hist",
    "early_stopping_rounds": 50,
}

lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "n_estimators": 2000,
    "learning_rate": 0.02,
    "num_leaves": 31,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "verbose": -1,
}

cb_params = {
    "iterations": 2000,
    "learning_rate": 0.03,
    "depth": 6,
    "subsample": 0.8,
    "random_seed": 42,
    "verbose": 0,
    "early_stopping_rounds": 50,
}

# Model training loop
for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_tr = X.iloc[train_idx].copy()
    y_tr = y[train_idx]
    X_va = X.iloc[val_idx].copy()
    y_va = y[val_idx]
    X_te_fold = X_test.copy()

    # OOF Target Encoding strictly within fold
    for col in cat_cols:
        tr_map = (
            pd.DataFrame({col: X_tr[col], "target": y_tr}).groupby(col)["target"].mean()
        )
        global_mean = y_tr.mean()
        X_tr[f"{col}_te"] = X_tr[col].map(tr_map).fillna(global_mean)
        X_va[f"{col}_te"] = X_va[col].map(tr_map).fillna(global_mean)
        X_te_fold[f"{col}_te"] = X_te_fold[col].map(tr_map).fillna(global_mean)

    # 1. XGBoost Model
    xgb_model = xgb.XGBRegressor(**xgb_params)
    xgb_model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    xgb_va_pred = xgb_model.predict(X_va)
    xgb_te_pred = xgb_model.predict(X_te_fold)

    # 2. LightGBM Model
    lgb_model = lgb.LGBMRegressor(**lgb_params)
    lgb_model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    lgb_va_pred = lgb_model.predict(X_va)
    lgb_te_pred = lgb_model.predict(X_te_fold)

    # 3. CatBoost Model
    cb_model = CatBoostRegressor(**cb_params)
    cb_model.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=False)
    cb_va_pred = cb_model.predict(X_va)
    cb_te_pred = cb_model.predict(X_te_fold)

    # Weighted Blend of Models
    fold_va_pred = 0.40 * xgb_va_pred + 0.35 * cb_va_pred + 0.25 * lgb_va_pred
    fold_te_pred = 0.40 * xgb_te_pred + 0.35 * cb_te_pred + 0.25 * lgb_te_pred

    oof_preds[val_idx] = fold_va_pred
    test_preds += fold_te_pred / kf.n_splits

# Post-processing
oof_preds = np.clip(oof_preds, 0.0, 100.0)
test_preds = np.clip(test_preds, 0.0, 100.0)

oof_rmse = np.sqrt(mean_squared_error(y, oof_preds))
print(f"Validation Metric (RMSE): {oof_rmse:.5f}")

# Save submission
os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame({"id": test_ids, "Female (%)": test_preds})
submission.to_csv("./working/submission.csv", index=False)
