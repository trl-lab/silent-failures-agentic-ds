import os
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

try:
    from catboost import CatBoostRegressor

    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

from sklearn.ensemble import HistGradientBoostingRegressor

# 1. Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "State Share"
y_train = train_df[target_col].values
X_train_raw = train_df.drop(columns=[target_col])
X_test_raw = test_df.copy()

# Combine for uniform feature parsing
n_train = len(X_train_raw)
combined = pd.concat([X_train_raw, X_test_raw], axis=0, ignore_index=True)

processed_data = pd.DataFrame()

# Clean numerical strings, parse dates, and identify categoricals
for col in combined.columns:
    s = combined[col]

    if pd.api.types.is_numeric_dtype(s):
        processed_data[col] = s.values
        continue

    s_str = s.astype(str).str.strip()

    # Try parsing numeric from currency/percent strings
    cleaned_str = (
        s_str.str.replace(r"[\$,%]", "", regex=True)
        .str.replace(",", "", regex=True)
        .str.strip()
    )
    converted_num = pd.to_numeric(cleaned_str, errors="coerce")

    if converted_num.notnull().mean() > 0.5:
        processed_data[col] = converted_num.values
        continue

    # Try datetime parsing
    try:
        converted_date = pd.to_datetime(s_str, errors="coerce")
        if converted_date.notnull().mean() > 0.5:
            processed_data[f"{col}_year"] = converted_date.dt.year.values
            processed_data[f"{col}_month"] = converted_date.dt.month.values
            processed_data[f"{col}_quarter"] = converted_date.dt.quarter.values
            processed_data[f"{col}_dayofweek"] = converted_date.dt.dayofweek.values
            continue
    except Exception:
        pass

    # Categorical column string handling
    processed_data[col] = s_str.fillna("Missing").values

# Frequency encoding for categorical columns
cat_cols = [c for c in processed_data.columns if processed_data[c].dtype == "object"]
for c in cat_cols:
    freq = processed_data[c].value_counts(normalize=True).to_dict()
    processed_data[f"{c}_freq"] = processed_data[c].map(freq).fillna(0).values

# Split back to train and test
X_train = processed_data.iloc[:n_train].copy().reset_index(drop=True)
X_test = processed_data.iloc[n_train:].copy().reset_index(drop=True)

# Group statistics feature engineering
num_cols = [c for c in X_train.columns if pd.api.types.is_numeric_dtype(X_train[c])]
if len(cat_cols) > 0 and len(num_cols) > 0:
    for cat in cat_cols[:3]:
        for num in num_cols[:3]:
            group_mean = X_train.groupby(cat)[num].mean().to_dict()
            X_train[f"{cat}_{num}_mean"] = (
                X_train[cat].map(group_mean).fillna(X_train[num].mean())
            )
            X_test[f"{cat}_{num}_mean"] = (
                X_test[cat].map(group_mean).fillna(X_train[num].mean())
            )

# Cast object columns to string explicitly
for c in cat_cols:
    X_train[c] = X_train[c].astype(str)
    X_test[c] = X_test[c].astype(str)

# 5-fold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(n_train)
test_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
    X_tr, y_tr = X_train.iloc[train_idx].copy(), y_train[train_idx]
    X_va, y_va = X_train.iloc[val_idx].copy(), y_train[val_idx]
    X_te = X_test.copy()

    # Target encoding for categoricals without data leakage
    for c in cat_cols:
        target_map = (
            pd.DataFrame({c: X_tr[c], "target": y_tr})
            .groupby(c)["target"]
            .mean()
            .to_dict()
        )
        global_mean = y_tr.mean()
        X_tr[f"{c}_te"] = X_tr[c].map(target_map).fillna(global_mean)
        X_va[f"{c}_te"] = X_va[c].map(target_map).fillna(global_mean)
        X_te[f"{c}_te"] = X_te[c].map(target_map).fillna(global_mean)

    if HAS_CATBOOST:
        model = CatBoostRegressor(
            iterations=1500,
            learning_rate=0.04,
            depth=6,
            cat_features=cat_cols if len(cat_cols) > 0 else None,
            eval_metric="RMSE",
            random_seed=42 + fold,
            verbose=0,
        )
        model.fit(
            X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=50, verbose=False
        )
        val_pred = model.predict(X_va)
        te_pred = model.predict(X_te)
    else:
        X_tr_hgb, X_va_hgb, X_te_hgb = X_tr.copy(), X_va.copy(), X_te.copy()
        for c in cat_cols:
            codes, _ = pd.factorize(pd.concat([X_tr[c], X_va[c], X_te[c]]))
            X_tr_hgb[c] = codes[: len(X_tr)]
            X_va_hgb[c] = codes[len(X_tr) : len(X_tr) + len(X_va)]
            X_te_hgb[c] = codes[len(X_tr) + len(X_va) :]

        model = HistGradientBoostingRegressor(
            max_iter=1000, learning_rate=0.04, max_depth=6, random_state=42 + fold
        )
        model.fit(X_tr_hgb, y_tr)
        val_pred = model.predict(X_va_hgb)
        te_pred = model.predict(X_te_hgb)

    oof_preds[val_idx] = val_pred
    test_preds += te_pred / kf.n_splits

# Print Out-Of-Fold RMSE metric
cv_rmse = np.sqrt(mean_squared_error(y_train, oof_preds))
print(f"Validation RMSE: {cv_rmse:.5f}")

# Save predictions to submission.csv
os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame({"id": np.arange(len(test_preds)), "State Share": test_preds})
submission.to_csv("./working/submission.csv", index=False)
