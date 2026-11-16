import os
import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from xgboost import XGBRegressor

# Load dataset
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "Total Enrollment"
y = train_df[target_col].values.astype(np.float64)
train_df = train_df.drop(columns=[target_col])

n_train = len(train_df)
combined_df = pd.concat([train_df, test_df], ignore_index=True)

# Process date columns
date_cols = []
for col in combined_df.columns:
    if "date" in col.lower() or pd.api.types.is_datetime64_any_dtype(combined_df[col]):
        date_cols.append(col)
    elif combined_df[col].dtype == "object":
        try:
            parsed = pd.to_datetime(combined_df[col], errors="coerce")
            if parsed.notna().sum() > 0.8 * len(combined_df):
                combined_df[col] = parsed
                date_cols.append(col)
        except Exception:
            pass

for dcol in date_cols:
    combined_df[dcol] = pd.to_datetime(combined_df[dcol], errors="coerce")
    combined_df[f"{dcol}_year"] = combined_df[dcol].dt.year
    combined_df[f"{dcol}_month"] = combined_df[dcol].dt.month
    combined_df[f"{dcol}_quarter"] = combined_df[dcol].dt.quarter
    combined_df[f"{dcol}_day"] = combined_df[dcol].dt.day
    combined_df[f"{dcol}_dayofweek"] = combined_df[dcol].dt.dayofweek
    combined_df[f"{dcol}_month_sin"] = np.sin(
        2 * np.pi * combined_df[f"{dcol}_month"] / 12
    )
    combined_df[f"{dcol}_month_cos"] = np.cos(
        2 * np.pi * combined_df[f"{dcol}_month"] / 12
    )
    combined_df.drop(columns=[dcol], inplace=True)

# Process categorical variables & interactions
cat_cols = combined_df.select_dtypes(include=["object", "category"]).columns.tolist()

if len(cat_cols) >= 2:
    col1, col2 = cat_cols[0], cat_cols[1]
    inter_name = f"{col1}_{col2}_inter"
    combined_df[inter_name] = (
        combined_df[col1].astype(str) + "_" + combined_df[col2].astype(str)
    )
    cat_cols.append(inter_name)

for col in cat_cols:
    freq = combined_df[col].value_counts()
    combined_df[f"{col}_freq"] = combined_df[col].map(freq)

# Split back to train/test
X_train = combined_df.iloc[:n_train].copy()
X_test = combined_df.iloc[n_train:].copy()

# Setup formats for CatBoost and Tree models
cb_cat_cols = [col for col in cat_cols if col in X_train.columns]
for c in cb_cat_cols:
    X_train[c] = X_train[c].astype(str).fillna("missing")
    X_test[c] = X_test[c].astype(str).fillna("missing")

xgb_lgb_X_train = X_train.copy()
xgb_lgb_X_test = X_test.copy()
for c in cb_cat_cols:
    xgb_lgb_X_train[c] = xgb_lgb_X_train[c].astype("category")
    xgb_lgb_X_test[c] = xgb_lgb_X_test[c].astype("category")

# Cross-Validation setup
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(n_train)
test_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y)):
    X_tr_cb, y_tr = X_train.iloc[train_idx], y[train_idx]
    X_va_cb, y_va = X_train.iloc[val_idx], y[val_idx]

    X_tr_tree = xgb_lgb_X_train.iloc[train_idx]
    X_va_tree = xgb_lgb_X_train.iloc[val_idx]

    # CatBoost Model
    cb = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.04,
        depth=6,
        random_seed=42 + fold,
        verbose=0,
        cat_features=cb_cat_cols,
    )
    cb.fit(
        X_tr_cb,
        y_tr,
        eval_set=(X_va_cb, y_va),
        early_stopping_rounds=50,
        verbose=False,
    )

    # XGBoost Model
    xgb = XGBRegressor(
        n_estimators=1000,
        learning_rate=0.04,
        max_depth=6,
        random_state=42 + fold,
        enable_categorical=True,
        tree_method="hist",
        early_stopping_rounds=50,
    )
    xgb.fit(X_tr_tree, y_tr, eval_set=[(X_va_tree, y_va)], verbose=False)

    # LightGBM Model
    lgb_model = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.04,
        max_depth=6,
        random_state=42 + fold,
        verbose=-1,
    )
    lgb_model.fit(
        X_tr_tree,
        y_tr,
        eval_set=[(X_va_tree, y_va)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )

    val_pred_cb = cb.predict(X_va_cb)
    val_pred_xgb = xgb.predict(X_va_tree)
    val_pred_lgb = lgb_model.predict(X_va_tree)

    fold_oof = 0.5 * val_pred_cb + 0.3 * val_pred_xgb + 0.2 * val_pred_lgb
    oof_preds[val_idx] = fold_oof

    test_pred_cb = cb.predict(X_test)
    test_pred_xgb = xgb.predict(xgb_lgb_X_test)
    test_pred_lgb = lgb_model.predict(xgb_lgb_X_test)

    test_preds += (
        0.5 * test_pred_cb + 0.3 * test_pred_xgb + 0.2 * test_pred_lgb
    ) / kf.n_splits

# Post-processing: Non-negative clipping
oof_preds = np.clip(oof_preds, a_min=0, a_max=None)
test_preds = np.clip(test_preds, a_min=0, a_max=None)

# Evaluate Metric
rmse = np.sqrt(mean_squared_error(y, oof_preds))
print(f"Validation RMSE: {rmse:.4f}")

# Output submission
os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame(
    {"id": np.arange(len(test_preds)), "Total Enrollment": test_preds}
)
submission.to_csv("./working/submission.csv", index=False)
