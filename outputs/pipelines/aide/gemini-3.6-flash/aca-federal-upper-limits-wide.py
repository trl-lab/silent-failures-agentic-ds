import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
from catboost import CatBoostRegressor

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "ACA FUL"
y_train = train_df[target_col].values

X_train_raw = train_df.drop(columns=[target_col])
X_test_raw = test_df.copy()

# Categorical and numerical columns identification
cat_cols = X_train_raw.select_dtypes(
    include=["object", "category", "string"]
).columns.tolist()
num_cols = X_train_raw.select_dtypes(
    include=["int64", "float64", "int32", "float32"]
).columns.tolist()


# Feature Engineering
def feature_engineering(X_tr, X_te, cat_cols, num_cols):
    df_all = pd.concat([X_tr, X_te], axis=0).reset_index(drop=True)
    n_train = len(X_tr)

    # Frequency encoding
    for c in cat_cols:
        freq = df_all[c].value_counts()
        df_all[f"{c}_freq"] = df_all[c].map(freq)

    # Expanded group aggregations (mean, std, median, min, max)
    selected_cats = cat_cols[: min(5, len(cat_cols))]
    for cat in selected_cats:
        for num in num_cols:
            if df_all[num].nunique() > 1:
                grp = df_all.groupby(cat, observed=False)[num]
                df_all[f"{num}_mean_by_{cat}"] = grp.transform("mean")
                df_all[f"{num}_std_by_{cat}"] = grp.transform("std").fillna(0)
                df_all[f"{num}_median_by_{cat}"] = grp.transform("median")
                df_all[f"{num}_min_by_{cat}"] = grp.transform("min")
                df_all[f"{num}_max_by_{cat}"] = grp.transform("max")

    # Pairwise numerical ratio and interaction features
    selected_nums = num_cols[: min(6, len(num_cols))]
    for i in range(len(selected_nums)):
        for j in range(i + 1, len(selected_nums)):
            col1, col2 = selected_nums[i], selected_nums[j]
            df_all[f"{col1}_ratio_{col2}"] = df_all[col1] / (df_all[col2].abs() + 1e-5)
            df_all[f"{col1}_mult_{col2}"] = df_all[col1] * df_all[col2]

    # Missing value indicators
    for num in num_cols:
        if df_all[num].isnull().sum() > 0:
            df_all[f"{num}_isna"] = df_all[num].isnull().astype(int)

    # Set categorical data types for LightGBM
    for c in cat_cols:
        df_all[c] = df_all[c].astype("category")

    train_fe = df_all.iloc[:n_train].copy()
    test_fe = df_all.iloc[n_train:].copy()
    return train_fe, test_fe


X_train, X_test = feature_engineering(X_train_raw, X_test_raw, cat_cols, num_cols)

# Target transformation check
use_log = (y_train >= 0).all() and (np.max(y_train) > 5 * np.median(y_train))
y_train_trans = np.log1p(y_train) if use_log else y_train

kf = KFold(n_splits=5, shuffle=True, random_state=42)

oof_preds_lgb = np.zeros(len(X_train))
test_preds_lgb = np.zeros(len(X_test))

oof_preds_cb = np.zeros(len(X_train))
test_preds_cb = np.zeros(len(X_test))

for fold, (tr_idx, va_idx) in enumerate(kf.split(X_train, y_train_trans)):
    X_tr, y_tr = X_train.iloc[tr_idx], y_train_trans[tr_idx]
    X_va, y_va = X_train.iloc[va_idx], y_train_trans[va_idx]

    # LightGBM Model
    model_lgb = lgb.LGBMRegressor(
        n_estimators=2000,
        learning_rate=0.03,
        num_leaves=31,
        random_state=42 + fold,
        n_jobs=-1,
        verbose=-1,
    )
    model_lgb.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )

    oof_preds_lgb[va_idx] = model_lgb.predict(X_va)
    test_preds_lgb += model_lgb.predict(X_test) / kf.n_splits

    # CatBoost Model
    X_tr_cb = X_tr.copy()
    X_va_cb = X_va.copy()
    X_te_cb = X_test.copy()
    for c in cat_cols:
        if c in X_tr_cb.columns:
            X_tr_cb[c] = X_tr_cb[c].astype(str).fillna("missing")
            X_va_cb[c] = X_va_cb[c].astype(str).fillna("missing")
            X_te_cb[c] = X_te_cb[c].astype(str).fillna("missing")

    model_cb = CatBoostRegressor(
        iterations=2000,
        learning_rate=0.03,
        depth=6,
        random_state=42 + fold,
        verbose=0,
        cat_features=cat_cols,
    )
    model_cb.fit(X_tr_cb, y_tr, eval_set=(X_va_cb, y_va), early_stopping_rounds=50)

    oof_preds_cb[va_idx] = model_cb.predict(X_va_cb)
    test_preds_cb += model_cb.predict(X_te_cb) / kf.n_splits

# Convert OOF predictions back to raw scale before finding optimal blending weights
if use_log:
    oof_lgb_raw = np.expm1(oof_preds_lgb)
    oof_cb_raw = np.expm1(oof_preds_cb)
    test_lgb_raw = np.expm1(test_preds_lgb)
    test_cb_raw = np.expm1(test_preds_cb)
else:
    oof_lgb_raw = oof_preds_lgb
    oof_cb_raw = oof_preds_cb
    test_lgb_raw = test_preds_lgb
    test_cb_raw = test_preds_cb

if (y_train >= 0).all():
    oof_lgb_raw = np.clip(oof_lgb_raw, 0, None)
    oof_cb_raw = np.clip(oof_cb_raw, 0, None)
    test_lgb_raw = np.clip(test_lgb_raw, 0, None)
    test_cb_raw = np.clip(test_cb_raw, 0, None)


# Optimize blending weight using SciPy
def loss_func(w):
    blend = w[0] * oof_lgb_raw + (1.0 - w[0]) * oof_cb_raw
    return np.sqrt(mean_squared_error(y_train, blend))


res = minimize(loss_func, [0.5], bounds=[(0.0, 1.0)], method="L-BFGS-B")
best_w = res.x[0]

oof_preds = best_w * oof_lgb_raw + (1.0 - best_w) * oof_cb_raw
test_preds = best_w * test_lgb_raw + (1.0 - best_w) * test_cb_raw

cv_rmse = np.sqrt(mean_squared_error(y_train, oof_preds))
print(f"Optimal LGB Weight: {best_w:.4f}, Optimal CatBoost Weight: {1.0 - best_w:.4f}")
print(f"5-Fold Cross-Validation RMSE: {cv_rmse:.5f}")

# Save submission file
os.makedirs("./working", exist_ok=True)
sub = pd.DataFrame({"id": np.arange(len(test_preds)), "ACA FUL": test_preds})
sub.to_csv("./working/submission.csv", index=False)
