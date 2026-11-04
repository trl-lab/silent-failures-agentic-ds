import os
import re
import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error
from scipy.optimize import minimize

# 1. Load data
train_path = "./input/train.parquet"
test_path = "./input/test.parquet"

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = "PAYMENTS_BY_ISSUER"

y = train_df[target_col].copy().to_numpy()
X_raw = train_df.drop(columns=[target_col]).copy()
X_test_raw = test_df.copy()


# 2. Data Cleaning & Preprocessing
def clean_features(df):
    df_clean = df.copy()
    for col in df_clean.columns:
        if df_clean[col].dtype == "object" or pd.api.types.is_string_dtype(
            df_clean[col]
        ):
            sample_non_na = df_clean[col].dropna().astype(str)
            if (
                len(sample_non_na) > 0
                and sample_non_na.str.contains(r"^\$?[0-9,.]+$").mean() > 0.5
            ):
                cleaned = (
                    df_clean[col]
                    .astype(str)
                    .str.replace("$", "", regex=False)
                    .str.replace(",", "", regex=False)
                    .str.strip()
                )
                df_clean[col] = pd.to_numeric(cleaned, errors="coerce")
    return df_clean


X = clean_features(X_raw)
X_test = clean_features(X_test_raw)

common_cols = [col for col in X.columns if col in X_test.columns]
X = X[common_cols].copy()
X_test = X_test[common_cols].copy()

# Identify initial categorical and numerical columns
cat_cols = []
num_cols = []

for col in common_cols:
    if (
        X[col].dtype == "object"
        or pd.api.types.is_categorical_dtype(X[col])
        or pd.api.types.is_string_dtype(X[col])
    ):
        cat_cols.append(col)
        X[col] = X[col].fillna("Missing").astype(str)
        X_test[col] = X_test[col].fillna("Missing").astype(str)
    else:
        num_cols.append(col)
        median_val = X[col].median()
        if pd.isna(median_val):
            median_val = 0
        X[col] = X[col].fillna(median_val)
        X_test[col] = X_test[col].fillna(median_val)

# 3. Feature Engineering
# Frequency encoding
for col in cat_cols:
    freq = pd.concat([X[col], X_test[col]], axis=0).value_counts(normalize=True)
    X[f"{col}_freq"] = X[col].map(freq).fillna(0).astype(float)
    X_test[f"{col}_freq"] = X_test[col].map(freq).fillna(0).astype(float)

# Group aggregations on key numerical features
if len(num_cols) > 0 and len(cat_cols) > 0:
    for cat in cat_cols[:3]:  # Top categorical variables
        for num in num_cols[:4]:
            combined = pd.concat([X[[cat, num]], X_test[[cat, num]]], axis=0)
            group_stats = combined.groupby(cat)[num].agg(["mean", "std"]).reset_index()
            group_stats.columns = [cat, f"{cat}_{num}_mean", f"{cat}_{num}_std"]

            X = X.merge(group_stats, on=cat, how="left")
            X_test = X_test.merge(group_stats, on=cat, how="left")

            X[f"{cat}_{num}_std"] = X[f"{cat}_{num}_std"].fillna(0)
            X_test[f"{cat}_{num}_std"] = X_test[f"{cat}_{num}_std"].fillna(0)
            X[f"{cat}_{num}_mean"] = X[f"{cat}_{num}_mean"].fillna(X[num].mean())
            X_test[f"{cat}_{num}_mean"] = X_test[f"{cat}_{num}_mean"].fillna(
                X[num].mean()
            )

# Prepare separate feature representations for LightGBM (with ordinal/frequency encoding) and CatBoost
all_features = [c for c in X.columns]
all_cat_cols = [c for c in all_features if c in cat_cols]

# For LightGBM: convert categorical features to category dtype
X_lgb = X.copy()
X_test_lgb = X_test.copy()
for col in all_cat_cols:
    X_lgb[col] = X_lgb[col].astype("category")
    X_test_lgb[col] = X_test_lgb[col].astype("category")

# 4. 5-Fold Cross-Validation Modeling
kf = KFold(n_splits=5, shuffle=True, random_state=42)

oof_cb = np.zeros(len(X))
test_cb = np.zeros(len(X_test))

oof_lgb = np.zeros(len(X))
test_lgb = np.zeros(len(X_test))

# CatBoost Parameters
cb_params = {
    "iterations": 1500,
    "learning_rate": 0.04,
    "depth": 6,
    "loss_function": "RMSE",
    "eval_metric": "RMSE",
    "random_seed": 42,
    "verbose": 0,
    "early_stopping_rounds": 50,
}

# LightGBM Parameters
lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.04,
    "num_leaves": 31,
    "max_depth": 6,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.85,
    "bagging_freq": 1,
    "min_child_samples": 20,
    "random_state": 42,
    "verbose": -1,
    "n_estimators": 1500,
}

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    # CatBoost fold data
    X_tr_cb, y_tr = X.iloc[train_idx], y[train_idx]
    X_va_cb, y_va = X.iloc[val_idx], y[val_idx]

    cb_train_pool = Pool(X_tr_cb, y_tr, cat_features=all_cat_cols)
    cb_val_pool = Pool(X_va_cb, y_va, cat_features=all_cat_cols)
    cb_test_pool = Pool(X_test, cat_features=all_cat_cols)

    cb_model = CatBoostRegressor(**cb_params)
    cb_model.fit(cb_train_pool, eval_set=cb_val_pool, use_best_model=True)

    oof_cb[val_idx] = cb_model.predict(cb_val_pool)
    test_cb += cb_model.predict(cb_test_pool) / kf.n_splits

    # LightGBM fold data
    X_tr_lgb = X_lgb.iloc[train_idx]
    X_va_lgb = X_lgb.iloc[val_idx]

    lgb_model = lgb.LGBMRegressor(**lgb_params)
    lgb_model.fit(
        X_tr_lgb,
        y_tr,
        eval_set=[(X_va_lgb, y_va)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )

    oof_lgb[val_idx] = lgb_model.predict(X_va_lgb)
    test_lgb += lgb_model.predict(X_test_lgb) / kf.n_splits

cb_rmse = root_mean_squared_error(y, oof_cb)
lgb_rmse = root_mean_squared_error(y, oof_lgb)
print(f"CatBoost OOF RMSE: {cb_rmse:.4f}")
print(f"LightGBM OOF RMSE: {lgb_rmse:.4f}")


# 5. Optimize ensemble weights
def blend_obj(weights):
    w1, w2 = weights
    pred = w1 * oof_cb + w2 * oof_lgb
    return root_mean_squared_error(y, pred)


res = minimize(
    blend_obj,
    [0.5, 0.5],
    bounds=[(0, 1), (0, 1)],
    constraints={"type": "eq", "fun": lambda w: sum(w) - 1.0},
)
w_cb, w_lgb = res.x
oof_ensemble = w_cb * oof_cb + w_lgb * oof_lgb
test_ensemble = w_cb * test_cb + w_lgb * test_lgb

oof_rmse = root_mean_squared_error(y, oof_ensemble)
print(f"Ensemble Weights: CatBoost={w_cb:.3f}, LightGBM={w_lgb:.3f}")
print(f"Out-of-Fold RMSE: {oof_rmse:.4f}")

# 6. Save final predictions
os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame(
    {"id": np.arange(len(test_ensemble)), "PAYMENTS_BY_ISSUER": test_ensemble}
)
submission.to_csv("./working/submission.csv", index=False)
print("Submission successfully saved to ./working/submission.csv")
