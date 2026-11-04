import os
import warnings
from catboost import CatBoostRegressor
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold
import xgboost as xgb

warnings.filterwarnings("ignore")

# 1. Load data
train_path = "./input/train.parquet"
test_path = "./input/test.parquet"
train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = "fatalities"
y = train_df[target_col].values.astype(np.float64)
X_train_raw = train_df.drop(columns=[target_col]).copy()
X_test_raw = test_df.copy()

n_train = len(train_df)
n_test = len(test_df)
combined = pd.concat([X_train_raw, X_test_raw], axis=0).reset_index(drop=True)

# 2. Feature Engineering
# 2a. Temporal feature extraction and cyclical transformations
date_cols = []
for col in combined.columns:
    if combined[col].dtype == "object":
        try:
            converted = pd.to_datetime(combined[col], format="mixed")
            if converted.notna().mean() > 0.8:
                combined[f"{col}_year"] = converted.dt.year.fillna(2000).astype(int)
                combined[f"{col}_month"] = converted.dt.month.fillna(1).astype(int)
                combined[f"{col}_day"] = converted.dt.day.fillna(1).astype(int)
                combined[f"{col}_dayofweek"] = converted.dt.dayofweek.fillna(0).astype(
                    int
                )
                combined[f"{col}_dayofyear"] = converted.dt.dayofyear.fillna(1).astype(
                    int
                )

                # Cyclical encodings
                combined[f"{col}_month_sin"] = np.sin(
                    2 * np.pi * combined[f"{col}_month"] / 12.0
                )
                combined[f"{col}_month_cos"] = np.cos(
                    2 * np.pi * combined[f"{col}_month"] / 12.0
                )
                combined[f"{col}_dow_sin"] = np.sin(
                    2 * np.pi * combined[f"{col}_dayofweek"] / 7.0
                )
                combined[f"{col}_dow_cos"] = np.cos(
                    2 * np.pi * combined[f"{col}_dayofweek"] / 7.0
                )
                date_cols.append(col)
        except Exception:
            pass

combined.drop(columns=date_cols, inplace=True)

# 2b. Categorical processing and pairwise interactions
cat_cols = combined.select_dtypes(include=["object", "category"]).columns.tolist()

if len(cat_cols) >= 2:
    top_cats = sorted(cat_cols, key=lambda c: combined[c].nunique(), reverse=True)[:4]
    for i in range(len(top_cats)):
        for j in range(i + 1, len(top_cats)):
            c1, c2 = top_cats[i], top_cats[j]
            inter_name = f"{c1}_x_{c2}"
            combined[inter_name] = (
                combined[c1].astype(str) + "__" + combined[c2].astype(str)
            )
            cat_cols.append(inter_name)

# Frequency encoding & label factorization
for col in cat_cols:
    freq = combined[col].value_counts(normalize=True).to_dict()
    combined[f"{col}_freq"] = combined[col].map(freq).fillna(0).astype(np.float32)
    combined[col] = pd.factorize(combined[col].astype(str))[0].astype(np.int32)

# 2c. Group aggregation statistics on numeric columns
numeric_cols = [
    c
    for c in combined.columns
    if c not in cat_cols and not c.endswith("_freq") and not c.endswith("_sin")
]
key_cat = cat_cols[0] if len(cat_cols) > 0 else None

if key_cat and len(numeric_cols) > 0:
    for num_col in numeric_cols[:5]:
        group_mean = combined.groupby(key_cat)[num_col].transform("mean")
        group_std = (
            combined.groupby(key_cat)[num_col].transform("std").fillna(0).astype(float)
        )
        combined[f"{num_col}_mean_by_{key_cat}"] = group_mean
        combined[f"{num_col}_diff_mean_by_{key_cat}"] = combined[num_col] - group_mean
        combined[f"{num_col}_std_by_{key_cat}"] = group_std

# Convert all column types to standard numeric
for col in combined.columns:
    if combined[col].dtype == "object":
        combined[col] = pd.factorize(combined[col].astype(str))[0]
combined = combined.astype(np.float32)

X = combined.iloc[:n_train].copy()
X_test = combined.iloc[n_train:].copy()

# 3. 5-Fold Cross-Validation with XGBoost, LightGBM, and CatBoost
kf = KFold(n_splits=5, shuffle=True, random_state=42)

oof_xgb = np.zeros(n_train, dtype=np.float64)
oof_lgb = np.zeros(n_train, dtype=np.float64)
oof_cb = np.zeros(n_train, dtype=np.float64)

test_xgb = np.zeros(n_test, dtype=np.float64)
test_lgb = np.zeros(n_test, dtype=np.float64)
test_cb = np.zeros(n_test, dtype=np.float64)

xgb_params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "learning_rate": 0.04,
    "max_depth": 7,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "n_estimators": 2000,
    "tree_method": "hist",
    "n_jobs": -1,
}

lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.04,
    "num_leaves": 63,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "n_estimators": 2000,
    "verbose": -1,
    "n_jobs": -1,
}

cb_params = {
    "loss_function": "RMSE",
    "eval_metric": "RMSE",
    "learning_rate": 0.04,
    "depth": 6,
    "random_seed": 42,
    "iterations": 2000,
    "verbose": False,
    "thread_count": -1,
}

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_tr, y_tr = X.iloc[train_idx], y[train_idx]
    X_va, y_va = X.iloc[val_idx], y[val_idx]

    # Train XGBoost
    model_xgb = xgb.XGBRegressor(**xgb_params, early_stopping_rounds=50)
    model_xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    oof_xgb[val_idx] = model_xgb.predict(X_va)
    test_xgb += model_xgb.predict(X_test) / kf.n_splits

    # Train LightGBM
    model_lgb = lgb.LGBMRegressor(**lgb_params)
    model_lgb.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )
    oof_lgb[val_idx] = model_lgb.predict(X_va)
    test_lgb += model_lgb.predict(X_test) / kf.n_splits

    # Train CatBoost
    model_cb = CatBoostRegressor(**cb_params, early_stopping_rounds=50)
    model_cb.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=False)
    oof_cb[val_idx] = model_cb.predict(X_va)
    test_cb += model_cb.predict(X_test) / kf.n_splits


# 4. Ensemble optimization via 3-way constrained blending
def loss_func(weights):
    w1, w2, w3 = weights
    pred = w1 * oof_xgb + w2 * oof_lgb + w3 * oof_cb
    return root_mean_squared_error(y, np.clip(pred, 0, None))


opt_res = minimize(
    loss_func,
    [1 / 3, 1 / 3, 1 / 3],
    bounds=[(0, 1), (0, 1), (0, 1)],
    constraints={"type": "eq", "fun": lambda w: sum(w) - 1.0},
)
w_xgb, w_lgb, w_cb = opt_res.x

oof_blend = np.clip(w_xgb * oof_xgb + w_lgb * oof_lgb + w_cb * oof_cb, 0, None)
test_blend = np.clip(w_xgb * test_xgb + w_lgb * test_lgb + w_cb * test_cb, 0, None)

cv_rmse = root_mean_squared_error(y, oof_blend)
print(f"Optimal weights: XGB={w_xgb:.4f}, LGB={w_lgb:.4f}, CB={w_cb:.4f}")
print(f"5-Fold CV RMSE: {cv_rmse:.5f}")

# 5. Save Final Submission
os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame({"id": np.arange(len(test_blend)), "fatalities": test_blend})
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved successfully.")
