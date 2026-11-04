import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error
from scipy.optimize import minimize
from catboost import CatBoostRegressor
import lightgbm as lgb
import warnings

warnings.filterwarnings("ignore")

# Create output directory
os.makedirs("./working", exist_ok=True)

# Load datasets
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "TOTAL_ACTIVE_ACCOUNTS_"
y = train_df[target_col].values
X_train = train_df.drop(columns=[target_col]).copy()
X_test = test_df.copy()

n_train = len(X_train)
df_all = pd.concat([X_train, X_test], axis=0).reset_index(drop=True)

# 1. Clean string-based numerical columns (remove $, %, commas)
for col in df_all.columns:
    if df_all[col].dtype == "object":
        sample_vals = df_all[col].dropna().astype(str)
        if (
            len(sample_vals) > 0
            and sample_vals.str.contains(r"^\$?[0-9,.]+%?$", regex=True).mean() > 0.7
        ):
            cleaned = (
                df_all[col]
                .astype(str)
                .str.replace("$", "", regex=False)
                .str.replace("%", "", regex=False)
                .str.replace(",", "", regex=False)
                .str.strip()
            )
            df_all[col] = pd.to_numeric(cleaned, errors="coerce")

# 2. Date parsing and temporal feature extraction
date_cols = []
for col in df_all.columns:
    if df_all[col].dtype == "object":
        try:
            parsed = pd.to_datetime(df_all[col], errors="coerce")
            if parsed.notnull().mean() > 0.6:
                date_cols.append(col)
                df_all[f"{col}_year"] = parsed.dt.year
                df_all[f"{col}_month"] = parsed.dt.month
                df_all[f"{col}_quarter"] = parsed.dt.quarter
                df_all[f"{col}_day"] = parsed.dt.day
                df_all[f"{col}_dayofweek"] = parsed.dt.dayofweek
                df_all[f"{col}_timestamp"] = parsed.astype("int64") // 10**9
        except Exception:
            pass

df_all = df_all.drop(columns=date_cols)

# 3. Identify Categorical & Numerical Columns
cat_cols = []
num_cols = []
for col in df_all.columns:
    if df_all[col].dtype == "object" or df_all[col].dtype.name == "category":
        df_all[col] = df_all[col].fillna("Missing").astype(str)
        freq = df_all[col].value_counts()
        df_all[f"{col}_freq"] = df_all[col].map(freq)
        cat_cols.append(col)
    elif df_all[col].nunique() < 20 and df_all[col].dtype not in [
        "float64",
        "float32",
        "int64",
        "int32",
    ]:
        df_all[col] = df_all[col].fillna("Missing").astype(str)
        cat_cols.append(col)
    else:
        num_cols.append(col)
        df_all[col] = df_all[col].fillna(df_all[col].median())

# 4. Pairwise Numerical Ratios & Differences
top_num = num_cols[:8]
for i in range(len(top_num)):
    for j in range(i + 1, len(top_num)):
        col_a, col_b = top_num[i], top_num[j]
        df_all[f"{col_a}_minus_{col_b}"] = df_all[col_a] - df_all[col_b]
        df_all[f"{col_a}_ratio_{col_b}"] = df_all[col_a] / (
            np.abs(df_all[col_b]) + 1e-4
        )

# 5. Group Aggregation Features on Top Categorical Columns
for cat in cat_cols[:6]:
    for num in num_cols[:8]:
        grouped = df_all.groupby(cat)[num]
        df_all[f"{cat}_{num}_mean"] = grouped.transform("mean")
        df_all[f"{cat}_{num}_std"] = grouped.transform("std").fillna(0)
        df_all[f"{cat}_{num}_min"] = grouped.transform("min")
        df_all[f"{cat}_{num}_max"] = grouped.transform("max")
        df_all[f"{cat}_{num}_median"] = grouped.transform("median")

# Prepare datasets for modeling
X_tr = df_all.iloc[:n_train].reset_index(drop=True).copy()
X_te = df_all.iloc[n_train:].reset_index(drop=True).copy()

# Encode categoricals for LightGBM
X_tr_lgb = X_tr.copy()
X_te_lgb = X_te.copy()
for col in cat_cols:
    X_tr_lgb[col] = X_tr_lgb[col].astype("category")
    X_te_lgb[col] = X_te_lgb[col].astype("category")

# 6. 5-Fold Cross-Validation Ensemble (CatBoost + LightGBM)
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

cb_oof = np.zeros(n_train)
lgb_oof = np.zeros(n_train)
cb_test = np.zeros(len(X_te))
lgb_test = np.zeros(len(X_te))

cat_features_idx = [X_tr.columns.get_loc(c) for c in cat_cols if c in X_tr.columns]

lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.025,
    "num_leaves": 31,
    "feature_fraction": 0.75,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "min_child_samples": 20,
    "verbose": -1,
    "random_state": 42,
    "n_jobs": -1,
}

for fold, (train_idx, val_idx) in enumerate(kf.split(X_tr, y)):
    # CatBoost training
    X_f_tr, y_f_tr = X_tr.iloc[train_idx], y[train_idx]
    X_f_val, y_f_val = X_tr.iloc[val_idx], y[val_idx]

    cb_model = CatBoostRegressor(
        iterations=2000,
        learning_rate=0.025,
        depth=6,
        l2_leaf_reg=3.0,
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=42 + fold,
        verbose=False,
        cat_features=cat_features_idx,
    )
    cb_model.fit(
        X_f_tr,
        y_f_tr,
        eval_set=(X_f_val, y_f_val),
        early_stopping_rounds=60,
        verbose=False,
    )
    cb_oof[val_idx] = cb_model.predict(X_f_val)
    cb_test += cb_model.predict(X_te) / n_splits

    # LightGBM training
    X_f_tr_lgb, X_f_val_lgb = X_tr_lgb.iloc[train_idx], X_tr_lgb.iloc[val_idx]
    trn_data = lgb.Dataset(X_f_tr_lgb, label=y_f_tr)
    val_data = lgb.Dataset(X_f_val_lgb, label=y_f_val, reference=trn_data)

    lgb_model = lgb.train(
        lgb_params,
        trn_data,
        num_boost_round=2000,
        valid_sets=[trn_data, val_data],
        callbacks=[lgb.early_stopping(60, verbose=False), lgb.log_evaluation(-1)],
    )
    lgb_oof[val_idx] = lgb_model.predict(
        X_f_val_lgb, num_iteration=lgb_model.best_iteration
    )
    lgb_test += (
        lgb_model.predict(X_te_lgb, num_iteration=lgb_model.best_iteration) / n_splits
    )


# 7. Optimal Weight Optimization for Blending
def blend_obj(weights):
    w1, w2 = weights
    pred = np.clip(w1 * cb_oof + w2 * lgb_oof, 0, None)
    return root_mean_squared_error(y, pred)


res = minimize(
    blend_obj,
    [0.5, 0.5],
    bounds=[(0, 1), (0, 1)],
    constraints={"type": "eq", "fun": lambda w: sum(w) - 1.0},
    method="SLSQP",
)
w_cb, w_lgb = res.x
print(f"Optimal Ensemble Weights: CatBoost={w_cb:.4f}, LightGBM={w_lgb:.4f}")

oof_preds = np.clip(w_cb * cb_oof + w_lgb * lgb_oof, 0, None)
test_preds = np.clip(w_cb * cb_test + w_lgb * lgb_test, 0, None)

cv_rmse = root_mean_squared_error(y, oof_preds)
print(f"5-Fold CV Optimized Ensemble Out-of-Fold RMSE: {cv_rmse:.5f}")

# 8. Save submission file
submission = pd.DataFrame(
    {"id": np.arange(len(test_preds)), "TOTAL_ACTIVE_ACCOUNTS_": test_preds}
)
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved successfully to ./working/submission.csv")
