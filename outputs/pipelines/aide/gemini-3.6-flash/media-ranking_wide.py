import os
import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from xgboost import XGBRegressor

# 1. Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "Overall"
assert (
    target_col in train_df.columns
), f"Target column '{target_col}' missing from train data"

# Clean target
train_clean = train_df.dropna(subset=[target_col]).reset_index(drop=True)
y = train_clean[target_col].values
y_min, y_max = y.min(), y.max()

X_train = train_clean.drop(columns=[target_col])
X_test = test_df.copy()

# Remove ID-like columns from feature set if present
id_cols = [c for c in ["id", "ID", "Id"] if c in X_train.columns]
if id_cols:
    X_train = X_train.drop(columns=id_cols)
    X_test = X_test.drop(columns=id_cols, errors="ignore")

# Combine for uniform feature engineering
n_train = len(X_train)
all_df = pd.concat([X_train, X_test], axis=0, ignore_index=True)

# Datetime processing
datetime_cols = []
for col in all_df.columns:
    if (
        pd.api.types.is_datetime64_any_dtype(all_df[col])
        or "date" in col.lower()
        or "time" in col.lower()
    ):
        try:
            dt = pd.to_datetime(all_df[col], errors="coerce")
            if dt.notna().sum() > 0:
                all_df[f"{col}_year"] = dt.dt.year
                all_df[f"{col}_month"] = dt.dt.month
                all_df[f"{col}_day"] = dt.dt.day
                all_df[f"{col}_dayofweek"] = dt.dt.dayofweek
                datetime_cols.append(col)
        except Exception:
            pass

if datetime_cols:
    all_df = all_df.drop(columns=datetime_cols)

# Categorical processing and frequency encoding
cat_cols = [
    c
    for c in all_df.columns
    if all_df[c].dtype == "object" or pd.api.types.is_categorical_dtype(all_df[c])
]
for col in cat_cols:
    all_df[col] = all_df[col].astype(str)
    freq_map = all_df[col].value_counts().to_dict()
    all_df[f"{col}_freq"] = all_df[col].map(freq_map)

# Numeric aggregations
num_cols = list(all_df.select_dtypes(include=[np.number]).columns)
if len(num_cols) > 2:
    all_df["num_mean"] = all_df[num_cols].mean(axis=1)
    all_df["num_std"] = all_df[num_cols].std(axis=1)
    all_df["num_min"] = all_df[num_cols].min(axis=1)
    all_df["num_max"] = all_df[num_cols].max(axis=1)
    all_df["num_nulls"] = all_df[num_cols].isnull().sum(axis=1)

# Initialize dataframes for each model family
lgb_df = all_df.copy()
cb_df = all_df.copy()
xgb_df = all_df.copy()

for col in cat_cols:
    lgb_df[col] = lgb_df[col].astype("category")
    cb_df[col] = cb_df[col].fillna("missing").astype(str)
    xgb_df[col] = xgb_df[col].astype("category")

# Compute Out-Of-Fold Target Encodings
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for col in cat_cols:
    te_name = f"{col}_te"
    lgb_df[te_name] = 0.0
    cb_df[te_name] = 0.0
    xgb_df[te_name] = 0.0

te_test_acc = {col: np.zeros(len(X_test)) for col in cat_cols}
smoothing = 10.0

for fold, (train_idx, val_idx) in enumerate(kf.split(train_clean, y)):
    tr_y = y[train_idx]
    global_mean = tr_y.mean()

    for col in cat_cols:
        tr_cat = train_clean.iloc[train_idx][col].astype(str)
        val_cat = train_clean.iloc[val_idx][col].astype(str)
        te_cat = test_df[col].astype(str) if col in test_df.columns else None

        stats = (
            pd.DataFrame({"cat": tr_cat, "target": tr_y})
            .groupby("cat")["target"]
            .agg(["count", "mean"])
        )
        smooth_map = (stats["count"] * stats["mean"] + smoothing * global_mean) / (
            stats["count"] + smoothing
        )
        smooth_dict = smooth_map.to_dict()

        te_val_vals = val_cat.map(smooth_dict).fillna(global_mean).values
        te_tr_vals = tr_cat.map(smooth_dict).fillna(global_mean).values

        lgb_df.loc[val_idx, f"{col}_te"] = te_val_vals
        cb_df.loc[val_idx, f"{col}_te"] = te_val_vals
        xgb_df.loc[val_idx, f"{col}_te"] = te_val_vals

        lgb_df.loc[train_idx, f"{col}_te"] = te_tr_vals
        cb_df.loc[train_idx, f"{col}_te"] = te_tr_vals
        xgb_df.loc[train_idx, f"{col}_te"] = te_tr_vals

        if te_cat is not None:
            te_test_acc[col] += (
                te_cat.map(smooth_dict).fillna(global_mean).values / kf.n_splits
            )

for col in cat_cols:
    lgb_df.loc[n_train:, f"{col}_te"] = te_test_acc[col]
    cb_df.loc[n_train:, f"{col}_te"] = te_test_acc[col]
    xgb_df.loc[n_train:, f"{col}_te"] = te_test_acc[col]

# Split dataframes to train/test
X_tr_lgb, X_te_lgb = lgb_df.iloc[:n_train].copy(), lgb_df.iloc[n_train:].copy()
X_tr_cb, X_te_cb = cb_df.iloc[:n_train].copy(), cb_df.iloc[n_train:].copy()
X_tr_xgb, X_te_xgb = xgb_df.iloc[:n_train].copy(), xgb_df.iloc[n_train:].copy()

# 2. 5-Fold Cross-Validation Model Training
oof_lgb = np.zeros(n_train)
test_lgb = np.zeros(len(X_te_lgb))

oof_cb = np.zeros(n_train)
test_cb = np.zeros(len(X_te_cb))

oof_xgb = np.zeros(n_train)
test_xgb = np.zeros(len(X_te_xgb))

lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.03,
    "num_leaves": 31,
    "colsample_bytree": 0.8,
    "subsample": 0.8,
    "random_state": 42,
    "n_estimators": 2000,
    "verbose": -1,
}

cb_params = {
    "iterations": 2000,
    "learning_rate": 0.03,
    "depth": 6,
    "loss_function": "RMSE",
    "eval_metric": "RMSE",
    "random_seed": 42,
    "verbose": 0,
}

xgb_params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "learning_rate": 0.03,
    "max_depth": 6,
    "colsample_bytree": 0.8,
    "subsample": 0.8,
    "enable_categorical": True,
    "tree_method": "hist",
    "random_state": 42,
    "n_estimators": 2000,
    "early_stopping_rounds": 50,
}

for fold, (train_idx, val_idx) in enumerate(kf.split(X_tr_lgb, y)):
    y_tr_fold, y_val_fold = y[train_idx], y[val_idx]

    # LightGBM Fold
    X_tr_fold_lgb, X_val_fold_lgb = (
        X_tr_lgb.iloc[train_idx],
        X_tr_lgb.iloc[val_idx],
    )
    model_lgb = lgb.LGBMRegressor(**lgb_params)
    model_lgb.fit(
        X_tr_fold_lgb,
        y_tr_fold,
        eval_set=[(X_val_fold_lgb, y_val_fold)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    oof_lgb[val_idx] = model_lgb.predict(X_val_fold_lgb)
    test_lgb += model_lgb.predict(X_te_lgb) / kf.n_splits

    # CatBoost Fold
    X_tr_fold_cb, X_val_fold_cb = X_tr_cb.iloc[train_idx], X_tr_cb.iloc[val_idx]
    model_cb = CatBoostRegressor(
        **cb_params, cat_features=cat_cols if len(cat_cols) > 0 else None
    )
    model_cb.fit(
        X_tr_fold_cb,
        y_tr_fold,
        eval_set=(X_val_fold_cb, y_val_fold),
        early_stopping_rounds=50,
    )
    oof_cb[val_idx] = model_cb.predict(X_val_fold_cb)
    test_cb += model_cb.predict(X_te_cb) / kf.n_splits

    # XGBoost Fold
    X_tr_fold_xgb, X_val_fold_xgb = (
        X_tr_xgb.iloc[train_idx],
        X_tr_xgb.iloc[val_idx],
    )
    model_xgb = XGBRegressor(**xgb_params)
    model_xgb.fit(
        X_tr_fold_xgb,
        y_tr_fold,
        eval_set=[(X_val_fold_xgb, y_val_fold)],
        verbose=False,
    )
    oof_xgb[val_idx] = model_xgb.predict(X_val_fold_xgb)
    test_xgb += model_xgb.predict(X_te_xgb) / kf.n_splits


# 3. Optimize 3-Model Ensemble Weights
def objective(w):
    blend = w[0] * oof_lgb + w[1] * oof_cb + w[2] * oof_xgb
    blend = np.clip(blend, y_min, y_max)
    return np.sqrt(mean_squared_error(y, blend))


cons = {"type": "eq", "fun": lambda w: 1.0 - sum(w)}
bounds = [(0, 1), (0, 1), (0, 1)]
init_w = [1 / 3, 1 / 3, 1 / 3]

res = minimize(objective, init_w, method="SLSQP", bounds=bounds, constraints=cons)
best_w = res.x / np.sum(res.x)

oof_ensemble = np.clip(
    best_w[0] * oof_lgb + best_w[1] * oof_cb + best_w[2] * oof_xgb,
    y_min,
    y_max,
)
cv_rmse = np.sqrt(mean_squared_error(y, oof_ensemble))
print(f"Validation RMSE: {cv_rmse:.5f}")

# 4. Generate & Save Submission
test_ensemble = np.clip(
    best_w[0] * test_lgb + best_w[1] * test_cb + best_w[2] * test_xgb,
    y_min,
    y_max,
)
os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame(
    {"id": np.arange(len(test_ensemble)), "Overall": test_ensemble}
)
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved to ./working/submission.csv successfully.")
