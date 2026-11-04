import os
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import OrdinalEncoder
import lightgbm as lgb
import xgboost as xgb
import catboost as cb

warnings.filterwarnings("ignore")

# 1. Load Data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "DEPSUM"
y = train_df[target_col].values

train_len = len(train_df)
combined_df = pd.concat(
    [train_df.drop(columns=[target_col]), test_df], axis=0
).reset_index(drop=True)

# Identify column types
cat_cols = []
num_cols = []

for c in combined_df.columns:
    if combined_df[c].dtype == "object" or combined_df[c].dtype.name == "category":
        cat_cols.append(c)
    else:
        num_cols.append(c)

# 2. Enhanced Feature Engineering
# Frequency encoding
for c in cat_cols:
    freq = combined_df[c].value_counts()
    combined_df[f"{c}_freq"] = combined_df[c].map(freq).fillna(0)

# Identify key categorical grouping columns with meaningful cardinality
key_cats = [c for c in cat_cols if 2 <= combined_df[c].nunique() <= 10000]
target_num_cols = num_cols[:8]  # Focus aggregations on primary numeric features

# Hierarchical group statistics and relative differences/ratios
for cat in key_cats:
    for num in target_num_cols:
        grp_mean = combined_df.groupby(cat)[num].transform("mean")
        grp_std = combined_df.groupby(cat)[num].transform("std").fillna(0)
        grp_min = combined_df.groupby(cat)[num].transform("min")
        grp_max = combined_df.groupby(cat)[num].transform("max")

        combined_df[f"{cat}_{num}_mean"] = grp_mean
        combined_df[f"{cat}_{num}_std"] = grp_std
        combined_df[f"{cat}_{num}_min"] = grp_min
        combined_df[f"{cat}_{num}_max"] = grp_max

        # Relative differences and ratios capturing local entity deviation
        combined_df[f"{cat}_{num}_diff"] = combined_df[num] - grp_mean
        combined_df[f"{cat}_{num}_ratio"] = combined_df[num] / (grp_mean.abs() + 1e-4)

# Ordinal encode categorical columns
if cat_cols:
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    combined_df[cat_cols] = encoder.fit_transform(combined_df[cat_cols].astype(str))

X = combined_df.iloc[:train_len].copy()
X_test = combined_df.iloc[train_len:].copy()

# 3. 5-Fold Cross Validation with Multi-Model Ensemble
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_lgb = np.zeros(train_len)
test_lgb = np.zeros(len(test_df))

oof_xgb = np.zeros(train_len)
test_xgb = np.zeros(len(test_df))

oof_cat = np.zeros(train_len)
test_cat = np.zeros(len(test_df))

lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.04,
    "num_leaves": 63,
    "min_child_samples": 25,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "n_estimators": 2500,
    "n_jobs": -1,
    "verbose": -1,
}

xgb_params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "learning_rate": 0.04,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "n_estimators": 2500,
    "n_jobs": -1,
    "early_stopping_rounds": 50,
}

cb_params = {
    "loss_function": "RMSE",
    "eval_metric": "RMSE",
    "learning_rate": 0.04,
    "depth": 6,
    "iterations": 2500,
    "random_seed": 42,
    "early_stopping_rounds": 50,
    "verbose": 0,
}

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_train, y_train = X.iloc[train_idx], y[train_idx]
    X_val, y_val = X.iloc[val_idx], y[val_idx]

    # LightGBM
    model_lgb = lgb.LGBMRegressor(**lgb_params)
    model_lgb.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    oof_lgb[val_idx] = model_lgb.predict(X_val)
    test_lgb += model_lgb.predict(X_test) / n_splits

    # XGBoost
    model_xgb = xgb.XGBRegressor(**xgb_params)
    model_xgb.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    oof_xgb[val_idx] = model_xgb.predict(X_val)
    test_xgb += model_xgb.predict(X_test) / n_splits

    # CatBoost
    model_cat = cb.CatBoostRegressor(**cb_params)
    model_cat.fit(
        X_train,
        y_train,
        eval_set=(X_val, y_val),
        verbose=False,
    )
    oof_cat[val_idx] = model_cat.predict(X_val)
    test_cat += model_cat.predict(X_test) / n_splits

lgb_rmse = root_mean_squared_error(y, oof_lgb)
xgb_rmse = root_mean_squared_error(y, oof_xgb)
cat_rmse = root_mean_squared_error(y, oof_cat)

print(f"LGBM Out-of-Fold RMSE: {lgb_rmse:.4f}")
print(f"XGBoost Out-of-Fold RMSE: {xgb_rmse:.4f}")
print(f"CatBoost Out-of-Fold RMSE: {cat_rmse:.4f}")


# 4. Optimal Blending Weights Optimization
def loss_func(weights):
    w = np.array(weights)
    w = w / np.sum(w)
    pred = w[0] * oof_lgb + w[1] * oof_xgb + w[2] * oof_cat
    return root_mean_squared_error(y, pred)


init_weights = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
bounds = [(0, 1), (0, 1), (0, 1)]
constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

opt_res = minimize(
    loss_func, init_weights, method="SLSQP", bounds=bounds, constraints=constraints
)
best_weights = opt_res.x / np.sum(opt_res.x)

print(f"Optimal Ensemble Weights (LGB, XGB, CAT): {best_weights.round(4)}")

oof_preds = (
    best_weights[0] * oof_lgb + best_weights[1] * oof_xgb + best_weights[2] * oof_cat
)
test_preds = (
    best_weights[0] * test_lgb + best_weights[1] * test_xgb + best_weights[2] * test_cat
)

cv_rmse = root_mean_squared_error(y, oof_preds)
print(f"Optimized Ensemble 5-Fold CV Out-of-Fold RMSE: {cv_rmse:.4f}")

# 5. Save Submission
os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame({"id": np.arange(len(test_df)), "DEPSUM": test_preds})
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved successfully to ./working/submission.csv")
