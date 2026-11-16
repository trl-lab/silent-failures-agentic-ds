import os
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from scipy.optimize import minimize

# 1. Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "usr"
features_to_exclude = [target_col]


def engineer_features(df):
    data = df.copy()
    eps = 1e-5

    # Memory and IO ratios
    data["io_rw_ratio"] = (data["lread"] + eps) / (data["lwrite"] + eps)
    data["char_rw_ratio"] = (data["rchar"] + eps) / (data["wchar"] + eps)
    data["sys_call_rw_ratio"] = (data["sread"] + eps) / (data["swrite"] + eps)

    # Total activity measures
    data["total_chars"] = data["rchar"] + data["wchar"]
    data["total_sys_rw"] = data["sread"] + data["swrite"]
    data["total_page_faults"] = data["pflt"] + data["vflt"]
    data["total_paging_req"] = data["pgin"] + data["pgout"]
    data["total_pages_moved"] = data["ppgin"] + data["ppgout"]

    # Average sizes and rates
    data["avg_read_chars"] = (data["rchar"] + eps) / (data["sread"] + eps)
    data["avg_write_chars"] = (data["wchar"] + eps) / (data["swrite"] + eps)
    data["pages_per_pgin"] = (data["ppgin"] + eps) / (data["pgin"] + eps)
    data["pages_per_pgout"] = (data["ppgout"] + eps) / (data["pgout"] + eps)
    data["pgscan_efficiency"] = (data["pgfree"] + eps) / (data["pgscan"] + eps)

    # Proportion of system calls
    data["sread_frac"] = (data["sread"] + eps) / (data["scall"] + eps)
    data["swrite_frac"] = (data["swrite"] + eps) / (data["scall"] + eps)
    data["fork_frac"] = (data["fork"] + eps) / (data["scall"] + eps)
    data["exec_frac"] = (data["exec"] + eps) / (data["scall"] + eps)
    data["io_sys_call_frac"] = (data["total_sys_rw"] + eps) / (data["scall"] + eps)

    # System pressure indicators
    data["runq_mem_ratio"] = (data["runqsz"] + eps) / (data["freemem"] + eps)
    data["mem_swap_ratio"] = (data["freemem"] + eps) / (data["freeswap"] + eps)
    data["page_fault_intensity"] = (data["total_page_faults"] + eps) / (
        data["scall"] + eps
    )
    data["runq_x_scall"] = data["runqsz"] * np.log1p(data["scall"])

    # Log transforms for skewed volume features
    skewed_cols = [
        "scall",
        "sread",
        "swrite",
        "rchar",
        "wchar",
        "pgin",
        "ppgin",
        "pgout",
        "ppgout",
        "pflt",
        "vflt",
    ]
    for col in skewed_cols:
        if col in data.columns:
            data[f"log_{col}"] = np.log1p(np.maximum(0, data[col]))

    return data


# Feature transformation
train_feat = engineer_features(train_df)
test_feat = engineer_features(test_df)

feature_cols = [c for c in train_feat.columns if c not in features_to_exclude]

X = train_feat[feature_cols].values
y = train_feat[target_col].values
X_test = test_feat[feature_cols].values

# 2. 5-Fold Cross Validation Setup
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_cat = np.zeros(len(train_feat))
test_cat = np.zeros(len(test_feat))

oof_lgb = np.zeros(len(train_feat))
test_lgb = np.zeros(len(test_feat))

oof_xgb = np.zeros(len(train_feat))
test_xgb = np.zeros(len(test_feat))

# Model configurations
cat_params = {
    "iterations": 3000,
    "learning_rate": 0.03,
    "depth": 7,
    "loss_function": "RMSE",
    "eval_metric": "RMSE",
    "random_seed": 42,
    "verbose": False,
}

lgb_params = {
    "n_estimators": 3000,
    "learning_rate": 0.03,
    "num_leaves": 31,
    "max_depth": 7,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}

xgb_params = {
    "n_estimators": 3000,
    "learning_rate": 0.03,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "n_jobs": -1,
    "early_stopping_rounds": 100,
}

print("Training diverse gradient boosted models across 5 folds...")

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_train_f, y_train_f = X[train_idx], y[train_idx]
    X_val_f, y_val_f = X[val_idx], y[val_idx]

    # 1. CatBoost
    cat_model = CatBoostRegressor(**cat_params)
    cat_model.fit(
        X_train_f,
        y_train_f,
        eval_set=(X_val_f, y_val_f),
        early_stopping_rounds=100,
        verbose=False,
    )
    oof_cat[val_idx] = cat_model.predict(X_val_f)
    test_cat += cat_model.predict(X_test) / n_splits

    # 2. LightGBM
    lgb_model = lgb.LGBMRegressor(**lgb_params)
    lgb_model.fit(
        X_train_f,
        y_train_f,
        eval_set=[(X_val_f, y_val_f)],
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(-1)],
    )
    oof_lgb[val_idx] = lgb_model.predict(X_val_f)
    test_lgb += lgb_model.predict(X_test) / n_splits

    # 3. XGBoost
    xgb_model = xgb.XGBRegressor(**xgb_params)
    xgb_model.fit(
        X_train_f,
        y_train_f,
        eval_set=[(X_val_f, y_val_f)],
        verbose=False,
    )
    oof_xgb[val_idx] = xgb_model.predict(X_val_f)
    test_xgb += xgb_model.predict(X_test) / n_splits

    fold_cat_rmse = np.sqrt(mean_squared_error(y_val_f, oof_cat[val_idx]))
    fold_lgb_rmse = np.sqrt(mean_squared_error(y_val_f, oof_lgb[val_idx]))
    fold_xgb_rmse = np.sqrt(mean_squared_error(y_val_f, oof_xgb[val_idx]))
    print(
        f"Fold {fold + 1} - CatBoost: {fold_cat_rmse:.4f} | LGB: {fold_lgb_rmse:.4f} | XGB: {fold_xgb_rmse:.4f}"
    )

cat_rmse = np.sqrt(mean_squared_error(y, oof_cat))
lgb_rmse = np.sqrt(mean_squared_error(y, oof_lgb))
xgb_rmse = np.sqrt(mean_squared_error(y, oof_xgb))
print(
    f"\nIndividual OOF RMSE: CatBoost={cat_rmse:.4f}, LightGBM={lgb_rmse:.4f}, XGBoost={xgb_rmse:.4f}"
)

# 3. Find optimal ensemble weights using constrained optimization
oof_matrix = np.column_stack([oof_cat, oof_lgb, oof_xgb])
test_matrix = np.column_stack([test_cat, test_lgb, test_xgb])


def loss_func(weights):
    weights = weights / np.sum(weights)
    blend = oof_matrix @ weights
    return np.sqrt(mean_squared_error(y, blend))


init_weights = np.array([1.0 / 3, 1.0 / 3, 1.0 / 3])
bounds = [(0, 1)] * 3
constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

opt_res = minimize(
    loss_func, init_weights, method="SLSQP", bounds=bounds, constraints=constraints
)
best_weights = opt_res.x / np.sum(opt_res.x)

print(
    f"Optimal weights: CatBoost={best_weights[0]:.4f}, LightGBM={best_weights[1]:.4f}, XGBoost={best_weights[2]:.4f}"
)

oof_ensemble = np.clip(oof_matrix @ best_weights, 0, 100)
test_ensemble = np.clip(test_matrix @ best_weights, 0, 100)

ensemble_rmse = np.sqrt(mean_squared_error(y, oof_ensemble))
print(f"Overall Ensemble OOF RMSE: {ensemble_rmse:.4f}")

# 4. Save Submission
os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame({"id": np.arange(len(test_ensemble)), "usr": test_ensemble})
submission.to_csv("./working/submission.csv", index=False)
print("Submission successfully saved to ./working/submission.csv")
