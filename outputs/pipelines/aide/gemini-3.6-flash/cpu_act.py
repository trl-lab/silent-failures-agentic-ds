import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from scipy.optimize import minimize
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from xgboost import XGBRegressor

# Load train and test datasets
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")


def engineer_features(df):
    data = df.copy()

    # System call ratios & process activity
    scall_safe = data["scall"] + 1e-5
    runqsz_safe = data["runqsz"] + 1e-5
    data["sread_ratio"] = data["sread"] / scall_safe
    data["swrite_ratio"] = data["swrite"] / scall_safe
    data["other_scall"] = data["scall"] - data["sread"] - data["swrite"]
    data["fork_exec_ratio"] = (data["fork"] + data["exec"]) / scall_safe
    data["scall_per_runq"] = data["scall"] / runqsz_safe
    data["fork_per_runq"] = data["fork"] / runqsz_safe
    data["exec_per_runq"] = data["exec"] / runqsz_safe

    # Character transfer efficiency & volume
    char_total = data["rchar"] + data["wchar"]
    data["char_total"] = char_total
    data["rchar_ratio"] = data["rchar"] / (char_total + 1e-5)
    data["rchar_per_sread"] = data["rchar"] / (data["sread"] + 1e-5)
    data["wchar_per_swrite"] = data["wchar"] / (data["swrite"] + 1e-5)
    data["char_per_scall"] = char_total / scall_safe

    # Logical I/O metrics
    total_io = data["lread"] + data["lwrite"]
    data["total_io"] = total_io
    data["lread_ratio"] = data["lread"] / (total_io + 1e-5)

    # Memory & Paging metrics
    total_pflt = data["pflt"] + data["vflt"]
    data["total_pflt"] = total_pflt
    data["vflt_ratio"] = data["vflt"] / (total_pflt + 1e-5)
    data["ppgin_per_pgin"] = data["ppgin"] / (data["pgin"] + 1e-5)
    data["ppgout_per_pgout"] = data["ppgout"] / (data["pgout"] + 1e-5)

    total_mem_swap = data["freemem"] + data["freeswap"]
    data["free_mem_ratio"] = data["freemem"] / (total_mem_swap + 1e-5)
    data["free_mem_swap_ratio"] = data["freemem"] / (data["freeswap"] + 1e-5)
    data["free_mem_per_runqsz"] = data["freemem"] / runqsz_safe
    data["pgfree_per_pgscan"] = data["pgfree"] / (data["pgscan"] + 1e-5)

    # CPU non-user time sum & estimated user mode time if available
    cpu_cols = [c for c in ["sys", "wio", "idle"] if c in data.columns]
    if len(cpu_cols) > 0:
        data["cpu_non_user_sum"] = data[cpu_cols].sum(axis=1)
        data["est_usr"] = 100.0 - data["cpu_non_user_sum"]

    # Log transforms of skewed attributes
    skewed_cols = [
        "rchar",
        "wchar",
        "scall",
        "sread",
        "swrite",
        "freemem",
        "freeswap",
        "lread",
        "lwrite",
        "pgscan",
        "pgfree",
        "total_io",
        "char_total",
        "total_pflt",
    ]
    for col in skewed_cols:
        if col in data.columns:
            data[f"log1p_{col}"] = np.log1p(np.maximum(0, data[col]))

    return data


# Process features
train_feat = engineer_features(train_df)
test_feat = engineer_features(test_df)

target = "usr"
features = [c for c in train_feat.columns if c != target]

X = train_feat[features]
y = train_feat[target]
X_test = test_feat[features]

# Fill any residual NaNs with column medians to ensure compatibility
X_median = X.median()
X = X.fillna(X_median)
X_test = X_test.fillna(X_median)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

oof_cat = np.zeros(len(train_df))
oof_xgb = np.zeros(len(train_df))
oof_lgb = np.zeros(len(train_df))
oof_et = np.zeros(len(train_df))

test_cat = np.zeros(len(test_df))
test_xgb = np.zeros(len(test_df))
test_lgb = np.zeros(len(test_df))
test_et = np.zeros(len(test_df))

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

    # 1. CatBoost Regressor
    cat_model = CatBoostRegressor(
        iterations=2500,
        learning_rate=0.025,
        depth=6,
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=42 + fold,
        verbose=0,
    )
    cat_model.fit(
        X_train,
        y_train,
        eval_set=(X_val, y_val),
        early_stopping_rounds=50,
        verbose=False,
    )
    oof_cat[val_idx] = cat_model.predict(X_val)
    test_cat += cat_model.predict(X_test) / kf.n_splits

    # 2. XGBoost Regressor
    xgb_model = XGBRegressor(
        n_estimators=2500,
        learning_rate=0.025,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        n_jobs=-1,
        early_stopping_rounds=50,
    )
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    oof_xgb[val_idx] = xgb_model.predict(X_val)
    test_xgb += xgb_model.predict(X_test) / kf.n_splits

    # 3. LightGBM Regressor
    lgb_model = lgb.LGBMRegressor(
        n_estimators=2500,
        learning_rate=0.025,
        max_depth=6,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        n_jobs=-1,
        verbose=-1,
    )
    lgb_model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=-1),
        ],
    )
    oof_lgb[val_idx] = lgb_model.predict(X_val)
    test_lgb += lgb_model.predict(X_test) / kf.n_splits

    # 4. ExtraTrees Regressor
    et_model = ExtraTreesRegressor(
        n_estimators=300,
        max_depth=18,
        min_samples_split=4,
        max_features=0.7,
        random_state=42 + fold,
        n_jobs=-1,
    )
    et_model.fit(X_train, y_train)
    oof_et[val_idx] = et_model.predict(X_val)
    test_et += et_model.predict(X_test) / kf.n_splits


# Optimize blending weights on OOF predictions using SLSQP
def blend_objective(weights):
    w = np.array(weights)
    w = w / np.sum(w)
    pred = w[0] * oof_cat + w[1] * oof_xgb + w[2] * oof_lgb + w[3] * oof_et
    pred = np.clip(pred, 0.0, 100.0)
    return np.sqrt(mean_squared_error(y, pred))


init_weights = [0.25, 0.25, 0.25, 0.25]
bounds = [(0, 1)] * 4
constraints = {"type": "eq", "fun": lambda w: 1.0 - np.sum(w)}

res = minimize(
    blend_objective,
    init_weights,
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
)
best_weights = res.x / np.sum(res.x)

oof_ensemble = (
    best_weights[0] * oof_cat
    + best_weights[1] * oof_xgb
    + best_weights[2] * oof_lgb
    + best_weights[3] * oof_et
)
oof_ensemble = np.clip(oof_ensemble, 0.0, 100.0)

cv_rmse = np.sqrt(mean_squared_error(y, oof_ensemble))
print(f"Validation RMSE: {cv_rmse:.5f}")

# Generate ensemble test predictions
final_test_preds = (
    best_weights[0] * test_cat
    + best_weights[1] * test_xgb
    + best_weights[2] * test_lgb
    + best_weights[3] * test_et
)
final_test_preds = np.clip(final_test_preds, 0.0, 100.0)

# Save submission
submission = pd.DataFrame(
    {"id": np.arange(len(final_test_preds)), "usr": final_test_preds}
)
submission.to_csv("./working/submission.csv", index=False)
