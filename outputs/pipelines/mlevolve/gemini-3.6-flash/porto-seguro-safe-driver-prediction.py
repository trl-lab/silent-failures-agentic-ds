import os
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

# Ensure output directories exist
os.makedirs("./submission", exist_ok=True)
os.makedirs("./working", exist_ok=True)


# 1. Official Metric Definitions
def gini(y_true, y_pred):
    assert len(y_true) == len(y_pred)
    all_data = np.asarray(
        np.column_stack((y_true, y_pred, np.arange(len(y_true)))), dtype=float
    )
    all_data = all_data[np.lexsort((all_data[:, 2], -all_data[:, 1]))]
    total_losses = all_data[:, 0].sum()
    gini_sum = all_data[:, 0].cumsum().sum() / total_losses
    gini_sum -= (len(y_true) + 1) / 2.0
    return gini_sum / len(y_true)


def normalized_gini(y_true, y_pred):
    return gini(y_true, y_pred) / gini(y_true, y_true)


def lgb_normalized_gini(preds, train_data):
    labels = train_data.get_label()
    score = normalized_gini(labels, preds)
    return "normalized_gini", score, True


print("Starting data processing and feature engineering...")

# 2. Data Loading
train_path = "./input/train.parquet"
test_path = "./input/test.parquet"

train = pd.read_parquet(train_path)
test = pd.read_parquet(test_path)

# 3. Missing Value Counts & Binary Flag Row Sum
raw_features = [
    c for c in train.columns if c not in ["id", "target"] and not c.endswith("_calc")
]
train["num_missing"] = (train[raw_features] == -1).sum(axis=1)
test["num_missing"] = (test[raw_features] == -1).sum(axis=1)

bin_cols = [c for c in raw_features if c.endswith("_bin")]
train["sum_bin"] = (train[bin_cols] == 1).sum(axis=1)
test["sum_bin"] = (test[bin_cols] == 1).sum(axis=1)

# Convert missing sentinel -1 in categorical columns to non-negative category indices
cat_cols_raw = [
    "ps_ind_02_cat",
    "ps_ind_04_cat",
    "ps_ind_05_cat",
    "ps_car_01_cat",
    "ps_car_02_cat",
    "ps_car_03_cat",
    "ps_car_04_cat",
    "ps_car_05_cat",
    "ps_car_06_cat",
    "ps_car_07_cat",
    "ps_car_08_cat",
    "ps_car_09_cat",
    "ps_car_10_cat",
    "ps_car_11_cat",
    "ps_car_11",
]
for col in cat_cols_raw:
    if col in train.columns:
        train[col] = train[col].replace(-1, 999).astype(int)
        test[col] = test[col].replace(-1, 999).astype(int)

# 4. Domain & Interaction Features
for df in [train, test]:
    df["ps_car_13_x_ps_reg_03"] = df["ps_car_13"] * df["ps_reg_03"]
    df["ps_car_12_x_ps_car_13"] = df["ps_car_12"] * df["ps_car_13"]
    df["ps_reg_01_x_ps_reg_02"] = df["ps_reg_01"] * df["ps_reg_02"]
    df["ps_car_15_sq"] = df["ps_car_15"] ** 2
    df["ps_car_13_x_ps_car_15"] = df["ps_car_13"] * df["ps_car_15"]
    df["ps_car_13_x_ps_car_14"] = df["ps_car_13"] * df["ps_car_14"]
    df["ps_reg_03_int"] = np.round((df["ps_reg_03"] * 40) ** 2)

# High-cardinality interaction frequency
train["ind05_x_car11"] = (
    train["ps_ind_05_cat"].astype(str) + "_" + train["ps_car_11_cat"].astype(str)
)
test["ind05_x_car11"] = (
    test["ps_ind_05_cat"].astype(str) + "_" + test["ps_car_11_cat"].astype(str)
)

freq_inter = train["ind05_x_car11"].value_counts(normalize=True)
train["ind05_x_car11_freq"] = train["ind05_x_car11"].map(freq_inter).fillna(0)
test["ind05_x_car11_freq"] = test["ind05_x_car11"].map(freq_inter).fillna(0)

train.drop(columns=["ind05_x_car11"], inplace=True)
test.drop(columns=["ind05_x_car11"], inplace=True)

# Composite categorical interactions ('ps_ind_05_cat' x 'ps_car_01_cat' and 'ps_car_11_cat' x 'ps_car_01_cat')
train["ind05_x_car01"] = (
    train["ps_ind_05_cat"].astype(str) + "_" + train["ps_car_01_cat"].astype(str)
)
test["ind05_x_car01"] = (
    test["ps_ind_05_cat"].astype(str) + "_" + test["ps_car_01_cat"].astype(str)
)

train["car11_x_car01"] = (
    train["ps_car_11_cat"].astype(str) + "_" + train["ps_car_01_cat"].astype(str)
)
test["car11_x_car01"] = (
    test["ps_car_11_cat"].astype(str) + "_" + test["ps_car_01_cat"].astype(str)
)

# 5. Group Aggregations
num_agg_cols = ["ps_car_13", "ps_reg_03"]
for num_col in num_agg_cols:
    group_stats = train.groupby("ps_car_11_cat")[num_col].agg(["mean", "std"])
    group_stats.columns = [
        f"{num_col}_by_car11_mean",
        f"{num_col}_by_car11_std",
    ]

    train = train.merge(group_stats, on="ps_car_11_cat", how="left")
    test = test.merge(group_stats, on="ps_car_11_cat", how="left")

    train[f"{num_col}_by_car11_std"] = train[f"{num_col}_by_car11_std"].fillna(0)
    test[f"{num_col}_by_car11_std"] = test[f"{num_col}_by_car11_std"].fillna(0)

# Additional group aggregations of 'ps_car_13' grouped by 'ps_ind_05_cat' and 'ps_car_01_cat'
for cat_col in ["ps_ind_05_cat", "ps_car_01_cat"]:
    group_stats = train.groupby(cat_col)["ps_car_13"].agg(["mean", "std"])
    group_stats.columns = [
        f"ps_car_13_by_{cat_col}_mean",
        f"ps_car_13_by_{cat_col}_std",
    ]
    train = train.merge(group_stats, on=cat_col, how="left")
    test = test.merge(group_stats, on=cat_col, how="left")
    train[f"ps_car_13_by_{cat_col}_std"] = train[f"ps_car_13_by_{cat_col}_std"].fillna(0)
    test[f"ps_car_13_by_{cat_col}_std"] = test[f"ps_car_13_by_{cat_col}_std"].fillna(0)

group_stats_joint = train.groupby(["ps_ind_05_cat", "ps_car_01_cat"])["ps_car_13"].agg(["mean", "std"])
group_stats_joint.columns = ["ps_car_13_by_ind05_car01_mean", "ps_car_13_by_ind05_car01_std"]
train = train.merge(group_stats_joint, on=["ps_ind_05_cat", "ps_car_01_cat"], how="left")
test = test.merge(group_stats_joint, on=["ps_ind_05_cat", "ps_car_01_cat"], how="left")
train["ps_car_13_by_ind05_car01_std"] = train["ps_car_13_by_ind05_car01_std"].fillna(0)
test["ps_car_13_by_ind05_car01_std"] = test["ps_car_13_by_ind05_car01_std"].fillna(0)

# 6. Frequency Encoding
freq_cols = [
    "ps_car_11_cat",
    "ps_ind_05_cat",
    "ps_car_01_cat",
    "ps_car_06_cat",
    "ps_car_09_cat",
    "ps_ind_03",
    "ps_ind_15",
]
for col in freq_cols:
    freq = train[col].value_counts(normalize=True)
    train[f"{col}_freq"] = train[col].map(freq).fillna(0)
    test[f"{col}_freq"] = test[col].map(freq).fillna(0)

# 7. Leak-Free Out-Of-Fold Target Encoding
te_cols = [
    "ps_car_11_cat",
    "ps_ind_05_cat",
    "ps_car_01_cat",
    "ps_car_09_cat",
    "ps_ind_02_cat",
    "ps_car_04_cat",
    "ind05_x_car01",
    "car11_x_car01",
]
smoothing = 10.0
global_mean = train["target"].mean()

skf_te = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for col in te_cols:
    train[f"{col}_te"] = np.nan
    for tr_idx, val_idx in skf_te.split(train, train["target"]):
        tr_df = train.iloc[tr_idx]
        stats = tr_df.groupby(col)["target"].agg(["count", "mean"])
        smooth_te = (stats["count"] * stats["mean"] + smoothing * global_mean) / (
            stats["count"] + smoothing
        )
        train.loc[train.index[val_idx], f"{col}_te"] = train.iloc[val_idx][col].map(
            smooth_te
        )
    train[f"{col}_te"] = train[f"{col}_te"].fillna(global_mean)

    stats_full = train.groupby(col)["target"].agg(["count", "mean"])
    smooth_te_full = (
        stats_full["count"] * stats_full["mean"] + smoothing * global_mean
    ) / (stats_full["count"] + smoothing)
    test[f"{col}_te"] = test[col].map(smooth_te_full).fillna(global_mean)

# 8. Drop Uninformative _calc Features
calc_cols = [c for c in train.columns if "_calc" in c]
train.drop(columns=calc_cols, inplace=True, errors="ignore")
test.drop(columns=calc_cols, inplace=True, errors="ignore")
train.drop(columns=["ind05_x_car01", "car11_x_car01"], inplace=True, errors="ignore")
test.drop(columns=["ind05_x_car01", "car11_x_car01"], inplace=True, errors="ignore")

# Prepare feature matrices
feature_cols = [c for c in train.columns if c not in ["id", "target"]]
X = train[feature_cols]
y = train["target"].values
X_test = test[feature_cols]

cat_cols = [
    c
    for c in [
        "ps_ind_02_cat",
        "ps_ind_04_cat",
        "ps_ind_05_cat",
        "ps_car_01_cat",
        "ps_car_02_cat",
        "ps_car_03_cat",
        "ps_car_04_cat",
        "ps_car_05_cat",
        "ps_car_06_cat",
        "ps_car_07_cat",
        "ps_car_08_cat",
        "ps_car_09_cat",
        "ps_car_10_cat",
        "ps_car_11_cat",
        "ps_car_11",
    ]
    if c in feature_cols
]

print(
    f"Features count: {len(feature_cols)}, Train shape: {X.shape}, Test shape: {X_test.shape}"
)

# 9. Model Hyperparameters
lgb_params = {
    "objective": "binary",
    "metric": "None",
    "boosting_type": "gbdt",
    "learning_rate": 0.03,
    "num_leaves": 24,
    "max_depth": 5,
    "min_child_samples": 80,
    "subsample": 0.7,
    "colsample_bytree": 0.6,
    "reg_alpha": 2.0,
    "reg_lambda": 8.0,
    "random_state": 42,
    "n_jobs": -1,
}

xgb_params = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "tree_method": "hist",
    "learning_rate": 0.03,
    "max_depth": 4,
    "min_child_weight": 20,
    "subsample": 0.7,
    "colsample_bytree": 0.6,
    "gamma": 1.0,
    "reg_alpha": 3.0,
    "reg_lambda": 8.0,
    "random_state": 42,
    "n_jobs": -1,
    "n_estimators": 1500,
}

cb_params = {
    "loss_function": "Logloss",
    "eval_metric": "AUC",
    "learning_rate": 0.04,
    "depth": 5,
    "subsample": 0.7,
    "l2_leaf_reg": 8.0,
    "random_seed": 42,
    "iterations": 1200,
    "verbose": 0,
    "thread_count": -1,
}

# 10. Stratified 5-Fold Cross Validation Setup
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_lgb = np.zeros(len(train))
oof_xgb = np.zeros(len(train))
oof_cb = np.zeros(len(train))
oof_ensemble = np.zeros(len(train))

test_preds_lgb = np.zeros(len(test))
test_preds_xgb = np.zeros(len(test))
test_preds_cb = np.zeros(len(test))

# 11. Model Training & Evaluation Loop
print("Starting ensemble training across 5 folds...")
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_tr, y_tr = X.iloc[train_idx], y[train_idx]
    X_va, y_va = X.iloc[val_idx], y[val_idx]

    # Model 1: LightGBM
    dtrain = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat_cols)
    dval = lgb.Dataset(X_va, label=y_va, reference=dtrain, categorical_feature=cat_cols)

    model_lgb = lgb.train(
        lgb_params,
        dtrain,
        num_boost_round=1500,
        valid_sets=[dval],
        feval=lgb_normalized_gini,
        callbacks=[
            lgb.early_stopping(50, verbose=False),
            lgb.log_evaluation(-1),
        ],
    )
    val_p_lgb = model_lgb.predict(X_va)
    tst_p_lgb = model_lgb.predict(X_test)
    oof_lgb[val_idx] = val_p_lgb
    test_preds_lgb += tst_p_lgb / n_splits

    # Model 2: XGBoost
    model_xgb = xgb.XGBClassifier(**xgb_params, early_stopping_rounds=50)
    model_xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

    val_p_xgb = model_xgb.predict_proba(X_va)[:, 1]
    tst_p_xgb = model_xgb.predict_proba(X_test)[:, 1]
    oof_xgb[val_idx] = val_p_xgb
    test_preds_xgb += tst_p_xgb / n_splits

    # Model 3: CatBoost
    model_cb = CatBoostClassifier(**cb_params)
    model_cb.fit(
        X_tr,
        y_tr,
        eval_set=(X_va, y_va),
        cat_features=cat_cols,
        early_stopping_rounds=50,
        verbose=False,
    )

    val_p_cb = model_cb.predict_proba(X_va)[:, 1]
    tst_p_cb = model_cb.predict_proba(X_test)[:, 1]
    oof_cb[val_idx] = val_p_cb
    test_preds_cb += tst_p_cb / n_splits

    # Rank Averaging for Fold Ensemble
    r_lgb = rankdata(val_p_lgb) / len(val_p_lgb)
    r_xgb = rankdata(val_p_xgb) / len(val_p_xgb)
    r_cb = rankdata(val_p_cb) / len(val_p_cb)
    oof_ensemble[val_idx] = (r_lgb + r_xgb + r_cb) / 3.0

    fold_score = normalized_gini(y_va, oof_ensemble[val_idx])
    print(f"Fold {fold + 1} Normalized Gini Score: {fold_score:.5f}")

# 12. Dynamic Optimization of Ensemble Blending Weights
r_oof_lgb = rankdata(oof_lgb) / len(oof_lgb)
r_oof_xgb = rankdata(oof_xgb) / len(oof_xgb)
r_oof_cb = rankdata(oof_cb) / len(oof_cb)

def loss_func(weights):
    w = np.array(weights)
    w = w / np.sum(w)
    blend = w[0] * r_oof_lgb + w[1] * r_oof_xgb + w[2] * r_oof_cb
    return -normalized_gini(y, blend)

res = minimize(loss_func, [1/3, 1/3, 1/3], method="Nelder-Mead")
best_weights = res.x / np.sum(res.x)
print(f"Optimal Ensemble Weights: LightGBM={best_weights[0]:.4f}, XGBoost={best_weights[1]:.4f}, CatBoost={best_weights[2]:.4f}")

oof_ensemble = best_weights[0] * r_oof_lgb + best_weights[1] * r_oof_xgb + best_weights[2] * r_oof_cb
overall_score = normalized_gini(y, oof_ensemble)

# 13. Generate Submission File
r_test_lgb = rankdata(test_preds_lgb) / len(test_preds_lgb)
r_test_xgb = rankdata(test_preds_xgb) / len(test_preds_xgb)
r_test_cb = rankdata(test_preds_cb) / len(test_preds_cb)

final_test_preds = (
    best_weights[0] * r_test_lgb
    + best_weights[1] * r_test_xgb
    + best_weights[2] * r_test_cb
)

submission = pd.DataFrame({"id": test["id"].values, "target": final_test_preds})
submission.to_csv("./submission/submission.csv", index=False)

print(f"Final Validation Score: {overall_score:.5f}")