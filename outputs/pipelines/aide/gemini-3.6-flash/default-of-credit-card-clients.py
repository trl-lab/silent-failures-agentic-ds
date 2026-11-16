import os
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")

# 1. Load Data
train_path = "./input/train.parquet"
test_path = "./input/test.parquet"

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

# Determine column naming structure dynamically
if "X1" in train_df.columns:
    limit_col = "X1"
    sex_col = "X2"
    edu_col = "X3"
    mar_col = "X4"
    age_col = "X5"
    pay_cols = ["X6", "X7", "X8", "X9", "X10", "X11"]
    bill_cols = ["X12", "X13", "X14", "X15", "X16", "X17"]
    pay_amt_cols = ["X18", "X19", "X20", "X21", "X22", "X23"]
else:
    limit_col = "LIMIT_BAL"
    sex_col = "SEX"
    edu_col = "EDUCATION"
    mar_col = "MARRIAGE"
    age_col = "AGE"
    pay_cols = [c for c in train_df.columns if c.startswith("PAY_") and "AMT" not in c]
    bill_cols = [c for c in train_df.columns if c.startswith("BILL_AMT")]
    pay_amt_cols = [c for c in train_df.columns if c.startswith("PAY_AMT")]

if "default payment next month" in train_df.columns:
    target_col = "default payment next month"
elif "Y" in train_df.columns:
    target_col = "Y"
else:
    target_col = [
        c
        for c in train_df.columns
        if c not in ["ID", "id"]
        and c not in pay_cols
        and c not in bill_cols
        and c not in pay_amt_cols
        and c not in [limit_col, sex_col, edu_col, mar_col, age_col]
    ][0]

full_features = pd.concat(
    [train_df.drop(columns=[target_col], errors="ignore"), test_df],
    axis=0,
    ignore_index=True,
)


# 2. Enhanced Feature Engineering Function
def engineer_features(df_input, reference_df):
    df = df_input.copy()

    # Data Cleaning for categorical values
    df[edu_col] = df[edu_col].apply(lambda x: 4 if x in [0, 5, 6] else x)
    df[mar_col] = df[mar_col].apply(lambda x: 3 if x == 0 else x)

    # Recency-weighted payment delay status
    recency_weights = [1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125]
    df["pay_recency_weighted"] = sum(
        df[pay_cols[i]] * recency_weights[i] for i in range(len(pay_cols))
    )

    # Monthly utilization, unpaid balance, and over-limit indicators
    for i in range(len(bill_cols)):
        df[f"util_{i}"] = df[bill_cols[i]] / (df[limit_col] + 1e-5)
        df[f"pay_to_limit_{i}"] = df[pay_amt_cols[i]] / (df[limit_col] + 1e-5)
        df[f"unpaid_{i}"] = df[bill_cols[i]] - df[pay_amt_cols[i]]
        df[f"is_overlimit_{i}"] = (df[bill_cols[i]] > df[limit_col]).astype(int)

    # Debt progression and compliance deltas month-over-month
    for i in range(len(bill_cols) - 1):
        df[f"pay_ratio_{i}"] = df[pay_amt_cols[i]] / (df[bill_cols[i + 1]].abs() + 1.0)
        df[f"unpaid_diff_{i}"] = df[f"unpaid_{i}"] - df[f"unpaid_{i+1}"]
        df[f"is_zero_pay_{i}"] = (
            (df[pay_amt_cols[i]] == 0) & (df[bill_cols[i + 1]] > 0)
        ).astype(int)

    # Financial risk behavior counts
    df["overlimit_count"] = df[
        [f"is_overlimit_{i}" for i in range(len(bill_cols))]
    ].sum(axis=1)
    df["zero_pay_count"] = df[
        [f"is_zero_pay_{i}" for i in range(len(bill_cols) - 1)]
    ].sum(axis=1)

    # Payment status aggregations
    df["pay_max"] = df[pay_cols].max(axis=1)
    df["pay_min"] = df[pay_cols].min(axis=1)
    df["pay_mean"] = df[pay_cols].mean(axis=1)
    df["pay_std"] = df[pay_cols].std(axis=1)
    df["pay_sum"] = df[pay_cols].sum(axis=1)
    df["num_delayed"] = (df[pay_cols] > 0).sum(axis=1)
    df["num_severe_delayed"] = (df[pay_cols] > 1).sum(axis=1)
    df["num_paid_duly"] = (df[pay_cols] <= 0).sum(axis=1)
    df["pay_trend"] = df[pay_cols[0]] - df[pay_cols[-1]]

    # Bill amount aggregations
    df["bill_mean"] = df[bill_cols].mean(axis=1)
    df["bill_max"] = df[bill_cols].max(axis=1)
    df["bill_min"] = df[bill_cols].min(axis=1)
    df["bill_std"] = df[bill_cols].std(axis=1)
    df["bill_sum"] = df[bill_cols].sum(axis=1)
    df["bill_trend"] = df[bill_cols[0]] - df[bill_cols[-1]]
    df["bill_to_limit"] = df["bill_mean"] / (df[limit_col] + 1e-5)

    # Payment amount aggregations
    df["pay_amt_mean"] = df[pay_amt_cols].mean(axis=1)
    df["pay_amt_max"] = df[pay_amt_cols].max(axis=1)
    df["pay_amt_min"] = df[pay_amt_cols].min(axis=1)
    df["pay_amt_std"] = df[pay_amt_cols].std(axis=1)
    df["pay_amt_sum"] = df[pay_amt_cols].sum(axis=1)
    df["pay_amt_to_limit"] = df["pay_amt_mean"] / (df[limit_col] + 1e-5)
    df["pay_amt_to_bill_mean"] = df["pay_amt_mean"] / (df["bill_mean"].abs() + 1.0)

    # Utilization aggregations
    util_cols = [f"util_{i}" for i in range(len(bill_cols))]
    df["util_mean"] = df[util_cols].mean(axis=1)
    df["util_max"] = df[util_cols].max(axis=1)
    df["util_std"] = df[util_cols].std(axis=1)

    # Interaction between recent payment status and credit utilization
    df["pay0_util0_interaction"] = df[pay_cols[0]] * df["util_0"]

    # Demographic relative ratio features
    df["limit_per_age"] = df[limit_col] / (df[age_col] + 1e-5)

    edu_limit_map = reference_df.groupby(edu_col)[limit_col].mean().to_dict()
    df["edu_limit_mean"] = df[edu_col].map(edu_limit_map)
    df["limit_rel_edu"] = df[limit_col] / (df["edu_limit_mean"] + 1e-5)

    edu_mar_limit_map = (
        reference_df.groupby([edu_col, mar_col])[limit_col].mean().to_dict()
    )
    df["limit_rel_edu_mar"] = df.apply(
        lambda r: r[limit_col]
        / (edu_mar_limit_map.get((r[edu_col], r[mar_col]), r[limit_col]) + 1e-5),
        axis=1,
    )

    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    return df


# Transform datasets
X_train_df = engineer_features(train_df, full_features)
X_test_df = engineer_features(test_df, full_features)

drop_cols = ["ID", "id", target_col]
y = train_df[target_col].values
X = X_train_df.drop(columns=drop_cols, errors="ignore")
X_test = X_test_df.drop(columns=drop_cols, errors="ignore")

# 3. Stratified 5-Fold Cross Validation Setup
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_lgb = np.zeros(len(X))
oof_xgb = np.zeros(len(X))
oof_cb = np.zeros(len(X))

test_lgb = np.zeros(len(X_test))
test_xgb = np.zeros(len(X_test))
test_cb = np.zeros(len(X_test))

# 4. Train Models Across Folds
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_tr, y_tr = X.iloc[train_idx], y[train_idx]
    X_va, y_va = X.iloc[val_idx], y[val_idx]

    # LightGBM Classifier
    lgb_model = lgb.LGBMClassifier(
        n_estimators=1500,
        learning_rate=0.03,
        max_depth=5,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        n_jobs=-1,
    )
    lgb_model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )
    oof_lgb[val_idx] = lgb_model.predict_proba(X_va)[:, 1]
    test_lgb += lgb_model.predict_proba(X_test)[:, 1] / n_splits

    # XGBoost Classifier
    xgb_model = xgb.XGBClassifier(
        n_estimators=1500,
        learning_rate=0.03,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        early_stopping_rounds=50,
        random_state=42 + fold,
        eval_metric="auc",
        n_jobs=-1,
    )
    xgb_model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    oof_xgb[val_idx] = xgb_model.predict_proba(X_va)[:, 1]
    test_xgb += xgb_model.predict_proba(X_test)[:, 1] / n_splits

    # CatBoost Classifier
    cb_model = CatBoostClassifier(
        iterations=1500,
        learning_rate=0.03,
        depth=5,
        eval_metric="AUC",
        early_stopping_rounds=50,
        random_seed=42 + fold,
        verbose=False,
        thread_count=-1,
    )
    cb_model.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=False)
    oof_cb[val_idx] = cb_model.predict_proba(X_va)[:, 1]
    test_cb += cb_model.predict_proba(X_test)[:, 1] / n_splits


# 5. Convert Predictions to Percentile Ranks for Optimal AUC Blending
oof_lgb_rank = rankdata(oof_lgb) / len(oof_lgb)
oof_xgb_rank = rankdata(oof_xgb) / len(oof_xgb)
oof_cb_rank = rankdata(oof_cb) / len(oof_cb)

test_lgb_rank = rankdata(test_lgb) / len(test_lgb)
test_xgb_rank = rankdata(test_xgb) / len(test_xgb)
test_cb_rank = rankdata(test_cb) / len(test_cb)


# 6. Optimize Ensemble Weights via SLSQP on Ranks
def loss_func(weights):
    w1, w2, w3 = weights
    pred = w1 * oof_lgb_rank + w2 * oof_xgb_rank + w3 * oof_cb_rank
    return -roc_auc_score(y, pred)


res = minimize(
    loss_func,
    [1 / 3, 1 / 3, 1 / 3],
    method="SLSQP",
    bounds=[(0, 1), (0, 1), (0, 1)],
    constraints={"type": "eq", "fun": lambda w: 1.0 - sum(w)},
)

weights = res.x
oof_ensemble = (
    weights[0] * oof_lgb_rank + weights[1] * oof_xgb_rank + weights[2] * oof_cb_rank
)
final_oof_auc = roc_auc_score(y, oof_ensemble)

# 7. Generate and Save Test Predictions
test_preds = (
    weights[0] * test_lgb_rank + weights[1] * test_xgb_rank + weights[2] * test_cb_rank
)

os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame(
    {"id": np.arange(len(test_df)), "default payment next month": test_preds}
)
submission.to_csv("./working/submission.csv", index=False)

print(f"Out-of-Fold ROC AUC Score: {final_oof_auc:.5f}")
