import os
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "default payment next month"
if target_col not in train_df.columns:
    # Fallback if target column has alternate name
    target_candidates = [
        c for c in train_df.columns if "default" in c.lower() or c == "Y"
    ]
    target_col = target_candidates[0]


# Standardize column mapping if original UCI names or alias names are used
def get_col(df, *candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def engineer_features(df):
    data = df.copy()

    # Identify key columns flexibly
    col_limit = get_col(data, "LIMIT_BAL", "X1")
    col_sex = get_col(data, "SEX", "X2")
    col_edu = get_col(data, "EDUCATION", "X3")
    col_mar = get_col(data, "MARRIAGE", "X4")
    col_age = get_col(data, "AGE", "X5")

    pay_cols = [
        get_col(data, f"PAY_{i}", f"PAY_0" if i == 1 else f"PAY_{i}", f"X{5+i}")
        for i in range(1, 7)
    ]
    pay_cols = [c for c in pay_cols if c is not None]

    bill_cols = [get_col(data, f"BILL_AMT{i}", f"X{11+i}") for i in range(1, 7)]
    bill_cols = [c for c in bill_cols if c is not None]

    pay_amt_cols = [get_col(data, f"PAY_AMT{i}", f"X{17+i}") for i in range(1, 7)]
    pay_amt_cols = [c for c in pay_amt_cols if c is not None]

    # Delinquency stats
    if pay_cols:
        data["pay_max"] = data[pay_cols].max(axis=1)
        data["pay_min"] = data[pay_cols].min(axis=1)
        data["pay_mean"] = data[pay_cols].mean(axis=1)
        data["pay_std"] = data[pay_cols].std(axis=1).fillna(0)
        data["pay_delay_count"] = (data[pay_cols] > 0).sum(axis=1)
        data["pay_recent_diff"] = data[pay_cols[0]] - data[pay_cols[-1]]

    # Bill amount stats and utilization
    if bill_cols and col_limit:
        data["bill_sum"] = data[bill_cols].sum(axis=1)
        data["bill_mean"] = data[bill_cols].mean(axis=1)
        data["bill_max"] = data[bill_cols].max(axis=1)
        data["bill_min"] = data[bill_cols].min(axis=1)
        data["bill_std"] = data[bill_cols].std(axis=1).fillna(0)

        # Credit limit utilization
        for i, b in enumerate(bill_cols, 1):
            data[f"utilization_{i}"] = data[b] / (data[col_limit] + 1.0)

        data["mean_utilization"] = data["bill_mean"] / (data[col_limit] + 1.0)
        data["max_utilization"] = data["bill_max"] / (data[col_limit] + 1.0)

        # Bill trend
        for i in range(len(bill_cols) - 1):
            data[f"bill_diff_{i+1}_{i+2}"] = data[bill_cols[i]] - data[bill_cols[i + 1]]

    # Payment amount stats
    if pay_amt_cols:
        data["pay_amt_sum"] = data[pay_amt_cols].sum(axis=1)
        data["pay_amt_mean"] = data[pay_amt_cols].mean(axis=1)
        data["pay_amt_std"] = data[pay_amt_cols].std(axis=1).fillna(0)

        # Payment to bill ratio
        if bill_cols:
            for i, (p, b) in enumerate(zip(pay_amt_cols, bill_cols), 1):
                data[f"pay_to_bill_ratio_{i}"] = data[p] / (
                    np.maximum(data[b], 0) + 1.0
                )
            data["overall_pay_ratio"] = data["pay_amt_sum"] / (
                np.maximum(data["bill_sum"], 0) + 1.0
            )

    # Demographic combinations
    if col_sex and col_edu:
        data["sex_edu"] = data[col_sex].astype(str) + "_" + data[col_edu].astype(str)
    if col_mar and col_edu:
        data["mar_edu"] = data[col_mar].astype(str) + "_" + data[col_edu].astype(str)

    return data


train_feat = engineer_features(train_df)
test_feat = engineer_features(test_df)

# Drop ID and target from feature set
drop_cols = [c for c in ["ID", "id", target_col] if c in train_feat.columns]
feature_cols = [c for c in train_feat.columns if c not in drop_cols]

# Identify categorical features
cat_candidates = [
    "SEX",
    "X2",
    "EDUCATION",
    "X3",
    "MARRIAGE",
    "X4",
    "sex_edu",
    "mar_edu",
]
cat_cols = [c for c in cat_candidates if c in feature_cols]

# Ensure string types for categoricals in CatBoost
for c in cat_cols:
    train_feat[c] = train_feat[c].astype(str).fillna("missing")
    test_feat[c] = test_feat[c].astype(str).fillna("missing")

X = train_feat[feature_cols]
y = train_df[target_col].values
X_test = test_feat[feature_cols]

# 5-fold Stratified Cross-Validation
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_preds = np.zeros(len(train_df))
test_preds = np.zeros(len(test_df))

cb_params = {
    "iterations": 1500,
    "learning_rate": 0.03,
    "depth": 5,
    "loss_function": "Logloss",
    "eval_metric": "AUC",
    "random_seed": 42,
    "verbose": False,
    "l2_leaf_reg": 4.0,
}

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train_f, y_train_f = X.iloc[train_idx], y[train_idx]
    X_val_f, y_val_f = X.iloc[val_idx], y[val_idx]

    train_pool = Pool(X_train_f, y_train_f, cat_features=cat_cols)
    val_pool = Pool(X_val_f, y_val_f, cat_features=cat_cols)
    test_pool = Pool(X_test, cat_features=cat_cols)

    model = CatBoostClassifier(**cb_params)
    model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=100, verbose=False)

    val_pred = model.predict_proba(val_pool)[:, 1]
    oof_preds[val_idx] = val_pred
    test_preds += model.predict_proba(test_pool)[:, 1] / n_splits

    fold_auc = roc_auc_score(y_val_f, val_pred)
    print(f"Fold {fold + 1} ROC-AUC: {fold_auc:.5f}")

overall_auc = roc_auc_score(y, oof_preds)
print(f"Overall OOF ROC-AUC: {overall_auc:.5f}")

# Save test predictions
os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame(
    {"id": np.arange(len(test_df)), "default payment next month": test_preds}
)
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved successfully to ./working/submission.csv")
