import os
import gc
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# 1. Load data
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

# Extract target and drop from train
target = train["isFraud"].values
train = train.drop(columns=["isFraud"])

n_train = len(train)
df = pd.concat([train, test], ignore_index=True)
del train, test
gc.collect()

# 2. Feature Engineering
# Row-wise missing value count
df["null_count"] = df.isnull().sum(axis=1).astype(np.int16)

# Time features
df["hour"] = (df["TransactionDT"] / 3600) % 24
df["day_of_week"] = (df["TransactionDT"] / (3600 * 24)) % 7

# Transaction Amount features
df["TransactionAmt_log"] = np.log1p(df["TransactionAmt"])
df["TransactionAmt_cents"] = (
    df["TransactionAmt"] - np.floor(df["TransactionAmt"])
).round(4)

# Interaction feature
if "card1" in df.columns and "addr1" in df.columns:
    df["card1_addr1"] = df["card1"].astype(str) + "_" + df["addr1"].astype(str)

# Email domain parsing and matching
if "P_emaildomain" in df.columns:
    p_str = df["P_emaildomain"].astype(str)
    df["P_emaildomain_vendor"] = p_str.str.split(".").str[0]
    df["P_emaildomain_suffix"] = p_str.str.split(".").str[-1]

if "R_emaildomain" in df.columns:
    r_str = df["R_emaildomain"].astype(str)
    df["R_emaildomain_vendor"] = r_str.str.split(".").str[0]
    df["R_emaildomain_suffix"] = r_str.str.split(".").str[-1]

if "P_emaildomain" in df.columns and "R_emaildomain" in df.columns:
    df["email_exact_match"] = (
        df["P_emaildomain"].astype(str) == df["R_emaildomain"].astype(str)
    ).astype(np.int8)

# Frequency Encoding
freq_cols = [
    c
    for c in [
        "card1",
        "card2",
        "card3",
        "card4",
        "card5",
        "card6",
        "addr1",
        "P_emaildomain",
        "R_emaildomain",
        "P_emaildomain_vendor",
        "R_emaildomain_vendor",
        "P_emaildomain_suffix",
        "R_emaildomain_suffix",
        "ProductCD",
        "DeviceInfo",
        "card1_addr1",
    ]
    if c in df.columns
]
for col in freq_cols:
    df[f"{col}_freq"] = df[col].map(df[col].value_counts(dropna=False))

# Group Aggregation features
if "card1" in df.columns:
    card1_amt_mean = df.groupby("card1")["TransactionAmt"].transform("mean")
    card1_amt_std = df.groupby("card1")["TransactionAmt"].transform("std")
    df["TransactionAmt_to_mean_card1"] = df["TransactionAmt"] / (card1_amt_mean + 1e-5)
    df["TransactionAmt_to_std_card1"] = df["TransactionAmt"] / (card1_amt_std + 1e-5)

# Convert categorical columns to category dtype for LightGBM
cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
for c in cat_cols:
    df[c] = df[c].astype("category")

# Split back into train and test
X_train = df.iloc[:n_train].copy()
X_test = df.iloc[n_train:].copy()
del df
gc.collect()

# 3. Model Training & Evaluation with 5-Fold Stratified CV
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_preds = np.zeros(len(X_train))
test_preds = np.zeros(len(X_test))

params = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 128,
    "max_depth": -1,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "verbosity": -1,
    "random_state": 42,
    "n_jobs": -1,
}

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, target)):
    X_tr, y_tr = X_train.iloc[train_idx], target[train_idx]
    X_va, y_va = X_train.iloc[val_idx], target[val_idx]

    trn_data = lgb.Dataset(X_tr, label=y_tr)
    val_data = lgb.Dataset(X_va, label=y_va)

    model = lgb.train(
        params,
        trn_data,
        num_boost_round=1500,
        valid_sets=[trn_data, val_data],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
    )

    val_pred = model.predict(X_va, num_iteration=model.best_iteration)
    oof_preds[val_idx] = val_pred
    test_preds += (
        model.predict(X_test, num_iteration=model.best_iteration) / skf.n_splits
    )

    fold_auc = roc_auc_score(y_va, val_pred)
    print(f"Fold {fold + 1} ROC-AUC: {fold_auc:.5f}")

overall_auc = roc_auc_score(target, oof_preds)
print(f"Overall OOF ROC-AUC: {overall_auc:.5f}")

# 4. Generate Submission File
os.makedirs("./working", exist_ok=True)
sub = pd.DataFrame({"id": np.arange(len(X_test)), "isFraud": test_preds})
sub.to_csv("./working/submission.csv", index=False)
print("Submission saved successfully to ./working/submission.csv")
