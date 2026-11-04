import gc
import glob
import os
import re
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

# Directory setup
input_dir = "./input"
working_dir = "./working"
os.makedirs(working_dir, exist_ok=True)

# Locate train and test directories
train_base_path = glob.glob(
    os.path.join(input_dir, "**", "*train_base*.parquet"), recursive=True
)[0]
train_dir = os.path.dirname(train_base_path)
test_base_path = glob.glob(
    os.path.join(input_dir, "**", "*test_base*.parquet"), recursive=True
)[0]
test_dir = os.path.dirname(test_base_path)


def process_data(data_dir, prefix="train"):
    base_file = glob.glob(os.path.join(data_dir, f"{prefix}_base*.parquet"))[0]
    df_base = pd.read_parquet(base_file)

    if "date_decision" in df_base.columns:
        df_base["date_decision"] = pd.to_datetime(df_base["date_decision"])
        df_base["decision_year"] = df_base["date_decision"].dt.year.astype("float32")
        df_base["decision_month"] = df_base["date_decision"].dt.month.astype("float32")
        df_base["decision_day"] = df_base["date_decision"].dt.day.astype("float32")
        df_base["decision_dayofweek"] = df_base["date_decision"].dt.dayofweek.astype(
            "float32"
        )
        df_base["decision_dayofyear"] = df_base["date_decision"].dt.dayofyear.astype(
            "float32"
        )
        df_base = df_base.drop(columns=["date_decision"])

    all_files = glob.glob(os.path.join(data_dir, f"{prefix}_*.parquet"))
    file_groups = {}
    for f in all_files:
        filename = os.path.basename(f)
        if filename.startswith(f"{prefix}_base"):
            continue
        m = re.match(rf"{prefix}_([a-zA-Z0-9_]+?)(_\d+)?\.parquet", filename)
        if m:
            group_name = m.group(1)
            file_groups.setdefault(group_name, []).append(f)

    for group_name, files in file_groups.items():
        dfs = [pd.read_parquet(f) for f in files]
        df_group = pd.concat(dfs, ignore_index=True)
        del dfs
        gc.collect()

        if df_group["case_id"].duplicated().any():
            num_cols = [
                c
                for c in df_group.columns
                if c not in ["case_id", "num_group1", "num_group2"]
                and df_group[c].dtype != "object"
            ]
            cat_cols = [
                c
                for c in df_group.columns
                if c not in ["case_id", "num_group1", "num_group2"]
                and df_group[c].dtype == "object"
            ]

            agg_dict = {}
            for col in num_cols:
                agg_dict[col] = ["max", "mean", "min", "std", "first", "last", "sum"]
            for col in cat_cols:
                agg_dict[col] = ["last"]

            if agg_dict:
                df_agg = df_group.groupby("case_id").agg(agg_dict)
                df_agg.columns = [
                    f"{group_name}_{col}_{stat}" for col, stat in df_agg.columns
                ]
                df_agg = df_agg.reset_index()
                df_base = df_base.merge(df_agg, on="case_id", how="left")
        else:
            cols_to_rename = {
                c: f"{group_name}_{c}" for c in df_group.columns if c != "case_id"
            }
            df_group = df_group.rename(columns=cols_to_rename)
            df_base = df_base.merge(df_group, on="case_id", how="left")

        del df_group
        gc.collect()

    for col in df_base.columns:
        if col in ["case_id", "target"]:
            continue
        if df_base[col].dtype == "object":
            df_base[col] = df_base[col].astype("category")
        elif df_base[col].dtype == "float64":
            df_base[col] = df_base[col].astype("float32")
        elif df_base[col].dtype == "int64":
            df_base[col] = df_base[col].astype("int32")

    return df_base


# Process train and test datasets
print("Processing training data...")
train_df = process_data(train_dir, prefix="train")
print(f"Train data shape: {train_df.shape}")

print("Processing testing data...")
test_df = process_data(test_dir, prefix="test")
print(f"Test data shape: {test_df.shape}")

# Align feature columns
features = [c for c in train_df.columns if c not in ["case_id", "target"]]
for col in features:
    if col not in test_df.columns:
        test_df[col] = np.nan

cat_cols = [c for c in features if str(train_df[c].dtype) == "category"]
for c in cat_cols:
    train_df[c] = train_df[c].astype("category")
    test_df[c] = test_df[c].astype("category")

X = train_df[features]
y = train_df["target"].values
X_test = test_df[features]

# Model setup and 5-fold cross-validation
params = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": 6,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "min_child_samples": 50,
    "n_estimators": 1500,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(train_df))
test_preds = np.zeros(len(test_df))

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_train, y_train = X.iloc[train_idx], y[train_idx]
    X_val, y_val = X.iloc[val_idx], y[val_idx]

    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
    )

    val_probs = model.predict_proba(X_val)[:, 1]
    oof_preds[val_idx] = val_probs
    fold_auc = roc_auc_score(y_val, val_probs)
    print(f"Fold {fold + 1} ROC-AUC: {fold_auc:.5f}")

    test_preds += model.predict_proba(X_test)[:, 1] / kf.n_splits

oof_auc = roc_auc_score(y, oof_preds)
print(f"OOF ROC-AUC Score: {oof_auc:.5f}")

# Save submission file
submission = pd.DataFrame({"id": np.arange(len(test_preds)), "target": test_preds})
submission.to_csv(os.path.join(working_dir, "submission.csv"), index=False)
print("Submission successfully saved to ./working/submission.csv")
