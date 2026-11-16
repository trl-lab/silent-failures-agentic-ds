import gc
import glob
import os
import re
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


def reduce_mem_usage(df):
    for col in df.columns:
        col_type = df[col].dtype
        if str(col_type)[:3] == "int":
            c_min = df[col].min()
            c_max = df[col].max()
            if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
            elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)
        elif str(col_type)[:5] == "float":
            c_min = df[col].min()
            c_max = df[col].max()
            if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                df[col] = df[col].astype(np.float32)
    return df


def load_and_aggregate_tables(data_dir, prefix="train"):
    # Locate base table
    base_file = os.path.join(data_dir, f"{prefix}_base.parquet")
    if not os.path.exists(base_file):
        # Fallback to single parquet if exists
        single_file = os.path.join(
            "./input",
            f"{prefix}.parquet" if prefix != "train" else "train.parquet",
        )
        if os.path.exists(single_file):
            return pd.read_parquet(single_file)
        raise FileNotFoundError(f"Base file not found for prefix {prefix}")

    df_base = pd.read_parquet(base_file)

    # Find all table files for prefix
    table_files = glob.glob(os.path.join(data_dir, f"{prefix}_*.parquet"))
    table_files = [f for f in table_files if not f.endswith(f"{prefix}_base.parquet")]

    # Process depth_0 tables
    depth_0_files = [
        f
        for f in table_files
        if ("_static_0" in f or "_static_cb_0" in f) and "_1" not in f
    ]
    for f in depth_0_files:
        df_sub = pd.read_parquet(f)
        df_sub = df_sub.drop_duplicates(subset=["case_id"])
        df_base = df_base.merge(df_sub, on="case_id", how="left")
        del df_sub
        gc.collect()

    # Process depth_1 and depth_2 tables with aggregations
    depth_other_files = [f for f in table_files if f not in depth_0_files]
    # Group file pairs (e.g. applprev_1_0 and applprev_1_1)
    file_groups = {}
    for f in depth_other_files:
        fname = os.path.basename(f)
        # Extract group prefix
        group_key = re.sub(r"_\d+\.parquet$", "", fname)
        if group_key not in file_groups:
            file_groups[group_key] = []
        file_groups[group_key].append(f)

    for group_key, files in file_groups.items():
        dfs = [pd.read_parquet(f) for f in files]
        df_sub = pd.concat(dfs, ignore_index=True)
        del dfs
        gc.collect()

        num_cols = [
            c
            for c in df_sub.columns
            if c not in ["case_id", "num_group1", "num_group2"]
            and pd.api.types.is_numeric_dtype(df_sub[c])
        ]
        cat_cols = [
            c
            for c in df_sub.columns
            if c not in ["case_id", "num_group1", "num_group2"]
            and not pd.api.types.is_numeric_dtype(df_sub[c])
        ]

        # Aggregate numeric features
        if num_cols:
            agg_dict = {c: ["max", "mean"] for c in num_cols[:40]}
            df_agg_num = df_sub.groupby("case_id").agg(agg_dict)
            df_agg_num.columns = [f"{col}_{agg}" for col, agg in df_agg_num.columns]
            df_agg_num = df_agg_num.reset_index()
            df_base = df_base.merge(df_agg_num, on="case_id", how="left")
            del df_agg_num

        # Aggregate count of records
        df_cnt = df_sub.groupby("case_id").size().reset_index(name=f"{group_key}_count")
        df_base = df_base.merge(df_cnt, on="case_id", how="left")
        del df_cnt, df_sub
        gc.collect()

    return df_base


def main():
    train_dir = "./input/train" if os.path.exists("./input/train") else "./input"
    test_dir = "./input/test" if os.path.exists("./input/test") else "./input"

    print("Loading and aggregating training data...")
    df_train = load_and_aggregate_tables(train_dir, prefix="train")
    print(f"Train data loaded: {df_train.shape}")

    print("Loading and aggregating test data...")
    df_test = load_and_aggregate_tables(test_dir, prefix="test")
    print(f"Test data loaded: {df_test.shape}")

    ignore_cols = ["case_id", "date_decision", "target", "MONTH", "WEEK_NUM"]
    feature_cols = [c for c in df_train.columns if c not in ignore_cols]

    # Align columns between train and test
    common_cols = [c for c in feature_cols if c in df_test.columns]

    # Remove columns with excessive missingness (>95%) or constant values
    valid_cols = []
    for c in common_cols:
        if df_train[c].isnull().mean() < 0.95 and df_train[c].nunique(dropna=False) > 1:
            valid_cols.append(c)

    print(f"Selected {len(valid_cols)} features for modeling.")

    # Convert object columns to category
    for c in valid_cols:
        if df_train[c].dtype == "object":
            df_train[c] = df_train[c].astype("category")
            df_test[c] = df_test[c].astype("category")

    df_train = reduce_mem_usage(df_train)
    df_test = reduce_mem_usage(df_test)
    gc.collect()

    X = df_train[valid_cols]
    y = df_train["target"].values
    X_test = df_test[valid_cols]

    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    oof_preds = np.zeros(len(df_train))
    test_preds = np.zeros(len(df_test))

    lgb_params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "n_estimators": 1000,
        "learning_rate": 0.05,
        "max_depth": 8,
        "num_leaves": 48,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, y_tr = X.iloc[train_idx], y[train_idx]
        X_va, y_va = X.iloc[val_idx], y[val_idx]

        model = LGBMClassifier(**lgb_params)
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[early_stopping(50, verbose=False), log_evaluation(-1)],
        )

        val_pred = model.predict_proba(X_va)[:, 1]
        oof_preds[val_idx] = val_pred
        test_preds += model.predict_proba(X_test)[:, 1] / n_splits

        fold_auc = roc_auc_score(y_va, val_pred)
        print(f"Fold {fold + 1} ROC AUC: {fold_auc:.5f}")

    cv_auc = roc_auc_score(y, oof_preds)
    print(f"\nOverall Validation ROC AUC: {cv_auc:.5f}")

    # Prepare submission
    os.makedirs("./working", exist_ok=True)
    submission = pd.DataFrame({"id": np.arange(len(test_preds)), "target": test_preds})
    submission.to_csv("./working/submission.csv", index=False)
    print("Submission successfully saved to ./working/submission.csv")


if __name__ == "__main__":
    main()
