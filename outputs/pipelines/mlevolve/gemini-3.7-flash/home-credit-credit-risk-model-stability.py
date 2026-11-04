import gc
import glob
import os
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit


def locate_files(data_dir):
    """Locate base and relational parquet files from either nested or flat directory structures."""
    if os.path.exists(os.path.join(data_dir, "train", "train_base.parquet")):
        train_dir = os.path.join(data_dir, "train")
        test_dir = os.path.join(data_dir, "test")
    elif os.path.exists(os.path.join(data_dir, "train_base.parquet")):
        train_dir = data_dir
        test_dir = data_dir
    else:
        train_dir = data_dir
        test_dir = data_dir
    return train_dir, test_dir


def get_table_files(base_dir, prefix):
    """Find all parquet files matching a given prefix pattern in the directory."""
    pattern = os.path.join(base_dir, f"{prefix}*.parquet")
    return sorted(glob.glob(pattern))


def downcast_dtypes(df):
    """Downcast float64 to float32 and int64 to int32 to optimize memory footprint."""
    for col in df.columns:
        col_type = df[col].dtype
        if col_type == "float64":
            df[col] = df[col].astype(np.float32)
        elif col_type == "int64" and col != "case_id":
            df[col] = df[col].astype(np.int32)
    return df


def process_dates(df, base_df):
    """Transform date columns to relative day offsets from date_decision."""
    date_cols = [
        c
        for c in df.columns
        if (c.endswith("D") or "date" in c.lower())
        and c not in ["date_decision", "case_id"]
    ]
    if not date_cols:
        return df

    date_decision_map = None
    if (
        "case_id" in df.columns
        and "date_decision" in base_df.columns
        and "case_id" in base_df.columns
    ):
        date_decision_map = base_df.set_index("case_id")["date_decision"].to_dict()

    for col in date_cols:
        try:
            date_s = pd.to_datetime(df[col], errors="coerce")
            if date_decision_map is not None:
                ref_dates = pd.to_datetime(
                    df["case_id"].map(date_decision_map), errors="coerce"
                )
                df[col] = (date_s - ref_dates).dt.days.astype(np.float32)
            else:
                df[col] = (date_s - pd.to_datetime("2020-01-01")).dt.days.astype(
                    np.float32
                )
        except Exception:
            df.drop(columns=[col], inplace=True, errors="ignore")
    return df


def aggregate_depth1_table(file_paths, base_df, prefix_name):
    """Aggregate depth-1 (1-to-many) table to case_id level with tailored aggregations."""
    if not file_paths:
        return None

    dfs = [pd.read_parquet(fp) for fp in file_paths]
    df = pd.concat(dfs, axis=0, ignore_index=True)
    del dfs
    gc.collect()

    if "case_id" not in df.columns:
        return None

    df = process_dates(df, base_df)

    num_cols = [
        c
        for c in df.columns
        if c not in ["case_id", "num_group1", "num_group2"]
        and (pd.api.types.is_numeric_dtype(df[c]) or c.endswith(("A", "P", "D")))
    ]
    for c in num_cols:
        if not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = pd.to_numeric(df[c], errors="coerce")

    cat_cols = [
        c
        for c in df.columns
        if c not in ["case_id", "num_group1", "num_group2"] and c not in num_cols
    ]

    agg_dict = {}
    for c in num_cols:
        agg_dict[c] = ["max", "mean", "min"]
    for c in cat_cols:
        agg_dict[c] = ["nunique"]

    if not agg_dict:
        return None

    grouped = df.groupby("case_id").agg(agg_dict)
    grouped.columns = [f"{prefix_name}_{col}_{stat}" for col, stat in grouped.columns]
    grouped.reset_index(inplace=True)
    grouped = downcast_dtypes(grouped)

    del df
    gc.collect()
    return grouped


def aggregate_depth2_table(file_paths, base_df, prefix_name):
    """Aggregate depth-2 (nested 1-to-many) table to case_id level."""
    if not file_paths:
        return None

    dfs = [pd.read_parquet(fp) for fp in file_paths]
    df = pd.concat(dfs, axis=0, ignore_index=True)
    del dfs
    gc.collect()

    if "case_id" not in df.columns:
        return None

    df = process_dates(df, base_df)

    num_cols = [
        c
        for c in df.columns
        if c not in ["case_id", "num_group1", "num_group2"]
        and (pd.api.types.is_numeric_dtype(df[c]) or c.endswith(("A", "P", "D")))
    ]
    for c in num_cols:
        if not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = pd.to_numeric(df[c], errors="coerce")

    agg_dict = {c: ["max", "mean"] for c in num_cols}
    if not agg_dict:
        return None

    grouped = df.groupby("case_id").agg(agg_dict)
    grouped.columns = [f"{prefix_name}_{col}_{stat}" for col, stat in grouped.columns]
    grouped.reset_index(inplace=True)
    grouped = downcast_dtypes(grouped)

    del df
    gc.collect()
    return grouped


def process_static_tables(file_paths, base_df, prefix_name):
    """Process depth-0 static tables and align to case_id."""
    if not file_paths:
        return None

    dfs = [pd.read_parquet(fp) for fp in file_paths]
    df = pd.concat(dfs, axis=0, ignore_index=True)
    del dfs
    gc.collect()

    if "case_id" not in df.columns:
        return None

    df = df.drop_duplicates(subset=["case_id"])
    df = process_dates(df, base_df)

    rename_dict = {c: f"{prefix_name}_{c}" for c in df.columns if c != "case_id"}
    df.rename(columns=rename_dict, inplace=True)
    df = downcast_dtypes(df)
    return df


def build_feature_table(base_dir, is_train=True):
    """Build unified feature matrix by joining base table with all aggregated relational tables."""
    base_file = os.path.join(
        base_dir, f"{'train' if is_train else 'test'}_base.parquet"
    )
    if not os.path.exists(base_file):
        alt_file = os.path.join(base_dir, f"{'train' if is_train else 'test'}.parquet")
        if os.path.exists(alt_file):
            base_file = alt_file

    base_df = pd.read_parquet(base_file)

    if "date_decision" in base_df.columns:
        dt = pd.to_datetime(base_df["date_decision"])
        base_df["decision_month"] = dt.dt.month.astype(np.int32)
        base_df["decision_day"] = dt.dt.day.astype(np.int32)
        base_df["decision_dayofweek"] = dt.dt.dayofweek.astype(np.int32)
        base_df["decision_year"] = dt.dt.year.astype(np.int32)

    table_configs = [
        ("static_0", "static", "depth0"),
        ("static_cb_0", "static_cb", "depth0"),
        ("applprev_1", "applprev1", "depth1"),
        ("applprev_2", "applprev2", "depth2"),
        ("credit_bureau_a_1", "cb_a1", "depth1"),
        ("credit_bureau_a_2", "cb_a2", "depth2"),
        ("credit_bureau_b_1", "cb_b1", "depth1"),
        ("credit_bureau_b_2", "cb_b2", "depth2"),
        ("tax_registry_a_1", "tax_a", "depth1"),
        ("tax_registry_b_1", "tax_b", "depth1"),
        ("tax_registry_c_1", "tax_c", "depth1"),
        ("other_1", "other1", "depth1"),
        ("person_1", "person1", "depth1"),
        ("person_2", "person2", "depth2"),
        ("deposit_1", "deposit1", "depth1"),
        ("debitcard_1", "debit1", "depth1"),
    ]

    prefix = "train" if is_train else "test"
    merged_df = base_df.copy()

    for pattern_suffix, prefix_name, depth_type in table_configs:
        pattern = f"{prefix}_{pattern_suffix}"
        files = get_table_files(base_dir, pattern)
        if not files:
            continue

        if depth_type == "depth0":
            agg_table = process_static_tables(files, base_df, prefix_name)
        elif depth_type == "depth1":
            agg_table = aggregate_depth1_table(files, base_df, prefix_name)
        elif depth_type == "depth2":
            agg_table = aggregate_depth2_table(files, base_df, prefix_name)
        else:
            agg_table = None

        if agg_table is not None:
            merged_df = merged_df.merge(agg_table, on="case_id", how="left")
            del agg_table
            gc.collect()

    merged_df = downcast_dtypes(merged_df)
    return merged_df


def handle_categorical_and_missing(train_df, valid_df, test_df):
    """Encode categoricals consistently and filter out zero-variance/high-missing features."""
    drop_meta_cols = ["case_id", "date_decision", "target", "WEEK_NUM", "MONTH"]
    candidate_features = [c for c in train_df.columns if c not in drop_meta_cols]

    missing_ratios = train_df[candidate_features].isnull().mean()
    valid_features = missing_ratios[missing_ratios < 0.95].index.tolist()

    final_features = []
    for col in valid_features:
        if train_df[col].dtype == "object" or train_df[col].dtype.name == "category":
            train_df[col] = train_df[col].astype("category")
            cat_categories = train_df[col].cat.categories
            train_df[col] = train_df[col].cat.codes.astype(np.int32)

            valid_df[col] = pd.Categorical(
                valid_df[col], categories=cat_categories
            ).codes.astype(np.int32)
            if test_df is not None and col in test_df.columns:
                test_df[col] = pd.Categorical(
                    test_df[col], categories=cat_categories
                ).codes.astype(np.int32)
            final_features.append(col)
        else:
            if train_df[col].std(skipna=True) > 1e-7:
                final_features.append(col)

    return train_df, valid_df, test_df, final_features


def get_lgb_model_params(custom_params=None):
    """Returns baseline LightGBM hyperparameters optimized for tabular credit risk."""
    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "learning_rate": 0.03,
        "num_leaves": 63,
        "max_depth": -1,
        "min_child_samples": 50,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 10.0,
        "n_estimators": 2500,
        "importance_type": "gain",
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }
    if custom_params:
        params.update(custom_params)
    return params


def get_model(model_type="lightgbm", params=None):
    """Instantiate and return classifier instance."""
    if model_type == "lightgbm":
        model_params = get_lgb_model_params(params)
        return lgb.LGBMClassifier(**model_params)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def get_loss_and_metric():
    """Returns loss objective name and validation metric callable."""
    loss_objective = "binary_logloss"

    def evaluation_metric(y_true, y_pred):
        return roc_auc_score(y_true, y_pred)

    return loss_objective, evaluation_metric


def main():
    working_dir = "./working"
    submission_dir = "./submission"
    os.makedirs(working_dir, exist_ok=True)
    os.makedirs(submission_dir, exist_ok=True)

    input_dir = "./input"
    train_dir, test_dir = locate_files(input_dir)

    # 1. Feature extraction and relational joining
    full_train = build_feature_table(train_dir, is_train=True)
    test_data = build_feature_table(test_dir, is_train=False)

    # 2. Stratified Validation Split (80% Train, 20% Validation)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(sss.split(full_train, full_train["target"]))

    train_split = full_train.iloc[train_idx].copy().reset_index(drop=True)
    valid_split = full_train.iloc[val_idx].copy().reset_index(drop=True)

    del full_train
    gc.collect()

    # 3. Clean, encode, and align features strictly learned on train
    train_split, valid_split, test_data, feature_cols = handle_categorical_and_missing(
        train_split, valid_split, test_data
    )

    for col in feature_cols:
        if col not in test_data.columns:
            test_data[col] = np.nan

    X_train, y_train = train_split[feature_cols], train_split["target"].values
    X_valid, y_valid = valid_split[feature_cols], valid_split["target"].values
    X_test = test_data[feature_cols]

    # 4. Model instantiation and training
    model = get_model("lightgbm")
    _, metric_fn = get_loss_and_metric()

    callbacks = [
        lgb.early_stopping(stopping_rounds=50, verbose=False),
        lgb.log_evaluation(period=0),
    ]

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="auc",
        callbacks=callbacks,
    )

    # 5. Model artifact serialization
    joblib.dump(model, os.path.join(working_dir, "lgb_model.joblib"))

    # 6. Evaluation on holdout validation set
    val_preds = model.predict_proba(X_valid)[:, 1]
    val_score = metric_fn(y_valid, val_preds)

    # 7. Test inference and submission generation
    test_preds = model.predict_proba(X_test)[:, 1]

    sample_sub_path = "./input/sample_submission.csv"
    submission_file = os.path.join(submission_dir, "submission.csv")

    if os.path.exists(sample_sub_path):
        sub_df = pd.read_csv(sample_sub_path)
        if "target" in sub_df.columns:
            sub_df["target"] = test_preds
        elif len(sub_df.columns) >= 2:
            sub_df.iloc[:, 1] = test_preds
        else:
            sub_df["target"] = test_preds
    else:
        sub_df = pd.DataFrame({"id": np.arange(len(test_data)), "target": test_preds})

    sub_df["target"] = sub_df["target"].fillna(0.0)
    sub_df.to_csv(submission_file, index=False)

    # 8. Print final validation score required by evaluation protocol
    print(f"Final Validation Score: {val_score}")


if __name__ == "__main__":
    main()
