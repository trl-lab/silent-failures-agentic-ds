import os
import gc
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder


def create_features(df):
    df = df.copy()

    # Time features
    if "TransactionDT" in df.columns:
        # TransactionDT is given in seconds from a reference point
        df["hour"] = (df["TransactionDT"] // 3600) % 24
        df["day_of_week"] = (df["TransactionDT"] // (3600 * 24)) % 7

    # Transaction Amount features
    if "TransactionAmt" in df.columns:
        df["TransactionAmt_to_mean_card1"] = df["TransactionAmt"] / (
            df.groupby("card1")["TransactionAmt"].transform("mean") + 1e-5
        )
        df["TransactionAmt_to_std_card1"] = df["TransactionAmt"] / (
            df.groupby("card1")["TransactionAmt"].transform("std") + 1e-5
        )
        df["TransactionAmt_decimal"] = (
            (df["TransactionAmt"] - df["TransactionAmt"].astype(int)) * 1000
        ).astype(int)
        df["TransactionAmt_log"] = np.log1p(df["TransactionAmt"])

    # Email domain processing
    for col in ["P_emaildomain", "R_emaildomain"]:
        if col in df.columns:
            df[f"{col}_prefix"] = (
                df[col].astype(str).apply(lambda x: x.split(".")[0] if "." in x else x)
            )

    return df


def main():
    print("Loading data...")
    train = pd.read_parquet("./input/train.parquet")
    test = pd.read_parquet("./input/test.parquet")

    target_col = "isFraud"
    y = train[target_col].values

    # Save test length for later
    n_test = len(test)

    # Feature engineering
    print("Engineering features...")
    df_all = pd.concat(
        [train.drop(columns=[target_col], errors="ignore"), test], ignore_index=True
    )
    del train, test
    gc.collect()

    df_all = create_features(df_all)

    # Identify categorical columns and label encode them
    cat_cols = [
        c
        for c in df_all.columns
        if df_all[c].dtype == "object" or df_all[c].dtype.name == "category"
    ]
    for col in cat_cols:
        le = LabelEncoder()
        df_all[col] = le.fit_transform(df_all[col].astype(str).fillna("missing"))

    # Frequency encoding for high cardinality columns
    freq_cols = ["card1", "card2", "card3", "card5", "addr1", "addr2"]
    for col in freq_cols:
        if col in df_all.columns:
            freq = df_all[col].value_counts(dropna=False)
            df_all[f"{col}_freq"] = df_all[col].map(freq)

    # Split back to train and test
    X = df_all.iloc[:-n_test].copy()
    X_test = df_all.iloc[-n_test:].copy()

    # Drop non-feature ID columns if present
    cols_to_drop = [c for c in ["TransactionID", "TransactionDT"] if c in X.columns]
    X = X.drop(columns=cols_to_drop)
    X_test = X_test.drop(columns=cols_to_drop)

    features = list(X.columns)
    print(f"Total features: {len(features)}")

    # 5-fold Stratified CV
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))

    lgb_params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 256,
        "max_depth": -1,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "min_child_samples": 20,
        "verbosity": -1,
        "random_state": 42,
        "n_jobs": -1,
    }

    print("Starting 5-fold cross validation...")
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, y_train = X.iloc[train_idx], y[train_idx]
        X_val, y_val = X.iloc[val_idx], y[val_idx]

        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        model = lgb.train(
            lgb_params,
            train_data,
            num_boost_round=2000,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
        )

        oof_preds[val_idx] = model.predict(X_val, num_iteration=model.best_iteration)
        test_preds += (
            model.predict(X_test, num_iteration=model.best_iteration) / n_splits
        )

        fold_auc = roc_auc_score(y_val, oof_preds[val_idx])
        print(f"Fold {fold + 1} ROC AUC: {fold_auc:.5f}")

    overall_auc = roc_auc_score(y, oof_preds)
    print(f"\nOverall Out-of-Fold ROC AUC: {overall_auc:.5f}")

    # Prepare submission
    os.makedirs("./working", exist_ok=True)
    submission = pd.DataFrame({"id": np.arange(len(test_preds)), "isFraud": test_preds})
    submission_path = "./working/submission.csv"
    submission.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path} with shape {submission.shape}")


if __name__ == "__main__":
    main()
