import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import warnings

warnings.filterwarnings("ignore")

# Define paths
TRAIN_PATH = "./input/train.parquet"
TEST_PATH = "./input/test.parquet"
OUTPUT_DIR = "./working"
os.makedirs(OUTPUT_DIR, exist_ok=True)
SUBMISSION_PATH = os.path.join(OUTPUT_DIR, "submission.csv")


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Handle known anomalies in DAYS_EMPLOYED
    if "DAYS_EMPLOYED" in df.columns:
        df["DAYS_EMPLOYED_ANOM"] = (df["DAYS_EMPLOYED"] == 365243).astype(int)
        df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace({365243: np.nan})

    # Domain-specific interaction features
    if "AMT_CREDIT" in df.columns and "AMT_INCOME_TOTAL" in df.columns:
        df["CREDIT_INCOME_PERCENT"] = df["AMT_CREDIT"] / (df["AMT_INCOME_TOTAL"] + 1e-5)
    if "AMT_ANNUITY" in df.columns and "AMT_INCOME_TOTAL" in df.columns:
        df["ANNUITY_INCOME_PERCENT"] = df["AMT_ANNUITY"] / (
            df["AMT_INCOME_TOTAL"] + 1e-5
        )
    if "AMT_ANNUITY" in df.columns and "AMT_CREDIT" in df.columns:
        df["PAYMENT_RATE"] = df["AMT_ANNUITY"] / (df["AMT_CREDIT"] + 1e-5)
    if "DAYS_EMPLOYED" in df.columns and "DAYS_BIRTH" in df.columns:
        df["DAYS_EMPLOYED_PERCENT"] = df["DAYS_EMPLOYED"] / (df["DAYS_BIRTH"] + 1e-5)
    if "AMT_INCOME_TOTAL" in df.columns and "CNT_FAM_MEMBERS" in df.columns:
        df["INCOME_PER_PERSON"] = df["AMT_INCOME_TOTAL"] / (
            df["CNT_FAM_MEMBERS"] + 1e-5
        )
    if "AMT_GOODS_PRICE" in df.columns and "AMT_CREDIT" in df.columns:
        df["GOODS_PRICE_CREDIT_RATIO"] = df["AMT_GOODS_PRICE"] / (
            df["AMT_CREDIT"] + 1e-5
        )

    # Aggregations over EXT_SOURCE columns
    ext_cols = [
        c for c in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"] if c in df.columns
    ]
    if len(ext_cols) >= 2:
        df["EXT_SOURCE_MEAN"] = df[ext_cols].mean(axis=1)
        df["EXT_SOURCE_STD"] = df[ext_cols].std(axis=1)
        df["EXT_SOURCE_PROD"] = df[ext_cols].prod(axis=1)
        df["EXT_SOURCE_MIN"] = df[ext_cols].min(axis=1)
        df["EXT_SOURCE_MAX"] = df[ext_cols].max(axis=1)

    return df


def main():
    print("Loading data...")
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

    # Separate target
    target_col = "TARGET"
    y = train_df[target_col].values
    train_df = train_df.drop(columns=[target_col])

    # Drop identifier columns if present
    id_cols = [c for c in ["SK_ID_CURR", "id", "ID"] if c in train_df.columns]
    X_train_raw = train_df.drop(columns=id_cols)
    X_test_raw = test_df.drop(columns=[c for c in id_cols if c in test_df.columns])

    # Feature engineering
    print("Engineering features...")
    X_train = engineer_features(X_train_raw)
    X_test = engineer_features(X_test_raw)

    # Encode categorical features
    cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    for col in cat_cols:
        X_train[col] = X_train[col].astype("category")
        X_test[col] = pd.Categorical(
            X_test[col], categories=X_train[col].cat.categories
        )

    # LightGBM Parameters
    lgb_params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "min_child_samples": 30,
        "lambda_l1": 0.1,
        "lambda_l2": 1.0,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }

    # Stratified K-Fold Cross Validation
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    oof_preds = np.zeros(len(X_train))
    test_preds = np.zeros(len(X_test))

    print("Starting 5-fold cross-validation...")
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y)):
        X_tr, y_tr = X_train.iloc[train_idx], y[train_idx]
        X_va, y_val = X_train.iloc[val_idx], y[val_idx]

        train_data = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat_cols)
        val_data = lgb.Dataset(
            X_va, label=y_val, reference=train_data, categorical_feature=cat_cols
        )

        model = lgb.train(
            lgb_params,
            train_data,
            num_boost_round=2000,
            valid_sets=[train_data, val_data],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )

        val_pred = model.predict(X_va, num_iteration=model.best_iteration)
        oof_preds[val_idx] = val_pred

        fold_auc = roc_auc_score(y_val, val_pred)
        print(
            f"Fold {fold + 1} ROC AUC: {fold_auc:.5f} (Best Iteration: {model.best_iteration})"
        )

        test_preds += (
            model.predict(X_test, num_iteration=model.best_iteration) / n_splits
        )

    overall_auc = roc_auc_score(y, oof_preds)
    print(f"\nOverall Out-of-Fold ROC AUC: {overall_auc:.5f}")

    # Generate submission file
    submission = pd.DataFrame(
        {"id": np.arange(len(test_preds), dtype=int), "TARGET": test_preds}
    )
    submission.to_csv(SUBMISSION_PATH, index=False)
    print(
        f"Submission successfully saved to {SUBMISSION_PATH} with shape {submission.shape}"
    )


if __name__ == "__main__":
    main()
