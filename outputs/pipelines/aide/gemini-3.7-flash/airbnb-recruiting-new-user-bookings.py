import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb


def compute_ndcg_at_k(y_true, y_pred_proba, k=5):
    """
    Computes NDCG@k for multiclass single-label predictions.
    y_true: 1D array of ground truth class indices
    y_pred_proba: 2D array (N, num_classes) of predicted class probabilities
    """
    top_k = np.argsort(y_pred_proba, axis=1)[:, ::-1][:, :k]
    hits = top_k == y_true[:, None]
    weights = 1.0 / np.log2(np.arange(2, k + 2))
    ndcg = (hits * weights).sum(axis=1)
    return np.mean(ndcg)


def preprocess_data(train_df, test_df):
    target_col = "country_destination"

    # Save target and drop from train
    y_raw = train_df[target_col].copy()
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)

    n_train = len(train_df)
    n_test = len(test_df)

    # Combine for consistent feature engineering
    df = pd.concat(
        [train_df.drop(columns=[target_col], errors="ignore"), test_df],
        ignore_index=True,
    )

    # Drop leakage column
    if "date_first_booking" in df.columns:
        df.drop(columns=["date_first_booking"], inplace=True)

    # Process age
    # Replace birth years entered as age (e.g. 1920-2000)
    df["age_cleaned"] = df["age"].copy()
    mask_year = (df["age_cleaned"] >= 1900) & (df["age_cleaned"] <= 2015)
    df.loc[mask_year, "age_cleaned"] = 2015 - df.loc[mask_year, "age_cleaned"]
    # Filter unreasonable ages
    mask_invalid = (df["age_cleaned"] < 14) | (df["age_cleaned"] > 100)
    df.loc[mask_invalid, "age_cleaned"] = np.nan
    df["age_is_na"] = df["age"].isna().astype(int)

    # Process date_account_created
    df["date_account_created"] = pd.to_datetime(
        df["date_account_created"], errors="coerce"
    )
    df["dac_year"] = df["date_account_created"].dt.year
    df["dac_month"] = df["date_account_created"].dt.month
    df["dac_day"] = df["date_account_created"].dt.day
    df["dac_dayofweek"] = df["date_account_created"].dt.dayofweek
    df["dac_dayofyear"] = df["date_account_created"].dt.dayofyear
    df["dac_quarter"] = df["date_account_created"].dt.quarter

    # Process timestamp_first_active
    tfa = pd.to_datetime(
        df["timestamp_first_active"].astype(str), format="%Y%m%d%H%M%S", errors="coerce"
    )
    df["tfa_year"] = tfa.dt.year
    df["tfa_month"] = tfa.dt.month
    df["tfa_day"] = tfa.dt.day
    df["tfa_hour"] = tfa.dt.hour
    df["tfa_dayofweek"] = tfa.dt.dayofweek
    df["tfa_dayofyear"] = tfa.dt.dayofyear

    # Time difference between first active and account created (in days)
    df["diff_dac_tfa_days"] = (df["date_account_created"] - tfa.dt.floor("D")).dt.days

    # Drop original raw date/timestamp/id columns
    cols_to_drop = ["date_account_created", "timestamp_first_active", "id", "age"]
    df.drop(columns=[col for col in cols_to_drop if col in df.columns], inplace=True)

    # Identify categorical columns
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    # Frequency encode high-cardinality categoricals
    for col in cat_cols:
        freq = df[col].value_counts(dropna=False)
        df[f"{col}_freq"] = df[col].map(freq)
        df[col] = df[col].astype(str).astype("category")

    X_train = df.iloc[:n_train].reset_index(drop=True)
    X_test = df.iloc[n_train:].reset_index(drop=True)

    return X_train, y, X_test, label_encoder


def main():
    print("Loading data...")
    train_df = pd.read_parquet("./input/train.parquet")
    test_df = pd.read_parquet("./input/test.parquet")

    print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

    X, y, X_test, label_encoder = preprocess_data(train_df, test_df)
    num_classes = len(label_encoder.classes_)

    print(f"Preprocessed train shape: {X.shape}, Number of classes: {num_classes}")

    # 5-fold Stratified Cross-Validation
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    oof_preds = np.zeros((len(X), num_classes), dtype=np.float32)
    test_preds = np.zeros((len(X_test), num_classes), dtype=np.float32)

    params = {
        "objective": "multiclass",
        "num_class": num_classes,
        "metric": "multi_logloss",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "max_depth": 7,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "min_child_samples": 20,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }

    fold_ndcg_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, y_tr = X.iloc[train_idx], y[train_idx]
        X_va, y_val = X.iloc[val_idx], y[val_idx]

        dtrain = lgb.Dataset(X_tr, label=y_tr)
        dval = lgb.Dataset(X_va, label=y_val, reference=dtrain)

        callbacks = [
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0),
        ]

        model = lgb.train(
            params, dtrain, num_boost_round=1000, valid_sets=[dval], callbacks=callbacks
        )

        val_preds = model.predict(X_va, num_iteration=model.best_iteration)
        oof_preds[val_idx] = val_preds

        fold_ndcg = compute_ndcg_at_k(y_val, val_preds, k=5)
        fold_ndcg_scores.append(fold_ndcg)
        print(f"Fold {fold + 1} NDCG@5: {fold_ndcg:.5f}")

        test_preds += (
            model.predict(X_test, num_iteration=model.best_iteration) / n_splits
        )

    overall_ndcg = compute_ndcg_at_k(y, oof_preds, k=5)
    print(f"\nMean 5-Fold NDCG@5: {np.mean(fold_ndcg_scores):.5f}")
    print(f"Overall OOF NDCG@5: {overall_ndcg:.5f}")

    # Generate test predictions
    top_pred_indices = np.argmax(test_preds, axis=1)
    predicted_countries = label_encoder.inverse_transform(top_pred_indices)

    os.makedirs("./working", exist_ok=True)
    submission = pd.DataFrame(
        {
            "id": np.arange(len(predicted_countries)),
            "country_destination": predicted_countries,
        }
    )

    submission_path = "./working/submission.csv"
    submission.to_csv(submission_path, index=False)
    print(
        f"Submission successfully saved to {submission_path} with {len(submission)} rows."
    )


if __name__ == "__main__":
    main()
