import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error
import lightgbm as lgb


def main():
    # 1. Load data
    train_path = "./input/train.parquet"
    test_path = "./input/test.parquet"

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    target_col = "CAPACITY"
    y = train_df[target_col].values

    # Drop target from train features
    train_features = train_df.drop(columns=[target_col])
    test_features = test_df.copy()

    # Combine for consistent feature engineering
    n_train = len(train_features)
    combined = pd.concat([train_features, test_features], axis=0).reset_index(drop=True)

    # 2. Feature Engineering & Preprocessing
    drop_cols = []

    # Process columns
    for col in combined.columns:
        # Check if column is datetime
        if pd.api.types.is_datetime64_any_dtype(combined[col]):
            combined[f"{col}_year"] = combined[col].dt.year
            combined[f"{col}_month"] = combined[col].dt.month
            combined[f"{col}_day"] = combined[col].dt.day
            drop_cols.append(col)
        # Check if column is object/string
        elif combined[col].dtype == "object" or pd.api.types.is_string_dtype(
            combined[col]
        ):
            # String length feature
            combined[f"{col}_len"] = combined[col].astype(str).str.len()
            # Frequency encoding
            freq = combined[col].value_counts(dropna=False)
            combined[f"{col}_freq"] = combined[col].map(freq)
            # Categorical type for LightGBM
            combined[col] = combined[col].astype("category")
        elif pd.api.types.is_numeric_dtype(combined[col]):
            # Leave numeric features as is
            pass
        else:
            try:
                combined[col] = combined[col].astype("category")
            except Exception:
                drop_cols.append(col)

    if drop_cols:
        combined.drop(columns=drop_cols, inplace=True)

    X_train = combined.iloc[:n_train].reset_index(drop=True)
    X_test = combined.iloc[n_train:].reset_index(drop=True)

    categorical_features = [
        col for col in X_train.columns if X_train[col].dtype.name == "category"
    ]

    # 3. Model Training with 5-Fold Cross-Validation
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    oof_preds = np.zeros(n_train)
    test_preds = np.zeros(len(X_test))

    lgb_params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "min_child_samples": 20,
        "random_state": 42,
        "verbosity": -1,
        "n_jobs": -1,
    }

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y)):
        X_tr, y_tr = X_train.iloc[train_idx], y[train_idx]
        X_va, y_val = X_train.iloc[val_idx], y[val_idx]

        trn_data = lgb.Dataset(
            X_tr,
            label=y_tr,
            categorical_feature=categorical_features,
            free_raw_data=False,
        )
        val_data = lgb.Dataset(
            X_va,
            label=y_val,
            reference=trn_data,
            categorical_feature=categorical_features,
            free_raw_data=False,
        )

        model = lgb.train(
            lgb_params,
            trn_data,
            num_boost_round=1500,
            valid_sets=[trn_data, val_data],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=-1),
            ],
        )

        val_pred = model.predict(X_va, num_iteration=model.best_iteration)
        oof_preds[val_idx] = val_pred

        test_preds += (
            model.predict(X_test, num_iteration=model.best_iteration) / n_splits
        )

    # Capacity is typically non-negative
    oof_preds = np.clip(oof_preds, 0, None)
    test_preds = np.clip(test_preds, 0, None)

    cv_rmse = root_mean_squared_error(y, oof_preds)
    print(f"5-Fold CV Out-Of-Fold RMSE: {cv_rmse:.4f}")

    # 4. Save Submission
    os.makedirs("./working", exist_ok=True)
    submission = pd.DataFrame(
        {"id": np.arange(len(test_preds)), "CAPACITY": test_preds}
    )
    submission_path = "./working/submission.csv"
    submission.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path} with shape {submission.shape}")


if __name__ == "__main__":
    main()
