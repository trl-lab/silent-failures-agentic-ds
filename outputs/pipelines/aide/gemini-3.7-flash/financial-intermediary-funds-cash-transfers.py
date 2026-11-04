import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error


def main():
    train_path = "./input/train.parquet"
    test_path = "./input/test.parquet"

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    target_col = "Amount in USD"

    # Separate target and features
    y = train_df[target_col].values
    train_features = train_df.drop(columns=[target_col]).copy()
    test_features = test_df.copy()

    # Process features
    combined = pd.concat([train_features, test_features], axis=0, ignore_index=True)

    # Detect and process date/time columns
    for col in combined.columns:
        if (
            combined[col].dtype == "object"
            or isinstance(combined[col].dtype, pd.DatetimeTZDtype)
            or np.issubdtype(combined[col].dtype, np.datetime64)
        ):
            # Attempt datetime conversion if string looks like date
            try:
                converted = pd.to_datetime(combined[col], errors="coerce")
                # If majority of non-null values parsed successfully, treat as datetime
                if (
                    converted.notnull().sum() > 0.5 * combined[col].notnull().sum()
                    and converted.notnull().sum() > 0
                ):
                    combined[f"{col}_year"] = converted.dt.year
                    combined[f"{col}_month"] = converted.dt.month
                    combined[f"{col}_day"] = converted.dt.day
                    combined[f"{col}_dayofweek"] = converted.dt.dayofweek
                    combined[f"{col}_dayofyear"] = converted.dt.dayofyear
                    combined.drop(columns=[col], inplace=True)
            except Exception:
                pass

    # Identify categorical and numerical columns
    cat_cols = []
    for col in combined.columns:
        if combined[col].dtype == "object" or str(combined[col].dtype) == "category":
            combined[col] = combined[col].astype("category")
            cat_cols.append(col)
        elif combined[col].dtype == "bool":
            combined[col] = combined[col].astype(int)

    n_train = len(train_df)
    X = combined.iloc[:n_train].copy().reset_index(drop=True)
    X_test = combined.iloc[n_train:].copy().reset_index(drop=True)

    # 5-Fold Cross-Validation Setup
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    oof_preds = np.zeros(n_train)
    test_preds = np.zeros(len(test_df))

    params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "n_estimators": 1500,
        "verbose": -1,
    }

    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        X_tr, y_tr = X.iloc[train_idx], y[train_idx]
        X_va, y_va = X.iloc[val_idx], y[val_idx]

        train_data = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat_cols)
        val_data = lgb.Dataset(
            X_va, label=y_va, reference=train_data, categorical_feature=cat_cols
        )

        model = lgb.train(
            params,
            train_data,
            valid_sets=[train_data, val_data],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )

        val_pred = model.predict(X_va, num_iteration=model.best_iteration)
        oof_preds[val_idx] = val_pred

        test_pred_fold = model.predict(X_test, num_iteration=model.best_iteration)
        test_preds += test_pred_fold / n_splits

    cv_rmse = root_mean_squared_error(y, oof_preds)
    print(f"5-Fold CV RMSE: {cv_rmse:.4f}")

    # Save submission file
    os.makedirs("./working", exist_ok=True)
    submission = pd.DataFrame({"id": np.arange(len(test_df)), target_col: test_preds})
    submission_path = "./working/submission.csv"
    submission.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path} with shape {submission.shape}")


if __name__ == "__main__":
    main()
