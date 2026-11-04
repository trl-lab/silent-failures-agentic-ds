import os
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


def eval_gini(y_true, y_pred):
    """Normalized Gini coefficient is equivalent to 2 * AUC - 1 for binary classification."""
    return 2.0 * roc_auc_score(y_true, y_pred) - 1.0


def main():
    print("Loading data...")
    train_df = pd.read_parquet("./input/train.parquet")
    test_df = pd.read_parquet("./input/test.parquet")

    target_col = "target"
    id_col = "id"

    test_ids = test_df[id_col].values

    feature_cols = [c for c in train_df.columns if c not in [id_col, target_col]]
    cat_cols = [c for c in feature_cols if c.endswith("_cat")]

    # Engineer missing value count feature (-1 indicates missing in Porto Seguro data)
    train_df["missing_count"] = (train_df[feature_cols] == -1).sum(axis=1)
    test_df["missing_count"] = (test_df[feature_cols] == -1).sum(axis=1)
    feature_cols.append("missing_count")

    # Treat -1 as np.nan for numerical and categorical handling
    for col in feature_cols:
        train_df[col] = train_df[col].replace(-1, np.nan)
        test_df[col] = test_df[col].replace(-1, np.nan)

    # Convert categorical columns to category dtype for LightGBM
    for col in cat_cols:
        train_df[col] = train_df[col].astype("category")
        test_df[col] = test_df[col].astype("category")

    X = train_df[feature_cols]
    y = train_df[target_col].values
    X_test = test_df[feature_cols]

    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    oof_preds = np.zeros(len(train_df))
    test_preds = np.zeros(len(test_df))

    lgb_params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 100,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }

    print("Starting 5-Fold Cross-Validation...")
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_tr, y_tr = X.iloc[train_idx], y[train_idx]
        X_va, y_va = X.iloc[val_idx], y[val_idx]

        train_data = lgb.Dataset(
            X_tr, label=y_tr, categorical_feature=cat_cols, free_raw_data=False
        )
        val_data = lgb.Dataset(
            X_va,
            label=y_va,
            reference=train_data,
            categorical_feature=cat_cols,
            free_raw_data=False,
        )

        model = lgb.train(
            lgb_params,
            train_data,
            num_boost_round=1500,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
        )

        val_pred = model.predict(X_va, num_iteration=model.best_iteration)
        oof_preds[val_idx] = val_pred
        test_preds += (
            model.predict(X_test, num_iteration=model.best_iteration) / n_splits
        )

        fold_gini = eval_gini(y_va, val_pred)
        print(f"Fold {fold} Normalized Gini: {fold_gini:.5f}")

    cv_gini = eval_gini(y, oof_preds)
    print(f"\nOverall Out-of-Fold Normalized Gini: {cv_gini:.5f}")

    os.makedirs("./working", exist_ok=True)
    submission = pd.DataFrame({id_col: test_ids, target_col: test_preds})
    submission.to_csv("./working/submission.csv", index=False)
    print("Submission saved successfully to ./working/submission.csv")


if __name__ == "__main__":
    main()
