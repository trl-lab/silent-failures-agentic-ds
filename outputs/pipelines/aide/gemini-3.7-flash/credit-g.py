import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb


def main():
    # Define paths
    train_path = "./input/train.parquet"
    test_path = "./input/test.parquet"
    output_dir = "./working"
    os.makedirs(output_dir, exist_ok=True)
    submission_path = os.path.join(output_dir, "submission.csv")

    # Load data
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    target_col = "class"

    # Process target
    le = LabelEncoder()
    y = le.fit_transform(train_df[target_col].astype(str))

    # In German credit risk, 'bad' is commonly treated as the positive risk class if present
    # Check classes mapping
    classes = list(le.classes_)
    if "bad" in classes and "good" in classes:
        # Ensure 'bad' is 1 and 'good' is 0
        bad_idx = classes.index("bad")
        y = (train_df[target_col].astype(str) == "bad").astype(int).values

    # Feature preparation
    feature_cols = [c for c in train_df.columns if c != target_col]
    X = train_df[feature_cols].copy()
    X_test = test_df[feature_cols].copy()

    # Identify categorical columns and convert to category dtype
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    for col in cat_cols:
        X[col] = X[col].astype("category")
        X_test[col] = X_test[col].astype("category")

    # Setup Stratified 5-Fold Cross Validation
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
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "random_state": 42,
        "n_estimators": 1000,
        "verbose": -1,
    }

    # Cross-validation training loop
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, y_train = X.iloc[train_idx], y[train_idx]
        X_val, y_val = X.iloc[val_idx], y[val_idx]

        model = lgb.LGBMClassifier(**lgb_params)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )

        val_preds = model.predict_proba(X_val)[:, 1]
        oof_preds[val_idx] = val_preds
        test_preds += model.predict_proba(X_test)[:, 1] / n_splits

    # Compute and display overall OOF ROC-AUC
    cv_auc = roc_auc_score(y, oof_preds)
    print(f"5-Fold CV ROC-AUC Score: {cv_auc:.5f}")

    # Generate submission file
    submission = pd.DataFrame({"id": np.arange(len(test_df)), "class": test_preds})
    submission.to_csv(submission_path, index=False)
    print(f"Saved submission to {submission_path}")


if __name__ == "__main__":
    main()
