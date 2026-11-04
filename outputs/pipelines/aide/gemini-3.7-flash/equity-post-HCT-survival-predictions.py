import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb


def compute_c_index(event_times, predicted_scores, event_observed):
    """
    Computes Harrell's Concordance Index (C-index).
    predicted_scores: predicted risk score (higher = higher risk / shorter survival).
    """
    try:
        from lifelines.utils import concordance_index

        # lifelines expects higher score = longer survival, so we pass -predicted_scores
        return concordance_index(event_times, -predicted_scores, event_observed)
    except ImportError:
        # Vectorized fallback calculation
        n = len(event_times)
        order = np.argsort(event_times)
        event_times = event_times[order]
        predicted_scores = predicted_scores[order]
        event_observed = event_observed[order]

        concordant = 0
        tied = 0
        total_pairs = 0

        for i in range(n):
            if event_observed[i] == 1:
                # Compare with all individuals who survived longer
                j_indices = np.where(event_times > event_times[i])[0]
                if len(j_indices) > 0:
                    diff = predicted_scores[i] - predicted_scores[j_indices]
                    concordant += np.sum(diff > 0)
                    tied += np.sum(diff == 0)
                    total_pairs += len(j_indices)

        return (concordant + 0.5 * tied) / max(total_pairs, 1)


def main():
    train_path = "./input/train.parquet"
    test_path = "./input/test.parquet"

    # Fallback to csv if parquet not present
    if not os.path.exists(train_path):
        train_path = "./input/original/train.csv"

    print("Loading datasets...")
    train = (
        pd.read_parquet(train_path)
        if train_path.endswith(".parquet")
        else pd.read_csv(train_path)
    )
    test = (
        pd.read_parquet(test_path)
        if os.path.exists(test_path)
        else pd.read_csv("./input/test.csv")
    )

    target_col = "efs"
    time_col = "efs_time"
    id_col = "ID" if "ID" in train.columns else "id"

    drop_cols = [id_col, target_col, time_col]
    feature_cols = [col for col in train.columns if col not in drop_cols]

    # Preprocess categorical features for LightGBM
    for col in feature_cols:
        if train[col].dtype == "object" or train[col].dtype.name == "category":
            train[col] = train[col].astype("category")
            test[col] = test[col].astype("category")

    # Cox objective target: positive for observed event, negative for censored
    train_efs = train[target_col].values
    train_time = train[time_col].values
    # Avoid exact 0 for censored negative times in LightGBM
    y_cox = np.where(
        train_efs == 1, np.maximum(train_time, 1e-5), -np.maximum(train_time, 1e-5)
    )

    params = {
        "objective": "cox",
        "metric": "None",
        "boosting_type": "gbdt",
        "learning_rate": 0.03,
        "num_leaves": 31,
        "min_child_samples": 50,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 1.0,
        "reg_lambda": 5.0,
        "random_state": 42,
        "verbose": -1,
        "n_jobs": -1,
    }

    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    oof_preds = np.zeros(len(train))
    test_preds = np.zeros(len(test))

    print(f"Training LightGBM with 5-fold CV across {len(feature_cols)} features...")
    for fold, (train_idx, val_idx) in enumerate(skf.split(train, train_efs)):
        X_train, y_train = train.iloc[train_idx][feature_cols], y_cox[train_idx]
        X_val, y_val = train.iloc[val_idx][feature_cols], y_cox[val_idx]

        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        model = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=[val_data],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )

        val_pred = model.predict(X_val)
        oof_preds[val_idx] = val_pred
        test_preds += model.predict(test[feature_cols]) / n_splits

        fold_c_index = compute_c_index(
            train_time[val_idx], val_pred, train_efs[val_idx]
        )
        print(f"Fold {fold + 1} C-index: {fold_c_index:.5f}")

    overall_c_index = compute_c_index(train_time, oof_preds, train_efs)
    print(f"\nOverall Out-of-Fold Concordance Index: {overall_c_index:.5f}")

    # Prepare submission
    os.makedirs("./working", exist_ok=True)
    submission = pd.DataFrame(
        {
            "id": np.arange(len(test)),
            "efs": test_preds,
        }
    )
    submission_path = "./working/submission.csv"
    submission.to_csv(submission_path, index=False)
    print(f"Saved predictions to {submission_path} with shape {submission.shape}")


if __name__ == "__main__":
    main()
