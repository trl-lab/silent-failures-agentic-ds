import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss

# 1. Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# 2. Extract feature columns and prepare base matrices
feature_cols = [c for c in train_df.columns if c.startswith("feat_")]
X_raw = train_df[feature_cols].values.astype(np.float32)
X_test_raw = test_df[feature_cols].values.astype(np.float32)

# Encode target classes
le = LabelEncoder()
y = le.fit_transform(train_df["target"].values)
num_classes = len(le.classes_)


# 3. Compute row-wise statistical features and normalized proportions
def extract_row_features(mat):
    row_sum = np.sum(mat, axis=1, keepdims=True)
    row_std = np.std(mat, axis=1, keepdims=True)
    row_mean = np.mean(mat, axis=1, keepdims=True)
    row_max = np.max(mat, axis=1, keepdims=True)
    row_nonzeros = np.sum(mat > 0, axis=1, keepdims=True)
    row_zeros = np.sum(mat == 0, axis=1, keepdims=True)
    row_p75 = np.percentile(mat, 75, axis=1, keepdims=True)
    row_p90 = np.percentile(mat, 90, axis=1, keepdims=True)

    # Row-normalized proportions (relative frequencies)
    norm_mat = mat / (row_sum + 1e-6)

    # Log1p of raw counts
    log_mat = np.log1p(mat)

    stats = np.hstack(
        [
            row_sum,
            row_std,
            row_mean,
            row_max,
            row_nonzeros,
            row_zeros,
            row_p75,
            row_p90,
        ]
    )

    return mat, log_mat, norm_mat, stats


train_mat, train_log, train_norm, train_stats = extract_row_features(X_raw)
test_mat, test_log, test_norm, test_stats = extract_row_features(X_test_raw)

# Base stacked features before fold-wise SVD
X_base = np.hstack([train_mat, train_log, train_norm, train_stats])
X_test_base = np.hstack([test_mat, test_log, test_norm, test_stats])

# 4. Model training with 5-Fold Stratified K-Fold and fold-wise SVD
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros((len(train_df), num_classes), dtype=np.float64)
test_preds = np.zeros((len(test_df), num_classes), dtype=np.float64)

lgb_params = {
    "objective": "multiclass",
    "num_class": num_classes,
    "metric": "multi_logloss",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "max_depth": -1,
    "feature_fraction": 0.75,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "min_child_samples": 20,
    "lambda_l1": 0.1,
    "lambda_l2": 1.0,
    "verbosity": -1,
    "n_jobs": -1,
    "random_state": 42,
}

n_svd_components = 16

for fold, (train_idx, val_idx) in enumerate(skf.split(X_base, y)):
    # Fit TruncatedSVD strictly on training fold log counts
    svd = TruncatedSVD(n_components=n_svd_components, random_state=42 + fold)
    train_svd = svd.fit_transform(train_log[train_idx])
    val_svd = svd.transform(train_log[val_idx])
    test_fold_svd = svd.transform(test_log)

    X_train_fold = np.hstack([X_base[train_idx], train_svd])
    X_val_fold = np.hstack([X_base[val_idx], val_svd])
    X_test_fold = np.hstack([X_test_base, test_fold_svd])

    train_data = lgb.Dataset(X_train_fold, label=y[train_idx])
    val_data = lgb.Dataset(X_val_fold, label=y[val_idx], reference=train_data)

    model = lgb.train(
        lgb_params,
        train_data,
        num_boost_round=1200,
        valid_sets=[train_data, val_data],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )

    oof_preds[val_idx] = model.predict(X_val_fold, num_iteration=model.best_iteration)
    test_preds += (
        model.predict(X_test_fold, num_iteration=model.best_iteration) / skf.n_splits
    )

# 5. Evaluation
cv_loss = log_loss(y, oof_preds)
print(f"5-Fold CV Log Loss: {cv_loss:.5f}")

# 6. Save submission
os.makedirs("./working", exist_ok=True)
predicted_classes = le.inverse_transform(np.argmax(test_preds, axis=1))

submission = pd.DataFrame({"id": np.arange(len(test_df)), "target": predicted_classes})
submission.to_csv("./working/submission.csv", index=False)
print("Submission successfully saved to ./working/submission.csv")
