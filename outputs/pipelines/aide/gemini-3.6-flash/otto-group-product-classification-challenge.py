import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD
import xgboost as xgb
import warnings

warnings.filterwarnings("ignore")

# Load dataset
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Identify feature columns
feature_cols = [c for c in train_df.columns if c.startswith("feat_")]

# Extract raw feature arrays
X_train_raw = train_df[feature_cols].values.astype(np.float32)
X_test_raw = test_df[feature_cols].values.astype(np.float32)

# Label encoding for target
le = LabelEncoder()
y_train = le.fit_transform(train_df["target"].values)


# 1. Feature Engineering: Row-wise summary statistics
def extract_row_stats(X):
    row_sum = np.sum(X, axis=1, keepdims=True)
    row_mean = np.mean(X, axis=1, keepdims=True)
    row_std = np.std(X, axis=1, keepdims=True)
    row_max = np.max(X, axis=1, keepdims=True)
    row_min = np.min(X, axis=1, keepdims=True)
    row_nonzero = np.sum(X > 0, axis=1, keepdims=True)
    return np.hstack([row_sum, row_mean, row_std, row_max, row_min, row_nonzero])


train_stats = extract_row_stats(X_train_raw)
test_stats = extract_row_stats(X_test_raw)

# 2. Feature Engineering: Log1p counts
X_train_log = np.log1p(X_train_raw)
X_test_log = np.log1p(X_test_raw)

# 3. Feature Engineering: TF-IDF + TruncatedSVD
tfidf = TfidfTransformer()
X_train_tfidf = tfidf.fit_transform(X_train_raw)
X_test_tfidf = tfidf.transform(X_test_raw)

svd = TruncatedSVD(n_components=10, random_state=42)
X_train_svd = svd.fit_transform(X_train_tfidf)
X_test_svd = svd.transform(X_test_tfidf)

# Combine all features
X_train_full = np.hstack([X_train_raw, X_train_log, train_stats, X_train_svd])
X_test_full = np.hstack([X_test_raw, X_test_log, test_stats, X_test_svd])

# 5-Fold Stratified K-Fold Cross-Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros((len(train_df), len(le.classes_)))
test_preds = np.zeros((len(test_df), len(le.classes_)))

xgb_params = {
    "objective": "multi:softprob",
    "num_class": len(le.classes_),
    "eval_metric": "mlogloss",
    "n_estimators": 1000,
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "tree_method": "hist",
    "random_state": 42,
    "n_jobs": -1,
    "early_stopping_rounds": 50,
}

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_full, y_train)):
    X_tr, y_tr = X_train_full[train_idx], y_train[train_idx]
    X_va, y_va = X_train_full[val_idx], y_train[val_idx]

    model = xgb.XGBClassifier(**xgb_params)
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

    val_probs = model.predict_proba(X_va)
    oof_preds[val_idx] = val_probs
    test_preds += model.predict_proba(X_test_full) / skf.n_splits

# Compute Out-of-Fold Multi-Class Log Loss
cv_log_loss = log_loss(y_train, oof_preds)
print(f"5-Fold Cross-Validation Multi-Class Log Loss: {cv_log_loss:.5f}")

# Save test predictions to submission.csv
test_pred_labels = le.inverse_transform(np.argmax(test_preds, axis=1))
submission = pd.DataFrame({"id": np.arange(len(test_df)), "target": test_pred_labels})
submission.to_csv("./working/submission.csv", index=False)
