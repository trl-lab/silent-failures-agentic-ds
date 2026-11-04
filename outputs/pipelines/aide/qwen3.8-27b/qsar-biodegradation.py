import os
import warnings

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)

os.makedirs("./working", exist_ok=True)

train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

TARGET = "Class"

# Drop rows with missing target, if any.
train = train.dropna(subset=[TARGET])

# Encode target to 0/1. The positive class is ready biodegradable (RB).
raw_y = train[TARGET]
if raw_y.dtype == object:
    uniq = sorted(raw_y.unique(), key=str)
    # For this dataset, sorted order places NRB/not-ready before RB/ready.
    # If labels are already 0/1 as strings, this also maps 0 -> 0 and 1 -> 1.
    y = raw_y.map({uniq[0]: 0, uniq[1]: 1}).astype(int)
else:
    y = raw_y.astype(int)
    uniq = np.unique(y)
    if len(uniq) == 2 and set(uniq) != {0, 1}:
        y = y.map({uniq[0]: 0, uniq[1]: 1}).astype(int)

# Feature selection: use only numeric molecular descriptors and exclude identifiers.
exclude_cols = {TARGET, "id", "ID", "Id", "index", "row_id", "sample_id"}
candidate_cols = [c for c in train.columns if c not in exclude_cols]

X = train[candidate_cols].select_dtypes(include=[np.number])
if X.shape[1] == 0:
    X = pd.to_numeric(train[candidate_cols], errors="coerce")

# Remove unusable features: all-NaN and constant features.
X = X.dropna(axis=1, how="all")
constant_mask = X.nunique(dropna=True) <= 1
X = X.loc[:, ~constant_mask]

feature_cols = X.columns.tolist()

# Align test features to the training feature set.
X_test = test.reindex(columns=feature_cols)
X_test = X_test.apply(pd.to_numeric, errors="coerce")

# Simple, leakage-aware model: median imputation + LightGBM.
pipe = Pipeline(
    [
        (
            "imputer",
            SimpleImputer(strategy="median", keep_empty_features=True),
        ),
        (
            "model",
            LGBMClassifier(
                objective="binary",
                n_estimators=500,
                learning_rate=0.05,
                num_leaves=31,
                max_depth=-1,
                min_child_samples=20,
                subsample=0.9,
                subsample_freq=1,
                colsample_bytree=0.9,
                reg_alpha=0.1,
                reg_lambda=0.1,
                random_state=SEED,
                n_jobs=-1,
                verbose=-1,
            ),
        ),
    ]
)

# Stratified 5-fold cross-validation is appropriate for this tabular binary task.
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
folds = list(cv.split(X, y))
auc_scores = []

for fold_idx, (train_idx, val_idx) in enumerate(folds, start=1):
    X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
    y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

    pipe.fit(X_train_fold, y_train_fold)
    val_pred = pipe.predict_proba(X_val_fold)[:, 1]
    fold_auc = roc_auc_score(y_val_fold, val_pred)
    auc_scores.append(fold_auc)
    print(f"Fold {fold_idx} ROC AUC: {fold_auc:.5f}")

cv_auc = float(np.mean(auc_scores))
print(f"5-fold CV ROC AUC: {cv_auc:.5f}")

# Refit on the full training set and predict the test set.
pipe.fit(X, y)
test_pred = pipe.predict_proba(X_test)[:, 1]
test_pred = np.clip(test_pred, 1e-6, 1.0 - 1e-6)

submission = pd.DataFrame(
    {
        "id": np.arange(len(X_test), dtype=int),
        "Class": test_pred,
    }
)
submission.to_csv("./working/submission.csv", index=False)
print("Saved ./working/submission.csv")
