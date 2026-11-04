import itertools
import os
import warnings
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")

# Define paths
TRAIN_PATH = "./input/train.parquet"
TEST_PATH = "./input/test.parquet"
OUTPUT_DIR = "./working"
os.makedirs(OUTPUT_DIR, exist_ok=True)
SUBMISSION_PATH = os.path.join(OUTPUT_DIR, "submission.csv")

# Load data
train_df = pd.read_parquet(TRAIN_PATH)
test_df = pd.read_parquet(TEST_PATH)

target_col = "Original Principal Amount (US$)"

# Separate features and target
y = train_df[target_col].values
X = train_df.drop(columns=[target_col]).copy()
X_test = test_df.copy()

if target_col in X_test.columns:
    X_test = X_test.drop(columns=[target_col])

n_train = len(X)
combined = pd.concat([X, X_test], axis=0).reset_index(drop=True)

# 1. Identify date columns and engineer pairwise duration features
date_cols = []
for col in combined.columns:
    if (
        pd.api.types.is_datetime64_any_dtype(combined[col])
        or "Date" in col
        or "date" in col
    ):
        dt_col = pd.to_datetime(combined[col], errors="coerce")
        if dt_col.notna().sum() > 0:
            date_cols.append(col)
            combined[col] = dt_col

# Compute pairwise differences (in days) between all date pairs
reference_epoch = pd.Timestamp("1950-01-01")
for i, col_a in enumerate(date_cols):
    # Continuous numeric timestamp relative to reference epoch
    combined[f"{col_a}_days_from_epoch"] = (
        (combined[col_a] - reference_epoch).dt.total_seconds() / 86400.0
    ).astype("float32")

    # Calendar parts
    combined[f"{col_a}_year"] = combined[col_a].dt.year
    combined[f"{col_a}_month"] = combined[col_a].dt.month
    combined[f"{col_a}_day"] = combined[col_a].dt.day
    combined[f"{col_a}_dayofweek"] = combined[col_a].dt.dayofweek

    for col_b in date_cols[i + 1 :]:
        diff_name = f"diff_days__{col_a}__minus__{col_b}"
        combined[diff_name] = (
            (combined[col_a] - combined[col_b]).dt.total_seconds() / 86400.0
        ).astype("float32")

# Drop raw datetime columns
combined.drop(columns=date_cols, inplace=True)

# 2. Process categorical columns with frequency encoding and category typing
for col in combined.columns:
    if combined[col].dtype == "object" or combined[col].dtype.name == "category":
        freq = combined[col].value_counts(dropna=False)
        combined[f"{col}_freq"] = combined[col].map(freq)
        combined[col] = combined[col].astype("category")

# Split back into train and test
X_processed = combined.iloc[:n_train].copy()
X_test_processed = combined.iloc[n_train:].copy()

# Cross Validation Setup
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_preds = np.zeros(n_train)
test_preds = np.zeros(len(X_test_processed))

categorical_features = [
    col for col in X_processed.columns if X_processed[col].dtype.name == "category"
]

lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "n_estimators": 2500,
    "learning_rate": 0.03,
    "num_leaves": 63,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}

for fold, (train_idx, val_idx) in enumerate(kf.split(X_processed, y)):
    X_tr, y_tr = X_processed.iloc[train_idx], y[train_idx]
    X_va, y_va = X_processed.iloc[val_idx], y[val_idx]

    model = lgb.LGBMRegressor(**lgb_params)

    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0),
        ],
        categorical_feature=categorical_features,
    )

    val_pred = model.predict(X_va)
    oof_preds[val_idx] = val_pred
    test_preds += model.predict(X_test_processed) / n_splits

    fold_rmse = root_mean_squared_error(y_va, val_pred)
    print(f"Fold {fold + 1} RMSE: {fold_rmse:.4f}")

# Overall CV Evaluation
cv_rmse = root_mean_squared_error(y, oof_preds)
print(f"\nOverall Out-of-Fold RMSE: {cv_rmse:.4f}")

# Post-processing: principal amount is non-negative
test_preds = np.clip(test_preds, 0, None)

# Save submission
submission = pd.DataFrame({"id": np.arange(len(test_preds)), target_col: test_preds})
submission.to_csv(SUBMISSION_PATH, index=False)
print(
    f"Submission saved successfully to {SUBMISSION_PATH} with shape {submission.shape}."
)
