import os
import gc
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error
import lightgbm as lgb
import warnings

warnings.filterwarnings("ignore")

# Define paths
TRAIN_PATH = "./input/train.parquet"
TEST_PATH = "./input/test.parquet"
OUTPUT_DIR = "./working"
SUBMISSION_PATH = os.path.join(OUTPUT_DIR, "submission.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load datasets
print("Loading data...")
train_df = pd.read_parquet(TRAIN_PATH)
test_df = pd.read_parquet(TEST_PATH)

target_col = "Supplier Contract Amount (USD)"

# Separate features and target
y = train_df[target_col].values
X = train_df.drop(columns=[target_col]).copy()
X_test = test_df.copy()

# Identify columns to drop (like unique IDs if present)
drop_cols = [
    col for col in X.columns if "id" in col.lower() and X[col].nunique() == len(X)
]
if drop_cols:
    X.drop(columns=drop_cols, inplace=True)
    X_test.drop(columns=drop_cols, inplace=True)

# Combine for uniform preprocessing
n_train = len(X)
combined = pd.concat([X, X_test], axis=0).reset_index(drop=True)

# Process Datetime columns
for col in combined.columns:
    if np.issubdtype(combined[col].dtype, np.datetime64) or "date" in col.lower():
        try:
            dt_series = pd.to_datetime(combined[col], errors="coerce")
            if dt_series.notnull().sum() > 0.1 * len(combined):
                combined[f"{col}_year"] = dt_series.dt.year.astype("float32")
                combined[f"{col}_month"] = dt_series.dt.month.astype("float32")
                combined[f"{col}_day"] = dt_series.dt.day.astype("float32")
                combined[f"{col}_dayofweek"] = dt_series.dt.dayofweek.astype("float32")
                combined[f"{col}_quarter"] = dt_series.dt.quarter.astype("float32")
                # Drop raw datetime column if it's not useful as category
                combined.drop(columns=[col], inplace=True)
        except Exception:
            pass

# Process Categorical and Text-like columns
categorical_cols = []
for col in combined.columns:
    if combined[col].dtype == "object" or combined[col].dtype.name == "category":
        # Frequency encoding
        freq = combined[col].value_counts(dropna=False)
        combined[f"{col}_freq"] = combined[col].map(freq).astype("float32")
        # Native categorical conversion
        combined[col] = combined[col].astype("category")
        categorical_cols.append(col)
    elif np.issubdtype(combined[col].dtype, np.floating) or np.issubdtype(
        combined[col].dtype, np.integer
    ):
        combined[col] = combined[col].astype("float32")

# Split back to train and test
X_processed = combined.iloc[:n_train].copy().reset_index(drop=True)
X_test_processed = combined.iloc[n_train:].copy().reset_index(drop=True)

del combined
gc.collect()

# LightGBM Parameters
lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "max_depth": -1,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "min_child_samples": 20,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}

# 5-Fold Cross-Validation
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_preds = np.zeros(n_train, dtype=np.float64)
test_preds = np.zeros(len(X_test_processed), dtype=np.float64)

print(f"Starting {n_splits}-fold cross-validation...")

for fold, (train_idx, val_idx) in enumerate(kf.split(X_processed, y)):
    X_tr, y_tr = X_processed.iloc[train_idx], y[train_idx]
    X_val, y_val = X_processed.iloc[val_idx], y[val_idx]

    train_data = lgb.Dataset(X_tr, label=y_tr, categorical_feature=categorical_cols)
    val_data = lgb.Dataset(
        X_val, label=y_val, reference=train_data, categorical_feature=categorical_cols
    )

    model = lgb.train(
        lgb_params,
        train_data,
        num_boost_round=2000,
        valid_sets=[train_data, val_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )

    val_pred = model.predict(X_val, num_iteration=model.best_iteration)
    # Clip negative predictions to 0 for contract amounts
    val_pred = np.clip(val_pred, 0, None)
    oof_preds[val_idx] = val_pred

    fold_rmse = root_mean_squared_error(y_val, val_pred)
    print(f"Fold {fold + 1} RMSE: {fold_rmse:,.4f}")

    fold_test_pred = model.predict(X_test_processed, num_iteration=model.best_iteration)
    fold_test_pred = np.clip(fold_test_pred, 0, None)
    test_preds += fold_test_pred / n_splits

cv_rmse = root_mean_squared_error(y, oof_preds)
print(f"\nOverall Out-of-Fold RMSE: {cv_rmse:,.4f}")

# Save test predictions
submission = pd.DataFrame({"id": np.arange(len(test_preds)), target_col: test_preds})

submission.to_csv(SUBMISSION_PATH, index=False)
print(
    f"Submission saved successfully to {SUBMISSION_PATH} with shape {submission.shape}."
)
