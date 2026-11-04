import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error

# 1. Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "gisacres"
y = train_df[target_col].values
X = train_df.drop(columns=[target_col]).copy()
X_test = test_df.copy()

# 2. Feature Engineering & Preprocessing
combined = pd.concat([X, X_test], axis=0, ignore_index=True)

# Process datetime columns
date_cols = []
for col in combined.columns:
    if np.issubdtype(combined[col].dtype, np.datetime64):
        date_cols.append(col)
    elif combined[col].dtype == "object":
        # Check if it can be parsed as datetime
        sample = combined[col].dropna()
        if len(sample) > 0:
            try:
                pd.to_datetime(sample.iloc[:100])
                date_cols.append(col)
            except Exception:
                pass

for col in date_cols:
    dt_series = pd.to_datetime(combined[col], errors="coerce")
    combined[f"{col}_year"] = dt_series.dt.year
    combined[f"{col}_month"] = dt_series.dt.month
    combined[f"{col}_day"] = dt_series.dt.day
    combined[f"{col}_dayofweek"] = dt_series.dt.dayofweek
    combined[f"{col}_dayofyear"] = dt_series.dt.dayofyear
    combined = combined.drop(columns=[col])

# Convert remaining object/string columns to category
cat_cols = []
for col in combined.columns:
    if combined[col].dtype == "object" or combined[col].dtype.name == "category":
        combined[col] = combined[col].astype("category")
        cat_cols.append(col)

# Split back to train and test
n_train = len(X)
X_processed = combined.iloc[:n_train].reset_index(drop=True)
X_test_processed = combined.iloc[n_train:].reset_index(drop=True)

# 3. 5-Fold Cross Validation Setup
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_preds = np.zeros(n_train)
test_preds = np.zeros(len(X_test_processed))

params = {
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

# 4. Train and Validate
for fold, (train_idx, val_idx) in enumerate(kf.split(X_processed, y)):
    X_tr, y_tr = X_processed.iloc[train_idx], y[train_idx]
    X_val, y_val = X_processed.iloc[val_idx], y[val_idx]

    trn_data = lgb.Dataset(X_tr, label=y_tr)
    val_data = lgb.Dataset(X_val, label=y_val, reference=trn_data)

    model = lgb.train(
        params,
        trn_data,
        num_boost_round=2000,
        valid_sets=[trn_data, val_data],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=0)],
    )

    val_pred = model.predict(X_val, num_iteration=model.best_iteration)
    oof_preds[val_idx] = np.clip(val_pred, 0, None)

    test_fold_pred = model.predict(X_test_processed, num_iteration=model.best_iteration)
    test_preds += np.clip(test_fold_pred, 0, None) / n_splits

# 5. Evaluate Out-of-Fold RMSE
cv_rmse = root_mean_squared_error(y, oof_preds)
print(f"5-Fold CV Out-Of-Fold RMSE: {cv_rmse:.4f}")

# 6. Save Submission
os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame({"id": np.arange(len(test_preds)), "gisacres": test_preds})
submission.to_csv("./working/submission.csv", index=False)
print(
    f"Submission saved successfully to ./working/submission.csv with {len(submission)} rows."
)
