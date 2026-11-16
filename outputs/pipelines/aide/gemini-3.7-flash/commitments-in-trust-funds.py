import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error

# 1. Load Data
train_path = "./input/train.parquet"
test_path = "./input/test.parquet"
output_dir = "./working"
os.makedirs(output_dir, exist_ok=True)

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = "New Commitments (US$)"
y = train_df[target_col].values
train_df = train_df.drop(columns=[target_col])

# Combine for uniform preprocessing
n_train = len(train_df)
combined_df = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)

# 2. Feature Engineering & Preprocessing
# Identify and parse date columns
date_cols = []
for col in combined_df.columns:
    if combined_df[col].dtype == "object":
        # Check if column is parseable as datetime
        sample = combined_df[col].dropna().head(50)
        if len(sample) > 0:
            try:
                pd.to_datetime(sample, format="%Y-%m-%d", errors="raise")
                date_cols.append(col)
            except Exception:
                try:
                    pd.to_datetime(sample, errors="raise")
                    date_cols.append(col)
                except Exception:
                    pass

for col in date_cols:
    dt = pd.to_datetime(combined_df[col], errors="coerce")
    combined_df[f"{col}_year"] = dt.dt.year
    combined_df[f"{col}_month"] = dt.dt.month
    combined_df[f"{col}_quarter"] = dt.dt.quarter
    combined_df[f"{col}_day"] = dt.dt.day
    combined_df[f"{col}_dayofweek"] = dt.dt.dayofweek
    combined_df[f"{col}_timestamp"] = dt.astype("int64") // 10**9
    combined_df.drop(columns=[col], inplace=True)

# Handle categorical and object columns
cat_cols = combined_df.select_dtypes(include=["object", "category"]).columns.tolist()

for col in cat_cols:
    combined_df[col] = combined_df[col].astype(str).fillna("missing")
    freq = combined_df[col].value_counts()
    combined_df[f"{col}_freq"] = combined_df[col].map(freq)
    combined_df[col] = combined_df[col].astype("category")

# Separate back to train and test sets
X = combined_df.iloc[:n_train].reset_index(drop=True)
X_test = combined_df.iloc[n_train:].reset_index(drop=True)

# 3. 5-Fold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(n_train)
test_preds = np.zeros(len(X_test))

lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "max_depth": -1,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_train_fold, y_train_fold = X.iloc[train_idx], y[train_idx]
    X_val_fold, y_val_fold = X.iloc[val_idx], y[val_idx]

    train_data = lgb.Dataset(X_train_fold, label=y_train_fold)
    val_data = lgb.Dataset(X_val_fold, label=y_val_fold, reference=train_data)

    model = lgb.train(
        lgb_params,
        train_data,
        num_boost_round=2000,
        valid_sets=[train_data, val_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(0),
        ],
    )

    val_pred = model.predict(X_val_fold, num_iteration=model.best_iteration)
    oof_preds[val_idx] = val_pred
    test_preds += (
        model.predict(X_test, num_iteration=model.best_iteration) / kf.n_splits
    )

    fold_rmse = root_mean_squared_error(y_val_fold, val_pred)
    print(f"Fold {fold + 1} RMSE: {fold_rmse:.4f}")

overall_rmse = root_mean_squared_error(y, oof_preds)
print(f"\nOverall Out-of-Fold RMSE: {overall_rmse:.4f}")

# 4. Save Submission
submission = pd.DataFrame({"id": np.arange(len(test_preds)), target_col: test_preds})
submission_path = os.path.join(output_dir, "submission.csv")
submission.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path} with shape {submission.shape}")
