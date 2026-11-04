import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error
import lightgbm as lgb

# 1. Load data
train_path = "./input/train.parquet"
test_path = "./input/test.parquet"

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = "HPSA Score"
y = train_df[target_col].values
X_train = train_df.drop(columns=[target_col]).copy()
X_test = test_df.copy()

# 2. Feature Engineering & Preprocessing
combined = pd.concat([X_train, X_test], axis=0).reset_index(drop=True)

# Missing count per row
combined["missing_count"] = combined.isnull().sum(axis=1)

# Detect and process date columns
date_cols = []
date_series_dict = {}

for col in list(combined.columns):
    if combined[col].dtype == "object":
        sample_non_null = combined[col].dropna()
        if len(sample_non_null) > 0:
            try:
                converted = pd.to_datetime(sample_non_null.iloc[:50], errors="coerce")
                if converted.notnull().mean() > 0.8:
                    full_dates = pd.to_datetime(combined[col], errors="coerce")
                    date_series_dict[col] = full_dates
                    combined[f"{col}_year"] = full_dates.dt.year
                    combined[f"{col}_month"] = full_dates.dt.month
                    combined[f"{col}_day"] = full_dates.dt.day
                    combined[f"{col}_dayofweek"] = full_dates.dt.dayofweek
                    # Continuous elapsed days from epoch
                    combined[f"{col}_elapsed_days"] = (
                        full_dates - pd.Timestamp("1970-01-01")
                    ).dt.total_seconds() / 86400.0
                    date_cols.append(col)
            except Exception:
                pass

# Pairwise date differences
date_col_keys = list(date_series_dict.keys())
for i in range(len(date_col_keys)):
    for j in range(i + 1, len(date_col_keys)):
        col1, col2 = date_col_keys[i], date_col_keys[j]
        diff_days = (
            date_series_dict[col1] - date_series_dict[col2]
        ).dt.total_seconds() / 86400.0
        combined[f"{col1}_minus_{col2}_days"] = diff_days

# Drop raw date string columns
combined.drop(columns=date_cols, inplace=True)

# Identify numerical and categorical columns
cat_cols = []
num_cols = []

for col in combined.columns:
    if combined[col].dtype == "object" or combined[col].dtype.name == "category":
        cat_cols.append(col)
        # Add frequency encoding
        freq = combined[col].value_counts(dropna=False, normalize=True)
        combined[f"{col}_freq"] = combined[col].map(freq).astype(float)
    else:
        if col != "missing_count":
            num_cols.append(col)

# Group-level aggregations on top categorical features
top_cats = [col for col in cat_cols if 2 < combined[col].nunique(dropna=True) <= 500][
    :5
]
top_nums = num_cols[:8]

for cat_col in top_cats:
    for num_col in top_nums:
        grouped = combined.groupby(cat_col)[num_col]
        agg_mean = grouped.transform("mean")
        agg_std = grouped.transform("std")
        combined[f"{cat_col}_{num_col}_grp_mean"] = agg_mean
        combined[f"{cat_col}_{num_col}_grp_std"] = agg_std

# Convert categorical columns to category dtype for LightGBM
for col in cat_cols:
    combined[col] = combined[col].astype("category")

# Split back to train and test
n_train = len(train_df)
X_train_proc = combined.iloc[:n_train].copy().reset_index(drop=True)
X_test_proc = combined.iloc[n_train:].copy().reset_index(drop=True)

# 3. Model Training & 5-Fold Cross-Validation
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_preds = np.zeros(n_train)
test_preds = np.zeros(len(test_df))

lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.04,
    "num_leaves": 63,
    "max_depth": -1,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "min_child_samples": 20,
    "random_state": 42,
    "verbosity": -1,
    "n_jobs": -1,
}

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_proc, y)):
    X_tr, y_tr = X_train_proc.iloc[train_idx], y[train_idx]
    X_va, y_val = X_train_proc.iloc[val_idx], y[val_idx]

    train_data = lgb.Dataset(
        X_tr, label=y_tr, categorical_feature=cat_cols, free_raw_data=False
    )
    val_data = lgb.Dataset(
        X_va,
        label=y_val,
        reference=train_data,
        categorical_feature=cat_cols,
        free_raw_data=False,
    )

    model = lgb.train(
        lgb_params,
        train_data,
        num_boost_round=2500,
        valid_sets=[val_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )

    val_pred = model.predict(X_va, num_iteration=model.best_iteration)
    oof_preds[val_idx] = val_pred
    fold_rmse = root_mean_squared_error(y_val, val_pred)
    print(f"Fold {fold + 1} RMSE: {fold_rmse:.5f}")

    test_preds += (
        model.predict(X_test_proc, num_iteration=model.best_iteration) / n_splits
    )

overall_rmse = root_mean_squared_error(y, oof_preds)
print(f"Overall Out-of-Fold RMSE: {overall_rmse:.5f}")

# 4. Save Submission
os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame({"id": np.arange(len(test_preds)), "HPSA Score": test_preds})
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved successfully to ./working/submission.csv")
