import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

# Ensure output directory exists
os.makedirs("./working", exist_ok=True)

# 1. Load data
print("Loading data...")
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "time_to_approve"
y = train_df[target_col].values
X_train = train_df.drop(columns=[target_col]).copy()
X_test = test_df.copy()

# 2. Feature Engineering & Preprocessing
combined = pd.concat([X_train, X_test], axis=0, ignore_index=True)

# Process date/time columns
date_cols = [
    col
    for col in combined.columns
    if "date" in col.lower() or pd.api.types.is_datetime64_any_dtype(combined[col])
]

timestamp_cols = []
for col in date_cols:
    dt = pd.to_datetime(combined[col], errors="coerce")
    combined[f"{col}_year"] = dt.dt.year
    combined[f"{col}_month"] = dt.dt.month
    combined[f"{col}_day"] = dt.dt.day
    combined[f"{col}_dayofweek"] = dt.dt.dayofweek
    combined[f"{col}_dayofyear"] = dt.dt.dayofyear

    ts_col = f"{col}_timestamp"
    combined[ts_col] = dt.astype("int64") // 10**9
    combined.loc[dt.isna(), ts_col] = np.nan
    timestamp_cols.append(ts_col)
    combined.drop(columns=[col], inplace=True)

# Pairwise date intervals (in days)
for i in range(len(timestamp_cols)):
    for j in range(i + 1, len(timestamp_cols)):
        col1, col2 = timestamp_cols[i], timestamp_cols[j]
        diff_name = f"diff_{col1}_minus_{col2}"
        combined[diff_name] = (combined[col1] - combined[col2]) / 86400.0


# Case-insensitive column search helper
def find_col(substring):
    for c in combined.columns:
        if (
            substring.lower() in c.lower()
            and not c.endswith("_year")
            and not c.endswith("_timestamp")
        ):
            return c
    return None


est_cost_col = find_col("estimated_cost") or find_col("estimated cost")
rev_cost_col = find_col("revised_cost") or find_col("revised cost")
prop_units_col = find_col("proposed_units") or find_col("proposed units")
exist_units_col = find_col("existing_units") or find_col("existing units")
prop_stories_col = find_col("proposed_stories") or find_col("proposed stories")
exist_stories_col = find_col("existing_stories") or find_col("existing stories")
prop_use_col = find_col("proposed_use") or find_col("proposed use")
exist_use_col = find_col("existing_use") or find_col("existing use")

# Cost interactions
if est_cost_col and rev_cost_col:
    est_c = pd.to_numeric(combined[est_cost_col], errors="coerce").fillna(0)
    rev_c = pd.to_numeric(combined[rev_cost_col], errors="coerce").fillna(0)
    combined["cost_diff"] = rev_c - est_c
    combined["cost_ratio"] = (rev_c + 1.0) / (est_c + 1.0)
    combined["log_est_cost"] = np.log1p(np.maximum(est_c, 0))
    combined["log_rev_cost"] = np.log1p(np.maximum(rev_c, 0))

# Unit interactions
if prop_units_col and exist_units_col:
    p_u = pd.to_numeric(combined[prop_units_col], errors="coerce").fillna(0)
    e_u = pd.to_numeric(combined[exist_units_col], errors="coerce").fillna(0)
    combined["units_diff"] = p_u - e_u

# Story interactions
if prop_stories_col and exist_stories_col:
    p_s = pd.to_numeric(combined[prop_stories_col], errors="coerce").fillna(0)
    e_s = pd.to_numeric(combined[exist_stories_col], errors="coerce").fillna(0)
    combined["stories_diff"] = p_s - e_s

# Use change flag
if prop_use_col and exist_use_col:
    combined["use_changed"] = (
        combined[prop_use_col].astype(str) != combined[exist_use_col].astype(str)
    ).astype(int)

# Group aggregations for cost
permit_type_col = find_col("permit_type") or find_col("permit type")
district_col = find_col("district") or find_col("zipcode") or find_col("zip")

if est_cost_col and permit_type_col:
    est_c = pd.to_numeric(combined[est_cost_col], errors="coerce")
    type_mean = combined.groupby(permit_type_col)[est_cost_col].transform("mean")
    type_std = combined.groupby(permit_type_col)[est_cost_col].transform("std")
    combined["cost_to_type_mean"] = est_c / (type_mean + 1.0)
    combined["cost_type_zscore"] = (est_c - type_mean) / (type_std + 1e-5)

if est_cost_col and district_col:
    est_c = pd.to_numeric(combined[est_cost_col], errors="coerce")
    dist_mean = combined.groupby(district_col)[est_cost_col].transform("mean")
    combined["cost_to_dist_mean"] = est_c / (dist_mean + 1.0)

# Categorical and high-cardinality processing
cat_cols = combined.select_dtypes(include=["object", "category"]).columns.tolist()
for col in cat_cols:
    freq = combined[col].value_counts(dropna=False)
    combined[f"{col}_freq"] = combined[col].map(freq)
    combined[col] = combined[col].astype("category")

# Separate back into train and test sets
n_train = len(train_df)
X_train = combined.iloc[:n_train].reset_index(drop=True)
X_test = combined.iloc[n_train:].reset_index(drop=True)

# 3. Model Training with 5-Fold Cross-Validation
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_preds = np.zeros(n_train)
test_preds = np.zeros(len(test_df))

lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.03,
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

print("Starting 5-fold cross-validation...")
for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y)):
    X_tr, y_tr = X_train.iloc[train_idx], y[train_idx]
    X_va, y_va = X_train.iloc[val_idx], y[val_idx]

    trn_data = lgb.Dataset(X_tr, label=y_tr)
    val_data = lgb.Dataset(X_va, label=y_va, reference=trn_data)

    model = lgb.train(
        lgb_params,
        trn_data,
        num_boost_round=3000,
        valid_sets=[trn_data, val_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )

    val_pred = model.predict(X_va, num_iteration=model.best_iteration)
    oof_preds[val_idx] = val_pred

    fold_rmse = np.sqrt(mean_squared_error(y_va, val_pred))
    print(f"Fold {fold + 1} RMSE: {fold_rmse:.4f}")

    test_preds += model.predict(X_test, num_iteration=model.best_iteration) / n_splits

# 4. Evaluation and Post-Processing
if np.min(y) >= 0:
    oof_preds = np.clip(oof_preds, 0, None)
    test_preds = np.clip(test_preds, 0, None)

cv_rmse = np.sqrt(mean_squared_error(y, oof_preds))
print(f"\nOverall Out-of-Fold Validation RMSE: {cv_rmse:.4f}")

# 5. Save Submission
sub = pd.DataFrame({"id": np.arange(len(test_preds)), "time_to_approve": test_preds})
sub.to_csv("./working/submission.csv", index=False)
print("Submission saved successfully to ./working/submission.csv")
