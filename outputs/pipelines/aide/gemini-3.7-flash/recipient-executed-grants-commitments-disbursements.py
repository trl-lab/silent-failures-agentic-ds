import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Paths
TRAIN_PATH = "./input/train.parquet"
TEST_PATH = "./input/test.parquet"
OUTPUT_DIR = "./working"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Target column
TARGET_COL = "Disbursements USD (Trust funds)"

# Load data
train_df = pd.read_parquet(TRAIN_PATH)
test_df = pd.read_parquet(TEST_PATH)

n_train = len(train_df)
y = train_df[TARGET_COL].values

# Combine for unified feature engineering
df = pd.concat(
    [train_df.drop(columns=[TARGET_COL], errors="ignore"), test_df], ignore_index=True
)

# 1. Identify and process date columns
date_cols = []
for col in df.columns:
    if df[col].dtype == "object":
        sample = df[col].dropna().head(20)
        if len(sample) > 0:
            try:
                pd.to_datetime(sample)
                date_cols.append(col)
            except (ValueError, TypeError):
                pass

parsed_dates = {}
for col in date_cols:
    try:
        dt_series = pd.to_datetime(df[col], errors="coerce")
        parsed_dates[col] = dt_series
        df[f"{col}_year"] = dt_series.dt.year
        df[f"{col}_month"] = dt_series.dt.month
        df[f"{col}_quarter"] = dt_series.dt.quarter
        df[f"{col}_day"] = dt_series.dt.day
        df[f"{col}_dayofweek"] = dt_series.dt.dayofweek
        df[f"{col}_days_since_epoch"] = (
            dt_series - pd.Timestamp("1970-01-01")
        ).dt.total_seconds() / (24 * 3600)
    except Exception:
        pass

# Compute pairwise date difference intervals (lifecycle durations)
date_names = list(parsed_dates.keys())
for i in range(len(date_names)):
    for j in range(i + 1, len(date_names)):
        col_a, col_b = date_names[i], date_names[j]
        diff_days = (parsed_dates[col_b] - parsed_dates[col_a]).dt.total_seconds() / (
            24 * 3600
        )
        df[f"duration_{col_a}_to_{col_b}"] = diff_days

# Drop raw date object columns
df.drop(columns=date_cols, inplace=True, errors="ignore")

# 2. Identify numerical columns and compute interactions & ratios
num_cols = [
    c
    for c in df.columns
    if np.issubdtype(df[c].dtype, np.number)
    and not c.endswith(("_year", "_month", "_quarter", "_day", "_dayofweek"))
]

# Create monetary differences and interaction ratios among top numeric fields
top_num_cols = num_cols[:8]
for i in range(len(top_num_cols)):
    for j in range(i + 1, len(top_num_cols)):
        c1, c2 = top_num_cols[i], top_num_cols[j]
        df[f"{c1}_diff_{c2}"] = df[c1] - df[c2]
        df[f"{c1}_ratio_{c2}"] = (df[c1] + 1e-5) / (df[c2] + 1e-5)

# 3. Categorical encoding and group aggregations
cat_cols = [
    c for c in df.columns if df[c].dtype == "object" or df[c].dtype.name == "category"
]

# Group aggregation on key numeric columns by top categoricals
main_cats = cat_cols[:4] if len(cat_cols) >= 4 else cat_cols
main_nums = top_num_cols[:3] if len(top_num_cols) >= 3 else top_num_cols

for cat in main_cats:
    for num in main_nums:
        grouped = df.groupby(cat)[num].agg(["mean", "std", "min", "max"]).reset_index()
        grouped.columns = [
            cat,
            f"{cat}_{num}_mean",
            f"{cat}_{num}_std",
            f"{cat}_{num}_min",
            f"{cat}_{num}_max",
        ]
        df = df.merge(grouped, on=cat, how="left")

for col in cat_cols:
    df[col] = df[col].astype(str)
    # Frequency encoding
    freq = df[col].value_counts()
    df[f"{col}_freq"] = df[col].map(freq)
    # Label encoding
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# Separate train and test sets
X = df.iloc[:n_train].copy().reset_index(drop=True)
X_test = df.iloc[n_train:].copy().reset_index(drop=True)

# 5-Fold Cross-Validation Setup
kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))

lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.03,
    "num_leaves": 45,
    "max_depth": -1,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": RANDOM_SEED,
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
        num_boost_round=3000,
        valid_sets=[train_data, val_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )

    val_pred = model.predict(X_val_fold, num_iteration=model.best_iteration)
    if (y >= 0).all():
        val_pred = np.clip(val_pred, a_min=0, a_max=None)
    oof_preds[val_idx] = val_pred

    fold_rmse = np.sqrt(mean_squared_error(y_val_fold, val_pred))
    print(f"Fold {fold + 1} RMSE: {fold_rmse:.4f}")

    fold_test_pred = model.predict(X_test, num_iteration=model.best_iteration)
    if (y >= 0).all():
        fold_test_pred = np.clip(fold_test_pred, a_min=0, a_max=None)
    test_preds += fold_test_pred / kf.n_splits

# Overall validation metric
overall_rmse = np.sqrt(mean_squared_error(y, oof_preds))
print(f"\nOverall Out-of-Fold RMSE: {overall_rmse:.4f}")

# Save test predictions
submission = pd.DataFrame({"id": np.arange(len(test_preds)), TARGET_COL: test_preds})
submission_path = os.path.join(OUTPUT_DIR, "submission.csv")
submission.to_csv(submission_path, index=False)
print(
    f"Submission saved successfully to {submission_path} with shape {submission.shape}"
)
