import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold

# 1. Load Data
train_path = "./input/train.parquet"
test_path = "./input/test.parquet"

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

# Ensure Date is datetime
train_df["Date"] = pd.to_datetime(train_df["Date"])
test_df["Date"] = pd.to_datetime(test_df["Date"])

# Ensure IsHoliday is boolean
train_df["IsHoliday"] = train_df["IsHoliday"].astype(bool)
test_df["IsHoliday"] = test_df["IsHoliday"].astype(bool)


# 2. Metric Definition: WMAE (Holiday weight = 5, Non-Holiday weight = 1)
def calculate_wmae(y_true, y_pred, is_holiday):
    weights = np.where(is_holiday, 5.0, 1.0)
    return np.sum(weights * np.abs(y_true - y_pred)) / np.sum(weights)


# 3. Feature Engineering
def engineer_features(df):
    df = df.copy()

    # Calendar features
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Week"] = df["Date"].dt.isocalendar().week.astype(int)
    df["Day"] = df["Date"].dt.day
    df["DayOfYear"] = df["Date"].dt.dayofyear
    df["Quarter"] = df["Date"].dt.quarter
    df["IsHolidayInt"] = df["IsHoliday"].astype(int)

    # Specific Walmart holiday indicators
    df["IsSuperBowl"] = (df["Week"] == 6).astype(int)
    df["IsLaborDay"] = (df["Week"] == 36).astype(int)
    df["IsThanksgiving"] = (df["Week"] == 47).astype(int)
    df["IsChristmas"] = ((df["Week"] == 51) | (df["Week"] == 52)).astype(int)

    # Continuous holiday distance features
    df["Dist_Thanksgiving"] = np.abs(df["Week"] - 47)
    df["Dist_Christmas"] = np.minimum(np.abs(df["Week"] - 51), np.abs(df["Week"] - 52))
    df["Dist_SuperBowl"] = np.abs(df["Week"] - 6)
    df["Dist_LaborDay"] = np.abs(df["Week"] - 36)

    # Markdown feature aggregations
    md_cols = [f"MarkDown{i}" for i in range(1, 6) if f"MarkDown{i}" in df.columns]
    if md_cols:
        for col in md_cols:
            df[f"{col}_isna"] = df[col].isna().astype(int)
            df[col] = df[col].fillna(0.0)
            df[f"{col}_log"] = np.log1p(np.maximum(0, df[col]))

        df["Total_MarkDown"] = df[md_cols].sum(axis=1)
        df["Total_MarkDown_log"] = np.log1p(np.maximum(0, df["Total_MarkDown"]))
        df["MarkDown_Count"] = (df[md_cols] > 0).sum(axis=1)

    # Cyclical week encoding
    df["Week_sin"] = np.sin(2 * np.pi * df["Week"] / 52.0)
    df["Week_cos"] = np.cos(2 * np.pi * df["Week"] / 52.0)

    return df


train_df = engineer_features(train_df)
test_df = engineer_features(test_df)

# Categorical column handling
cat_cols = ["Store", "Dept"]
if "Type" in train_df.columns:
    cat_cols.append("Type")

for col in cat_cols:
    train_df[col] = train_df[col].astype("category")
    test_df[col] = test_df[col].astype("category")

# Identify feature columns
excluded_cols = {"Weekly_Sales", "Date", "IsHoliday"}
feature_cols = [c for c in train_df.columns if c not in excluded_cols]

# 4. Cross-Validation Setup
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_preds = np.zeros(len(train_df))
test_preds = np.zeros(len(test_df))

lgb_params = {
    "objective": "regression_l1",
    "metric": "l1",
    "boosting_type": "gbdt",
    "n_estimators": 3000,
    "learning_rate": 0.05,
    "num_leaves": 255,
    "min_child_samples": 25,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.85,
    "bagging_freq": 1,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}

# 5. Model Training & Out-of-Fold Cross-Validation
print("Starting 5-fold Cross-Validation...")

for fold, (train_idx, val_idx) in enumerate(kf.split(train_df)):
    X_train_df = train_df.iloc[train_idx].copy()
    X_val_df = train_df.iloc[val_idx].copy()

    y_train = X_train_df["Weekly_Sales"].values
    y_val = X_val_df["Weekly_Sales"].values

    # Calculate sample weights for WMAE optimization
    w_train = np.where(X_train_df["IsHoliday"].values, 5.0, 1.0)
    w_val = np.where(X_val_df["IsHoliday"].values, 5.0, 1.0)

    # Target encodings computed strictly on train fold
    store_dept_stats = (
        X_train_df.groupby(["Store", "Dept"], observed=False)["Weekly_Sales"]
        .agg(["mean", "median", "std", "min", "max"])
        .reset_index()
    )
    store_dept_stats.columns = [
        "Store",
        "Dept",
        "store_dept_mean",
        "store_dept_median",
        "store_dept_std",
        "store_dept_min",
        "store_dept_max",
    ]

    dept_week_stats = (
        X_train_df.groupby(["Dept", "Week"], observed=False)["Weekly_Sales"]
        .agg(["mean", "median"])
        .reset_index()
    )
    dept_week_stats.columns = [
        "Dept",
        "Week",
        "dept_week_mean",
        "dept_week_median",
    ]

    store_week_stats = (
        X_train_df.groupby(["Store", "Week"], observed=False)["Weekly_Sales"]
        .agg(["mean", "median"])
        .reset_index()
    )
    store_week_stats.columns = [
        "Store",
        "Week",
        "store_week_mean",
        "store_week_median",
    ]

    # Merge aggregations
    def merge_stats(df):
        return (
            df.merge(store_dept_stats, on=["Store", "Dept"], how="left")
            .merge(dept_week_stats, on=["Dept", "Week"], how="left")
            .merge(store_week_stats, on=["Store", "Week"], how="left")
        )

    X_train_fold = merge_stats(X_train_df)
    X_val_fold = merge_stats(X_val_df)
    X_test_fold = merge_stats(test_df)

    fold_feature_cols = [
        c
        for c in X_train_fold.columns
        if c not in excluded_cols
        and c
        in feature_cols
        + [
            "store_dept_mean",
            "store_dept_median",
            "store_dept_std",
            "store_dept_min",
            "store_dept_max",
            "dept_week_mean",
            "dept_week_median",
            "store_week_mean",
            "store_week_median",
        ]
    ]

    X_tr = X_train_fold[fold_feature_cols]
    X_v = X_val_fold[fold_feature_cols]
    X_te = X_test_fold[fold_feature_cols]

    model = lgb.LGBMRegressor(**lgb_params)
    model.fit(
        X_tr,
        y_train,
        sample_weight=w_train,
        eval_set=[(X_v, y_val)],
        eval_sample_weight=[w_val],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )

    val_pred = model.predict(X_v)
    oof_preds[val_idx] = val_pred
    test_preds += model.predict(X_te) / n_splits

    fold_wmae = calculate_wmae(y_val, val_pred, X_val_df["IsHoliday"].values)
    print(f"Fold {fold + 1} WMAE: {fold_wmae:.4f}")

overall_wmae = calculate_wmae(
    train_df["Weekly_Sales"].values, oof_preds, train_df["IsHoliday"].values
)
print(f"Overall Out-of-Fold WMAE: {overall_wmae:.4f}")

# 6. Save Submission
os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame(
    {"id": np.arange(len(test_preds)), "Weekly_Sales": test_preds}
)
submission.to_csv("./working/submission.csv", index=False)
print(
    "Submission saved successfully to ./working/submission.csv with shape:",
    submission.shape,
)
