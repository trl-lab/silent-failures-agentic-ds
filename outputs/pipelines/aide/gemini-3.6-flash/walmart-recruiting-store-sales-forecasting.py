import os
import numpy as np
import pandas as pd
import lightgbm as lgb

# 1. Load data
train_path = "./input/train.parquet"
test_path = "./input/test.parquet"

train = pd.read_parquet(train_path)
test = pd.read_parquet(test_path)

# Ensure Date is datetime
train["Date"] = pd.to_datetime(train["Date"])
test["Date"] = pd.to_datetime(test["Date"])

# Create lookup tables strictly from historical sales data
sales_lookup = (
    train[["Store", "Dept", "Date", "Weekly_Sales"]]
    .drop_duplicates(subset=["Store", "Dept", "Date"])
    .rename(columns={"Date": "lookup_date", "Weekly_Sales": "lookup_sales"})
)

dept_lookup = (
    train.groupby(["Dept", "Date"])["Weekly_Sales"]
    .sum()
    .reset_index()
    .rename(columns={"Date": "lookup_date", "Weekly_Sales": "dept_lookup_sales"})
)

store_lookup = (
    train.groupby(["Store", "Date"])["Weekly_Sales"]
    .sum()
    .reset_index()
    .rename(columns={"Date": "lookup_date", "Weekly_Sales": "store_lookup_sales"})
)

# Expanded lag offsets around 1 and 2 years (50..54 and 102..105)
lags = [50, 51, 52, 53, 54, 102, 103, 104, 105]


def add_hierarchical_lag_features(df, s_lookup, d_lookup, st_lookup, lag_list):
    for lag in lag_list:
        df["lookup_date"] = df["Date"] - pd.Timedelta(weeks=lag)

        # Store-Dept specific lag
        df = df.merge(s_lookup, on=["Store", "Dept", "lookup_date"], how="left")
        df[f"sales_lag_{lag}"] = df["lookup_sales"]
        df = df.drop(columns=["lookup_sales"])

        # Department macro lag
        df = df.merge(d_lookup, on=["Dept", "lookup_date"], how="left")
        df[f"dept_lag_{lag}"] = df["dept_lookup_sales"]
        df = df.drop(columns=["dept_lookup_sales"])

        # Store macro lag
        df = df.merge(st_lookup, on=["Store", "lookup_date"], how="left")
        df[f"store_lag_{lag}"] = df["store_lookup_sales"]
        df = df.drop(columns=["store_lookup_sales", "lookup_date"])

    # Summary statistics across lags
    sales_cols = [f"sales_lag_{w}" for w in lag_list]
    dept_cols = [f"dept_lag_{w}" for w in lag_list]
    store_cols = [f"store_lag_{w}" for w in lag_list]

    df["sales_lag_mean"] = df[sales_cols].mean(axis=1)
    df["sales_lag_std"] = df[sales_cols].std(axis=1)
    df["sales_lag_min"] = df[sales_cols].min(axis=1)
    df["sales_lag_max"] = df[sales_cols].max(axis=1)

    df["dept_lag_mean"] = df[dept_cols].mean(axis=1)
    df["dept_lag_std"] = df[dept_cols].std(axis=1)

    df["store_lag_mean"] = df[store_cols].mean(axis=1)
    df["store_lag_std"] = df[store_cols].std(axis=1)

    return df


train = add_hierarchical_lag_features(
    train, sales_lookup, dept_lookup, store_lookup, lags
)
test = add_hierarchical_lag_features(
    test, sales_lookup, dept_lookup, store_lookup, lags
)

# 2. Preprocessing & Temporal Feature Engineering
for df in [train, test]:
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Week"] = df["Date"].dt.isocalendar().week.astype(int)
    df["Day"] = df["Date"].dt.day
    df["IsHoliday"] = df["IsHoliday"].astype(int)

cat_cols = ["Store", "Dept"]
if "Type" in train.columns:
    cat_cols.append("Type")

for c in cat_cols:
    train[c] = train[c].astype("category")
    test[c] = test[c].astype("category")

feature_cols = [c for c in train.columns if c not in ["Date", "Weekly_Sales"]]

# 3. Validation setup (Time-based split: last 20% of dates for validation)
dates = train["Date"].sort_values().unique()
split_date = dates[int(len(dates) * 0.8)]

tr_mask = train["Date"] < split_date
val_mask = train["Date"] >= split_date

X_tr, y_tr = train.loc[tr_mask, feature_cols], train.loc[tr_mask, "Weekly_Sales"]
w_tr = np.where(train.loc[tr_mask, "IsHoliday"] == 1, 5, 1)

X_val, y_val = (
    train.loc[val_mask, feature_cols],
    train.loc[val_mask, "Weekly_Sales"],
)
w_val = np.where(train.loc[val_mask, "IsHoliday"] == 1, 5, 1)


# Metric calculation helper
def compute_wmae(y_true, y_pred, weights):
    return np.sum(weights * np.abs(y_true - y_pred)) / np.sum(weights)


# 4. Train LightGBM model on validation split
params = {
    "objective": "regression_l1",
    "metric": "mae",
    "boosting_type": "gbdt",
    "n_estimators": 1500,
    "learning_rate": 0.05,
    "num_leaves": 127,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}

train_data = lgb.Dataset(X_tr, label=y_tr, weight=w_tr)
val_data = lgb.Dataset(X_val, label=y_val, weight=w_val, reference=train_data)

model = lgb.train(
    params,
    train_data,
    valid_sets=[val_data],
    callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
)

val_preds = model.predict(X_val)
val_wmae = compute_wmae(y_val, val_preds, w_val)
print(f"Validation WMAE: {val_wmae:.4f}")

# 5. Retrain on full dataset and predict on test
X_full = train[feature_cols]
y_full = train["Weekly_Sales"]
w_full = np.where(train["IsHoliday"] == 1, 5, 1)

full_train_data = lgb.Dataset(X_full, label=y_full, weight=w_full)
final_model = lgb.train(
    params, full_train_data, num_boost_round=model.best_iteration or 600
)

X_test = test[feature_cols]
test_preds = final_model.predict(X_test)

# 6. Save submission file
os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame(
    {"id": np.arange(len(test_preds)), "Weekly_Sales": test_preds}
)
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved successfully.")
