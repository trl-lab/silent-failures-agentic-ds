import os
import numpy as np
import pandas as pd
import lightgbm as lgb

os.makedirs("./working", exist_ok=True)

train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

train["Date"] = pd.to_datetime(train["Date"])
test["Date"] = pd.to_datetime(test["Date"])

train["IsHoliday"] = train["IsHoliday"].astype(int)
if "IsHoliday" not in test.columns:
    test["IsHoliday"] = False
test["IsHoliday"] = test["IsHoliday"].fillna(False).astype(int)

train["Weekly_Sales"] = train["Weekly_Sales"].astype(float)

sales_lookup = train.groupby(["Store", "Dept", "Date"], as_index=False)[
    "Weekly_Sales"
].sum()

LAGS = [48, 49, 50, 51, 52, 53, 54, 55, 56]


def add_features(df, lookup):
    out = df.copy()

    out["year"] = out["Date"].dt.year
    out["month"] = out["Date"].dt.month
    out["week"] = out["Date"].dt.isocalendar().week.astype(int)
    out["dayofweek"] = out["Date"].dt.dayofweek
    out["dayofyear"] = out["Date"].dt.dayofyear
    out["quarter"] = out["Date"].dt.quarter

    for lag in LAGS:
        keys = out[["Store", "Dept"]].copy()
        keys["Date"] = out["Date"] - pd.Timedelta(weeks=lag)
        keys = keys.merge(lookup, on=["Store", "Dept", "Date"], how="left")
        out[f"sales_lag_{lag}"] = keys["Weekly_Sales"].fillna(0.0).to_numpy()

    return out


train_feat = add_features(train, sales_lookup)
test_feat = add_features(test, sales_lookup)

feature_cols = [
    "Store",
    "Dept",
    "IsHoliday",
    "year",
    "month",
    "week",
    "dayofweek",
    "dayofyear",
    "quarter",
] + [f"sales_lag_{lag}" for lag in LAGS]

categorical_cols = ["Store", "Dept"]

unique_dates = np.sort(train["Date"].unique())
val_dates = unique_dates[-13:]

val_mask = train_feat["Date"].isin(val_dates)
train_mask = ~val_mask

X_train = train_feat.loc[train_mask, feature_cols]
y_train = train_feat.loc[train_mask, "Weekly_Sales"]

X_val = train_feat.loc[val_mask, feature_cols]
y_val = train_feat.loc[val_mask, "Weekly_Sales"]

params = {
    "objective": "regression_l1",
    "metric": "l1",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "min_child_samples": 50,
    "lambda_l1": 0.1,
    "lambda_l2": 0.1,
    "seed": 42,
}

dtrain = lgb.Dataset(X_train, label=y_train, categorical_feature=categorical_cols)
dval = lgb.Dataset(
    X_val,
    label=y_val,
    categorical_feature=categorical_cols,
    reference=dtrain,
)

model = lgb.train(
    params,
    dtrain,
    num_boost_round=2000,
    valid_sets=[dval],
    valid_names=["valid"],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
)

val_pred = model.predict(X_val, num_iteration=model.best_iteration)


def wmae(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    denom = np.sum(y_true)
    if denom <= 0:
        denom = np.sum(np.abs(y_true))

    if denom <= 0:
        return np.inf

    return float(np.sum(np.abs(y_true - y_pred)) / denom)


print(f"Validation WMAE: {wmae(y_val, val_pred):.6f}")

dtrain_full = lgb.Dataset(
    train_feat[feature_cols],
    label=train_feat["Weekly_Sales"],
    categorical_feature=categorical_cols,
)

best_iter = max(1, int(model.best_iteration))

model_full = lgb.train(
    params,
    dtrain_full,
    num_boost_round=best_iter,
    callbacks=[lgb.log_evaluation(-1)],
)

test_pred = model_full.predict(test_feat[feature_cols], num_iteration=best_iter)

submission = pd.DataFrame(
    {
        "id": np.arange(len(test), dtype=int),
        "Weekly_Sales": test_pred,
    }
)

submission.to_csv("./working/submission.csv", index=False)
