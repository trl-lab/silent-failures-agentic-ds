import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import root_mean_squared_error

# 1. Load input datasets
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Mark source datasets to avoid target leakage/NaNs during splitting
train_df["is_test"] = False
test_df["is_test"] = True
test_df["sales"] = np.nan

# Combine datasets for consistent feature computation
df_all = pd.concat([train_df, test_df], ignore_index=True)
df_all["date"] = pd.to_datetime(df_all["date"])
df_all = df_all.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)

# Log1p transformation of target variable
df_all["log_sales"] = np.log1p(df_all["sales"].clip(lower=0))

# 2. Extract Calendar & Payday Features
df_all["year"] = df_all["date"].dt.year
df_all["month"] = df_all["date"].dt.month
df_all["day"] = df_all["date"].dt.day
df_all["dayofweek"] = df_all["date"].dt.dayofweek
df_all["dayofyear"] = df_all["date"].dt.dayofyear
df_all["is_weekend"] = df_all["dayofweek"].isin([5, 6]).astype(int)
df_all["is_payday"] = ((df_all["day"] == 15) | (df_all["date"].dt.is_month_end)).astype(
    int
)

# 3. Promotional Features
df_all["onpromotion"] = df_all["onpromotion"].astype(float)
g_promo = df_all.groupby(["store_nbr", "family"])["onpromotion"]
df_all["promo_lag_16"] = g_promo.shift(16)
df_all["promo_roll_mean_7"] = g_promo.transform(
    lambda x: x.rolling(7, min_periods=1).mean()
)
df_all["promo_roll_mean_14"] = g_promo.transform(
    lambda x: x.rolling(14, min_periods=1).mean()
)

# 4. Target Lags & Rolling Statistics (Gap >= 16 days for 16-day forecast horizon)
g_target = df_all.groupby(["store_nbr", "family"])["log_sales"]
for lag in [16, 17, 18, 19, 20, 21, 28, 35, 42, 56]:
    df_all[f"target_lag_{lag}"] = g_target.shift(lag)

g_lag16 = df_all.groupby(["store_nbr", "family"])["target_lag_16"]
for w in [7, 14, 28, 56]:
    df_all[f"target_roll_mean_{w}"] = g_lag16.transform(
        lambda x: x.rolling(w, min_periods=1).mean()
    )
    df_all[f"target_roll_std_{w}"] = g_lag16.transform(
        lambda x: x.rolling(w, min_periods=1).std()
    )
    df_all[f"target_roll_max_{w}"] = g_lag16.transform(
        lambda x: x.rolling(w, min_periods=1).max()
    )
    df_all[f"target_roll_min_{w}"] = g_lag16.transform(
        lambda x: x.rolling(w, min_periods=1).min()
    )

# 5. Hierarchical Aggregated Demands
family_daily = (
    df_all[~df_all["is_test"]]
    .groupby(["family", "date"])["log_sales"]
    .mean()
    .reset_index()
)
family_daily = family_daily.sort_values(["family", "date"]).reset_index(drop=True)
family_daily["family_lag_16"] = family_daily.groupby("family")["log_sales"].shift(16)
family_daily["family_roll_mean_14"] = family_daily.groupby("family")[
    "family_lag_16"
].transform(lambda x: x.rolling(14, min_periods=1).mean())
df_all = df_all.merge(
    family_daily[["family", "date", "family_lag_16", "family_roll_mean_14"]],
    on=["family", "date"],
    how="left",
)

store_daily = (
    df_all[~df_all["is_test"]]
    .groupby(["store_nbr", "date"])["log_sales"]
    .mean()
    .reset_index()
)
store_daily = store_daily.sort_values(["store_nbr", "date"]).reset_index(drop=True)
store_daily["store_lag_16"] = store_daily.groupby("store_nbr")["log_sales"].shift(16)
store_daily["store_roll_mean_14"] = store_daily.groupby("store_nbr")[
    "store_lag_16"
].transform(lambda x: x.rolling(14, min_periods=1).mean())
df_all = df_all.merge(
    store_daily[["store_nbr", "date", "store_lag_16", "store_roll_mean_14"]],
    on=["store_nbr", "date"],
    how="left",
)

# Categorical column casting
df_all["store_nbr"] = df_all["store_nbr"].astype("category")
df_all["family"] = df_all["family"].astype("category")

# Filter modeling range from 2015 onwards
df_model = df_all[df_all["date"] >= "2015-01-01"].copy()

# 6. Dynamic Train / Validation / Test Splitting
max_train_date = df_model.loc[~df_model["is_test"], "date"].max()
val_start_date = max_train_date - pd.Timedelta(days=15)

train_mask = (~df_model["is_test"]) & (df_model["date"] < val_start_date)
val_mask = (~df_model["is_test"]) & (df_model["date"] >= val_start_date)
test_mask = df_model["is_test"]

feature_cols = [
    c
    for c in df_model.columns
    if c not in ["id", "date", "sales", "log_sales", "is_test"]
]

X_train, y_train = (
    df_model.loc[train_mask, feature_cols],
    df_model.loc[train_mask, "log_sales"],
)
X_val, y_val = df_model.loc[val_mask, feature_cols], df_model.loc[val_mask, "log_sales"]
X_test = df_model.loc[test_mask, feature_cols]

# 7. Train LightGBM Model
lgb_train = lgb.Dataset(
    X_train, label=y_train, categorical_feature=["store_nbr", "family"]
)
lgb_val = lgb.Dataset(
    X_val, label=y_val, reference=lgb_train, categorical_feature=["store_nbr", "family"]
)

params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 127,
    "max_depth": -1,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "min_child_samples": 20,
    "n_jobs": -1,
    "seed": 42,
    "verbose": -1,
}

model = lgb.train(
    params,
    lgb_train,
    num_boost_round=1500,
    valid_sets=[lgb_train, lgb_val],
    callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)],
)

# 8. Validation Evaluation (RMSE on log_sales equals RMSLE on sales)
val_preds = model.predict(X_val)
val_preds = np.clip(val_preds, 0, None)
val_rmsle = root_mean_squared_error(y_val, val_preds)
print(f"Validation RMSLE: {val_rmsle:.5f}")

# 9. Test Prediction & Submission Formatting
test_preds = model.predict(X_test)
test_preds_sales = np.expm1(np.clip(test_preds, 0, None))

test_sub_df = df_model.loc[test_mask, ["id"]].copy()
test_sub_df["sales"] = test_preds_sales

os.makedirs("./working", exist_ok=True)
sub = test_df[["id"]].merge(test_sub_df, on="id", how="left")
sub["sales"] = sub["sales"].fillna(0).clip(lower=0)
sub.to_csv("./working/submission.csv", index=False)
