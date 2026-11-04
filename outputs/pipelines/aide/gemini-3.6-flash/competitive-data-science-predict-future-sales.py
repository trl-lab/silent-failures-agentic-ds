import os
import numpy as np
import pandas as pd
from itertools import product
import lightgbm as lgb

# 1. Load Data
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

# 2. Monthly Target Aggregation
monthly_sales = train.groupby(["date_block_num", "shop_id", "item_id"], as_index=False)[
    "item_cnt_day"
].sum()
monthly_sales.rename(columns={"item_cnt_day": "item_cnt_month"}, inplace=True)
monthly_sales["item_cnt_month"] = monthly_sales["item_cnt_month"].clip(0, 20)

# 3. Lifecycle & Price Trend Features
item_first_sale = train.groupby("item_id")["date_block_num"].min().reset_index()
item_first_sale.rename(columns={"date_block_num": "item_first_block"}, inplace=True)

shop_item_first_sale = (
    train.groupby(["shop_id", "item_id"])["date_block_num"].min().reset_index()
)
shop_item_first_sale.rename(
    columns={"date_block_num": "shop_item_first_block"}, inplace=True
)

item_monthly_price = (
    train.groupby(["date_block_num", "item_id"])["item_price"].mean().reset_index()
)
item_global_price = train.groupby("item_id")["item_price"].mean().reset_index()
item_global_price.rename(columns={"item_price": "item_avg_global_price"}, inplace=True)

item_price_df = pd.merge(
    item_monthly_price, item_global_price, on="item_id", how="left"
)
item_price_df["item_price_ratio"] = item_price_df["item_price"] / (
    item_price_df["item_avg_global_price"] + 1e-5
)
item_price_df = item_price_df[["date_block_num", "item_id", "item_price_ratio"]]

# 4. Construct Monthly Grid
grid = []
for block in range(34):
    cur_sales = train[train["date_block_num"] == block]
    shops = cur_sales["shop_id"].unique()
    items = cur_sales["item_id"].unique()
    grid.append(np.array(list(product([block], shops, items)), dtype="int32"))

test_pairs = test[["shop_id", "item_id"]].drop_duplicates()
test_pairs["date_block_num"] = 34
grid.append(test_pairs[["date_block_num", "shop_id", "item_id"]].values)

grid_df = pd.DataFrame(
    np.vstack(grid), columns=["date_block_num", "shop_id", "item_id"]
)
grid_df.sort_values(["date_block_num", "shop_id", "item_id"], inplace=True)
grid_df.reset_index(drop=True, inplace=True)

grid_df = pd.merge(
    grid_df, monthly_sales, on=["date_block_num", "shop_id", "item_id"], how="left"
)
grid_df["item_cnt_month"] = grid_df["item_cnt_month"].fillna(0)

# Add Group Means
item_means = grid_df.groupby(["date_block_num", "item_id"], as_index=False)[
    "item_cnt_month"
].mean()
item_means.rename(columns={"item_cnt_month": "item_avg_cnt"}, inplace=True)

shop_means = grid_df.groupby(["date_block_num", "shop_id"], as_index=False)[
    "item_cnt_month"
].mean()
shop_means.rename(columns={"item_cnt_month": "shop_avg_cnt"}, inplace=True)

grid_df = pd.merge(grid_df, item_means, on=["date_block_num", "item_id"], how="left")
grid_df = pd.merge(grid_df, shop_means, on=["date_block_num", "shop_id"], how="left")
grid_df = pd.merge(grid_df, item_price_df, on=["date_block_num", "item_id"], how="left")


# 5. Lag Feature Generator
def add_lag(df, lags, col, on=["date_block_num", "shop_id", "item_id"]):
    tmp = df[on + [col]].drop_duplicates()
    for i in lags:
        shifted = tmp.copy()
        shifted.columns = on + [f"{col}_lag_{i}"]
        shifted["date_block_num"] += i
        df = pd.merge(df, shifted, on=on, how="left")
        df[f"{col}_lag_{i}"] = df[f"{col}_lag_{i}"].fillna(0)
    return df


grid_df = add_lag(
    grid_df,
    [1, 2, 3, 6, 12],
    "item_cnt_month",
    on=["date_block_num", "shop_id", "item_id"],
)
grid_df = add_lag(grid_df, [1, 2, 3], "item_avg_cnt", on=["date_block_num", "item_id"])
grid_df = add_lag(grid_df, [1, 2, 3], "shop_avg_cnt", on=["date_block_num", "shop_id"])
grid_df = add_lag(grid_df, [1], "item_price_ratio", on=["date_block_num", "item_id"])

# 6. Sales Recency & Target Dynamics Features
monthly_pos_sales = monthly_sales[monthly_sales["item_cnt_month"] > 0][
    ["date_block_num", "shop_id", "item_id"]
].drop_duplicates()
monthly_pos_sales["shop_item_last_sale_block"] = monthly_pos_sales["date_block_num"]

item_pos_sales = (
    monthly_sales.groupby(["date_block_num", "item_id"])["item_cnt_month"]
    .sum()
    .reset_index()
)
item_pos_sales = item_pos_sales[item_pos_sales["item_cnt_month"] > 0][
    ["date_block_num", "item_id"]
].drop_duplicates()
item_pos_sales["item_last_sale_block"] = item_pos_sales["date_block_num"]

grid_df = grid_df.sort_values("date_block_num").reset_index(drop=True)
monthly_pos_sales = monthly_pos_sales.sort_values("date_block_num").reset_index(
    drop=True
)
item_pos_sales = item_pos_sales.sort_values("date_block_num").reset_index(drop=True)

# Merge shop-item last sale block
grid_df = pd.merge_asof(
    grid_df,
    monthly_pos_sales,
    on="date_block_num",
    by=["shop_id", "item_id"],
    allow_exact_matches=False,
    direction="backward",
)
grid_df["shop_item_months_since_last_sale"] = (
    grid_df["date_block_num"] - grid_df["shop_item_last_sale_block"]
).fillna(-1)
grid_df.drop(columns=["shop_item_last_sale_block"], inplace=True)

# Merge national item last sale block
grid_df = pd.merge_asof(
    grid_df,
    item_pos_sales,
    on="date_block_num",
    by="item_id",
    allow_exact_matches=False,
    direction="backward",
)
grid_df["item_months_since_last_sale"] = (
    grid_df["date_block_num"] - grid_df["item_last_sale_block"]
).fillna(-1)
grid_df.drop(columns=["item_last_sale_block"], inplace=True)

# Relative Demand Intensity
grid_df["shop_item_ratio_lag_1"] = grid_df["item_cnt_month_lag_1"] / (
    grid_df["item_avg_cnt_lag_1"] + 1e-5
)

# Lifecycle & Momentum Features
grid_df = pd.merge(grid_df, item_first_sale, on="item_id", how="left")
grid_df = pd.merge(grid_df, shop_item_first_sale, on=["shop_id", "item_id"], how="left")

grid_df["item_age"] = grid_df["date_block_num"] - grid_df["item_first_block"].fillna(
    grid_df["date_block_num"]
)
grid_df["shop_item_age"] = grid_df["date_block_num"] - grid_df[
    "shop_item_first_block"
].fillna(grid_df["date_block_num"])

grid_df["cnt_roll_mean_3"] = (
    grid_df["item_cnt_month_lag_1"]
    + grid_df["item_cnt_month_lag_2"]
    + grid_df["item_cnt_month_lag_3"]
) / 3.0
grid_df["cnt_roll_mean_6"] = (
    grid_df["item_cnt_month_lag_1"]
    + grid_df["item_cnt_month_lag_2"]
    + grid_df["item_cnt_month_lag_3"]
    + grid_df["item_cnt_month_lag_6"]
) / 4.0

grid_df["delta_lag_1_2"] = (
    grid_df["item_cnt_month_lag_1"] - grid_df["item_cnt_month_lag_2"]
)
grid_df["delta_lag_1_3"] = (
    grid_df["item_cnt_month_lag_1"] - grid_df["item_cnt_month_lag_3"]
)
grid_df["month"] = grid_df["date_block_num"] % 12

# 7. Train/Val/Test Split
train_df = grid_df[grid_df["date_block_num"].between(12, 32)].copy()
val_df = grid_df[grid_df["date_block_num"] == 33].copy()
test_block_df = grid_df[grid_df["date_block_num"] == 34].copy()

test_df = pd.merge(test, test_block_df, on=["shop_id", "item_id"], how="left").fillna(0)

features = [
    "shop_id",
    "item_id",
    "month",
    "item_cnt_month_lag_1",
    "item_cnt_month_lag_2",
    "item_cnt_month_lag_3",
    "item_cnt_month_lag_6",
    "item_cnt_month_lag_12",
    "item_avg_cnt_lag_1",
    "item_avg_cnt_lag_2",
    "item_avg_cnt_lag_3",
    "shop_avg_cnt_lag_1",
    "shop_avg_cnt_lag_2",
    "shop_avg_cnt_lag_3",
    "cnt_roll_mean_3",
    "cnt_roll_mean_6",
    "delta_lag_1_2",
    "delta_lag_1_3",
    "item_price_ratio_lag_1",
    "item_age",
    "shop_item_age",
    "shop_item_months_since_last_sale",
    "item_months_since_last_sale",
    "shop_item_ratio_lag_1",
]

X_train, y_train = train_df[features], train_df["item_cnt_month"]
X_val, y_val = val_df[features], val_df["item_cnt_month"]
X_test = test_df[features]

# 8. Model Training
model = lgb.LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.03,
    num_leaves=63,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
)

model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    eval_metric="rmse",
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
)

# 9. Validation Evaluation
val_preds = model.predict(X_val).clip(0, 20)
val_rmse = np.sqrt(np.mean((val_preds - y_val.values) ** 2))
print(f"Validation RMSE: {val_rmse:.5f}")

# 10. Generate Predictions & Submission
test_preds = model.predict(X_test).clip(0, 20)

os.makedirs("./working", exist_ok=True)
if "id" in test.columns:
    submission = pd.DataFrame({"id": test["id"], "item_cnt_day": test_preds})
elif "ID" in test.columns:
    submission = pd.DataFrame({"id": test["ID"], "item_cnt_day": test_preds})
else:
    submission = pd.DataFrame(
        {"id": np.arange(len(test_preds)), "item_cnt_day": test_preds}
    )

submission.to_csv("./working/submission.csv", index=False)
print("Submission saved successfully.")
