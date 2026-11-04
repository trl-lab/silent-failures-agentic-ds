import os
import gc
import numpy as np
import pandas as pd
from itertools import product
from sklearn.metrics import root_mean_squared_error
import lightgbm as lgb


def main():
    os.makedirs("./working", exist_ok=True)

    # 1. Load data
    train = pd.read_parquet("./input/train.parquet")
    test = pd.read_parquet("./input/test.parquet")

    # Standardize column names if necessary
    if "ID" in test.columns and "id" not in test.columns:
        test = test.rename(columns={"ID": "id"})
    if "id" not in test.columns:
        test["id"] = np.arange(len(test))

    # Clean train anomalies
    if "item_price" in train.columns:
        train = train[(train["item_price"] > 0) & (train["item_price"] < 100000)]
    if "item_cnt_day" in train.columns:
        train = train[(train["item_cnt_day"] >= 0) & (train["item_cnt_day"] <= 1000)]

    # Target column check
    target_col = (
        "item_cnt_day" if "item_cnt_day" in train.columns else train.columns[-1]
    )

    # 2. Build monthly grid (to capture zero sales)
    grid = []
    for block in range(
        train["date_block_num"].min(), train["date_block_num"].max() + 1
    ):
        cur_train = train[train["date_block_num"] == block]
        shops = cur_train["shop_id"].unique()
        items = cur_train["item_id"].unique()
        grid.append(np.array(list(product([block], shops, items)), dtype="int32"))

    grid_df = pd.DataFrame(
        np.vstack(grid), columns=["date_block_num", "shop_id", "item_id"]
    )

    # Aggregate monthly sales
    monthly_sales = train.groupby(
        ["date_block_num", "shop_id", "item_id"], as_index=False
    ).agg(target=(target_col, "sum"))
    monthly_sales["target"] = monthly_sales["target"].clip(0, 20).astype(np.float32)

    # Merge grid with monthly target
    df = pd.merge(
        grid_df, monthly_sales, on=["date_block_num", "shop_id", "item_id"], how="left"
    )
    df["target"] = df["target"].fillna(0).astype(np.float32)

    # Prepare test set for date_block_num = 34
    test_block = train["date_block_num"].max() + 1
    test_df = test[["id", "shop_id", "item_id"]].copy()
    test_df["date_block_num"] = test_block
    test_df["target"] = 0.0

    # Combine train grid and test
    df = pd.concat(
        [df, test_df[["date_block_num", "shop_id", "item_id", "target"]]],
        ignore_index=True,
    )
    df["date_block_num"] = df["date_block_num"].astype(np.int16)
    df["shop_id"] = df["shop_id"].astype(np.int16)
    df["item_id"] = df["item_id"].astype(np.int32)

    del grid, grid_df, monthly_sales
    gc.collect()

    # 3. Feature Engineering: Lag Features
    def add_lags(df_base, group_cols, lag_col, lags, prefix):
        gb = (
            df_base.groupby(["date_block_num"] + group_cols)[lag_col]
            .mean()
            .reset_index()
        )
        gb.rename(columns={lag_col: f"{prefix}_mean"}, inplace=True)

        res = df_base
        for lag in lags:
            shifted = gb.copy()
            shifted["date_block_num"] += lag
            col_name = f"{prefix}_lag_{lag}"
            shifted.rename(columns={f"{prefix}_mean": col_name}, inplace=True)
            res = pd.merge(res, shifted, on=["date_block_num"] + group_cols, how="left")
            res[col_name] = res[col_name].fillna(0).astype(np.float32)
        return res

    # Interaction & aggregated lags
    df = add_lags(df, ["shop_id", "item_id"], "target", [1, 2, 3, 6], "shop_item")
    df = add_lags(df, ["item_id"], "target", [1, 2, 3, 6], "item")
    df = add_lags(df, ["shop_id"], "target", [1, 2, 3], "shop")

    # Calendar features
    df["month"] = (df["date_block_num"] % 12).astype(np.int8)

    # Filter out initial months used for lag generation
    df = df[df["date_block_num"] >= 6].reset_index(drop=True)

    # 4. Train / Validation Split
    val_block = test_block - 1
    train_mask = df["date_block_num"] < val_block
    val_mask = df["date_block_num"] == val_block
    test_mask = df["date_block_num"] == test_block

    features = [c for c in df.columns if c not in ["date_block_num", "target"]]

    X_train, y_train = df[train_mask][features], df[train_mask]["target"]
    X_val, y_val = df[val_mask][features], df[val_mask]["target"]
    X_test = df[test_mask][features]

    # 5. Train LightGBM model
    params = {
        "objective": "rmse",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "n_jobs": -1,
        "random_state": 42,
        "verbose": -1,
    }

    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        valid_sets=[dtrain, dval],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
    )

    # 6. Evaluation on Validation Set
    val_preds = np.clip(model.predict(X_val), 0, 20)
    val_rmse = root_mean_squared_error(y_val, val_preds)
    print(f"Validation RMSE: {val_rmse:.5f}")

    # 7. Predict on Test Set & Create Submission
    test_preds = np.clip(model.predict(X_test), 0, 20)

    test_result = df[test_mask][["shop_id", "item_id"]].copy()
    test_result["item_cnt_day"] = test_preds

    # Match predictions back to original test ID ordering
    submission = pd.merge(
        test[["id", "shop_id", "item_id"]],
        test_result,
        on=["shop_id", "item_id"],
        how="left",
    )
    submission["item_cnt_day"] = submission["item_cnt_day"].fillna(0).clip(0, 20)
    submission = (
        submission[["id", "item_cnt_day"]].sort_values("id").reset_index(drop=True)
    )

    submission.to_csv("./working/submission.csv", index=False)
    print(
        f"Submission saved successfully to ./working/submission.csv with shape {submission.shape}"
    )


if __name__ == "__main__":
    main()
