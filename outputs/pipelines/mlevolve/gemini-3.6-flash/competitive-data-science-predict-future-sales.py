import gc
import json
import os
import re
from itertools import product
import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize, nnls
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb


def run_pipeline():
    input_dir = "./input"
    submission_dir = "./submission"
    os.makedirs(submission_dir, exist_ok=True)

    print("=== Step 1: Data Processing & Feature Engineering ===")
    train_sales = pd.read_parquet(os.path.join(input_dir, "train.parquet"))
    test_df = pd.read_parquet(os.path.join(input_dir, "test.parquet"))

    def load_auxiliary_file(filename):
        for candidate in [
            os.path.join(input_dir, f"{filename}.csv"),
            os.path.join(input_dir, "original", f"{filename}.csv"),
            os.path.join(input_dir, f"{filename}.parquet"),
            os.path.join(input_dir, "original", f"{filename}.parquet"),
        ]:
            if os.path.exists(candidate):
                return pd.read_parquet(candidate) if candidate.endswith(".parquet") else pd.read_csv(candidate)
        return pd.DataFrame()

    items = load_auxiliary_file("items")
    item_cats = load_auxiliary_file("item_categories")
    shops = load_auxiliary_file("shops")

    shop_mapping = {0: 57, 1: 58, 10: 11}
    train_sales["shop_id"] = (
        train_sales["shop_id"].replace(shop_mapping).astype(np.int16)
    )
    test_df["shop_id"] = test_df["shop_id"].replace(shop_mapping).astype(np.int16)

    train_sales = train_sales[
        (train_sales["item_price"] > 0) & (train_sales["item_price"] < 100000)
    ]
    train_sales["item_cnt_day"] = train_sales["item_cnt_day"].clip(0, 1000)

    if not shops.empty:
        shops["shop_id"] = shops["shop_id"].replace(shop_mapping).astype(np.int16)
        shops["city"] = shops["shop_name"].apply(
            lambda x: x.split()[0].replace("!", "").replace('"', "").replace(".", "")
        )
        shops["city_code"] = LabelEncoder().fit_transform(shops["city"]).astype(np.int8)
        shop_city_map = shops.set_index("shop_id")["city_code"].to_dict()
    else:
        shop_city_map = {}

    if not item_cats.empty:
        split_cats = item_cats["item_category_name"].str.split("-")
        item_cats["main_cat"] = split_cats.apply(lambda x: x[0].strip())
        item_cats["sub_cat"] = split_cats.apply(
            lambda x: x[1].strip() if len(x) > 1 else x[0].strip()
        )
        item_cats["main_cat_code"] = (
            LabelEncoder().fit_transform(item_cats["main_cat"]).astype(np.int8)
        )
        item_cats["sub_cat_code"] = (
            LabelEncoder().fit_transform(item_cats["sub_cat"]).astype(np.int8)
        )

    if not items.empty and not item_cats.empty:
        items = items.merge(
            item_cats[["item_category_id", "main_cat_code", "sub_cat_code"]],
            on="item_category_id",
            how="left",
        )

    grid = []
    for month in range(34):
        sales_m = train_sales[train_sales["date_block_num"] == month]
        shops_m = sales_m["shop_id"].unique()
        items_m = sales_m["item_id"].unique()
        grid.append(np.array(list(product([month], shops_m, items_m)), dtype=np.int32))

    grid = pd.DataFrame(
        np.vstack(grid), columns=["date_block_num", "shop_id", "item_id"]
    )
    grid["date_block_num"] = grid["date_block_num"].astype(np.int8)
    grid["shop_id"] = grid["shop_id"].astype(np.int16)
    grid["item_id"] = grid["item_id"].astype(np.int16)

    test_id_col = (
        "id" if "id" in test_df.columns else ("ID" if "ID" in test_df.columns else None)
    )
    test_grid = test_df[["shop_id", "item_id"]].copy()
    test_grid["date_block_num"] = np.int8(34)
    if test_id_col:
        test_grid[test_id_col] = test_df[test_id_col]

    grid = pd.concat([grid, test_grid], ignore_index=True)

    if not items.empty and "item_category_id" in items.columns:
        item_cols = ["item_id", "item_category_id"]
        if "main_cat_code" in items.columns:
            item_cols.extend(["main_cat_code", "sub_cat_code"])
        grid = grid.merge(
            items[item_cols],
            on="item_id",
            how="left",
        )
        grid["item_category_id"] = grid["item_category_id"].fillna(-1).astype(np.int16)
        if "main_cat_code" in grid.columns:
            grid["main_cat_code"] = grid["main_cat_code"].fillna(-1).astype(np.int8)
            grid["sub_cat_code"] = grid["sub_cat_code"].fillna(-1).astype(np.int8)

    grid["city_code"] = grid["shop_id"].map(shop_city_map).fillna(-1).astype(np.int8)

    monthly_sales = (
        train_sales.groupby(["date_block_num", "shop_id", "item_id"])["item_cnt_day"]
        .sum()
        .reset_index()
    )
    monthly_sales.rename(columns={"item_cnt_day": "item_cnt_month"}, inplace=True)
    monthly_sales["item_cnt_month"] = monthly_sales["item_cnt_month"].clip(0, 20)

    grid = grid.merge(
        monthly_sales, on=["date_block_num", "shop_id", "item_id"], how="left"
    )
    grid["item_cnt_month"] = grid["item_cnt_month"].fillna(0).astype(np.float32)

    def add_lag_features(df, lags, target_col, group_cols):
        col_prefix = f"target_{'_'.join(group_cols)}" if group_cols else "target_global"
        for lag in lags:
            col_name = f"{col_prefix}_lag_{lag}"
            if group_cols:
                gb = (
                    df.groupby(group_cols + ["date_block_num"])[target_col]
                    .mean()
                    .reset_index()
                )
            else:
                gb = df.groupby(["date_block_num"])[target_col].mean().reset_index()
            gb.rename(columns={target_col: col_name}, inplace=True)
            gb["date_block_num"] = (gb["date_block_num"] + lag).astype(np.int8)

            merge_cols = (
                group_cols + ["date_block_num"] if group_cols else ["date_block_num"]
            )
            df = df.merge(gb, on=merge_cols, how="left")
            df[col_name] = df[col_name].fillna(0).astype(np.float32)
        return df

    grid = add_lag_features(
        grid,
        [1, 2, 3, 4, 5, 6, 12],
        "item_cnt_month",
        ["shop_id", "item_id"],
    )
    grid = add_lag_features(grid, [1, 2, 3, 6, 12], "item_cnt_month", ["item_id"])
    grid = add_lag_features(grid, [1, 2, 3, 6, 12], "item_cnt_month", ["shop_id"])
    if "item_category_id" in grid.columns:
        grid = add_lag_features(grid, [1, 2, 3, 6], "item_cnt_month", ["item_category_id"])
        grid = add_lag_features(
            grid, [1, 2, 3], "item_cnt_month", ["shop_id", "item_category_id"]
        )
    if "main_cat_code" in grid.columns:
        grid = add_lag_features(
            grid, [1, 2, 3], "item_cnt_month", ["shop_id", "main_cat_code"]
        )
    if "sub_cat_code" in grid.columns:
        grid = add_lag_features(
            grid, [1, 2, 3], "item_cnt_month", ["shop_id", "sub_cat_code"]
        )
    if "city_code" in grid.columns:
        grid = add_lag_features(grid, [1, 2], "item_cnt_month", ["city_code", "item_id"])
    grid = add_lag_features(grid, [1, 2, 3], "item_cnt_month", [])

    item_prices = (
        train_sales.groupby(["date_block_num", "item_id"])["item_price"]
        .mean()
        .reset_index()
    )
    global_item_prices = (
        train_sales.groupby("item_id")["item_price"]
        .mean()
        .reset_index()
        .rename(columns={"item_price": "global_avg_price"})
    )

    item_prices["date_block_num"] = (item_prices["date_block_num"] + 1).astype(np.int8)
    item_prices.rename(columns={"item_price": "item_price_lag_1"}, inplace=True)

    grid = grid.merge(item_prices, on=["date_block_num", "item_id"], how="left")
    grid = grid.merge(global_item_prices, on="item_id", how="left")

    grid["item_price_lag_1"] = (
        grid["item_price_lag_1"]
        .fillna(grid["global_avg_price"])
        .fillna(0)
        .astype(np.float32)
    )
    grid["global_avg_price"] = grid["global_avg_price"].fillna(0).astype(np.float32)

    grid["price_ratio_lag1"] = (
        (grid["item_price_lag_1"] - grid["global_avg_price"])
        / (grid["global_avg_price"] + 1e-5)
    ).astype(np.float32)

    item_first_sale = (
        train_sales.groupby("item_id")["date_block_num"]
        .min()
        .reset_index()
        .rename(columns={"date_block_num": "item_first_month"})
    )
    shop_item_first_sale = (
        train_sales.groupby(["shop_id", "item_id"])["date_block_num"]
        .min()
        .reset_index()
        .rename(columns={"date_block_num": "shop_item_first_month"})
    )

    grid = grid.merge(item_first_sale, on="item_id", how="left")
    grid = grid.merge(shop_item_first_sale, on=["shop_id", "item_id"], how="left")

    grid["item_age"] = (
        grid["date_block_num"] - grid["item_first_month"].fillna(grid["date_block_num"])
    ).astype(np.int8)
    grid["shop_item_age"] = (
        grid["date_block_num"]
        - grid["shop_item_first_month"].fillna(grid["date_block_num"])
    ).astype(np.int8)
    grid["is_new_item"] = (grid["item_age"] <= 1).astype(np.int8)

    grid.drop(columns=["item_first_month", "shop_item_first_month"], inplace=True)

    lag_col = "target_shop_id_item_id_lag_"
    grid["momentum_lag1_lag2"] = (grid[f"{lag_col}1"] - grid[f"{lag_col}2"]).astype(
        np.float32
    )
    grid["momentum_lag1_lag3"] = (grid[f"{lag_col}1"] - grid[f"{lag_col}3"]).astype(
        np.float32
    )

    # Multi-month rolling lag statistics (3-month and 6-month target lag means and stds)
    ewm_w = np.array([1.0, 0.5, 0.25], dtype=np.float32)
    ewm_w /= ewm_w.sum()

    grid["target_shop_id_item_id_lag_ewm_3"] = (
        grid[f"{lag_col}1"] * ewm_w[0]
        + grid[f"{lag_col}2"] * ewm_w[1]
        + grid[f"{lag_col}3"] * ewm_w[2]
    ).astype(np.float32)

    grid["target_item_id_lag_ewm_3"] = (
        grid["target_item_id_lag_1"] * ewm_w[0]
        + grid["target_item_id_lag_2"] * ewm_w[1]
        + grid["target_item_id_lag_3"] * ewm_w[2]
    ).astype(np.float32)

    if "target_shop_id_main_cat_code_lag_1" in grid.columns:
        grid["target_shop_id_main_cat_code_lag_ewm_3"] = (
            grid["target_shop_id_main_cat_code_lag_1"] * ewm_w[0]
            + grid["target_shop_id_main_cat_code_lag_2"] * ewm_w[1]
            + grid["target_shop_id_main_cat_code_lag_3"] * ewm_w[2]
        ).astype(np.float32)

    if "target_shop_id_sub_cat_code_lag_1" in grid.columns:
        grid["target_shop_id_sub_cat_code_lag_ewm_3"] = (
            grid["target_shop_id_sub_cat_code_lag_1"] * ewm_w[0]
            + grid["target_shop_id_sub_cat_code_lag_2"] * ewm_w[1]
            + grid["target_shop_id_sub_cat_code_lag_3"] * ewm_w[2]
        ).astype(np.float32)

    grid["target_shop_id_item_id_lag_mean_3"] = grid[
        [f"{lag_col}1", f"{lag_col}2", f"{lag_col}3"]
    ].mean(axis=1).astype(np.float32)

    grid["target_shop_id_item_id_lag_mean_6"] = grid[
        [f"{lag_col}1", f"{lag_col}2", f"{lag_col}3", f"{lag_col}4", f"{lag_col}5", f"{lag_col}6"]
    ].mean(axis=1).astype(np.float32)

    grid["target_shop_id_item_id_lag_std_3"] = grid[
        [f"{lag_col}1", f"{lag_col}2", f"{lag_col}3"]
    ].std(axis=1).astype(np.float32)

    grid["target_shop_id_item_id_lag_std_6"] = grid[
        [f"{lag_col}1", f"{lag_col}2", f"{lag_col}3", f"{lag_col}4", f"{lag_col}5", f"{lag_col}6"]
    ].std(axis=1).astype(np.float32)

    grid["seasonal_lag_ratio"] = (
        grid[f"{lag_col}1"] / (grid[f"{lag_col}12"] + 1e-5)
    ).astype(np.float32)

    grid["relative_shop_item_share"] = (
        grid[f"{lag_col}1"] / (grid["target_item_id_lag_1"] + 1e-5)
    ).astype(np.float32)

    # Time-since-last-sale features
    sales_pos = train_sales[train_sales["item_cnt_day"] > 0]
    item_sales_map = sales_pos.groupby("date_block_num")["item_id"].unique().to_dict()

    shop_item_sales = sales_pos[["date_block_num", "shop_id", "item_id"]].drop_duplicates()
    shop_item_sales["shop_item_key"] = shop_item_sales["shop_id"].astype(np.int64) * 100000 + shop_item_sales["item_id"].astype(np.int64)
    shop_item_map = shop_item_sales.groupby("date_block_num")["shop_item_key"].unique().to_dict()

    last_sale_item = {}
    last_sale_shop_item = {}

    item_since_list = np.full(len(grid), -1, dtype=np.int16)
    shop_item_since_list = np.full(len(grid), -1, dtype=np.int16)

    grid["_row_idx"] = np.arange(len(grid))
    grid["_shop_item_key"] = grid["shop_id"].astype(np.int64) * 100000 + grid["item_id"].astype(np.int64)

    for db in range(35):
        db_mask = grid["date_block_num"] == db
        if not db_mask.any():
            continue
        sub_grid = grid[db_mask]

        sub_items = sub_grid["item_id"].values
        sub_si_keys = sub_grid["_shop_item_key"].values
        sub_indices = sub_grid["_row_idx"].values

        item_deltas = np.array([db - last_sale_item[item] if item in last_sale_item else -1 for item in sub_items], dtype=np.int16)
        si_deltas = np.array([db - last_sale_shop_item[si] if si in last_sale_shop_item else -1 for si in sub_si_keys], dtype=np.int16)

        item_since_list[sub_indices] = item_deltas
        shop_item_since_list[sub_indices] = si_deltas

        for item in item_sales_map.get(db, []):
            last_sale_item[item] = db
        for si in shop_item_map.get(db, []):
            last_sale_shop_item[si] = db

    grid["months_since_last_item_sale"] = item_since_list
    grid["months_since_last_shop_item_sale"] = shop_item_since_list
    grid.drop(columns=["_row_idx", "_shop_item_key"], inplace=True)

    grid["month"] = (grid["date_block_num"] % 12).astype(np.int8)
    days_map = {
        0: 31,
        1: 28,
        2: 31,
        3: 30,
        4: 31,
        5: 30,
        6: 31,
        7: 31,
        8: 30,
        9: 31,
        10: 30,
        11: 31,
    }
    grid["days_in_month"] = grid["month"].map(days_map).astype(np.int8)

    grid = grid[grid["date_block_num"] >= 12].copy()

    exclude_cols = [
        "item_cnt_month",
        "date_block_num",
        test_id_col,
    ]
    feature_cols = [c for c in grid.columns if c not in exclude_cols and c is not None]

    train_mask = (grid["date_block_num"] >= 12) & (grid["date_block_num"] <= 32)
    val_mask = grid["date_block_num"] == 33
    test_mask = grid["date_block_num"] == 34

    train_df = grid[train_mask].reset_index(drop=True)
    val_df = grid[val_mask].reset_index(drop=True)
    test_df_processed = grid[test_mask].reset_index(drop=True)

    if test_id_col and test_id_col in test_df_processed.columns:
        test_df_processed.sort_values(test_id_col, inplace=True)
        test_df_processed.reset_index(drop=True, inplace=True)

    X_train, y_train = train_df[feature_cols], train_df["item_cnt_month"]
    X_val, y_val = val_df[feature_cols], val_df["item_cnt_month"]
    X_test = test_df_processed[feature_cols]

    del train_sales, grid, train_df, val_df
    gc.collect()

    print("=== Step 2 & 3: Model Training and Evaluation ===")
    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

    lgb_params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "n_estimators": 1500,
        "learning_rate": 0.03,
        "num_leaves": 63,
        "max_depth": 8,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "min_data_in_leaf": 50,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }

    model_lgb = lgb.train(
        lgb_params,
        dtrain,
        num_boost_round=1500,
        valid_sets=[dtrain, dval],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )

    val_preds_lgb = model_lgb.predict(X_val, num_iteration=model_lgb.best_iteration)
    val_preds_lgb = np.clip(val_preds_lgb, 0, 20)

    test_preds_lgb = model_lgb.predict(X_test, num_iteration=model_lgb.best_iteration)
    test_preds_lgb = np.clip(test_preds_lgb, 0, 20)

    lgb_val_rmse = np.sqrt(np.mean((y_val.values - val_preds_lgb) ** 2))
    print(f"LightGBM Validation RMSE: {lgb_val_rmse:.5f}")

    lgb_tweedie_params = {
        "objective": "tweedie",
        "tweedie_variance_power": 1.1,
        "metric": "rmse",
        "boosting_type": "gbdt",
        "n_estimators": 1500,
        "learning_rate": 0.03,
        "num_leaves": 63,
        "max_depth": 8,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "min_data_in_leaf": 50,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }

    model_lgb_tw = lgb.train(
        lgb_tweedie_params,
        dtrain,
        num_boost_round=1500,
        valid_sets=[dtrain, dval],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )

    val_preds_lgb_tw = model_lgb_tw.predict(X_val, num_iteration=model_lgb_tw.best_iteration)
    val_preds_lgb_tw = np.clip(val_preds_lgb_tw, 0, 20)

    test_preds_lgb_tw = model_lgb_tw.predict(X_test, num_iteration=model_lgb_tw.best_iteration)
    test_preds_lgb_tw = np.clip(test_preds_lgb_tw, 0, 20)

    lgb_tw_val_rmse = np.sqrt(np.mean((y_val.values - val_preds_lgb_tw) ** 2))
    print(f"LightGBM Tweedie Validation RMSE: {lgb_tw_val_rmse:.5f}")

    model_cb = cb.CatBoostRegressor(
        iterations=1500,
        learning_rate=0.03,
        depth=7,
        loss_function="Tweedie:variance_power=1.1",
        eval_metric="RMSE",
        random_seed=42,
        verbose=0,
        thread_count=-1,
    )

    model_cb.fit(
        X_train,
        y_train,
        eval_set=(X_val, y_val),
        early_stopping_rounds=50,
        verbose=False,
    )

    val_preds_cb = model_cb.predict(X_val)
    val_preds_cb = np.clip(val_preds_cb, 0, 20)

    test_preds_cb = model_cb.predict(X_test)
    test_preds_cb = np.clip(test_preds_cb, 0, 20)

    cb_val_rmse = np.sqrt(np.mean((y_val.values - val_preds_cb) ** 2))
    print(f"CatBoost Validation RMSE: {cb_val_rmse:.5f}")

    model_xgb = xgb.XGBRegressor(
        n_estimators=1500,
        learning_rate=0.03,
        max_depth=7,
        objective="reg:tweedie",
        tweedie_variance_power=1.1,
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=50,
        eval_metric="rmse",
    )

    model_xgb.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    val_preds_xgb = model_xgb.predict(X_val)
    val_preds_xgb = np.clip(val_preds_xgb, 0, 20)

    test_preds_xgb = model_xgb.predict(X_test)
    test_preds_xgb = np.clip(test_preds_xgb, 0, 20)

    xgb_val_rmse = np.sqrt(np.mean((y_val.values - val_preds_xgb) ** 2))
    print(f"XGBoost Validation RMSE: {xgb_val_rmse:.5f}")

    y_val_arr = y_val.values

    val_matrix = np.column_stack([val_preds_lgb, val_preds_lgb_tw, val_preds_cb, val_preds_xgb])
    test_matrix = np.column_stack([test_preds_lgb, test_preds_lgb_tw, test_preds_cb, test_preds_xgb])

    weights, _ = nnls(val_matrix, y_val_arr)
    if weights.sum() > 0:
        weights = weights / weights.sum()

    w1, w2, w3, w4 = weights
    print(
        f"Optimal blend weights (LGB RMSE, LGB Tweedie, CB Tweedie, XGB Tweedie): {w1:.4f}, {w2:.4f}, {w3:.4f}, {w4:.4f}"
    )

    final_val_preds = np.clip(val_matrix @ weights, 0, 20)
    final_test_preds = np.clip(test_matrix @ weights, 0, 20)

    final_val_rmse = np.sqrt(np.mean((y_val_arr - final_val_preds) ** 2))

    if test_id_col is not None and test_id_col in test_df_processed.columns:
        sub_ids = test_df_processed[test_id_col].astype(int)
    else:
        sub_ids = np.arange(len(final_test_preds), dtype=int)

    submission_df = pd.DataFrame({"id": sub_ids, "item_cnt_day": final_test_preds})
    sub_path = os.path.join(submission_dir, "submission.csv")
    submission_df.to_csv(sub_path, index=False)

    print(f"Final Validation Score: {final_val_rmse:.5f}")


if __name__ == "__main__":
    run_pipeline()
