import gc
import os
from itertools import product
from catboost import CatBoostRegressor
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb


def load_and_preprocess_data():
    os.makedirs("./working", exist_ok=True)

    train_path = "./input/train.parquet"
    test_path = "./input/test.parquet"

    train = pd.read_parquet(train_path)
    test = pd.read_parquet(test_path)

    if "ID" in test.columns:
        test = test.rename(columns={"ID": "id"})
    elif "id" not in test.columns:
        test["id"] = np.arange(len(test))

    items = None
    shops = None
    cats = None

    for prefix in ["./input/", "./input/original/"]:
        if items is None:
            if os.path.exists(f"{prefix}items.parquet"):
                items = pd.read_parquet(f"{prefix}items.parquet")
            elif os.path.exists(f"{prefix}items.csv"):
                items = pd.read_csv(f"{prefix}items.csv")
        if shops is None:
            if os.path.exists(f"{prefix}shops.parquet"):
                shops = pd.read_parquet(f"{prefix}shops.parquet")
            elif os.path.exists(f"{prefix}shops.csv"):
                shops = pd.read_csv(f"{prefix}shops.csv")
        if cats is None:
            if os.path.exists(f"{prefix}item_categories.parquet"):
                cats = pd.read_parquet(f"{prefix}item_categories.parquet")
            elif os.path.exists(f"{prefix}item_categories.csv"):
                cats = pd.read_csv(f"{prefix}item_categories.csv")

    train = train[train["item_price"] > 0]
    train = train[train["item_price"] < 100000]
    train = train[train["item_cnt_day"] > 0]
    train = train[train["item_cnt_day"] < 1000]

    shop_mapping = {0: 57, 1: 58, 10: 11, 39: 40}
    train["shop_id"] = train["shop_id"].replace(shop_mapping)
    test["shop_id"] = test["shop_id"].replace(shop_mapping)

    if shops is not None and "shop_name" in shops.columns:
        shops["shop_id"] = shops["shop_id"].replace(shop_mapping)
        shops = shops.drop_duplicates(subset=["shop_id"])
        shops["city"] = shops["shop_name"].apply(lambda x: x.split()[0].strip("!"))
        shops["city_code"] = LabelEncoder().fit_transform(shops["city"]).astype(np.int8)
        shops = shops[["shop_id", "city_code"]]
    else:
        unique_shops = np.unique(
            np.concatenate([train["shop_id"].unique(), test["shop_id"].unique()])
        )
        shops = pd.DataFrame(
            {
                "shop_id": unique_shops,
                "city_code": np.zeros(len(unique_shops), dtype=np.int8),
            }
        )

    if cats is not None and "item_category_name" in cats.columns:
        cats["split"] = cats["item_category_name"].apply(lambda x: x.split("-"))
        cats["type"] = cats["split"].apply(lambda x: x[0].strip())
        cats["subtype"] = cats["split"].apply(
            lambda x: x[1].strip() if len(x) > 1 else x[0].strip()
        )
        cats["type_code"] = LabelEncoder().fit_transform(cats["type"]).astype(np.int8)
        cats["subtype_code"] = (
            LabelEncoder().fit_transform(cats["subtype"]).astype(np.int8)
        )
        cats = cats[["item_category_id", "type_code", "subtype_code"]]
    else:
        cats = pd.DataFrame(
            {"item_category_id": np.arange(100), "type_code": 0, "subtype_code": 0}
        )

    svd_cols = [f"item_name_svd_{i}" for i in range(8)]
    if items is not None:
        if "item_name" in items.columns:
            items["item_name_len"] = (
                items["item_name"].astype(str).str.len().astype(np.int16)
            )
            items["item_name_words"] = (
                items["item_name"].astype(str).str.split().str.len().astype(np.int8)
            )
            tfidf = TfidfVectorizer(max_features=1000, token_pattern=r"(?u)\b\w+\b")
            tfidf_mat = tfidf.fit_transform(items["item_name"].astype(str))
            svd = TruncatedSVD(n_components=8, random_state=42)
            svd_mat = svd.fit_transform(tfidf_mat).astype(np.float32)
            for idx, col in enumerate(svd_cols):
                items[col] = svd_mat[:, idx]
        else:
            items["item_name_len"] = np.int16(0)
            items["item_name_words"] = np.int8(0)
            for col in svd_cols:
                items[col] = np.float32(0.0)
        if "item_category_id" not in items.columns:
            items["item_category_id"] = 0
        keep_cols = ["item_id", "item_category_id", "item_name_len", "item_name_words"] + svd_cols
        items = items[keep_cols].drop_duplicates(subset=["item_id"])
    else:
        unique_items = np.unique(
            np.concatenate([train["item_id"].unique(), test["item_id"].unique()])
        )
        items_dict = {
            "item_id": unique_items,
            "item_category_id": 0,
            "item_name_len": 0,
            "item_name_words": 0,
        }
        for col in svd_cols:
            items_dict[col] = np.float32(0.0)
        items = pd.DataFrame(items_dict)

    items = pd.merge(items, cats, on="item_category_id", how="left").fillna(0)
    cat_le = LabelEncoder()
    items["item_category_id"] = cat_le.fit_transform(items["item_category_id"]).astype(np.int16)
    items["type_code"] = items["type_code"].astype(np.int8)
    items["subtype_code"] = items["subtype_code"].astype(np.int8)
    items["item_name_len"] = items["item_name_len"].astype(np.int16)
    items["item_name_words"] = items["item_name_words"].astype(np.int8)
    for col in svd_cols:
        items[col] = items[col].astype(np.float32)

    grid = []
    for block in range(34):
        cur_train = train[train["date_block_num"] == block]
        cur_shops = cur_train["shop_id"].unique()
        cur_items = cur_train["item_id"].unique()
        grid.append(
            np.array(list(product([block], cur_shops, cur_items)), dtype=np.int32)
        )

    grid = pd.DataFrame(
        np.vstack(grid), columns=["date_block_num", "shop_id", "item_id"]
    )
    grid["date_block_num"] = grid["date_block_num"].astype(np.int8)
    grid["shop_id"] = grid["shop_id"].astype(np.int8)
    grid["item_id"] = grid["item_id"].astype(np.int32)

    monthly_sales = (
        train.groupby(["date_block_num", "shop_id", "item_id"])["item_cnt_day"]
        .sum()
        .reset_index()
    )
    monthly_sales.columns = ["date_block_num", "shop_id", "item_id", "item_cnt_month"]
    monthly_sales["item_cnt_month"] = (
        monthly_sales["item_cnt_month"].clip(0, 20).astype(np.float32)
    )

    grid = pd.merge(
        grid, monthly_sales, on=["date_block_num", "shop_id", "item_id"], how="left"
    )
    grid["item_cnt_month"] = grid["item_cnt_month"].fillna(0).astype(np.float32)

    test_block = test[["shop_id", "item_id"]].copy()
    test_block["date_block_num"] = np.int8(34)
    test_block["shop_id"] = test_block["shop_id"].astype(np.int8)
    test_block["item_id"] = test_block["item_id"].astype(np.int32)
    test_block["item_cnt_month"] = np.float32(0.0)

    grid = pd.concat([grid, test_block], ignore_index=True)
    grid = pd.merge(grid, shops, on="shop_id", how="left")
    grid = pd.merge(grid, items, on="item_id", how="left")
    grid["city_code"] = grid["city_code"].fillna(0).astype(np.int8)
    grid["item_category_id"] = grid["item_category_id"].fillna(0).astype(np.int16)
    grid["type_code"] = grid["type_code"].fillna(0).astype(np.int8)
    grid["subtype_code"] = grid["subtype_code"].fillna(0).astype(np.int8)
    grid["item_name_len"] = grid["item_name_len"].fillna(0).astype(np.int16)
    grid["item_name_words"] = grid["item_name_words"].fillna(0).astype(np.int8)
    for col in svd_cols:
        grid[col] = grid[col].fillna(0.0).astype(np.float32)

    grid = grid.sort_values(["date_block_num", "shop_id", "item_id"]).reset_index(
        drop=True
    )
    last_sale_shop_item = {}
    last_sale_item = {}
    last_qty_shop_item = {}
    last_qty_item = {}
    months_since_shop_item = []
    months_since_item = []
    qty_at_last_shop_item = []
    qty_at_last_item = []

    for block in range(35):
        block_mask = grid["date_block_num"] == block
        sub = grid.loc[block_mask, ["shop_id", "item_id", "item_cnt_month"]]
        s_arr = sub["shop_id"].values
        i_arr = sub["item_id"].values
        c_arr = sub["item_cnt_month"].values
        n_rows = len(sub)
        delta_si = np.full(n_rows, fill_value=-1, dtype=np.int8)
        delta_i = np.full(n_rows, fill_value=-1, dtype=np.int8)
        q_si = np.zeros(n_rows, dtype=np.float32)
        q_i = np.zeros(n_rows, dtype=np.float32)

        for idx in range(n_rows):
            s = s_arr[idx]
            i = i_arr[idx]
            si_key = (s, i)
            if si_key in last_sale_shop_item:
                delta_si[idx] = block - last_sale_shop_item[si_key]
                q_si[idx] = last_qty_shop_item[si_key]
            if i in last_sale_item:
                delta_i[idx] = block - last_sale_item[i]
                q_i[idx] = last_qty_item[i]
            if c_arr[idx] > 0:
                last_sale_shop_item[si_key] = block
                last_sale_item[i] = block
                last_qty_shop_item[si_key] = c_arr[idx]
                last_qty_item[i] = c_arr[idx]

        months_since_shop_item.append(delta_si)
        months_since_item.append(delta_i)
        qty_at_last_shop_item.append(q_si)
        qty_at_last_item.append(q_i)

    grid["months_since_shop_item_sale"] = np.concatenate(
        months_since_shop_item
    ).astype(np.int8)
    grid["months_since_item_sale"] = np.concatenate(months_since_item).astype(np.int8)
    grid["last_sale_shop_item_qty"] = np.concatenate(qty_at_last_shop_item).astype(
        np.float32
    )
    grid["last_sale_item_qty"] = np.concatenate(qty_at_last_item).astype(np.float32)

    def add_lags(df, lag_df, lags, on_cols, col_name):
        for lag in lags:
            temp = lag_df[on_cols + [col_name]].copy()
            temp["date_block_num"] = temp["date_block_num"] + lag
            new_col = f"{col_name}_lag_{lag}"
            temp = temp.rename(columns={col_name: new_col})
            df = pd.merge(
                df,
                temp,
                on=["date_block_num"] + [c for c in on_cols if c != "date_block_num"],
                how="left",
            )
            df[new_col] = df[new_col].fillna(0).astype(np.float32)
        return df

    grid = add_lags(
        grid,
        grid[["date_block_num", "shop_id", "item_id", "item_cnt_month"]],
        [1, 2, 3, 4, 5, 6, 11, 12],
        ["date_block_num", "shop_id", "item_id"],
        "item_cnt_month",
    )

    group = grid.groupby(["date_block_num"])["item_cnt_month"].mean().reset_index()
    group.columns = ["date_block_num", "date_avg_item_cnt"]
    grid = add_lags(grid, group, [1], ["date_block_num"], "date_avg_item_cnt")

    group = (
        grid.groupby(["date_block_num", "item_id"])["item_cnt_month"]
        .mean()
        .reset_index()
    )
    group.columns = ["date_block_num", "item_id", "date_item_avg_item_cnt"]
    grid = add_lags(
        grid,
        group,
        [1, 2, 3, 6],
        ["date_block_num", "item_id"],
        "date_item_avg_item_cnt",
    )

    group = (
        grid.groupby(["date_block_num", "shop_id"])["item_cnt_month"]
        .mean()
        .reset_index()
    )
    group.columns = ["date_block_num", "shop_id", "date_shop_avg_item_cnt"]
    grid = add_lags(
        grid,
        group,
        [1, 2, 3, 6],
        ["date_block_num", "shop_id"],
        "date_shop_avg_item_cnt",
    )

    group = (
        grid.groupby(["date_block_num", "item_category_id"])["item_cnt_month"]
        .mean()
        .reset_index()
    )
    group.columns = ["date_block_num", "item_category_id", "date_cat_avg_item_cnt"]
    grid = add_lags(
        grid,
        group,
        [1, 2],
        ["date_block_num", "item_category_id"],
        "date_cat_avg_item_cnt",
    )

    group = (
        grid.groupby(["date_block_num", "shop_id", "item_category_id"])[
            "item_cnt_month"
        ]
        .mean()
        .reset_index()
    )
    group.columns = [
        "date_block_num",
        "shop_id",
        "item_category_id",
        "date_shop_cat_avg_item_cnt",
    ]
    grid = add_lags(
        grid,
        group,
        [1, 2],
        ["date_block_num", "shop_id", "item_category_id"],
        "date_shop_cat_avg_item_cnt",
    )

    group = (
        grid.groupby(["date_block_num", "city_code"])["item_cnt_month"]
        .mean()
        .reset_index()
    )
    group.columns = ["date_block_num", "city_code", "date_city_avg_item_cnt"]
    grid = add_lags(
        grid, group, [1], ["date_block_num", "city_code"], "date_city_avg_item_cnt"
    )

    group = (
        grid.groupby(["date_block_num", "subtype_code"])["item_cnt_month"]
        .mean()
        .reset_index()
    )
    group.columns = ["date_block_num", "subtype_code", "date_subtype_avg_item_cnt"]
    grid = add_lags(
        grid,
        group,
        [1, 2],
        ["date_block_num", "subtype_code"],
        "date_subtype_avg_item_cnt",
    )

    group = (
        grid.groupby(["date_block_num", "shop_id", "subtype_code"])["item_cnt_month"]
        .mean()
        .reset_index()
    )
    group.columns = [
        "date_block_num",
        "shop_id",
        "subtype_code",
        "date_shop_subtype_avg_item_cnt",
    ]
    grid = add_lags(
        grid,
        group,
        [1, 2],
        ["date_block_num", "shop_id", "subtype_code"],
        "date_shop_subtype_avg_item_cnt",
    )

    group = (
        grid.groupby(["date_block_num", "city_code", "item_id"])["item_cnt_month"]
        .mean()
        .reset_index()
    )
    group.columns = [
        "date_block_num",
        "city_code",
        "item_id",
        "date_city_item_avg_item_cnt",
    ]
    grid = add_lags(
        grid,
        group,
        [1],
        ["date_block_num", "city_code", "item_id"],
        "date_city_item_avg_item_cnt",
    )

    group = (
        grid.groupby(["date_block_num", "shop_id", "type_code"])["item_cnt_month"]
        .mean()
        .reset_index()
    )
    group.columns = [
        "date_block_num",
        "shop_id",
        "type_code",
        "date_shop_type_avg_item_cnt",
    ]
    grid = add_lags(
        grid,
        group,
        [1, 2],
        ["date_block_num", "shop_id", "type_code"],
        "date_shop_type_avg_item_cnt",
    )

    group = (
        grid.groupby(["date_block_num", "city_code", "item_category_id"])["item_cnt_month"]
        .mean()
        .reset_index()
    )
    group.columns = [
        "date_block_num",
        "city_code",
        "item_category_id",
        "date_city_cat_avg_item_cnt",
    ]
    grid = add_lags(
        grid,
        group,
        [1],
        ["date_block_num", "city_code", "item_category_id"],
        "date_city_cat_avg_item_cnt",
    )

    grid["item_cnt_month_roll_mean_3"] = (
        (
            grid["item_cnt_month_lag_1"]
            + grid["item_cnt_month_lag_2"]
            + grid["item_cnt_month_lag_3"]
        )
        / 3.0
    ).astype(np.float32)
    grid["item_cnt_month_roll_mean_6"] = (
        (
            grid["item_cnt_month_lag_1"]
            + grid["item_cnt_month_lag_2"]
            + grid["item_cnt_month_lag_3"]
            + grid["item_cnt_month_lag_4"]
            + grid["item_cnt_month_lag_5"]
            + grid["item_cnt_month_lag_6"]
        )
        / 6.0
    ).astype(np.float32)
    grid["item_cnt_month_roll_std_3"] = (
        grid[["item_cnt_month_lag_1", "item_cnt_month_lag_2", "item_cnt_month_lag_3"]]
        .std(axis=1)
        .astype(np.float32)
    )
    grid["item_cnt_month_diff_1_2"] = (
        grid["item_cnt_month_lag_1"] - grid["item_cnt_month_lag_2"]
    ).astype(np.float32)
    grid["item_cnt_month_accel"] = (
        (grid["item_cnt_month_lag_1"] - grid["item_cnt_month_lag_2"])
        - (grid["item_cnt_month_lag_2"] - grid["item_cnt_month_lag_3"])
    ).astype(np.float32)
    grid["item_cnt_month_ratio_roll_3"] = (
        grid["item_cnt_month_lag_1"] / (grid["item_cnt_month_roll_mean_3"] + 1e-3)
    ).astype(np.float32)

    grid["date_item_avg_diff_1_2"] = (
        grid["date_item_avg_item_cnt_lag_1"] - grid["date_item_avg_item_cnt_lag_2"]
    ).astype(np.float32)
    grid["date_shop_avg_diff_1_2"] = (
        grid["date_shop_avg_item_cnt_lag_1"] - grid["date_shop_avg_item_cnt_lag_2"]
    ).astype(np.float32)
    grid["date_cat_avg_diff_1_2"] = (
        grid["date_cat_avg_item_cnt_lag_1"] - grid["date_cat_avg_item_cnt_lag_2"]
    ).astype(np.float32)
    grid["item_cnt_month_diff_11_12"] = (
        grid["item_cnt_month_lag_11"] - grid["item_cnt_month_lag_12"]
    ).astype(np.float32)

    grid["shop_item_share_in_shop"] = (
        grid["item_cnt_month_lag_1"] / (grid["date_shop_avg_item_cnt_lag_1"] + 1e-4)
    ).astype(np.float32)
    grid["shop_item_share_in_item"] = (
        grid["item_cnt_month_lag_1"] / (grid["date_item_avg_item_cnt_lag_1"] + 1e-4)
    ).astype(np.float32)
    grid["active_months_last_3"] = (
        (grid["item_cnt_month_lag_1"] > 0).astype(np.int8)
        + (grid["item_cnt_month_lag_2"] > 0).astype(np.int8)
        + (grid["item_cnt_month_lag_3"] > 0).astype(np.int8)
    ).astype(np.int8)
    grid["active_months_ratio_6"] = (
        (
            (grid["item_cnt_month_lag_1"] > 0).astype(np.float32)
            + (grid["item_cnt_month_lag_2"] > 0).astype(np.float32)
            + (grid["item_cnt_month_lag_3"] > 0).astype(np.float32)
            + (grid["item_cnt_month_lag_4"] > 0).astype(np.float32)
            + (grid["item_cnt_month_lag_5"] > 0).astype(np.float32)
            + (grid["item_cnt_month_lag_6"] > 0).astype(np.float32)
        )
        / 6.0
    ).astype(np.float32)

    item_avg_price = (
        train.groupby("item_id")["item_price"]
        .mean()
        .reset_index()
        .rename(columns={"item_price": "item_avg_price"})
    )
    item_avg_price["log_item_avg_price"] = np.log1p(item_avg_price["item_avg_price"]).astype(np.float32)
    item_cat_map = items[["item_id", "item_category_id"]].drop_duplicates()
    train_with_cat = pd.merge(train[["item_id", "item_price"]], item_cat_map, on="item_id", how="left")
    cat_avg_price = (
        train_with_cat.groupby("item_category_id")["item_price"]
        .mean()
        .reset_index()
        .rename(columns={"item_price": "cat_avg_price"})
    )
    item_avg_price = pd.merge(item_avg_price, item_cat_map, on="item_id", how="left")
    item_avg_price = pd.merge(item_avg_price, cat_avg_price, on="item_category_id", how="left")
    item_avg_price["item_to_cat_price_ratio"] = (
        item_avg_price["item_avg_price"] / (item_avg_price["cat_avg_price"] + 1e-4)
    ).astype(np.float32)

    grid = pd.merge(
        grid,
        item_avg_price[["item_id", "log_item_avg_price", "item_to_cat_price_ratio"]],
        on="item_id",
        how="left",
    )
    grid["log_item_avg_price"] = grid["log_item_avg_price"].fillna(0).astype(np.float32)
    grid["item_to_cat_price_ratio"] = grid["item_to_cat_price_ratio"].fillna(1.0).astype(np.float32)

    item_block_price = (
        train.groupby(["date_block_num", "item_id"])["item_price"]
        .mean()
        .reset_index()
        .rename(columns={"item_price": "date_item_avg_price"})
    )

    price_df = pd.merge(item_block_price, item_avg_price[["item_id", "item_avg_price"]], on="item_id", how="left")
    price_df["delta_price"] = (
        (price_df["date_item_avg_price"] - price_df["item_avg_price"])
        / (price_df["item_avg_price"] + 1e-6)
    ).astype(np.float32)
    grid = add_lags(
        grid,
        price_df[["date_block_num", "item_id", "delta_price"]],
        [1, 2, 3],
        ["date_block_num", "item_id"],
        "delta_price",
    )

    shop_item_block_price = (
        train.groupby(["date_block_num", "shop_id", "item_id"])["item_price"]
        .mean()
        .reset_index()
        .rename(columns={"item_price": "date_shop_item_avg_price"})
    )
    shop_price_df = pd.merge(
        shop_item_block_price, item_avg_price[["item_id", "item_avg_price"]], on="item_id", how="left"
    )
    shop_price_df["delta_price_shop"] = (
        (shop_price_df["date_shop_item_avg_price"] - shop_price_df["item_avg_price"])
        / (shop_price_df["item_avg_price"] + 1e-6)
    ).astype(np.float32)
    grid = add_lags(
        grid,
        shop_price_df[["date_block_num", "shop_id", "item_id", "delta_price_shop"]],
        [1],
        ["date_block_num", "shop_id", "item_id"],
        "delta_price_shop",
    )

    item_first_sale = (
        train.groupby("item_id")["date_block_num"]
        .min()
        .reset_index()
        .rename(columns={"date_block_num": "item_first_sale_block"})
    )
    shop_item_first_sale = (
        train.groupby(["shop_id", "item_id"])["date_block_num"]
        .min()
        .reset_index()
        .rename(columns={"date_block_num": "shop_item_first_sale_block"})
    )

    grid = pd.merge(grid, item_first_sale, on="item_id", how="left")
    grid = pd.merge(grid, shop_item_first_sale, on=["shop_id", "item_id"], how="left")

    grid["item_first_sale_block"] = (
        grid["item_first_sale_block"].fillna(34).astype(np.int8)
    )
    grid["shop_item_first_sale_block"] = (
        grid["shop_item_first_sale_block"].fillna(34).astype(np.int8)
    )

    grid["item_age"] = (grid["date_block_num"] - grid["item_first_sale_block"]).astype(
        np.int8
    )
    grid["shop_item_age"] = (
        grid["date_block_num"] - grid["shop_item_first_sale_block"]
    ).astype(np.int8)
    grid["is_new_item"] = (grid["item_age"] == 0).astype(np.int8)
    grid["is_new_shop_item"] = (grid["shop_item_age"] == 0).astype(np.int8)

    grid["month"] = (grid["date_block_num"] % 12).astype(np.int8)
    grid["year"] = (2013 + grid["date_block_num"] // 12).astype(np.int16)

    calendar_stats = {}
    for block in range(35):
        y = 2013 + block // 12
        m = 1 + block % 12
        dates = pd.date_range(start=f"{y}-{m:02d}-01", periods=pd.Period(f"{y}-{m:02d}").days_in_month)
        calendar_stats[block] = {
            "days_in_month": len(dates),
            "sundays_in_month": int((dates.dayofweek == 6).sum()),
            "weekends_in_month": int((dates.dayofweek >= 5).sum()),
        }
    cal_df = pd.DataFrame.from_dict(calendar_stats, orient="index").reset_index()
    cal_df.columns = ["date_block_num", "days_in_month", "sundays_in_month", "weekends_in_month"]
    cal_df["date_block_num"] = cal_df["date_block_num"].astype(np.int8)
    cal_df["days_in_month"] = cal_df["days_in_month"].astype(np.int8)
    cal_df["sundays_in_month"] = cal_df["sundays_in_month"].astype(np.int8)
    cal_df["weekends_in_month"] = cal_df["weekends_in_month"].astype(np.int8)
    grid = pd.merge(grid, cal_df, on="date_block_num", how="left")

    grid = grid[grid["date_block_num"] >= 12].reset_index(drop=True)

    del train, monthly_sales, price_df, shop_price_df, group, item_avg_price, item_block_price, shop_item_block_price, cal_df, train_with_cat, cat_avg_price
    gc.collect()

    train_data = grid[grid["date_block_num"] < 33].reset_index(drop=True)
    val_data = grid[grid["date_block_num"] == 33].reset_index(drop=True)
    test_grid = grid[grid["date_block_num"] == 34].reset_index(drop=True)

    test_data = pd.merge(
        test[["id", "shop_id", "item_id"]],
        test_grid,
        on=["shop_id", "item_id"],
        how="left",
    )
    test_data = test_data.fillna(0)
    test_data = test_data.sort_values("id").reset_index(drop=True)

    return train_data, val_data, test_data


def get_feature_columns(df):
    exclude_cols = [
        "item_cnt_month",
        "date_block_num",
        "id",
        "item_first_sale_block",
        "shop_item_first_sale_block",
    ]
    features = [col for col in df.columns if col not in exclude_cols]
    categorical_features = [
        col
        for col in [
            "shop_id",
            "item_id",
            "city_code",
            "item_category_id",
            "type_code",
            "subtype_code",
            "month",
            "year",
        ]
        if col in features
    ]
    return features, categorical_features


def compute_rmse(y_true, y_pred):
    y_true_clipped = np.clip(y_true, 0.0, 20.0)
    y_pred_clipped = np.clip(y_pred, 0.0, 20.0)
    return float(root_mean_squared_error(y_true_clipped, y_pred_clipped))


def get_lgb_model_params(seed: int = 42) -> dict:
    return {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "n_estimators": 1500,
        "learning_rate": 0.03,
        "num_leaves": 63,
        "max_depth": 7,
        "min_child_samples": 50,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.75,
        "reg_alpha": 0.1,
        "reg_lambda": 1.5,
        "random_state": seed,
        "n_jobs": -1,
        "verbose": -1,
    }


def get_xgb_model_params(seed: int = 42) -> dict:
    return {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "n_estimators": 1500,
        "learning_rate": 0.03,
        "max_depth": 7,
        "min_child_weight": 40,
        "subsample": 0.8,
        "colsample_bytree": 0.75,
        "reg_alpha": 0.1,
        "reg_lambda": 2.0,
        "random_state": seed,
        "n_jobs": -1,
        "tree_method": "hist",
    }


def get_cb_model_params(seed: int = 42) -> dict:
    return {
        "iterations": 1500,
        "learning_rate": 0.03,
        "depth": 7,
        "rsm": 0.75,
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "random_seed": seed,
        "l2_leaf_reg": 5.0,
        "early_stopping_rounds": 50,
        "verbose": False,
        "thread_count": -1,
    }


def main():
    train_data, val_data, test_data = load_and_preprocess_data()
    features, cat_cols = get_feature_columns(train_data)

    X_train = train_data[features]
    y_train = train_data["item_cnt_month"].clip(0.0, 20.0).values
    X_val = val_data[features]
    y_val = val_data["item_cnt_month"].clip(0.0, 20.0).values
    X_test = test_data[features]

    sample_weights = np.exp(0.04 * (train_data["date_block_num"].values - 32)).astype(np.float32)

    lgb_params_1 = get_lgb_model_params(seed=42)
    lgb_model_1 = lgb.LGBMRegressor(**lgb_params_1)
    lgb_model_1.fit(
        X_train,
        y_train,
        sample_weight=sample_weights,
        eval_set=[(X_val, y_val)],
        categorical_feature=cat_cols,
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    val_preds_lgb1 = np.clip(lgb_model_1.predict(X_val), 0.0, 20.0)
    test_preds_lgb1 = np.clip(lgb_model_1.predict(X_test), 0.0, 20.0)

    lgb_params_2 = get_lgb_model_params(seed=100)
    lgb_params_2.update({"max_depth": 8, "num_leaves": 127, "colsample_bytree": 0.65, "subsample": 0.85})
    lgb_model_2 = lgb.LGBMRegressor(**lgb_params_2)
    lgb_model_2.fit(
        X_train,
        y_train,
        sample_weight=sample_weights,
        eval_set=[(X_val, y_val)],
        categorical_feature=cat_cols,
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    val_preds_lgb2 = np.clip(lgb_model_2.predict(X_val), 0.0, 20.0)
    test_preds_lgb2 = np.clip(lgb_model_2.predict(X_test), 0.0, 20.0)

    xgb_params_1 = get_xgb_model_params(seed=42)
    xgb_model_1 = xgb.XGBRegressor(**xgb_params_1, early_stopping_rounds=50)
    xgb_model_1.fit(
        X_train,
        y_train,
        sample_weight=sample_weights,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    val_preds_xgb1 = np.clip(xgb_model_1.predict(X_val), 0.0, 20.0)
    test_preds_xgb1 = np.clip(xgb_model_1.predict(X_test), 0.0, 20.0)

    xgb_params_2 = get_xgb_model_params(seed=100)
    xgb_params_2.update({"max_depth": 8, "colsample_bytree": 0.65, "subsample": 0.85})
    xgb_model_2 = xgb.XGBRegressor(**xgb_params_2, early_stopping_rounds=50)
    xgb_model_2.fit(
        X_train,
        y_train,
        sample_weight=sample_weights,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    val_preds_xgb2 = np.clip(xgb_model_2.predict(X_val), 0.0, 20.0)
    test_preds_xgb2 = np.clip(xgb_model_2.predict(X_test), 0.0, 20.0)

    cb_params = get_cb_model_params(seed=42)
    cb_model = CatBoostRegressor(**cb_params)
    cb_model.fit(
        X_train,
        y_train,
        sample_weight=sample_weights,
        eval_set=(X_val, y_val),
        cat_features=cat_cols,
        verbose=False,
    )
    val_preds_cb = np.clip(cb_model.predict(X_val), 0.0, 20.0)
    test_preds_cb = np.clip(cb_model.predict(X_test), 0.0, 20.0)

    models_val = [val_preds_lgb1, val_preds_lgb2, val_preds_xgb1, val_preds_xgb2, val_preds_cb]
    models_test = [test_preds_lgb1, test_preds_lgb2, test_preds_xgb1, test_preds_xgb2, test_preds_cb]
    num_models = len(models_val)

    def blend_loss(weights):
        w = np.array(weights)
        w = np.maximum(w, 0.0)
        s = np.sum(w)
        if s > 0:
            w = w / s
        else:
            w = np.ones(num_models) / num_models
        pred = np.zeros_like(y_val)
        for i in range(num_models):
            pred += w[i] * models_val[i]
        return compute_rmse(y_val, pred)

    opt_res = minimize(
        blend_loss,
        [1.0 / num_models] * num_models,
        bounds=[(0.0, 1.0)] * num_models,
        constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        method="SLSQP",
    )
    weights = opt_res.x / np.sum(opt_res.x)

    final_val_preds = np.zeros_like(y_val)
    for i in range(num_models):
        final_val_preds += weights[i] * models_val[i]
    final_val_preds = np.clip(final_val_preds, 0.0, 20.0)
    final_val_score = compute_rmse(y_val, final_val_preds)

    test_preds = np.zeros(len(test_data), dtype=np.float32)
    for i in range(num_models):
        test_preds += weights[i] * models_test[i]
    test_preds = np.clip(test_preds, 0.0, 20.0)

    os.makedirs("./submission", exist_ok=True)
    submission = pd.DataFrame(
        {
            "id": test_data["id"].astype(int).values,
            "item_cnt_day": test_preds.astype(np.float32),
        }
    )
    submission = submission.sort_values("id").reset_index(drop=True)
    submission.to_csv("./submission/submission.csv", index=False)

    print(f"Final Validation Score: {final_val_score}")


if __name__ == "__main__":
    main()
