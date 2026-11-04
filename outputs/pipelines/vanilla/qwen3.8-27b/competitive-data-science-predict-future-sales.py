import pandas as pd
import numpy as np
import lightgbm as lgb

TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/kaggle/competitive-data-science-predict-future-sales/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/kaggle/competitive-data-science-predict-future-sales/data/test.parquet"
TARGET = "item_cnt_day"
BLOCK_COL = "date_block_num"
ID_COLS = ["shop_id", "item_id"]
SMOOTHING = 20.0


def clean_frame(df):
    df = df.copy()

    if TARGET in df.columns:
        df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce").fillna(0.0)

    for col in ID_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(-1).astype("int64")
        else:
            df[col] = -1

    if "item_price" in df.columns:
        df["item_price"] = pd.to_numeric(df["item_price"], errors="coerce").fillna(0.0)
    else:
        df["item_price"] = 0.0

    if BLOCK_COL in df.columns:
        df[BLOCK_COL] = pd.to_numeric(df[BLOCK_COL], errors="coerce").fillna(0).astype("int64")
    else:
        df[BLOCK_COL] = 0

    df["shop_item_key"] = df["shop_id"].astype("int64") * 1_000_000 + df["item_id"].astype("int64")
    return df


def add_time_features(train, test):
    for col in ["shop_enc", "item_enc", "shop_item_enc", "shop_item_last", "global_mean"]:
        train[col] = 0.0
        test[col] = 0.0

    shop_sum = {}
    shop_cnt = {}
    item_sum = {}
    item_cnt = {}
    si_sum = {}
    si_cnt = {}
    last_si_sum = {}

    global_sum = 0.0
    global_cnt = 0
    global_mean = 0.0

    for _, block_df in train.groupby(BLOCK_COL, sort=True):
        idx = block_df.index
        shop_keys = block_df["shop_id"].to_numpy()
        item_keys = block_df["item_id"].to_numpy()
        si_keys = block_df["shop_item_key"].to_numpy()

        shop_mean_dict = {
            k: (shop_sum[k] + SMOOTHING * global_mean) / (shop_cnt[k] + SMOOTHING)
            for k in shop_sum
        }
        item_mean_dict = {
            k: (item_sum[k] + SMOOTHING * global_mean) / (item_cnt[k] + SMOOTHING)
            for k in item_sum
        }
        si_mean_dict = {
            k: (si_sum[k] + SMOOTHING * global_mean) / (si_cnt[k] + SMOOTHING)
            for k in si_sum
        }

        train.loc[idx, "shop_enc"] = pd.Series(shop_keys).map(shop_mean_dict).fillna(global_mean).to_numpy()
        train.loc[idx, "item_enc"] = pd.Series(item_keys).map(item_mean_dict).fillna(global_mean).to_numpy()
        train.loc[idx, "shop_item_enc"] = pd.Series(si_keys).map(si_mean_dict).fillna(global_mean).to_numpy()
        train.loc[idx, "shop_item_last"] = pd.Series(si_keys).map(last_si_sum).fillna(0.0).to_numpy()
        train.loc[idx, "global_mean"] = global_mean

        shop_agg = block_df.groupby("shop_id")[TARGET].agg(["sum", "count"])
        for k, v in shop_agg["sum"].to_dict().items():
            shop_sum[k] = shop_sum.get(k, 0.0) + float(v)
        for k, v in shop_agg["count"].to_dict().items():
            shop_cnt[k] = shop_cnt.get(k, 0) + int(v)

        item_agg = block_df.groupby("item_id")[TARGET].agg(["sum", "count"])
        for k, v in item_agg["sum"].to_dict().items():
            item_sum[k] = item_sum.get(k, 0.0) + float(v)
        for k, v in item_agg["count"].to_dict().items():
            item_cnt[k] = item_cnt.get(k, 0) + int(v)

        si_agg = block_df.groupby("shop_item_key")[TARGET].agg(["sum", "count"])
        for k, v in si_agg["sum"].to_dict().items():
            si_sum[k] = si_sum.get(k, 0.0) + float(v)
        for k, v in si_agg["count"].to_dict().items():
            si_cnt[k] = si_cnt.get(k, 0) + int(v)

        last_si_sum = si_agg["sum"].to_dict()

        global_sum += float(block_df[TARGET].sum())
        global_cnt += int(len(block_df))
        global_mean = global_sum / global_cnt if global_cnt > 0 else 0.0

    shop_mean_dict = {
        k: (shop_sum[k] + SMOOTHING * global_mean) / (shop_cnt[k] + SMOOTHING)
        for k in shop_sum
    }
    item_mean_dict = {
        k: (item_sum[k] + SMOOTHING * global_mean) / (item_cnt[k] + SMOOTHING)
        for k in item_sum
    }
    si_mean_dict = {
        k: (si_sum[k] + SMOOTHING * global_mean) / (si_cnt[k] + SMOOTHING)
        for k in si_sum
    }

    test["shop_enc"] = pd.Series(test["shop_id"].to_numpy()).map(shop_mean_dict).fillna(global_mean).to_numpy()
    test["item_enc"] = pd.Series(test["item_id"].to_numpy()).map(item_mean_dict).fillna(global_mean).to_numpy()
    test["shop_item_enc"] = pd.Series(test["shop_item_key"].to_numpy()).map(si_mean_dict).fillna(global_mean).to_numpy()
    test["shop_item_last"] = pd.Series(test["shop_item_key"].to_numpy()).map(last_si_sum).fillna(0.0).to_numpy()
    test["global_mean"] = global_mean

    return train, test


def main():
    train = pd.read_parquet(TRAIN_PATH)
    test = pd.read_parquet(TEST_PATH)

    train = clean_frame(train)
    test = clean_frame(test)

    train, test = add_time_features(train, test)

    feature_cols = [
        BLOCK_COL,
        "item_price",
        "shop_enc",
        "item_enc",
        "shop_item_enc",
        "shop_item_last",
        "global_mean",
    ]
    feature_cols = [col for col in feature_cols if col in train.columns and col in test.columns]

    X_all = train[feature_cols].copy()
    y_all = train[TARGET].copy()

    if BLOCK_COL in train.columns and train[BLOCK_COL].nunique() > 1:
        val_block = train[BLOCK_COL].max()
        val_mask = train[BLOCK_COL] == val_block
        if val_mask.sum() > 0 and (~val_mask).sum() > 0:
            X_train = X_all.loc[~val_mask].copy()
            y_train = y_all.loc[~val_mask].copy()
            X_val = X_all.loc[val_mask].copy()
            y_val = y_all.loc[val_mask].copy()
        else:
            from sklearn.model_selection import train_test_split

            X_train, X_val, y_train, y_val = train_test_split(
                X_all, y_all, test_size=0.2, random_state=42
            )
    else:
        from sklearn.model_selection import train_test_split

        X_train, X_val, y_train, y_val = train_test_split(
            X_all, y_all, test_size=0.2, random_state=42
        )

    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=2000,
        learning_rate=0.05,
        num_leaves=127,
        max_depth=-1,
        min_child_samples=100,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )

    try:
        callbacks = [lgb.early_stopping(stopping_rounds=100, verbose=False)]
    except AttributeError:
        callbacks = []

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="rmse",
        callbacks=callbacks,
    )

    best_iter = getattr(model, "best_iteration_", None)
    if best_iter is None or best_iter <= 0:
        best_iter = model.n_estimators
    else:
        best_iter = int(best_iter)

    final_model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=best_iter,
        learning_rate=0.05,
        num_leaves=127,
        max_depth=-1,
        min_child_samples=100,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    final_model.fit(X_all, y_all)

    X_test = test[feature_cols].copy()
    preds = final_model.predict(X_test)
    preds = np.clip(preds, 0.0, None)

    submission = pd.DataFrame(
        {
            "id": np.arange(len(preds), dtype="int64"),
            "item_cnt_day": preds.astype("float64"),
        }
    )
    submission.to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    main()