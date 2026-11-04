import gc
import json
import os
from catboost import CatBoostRegressor
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
import xgboost as xgb


def compute_microstructure_features(df: pd.DataFrame) -> pd.DataFrame:
    """Computes instant order book, spread, pairwise price/size triplet imbalance ratios."""
    df = df.copy()
    eps = 1e-8

    df["mid_price"] = (df["ask_price"] + df["bid_price"]) / 2.0
    df["spread"] = df["ask_price"] - df["bid_price"]
    df["relative_spread"] = df["spread"] / (df["wap"] + eps)
    df["wap_mid_diff"] = df["wap"] - df["mid_price"]
    df["price_spread_imb"] = (df["ask_price"] - df["bid_price"]) / (df["ask_price"] + df["bid_price"] + eps)

    # Size and volume imbalances
    df["total_volume"] = df["bid_size"] + df["ask_size"]
    df["volume_imbalance"] = (df["bid_size"] - df["ask_size"]) / (df["total_volume"] + eps)
    df["matched_imbalance_ratio"] = df["matched_size"] / (df["imbalance_size"] + eps)
    df["matched_total_ratio"] = df["matched_size"] / (df["total_volume"] + eps)
    df["imb_matched_diff_ratio"] = (df["imbalance_size"] - df["matched_size"]) / (
        df["imbalance_size"] + df["matched_size"] + eps
    )
    df["imb_bid_diff_ratio"] = (df["imbalance_size"] - df["bid_size"]) / (
        df["imbalance_size"] + df["bid_size"] + eps
    )
    df["imb_ask_diff_ratio"] = (df["imbalance_size"] - df["ask_size"]) / (
        df["imbalance_size"] + df["ask_size"] + eps
    )

    # Signed price pressure from auction imbalance
    df["signed_imbalance"] = df["imbalance_buy_sell_flag"] * df["imbalance_size"]
    df["pressure_to_matched"] = df["signed_imbalance"] / (df["matched_size"] + eps)
    df["pressure_to_vol"] = df["signed_imbalance"] / (df["total_volume"] + eps)

    # Pairwise price differentials & triplet imbalance ratios across prices
    prices = ["wap", "reference_price", "bid_price", "ask_price", "near_price", "far_price"]
    for i in range(len(prices)):
        for j in range(i + 1, len(prices)):
            p1, p2 = prices[i], prices[j]
            diff_col = f"{p1}_{p2}_diff"
            imb_col = f"{p1}_{p2}_imb"
            df[diff_col] = df[p1] - df[p2]
            df[imb_col] = (df[p1] - df[p2]) / (df[p1] + df[p2] + eps)

    # Auction book dynamics and crossing signals
    df["is_auction_phase"] = (df["seconds_in_bucket"] >= 300).astype(np.int32)
    df["near_ref_spread_ratio"] = (df["near_price"] - df["reference_price"]) / (df["spread"] + eps)
    df["far_ref_spread_ratio"] = (df["far_price"] - df["reference_price"]) / (df["spread"] + eps)

    # Triplet price interactions
    df["wap_spread_ratio"] = df["wap"] / (df["spread"] + eps)
    df["ask_bid_spread_sum"] = (df["ask_price"] + df["bid_price"]) / (df["spread"] + eps)

    # Cyclical seconds in bucket
    df["seconds_norm"] = df["seconds_in_bucket"] / 540.0
    df["sin_seconds"] = np.sin(2 * np.pi * df["seconds_norm"])
    df["cos_seconds"] = np.cos(2 * np.pi * df["seconds_norm"])

    return df


def compute_cross_sectional_features(df: pd.DataFrame) -> pd.DataFrame:
    """Computes cross-sectional market-wide statistics at each time snapshot."""
    eps = 1e-8
    time_group = df.groupby(["date_id", "seconds_in_bucket"])

    market_wap_mean = time_group["wap"].transform("mean")
    market_wap_std = time_group["wap"].transform("std").fillna(0)
    market_vol_sum = time_group["total_volume"].transform("sum")
    market_imb_sum = time_group["signed_imbalance"].transform("sum")
    market_spread_mean = time_group["spread"].transform("mean")

    # Volume-weighted market index calculation
    df["wap_vol"] = df["wap"] * df["total_volume"]
    market_wap_vol_sum = df.groupby(["date_id", "seconds_in_bucket"])["wap_vol"].transform("sum")
    market_vwap = market_wap_vol_sum / (market_vol_sum + eps)
    df["market_vwap"] = market_vwap
    df = df.drop(columns=["wap_vol"])

    df["market_wap_mean"] = market_wap_mean
    df["market_wap_std"] = market_wap_std
    df["stock_wap_rel_market"] = df["wap"] - market_wap_mean
    df["stock_vol_rel_market"] = df["total_volume"] / (market_vol_sum + eps)
    df["stock_spread_rel_market"] = df["spread"] - market_spread_mean
    df["market_imb_sum"] = market_imb_sum

    return df


def compute_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Computes multi-horizon lag, return, diff, acceleration, and rolling momentum features."""
    df = df.copy()
    orig_idx_name = "__orig_order_idx__"
    df[orig_idx_name] = np.arange(len(df))

    # Sort strictly by stock_id, date_id, seconds_in_bucket for sequential time operations
    df = df.sort_values(by=["stock_id", "date_id", "seconds_in_bucket"]).reset_index(
        drop=True
    )
    stock_date_group = df.groupby(["stock_id", "date_id"], sort=False)

    # Multi-horizon temporal returns, market-relative percentage returns, and differentials
    for lag in [1, 2, 3, 4, 6, 10, 12]:
        wap_shift = stock_date_group["wap"].shift(lag)
        df[f"wap_diff_{lag}"] = (df["wap"] - wap_shift).fillna(0)
        df[f"wap_ret_{lag}"] = ((df["wap"] - wap_shift) / (wap_shift + 1e-8)).fillna(0) * 10000.0

        # Historical market-relative percentage returns: stock return minus volume-weighted index return
        if "market_vwap" in df.columns:
            market_vwap_shift = stock_date_group["market_vwap"].shift(lag)
            market_vwap_ret = (
                (df["market_vwap"] - market_vwap_shift) / (market_vwap_shift + 1e-8)
            ).fillna(0) * 10000.0
            df[f"market_rel_wap_ret_{lag}"] = df[f"wap_ret_{lag}"] - market_vwap_ret

        df[f"vol_diff_{lag}"] = stock_date_group["total_volume"].diff(lag).fillna(0)
        df[f"imb_diff_{lag}"] = stock_date_group["signed_imbalance"].diff(lag).fillna(0)
        df[f"spread_diff_{lag}"] = stock_date_group["spread"].diff(lag).fillna(0)

    # Market-relative return momentum at horizon 6 (60-second window)
    if "stock_wap_rel_market" in df.columns:
        df["market_rel_wap_diff_6"] = stock_date_group["stock_wap_rel_market"].diff(6).fillna(0)

    # Acceleration (diff of diffs)
    df["wap_accel_1"] = stock_date_group["wap_diff_1"].diff(1).fillna(0)
    df["wap_accel_6"] = stock_date_group["wap_diff_6"].diff(6).fillna(0)

    # Rolling price volatility and volume momentum
    df["wap_rolling_std_3"] = (
        stock_date_group["wap"].rolling(3, min_periods=1).std().fillna(0).to_numpy()
    )
    df["wap_rolling_std_6"] = (
        stock_date_group["wap"].rolling(6, min_periods=1).std().fillna(0).to_numpy()
    )
    df["wap_rolling_std_12"] = (
        stock_date_group["wap"].rolling(12, min_periods=1).std().fillna(0).to_numpy()
    )
    df["vol_rolling_mean_6"] = (
        stock_date_group["total_volume"]
        .rolling(6, min_periods=1)
        .mean()
        .fillna(0)
        .to_numpy()
    )
    df["imb_rolling_mean_6"] = (
        stock_date_group["signed_imbalance"]
        .rolling(6, min_periods=1)
        .mean()
        .fillna(0)
        .to_numpy()
    )

    # Restore initial dataset ordering
    df = (
        df.sort_values(by=orig_idx_name)
        .reset_index(drop=True)
        .drop(columns=[orig_idx_name])
    )
    return df


def align_snapshot_predictions(
    df: pd.DataFrame, preds: np.ndarray, time_cols: list
) -> np.ndarray:
    """Applies cross-sectional zero-centering per snapshot if timestamp columns are present."""
    available_cols = [col for col in time_cols if col in df.columns]
    if not available_cols:
        return preds

    temp_df = pd.DataFrame(
        {
            "pred": preds,
            "group": df[available_cols].astype(str).agg("_".join, axis=1),
        }
    )
    group_means = temp_df.groupby("group")["pred"].transform("mean")
    adjusted_preds = (temp_df["pred"] - group_means).to_numpy()
    return adjusted_preds


def main():
    os.makedirs("./submission", exist_ok=True)
    os.makedirs("./working", exist_ok=True)

    print("Loading raw parquet datasets...")
    train_raw = pd.read_parquet("./input/train.parquet")
    test_raw = pd.read_parquet("./input/test.parquet")

    # Drop records missing target in training data
    if "target" in train_raw.columns:
        train_raw = train_raw.dropna(subset=["target"]).reset_index(drop=True)

    # Temporal holdout validation partition (last 15% of trading days)
    unique_dates = np.sort(train_raw["date_id"].unique())
    split_idx = int(len(unique_dates) * 0.85)
    split_date = unique_dates[split_idx]

    train_mask = train_raw["date_id"] < split_date
    val_mask = train_raw["date_id"] >= split_date

    train_split = train_raw[train_mask].copy().reset_index(drop=True)
    val_split = train_raw[val_mask].copy().reset_index(drop=True)
    test_split = test_raw.copy().reset_index(drop=True)

    print(
        f"Data partitioning: Train dates [{unique_dates[0]}..{unique_dates[split_idx-1]}], "
        f"Validation dates [{split_date}..{unique_dates[-1]}]"
    )
    print(
        f"Train samples: {len(train_split)}, Validation samples: {len(val_split)}, Test samples: {len(test_split)}"
    )

    # Compute stock-level baseline statistics strictly fitted on train split (preventing leakage)
    train_split_temp = train_split.copy()
    train_split_temp["temp_spread"] = (
        train_split_temp["ask_price"] - train_split_temp["bid_price"]
    )
    train_split_temp["temp_vol"] = (
        train_split_temp["bid_size"] + train_split_temp["ask_size"]
    )
    stock_stats = (
        train_split_temp.groupby("stock_id")
        .agg(
            stock_mean_spread=("temp_spread", "mean"),
            stock_mean_vol=("temp_vol", "mean"),
            stock_mean_imb=("imbalance_size", "mean"),
            stock_wap_std=("wap", "std"),
        )
        .reset_index()
    )
    del train_split_temp
    gc.collect()

    # Feature processing pipeline across splits
    processed_dfs = []
    for split_name, split_df in [
        ("train", train_split),
        ("val", val_split),
        ("test", test_split),
    ]:
        print(f"Engineering features for {split_name} split...")
        df_feat = compute_microstructure_features(split_df)
        df_feat = compute_cross_sectional_features(df_feat)
        df_feat = compute_temporal_features(df_feat)
        df_feat = df_feat.merge(stock_stats, on="stock_id", how="left")
        df_feat["signed_imbal_to_stock_vol"] = df_feat["signed_imbalance"] / (
            df_feat["stock_mean_vol"] + 1e-8
        )

        # Cast floats to float32 to conserve memory and optimize boosting performance
        float_cols = df_feat.select_dtypes(include=["float64"]).columns
        df_feat[float_cols] = df_feat[float_cols].astype(np.float32)
        int_cols = df_feat.select_dtypes(include=["int64"]).columns
        df_feat[int_cols] = df_feat[int_cols].astype(np.int32)

        processed_dfs.append(df_feat)
        del df_feat
        gc.collect()

    train_processed, val_processed, test_processed = processed_dfs

    ignore_cols = {"row_id", "target", "time_id", "date_id"}
    feature_cols = [
        c
        for c in train_processed.columns
        if c not in ignore_cols
        and c in val_processed.columns
        and c in test_processed.columns
    ]

    print(f"Total active training features: {len(feature_cols)}")

    X_train = train_processed[feature_cols]
    y_train = train_processed["target"].to_numpy()

    X_val = val_processed[feature_cols]
    y_val = val_processed["target"].to_numpy()

    X_test = test_processed[feature_cols]

    # Heterogeneous gradient boosting ensemble: LightGBM + XGBoost Hist
    lgb_params = {
        "objective": "regression_l1",
        "metric": "mae",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 128,
        "max_depth": -1,
        "min_child_samples": 64,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.7,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "n_estimators": 2500,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
        "force_col_wise": True,
    }

    lgb_callbacks = [
        lgb.early_stopping(stopping_rounds=50, first_metric_only=True, verbose=False),
        lgb.log_evaluation(period=100),
    ]

    print("Fitting LightGBM regressor with early stopping...")
    lgb_model = lgb.LGBMRegressor(**lgb_params)
    lgb_model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_names=["val"],
        callbacks=lgb_callbacks,
    )
    print(f"LightGBM best iteration achieved: {lgb_model.best_iteration_}")

    xgb_params = {
        "objective": "reg:absoluteerror",
        "eval_metric": "mae",
        "tree_method": "hist",
        "learning_rate": 0.06,
        "max_depth": 8,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "n_estimators": 2000,
        "random_state": 42,
        "n_jobs": -1,
        "early_stopping_rounds": 50,
    }

    print("Fitting XGBoost Histogram regressor with early stopping...")
    xgb_model = xgb.XGBRegressor(**xgb_params)
    xgb_model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=100,
    )
    print(f"XGBoost best iteration achieved: {xgb_model.best_iteration}")

    cat_params = {
        "loss_function": "MAE",
        "eval_metric": "MAE",
        "iterations": 1500,
        "learning_rate": 0.08,
        "depth": 7,
        "random_seed": 42,
        "thread_count": -1,
        "early_stopping_rounds": 50,
        "verbose": 100,
    }

    print("Fitting CatBoost regressor with early stopping...")
    cat_model = CatBoostRegressor(**cat_params)
    cat_model.fit(
        X_train,
        y_train,
        eval_set=(X_val, y_val),
        use_best_model=True,
    )
    print(f"CatBoost best iteration achieved: {cat_model.get_best_iteration()}")

    # Generate individual validation predictions
    val_preds_lgb = lgb_model.predict(X_val)
    val_preds_xgb = xgb_model.predict(X_val)
    val_preds_cat = cat_model.predict(X_val)

    lgb_mae = float(mean_absolute_error(y_val, val_preds_lgb))
    xgb_mae = float(mean_absolute_error(y_val, val_preds_xgb))
    cat_mae = float(mean_absolute_error(y_val, val_preds_cat))
    print(f"LightGBM Validation MAE: {lgb_mae:.5f}")
    print(f"XGBoost Validation MAE: {xgb_mae:.5f}")
    print(f"CatBoost Validation MAE: {cat_mae:.5f}")

    # Optimize 3-way ensemble blending weights via constrained simplex grid search
    best_weights = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
    best_raw_mae = float("inf")
    grid_steps = np.linspace(0.0, 1.0, 51)
    for w1 in grid_steps:
        for w2 in grid_steps:
            if w1 + w2 <= 1.0001:
                w3 = max(0.0, 1.0 - w1 - w2)
                blend = w1 * val_preds_lgb + w2 * val_preds_xgb + w3 * val_preds_cat
                mae = float(mean_absolute_error(y_val, blend))
                if mae < best_raw_mae:
                    best_raw_mae = mae
                    best_weights = (w1, w2, w3)

    w_lgb, w_xgb, w_cat = best_weights
    print(
        f"Optimal 3-Way Blend: LightGBM = {w_lgb:.2f}, XGBoost = {w_xgb:.2f}, CatBoost = {w_cat:.2f}"
    )
    val_preds_raw = (
        w_lgb * val_preds_lgb + w_xgb * val_preds_xgb + w_cat * val_preds_cat
    )
    val_mae_raw = best_raw_mae

    time_cols = ["date_id", "seconds_in_bucket"]
    val_preds_adj = align_snapshot_predictions(val_processed, val_preds_raw, time_cols)
    val_mae_adj = float(mean_absolute_error(y_val, val_preds_adj))

    print(f"Validation MAE (Raw Ensemble): {val_mae_raw:.5f}")
    print(f"Validation MAE (Zero-Centered Ensemble): {val_mae_adj:.5f}")

    if val_mae_adj < val_mae_raw:
        final_val_score = val_mae_adj
        apply_adjustment = True
        print("Selected zero-centering snapshot alignment for test inference.")
    else:
        final_val_score = val_mae_raw
        apply_adjustment = False
        print("Selected raw ensemble outputs for test inference.")

    # Test inference
    print("Generating predictions for test dataset...")
    test_preds_lgb = lgb_model.predict(X_test)
    test_preds_xgb = xgb_model.predict(X_test)
    test_preds_cat = cat_model.predict(X_test)
    test_preds_raw = (
        w_lgb * test_preds_lgb + w_xgb * test_preds_xgb + w_cat * test_preds_cat
    )

    if apply_adjustment:
        test_preds = align_snapshot_predictions(
            test_processed, test_preds_raw, time_cols
        )
    else:
        test_preds = test_preds_raw

    test_preds = np.nan_to_num(test_preds, nan=0.0)

    # Construct submission dataframe
    submission_df = pd.DataFrame(
        {
            "id": np.arange(len(test_preds)),
            "target": test_preds,
        }
    )

    submission_path = "./submission/submission.csv"
    submission_df.to_csv(submission_path, index=False)
    print(
        f"Submission successfully written to {submission_path} with {len(submission_df)} rows."
    )

    print(f"Final Validation Score: {final_val_score}")


if __name__ == "__main__":
    main()
