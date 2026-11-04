import gc
import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generate microstructure, cross-sectional, and lag features."""
    df = df.copy()

    # Basic spread and price features
    df["bid_ask_spread"] = df["ask_price"] - df["bid_price"]
    df["mid_price"] = (df["ask_price"] + df["bid_price"]) / 2.0
    df["wap_mid_diff"] = df["wap"] - df["mid_price"]
    df["wap_ref_diff"] = df["wap"] - df["reference_price"]
    df["ask_ref_diff"] = df["ask_price"] - df["reference_price"]
    df["bid_ref_diff"] = df["bid_price"] - df["reference_price"]

    # Imbalance and size features
    df["volume"] = df["ask_size"] + df["bid_size"]
    df["size_imbalance"] = (df["bid_size"] - df["ask_size"]) / (df["volume"] + 1e-8)
    df["matched_imbalance_ratio"] = (df["imbalance_size"] - df["matched_size"]) / (
        df["imbalance_size"] + df["matched_size"] + 1e-8
    )
    df["signed_imbalance"] = df["imbalance_size"] * df["imbalance_buy_sell_flag"]

    # Price & volume interaction
    df["bid_liquidity"] = df["bid_size"] * df["bid_price"]
    df["ask_liquidity"] = df["ask_size"] * df["ask_price"]
    df["liquidity_imbalance"] = (df["bid_liquidity"] - df["ask_liquidity"]) / (
        df["bid_liquidity"] + df["ask_liquidity"] + 1e-8
    )

    # Sort to compute sequential lags per stock within each day
    sort_cols = ["date_id", "stock_id", "seconds_in_bucket"]
    df = df.sort_values(sort_cols).reset_index(drop=True)

    # Within-day lags & diffs
    group = df.groupby(["date_id", "stock_id"])
    for lag in [1, 2, 3, 6]:
        df[f"wap_diff_lag_{lag}"] = group["wap"].diff(lag)
        df[f"imbalance_diff_lag_{lag}"] = group["signed_imbalance"].diff(lag)
        df[f"spread_diff_lag_{lag}"] = group["bid_ask_spread"].diff(lag)

    # Cross-sectional aggregate market features per time step
    time_group = df.groupby(["date_id", "seconds_in_bucket"])
    cs_wap_mean = time_group["wap"].transform("mean")
    cs_wap_std = time_group["wap"].transform("std")
    cs_imbalance_mean = time_group["signed_imbalance"].transform("mean")
    cs_spread_mean = time_group["bid_ask_spread"].transform("mean")

    df["market_wap_mean"] = cs_wap_mean
    df["market_wap_std"] = cs_wap_std
    df["market_imbalance_mean"] = cs_imbalance_mean
    df["market_spread_mean"] = cs_spread_mean
    df["wap_vs_market_mean"] = df["wap"] - cs_wap_mean

    # Downcast float columns to float32 to reduce memory footprint
    float_cols = df.select_dtypes(include=["float64"]).columns
    for c in float_cols:
        if c != "target":
            df[c] = df[c].astype(np.float32)

    return df


def zero_center_predictions(df: pd.DataFrame, pred_col: str) -> np.ndarray:
    """Subtract cross-sectional mean per time step (index-neutralization)."""
    mean_pred = df.groupby(["date_id", "seconds_in_bucket"])[pred_col].transform("mean")
    return df[pred_col] - mean_pred


def main():
    print("Loading data...")
    train_path = "./input/train.parquet"
    test_path = "./input/test.parquet"

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    # Filter out missing targets in training
    train_df = train_df.dropna(subset=["target"]).reset_index(drop=True)

    print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

    # Combine for feature engineering or process separately
    print("Generating features...")
    train_feat = create_features(train_df)
    test_feat = create_features(test_df)

    del train_df
    gc.collect()

    # Define feature set
    ignore_cols = ["row_id", "target", "date_id", "time_id"]
    feature_cols = [c for c in train_feat.columns if c not in ignore_cols]
    print(f"Total features: {len(feature_cols)}")

    # 5-fold cross validation grouped by date_id
    unique_dates = train_feat["date_id"].unique()
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    oof_preds = np.zeros(len(train_feat), dtype=np.float32)
    test_preds = np.zeros(len(test_feat), dtype=np.float32)

    lgb_params = {
        "objective": "mae",
        "metric": "mae",
        "boosting_type": "gbdt",
        "learning_rate": 0.08,
        "num_leaves": 63,
        "max_depth": 8,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "min_child_samples": 64,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }

    models = []
    print("Training LightGBM models...")
    for fold, (trn_date_idx, val_date_idx) in enumerate(kf.split(unique_dates)):
        trn_dates = unique_dates[trn_date_idx]
        val_dates = unique_dates[val_date_idx]

        train_mask = train_feat["date_id"].isin(trn_dates)
        val_mask = train_feat["date_id"].isin(val_dates)

        X_train, y_train = (
            train_feat.loc[train_mask, feature_cols],
            train_feat.loc[train_mask, "target"],
        )
        X_val, y_val = (
            train_feat.loc[val_mask, feature_cols],
            train_feat.loc[val_mask, "target"],
        )

        trn_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=trn_data)

        model = lgb.train(
            lgb_params,
            trn_data,
            num_boost_round=1200,
            valid_sets=[trn_data, val_data],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(200)],
        )

        val_pred = model.predict(X_val, num_iteration=model.best_iteration)
        oof_preds[val_mask] = val_pred

        test_pred_fold = model.predict(
            test_feat[feature_cols], num_iteration=model.best_iteration
        )
        test_preds += test_pred_fold / 5.0

        fold_mae = mean_absolute_error(y_val, val_pred)
        print(f"Fold {fold + 1} MAE (raw): {fold_mae:.5f}")
        models.append(model)

    # Evaluate raw OOF
    overall_oof_mae = mean_absolute_error(train_feat["target"], oof_preds)
    print(f"Overall OOF MAE (raw): {overall_oof_mae:.5f}")

    # Cross-sectional zero-centering post-processing
    train_feat["pred_raw"] = oof_preds
    oof_centered = zero_center_predictions(train_feat, "pred_raw")
    centered_oof_mae = mean_absolute_error(train_feat["target"], oof_centered)
    print(f"Overall OOF MAE (zero-centered): {centered_oof_mae:.5f}")

    # Process test predictions
    test_feat["pred_raw"] = test_preds
    final_test_pred = zero_center_predictions(test_feat, "pred_raw")

    # Save submission
    os.makedirs("./working", exist_ok=True)
    submission = pd.DataFrame(
        {
            "id": np.arange(len(final_test_pred)),
            "target": final_test_pred.values,
        }
    )
    submission.to_csv("./working/submission.csv", index=False)
    print(f"Submission saved successfully with shape: {submission.shape}")


if __name__ == "__main__":
    main()
