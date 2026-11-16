import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor


def run_pipeline():
    # 1. Load raw data
    os.makedirs("./submission", exist_ok=True)
    os.makedirs("./working", exist_ok=True)

    train_df = pd.read_parquet("./input/train.parquet")
    test_df = pd.read_parquet("./input/test.parquet")

    # Track test set row ordering
    test_df["id"] = np.arange(len(test_df))

    train_df["is_test"] = 0
    test_df["is_test"] = 1

    target_col = "traffic_volume"
    test_df[target_col] = np.nan

    # 2. Data Cleaning & Anomaly Correction (separately to prevent test leakage)
    for df_split in [train_df, test_df]:
        df_split.loc[df_split["temp"] < 100.0, "temp"] = np.nan
        df_split["temp"] = df_split["temp"].interpolate(method="linear").bfill().ffill()

    df = pd.concat([train_df, test_df], ignore_index=True)
    df["date_time"] = pd.to_datetime(df["date_time"])
    df = df.sort_values("date_time").reset_index(drop=True)

    # Fill missing values in categorical columns to avoid type errors during sorting/encoding
    df["holiday"] = df["holiday"].fillna("None").astype(str)
    df["weather_main"] = df["weather_main"].fillna("Missing").astype(str)
    df["weather_description"] = df["weather_description"].fillna("Missing").astype(str)

    df["temp_celsius"] = df["temp"] - 273.15
    df["temp_sq"] = df["temp_celsius"] ** 2

    # Rain and snow outlier management
    df["rain_1h"] = np.minimum(df["rain_1h"], 100.0)
    df["log_rain"] = np.log1p(df["rain_1h"])
    df["is_raining"] = (df["rain_1h"] > 0).astype(int)

    df["log_snow"] = np.log1p(df["snow_1h"])
    df["is_snowing"] = (df["snow_1h"] > 0).astype(int)

    # Cloud features
    df["cloud_ratio"] = df["clouds_all"] / 100.0

    # 3. Temporal Decomposition & Cyclical Encodings
    dt = df["date_time"].dt
    df["hour"] = dt.hour
    df["dayofweek"] = dt.dayofweek
    df["day"] = dt.day
    df["month"] = dt.month
    df["year"] = dt.year
    df["dayofyear"] = dt.dayofyear
    df["quarter"] = dt.quarter

    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["is_rush_morning"] = (
        (df["hour"] >= 6) & (df["hour"] <= 9) & (df["dayofweek"] < 5)
    ).astype(int)
    df["is_rush_evening"] = (
        (df["hour"] >= 15) & (df["hour"] <= 18) & (df["dayofweek"] < 5)
    ).astype(int)
    df["is_holiday"] = (df["holiday"] != "None").astype(int)

    df["sin_hour"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["cos_hour"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["sin_dow"] = np.sin(2 * np.pi * df["dayofweek"] / 7.0)
    df["cos_dow"] = np.cos(2 * np.pi * df["dayofweek"] / 7.0)
    df["sin_month"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["cos_month"] = np.cos(2 * np.pi * df["month"] / 12.0)
    df["sin_doy"] = np.sin(2 * np.pi * df["dayofyear"] / 365.25)
    df["cos_doy"] = np.cos(2 * np.pi * df["dayofyear"] / 365.25)

    # Domain Interaction Terms
    df["temp_cloud_inter"] = df["temp_celsius"] * df["cloud_ratio"]
    df["temp_rain_inter"] = df["temp_celsius"] * df["log_rain"]
    df["temp_snow_inter"] = df["temp_celsius"] * df["log_snow"]
    df["bad_weather_score"] = (
        df["is_raining"] + df["is_snowing"] + (df["cloud_ratio"] > 0.8).astype(int)
    )

    # 4. Fit Categorical Statistics Leak-Free on Training Set
    train_mask = df["is_test"] == 0
    test_mask = df["is_test"] == 1

    cat_cols = ["holiday", "weather_main", "weather_description"]

    for col in cat_cols:
        freq_map = df.loc[train_mask, col].value_counts(normalize=True).to_dict()
        df[f"{col}_freq"] = df[col].map(freq_map).fillna(0.0)

        categories = sorted(df.loc[train_mask, col].unique())
        cat_type = pd.CategoricalDtype(categories=categories, ordered=False)
        df[f"{col}_code"] = df[col].astype(cat_type).cat.codes

    train_processed = (
        df[train_mask].drop(columns=["is_test", "id"]).reset_index(drop=True)
    )
    test_processed = (
        df[test_mask].sort_values("id").drop(columns=["is_test"]).reset_index(drop=True)
    )

    # 5. 5-Fold Cross-Validation Pipeline with Out-Of-Fold Target Summary Features
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    n_train = len(train_processed)
    n_test = len(test_processed)
    test_ids = test_processed["id"].values

    oof_preds_lgb = np.zeros(n_train)
    oof_preds_xgb = np.zeros(n_train)
    oof_preds_cb = np.zeros(n_train)

    test_preds_lgb = np.zeros(n_test)
    test_preds_xgb = np.zeros(n_test)
    test_preds_cb = np.zeros(n_test)

    groups = [
        (["hour", "dayofweek"], "target_hour_dow"),
        (["hour", "is_weekend"], "target_hour_weekend"),
        (["hour", "weather_main"], "target_hour_weather"),
    ]

    drop_cols = [
        "date_time",
        "holiday",
        "weather_main",
        "weather_description",
        "traffic_volume",
        "id",
    ]

    y_true_all = train_processed[target_col].values

    for fold, (train_idx, val_idx) in enumerate(kf.split(train_processed)):
        X_tr = train_processed.iloc[train_idx].copy()
        X_va = train_processed.iloc[val_idx].copy()
        X_te = test_processed.copy()

        # Compute historical target aggregations strictly on fold training data
        for group_cols, group_prefix in groups:
            stats = (
                X_tr.groupby(group_cols)[target_col]
                .agg(["mean", "std", "median"])
                .reset_index()
            )
            stats.columns = group_cols + [
                f"{group_prefix}_mean",
                f"{group_prefix}_std",
                f"{group_prefix}_median",
            ]

            X_tr = X_tr.merge(stats, on=group_cols, how="left")
            X_va = X_va.merge(stats, on=group_cols, how="left")
            X_te = X_te.merge(stats, on=group_cols, how="left")

            overall_mean = X_tr[target_col].mean()
            overall_median = X_tr[target_col].median()

            X_tr[f"{group_prefix}_std"] = X_tr[f"{group_prefix}_std"].fillna(0.0)
            X_va[f"{group_prefix}_std"] = X_va[f"{group_prefix}_std"].fillna(0.0)
            X_te[f"{group_prefix}_std"] = X_te[f"{group_prefix}_std"].fillna(0.0)

            X_tr[f"{group_prefix}_mean"] = X_tr[f"{group_prefix}_mean"].fillna(overall_mean)
            X_va[f"{group_prefix}_mean"] = X_va[f"{group_prefix}_mean"].fillna(overall_mean)
            X_te[f"{group_prefix}_mean"] = X_te[f"{group_prefix}_mean"].fillna(overall_mean)

            X_tr[f"{group_prefix}_median"] = X_tr[f"{group_prefix}_median"].fillna(overall_median)
            X_va[f"{group_prefix}_median"] = X_va[f"{group_prefix}_median"].fillna(overall_median)
            X_te[f"{group_prefix}_median"] = X_te[f"{group_prefix}_median"].fillna(overall_median)

        feature_cols = [col for col in X_tr.columns if col not in drop_cols]

        X_train = X_tr[feature_cols].copy()
        y_train = X_tr[target_col].values

        X_val = X_va[feature_cols].copy()
        y_val = X_va[target_col].values

        X_test = X_te[feature_cols].copy()

        # Train LightGBM
        lgb_model = lgb.LGBMRegressor(
            n_estimators=3000,
            learning_rate=0.03,
            num_leaves=63,
            max_depth=-1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42 + fold,
            n_jobs=-1,
        )
        lgb_model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="rmse",
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
        )
        oof_preds_lgb[val_idx] = lgb_model.predict(X_val)
        test_preds_lgb += lgb_model.predict(X_test) / n_splits

        # Train XGBoost
        xgb_model = xgb.XGBRegressor(
            n_estimators=3000,
            learning_rate=0.03,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42 + fold,
            early_stopping_rounds=50,
            eval_metric="rmse",
            n_jobs=-1,
        )
        xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        oof_preds_xgb[val_idx] = xgb_model.predict(X_val)
        test_preds_xgb += xgb_model.predict(X_test) / n_splits

        # Train CatBoost
        cb_model = CatBoostRegressor(
            iterations=3000,
            learning_rate=0.03,
            depth=6,
            random_seed=42 + fold,
            verbose=0,
            early_stopping_rounds=50,
        )
        cb_model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)
        oof_preds_cb[val_idx] = cb_model.predict(X_val)
        test_preds_cb += cb_model.predict(X_test) / n_splits

    # 6. Optimize Ensemble Weights on Full OOF Predictions
    def blend_objective(weights):
        w1, w2, w3 = weights
        pred = w1 * oof_preds_lgb + w2 * oof_preds_xgb + w3 * oof_preds_cb
        pred = np.maximum(pred, 0.0)
        return np.sqrt(mean_squared_error(y_true_all, pred))

    init_weights = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
    bounds = [(0, 1), (0, 1), (0, 1)]
    constraints = {"type": "eq", "fun": lambda w: sum(w) - 1.0}

    opt_res = minimize(
        blend_objective,
        init_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    w1, w2, w3 = opt_res.x

    # 7. Compute Final Validation Predictions & Save Submission
    oof_preds_final = np.maximum(
        w1 * oof_preds_lgb + w2 * oof_preds_xgb + w3 * oof_preds_cb, 0.0
    )
    test_preds_final = np.maximum(
        w1 * test_preds_lgb + w2 * test_preds_xgb + w3 * test_preds_cb, 0.0
    )

    final_val_score = np.sqrt(mean_squared_error(y_true_all, oof_preds_final))

    submission_df = pd.DataFrame(
        {"id": test_ids.astype(int), "traffic_volume": test_preds_final}
    )
    submission_df.to_csv("./submission/submission.csv", index=False)

    print(f"Final Validation Score: {final_val_score}")


if __name__ == "__main__":
    run_pipeline()