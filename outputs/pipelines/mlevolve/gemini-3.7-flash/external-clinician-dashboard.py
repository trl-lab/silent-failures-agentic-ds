from itertools import combinations
import json
import math
import os
from typing import Any, Dict, List, Tuple
from catboost import CatBoostRegressor
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb


# =========================================================================
# Main End-to-End Pipeline
# =========================================================================
def main():
    print("[1/5] Loading raw datasets...")
    input_dir = "./input"
    working_dir = "./working"
    submission_dir = "./submission"

    os.makedirs(working_dir, exist_ok=True)
    os.makedirs(submission_dir, exist_ok=True)

    train_df = pd.read_parquet(os.path.join(input_dir, "train.parquet")).copy()
    test_df = pd.read_parquet(os.path.join(input_dir, "test.parquet")).copy()

    print(f"Loaded train shape: {train_df.shape}, test shape: {test_df.shape}")

    target_col = "Score"
    id_col = None
    for candidate in ["id", "ID", "Id", "index"]:
        if candidate in test_df.columns:
            id_col = candidate
            break

    if id_col is None:
        train_df["id"] = np.arange(len(train_df))
        test_df["id"] = np.arange(len(test_df))
        id_col = "id"

    test_ids = test_df[id_col].values.astype(int)

    # Process potential datetime columns
    all_cols = [c for c in train_df.columns if c not in [target_col, id_col]]
    datetime_cols = []
    for col in all_cols:
        if pd.api.types.is_datetime64_any_dtype(train_df[col]):
            datetime_cols.append(col)
        elif train_df[col].dtype == "object":
            sample = train_df[col].dropna().head(20)
            if len(sample) > 0:
                try:
                    pd.to_datetime(sample, format="mixed")
                    datetime_cols.append(col)
                except Exception:
                    pass

    for col in datetime_cols:
        for df in [train_df, test_df]:
            dt_series = pd.to_datetime(df[col], errors="coerce")
            df[f"{col}_year"] = dt_series.dt.year.astype("float32")
            df[f"{col}_month"] = dt_series.dt.month.astype("float32")
            df[f"{col}_day"] = dt_series.dt.day.astype("float32")
            df[f"{col}_dayofweek"] = dt_series.dt.dayofweek.astype("float32")
            df[f"{col}_quarter"] = dt_series.dt.quarter.astype("float32")
            df[f"{col}_is_month_end"] = dt_series.dt.is_month_end.astype("float32")
            df.drop(columns=[col], inplace=True)

    feature_cols = [c for c in train_df.columns if c not in [target_col, id_col]]
    categorical_cols = []
    numerical_cols = []

    for col in feature_cols:
        if pd.api.types.is_numeric_dtype(train_df[col]):
            numerical_cols.append(col)
        else:
            categorical_cols.append(col)

    print(
        f"Initial Categorical features ({len(categorical_cols)}), Numerical features ({len(numerical_cols)})"
    )

    print("[2/5] Creating Stratified CV Folds and Feature Engineering...")
    n_splits = 5
    num_bins = 10
    target_series = train_df[target_col].values
    binned_target = pd.qcut(target_series, q=num_bins, labels=False, duplicates="drop")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    train_df["fold"] = -1
    for fold, (_, val_idx) in enumerate(skf.split(train_df, binned_target)):
        train_df.loc[val_idx, "fold"] = fold

    # Missingness indicators
    for col in numerical_cols:
        if train_df[col].isna().sum() > 0 or test_df[col].isna().sum() > 0:
            train_df[f"{col}_isna"] = train_df[col].isna().astype("float32")
            test_df[f"{col}_isna"] = test_df[col].isna().astype("float32")

    # Frequency encoding
    for col in categorical_cols:
        train_df[col] = train_df[col].fillna("__MISSING__").astype(str)
        test_df[col] = test_df[col].fillna("__MISSING__").astype(str)

        freq_map = train_df[col].value_counts(normalize=True).to_dict()
        train_df[f"{col}_freq"] = (
            train_df[col].map(freq_map).fillna(0.0).astype("float32")
        )
        test_df[f"{col}_freq"] = (
            test_df[col].map(freq_map).fillna(0.0).astype("float32")
        )

        count_map = train_df[col].value_counts().to_dict()
        train_df[f"{col}_count"] = (
            train_df[col].map(count_map).fillna(0.0).astype("float32")
        )
        test_df[f"{col}_count"] = (
            test_df[col].map(count_map).fillna(0.0).astype("float32")
        )

    # High-order Pairwise Categorical Interactions
    top_cats_for_cross = [c for c in categorical_cols if 1 < train_df[c].nunique() <= 100][:6]
    for c1, c2 in combinations(top_cats_for_cross, 2):
        cross_col = f"{c1}_x_{c2}"
        train_df[cross_col] = train_df[c1] + "___" + train_df[c2]
        test_df[cross_col] = test_df[c1] + "___" + test_df[c2]

        freq_cross = train_df[cross_col].value_counts(normalize=True).to_dict()
        train_df[f"{cross_col}_freq"] = train_df[cross_col].map(freq_cross).fillna(0.0).astype("float32")
        test_df[f"{cross_col}_freq"] = test_df[cross_col].map(freq_cross).fillna(0.0).astype("float32")

        cross_cats = {val: idx for idx, val in enumerate(train_df[cross_col].unique())}
        train_df[f"{cross_col}_code"] = train_df[cross_col].map(cross_cats).fillna(-1).astype("int32")
        test_df[f"{cross_col}_code"] = test_df[cross_col].map(cross_cats).fillna(-1).astype("int32")

    # Multi-Statistic Out-Of-Fold Smoothed Target Encoding
    smoothing_weight = 10.0
    global_mean = float(train_df[target_col].mean())
    global_std = float(train_df[target_col].std())
    global_min = float(train_df[target_col].min())
    global_max = float(train_df[target_col].max())

    for col in categorical_cols:
        for stat in ["mean", "std", "min", "max"]:
            train_df[f"{col}_te_{stat}"] = np.nan

        for fold in range(n_splits):
            tr_mask = train_df["fold"] != fold
            val_mask = train_df["fold"] == fold

            fold_tr = train_df[tr_mask]
            stats = fold_tr.groupby(col)[target_col].agg(["count", "mean", "std", "min", "max"])
            smoothed_mean = (
                stats["count"] * stats["mean"] + smoothing_weight * global_mean
            ) / (stats["count"] + smoothing_weight)

            train_df.loc[val_mask, f"{col}_te_mean"] = (
                train_df.loc[val_mask, col].map(smoothed_mean).fillna(global_mean).astype("float32")
            )
            train_df.loc[val_mask, f"{col}_te_std"] = (
                train_df.loc[val_mask, col].map(stats["std"]).fillna(global_std).astype("float32")
            )
            train_df.loc[val_mask, f"{col}_te_min"] = (
                train_df.loc[val_mask, col].map(stats["min"]).fillna(global_min).astype("float32")
            )
            train_df.loc[val_mask, f"{col}_te_max"] = (
                train_df.loc[val_mask, col].map(stats["max"]).fillna(global_max).astype("float32")
            )

        stats_full = train_df.groupby(col)[target_col].agg(["count", "mean", "std", "min", "max"])
        smoothed_mean_full = (
            stats_full["count"] * stats_full["mean"] + smoothing_weight * global_mean
        ) / (stats_full["count"] + smoothing_weight)
        test_df[f"{col}_te_mean"] = test_df[col].map(smoothed_mean_full).fillna(global_mean).astype("float32")
        test_df[f"{col}_te_std"] = test_df[col].map(stats_full["std"]).fillna(global_std).astype("float32")
        test_df[f"{col}_te_min"] = test_df[col].map(stats_full["min"]).fillna(global_min).astype("float32")
        test_df[f"{col}_te_max"] = test_df[col].map(stats_full["max"]).fillna(global_max).astype("float32")

    # Hierarchical group aggregations
    top_cats = [c for c in categorical_cols if 1 < train_df[c].nunique() < 500][:3]
    top_nums = numerical_cols[:5] if len(numerical_cols) >= 5 else numerical_cols

    if len(top_cats) > 0 and len(top_nums) > 0:
        for cat_col in top_cats:
            for num_col in top_nums:
                agg_df = (
                    train_df.groupby(cat_col)[num_col]
                    .agg(["mean", "std", "min", "max"])
                    .reset_index()
                )
                agg_df.columns = [
                    cat_col,
                    f"{num_col}_grp_{cat_col}_mean",
                    f"{num_col}_grp_{cat_col}_std",
                    f"{num_col}_grp_{cat_col}_min",
                    f"{num_col}_grp_{cat_col}_max",
                ]

                train_df = train_df.merge(agg_df, on=cat_col, how="left")
                test_df = test_df.merge(agg_df, on=cat_col, how="left")

                train_df[f"{num_col}_diff_from_{cat_col}_mean"] = (
                    train_df[num_col] - train_df[f"{num_col}_grp_{cat_col}_mean"]
                ).astype("float32")
                test_df[f"{num_col}_diff_from_{cat_col}_mean"] = (
                    test_df[num_col] - test_df[f"{num_col}_grp_{cat_col}_mean"]
                ).astype("float32")

                train_df[f"{num_col}_ratio_to_{cat_col}_mean"] = (
                    train_df[num_col]
                    / (train_df[f"{num_col}_grp_{cat_col}_mean"].abs() + 1e-5)
                ).astype("float32")
                test_df[f"{num_col}_ratio_to_{cat_col}_mean"] = (
                    test_df[num_col]
                    / (test_df[f"{num_col}_grp_{cat_col}_mean"].abs() + 1e-5)
                ).astype("float32")

    # Pairwise numerical interactions and robust transforms
    for col in numerical_cols:
        train_df[f"{col}_log1p"] = np.log1p(np.maximum(0, train_df[col])).astype("float32")
        test_df[f"{col}_log1p"] = np.log1p(np.maximum(0, test_df[col])).astype("float32")
        train_df[f"{col}_sqrt"] = np.sqrt(np.maximum(0, train_df[col])).astype("float32")
        test_df[f"{col}_sqrt"] = np.sqrt(np.maximum(0, test_df[col])).astype("float32")

    if len(top_nums) >= 2:
        for i in range(len(top_nums)):
            for j in range(i + 1, min(i + 3, len(top_nums))):
                col_a = top_nums[i]
                col_b = top_nums[j]

                train_df[f"{col_a}_add_{col_b}"] = (
                    train_df[col_a] + train_df[col_b]
                ).astype("float32")
                test_df[f"{col_a}_add_{col_b}"] = (
                    test_df[col_a] + test_df[col_b]
                ).astype("float32")

                train_df[f"{col_a}_sub_{col_b}"] = (
                    train_df[col_a] - train_df[col_b]
                ).astype("float32")
                test_df[f"{col_a}_sub_{col_b}"] = (
                    test_df[col_a] - test_df[col_b]
                ).astype("float32")

                train_df[f"{col_a}_mul_{col_b}"] = (
                    train_df[col_a] * train_df[col_b]
                ).astype("float32")
                test_df[f"{col_a}_mul_{col_b}"] = (
                    test_df[col_a] * test_df[col_b]
                ).astype("float32")

                train_df[f"{col_a}_div_{col_b}"] = (
                    train_df[col_a] / (train_df[col_b].abs() + 1e-5)
                ).astype("float32")
                test_df[f"{col_a}_div_{col_b}"] = (
                    test_df[col_a] / (test_df[col_b].abs() + 1e-5)
                ).astype("float32")

    # Categorical Ordinal codes
    for col in categorical_cols:
        train_df[col] = train_df[col].astype("category")
        categories = train_df[col].cat.categories
        train_df[f"{col}_code"] = train_df[col].cat.codes.astype("int32")
        test_df[f"{col}_code"] = pd.Categorical(
            test_df[col], categories=categories
        ).codes.astype("int32")

    # Impute missing & non-finite numerical values with training medians
    str_cols_to_drop = [
        c for c in train_df.columns
        if train_df[c].dtype == "object" or str(train_df[c].dtype) == "category"
    ]
    num_final_cols = [
        c
        for c in train_df.columns
        if c not in [target_col, id_col, "fold"] + str_cols_to_drop
    ]
    for col in num_final_cols:
        train_df[col] = train_df[col].replace([np.inf, -np.inf], np.nan)
        test_df[col] = test_df[col].replace([np.inf, -np.inf], np.nan)

        median_val = train_df[col].median()
        if pd.isna(median_val):
            median_val = 0.0
        train_df[col] = train_df[col].fillna(median_val).astype("float32")
        test_df[col] = test_df[col].fillna(median_val).astype("float32")

    train_engineered = train_df.drop(columns=str_cols_to_drop)
    test_engineered = test_df.drop(columns=str_cols_to_drop)

    final_features = [
        c for c in train_engineered.columns if c not in [target_col, id_col, "fold"]
    ]
    print(f"Total engineered features: {len(final_features)}")

    # =========================================================================
    # 3. Model Training: LightGBM, XGBoost, and CatBoost
    # =========================================================================
    print(f"[3/5] Executing {n_splits}-Fold Cross Validation...")

    oof_lgb = np.zeros(len(train_engineered), dtype=np.float32)
    oof_xgb = np.zeros(len(train_engineered), dtype=np.float32)
    oof_cat = np.zeros(len(train_engineered), dtype=np.float32)

    test_lgb_folds = np.zeros((n_splits, len(test_engineered)), dtype=np.float32)
    test_xgb_folds = np.zeros((n_splits, len(test_engineered)), dtype=np.float32)
    test_cat_folds = np.zeros((n_splits, len(test_engineered)), dtype=np.float32)

    y_all = train_engineered[target_col].values.astype(np.float32)
    X_test = test_engineered[final_features].values.astype(np.float32)

    lgb_params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": -1,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "min_child_samples": 20,
        "lambda_l1": 0.1,
        "lambda_l2": 1.0,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }

    xgb_params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "learning_rate": 0.03,
        "max_depth": 6,
        "min_child_weight": 3,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": -1,
        "tree_method": "hist",
    }

    cat_params = {
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "iterations": 1500,
        "learning_rate": 0.03,
        "depth": 6,
        "l2_leaf_reg": 3.0,
        "random_seed": 42,
        "verbose": False,
        "thread_count": -1,
    }

    for fold in range(n_splits):
        train_idx = np.where(train_engineered["fold"] != fold)[0]
        val_idx = np.where(train_engineered["fold"] == fold)[0]

        X_tr = train_engineered.iloc[train_idx][final_features].values.astype(
            np.float32
        )
        y_tr = y_all[train_idx]
        X_val = train_engineered.iloc[val_idx][final_features].values.astype(np.float32)
        y_val = y_all[val_idx]

        # ----------------- 1. LightGBM -----------------
        trn_data = lgb.Dataset(X_tr, label=y_tr)
        vld_data = lgb.Dataset(X_val, label=y_val, reference=trn_data)

        lgb_model = lgb.train(
            lgb_params,
            trn_data,
            num_boost_round=1500,
            valid_sets=[trn_data, vld_data],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )
        oof_lgb[val_idx] = lgb_model.predict(
            X_val, num_iteration=lgb_model.best_iteration
        )
        test_lgb_folds[fold] = lgb_model.predict(
            X_test, num_iteration=lgb_model.best_iteration
        )

        # ----------------- 2. XGBoost -----------------
        dtrain = xgb.DMatrix(X_tr, label=y_tr)
        dval = xgb.DMatrix(X_val, label=y_val)
        dtest = xgb.DMatrix(X_test)

        xgb_model = xgb.train(
            xgb_params,
            dtrain,
            num_boost_round=1500,
            evals=[(dval, "val")],
            early_stopping_rounds=50,
            verbose_eval=False,
        )
        oof_xgb[val_idx] = xgb_model.predict(dval)
        test_xgb_folds[fold] = xgb_model.predict(dtest)

        # ----------------- 3. CatBoost -----------------
        cb_model = CatBoostRegressor(**cat_params)
        cb_model.fit(
            X_tr,
            y_tr,
            eval_set=(X_val, y_val),
            early_stopping_rounds=50,
            verbose=False,
        )
        oof_cat[val_idx] = cb_model.predict(X_val)
        test_cat_folds[fold] = cb_model.predict(X_test)

        fold_rmse_lgb = np.sqrt(mean_squared_error(y_val, oof_lgb[val_idx]))
        fold_rmse_xgb = np.sqrt(mean_squared_error(y_val, oof_xgb[val_idx]))
        fold_rmse_cat = np.sqrt(mean_squared_error(y_val, oof_cat[val_idx]))
        print(
            f"Fold {fold+1}/{n_splits} - LGB RMSE: {fold_rmse_lgb:.5f} | XGB RMSE: {fold_rmse_xgb:.5f} | CatBoost RMSE: {fold_rmse_cat:.5f}"
        )

    # Out-of-fold single model performance
    rmse_lgb = np.sqrt(mean_squared_error(y_all, oof_lgb))
    rmse_xgb = np.sqrt(mean_squared_error(y_all, oof_xgb))
    rmse_cat = np.sqrt(mean_squared_error(y_all, oof_cat))
    print(
        f"[4/5] Single Model OOF RMSE: LGB: {rmse_lgb:.5f} | XGB: {rmse_xgb:.5f} | CatBoost: {rmse_cat:.5f}"
    )

    # Optimal ensemble weights via SLSQP constrained optimization
    oof_matrix = np.column_stack([oof_lgb, oof_xgb, oof_cat])
    test_matrix = np.column_stack(
        [
            test_lgb_folds.mean(axis=0),
            test_xgb_folds.mean(axis=0),
            test_cat_folds.mean(axis=0),
        ]
    )

    def blend_objective(weights):
        pred = oof_matrix @ weights
        return np.sqrt(mean_squared_error(y_all, pred))

    init_weights = np.array([1 / 3, 1 / 3, 1 / 3])
    bounds = [(0, 1), (0, 1), (0, 1)]
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

    res = minimize(
        blend_objective,
        init_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    opt_weights = res.x
    print(
        f"Optimized Blend Weights -> LGB: {opt_weights[0]:.4f}, XGB: {opt_weights[1]:.4f}, CatBoost: {opt_weights[2]:.4f}"
    )

    final_oof_preds = oof_matrix @ opt_weights
    final_score = float(np.sqrt(mean_squared_error(y_all, final_oof_preds)))
    print(f"Blended Out-Of-Fold RMSE: {final_score:.5f}")

    # =========================================================================
    # 4. Final Inference and Submission
    # =========================================================================
    print("[5/5] Generating submission file...")
    final_test_preds = test_matrix @ opt_weights
    min_target = float(y_all.min())
    max_target = float(y_all.max())
    final_test_preds = np.clip(final_test_preds, min_target, max_target)

    submission_df = pd.DataFrame(
        {
            "id": test_ids,
            "Score": final_test_preds,
        }
    )

    submission_path = os.path.join(submission_dir, "submission.csv")
    submission_df.to_csv(submission_path, index=False)

    print(f"Saved submission to {submission_path} with shape {submission_df.shape}")
    print(f"Final Validation Score: {final_score}")


if __name__ == "__main__":
    main()