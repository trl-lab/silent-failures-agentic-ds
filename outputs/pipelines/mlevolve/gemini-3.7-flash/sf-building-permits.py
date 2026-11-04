import json
import os
import re
import warnings
from catboost import CatBoostRegressor
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import xgboost as xgb

warnings.filterwarnings("ignore")


# =========================================================================
# 1. Model Configuration Factories
# =========================================================================
def get_lgb_params(seed: int = 42):
    return {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "n_estimators": 3500,
        "learning_rate": 0.025,
        "num_leaves": 63,
        "max_depth": 8,
        "min_child_samples": 20,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.75,
        "reg_alpha": 0.5,
        "reg_lambda": 2.5,
        "random_state": seed,
        "n_jobs": -1,
        "verbose": -1,
    }


def get_xgb_params(seed: int = 42):
    return {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "n_estimators": 3500,
        "learning_rate": 0.025,
        "max_depth": 7,
        "min_child_weight": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.75,
        "reg_alpha": 0.5,
        "reg_lambda": 2.5,
        "random_state": seed,
        "n_jobs": -1,
        "early_stopping_rounds": 50,
    }


def get_catboost_params(seed: int = 42):
    return {
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "iterations": 3500,
        "learning_rate": 0.03,
        "depth": 6,
        "l2_leaf_reg": 6.0,
        "rsm": 0.75,
        "random_seed": seed,
        "verbose": False,
        "thread_count": -1,
        "early_stopping_rounds": 50,
    }


# =========================================================================
# 2. Main Execution Pipeline
# =========================================================================
def main():
    os.makedirs("./working", exist_ok=True)
    os.makedirs("./submission", exist_ok=True)

    # 1. Load Data
    train_df = pd.read_parquet("./input/train.parquet")
    test_df = pd.read_parquet("./input/test.parquet")

    target_col = "time_to_approve"
    if target_col in train_df.columns:
        y_train = train_df[target_col].values.astype(np.float32)
        X_train_raw = train_df.drop(columns=[target_col]).copy()
    else:
        raise ValueError(f"Target column '{target_col}' not found in train.parquet")

    X_test_raw = test_df.copy()
    n_train = len(X_train_raw)
    n_test = len(X_test_raw)

    # Standardize column naming dictionary for robust matching
    col_map = {
        col.lower().strip().replace(" ", "_").replace("-", "_"): col
        for col in X_train_raw.columns
    }

    def find_col(possible_names):
        for name in possible_names:
            norm = name.lower().strip().replace(" ", "_").replace("-", "_")
            if norm in col_map:
                return col_map[norm]
        return None

    # Identify key column types dynamically
    desc_col = find_col(["description", "permit_description"])
    est_cost_col = find_col(["estimated_cost", "est_cost", "cost"])
    rev_cost_col = find_col(["revised_cost", "rev_cost"])
    exist_stories_col = find_col(["number_of_existing_stories", "existing_stories"])
    prop_stories_col = find_col(["number_of_proposed_stories", "proposed_stories"])
    exist_units_col = find_col(["existing_units", "number_of_existing_units"])
    prop_units_col = find_col(["proposed_units", "number_of_proposed_units"])
    exist_use_col = find_col(["existing_use", "existing_use_code"])
    prop_use_col = find_col(["proposed_use", "proposed_use_code"])
    permit_type_col = find_col(["permit_type", "permit_type_definition"])
    neighborhood_col = find_col(
        ["neighborhoods_analysis_boundaries", "neighborhood", "neighborhoods"]
    )
    zipcode_col = find_col(["zipcode", "zip_code", "postal_code"])
    block_col = find_col(["block"])
    lot_col = find_col(["lot"])

    # 2. Extract Tabular Features across full dataset
    full_df = pd.concat([X_train_raw, X_test_raw], axis=0, ignore_index=True)
    feat_df = pd.DataFrame(index=full_df.index)

    # Category Interactions
    if permit_type_col and prop_use_col:
        full_df["permit_type_x_prop_use"] = (
            full_df[permit_type_col].fillna("MISSING").astype(str)
            + "___"
            + full_df[prop_use_col].fillna("MISSING").astype(str)
        )
    if permit_type_col and neighborhood_col:
        full_df["permit_type_x_neighborhood"] = (
            full_df[permit_type_col].fillna("MISSING").astype(str)
            + "___"
            + full_df[neighborhood_col].fillna("MISSING").astype(str)
        )
    if prop_use_col and neighborhood_col:
        full_df["prop_use_x_neighborhood"] = (
            full_df[prop_use_col].fillna("MISSING").astype(str)
            + "___"
            + full_df[neighborhood_col].fillna("MISSING").astype(str)
        )

    # Date & Temporal Features
    date_cols = [c for c in full_df.columns if "date" in c.lower()]
    parsed_dates = {}
    for d_col in date_cols:
        parsed_dt = pd.to_datetime(full_df[d_col], errors="coerce")
        parsed_dates[d_col] = parsed_dt
        prefix = d_col.lower().replace(" ", "_")
        feat_df[f"{prefix}_year"] = parsed_dt.dt.year.fillna(-1).astype(np.float32)
        feat_df[f"{prefix}_month"] = parsed_dt.dt.month.fillna(-1).astype(np.float32)
        feat_df[f"{prefix}_day"] = parsed_dt.dt.day.fillna(-1).astype(np.float32)
        feat_df[f"{prefix}_dayofweek"] = parsed_dt.dt.dayofweek.fillna(-1).astype(
            np.float32
        )
        feat_df[f"{prefix}_dayofyear"] = parsed_dt.dt.dayofyear.fillna(-1).astype(
            np.float32
        )
        feat_df[f"{prefix}_quarter"] = parsed_dt.dt.quarter.fillna(-1).astype(
            np.float32
        )
        feat_df[f"{prefix}_is_weekend"] = (parsed_dt.dt.dayofweek >= 5).astype(
            np.float32
        )
        feat_df[f"{prefix}_sin_month"] = np.sin(
            2 * np.pi * feat_df[f"{prefix}_month"] / 12.0
        )
        feat_df[f"{prefix}_cos_month"] = np.cos(
            2 * np.pi * feat_df[f"{prefix}_month"] / 12.0
        )
        feat_df[f"{prefix}_sin_dow"] = np.sin(
            2 * np.pi * feat_df[f"{prefix}_dayofweek"] / 7.0
        )
        feat_df[f"{prefix}_cos_dow"] = np.cos(
            2 * np.pi * feat_df[f"{prefix}_dayofweek"] / 7.0
        )

        min_date = pd.Timestamp("1980-01-01")
        feat_df[f"{prefix}_elapsed_days"] = (
            parsed_dt - min_date
        ).dt.total_seconds() / (24 * 3600)
        feat_df[f"{prefix}_elapsed_days"] = (
            feat_df[f"{prefix}_elapsed_days"].fillna(-1).astype(np.float32)
        )

    # Pairwise Date Difference Intervals
    for i in range(len(date_cols)):
        for j in range(i + 1, len(date_cols)):
            col1 = date_cols[i]
            col2 = date_cols[j]
            p1 = col1.lower().replace(" ", "_")
            p2 = col2.lower().replace(" ", "_")
            diff = (parsed_dates[col1] - parsed_dates[col2]).dt.total_seconds() / (24 * 3600)
            feat_df[f"datediff_{p1}_minus_{p2}"] = diff.fillna(-99999).astype(np.float32)
            feat_df[f"datediff_abs_{p1}_{p2}"] = np.abs(diff).fillna(-99999).astype(np.float32)

    # Financial Complexity Features
    if est_cost_col and rev_cost_col:
        est_cost = (
            pd.to_numeric(full_df[est_cost_col], errors="coerce").fillna(0).values
        )
        rev_cost = (
            pd.to_numeric(full_df[rev_cost_col], errors="coerce").fillna(0).values
        )
        feat_df["est_cost_raw"] = est_cost
        feat_df["rev_cost_raw"] = rev_cost
        feat_df["log_est_cost"] = np.log1p(np.maximum(0, est_cost))
        feat_df["log_rev_cost"] = np.log1p(np.maximum(0, rev_cost))
        feat_df["cost_diff"] = rev_cost - est_cost
        feat_df["cost_ratio"] = (rev_cost + 1.0) / (est_cost + 1.0)
        feat_df["has_revised_cost"] = (rev_cost > 0).astype(np.float32)
        feat_df["est_cost_is_na"] = full_df[est_cost_col].isna().astype(np.float32)
        feat_df["rev_cost_is_na"] = full_df[rev_cost_col].isna().astype(np.float32)
    elif est_cost_col:
        est_cost = (
            pd.to_numeric(full_df[est_cost_col], errors="coerce").fillna(0).values
        )
        feat_df["est_cost_raw"] = est_cost
        feat_df["log_est_cost"] = np.log1p(np.maximum(0, est_cost))

    # Structural Project Features
    if exist_stories_col and prop_stories_col:
        ex_stories = (
            pd.to_numeric(full_df[exist_stories_col], errors="coerce").fillna(0).values
        )
        pr_stories = (
            pd.to_numeric(full_df[prop_stories_col], errors="coerce").fillna(0).values
        )
        feat_df["exist_stories"] = ex_stories
        feat_df["prop_stories"] = pr_stories
        feat_df["story_delta"] = pr_stories - ex_stories
        feat_df["story_ratio"] = (pr_stories + 1.0) / (ex_stories + 1.0)
        feat_df["is_story_increase"] = (pr_stories > ex_stories).astype(np.float32)

    if exist_units_col and prop_units_col:
        ex_units = (
            pd.to_numeric(full_df[exist_units_col], errors="coerce").fillna(0).values
        )
        pr_units = (
            pd.to_numeric(full_df[prop_units_col], errors="coerce").fillna(0).values
        )
        feat_df["exist_units"] = ex_units
        feat_df["prop_units"] = pr_units
        feat_df["unit_delta"] = pr_units - ex_units
        feat_df["unit_ratio"] = (pr_units + 1.0) / (ex_units + 1.0)
        feat_df["is_unit_increase"] = (pr_units > ex_units).astype(np.float32)

    # Cost per structural unit
    if est_cost_col and prop_stories_col:
        feat_df["cost_per_prop_story"] = feat_df["log_est_cost"] / (
            feat_df["prop_stories"] + 1.0
        )
    if est_cost_col and prop_units_col:
        feat_df["cost_per_prop_unit"] = feat_df["log_est_cost"] / (
            feat_df["prop_units"] + 1.0
        )

    # Use Category Transitions
    if exist_use_col and prop_use_col:
        ex_u = full_df[exist_use_col].fillna("MISSING").astype(str)
        pr_u = full_df[prop_use_col].fillna("MISSING").astype(str)
        feat_df["is_use_changed"] = (ex_u != pr_u).astype(np.float32)

    # Generic Numeric Columns
    numeric_cols = full_df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        norm_name = col.lower().strip().replace(" ", "_")
        if norm_name not in feat_df.columns:
            feat_df[f"num_{norm_name}"] = (
                full_df[col].fillna(full_df[col].median()).astype(np.float32)
            )
            feat_df[f"num_{norm_name}_isna"] = full_df[col].isna().astype(np.float32)

    # Categorical Frequency Encodings & Label Encodings
    cat_cols = full_df.select_dtypes(include=["object", "category"]).columns.tolist()
    if desc_col and desc_col in cat_cols:
        cat_cols.remove(desc_col)
    for d_col in date_cols:
        if d_col in cat_cols:
            cat_cols.remove(d_col)

    for col in cat_cols:
        norm_name = col.lower().strip().replace(" ", "_")
        series_str = full_df[col].fillna("__MISSING__").astype(str)
        freq_map = series_str.value_counts(normalize=True).to_dict()
        feat_df[f"freq_{norm_name}"] = (
            series_str.map(freq_map).fillna(0).astype(np.float32)
        )
        feat_df[f"cat_{norm_name}"] = pd.factorize(series_str)[0].astype(np.int32)
        feat_df[f"isna_{norm_name}"] = full_df[col].isna().astype(np.float32)

    # 3. Text NLP with TF-IDF + SVD (fitted strictly on training text) & Domain Keywords
    if desc_col and desc_col in full_df.columns:
        desc_text = full_df[desc_col].fillna("").astype(str)
        desc_lower = desc_text.str.lower()

        feat_df["desc_char_len"] = desc_text.str.len().astype(np.float32)
        feat_df["desc_word_count"] = desc_text.apply(lambda s: len(s.split())).astype(
            np.float32
        )
        feat_df["desc_digit_count"] = desc_text.apply(
            lambda s: sum(c.isdigit() for c in s)
        ).astype(np.float32)
        feat_df["desc_upper_count"] = desc_text.apply(
            lambda s: sum(c.isupper() for c in s)
        ).astype(np.float32)

        # Domain keyword indicators
        domain_keywords = [
            "otc", "over the counter", "over_the_counter", "seismic", "demolition",
            "tenant improvement", "remodel", "renovation", "electrical", "plumbing",
            "fire", "roof", "solar", "sign", "kitchen", "bath", "foundation",
            "addition", "alteration", "repair", "replace", "commercial", "residential", "deck"
        ]
        for kw in domain_keywords:
            kw_clean = kw.replace(" ", "_").replace("-", "_")
            feat_df[f"kw_{kw_clean}"] = desc_lower.str.contains(kw, regex=False).astype(np.float32)

        train_text = desc_text.iloc[:n_train]
        test_text = desc_text.iloc[n_train:]

        tfidf = TfidfVectorizer(
            max_features=10000, ngram_range=(1, 2), min_df=3, sublinear_tf=True
        )
        svd = TruncatedSVD(n_components=64, random_state=42)

        tfidf_train = tfidf.fit_transform(train_text)
        tfidf_test = tfidf.transform(test_text)

        svd_train = svd.fit_transform(tfidf_train)
        svd_test = svd.transform(tfidf_test)

        for k in range(64):
            feat_df[f"tfidf_svd_{k}"] = np.concatenate(
                [svd_train[:, k], svd_test[:, k]]
            ).astype(np.float32)

    # 4. Multi-level Group Aggregations (computed strictly on training slice)
    group_keys = [
        k for k in [
            permit_type_col,
            prop_use_col,
            neighborhood_col,
            "permit_type_x_prop_use",
            "permit_type_x_neighborhood",
        ] if k and k in full_df.columns
    ]
    num_metrics = [
        m for m in ["log_est_cost", "cost_ratio", "story_delta", "unit_delta"] if m in feat_df.columns
    ]

    for g_key in group_keys:
        norm_g = g_key.lower().strip().replace(" ", "_")
        g_series_train = full_df[g_key].iloc[:n_train].fillna("__MISSING__").astype(str)
        g_series_full = full_df[g_key].fillna("__MISSING__").astype(str)

        for num_m in num_metrics:
            train_num = feat_df[num_m].iloc[:n_train]
            temp_df = pd.DataFrame({"grp": g_series_train, "val": train_num})
            agg_stats = temp_df.groupby("grp")["val"].agg(["mean", "std", "min", "max"])

            mean_map = agg_stats["mean"].to_dict()
            std_map = agg_stats["std"].fillna(0).to_dict()
            min_map = agg_stats["min"].to_dict()
            max_map = agg_stats["max"].to_dict()

            global_mean = float(train_num.mean())
            global_std = float(train_num.std()) if len(train_num) > 1 else 0.0
            global_min = float(train_num.min())
            global_max = float(train_num.max())

            m_series = g_series_full.map(mean_map).fillna(global_mean).astype(np.float32)
            s_series = g_series_full.map(std_map).fillna(global_std).astype(np.float32)
            min_series = g_series_full.map(min_map).fillna(global_min).astype(np.float32)
            max_series = g_series_full.map(max_map).fillna(global_max).astype(np.float32)

            feat_df[f"{num_m}_mean_by_{norm_g}"] = m_series
            feat_df[f"{num_m}_std_by_{norm_g}"] = s_series
            feat_df[f"{num_m}_min_by_{norm_g}"] = min_series
            feat_df[f"{num_m}_max_by_{norm_g}"] = max_series
            feat_df[f"{num_m}_diff_mean_{norm_g}"] = feat_df[num_m] - m_series
            feat_df[f"{num_m}_ratio_mean_{norm_g}"] = (feat_df[num_m] + 1.0) / (m_series + 1.0)

    # 4. Separate back into Train and Test
    X_train_df = feat_df.iloc[:n_train].copy().reset_index(drop=True)
    X_test_df = feat_df.iloc[n_train:].copy().reset_index(drop=True)

    # 5. Leak-Free Out-Of-Fold (OOF) Target Encodings
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    folds = np.zeros(n_train, dtype=np.int32)

    target_encoding_cols = [
        c
        for c in [
            permit_type_col,
            prop_use_col,
            neighborhood_col,
            zipcode_col,
            block_col,
            "permit_type_x_prop_use",
            "permit_type_x_neighborhood",
            "prop_use_x_neighborhood",
        ]
        if c and c in full_df.columns
    ]

    for fold_idx, (trn_idx, val_idx) in enumerate(kf.split(X_train_df, y_train)):
        folds[val_idx] = fold_idx

    for raw_c in target_encoding_cols:
        norm_name = raw_c.lower().strip().replace(" ", "_")
        series_train = full_df[raw_c].iloc[:n_train].fillna("__MISSING__").astype(str)
        series_test = full_df[raw_c].iloc[n_train:].fillna("__MISSING__").astype(str)

        oof_mean = np.zeros(n_train, dtype=np.float32)
        global_mean = float(np.mean(y_train))
        smoothing = 10.0

        for trn_idx, val_idx in kf.split(X_train_df, y_train):
            cat_trn = series_train.iloc[trn_idx]
            y_trn = y_train[trn_idx]

            stats = (
                pd.DataFrame({"cat": cat_trn, "y": y_trn})
                .groupby("cat")["y"]
                .agg(["count", "mean"])
            )
            smoothed_mean = (
                stats["count"] * stats["mean"] + smoothing * global_mean
            ) / (stats["count"] + smoothing)
            mean_map = smoothed_mean.to_dict()

            oof_mean[val_idx] = (
                series_train.iloc[val_idx].map(mean_map).fillna(global_mean).values
            )

        full_stats = (
            pd.DataFrame({"cat": series_train, "y": y_train})
            .groupby("cat")["y"]
            .agg(["count", "mean"])
        )
        full_smoothed = (
            full_stats["count"] * full_stats["mean"] + smoothing * global_mean
        ) / (full_stats["count"] + smoothing)
        test_mean = series_test.map(full_smoothed.to_dict()).fillna(global_mean).values

        X_train_df[f"oof_te_{norm_name}"] = oof_mean
        X_test_df[f"oof_te_{norm_name}"] = test_mean.astype(np.float32)

    # 6. Feature Matrix Preparation
    feature_cols = [c for c in X_test_df.columns if c in X_train_df.columns]
    X_train = (
        X_train_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0).values
    )
    X_test = X_test_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0).values

    n_samples = len(y_train)
    n_folds = int(np.max(folds) + 1)

    # 7. Model Training & Out-Of-Fold Predictions
    models_to_train = ["lgb", "xgb", "cat"]
    oof_predictions = {
        m: np.zeros(n_samples, dtype=np.float32) for m in models_to_train
    }
    test_predictions = {m: np.zeros(n_test, dtype=np.float32) for m in models_to_train}

    for fold in range(n_folds):
        trn_idx = np.where(folds != fold)[0]
        val_idx = np.where(folds == fold)[0]

        X_tr, y_tr = X_train[trn_idx], y_train[trn_idx]
        X_va, y_va = X_train[val_idx], y_train[val_idx]

        # Model 1: LightGBM
        lgb_model = lgb.LGBMRegressor(**get_lgb_params(seed=42 + fold))
        lgb_callbacks = [
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0),
        ]
        lgb_model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=lgb_callbacks,
        )
        oof_predictions["lgb"][val_idx] = lgb_model.predict(X_va)
        test_predictions["lgb"] += lgb_model.predict(X_test) / n_folds

        # Model 2: XGBoost
        xgb_model = xgb.XGBRegressor(**get_xgb_params(seed=42 + fold))
        xgb_model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            verbose=False,
        )
        oof_predictions["xgb"][val_idx] = xgb_model.predict(X_va)
        test_predictions["xgb"] += xgb_model.predict(X_test) / n_folds

        # Model 3: CatBoost
        cat_model = CatBoostRegressor(**get_catboost_params(seed=42 + fold))
        cat_model.fit(
            X_tr,
            y_tr,
            eval_set=(X_va, y_va),
            verbose=False,
        )
        oof_predictions["cat"][val_idx] = cat_model.predict(X_va)
        test_predictions["cat"] += cat_model.predict(X_test) / n_folds

    # 8. Optimal Ensemble Blending (SLSQP Optimization on OOF predictions)
    def blend_objective(weights):
        weights = np.array(weights)
        weights = weights / np.sum(weights)
        blended_oof = np.zeros(n_samples, dtype=np.float32)
        for i, m in enumerate(models_to_train):
            blended_oof += weights[i] * oof_predictions[m]
        blended_oof = np.maximum(0, blended_oof)
        return np.sqrt(mean_squared_error(y_train, blended_oof))

    initial_weights = [1.0 / len(models_to_train)] * len(models_to_train)
    bounds = [(0, 1) for _ in models_to_train]
    constraints = {"type": "eq", "fun": lambda w: 1 - sum(w)}

    opt_result = minimize(
        blend_objective,
        initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    opt_weights = opt_result.x / np.sum(opt_result.x)

    oof_ensemble = np.zeros(n_samples, dtype=np.float32)
    final_test_preds = np.zeros(n_test, dtype=np.float32)

    for i, m in enumerate(models_to_train):
        oof_ensemble += opt_weights[i] * oof_predictions[m]
        final_test_preds += opt_weights[i] * test_predictions[m]

    # Target is strictly non-negative duration
    oof_ensemble = np.maximum(0, oof_ensemble)
    final_test_preds = np.maximum(0, final_test_preds)

    # 9. Save Submission File
    submission_path = "./submission/submission.csv"
    submission_df = pd.DataFrame(
        {
            "id": np.arange(n_test, dtype=np.int64),
            "time_to_approve": final_test_preds,
        }
    )
    submission_df.to_csv(submission_path, index=False)

    # 10. Print Final Validation Score (Exact RMSE)
    final_score = np.sqrt(mean_squared_error(y_train, oof_ensemble))
    print(f"Final Validation Score: {final_score}")


if __name__ == "__main__":
    main()
