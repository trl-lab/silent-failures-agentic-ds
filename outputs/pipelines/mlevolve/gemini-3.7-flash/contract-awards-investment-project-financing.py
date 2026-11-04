"""Contract Awards Investment Project Financing

End-to-End Predictive Pipeline with GBDT Ensemble & Scale Calibration.
Evaluation Metric: RMSE
"""

import os
import re
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Ridge
from sklearn.metrics import root_mean_squared_error
from scipy.optimize import minimize

import lightgbm as lgb
import xgboost as xgb
import catboost as cb

warnings.filterwarnings("ignore")


# =========================================================================
# 1. Feature Engineering & Preprocessing
# =========================================================================


def engineer_features():
    print("=" * 70)
    print("1. Loading Data & Engineering Features")
    print("=" * 70)

    train_df = pd.read_parquet("./input/train.parquet")
    test_df = pd.read_parquet("./input/test.parquet")

    target_col = "Supplier Contract Amount (USD)"
    if target_col not in train_df.columns:
        matches = [
            c for c in train_df.columns if "supplier contract amount" in c.lower()
        ]
        if matches:
            target_col = matches[0]

    # Filter invalid target rows from train
    initial_len = len(train_df)
    train_df = train_df.dropna(subset=[target_col]).reset_index(drop=True)
    train_df = train_df[train_df[target_col] > 0].reset_index(drop=True)
    print(f"Train samples filtered: {initial_len:,} -> {len(train_df):,}")

    train_df["target"] = train_df[target_col].astype(np.float64)
    train_df["target_log1p"] = np.log1p(train_df["target"])

    # 5-Fold Stratified Validation Split based on target quantiles
    n_splits = 5
    num_bins = 20
    train_df["target_bin"] = pd.qcut(
        train_df["target_log1p"], q=num_bins, labels=False, duplicates="drop"
    )

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    train_df["fold"] = -1
    for fold, (_, val_idx) in enumerate(skf.split(train_df, train_df["target_bin"])):
        train_df.loc[val_idx, "fold"] = fold
    train_df = train_df.drop(columns=["target_bin"])

    # Dynamic Column Categorization
    exclude_cols = {target_col, "target", "target_log1p", "fold", "id"}
    candidate_cols = [c for c in train_df.columns if c not in exclude_cols]

    date_cols = []
    text_cols = []
    cat_cols = []
    num_cols = []

    for c in candidate_cols:
        if pd.api.types.is_datetime64_any_dtype(train_df[c]):
            date_cols.append(c)
        elif pd.api.types.is_numeric_dtype(train_df[c]):
            num_cols.append(c)
        elif pd.api.types.is_string_dtype(train_df[c]) or pd.api.types.is_object_dtype(
            train_df[c]
        ):
            sample_non_null = train_df[c].dropna().head(50)
            is_date = False
            if len(sample_non_null) > 0 and (
                "date" in c.lower() or "year" in c.lower()
            ):
                try:
                    pd.to_datetime(sample_non_null, errors="raise")
                    is_date = True
                except Exception:
                    is_date = False
            if is_date:
                date_cols.append(c)
            else:
                avg_len = (
                    sample_non_null.astype(str).str.len().mean()
                    if len(sample_non_null) > 0
                    else 0
                )
                if avg_len > 60 or "description" in c.lower() or "name" in c.lower():
                    text_cols.append(c)
                else:
                    cat_cols.append(c)
        else:
            cat_cols.append(c)

    print(
        f"Features identified: Dates={len(date_cols)}, Text={len(text_cols)}, "
        f"Categorical={len(cat_cols)}, Numerical={len(num_cols)}"
    )

    # Date Feature Engineering
    parsed_date_cols = []
    ref_date = pd.Timestamp("1990-01-01")
    for dc in date_cols:
        col_clean = f"dt_{re.sub(r'[^a-zA-Z0-9]', '_', dc)}"
        train_dt = pd.to_datetime(train_df[dc], errors="coerce")
        test_dt = pd.to_datetime(test_df[dc], errors="coerce")

        for df, dt_series in [(train_df, train_dt), (test_df, test_dt)]:
            df[f"{col_clean}_year"] = dt_series.dt.year.astype("float32")
            df[f"{col_clean}_month"] = dt_series.dt.month.astype("float32")
            df[f"{col_clean}_day"] = dt_series.dt.day.astype("float32")
            df[f"{col_clean}_dayofweek"] = dt_series.dt.dayofweek.astype("float32")
            df[f"{col_clean}_dayofyear"] = dt_series.dt.dayofyear.astype("float32")
            df[f"{col_clean}_quarter"] = dt_series.dt.quarter.astype("float32")
            df[f"{col_clean}_is_month_end"] = dt_series.dt.is_month_end.astype(
                "float32"
            )
            df[f"{col_clean}_days_since_ref"] = (dt_series - ref_date).dt.days.astype(
                "float32"
            )
        parsed_date_cols.append(col_clean)

    if len(parsed_date_cols) >= 2:
        for i in range(len(parsed_date_cols)):
            for j in range(i + 1, len(parsed_date_cols)):
                d1 = f"{parsed_date_cols[i]}_days_since_ref"
                d2 = f"{parsed_date_cols[j]}_days_since_ref"
                diff_col = f"diff_days_{parsed_date_cols[i]}_vs_{parsed_date_cols[j]}"
                train_df[diff_col] = train_df[d1] - train_df[d2]
                test_df[diff_col] = test_df[d1] - test_df[d2]

    # Domain Indicator: Domestic vs Cross-border Contractor
    borrower_col = next(
        (c for c in candidate_cols if "borrower country" in c.lower()), None
    )
    supplier_col = next(
        (c for c in candidate_cols if "supplier country" in c.lower()), None
    )
    if borrower_col and supplier_col:
        train_df["is_domestic_supplier"] = (
            train_df[borrower_col].astype(str).str.strip().str.lower()
            == train_df[supplier_col].astype(str).str.strip().str.lower()
        ).astype("float32")
        test_df["is_domestic_supplier"] = (
            test_df[borrower_col].astype(str).str.strip().str.lower()
            == test_df[supplier_col].astype(str).str.strip().str.lower()
        ).astype("float32")

    # High-Order Cross-Categorical Interactions
    interaction_candidates = [
        ("borrower country", "supplier country"),
        ("procurement category", "procurement method"),
        ("procurement group", "procurement method"),
        ("major sector", "procurement method"),
        ("borrower country", "procurement category"),
    ]
    for k1, k2 in interaction_candidates:
        col1 = next((c for c in candidate_cols if k1 in c.lower()), None)
        col2 = next((c for c in candidate_cols if k2 in c.lower()), None)
        if col1 and col2 and col1 != col2:
            inter_col = f"inter_{re.sub(r'[^a-zA-Z0-9]', '_', col1)}_x_{re.sub(r'[^a-zA-Z0-9]', '_', col2)}"
            train_df[inter_col] = (
                train_df[col1].astype(str) + "___" + train_df[col2].astype(str)
            )
            test_df[inter_col] = (
                test_df[col1].astype(str) + "___" + test_df[col2].astype(str)
            )
            cat_cols.append(inter_col)

    # Group Aggregation Statistics (Count and Smoothed Target Mean per Project and Sector)
    group_keywords = [
        "project",
        "sector",
        "borrower country",
        "supplier country",
        "procurement method",
    ]
    global_target_mean = train_df["target_log1p"].mean()
    smooth_weight = 10.0
    for kw in group_keywords:
        grp_col = next(
            (c for c in candidate_cols if kw in c.lower() and "date" not in c.lower()),
            None,
        )
        if grp_col:
            clean_grp = f"grp_{re.sub(r'[^a-zA-Z0-9]', '_', grp_col)}"
            vc = train_df[grp_col].astype(str).value_counts()
            train_df[f"{clean_grp}_count"] = (
                train_df[grp_col].astype(str).map(vc).fillna(0).astype("float32")
            )
            test_df[f"{clean_grp}_count"] = (
                test_df[grp_col].astype(str).map(vc).fillna(0).astype("float32")
            )

            # Smoothed Out-Of-Fold Group Target Mean
            train_df[f"{clean_grp}_mean"] = np.nan
            for fold in range(n_splits):
                tr_idx = train_df["fold"] != fold
                va_idx = train_df["fold"] == fold
                fold_stats = (
                    train_df.loc[tr_idx]
                    .groupby(train_df.loc[tr_idx, grp_col].astype(str))["target_log1p"]
                    .agg(["count", "mean"])
                )
                smoothed_val = (
                    fold_stats["count"] * fold_stats["mean"]
                    + smooth_weight * global_target_mean
                ) / (fold_stats["count"] + smooth_weight)
                target_map = smoothed_val.to_dict()
                train_df.loc[va_idx, f"{clean_grp}_mean"] = (
                    train_df.loc[va_idx, grp_col]
                    .astype(str)
                    .map(target_map)
                    .fillna(global_target_mean)
                    .astype("float32")
                )

            all_stats = train_df.groupby(train_df[grp_col].astype(str))["target_log1p"].agg(
                ["count", "mean"]
            )
            all_smoothed = (
                all_stats["count"] * all_stats["mean"]
                + smooth_weight * global_target_mean
            ) / (all_stats["count"] + smooth_weight)
            test_df[f"{clean_grp}_mean"] = (
                test_df[grp_col]
                .astype(str)
                .map(all_smoothed.to_dict())
                .fillna(global_target_mean)
                .astype("float32")
            )

    # Multi-Text Feature Engineering (TF-IDF + TruncatedSVD + OOF Ridge Meta-Features)
    for tc in text_cols:
        prefix = f"txt_{re.sub(r'[^a-zA-Z0-9]', '_', tc)}"
        for df in [train_df, test_df]:
            text_series = df[tc].fillna("").astype(str)
            df[f"{prefix}_char_len"] = text_series.str.len().astype("float32")
            df[f"{prefix}_word_count"] = (
                text_series.str.split().str.len().astype("float32")
            )
            df[f"{prefix}_digit_count"] = text_series.apply(
                lambda s: sum(c.isdigit() for c in s)
            ).astype("float32")
            df[f"{prefix}_upper_count"] = text_series.apply(
                lambda s: sum(c.isupper() for c in s)
            ).astype("float32")

        train_text = train_df[tc].fillna("").astype(str)
        test_text = test_df[tc].fillna("").astype(str)

        tfidf = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=15000,
            sublinear_tf=True,
            min_df=3,
            strip_accents="unicode",
        )
        train_tfidf = tfidf.fit_transform(train_text)
        test_tfidf = tfidf.transform(test_text)

        # OOF Ridge Meta-Features on target_log1p
        train_df[f"{prefix}_ridge_oof"] = np.nan
        test_ridge_preds = np.zeros(len(test_df), dtype=np.float32)
        for fold in range(n_splits):
            tr_idx = train_df["fold"] != fold
            va_idx = train_df["fold"] == fold
            ridge_text = Ridge(alpha=5.0, random_state=42)
            ridge_text.fit(
                train_tfidf[tr_idx], train_df.loc[tr_idx, "target_log1p"]
            )
            train_df.loc[va_idx, f"{prefix}_ridge_oof"] = ridge_text.predict(
                train_tfidf[va_idx]
            ).astype("float32")
            test_ridge_preds += (
                ridge_text.predict(test_tfidf).astype("float32") / n_splits
            )
        test_df[f"{prefix}_ridge_oof"] = test_ridge_preds

        n_components = min(16, max(2, train_tfidf.shape[1] - 1))
        if train_tfidf.shape[1] > 2:
            svd = TruncatedSVD(n_components=n_components, random_state=42)
            train_svd = svd.fit_transform(train_tfidf)
            test_svd = svd.transform(test_tfidf)

            for k in range(n_components):
                train_df[f"{prefix}_svd_{k}"] = train_svd[:, k].astype("float32")
                test_df[f"{prefix}_svd_{k}"] = test_svd[:, k].astype("float32")

    # Categorical Encodings (Frequency & Out-Of-Fold Smoothed Target Encoding)
    smooth_weight = 10.0
    global_target_mean = train_df["target_log1p"].mean()

    for cc in cat_cols:
        clean_cc = f"cat_{re.sub(r'[^a-zA-Z0-9]', '_', cc)}"

        freq_map = train_df[cc].astype(str).value_counts(dropna=False).to_dict()
        train_df[f"{clean_cc}_freq"] = (
            train_df[cc].astype(str).map(freq_map).fillna(0).astype("float32")
        )
        test_df[f"{clean_cc}_freq"] = (
            test_df[cc].astype(str).map(freq_map).fillna(0).astype("float32")
        )

        train_df[f"{clean_cc}_target_enc"] = np.nan
        for fold in range(n_splits):
            tr_idx = train_df["fold"] != fold
            va_idx = train_df["fold"] == fold

            fold_stats = (
                train_df.loc[tr_idx]
                .groupby(train_df.loc[tr_idx, cc].astype(str))["target_log1p"]
                .agg(["count", "mean"])
            )
            smoothed_val = (
                fold_stats["count"] * fold_stats["mean"]
                + smooth_weight * global_target_mean
            ) / (fold_stats["count"] + smooth_weight)
            target_map = smoothed_val.to_dict()

            train_df.loc[va_idx, f"{clean_cc}_target_enc"] = (
                train_df.loc[va_idx, cc]
                .astype(str)
                .map(target_map)
                .fillna(global_target_mean)
                .astype("float32")
            )

        all_stats = train_df.groupby(train_df[cc].astype(str))["target_log1p"].agg(
            ["count", "mean"]
        )
        all_smoothed = (
            all_stats["count"] * all_stats["mean"] + smooth_weight * global_target_mean
        ) / (all_stats["count"] + smooth_weight)
        all_target_map = all_smoothed.to_dict()
        test_df[f"{clean_cc}_target_enc"] = (
            test_df[cc]
            .astype(str)
            .map(all_target_map)
            .fillna(global_target_mean)
            .astype("float32")
        )

        # Standardize categorical column format
        train_df[cc] = train_df[cc].fillna("missing").astype(str).astype("category")
        test_df[cc] = test_df[cc].fillna("missing").astype(str).astype("category")

    # Numerical Missingness Flags
    for nc in num_cols:
        train_df[f"{nc}_isna"] = train_df[nc].isna().astype("float32")
        test_df[f"{nc}_isna"] = test_df[nc].isna().astype("float32")

    # Drop raw unparsed dates and text columns
    drop_raw_dates = [c for c in date_cols if c in train_df.columns]
    drop_raw_texts = [c for c in text_cols if c in train_df.columns]
    train_df = train_df.drop(columns=drop_raw_dates + drop_raw_texts)
    test_df = test_df.drop(columns=drop_raw_dates + drop_raw_texts)

    print(
        f"Feature engineering completed: Train shape = {train_df.shape}, Test shape = {test_df.shape}"
    )
    return train_df, test_df, cat_cols, target_col


# =========================================================================
# 2. Model Training, Blending, Calibration & Evaluation
# =========================================================================


def train_and_evaluate(train_df, test_df, cat_cols, target_col):
    print("=" * 70)
    print("2. Multi-Model Cross-Validation & Calibration")
    print("=" * 70)

    y_raw = train_df["target"].values.astype(np.float64)
    y_log = train_df["target_log1p"].values.astype(np.float32)
    folds = train_df["fold"].values

    exclude_cols = {
        target_col,
        "target",
        "target_log1p",
        "fold",
        "id",
        "target_bin",
    }
    feature_cols = [c for c in train_df.columns if c not in exclude_cols]

    X = train_df[feature_cols].copy()
    X_test = test_df[feature_cols].copy()

    # Align categorical features across train and test
    for c in cat_cols:
        if c in feature_cols:
            all_cats = pd.concat([X[c], X_test[c]]).unique()
            X[c] = pd.Categorical(X[c], categories=all_cats)
            X_test[c] = pd.Categorical(X_test[c], categories=all_cats)

    n_splits = len(np.unique(folds))
    n_test = len(test_df)

    oof_lgb = np.zeros(len(train_df), dtype=np.float32)
    test_lgb = np.zeros(n_test, dtype=np.float32)

    oof_xgb = np.zeros(len(train_df), dtype=np.float32)
    test_xgb = np.zeros(n_test, dtype=np.float32)

    oof_cb = np.zeros(len(train_df), dtype=np.float32)
    test_cb = np.zeros(n_test, dtype=np.float32)

    # Convert categoricals to strings for CatBoost compatibility
    X_cb = X.copy()
    X_test_cb = X_test.copy()
    active_cat_cols = [c for c in cat_cols if c in feature_cols]
    for c in active_cat_cols:
        X_cb[c] = X_cb[c].astype(str)
        X_test_cb[c] = X_test_cb[c].astype(str)

    for fold in range(n_splits):
        tr_mask = folds != fold
        va_mask = folds == fold

        X_tr, y_tr = X.loc[tr_mask], y_log[tr_mask]
        X_va, y_va = X.loc[va_mask], y_log[va_mask]

        # Model A: LightGBM Regressor
        lgb_params = {
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "n_estimators": 2500,
            "learning_rate": 0.03,
            "num_leaves": 47,
            "max_depth": 7,
            "min_child_samples": 25,
            "subsample": 0.85,
            "subsample_freq": 1,
            "colsample_bytree": 0.75,
            "reg_alpha": 0.5,
            "reg_lambda": 1.5,
            "random_state": 42 + fold,
            "n_jobs": -1,
            "verbose": -1,
        }
        model_lgb = lgb.LGBMRegressor(**lgb_params)
        model_lgb.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )
        oof_lgb[va_mask] = model_lgb.predict(X_va)
        test_lgb += model_lgb.predict(X_test) / n_splits

        # Model B: XGBoost Regressor
        xgb_params = {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "tree_method": "hist",
            "n_estimators": 2500,
            "learning_rate": 0.03,
            "max_depth": 6,
            "min_child_weight": 5,
            "subsample": 0.85,
            "colsample_bytree": 0.75,
            "colsample_bylevel": 0.85,
            "reg_alpha": 0.5,
            "reg_lambda": 2.0,
            "enable_categorical": True,
            "random_state": 42 + fold,
            "n_jobs": -1,
            "early_stopping_rounds": 50,
        }
        model_xgb = xgb.XGBRegressor(**xgb_params)
        model_xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        oof_xgb[va_mask] = model_xgb.predict(X_va)
        test_xgb += model_xgb.predict(X_test) / n_splits

        # Model C: CatBoost Regressor
        cb_params = {
            "loss_function": "RMSE",
            "eval_metric": "RMSE",
            "iterations": 2500,
            "learning_rate": 0.035,
            "depth": 6,
            "l2_leaf_reg": 3.0,
            "random_strength": 0.5,
            "bagging_temperature": 0.2,
            "random_seed": 42 + fold,
            "thread_count": -1,
            "early_stopping_rounds": 50,
            "verbose": False,
        }
        model_cb = cb.CatBoostRegressor(**cb_params)
        model_cb.fit(
            X_cb.loc[tr_mask],
            y_tr,
            cat_features=active_cat_cols if len(active_cat_cols) > 0 else None,
            eval_set=(X_cb.loc[va_mask], y_va),
            verbose=False,
        )
        oof_cb[va_mask] = model_cb.predict(X_cb.loc[va_mask])
        test_cb += model_cb.predict(X_test_cb) / n_splits

        fold_rmse = root_mean_squared_error(y_raw[va_mask], np.expm1(oof_lgb[va_mask]))
        print(f"Fold {fold+1}/{n_splits} complete | Fold LGB RMSE: {fold_rmse:,.2f}")

    # Optimal Linear Blending in Log Space
    oof_matrix = np.column_stack([oof_lgb, oof_xgb, oof_cb])
    meta_model = Ridge(alpha=10.0, positive=True, fit_intercept=False)
    meta_model.fit(oof_matrix, y_log)

    weights = meta_model.coef_
    if weights.sum() > 0:
        weights = weights / weights.sum()
    else:
        weights = np.array([1 / 3, 1 / 3, 1 / 3])

    print(
        f"Ensemble Blend Weights: LGB={weights[0]:.3f}, XGB={weights[1]:.3f}, CatBoost={weights[2]:.3f}"
    )

    oof_blend_log = weights[0] * oof_lgb + weights[1] * oof_xgb + weights[2] * oof_cb
    test_blend_log = (
        weights[0] * test_lgb + weights[1] * test_xgb + weights[2] * test_cb
    )

    # Scale Calibration for Raw RMSE Metric
    uncalibrated_oof_raw = np.clip(np.expm1(oof_blend_log), 0, None)
    alpha_optimal = np.sum(uncalibrated_oof_raw * y_raw) / np.sum(
        uncalibrated_oof_raw**2 + 1e-8
    )
    print(f"Optimal Out-Of-Fold Scale Multiplier alpha*: {alpha_optimal:.4f}")

    oof_preds_calibrated = np.clip(uncalibrated_oof_raw * alpha_optimal, 0, None)
    uncalibrated_test_raw = np.clip(np.expm1(test_blend_log), 0, None)
    test_preds_calibrated = np.clip(uncalibrated_test_raw * alpha_optimal, 0, None)

    final_rmse = root_mean_squared_error(y_raw, oof_preds_calibrated)
    print(f"Calibrated Ensemble Out-Of-Fold RMSE: {final_rmse:,.4f}")

    # Generate Submission
    os.makedirs("./submission", exist_ok=True)
    submission_path = "./submission/submission.csv"

    if "id" in test_df.columns:
        test_ids = test_df["id"].values
    else:
        test_ids = np.arange(len(test_df))

    sub_col_name = "Supplier Contract Amount (USD)"
    submission_df = pd.DataFrame({"id": test_ids, sub_col_name: test_preds_calibrated})

    assert len(submission_df) == len(test_df), "Mismatch in test submission count."
    assert list(submission_df.columns) == ["id", sub_col_name]
    assert not submission_df[sub_col_name].isna().any(), "NaN in predictions."

    submission_df.to_csv(submission_path, index=False)
    print(f"Saved submission file to {submission_path}")

    # REQUIRED FINAL LINE
    print(f"Final Validation Score: {final_rmse}")


def main():
    train_df, test_df, cat_cols, target_col = engineer_features()
    train_and_evaluate(train_df, test_df, cat_cols, target_col)


if __name__ == "__main__":
    main()
