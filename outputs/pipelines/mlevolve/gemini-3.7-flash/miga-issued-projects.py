import json
import math
import os
import re
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.optimize import minimize
from sklearn.decomposition import TruncatedSVD
import warnings
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, HuberRegressor, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")
from catboost import CatBoostRegressor
import lightgbm as lgb
import xgboost as xgb


# ============================================================================
# 2. End-to-End Feature Engineering Pipeline
# ============================================================================


def run_feature_engineering():
    os.makedirs("./working", exist_ok=True)
    os.makedirs("./submission", exist_ok=True)

    train_path = "./input/train.parquet"
    test_path = "./input/test.parquet"

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    target_col = "Gross Issued Amount (US$)"
    if target_col not in train_df.columns:
        for col in train_df.columns:
            if "gross" in col.lower() and "amount" in col.lower():
                target_col = col
                break

    train_df = train_df.dropna(subset=[target_col]).reset_index(drop=True)
    n_train = len(train_df)
    n_test = len(test_df)

    y_train = train_df[target_col].values.astype(np.float64)
    train_features = train_df.drop(columns=[target_col])
    test_features = test_df.copy()

    test_ids = np.arange(n_test, dtype=np.int64)
    if "id" in test_df.columns:
        test_ids = test_df["id"].values
    elif "ID" in test_df.columns:
        test_ids = test_df["ID"].values

    combined_df = pd.concat([train_features, test_features], axis=0).reset_index(
        drop=True
    )

    date_cols = []
    for col in combined_df.columns:
        if combined_df[col].dtype == "object" or np.issubdtype(
            combined_df[col].dtype, np.datetime64
        ):
            is_date_name = any(
                kw in col.lower()
                for kw in [
                    "date",
                    "year",
                    "fiscal",
                    "effective",
                    "expiry",
                    "maturity",
                    "issued",
                ]
            )
            sample_non_null = combined_df[col].dropna().astype(str).head(50)
            if is_date_name or (
                len(sample_non_null) > 0
                and sample_non_null.str.contains(
                    r"\d{4}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4}", regex=True
                ).mean()
                > 0.5
            ):
                try:
                    parsed_series = pd.to_datetime(combined_df[col], errors="coerce")
                    if parsed_series.notnull().mean() > 0.2:
                        date_cols.append(col)
                        combined_df[col + "_parsed"] = parsed_series
                except Exception:
                    pass

    for col in date_cols:
        parsed = combined_df[col + "_parsed"]
        combined_df[col + "_year"] = parsed.dt.year.astype(np.float32)
        combined_df[col + "_month"] = parsed.dt.month.astype(np.float32)
        combined_df[col + "_quarter"] = parsed.dt.quarter.astype(np.float32)
        combined_df[col + "_day"] = parsed.dt.day.astype(np.float32)
        combined_df[col + "_dayofweek"] = parsed.dt.dayofweek.astype(np.float32)
        combined_df[col + "_dayofyear"] = parsed.dt.dayofyear.astype(np.float32)
        combined_df[col + "_is_month_end"] = parsed.dt.is_month_end.astype(np.float32)
        combined_df.drop(columns=[col + "_parsed"], inplace=True)

    parsed_date_cols = [c for c in date_cols if c + "_year" in combined_df.columns]
    duration_cols = []
    for i in range(len(parsed_date_cols)):
        for j in range(i + 1, len(parsed_date_cols)):
            col1 = parsed_date_cols[i]
            col2 = parsed_date_cols[j]
            dur_name = f"duration_days_{col1}_{col2}"
            combined_df[dur_name] = (
                combined_df[col2 + "_year"] - combined_df[col1 + "_year"]
            ) * 365.25 + (
                combined_df[col2 + "_dayofyear"] - combined_df[col1 + "_dayofyear"]
            )
            duration_cols.append(dur_name)

    for i in range(len(duration_cols)):
        for j in range(i + 1, len(duration_cols)):
            d1 = duration_cols[i]
            d2 = duration_cols[j]
            combined_df[f"ratio_{d1}_{d2}"] = (
                combined_df[d1].fillna(0.0) + 1.0
            ) / (combined_df[d2].fillna(0.0).abs() + 1.0)

    text_cols = []
    cat_cols = []
    num_cols = []

    for col in combined_df.columns:
        if (
            col.endswith(
                (
                    "_year",
                    "_month",
                    "_quarter",
                    "_day",
                    "_dayofweek",
                    "_dayofyear",
                    "_is_month_end",
                )
            )
            or col.startswith("duration_days_")
            or col.startswith("ratio_")
        ):
            num_cols.append(col)
            continue

        if (
            combined_df[col].dtype == "object"
            or combined_df[col].dtype.name == "category"
        ):
            unique_ratio = combined_df[col].nunique() / max(1, len(combined_df))
            avg_len = (
                combined_df[col].dropna().astype(str).str.len().mean()
                if combined_df[col].notnull().sum() > 0
                else 0
            )

            if avg_len > 40 or unique_ratio > 0.6:
                text_cols.append(col)
            else:
                cat_cols.append(col)
        elif np.issubdtype(combined_df[col].dtype, np.number):
            num_cols.append(col)

    # High-Order Categorical Interactions
    top_cats = sorted(
        cat_cols, key=lambda c: combined_df[c].nunique(), reverse=False
    )[:6]
    interaction_cols = []
    for i in range(len(top_cats)):
        for j in range(i + 1, len(top_cats)):
            c1, c2 = top_cats[i], top_cats[j]
            inter_col = f"inter_{c1}_x_{c2}"
            combined_df[inter_col] = (
                combined_df[c1].astype(str) + "___" + combined_df[c2].astype(str)
            )
            interaction_cols.append(inter_col)
    cat_cols.extend(interaction_cols)

    # 10-Fold Stratified Quantile Split
    n_splits = 10
    n_bins = min(10, max(2, len(y_train) // 20))
    binned_y = pd.qcut(y_train, q=n_bins, labels=False, duplicates="drop")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    folds = np.zeros(n_train, dtype=int)
    for fold, (_, val_idx) in enumerate(skf.split(np.zeros(n_train), binned_y)):
        folds[val_idx] = fold

    # Multi-field Text Representations & Out-of-fold Ridge Text Regressors
    text_svd_col_names = []
    oof_text_ridge = np.zeros(n_train, dtype=np.float32)
    test_text_ridge = np.zeros(n_test, dtype=np.float32)

    if text_cols:
        combined_text_train = (
            train_features[text_cols].fillna("").astype(str).agg(" ".join, axis=1)
        )
        combined_text_test = (
            test_features[text_cols].fillna("").astype(str).agg(" ".join, axis=1)
        )

        all_text = pd.concat(
            [combined_text_train, combined_text_test], axis=0
        ).reset_index(drop=True)
        combined_df["text_word_count"] = all_text.apply(
            lambda s: len(s.split())
        ).astype(np.float32)
        combined_df["text_char_len"] = all_text.apply(len).astype(np.float32)
        combined_df["text_digit_count"] = all_text.apply(
            lambda s: sum(c.isdigit() for c in s)
        ).astype(np.float32)
        combined_df["text_upper_count"] = all_text.apply(
            lambda s: sum(c.isupper() for c in s)
        ).astype(np.float32)
        num_cols.extend(
            [
                "text_word_count",
                "text_char_len",
                "text_digit_count",
                "text_upper_count",
            ]
        )

        # Word & Char n-gram TF-IDF with TruncatedSVD
        tfidf_word = TfidfVectorizer(
            max_features=3000, stop_words="english", ngram_range=(1, 2)
        )
        tfidf_char = TfidfVectorizer(
            max_features=3000, analyzer="char", ngram_range=(3, 5)
        )

        word_tr = tfidf_word.fit_transform(combined_text_train)
        word_te = tfidf_word.transform(combined_text_test)
        char_tr = tfidf_char.fit_transform(combined_text_train)
        char_te = tfidf_char.transform(combined_text_test)

        svd_w = TruncatedSVD(n_components=min(15, n_train - 1), random_state=42)
        svd_c = TruncatedSVD(n_components=min(10, n_train - 1), random_state=42)

        svd_w_tr = svd_w.fit_transform(word_tr)
        svd_w_te = svd_w.transform(word_te)
        svd_c_tr = svd_c.fit_transform(char_tr)
        svd_c_te = svd_c.transform(char_te)

        for comp_idx in range(svd_w_tr.shape[1]):
            col_name = f"svd_word_{comp_idx}"
            text_svd_col_names.append(col_name)
            combined_df[col_name] = np.concatenate([svd_w_tr[:, comp_idx], svd_w_te[:, comp_idx]])

        for comp_idx in range(svd_c_tr.shape[1]):
            col_name = f"svd_char_{comp_idx}"
            text_svd_col_names.append(col_name)
            combined_df[col_name] = np.concatenate([svd_c_tr[:, comp_idx], svd_c_te[:, comp_idx]])

        # Out-of-fold Ridge Text Regressor
        tfidf_comb_tr = sp.hstack([word_tr, char_tr], format="csr")
        tfidf_comb_te = sp.hstack([word_te, char_te], format="csr")
        log_y = np.log1p(np.maximum(0.0, y_train))

        for fold in range(n_splits):
            tr_idx = folds != fold
            va_idx = folds == fold
            ridge = Ridge(alpha=10.0, random_state=42)
            ridge.fit(tfidf_comb_tr[tr_idx], log_y[tr_idx])
            oof_text_ridge[va_idx] = np.expm1(ridge.predict(tfidf_comb_tr[va_idx]))
            test_text_ridge += np.expm1(ridge.predict(tfidf_comb_te)) / n_splits

    train_slice = combined_df.iloc[:n_train].copy()
    test_slice = combined_df.iloc[n_train:].copy()
    train_slice["fold"] = folds

    if text_cols:
        train_slice["text_ridge_oof"] = oof_text_ridge
        test_slice["text_ridge_oof"] = test_text_ridge
        num_cols.append("text_ridge_oof")

    # Frequency, Label, and Smoothed Out-of-Fold Target Statistics
    global_target_mean = float(np.mean(y_train))
    smoothing_m = 10.0

    for c in cat_cols:
        freq_map = train_slice[c].value_counts(normalize=True).to_dict()
        train_slice[f"{c}_freq"] = (
            train_slice[c].map(freq_map).fillna(0).astype(np.float32)
        )
        test_slice[f"{c}_freq"] = (
            test_slice[c].map(freq_map).fillna(0).astype(np.float32)
        )
        num_cols.append(f"{c}_freq")

        le = LabelEncoder()
        train_vals = train_slice[c].fillna("<<MISSING>>").astype(str)
        test_vals = test_slice[c].fillna("<<MISSING>>").astype(str)
        le.fit(pd.concat([train_vals, test_vals], axis=0))
        train_slice[f"{c}_label"] = le.transform(train_vals).astype(np.int32)
        test_slice[f"{c}_label"] = le.transform(test_vals).astype(np.int32)
        num_cols.append(f"{c}_label")

        target_oof_mean = np.full(n_train, global_target_mean, dtype=np.float64)
        target_oof_std = np.zeros(n_train, dtype=np.float64)
        target_oof_min = np.zeros(n_train, dtype=np.float64)
        target_oof_max = np.zeros(n_train, dtype=np.float64)

        for fold in range(n_splits):
            tr_idx = folds != fold
            val_idx = folds == fold

            fold_tr_cat = train_slice.iloc[tr_idx][c].fillna("<<MISSING>>")
            fold_tr_y = y_train[tr_idx]

            grouped = pd.Series(fold_tr_y).groupby(fold_tr_cat.values)
            counts = grouped.count()
            raw_means = grouped.mean()
            stds = grouped.std().fillna(0)
            mins = grouped.min()
            maxs = grouped.max()

            smoothed_means = (counts * raw_means + smoothing_m * global_target_mean) / (
                counts + smoothing_m
            )

            val_cats = train_slice.iloc[val_idx][c].fillna("<<MISSING>>")
            target_oof_mean[val_idx] = (
                val_cats.map(smoothed_means).fillna(global_target_mean).values
            )
            target_oof_std[val_idx] = val_cats.map(stds).fillna(0).values
            target_oof_min[val_idx] = (
                val_cats.map(mins).fillna(global_target_mean).values
            )
            target_oof_max[val_idx] = (
                val_cats.map(maxs).fillna(global_target_mean).values
            )

        full_tr_cat = train_slice[c].fillna("<<MISSING>>")
        full_grp = pd.Series(y_train).groupby(full_tr_cat.values)
        f_counts = full_grp.count()
        f_raw_means = full_grp.mean()
        f_stds = full_grp.std().fillna(0)
        f_mins = full_grp.min()
        f_maxs = full_grp.max()
        f_smoothed_means = (
            f_counts * f_raw_means + smoothing_m * global_target_mean
        ) / (f_counts + smoothing_m)

        test_cats = test_slice[c].fillna("<<MISSING>>")
        test_target_mean = (
            test_cats.map(f_smoothed_means).fillna(global_target_mean).values
        )
        test_target_std = test_cats.map(f_stds).fillna(0).values
        test_target_min = test_cats.map(f_mins).fillna(global_target_mean).values
        test_target_max = test_cats.map(f_maxs).fillna(global_target_mean).values

        train_slice[f"{c}_target_oof_mean"] = target_oof_mean.astype(np.float32)
        train_slice[f"{c}_target_oof_std"] = target_oof_std.astype(np.float32)
        train_slice[f"{c}_target_oof_min"] = target_oof_min.astype(np.float32)
        train_slice[f"{c}_target_oof_max"] = target_oof_max.astype(np.float32)

        test_slice[f"{c}_target_oof_mean"] = test_target_mean.astype(np.float32)
        test_slice[f"{c}_target_oof_std"] = test_target_std.astype(np.float32)
        test_slice[f"{c}_target_oof_min"] = test_target_min.astype(np.float32)
        test_slice[f"{c}_target_oof_max"] = test_target_max.astype(np.float32)

        num_cols.extend(
            [
                f"{c}_target_oof_mean",
                f"{c}_target_oof_std",
                f"{c}_target_oof_min",
                f"{c}_target_oof_max",
            ]
        )

    feature_cols = [c for c in sorted(list(set(num_cols))) if c in train_slice.columns]

    X_train_final = train_slice[feature_cols].copy()
    X_train_final[target_col] = y_train
    X_train_final["fold"] = train_slice["fold"]

    X_test_final = test_slice[feature_cols].copy()
    X_test_final["id"] = test_ids

    for col in feature_cols:
        med = X_train_final[col].median()
        if pd.isna(med):
            med = 0.0
        X_train_final[col] = X_train_final[col].fillna(med)
        X_test_final[col] = X_test_final[col].fillna(med)

    X_train_final.to_parquet("./working/train_engineered.parquet", index=False)
    X_test_final.to_parquet("./working/test_engineered.parquet", index=False)

    metadata = {
        "target_col": target_col,
        "n_train": n_train,
        "n_test": n_test,
        "n_features": len(feature_cols),
        "feature_cols": feature_cols,
        "n_folds": n_splits,
    }

    with open("./working/feature_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    return X_train_final, X_test_final, feature_cols, target_col, n_splits


# ============================================================================
# 3. Model Training, Ensemble Blending, and Evaluation
# ============================================================================


def main():
    X_train_final, X_test_final, feature_cols, target_col, n_splits = (
        run_feature_engineering()
    )

    X = X_train_final[feature_cols].copy()
    y = X_train_final[target_col].values.astype(np.float64)
    folds = X_train_final["fold"].values.astype(int)

    X_test = X_test_final[feature_cols].copy()
    test_ids = X_test_final["id"].values

    n_train = len(X)
    n_test = len(X_test)

    oof_lgb = np.zeros(n_train, dtype=np.float64)
    oof_lgb_huber = np.zeros(n_train, dtype=np.float64)
    oof_xgb = np.zeros(n_train, dtype=np.float64)
    oof_cat = np.zeros(n_train, dtype=np.float64)
    oof_cat_log = np.zeros(n_train, dtype=np.float64)
    oof_et = np.zeros(n_train, dtype=np.float64)
    oof_gbr_huber = np.zeros(n_train, dtype=np.float64)

    test_preds_lgb = np.zeros(n_test, dtype=np.float64)
    test_preds_lgb_huber = np.zeros(n_test, dtype=np.float64)
    test_preds_xgb = np.zeros(n_test, dtype=np.float64)
    test_preds_cat = np.zeros(n_test, dtype=np.float64)
    test_preds_cat_log = np.zeros(n_test, dtype=np.float64)
    test_preds_et = np.zeros(n_test, dtype=np.float64)
    test_preds_gbr_huber = np.zeros(n_test, dtype=np.float64)

    lgb_params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": 6,
        "min_child_samples": 15,
        "subsample": 0.85,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "n_estimators": 1500,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }

    lgb_huber_params = {
        "objective": "huber",
        "metric": "rmse",
        "alpha": 0.9,
        "boosting_type": "gbdt",
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": 6,
        "min_child_samples": 15,
        "subsample": 0.85,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "n_estimators": 1500,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }

    xgb_params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "booster": "gbtree",
        "learning_rate": 0.03,
        "max_depth": 5,
        "min_child_weight": 3,
        "subsample": 0.85,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "n_estimators": 1500,
        "random_state": 42,
        "n_jobs": -1,
        "tree_method": "hist",
        "early_stopping_rounds": 50,
    }

    cat_params = {
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "iterations": 1000,
        "learning_rate": 0.03,
        "depth": 6,
        "l2_leaf_reg": 3.0,
        "random_seed": 42,
        "verbose": 0,
        "early_stopping_rounds": 50,
    }

    cat_log_params = {
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "iterations": 1000,
        "learning_rate": 0.03,
        "depth": 6,
        "l2_leaf_reg": 5.0,
        "random_seed": 42,
        "verbose": 0,
        "early_stopping_rounds": 50,
    }

    for fold in range(n_splits):
        train_idx = folds != fold
        val_idx = folds == fold

        X_tr, y_tr = X.iloc[train_idx], y[train_idx]
        X_va, y_va = X.iloc[val_idx], y[val_idx]

        log_y_tr = np.log1p(np.maximum(0.0, y_tr))
        log_y_va = np.log1p(np.maximum(0.0, y_va))

        # Model 1: LightGBM Regressor (RMSE)
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
        oof_lgb[val_idx] = model_lgb.predict(X_va)
        test_preds_lgb += model_lgb.predict(X_test) / n_splits

        # Model 2: LightGBM Regressor (Huber Loss)
        model_lgb_huber = lgb.LGBMRegressor(**lgb_huber_params)
        model_lgb_huber.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )
        oof_lgb_huber[val_idx] = model_lgb_huber.predict(X_va)
        test_preds_lgb_huber += model_lgb_huber.predict(X_test) / n_splits

        # Model 3: XGBoost Regressor
        model_xgb = xgb.XGBRegressor(**xgb_params)
        model_xgb.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            verbose=False,
        )
        oof_xgb[val_idx] = model_xgb.predict(X_va)
        test_preds_xgb += model_xgb.predict(X_test) / n_splits

        # Model 4: CatBoost Regressor (Direct RMSE)
        model_cat = CatBoostRegressor(**cat_params)
        model_cat.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=False)
        oof_cat[val_idx] = model_cat.predict(X_va)
        test_preds_cat += model_cat.predict(X_test) / n_splits

        # Model 5: CatBoost Regressor (Log-Transformed Target)
        model_cat_log = CatBoostRegressor(**cat_log_params)
        model_cat_log.fit(X_tr, log_y_tr, eval_set=(X_va, log_y_va), verbose=False)
        oof_cat_log[val_idx] = np.expm1(model_cat_log.predict(X_va))
        test_preds_cat_log += np.expm1(model_cat_log.predict(X_test)) / n_splits

        # Model 6: ExtraTrees Regressor
        model_et = ExtraTreesRegressor(
            n_estimators=300,
            max_depth=12,
            min_samples_split=4,
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=42 + fold,
            n_jobs=-1,
        )
        model_et.fit(X_tr, y_tr)
        oof_et[val_idx] = model_et.predict(X_va)
        test_preds_et += model_et.predict(X_test) / n_splits

        # Model 7: Gradient Boosting Regressor (Huber Loss)
        model_gbr_huber = GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=4,
            min_samples_split=4,
            min_samples_leaf=2,
            subsample=0.85,
            loss="huber",
            random_state=42 + fold,
        )
        model_gbr_huber.fit(X_tr, y_tr)
        oof_gbr_huber[val_idx] = model_gbr_huber.predict(X_va)
        test_preds_gbr_huber += model_gbr_huber.predict(X_test) / n_splits

    # Bounded SLSQP Optimization for Optimal Ensemble Weights
    oof_matrix = np.column_stack([
        oof_lgb,
        oof_lgb_huber,
        oof_xgb,
        oof_cat,
        oof_cat_log,
        oof_et,
        oof_gbr_huber,
    ])
    test_matrix = np.column_stack([
        test_preds_lgb,
        test_preds_lgb_huber,
        test_preds_xgb,
        test_preds_cat,
        test_preds_cat_log,
        test_preds_et,
        test_preds_gbr_huber,
    ])

    num_models = oof_matrix.shape[1]

    def loss_func(weights):
        weights = weights / (np.sum(weights) + 1e-12)
        blended_oof = np.dot(oof_matrix, weights)
        return float(np.sqrt(mean_squared_error(y, blended_oof)))

    init_weights = np.ones(num_models) / num_models
    bounds = [(0.0, 1.0)] * num_models
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

    opt_res = minimize(
        loss_func,
        init_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    if opt_res.success:
        optimal_weights = opt_res.x / np.sum(opt_res.x)
    else:
        optimal_weights = init_weights

    upper_bound = float(np.max(y) * 1.5)
    final_oof_preds = np.dot(oof_matrix, optimal_weights)
    final_oof_preds = np.clip(final_oof_preds, a_min=0.0, a_max=upper_bound)

    final_test_preds = np.dot(test_matrix, optimal_weights)
    final_test_preds = np.clip(final_test_preds, a_min=0.0, a_max=upper_bound)

    final_rmse = float(np.sqrt(mean_squared_error(y, final_oof_preds)))

    # Save final submission
    os.makedirs("./submission", exist_ok=True)
    submission_path = "./submission/submission.csv"

    submission_df = pd.DataFrame(
        {
            "id": np.arange(len(final_test_preds), dtype=np.int64),
            "Gross Issued Amount (US$)": final_test_preds,
        }
    )
    submission_df.to_csv(submission_path, index=False)

    # Required output format
    print(f"Final Validation Score: {final_rmse}")


if __name__ == "__main__":
    main()
