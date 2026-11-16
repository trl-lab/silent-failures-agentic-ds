import os
import re
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import ExtraTreesRegressor
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

warnings.filterwarnings("ignore")


def calc_oof_target_encoding(train_series, test_series, y_trn, folds, smoothing=10.0):
    global_mean = float(np.mean(y_trn))
    oof_encoded = np.zeros(len(train_series), dtype=np.float64)
    n_folds = len(np.unique(folds))

    for f in range(n_folds):
        tr_idx = np.where(folds != f)[0]
        va_idx = np.where(folds == f)[0]

        tr_s = train_series.iloc[tr_idx]
        tr_y = y_trn[tr_idx]
        fold_global_mean = float(np.mean(tr_y))

        df_tr = pd.DataFrame({"cat": tr_s, "y": tr_y})
        grouped = df_tr.groupby("cat")["y"]
        counts = grouped.count()
        sums = grouped.sum()
        means = (sums + smoothing * fold_global_mean) / (counts + smoothing)

        va_s = train_series.iloc[va_idx]
        oof_encoded[va_idx] = va_s.map(means).fillna(fold_global_mean).values

    df_all = pd.DataFrame({"cat": train_series, "y": y_trn})
    grouped_all = df_all.groupby("cat")["y"]
    counts_all = grouped_all.count()
    sums_all = grouped_all.sum()
    means_all = (sums_all + smoothing * global_mean) / (counts_all + smoothing)
    test_encoded = test_series.map(means_all).fillna(global_mean).values

    return oof_encoded, test_encoded


# ==============================================================================
# 1. Feature Engineering & Preprocessing Pipeline
# ==============================================================================
def process_and_engineer_features():
    train_path = "./input/train.parquet"
    test_path = "./input/test.parquet"

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    target_col = "AwardTotal"
    if target_col not in train_df.columns:
        raise ValueError(f"Target column '{target_col}' not found in training data.")

    if "id" in test_df.columns:
        test_ids = test_df["id"].values
    else:
        test_ids = np.arange(len(test_df), dtype=int)

    y_train_raw = train_df[target_col].values.astype(np.float64)
    train_features = train_df.drop(columns=[target_col]).copy()
    test_features = test_df.copy()
    if "id" in test_features.columns and "id" not in train_df.columns:
        test_features = test_features.drop(columns=["id"])

    n_train = len(train_features)
    n_test = len(test_features)
    all_features = pd.concat([train_features, test_features], axis=0, ignore_index=True)

    n_splits = 5
    y_binned = pd.qcut(
        pd.Series(y_train_raw).rank(method="first"), q=n_splits * 2, labels=False
    )
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    fold_indices = np.zeros(n_train, dtype=int)
    for fold, (_, val_idx) in enumerate(skf.split(np.zeros(n_train), y_binned)):
        fold_indices[val_idx] = fold

    date_cols = []
    text_cols = []
    cat_cols = []
    num_cols = []

    for col in all_features.columns:
        col_series = all_features[col]
        dtype_str = str(col_series.dtype).lower()

        if "datetime" in dtype_str or "date" in col.lower() or "time" in col.lower():
            date_cols.append(col)
        elif dtype_str in ["object", "string", "category"]:
            valid_samples = col_series.dropna().astype(str)
            if len(valid_samples) > 0:
                avg_length = valid_samples.map(len).mean()
                unique_ratio = valid_samples.nunique() / max(1, len(valid_samples))
                if avg_length > 40 or (avg_length > 20 and unique_ratio > 0.6):
                    text_cols.append(col)
                else:
                    cat_cols.append(col)
            else:
                cat_cols.append(col)
        else:
            num_cols.append(col)

    engineered_dfs = []

    # A. Date Features
    date_feat_df = pd.DataFrame(index=all_features.index)
    parsed_date_cols = []
    for col in date_cols:
        try:
            dt_series = pd.to_datetime(all_features[col], errors="coerce")
            month = dt_series.dt.month.fillna(1).astype(float)
            day = dt_series.dt.day.fillna(1).astype(float)
            dayofweek = dt_series.dt.dayofweek.fillna(0).astype(float)
            dayofyear = dt_series.dt.dayofyear.fillna(1).astype(float)

            date_feat_df[f"{col}_year"] = dt_series.dt.year.fillna(-1).astype(int)
            date_feat_df[f"{col}_month"] = dt_series.dt.month.fillna(-1).astype(int)
            date_feat_df[f"{col}_day"] = dt_series.dt.day.fillna(-1).astype(int)
            date_feat_df[f"{col}_dayofweek"] = dt_series.dt.dayofweek.fillna(-1).astype(
                int
            )
            date_feat_df[f"{col}_dayofyear"] = dt_series.dt.dayofyear.fillna(-1).astype(
                int
            )
            date_feat_df[f"{col}_is_month_end"] = dt_series.dt.is_month_end.fillna(
                False
            ).astype(int)

            # Cyclical date representations
            date_feat_df[f"{col}_month_sin"] = np.sin(2 * np.pi * month / 12.0)
            date_feat_df[f"{col}_month_cos"] = np.cos(2 * np.pi * month / 12.0)
            date_feat_df[f"{col}_dayofweek_sin"] = np.sin(2 * np.pi * dayofweek / 7.0)
            date_feat_df[f"{col}_dayofweek_cos"] = np.cos(2 * np.pi * dayofweek / 7.0)
            date_feat_df[f"{col}_dayofyear_sin"] = np.sin(2 * np.pi * dayofyear / 365.25)
            date_feat_df[f"{col}_dayofyear_cos"] = np.cos(2 * np.pi * dayofyear / 365.25)

            parsed_date_cols.append((col, dt_series))
        except Exception:
            pass

    if len(parsed_date_cols) >= 2:
        for i in range(len(parsed_date_cols)):
            for j in range(i + 1, len(parsed_date_cols)):
                name_i, dt_i = parsed_date_cols[i]
                name_j, dt_j = parsed_date_cols[j]
                duration_days = (dt_j - dt_i).dt.days
                date_feat_df[f"duration_days_{name_j}_minus_{name_i}"] = (
                    duration_days.fillna(-999).astype(float)
                )
                date_feat_df[f"duration_months_{name_j}_minus_{name_i}"] = (
                    (duration_days / 30.4375).fillna(-999).astype(float)
                )

    if date_feat_df.shape[1] > 0:
        engineered_dfs.append(date_feat_df)

    # B. Numerical Features
    num_feat_df = pd.DataFrame(index=all_features.index)
    for col in num_cols:
        col_data = pd.to_numeric(all_features[col], errors="coerce")
        num_feat_df[f"{col}_isnan"] = col_data.isna().astype(int)

        train_median = col_data.iloc[:n_train].median()
        if pd.isna(train_median):
            train_median = 0.0

        col_imputed = col_data.fillna(train_median)
        num_feat_df[col] = col_imputed

        if (col_imputed >= 0).all():
            num_feat_df[f"{col}_log1p"] = np.log1p(col_imputed)

    valid_num_cols = [
        c for c in num_cols if c in num_feat_df.columns and not c.endswith("_isnan")
    ]
    if len(valid_num_cols) >= 2:
        top_nums = valid_num_cols[:6]
        for i in range(len(top_nums)):
            for j in range(i + 1, len(top_nums)):
                c1, c2 = top_nums[i], top_nums[j]
                v1, v2 = num_feat_df[c1].values, num_feat_df[c2].values
                num_feat_df[f"ratio_{c1}_over_{c2}"] = v1 / (np.abs(v2) + 1e-5)
                num_feat_df[f"diff_{c1}_minus_{c2}"] = v1 - v2
                num_feat_df[f"prod_{c1}_times_{c2}"] = v1 * v2

    if num_feat_df.shape[1] > 0:
        engineered_dfs.append(num_feat_df)

    # C. Categorical Features
    cat_feat_df = pd.DataFrame(index=all_features.index)
    for col in cat_cols:
        cat_series = all_features[col].fillna("__MISSING__").astype(str)
        train_counts = cat_series.iloc[:n_train].value_counts()
        cat_feat_df[f"{col}_freq"] = (
            cat_series.map(train_counts).fillna(0).astype(float)
        )
        unique_cats = {val: idx for idx, val in enumerate(train_counts.index)}
        cat_feat_df[f"{col}_label"] = cat_series.map(
            lambda x: unique_cats.get(x, -1)
        ).astype(int)

        oof_te, test_te = calc_oof_target_encoding(
            cat_series.iloc[:n_train],
            cat_series.iloc[n_train:],
            y_train_raw,
            fold_indices,
            smoothing=10.0,
        )
        cat_feat_df[f"{col}_target_enc"] = np.concatenate([oof_te, test_te])

    if len(cat_cols) >= 2:
        top_cats_inter = cat_cols[:4]
        for i in range(len(top_cats_inter)):
            for j in range(i + 1, len(top_cats_inter)):
                c1, c2 = top_cats_inter[i], top_cats_inter[j]
                inter_series = (
                    all_features[c1].fillna("__MISSING__").astype(str)
                    + "_"
                    + all_features[c2].fillna("__MISSING__").astype(str)
                )
                inter_counts = inter_series.iloc[:n_train].value_counts()
                cat_feat_df[f"{c1}_{c2}_freq"] = (
                    inter_series.map(inter_counts).fillna(0).astype(float)
                )
                oof_te_inter, test_te_inter = calc_oof_target_encoding(
                    inter_series.iloc[:n_train],
                    inter_series.iloc[n_train:],
                    y_train_raw,
                    fold_indices,
                    smoothing=15.0,
                )
                cat_feat_df[f"{c1}_{c2}_target_enc"] = np.concatenate(
                    [oof_te_inter, test_te_inter]
                )

    if len(cat_cols) > 0 and len(valid_num_cols) > 0:
        top_cats = cat_cols[:3]
        top_nums = valid_num_cols[:3]
        for cat in top_cats:
            for num in top_nums:
                train_group = pd.DataFrame(
                    {
                        "cat": all_features[cat]
                        .iloc[:n_train]
                        .fillna("__MISSING__")
                        .astype(str),
                        "num": num_feat_df[num].iloc[:n_train],
                    }
                )
                group_stats = (
                    train_group.groupby("cat")["num"]
                    .agg(["mean", "std", "median"])
                    .reset_index()
                )
                group_stats.columns = [
                    "cat",
                    f"{cat}_{num}_grp_mean",
                    f"{cat}_{num}_grp_std",
                    f"{cat}_{num}_grp_median",
                ]

                merged = pd.DataFrame(
                    {"cat": all_features[cat].fillna("__MISSING__").astype(str)}
                ).merge(group_stats, on="cat", how="left")
                cat_feat_df[f"{cat}_{num}_grp_mean"] = (
                    merged[f"{cat}_{num}_grp_mean"]
                    .fillna(train_group["num"].mean())
                    .values
                )
                cat_feat_df[f"{cat}_{num}_grp_std"] = (
                    merged[f"{cat}_{num}_grp_std"].fillna(0.0).values
                )
                cat_feat_df[f"{cat}_{num}_grp_median"] = (
                    merged[f"{cat}_{num}_grp_median"]
                    .fillna(train_group["num"].median())
                    .values
                )

    if cat_feat_df.shape[1] > 0:
        engineered_dfs.append(cat_feat_df)

    # D. Text Features
    text_feat_df = pd.DataFrame(index=all_features.index)
    for col in text_cols:
        raw_text = all_features[col].fillna("").astype(str)
        text_feat_df[f"{col}_char_len"] = raw_text.map(len).astype(float)
        text_feat_df[f"{col}_word_count"] = raw_text.map(
            lambda x: len(re.findall(r"\w+", x))
        ).astype(float)
        text_feat_df[f"{col}_digit_count"] = raw_text.map(
            lambda x: len(re.findall(r"\d", x))
        ).astype(float)
        text_feat_df[f"{col}_caps_count"] = raw_text.map(
            lambda x: sum(1 for c in x if c.isupper())
        ).astype(float)
        text_feat_df[f"{col}_caps_ratio"] = text_feat_df[f"{col}_caps_count"] / (
            text_feat_df[f"{col}_char_len"] + 1e-5
        )

        train_text = raw_text.iloc[:n_train]
        n_components = min(32, max(5, len(train_text) // 250))

        tfidf = TfidfVectorizer(
            max_features=10000, ngram_range=(1, 2), min_df=3, stop_words="english"
        )
        svd = TruncatedSVD(n_components=n_components, random_state=42)

        tfidf_train = tfidf.fit_transform(train_text)
        svd.fit(tfidf_train)

        tfidf_all = tfidf.transform(raw_text)
        svd_all = svd.transform(tfidf_all)

        for comp_idx in range(n_components):
            text_feat_df[f"{col}_svd_{comp_idx}"] = svd_all[:, comp_idx]

    if text_feat_df.shape[1] > 0:
        engineered_dfs.append(text_feat_df)

    X_all = pd.concat(engineered_dfs, axis=1)
    X_all = X_all.replace([np.inf, -np.inf], np.nan)
    for col in X_all.columns:
        if X_all[col].isna().any():
            med = X_all[col].iloc[:n_train].median()
            X_all[col] = X_all[col].fillna(med if not pd.isna(med) else 0.0)

    clean_columns = [re.sub(r"[\[\]<>, \"']", "_", str(c)) for c in X_all.columns]
    X_all.columns = clean_columns

    X_train = X_all.iloc[:n_train].copy().reset_index(drop=True)
    X_test = X_all.iloc[n_train:].copy().reset_index(drop=True)

    n_splits = 5
    y_binned = pd.qcut(
        pd.Series(y_train_raw).rank(method="first"), q=n_splits * 2, labels=False
    )
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    fold_indices = np.zeros(n_train, dtype=int)
    for fold, (_, val_idx) in enumerate(skf.split(X_train, y_binned)):
        fold_indices[val_idx] = fold

    return (
        X_train.values.astype(np.float32),
        y_train_raw,
        X_test.values.astype(np.float32),
        test_ids,
        fold_indices,
        n_splits,
    )


# ==============================================================================
# 2. Model Configurations (LightGBM, XGBoost, CatBoost, ExtraTrees)
# ==============================================================================
def get_lightgbm_params(seed: int = 42) -> dict:
    return {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 20,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "n_estimators": 2500,
        "random_state": seed,
        "n_jobs": -1,
        "verbose": -1,
    }


def get_xgboost_params(seed: int = 42) -> dict:
    return {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "learning_rate": 0.03,
        "max_depth": 6,
        "min_child_weight": 3,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "n_estimators": 2500,
        "random_state": seed,
        "n_jobs": -1,
    }


def get_catboost_params(seed: int = 42) -> dict:
    return {
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "learning_rate": 0.04,
        "depth": 6,
        "l2_leaf_reg": 3.0,
        "iterations": 2500,
        "random_seed": seed,
        "verbose": False,
        "thread_count": -1,
    }


# ==============================================================================
# 3. Model Training, Ensemble Blending & Evaluation
# ==============================================================================
def main():
    os.makedirs("./working", exist_ok=True)
    os.makedirs("./submission", exist_ok=True)

    X_train_np, y_train, X_test_np, test_ids, fold_indices, n_splits = (
        process_and_engineer_features()
    )
    n_samples, n_features = X_train_np.shape
    n_test = len(X_test_np)

    oof_lgb = np.zeros(n_samples, dtype=np.float64)
    oof_xgb = np.zeros(n_samples, dtype=np.float64)
    oof_cb = np.zeros(n_samples, dtype=np.float64)
    oof_et = np.zeros(n_samples, dtype=np.float64)

    test_preds_lgb = np.zeros(n_test, dtype=np.float64)
    test_preds_xgb = np.zeros(n_test, dtype=np.float64)
    test_preds_cb = np.zeros(n_test, dtype=np.float64)
    test_preds_et = np.zeros(n_test, dtype=np.float64)

    for fold in range(n_splits):
        val_idx = np.where(fold_indices == fold)[0]
        trn_idx = np.where(fold_indices != fold)[0]

        X_tr, y_tr = X_train_np[trn_idx], y_train[trn_idx]
        X_va, y_va = X_train_np[val_idx], y_train[val_idx]

        # 1. LightGBM
        lgb_params = get_lightgbm_params(seed=42 + fold)
        lgb_model = lgb.LGBMRegressor(**lgb_params)
        callbacks = [
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0),
        ]
        lgb_model.fit(
            X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric="rmse", callbacks=callbacks
        )
        val_pred_lgb = lgb_model.predict(X_va)
        oof_lgb[val_idx] = val_pred_lgb
        test_preds_lgb += lgb_model.predict(X_test_np) / n_splits
        rmse_lgb_fold = np.sqrt(mean_squared_error(y_va, val_pred_lgb))

        # 2. XGBoost
        xgb_params = get_xgboost_params(seed=42 + fold)
        xgb_model = xgb.XGBRegressor(**xgb_params, early_stopping_rounds=50)
        xgb_model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

        val_pred_xgb = xgb_model.predict(X_va)
        oof_xgb[val_idx] = val_pred_xgb
        test_preds_xgb += xgb_model.predict(X_test_np) / n_splits
        rmse_xgb_fold = np.sqrt(mean_squared_error(y_va, val_pred_xgb))

        # 3. CatBoost
        cb_params = get_catboost_params(seed=42 + fold)
        cb_model = CatBoostRegressor(**cb_params, early_stopping_rounds=50)
        cb_model.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=False)

        val_pred_cb = cb_model.predict(X_va)
        oof_cb[val_idx] = val_pred_cb
        test_preds_cb += cb_model.predict(X_test_np) / n_splits
        rmse_cb_fold = np.sqrt(mean_squared_error(y_va, val_pred_cb))

        # 4. ExtraTrees Regressor
        et_model = ExtraTreesRegressor(
            n_estimators=300,
            max_depth=15,
            min_samples_split=5,
            random_state=42 + fold,
            n_jobs=-1,
        )
        et_model.fit(X_tr, y_tr)

        val_pred_et = et_model.predict(X_va)
        oof_et[val_idx] = val_pred_et
        test_preds_et += et_model.predict(X_test_np) / n_splits
        rmse_et_fold = np.sqrt(mean_squared_error(y_va, val_pred_et))

        print(
            f"Fold {fold+1}/{n_splits} | LGB RMSE: {rmse_lgb_fold:.4f} | XGB RMSE: {rmse_xgb_fold:.4f} | CB RMSE: {rmse_cb_fold:.4f} | ET RMSE: {rmse_et_fold:.4f}"
        )

    # Optimal Blending Optimization
    def blend_objective(weights):
        w1, w2, w3, w4 = weights
        pred = w1 * oof_lgb + w2 * oof_xgb + w3 * oof_cb + w4 * oof_et
        return np.sqrt(mean_squared_error(y_train, pred))

    initial_weights = [0.35, 0.35, 0.20, 0.10]
    bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]
    constraints = {"type": "eq", "fun": lambda w: sum(w) - 1.0}

    opt_result = minimize(
        blend_objective,
        initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    if opt_result.success:
        best_weights = opt_result.x
    else:
        best_weights = np.array([0.35, 0.35, 0.20, 0.10])

    final_oof_preds = (
        best_weights[0] * oof_lgb
        + best_weights[1] * oof_xgb
        + best_weights[2] * oof_cb
        + best_weights[3] * oof_et
    )
    final_oof_preds = np.clip(final_oof_preds, a_min=0.0, a_max=None)
    final_rmse = float(np.sqrt(mean_squared_error(y_train, final_oof_preds)))

    final_test_preds = (
        best_weights[0] * test_preds_lgb
        + best_weights[1] * test_preds_xgb
        + best_weights[2] * test_preds_cb
        + best_weights[3] * test_preds_et
    )
    final_test_preds = np.clip(final_test_preds, a_min=0.0, a_max=None)

    submission_df = pd.DataFrame(
        {"id": test_ids.astype(int), "AwardTotal": final_test_preds}
    )

    submission_path = "./submission/submission.csv"
    submission_df.to_csv(submission_path, index=False)

    print(f"Final Validation Score: {final_rmse}")


if __name__ == "__main__":
    main()