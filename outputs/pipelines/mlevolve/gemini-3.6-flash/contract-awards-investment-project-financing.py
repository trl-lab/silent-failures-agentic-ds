import os
import warnings
import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import xgboost as xgb

warnings.filterwarnings("ignore")


def main():
    input_dir = "./input"
    submission_dir = "./submission"
    os.makedirs(submission_dir, exist_ok=True)

    train_path = os.path.join(input_dir, "train.parquet")
    test_path = os.path.join(input_dir, "test.parquet")

    print("Loading raw dataset...")
    df_train = pd.read_parquet(train_path)
    df_test = pd.read_parquet(test_path)

    target_col = "Supplier Contract Amount (USD)"
    if target_col not in df_train.columns:
        raise ValueError(f"Target column '{target_col}' not found in training data.")

    y_train_raw = df_train[target_col].values
    y_train_log = np.log1p(np.maximum(0, y_train_raw))

    train_len = len(df_train)
    df_all = pd.concat(
        [df_train.drop(columns=[target_col], errors="ignore"), df_test],
        axis=0,
        ignore_index=True,
    )

    # Identify column types
    date_cols = [
        c
        for c in df_all.columns
        if "date" in c.lower() or "year" in c.lower() or "month" in c.lower()
    ]
    num_cols = [
        c
        for c in df_all.columns
        if df_all[c].dtype in ["int64", "float64", "int32", "float32"]
        and c not in date_cols
    ]
    cat_cols = [
        c
        for c in df_all.columns
        if df_all[c].dtype == "object" or df_all[c].dtype.name == "category"
    ]

    print(
        f"Dataset shape: {df_all.shape} | Numeric: {len(num_cols)} | Categorical: {len(cat_cols)} | Dates: {len(date_cols)}"
    )

    # 1. Process Date Columns & Time Deltas
    parsed_dates = {}
    for dc in date_cols:
        try:
            dt_series = pd.to_datetime(df_all[dc], errors="coerce")
            df_all[f"{dc}_year"] = dt_series.dt.year
            df_all[f"{dc}_month"] = dt_series.dt.month
            df_all[f"{dc}_quarter"] = dt_series.dt.quarter
            df_all[f"{dc}_day"] = dt_series.dt.day
            df_all[f"{dc}_dayofweek"] = dt_series.dt.dayofweek
            parsed_dates[dc] = dt_series
        except Exception:
            pass

    # Time diffs between valid pairs of dates
    p_keys = list(parsed_dates.keys())
    for i in range(len(p_keys)):
        for j in range(i + 1, len(p_keys)):
            d1, d2 = p_keys[i], p_keys[j]
            diff = (parsed_dates[d1] - parsed_dates[d2]).dt.days
            if diff.notna().sum() > 0:
                df_all[f"diff_days_{d1}_vs_{d2}"] = diff

    df_all.drop(columns=list(parsed_dates.keys()), inplace=True, errors="ignore")

    # Re-evaluate categorical columns
    cat_cols = [
        c
        for c in df_all.columns
        if df_all[c].dtype == "object" or df_all[c].dtype.name == "category"
    ]

    # 2. Text Feature Extraction (Unified TF-IDF Word + Char N-grams -> 30-dim SVD & In-fold Ridge prep)
    from scipy.sparse import hstack

    text_cols = [c for c in cat_cols if df_all[c].nunique() > 500]
    tfidf_combined = None
    if len(text_cols) > 0:
        combined_text = (
            df_all[text_cols].fillna("").astype(str).agg(" ".join, axis=1).tolist()
        )
        tfidf_word = TfidfVectorizer(
            max_features=5000, ngram_range=(1, 2), stop_words="english"
        )
        tfidf_char = TfidfVectorizer(
            max_features=5000, analyzer="char_wb", ngram_range=(3, 5)
        )

        word_vecs = tfidf_word.fit_transform(combined_text)
        char_vecs = tfidf_char.fit_transform(combined_text)

        tfidf_combined = hstack([word_vecs, char_vecs]).tocsr()

        svd = TruncatedSVD(n_components=30, random_state=42)
        svd_feats = svd.fit_transform(tfidf_combined)

        for col_idx in range(svd_feats.shape[1]):
            df_all[f"unified_text_svd_{col_idx}"] = svd_feats[:, col_idx]

    # 3. Categorical Frequency & Categorical Aggregation Features
    key_cats = [c for c in cat_cols if df_all[c].nunique() < 5000]
    for c in key_cats:
        freq = df_all.iloc[:train_len][c].value_counts(normalize=True).to_dict()
        df_all[f"{c}_freq"] = df_all[c].map(freq).fillna(0).astype("float32")

    # Interaction pairs among key categoricals
    top_cats = key_cats[:4]
    interaction_cols = []
    for i in range(len(top_cats)):
        for j in range(i + 1, len(top_cats)):
            c1, c2 = top_cats[i], top_cats[j]
            pair_col = f"{c1}_x_{c2}"
            df_all[pair_col] = df_all[c1].astype(str) + "___" + df_all[c2].astype(str)
            interaction_cols.append(pair_col)

    target_stat_cols = top_cats + interaction_cols

    num_to_agg = [c for c in num_cols if df_all[c].nunique() > 10][:5]
    group_cols = [c for c in key_cats if 5 < df_all[c].nunique() < 200][:3]

    for gc in group_cols:
        for nc in num_to_agg:
            train_sub = df_all.iloc[:train_len][[gc, nc]].copy()
            grp = train_sub.groupby(gc)[nc].agg(["mean", "std", "min", "max", "median"]).reset_index()
            grp.columns = [
                gc,
                f"{gc}_{nc}_mean",
                f"{gc}_{nc}_std",
                f"{gc}_{nc}_min",
                f"{gc}_{nc}_max",
                f"{gc}_{nc}_median",
            ]
            df_all = df_all.merge(grp, on=gc, how="left")

    # Format Categorical Features
    cat_final_cols = [
        c
        for c in df_all.columns
        if df_all[c].dtype == "object" or df_all[c].dtype.name == "category"
    ]
    for c in cat_final_cols:
        df_all[c] = df_all[c].astype(str).astype("category")

    X = df_all.iloc[:train_len].copy()
    X_test = df_all.iloc[train_len:].copy()

    n_samples = len(X)
    n_test = len(X_test)

    n_splits = 5
    cv_kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    oof_lgb = np.zeros(n_samples)
    oof_xgb = np.zeros(n_samples)
    oof_cat = np.zeros(n_samples)

    test_lgb = np.zeros(n_test)
    test_xgb = np.zeros(n_test)
    test_cat = np.zeros(n_test)

    print(
        f"Starting {n_splits}-fold CV training on {n_samples} samples with {X.shape[1]} features..."
    )

    for fold, (train_idx, val_idx) in enumerate(cv_kf.split(X)):
        X_tr, y_tr = X.iloc[train_idx].copy(), y_train_log[train_idx]
        X_va, y_va = X.iloc[val_idx].copy(), y_train_log[val_idx]
        X_te_fold = X_test.copy()

        # In-fold Ridge Regression meta-feature on combined text TF-IDF
        if tfidf_combined is not None:
            ridge_text = Ridge(alpha=5.0, random_state=42)
            ridge_text.fit(tfidf_combined[train_idx], y_tr)
            X_tr["text_pred_log"] = ridge_text.predict(tfidf_combined[train_idx]).astype("float32")
            X_va["text_pred_log"] = ridge_text.predict(tfidf_combined[val_idx]).astype("float32")
            X_te_fold["text_pred_log"] = ridge_text.predict(tfidf_combined[train_len:]).astype("float32")

        # In-fold Multi-granularity Target Encoding (mean, std, median) on target_stat_cols
        global_mean = float(np.mean(y_tr))
        global_std = float(np.std(y_tr))
        global_median = float(np.median(y_tr))
        smoothing = 10.0

        for c in target_stat_cols:
            df_tr_c = pd.DataFrame({c: X_tr[c].astype(str), "target": y_tr})
            stats = df_tr_c.groupby(c, observed=False)["target"].agg(["count", "mean", "std", "median"])

            smooth_mean = (
                (stats["count"] * stats["mean"] + smoothing * global_mean)
                / (stats["count"] + smoothing)
            ).to_dict()
            std_map = stats["std"].fillna(global_std).to_dict()
            med_map = stats["median"].fillna(global_median).to_dict()

            for stat_name, mapping, fill_val in [
                ("mean", smooth_mean, global_mean),
                ("std", std_map, global_std),
                ("median", med_map, global_median),
            ]:
                col_name = f"{c}_te_{stat_name}"
                X_tr[col_name] = X_tr[c].astype(str).map(mapping).fillna(fill_val).astype("float32").values
                X_va[col_name] = X_va[c].astype(str).map(mapping).fillna(fill_val).astype("float32").values
                X_te_fold[col_name] = X_te_fold[c].astype(str).map(mapping).fillna(fill_val).astype("float32").values

        # LightGBM Regressor
        lgb_model = lgb.LGBMRegressor(
            objective="regression",
            metric="rmse",
            learning_rate=0.03,
            num_leaves=45,
            feature_fraction=0.65,
            bagging_fraction=0.8,
            bagging_freq=1,
            min_child_samples=25,
            reg_alpha=0.1,
            reg_lambda=1.0,
            n_estimators=1500,
            random_state=42 + fold,
            verbosity=-1,
            n_jobs=-1,
        )
        lgb_model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[lgb.early_stopping(30, verbose=False)],
        )
        oof_lgb[val_idx] = lgb_model.predict(X_va)
        test_lgb += lgb_model.predict(X_te_fold) / n_splits

        # XGBoost Regressor
        xgb_model = xgb.XGBRegressor(
            objective="reg:squarederror",
            eval_metric="rmse",
            learning_rate=0.03,
            max_depth=7,
            subsample=0.8,
            colsample_bytree=0.65,
            min_child_weight=5,
            reg_alpha=0.1,
            reg_lambda=1.0,
            n_estimators=1500,
            tree_method="hist",
            enable_categorical=True,
            random_state=42 + fold,
            early_stopping_rounds=30,
            n_jobs=-1,
        )
        xgb_model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        oof_xgb[val_idx] = xgb_model.predict(X_va)
        test_xgb += xgb_model.predict(X_te_fold) / n_splits

        # CatBoost Regressor
        cat_model = cb.CatBoostRegressor(
            loss_function="RMSE",
            eval_metric="RMSE",
            learning_rate=0.03,
            depth=7,
            rsm=0.65,
            l2_leaf_reg=3.0,
            iterations=1500,
            random_seed=42 + fold,
            cat_features=cat_final_cols if cat_final_cols else None,
            verbose=False,
        )
        cat_model.fit(
            X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=30, verbose=False
        )
        oof_cat[val_idx] = cat_model.predict(X_va)
        test_cat += cat_model.predict(X_te_fold) / n_splits

        fold_var_lgb = np.mean((y_va - oof_lgb[val_idx]) ** 2)
        fold_rmse = np.sqrt(
            mean_squared_error(
                y_train_raw[val_idx], np.exp(oof_lgb[val_idx] + 0.5 * fold_var_lgb) - 1.0
            )
        )
        print(f"Fold {fold + 1}/{n_splits} complete. LGB Raw RMSE (with bias correction): {fold_rmse:.2f}")

    # Optimize blending weights on raw scale RMSE using log-normal bias correction
    def loss_func(weights):
        w1, w2, w3 = weights
        pred_log = w1 * oof_lgb + w2 * oof_xgb + w3 * oof_cat
        res_var = np.mean((y_train_log - pred_log) ** 2)
        pred_raw = np.exp(pred_log + 0.5 * res_var) - 1.0
        return np.sqrt(mean_squared_error(y_train_raw, pred_raw))

    init_weights = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
    bounds = [(0, 1), (0, 1), (0, 1)]
    constraints = {"type": "eq", "fun": lambda w: 1.0 - sum(w)}

    res = minimize(
        loss_func,
        init_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    best_w = res.x if res.success else init_weights

    oof_blend_log = best_w[0] * oof_lgb + best_w[1] * oof_xgb + best_w[2] * oof_cat
    blend_var = np.mean((y_train_log - oof_blend_log) ** 2)
    oof_blend_raw = np.exp(oof_blend_log + 0.5 * blend_var) - 1.0
    score = np.sqrt(mean_squared_error(y_train_raw, oof_blend_raw))

    print(
        f"Optimal Weights: LGB={best_w[0]:.3f}, XGB={best_w[1]:.3f}, CAT={best_w[2]:.3f}"
    )

    test_blend_log = best_w[0] * test_lgb + best_w[1] * test_xgb + best_w[2] * test_cat
    test_blend_raw = np.exp(test_blend_log + 0.5 * blend_var) - 1.0

    sub_df = pd.DataFrame(
        {
            "id": np.arange(len(test_blend_raw)),
            "Supplier Contract Amount (USD)": test_blend_raw,
        }
    )

    sub_path = os.path.join(submission_dir, "submission.csv")
    sub_df.to_csv(sub_path, index=False)
    print(f"Saved submission file to {sub_path} with {len(sub_df)} records.")

    print(f"Final Validation Score: {score}")


if __name__ == "__main__":
    main()