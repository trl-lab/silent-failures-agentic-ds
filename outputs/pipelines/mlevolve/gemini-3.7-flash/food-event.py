import gc
import json
import os
import warnings
from catboost import CatBoostClassifier
from itertools import combinations
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.sparse import hstack
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
import xgboost as xgb

warnings.filterwarnings("ignore")


def main():
    os.makedirs("./working", exist_ok=True)
    os.makedirs("./submission", exist_ok=True)

    print("Loading raw datasets...")
    train_raw = pd.read_parquet("./input/train.parquet")
    test_raw = pd.read_parquet("./input/test.parquet")

    target_col = "outcomes"
    if target_col not in train_raw.columns:
        raise ValueError(f"Target column '{target_col}' not found in train data.")

    # 1. Target Encoding and Fold Splitting
    target_encoder = LabelEncoder()
    train_raw["target"] = target_encoder.fit_transform(
        train_raw[target_col].astype(str)
    )
    n_classes = len(target_encoder.classes_)

    target_classes = {int(i): str(cls) for i, cls in enumerate(target_encoder.classes_)}
    with open("./working/target_classes.json", "w") as f:
        json.dump(target_classes, f, indent=2)
    joblib.dump(target_encoder, "./working/target_encoder.joblib")

    print(
        f"Dataset loaded: Train={train_raw.shape}, Test={test_raw.shape}, Target Classes={n_classes}"
    )

    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    train_raw["fold"] = -1
    for fold, (_, val_idx) in enumerate(skf.split(train_raw, train_raw["target"])):
        train_raw.loc[val_idx, "fold"] = fold

    # 2. Automated Modality Detection
    exclude_cols = [target_col, "target", "fold", "id", "ID", "Id"]
    feature_cols = [c for c in train_raw.columns if c not in exclude_cols]

    date_cols = []
    text_cols = []
    cat_cols = []
    num_cols = []

    for col in feature_cols:
        col_type = train_raw[col].dtype
        sample_non_null = train_raw[col].dropna()

        if (
            "date" in col.lower()
            or "time" in col.lower()
            or np.issubdtype(col_type, np.datetime64)
        ):
            date_cols.append(col)
        elif np.issubdtype(col_type, np.number):
            num_cols.append(col)
        else:
            if len(sample_non_null) > 0:
                avg_words = (
                    sample_non_null.astype(str).apply(lambda x: len(x.split())).mean()
                )
                avg_len = sample_non_null.astype(str).apply(len).mean()
                if avg_words > 3 or avg_len > 40:
                    text_cols.append(col)
                else:
                    cat_cols.append(col)
            else:
                cat_cols.append(col)

    print(
        f"Feature types -> Numeric: {len(num_cols)}, Categorical: {len(cat_cols)}, Text: {len(text_cols)}, Date: {len(date_cols)}"
    )

    # 3. Base Features Extraction (Dates, Missingness, and Numerics)
    def extract_base_features(df, train_ref_medians=None):
        out = pd.DataFrame(index=df.index)

        missing_counts = df[feature_cols].isnull().sum(axis=1)
        out["row_missing_count"] = missing_counts.astype(np.float32)
        out["row_missing_ratio"] = (missing_counts / len(feature_cols)).astype(
            np.float32
        )

        parsed_dates = {}
        for dcol in date_cols:
            parsed = pd.to_datetime(df[dcol], errors="coerce")
            parsed_dates[dcol] = parsed
            year = parsed.dt.year.fillna(-1).astype(np.float32)
            month = parsed.dt.month.fillna(0).astype(np.float32)
            day = parsed.dt.day.fillna(0).astype(np.float32)
            dayofweek = parsed.dt.dayofweek.fillna(0).astype(np.float32)
            quarter = parsed.dt.quarter.fillna(0).astype(np.float32)

            out[f"{dcol}_year"] = year
            out[f"{dcol}_month"] = month
            out[f"{dcol}_day"] = day
            out[f"{dcol}_dayofweek"] = dayofweek
            out[f"{dcol}_quarter"] = quarter

            # Cyclical temporal encodings
            out[f"{dcol}_month_sin"] = np.sin(2 * np.pi * month / 12.0).astype(np.float32)
            out[f"{dcol}_month_cos"] = np.cos(2 * np.pi * month / 12.0).astype(np.float32)
            out[f"{dcol}_day_sin"] = np.sin(2 * np.pi * day / 31.0).astype(np.float32)
            out[f"{dcol}_day_cos"] = np.cos(2 * np.pi * day / 31.0).astype(np.float32)
            out[f"{dcol}_dow_sin"] = np.sin(2 * np.pi * dayofweek / 7.0).astype(np.float32)
            out[f"{dcol}_dow_cos"] = np.cos(2 * np.pi * dayofweek / 7.0).astype(np.float32)

            # Elapsed days timeline features
            epoch_days = (parsed - pd.Timestamp("2000-01-01")).dt.total_seconds() / 86400.0
            out[f"{dcol}_elapsed_days"] = epoch_days.fillna(-999.0).astype(np.float32)

        if len(date_cols) >= 2:
            d_keys = list(parsed_dates.keys())
            for i in range(len(d_keys)):
                for j in range(i + 1, len(d_keys)):
                    diff = (
                        parsed_dates[d_keys[i]] - parsed_dates[d_keys[j]]
                    ).dt.total_seconds() / 86400.0
                    out[f"diff_days_{d_keys[i]}_{d_keys[j]}"] = diff.fillna(
                        -999.0
                    ).astype(np.float32)

        computed_medians = {}
        for ncol in num_cols:
            val = pd.to_numeric(df[ncol], errors="coerce")
            out[f"{ncol}_is_null"] = val.isnull().astype(np.float32)
            med = (
                train_ref_medians[ncol]
                if train_ref_medians is not None
                else (val.median() if not val.dropna().empty else 0.0)
            )
            computed_medians[ncol] = med
            filled = val.fillna(med).astype(np.float32)
            out[ncol] = filled
            out[f"{ncol}_log1p"] = np.log1p(np.maximum(0, filled)).astype(np.float32)

        return out, computed_medians

    print("Extracting base tabular and date features...")
    train_base, train_medians = extract_base_features(train_raw)
    test_base, _ = extract_base_features(test_raw, train_ref_medians=train_medians)

    # 4. Text Feature Engineering (TF-IDF + 60-dim SVD + OOF Linear Probabilities)
    text_feature_dfs_train = []
    text_feature_dfs_test = []

    for tcol in text_cols:
        train_text = train_raw[tcol].fillna("missing").astype(str)
        test_text = test_raw[tcol].fillna("missing").astype(str)

        train_tstats = pd.DataFrame(index=train_raw.index)
        test_tstats = pd.DataFrame(index=test_raw.index)

        train_tstats[f"{tcol}_char_len"] = train_text.apply(len).astype(np.float32)
        train_tstats[f"{tcol}_word_count"] = train_text.apply(
            lambda x: len(x.split())
        ).astype(np.float32)
        test_tstats[f"{tcol}_char_len"] = test_text.apply(len).astype(np.float32)
        test_tstats[f"{tcol}_word_count"] = test_text.apply(
            lambda x: len(x.split())
        ).astype(np.float32)

        text_feature_dfs_train.append(train_tstats)
        text_feature_dfs_test.append(test_tstats)

        tfidf_word = TfidfVectorizer(
            max_features=15000, ngram_range=(1, 2), min_df=2, sublinear_tf=True
        )
        tfidf_char = TfidfVectorizer(
            max_features=15000,
            ngram_range=(3, 5),
            analyzer="char",
            min_df=3,
            sublinear_tf=True,
        )

        train_word_vec = tfidf_word.fit_transform(train_text)
        test_word_vec = tfidf_word.transform(test_text)
        train_char_vec = tfidf_char.fit_transform(train_text)
        test_char_vec = tfidf_char.transform(test_text)

        train_comb_vec = hstack([train_word_vec, train_char_vec]).tocsr()
        test_comb_vec = hstack([test_word_vec, test_char_vec]).tocsr()

        n_components = min(60, train_comb_vec.shape[1] - 1)
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        train_svd = svd.fit_transform(train_comb_vec)
        test_svd = svd.transform(test_comb_vec)

        svd_cols = [f"{tcol}_svd_{k}" for k in range(n_components)]
        train_svd_df = pd.DataFrame(
            train_svd, columns=svd_cols, index=train_raw.index
        ).astype(np.float32)
        test_svd_df = pd.DataFrame(
            test_svd, columns=svd_cols, index=test_raw.index
        ).astype(np.float32)

        text_feature_dfs_train.append(train_svd_df)
        text_feature_dfs_test.append(test_svd_df)

        # Out-of-fold linear model probability meta-features
        oof_text_probs = np.zeros((train_raw.shape[0], n_classes), dtype=np.float32)
        test_text_probs = np.zeros((test_raw.shape[0], n_classes), dtype=np.float32)

        for fold in range(n_splits):
            trn_idx = (train_raw["fold"] != fold).values
            val_idx = (train_raw["fold"] == fold).values

            lr_text = LogisticRegression(
                C=1.0, max_iter=250, random_state=42, n_jobs=-1
            )
            lr_text.fit(train_comb_vec[trn_idx], train_raw.loc[trn_idx, "target"].values)

            oof_text_probs[val_idx] = lr_text.predict_proba(train_comb_vec[val_idx]).astype(
                np.float32
            )
            test_text_probs += (
                lr_text.predict_proba(test_comb_vec).astype(np.float32) / n_splits
            )

        prob_cols = [f"{tcol}_oof_prob_cls_{k}" for k in range(n_classes)]
        train_prob_df = pd.DataFrame(
            oof_text_probs, columns=prob_cols, index=train_raw.index
        )
        test_prob_df = pd.DataFrame(
            test_text_probs, columns=prob_cols, index=test_raw.index
        )

        text_feature_dfs_train.append(train_prob_df)
        text_feature_dfs_test.append(test_prob_df)

    # 5. Categorical Encodings (Single & Pairwise: Label, Frequency & Smoothed OOF Target Encoding)
    cat_train_df = pd.DataFrame(index=train_raw.index)
    cat_test_df = pd.DataFrame(index=test_raw.index)

    for ccol in cat_cols:
        train_series = train_raw[ccol].fillna("MISSING").astype(str)
        test_series = test_raw[ccol].fillna("MISSING").astype(str)

        le = LabelEncoder()
        full_vals = pd.concat([train_series, test_series], axis=0).unique()
        le.fit(full_vals)
        cat_train_df[f"{ccol}_le"] = le.transform(train_series).astype(np.float32)
        cat_test_df[f"{ccol}_le"] = le.transform(test_series).astype(np.float32)

        freq_map = train_series.value_counts(normalize=True).to_dict()
        cat_train_df[f"{ccol}_freq"] = (
            train_series.map(freq_map).fillna(0.0).astype(np.float32)
        )
        cat_test_df[f"{ccol}_freq"] = (
            test_series.map(freq_map).fillna(0.0).astype(np.float32)
        )

        smoothing = 10.0
        for cls_idx in range(n_classes):
            te_col_name = f"{ccol}_te_class_{cls_idx}"
            cat_train_df[te_col_name] = 0.0
            cat_test_df[te_col_name] = 0.0

            global_prior = (train_raw["target"] == cls_idx).mean()

            for fold in range(n_splits):
                trn_idx = train_raw["fold"] != fold
                vld_idx = train_raw["fold"] == fold

                trn_cat = train_series.loc[trn_idx]
                trn_tgt = (train_raw["target"].loc[trn_idx] == cls_idx).astype(float)

                stats = trn_tgt.groupby(trn_cat).agg(["count", "mean"])
                smooth_te = (
                    stats["count"] * stats["mean"] + smoothing * global_prior
                ) / (stats["count"] + smoothing)
                te_dict = smooth_te.to_dict()

                vld_cat = train_series.loc[vld_idx]
                cat_train_df.loc[vld_idx, te_col_name] = (
                    vld_cat.map(te_dict).fillna(global_prior).astype(np.float32)
                )

            full_stats = (
                (train_raw["target"] == cls_idx)
                .astype(float)
                .groupby(train_series)
                .agg(["count", "mean"])
            )
            full_smooth_te = (
                full_stats["count"] * full_stats["mean"] + smoothing * global_prior
            ) / (full_stats["count"] + smoothing)
            full_te_dict = full_smooth_te.to_dict()
            cat_test_df[te_col_name] = (
                test_series.map(full_te_dict).fillna(global_prior).astype(np.float32)
            )

    # Pairwise categorical interactions
    for c1, c2 in combinations(cat_cols, 2):
        pair_col = f"{c1}_x_{c2}"
        train_pair = (
            train_raw[c1].fillna("MISSING").astype(str)
            + "__"
            + train_raw[c2].fillna("MISSING").astype(str)
        )
        test_pair = (
            test_raw[c1].fillna("MISSING").astype(str)
            + "__"
            + test_raw[c2].fillna("MISSING").astype(str)
        )

        le_pair = LabelEncoder()
        full_pair_vals = pd.concat([train_pair, test_pair], axis=0).unique()
        le_pair.fit(full_pair_vals)
        cat_train_df[f"{pair_col}_le"] = le_pair.transform(train_pair).astype(
            np.float32
        )
        cat_test_df[f"{pair_col}_le"] = le_pair.transform(test_pair).astype(np.float32)

        pair_freq_map = train_pair.value_counts(normalize=True).to_dict()
        cat_train_df[f"{pair_col}_freq"] = (
            train_pair.map(pair_freq_map).fillna(0.0).astype(np.float32)
        )
        cat_test_df[f"{pair_col}_freq"] = (
            test_pair.map(pair_freq_map).fillna(0.0).astype(np.float32)
        )

        smoothing_pair = 10.0
        for cls_idx in range(n_classes):
            te_pair_col_name = f"{pair_col}_te_class_{cls_idx}"
            cat_train_df[te_pair_col_name] = 0.0
            cat_test_df[te_pair_col_name] = 0.0

            global_prior = (train_raw["target"] == cls_idx).mean()

            for fold in range(n_splits):
                trn_idx = train_raw["fold"] != fold
                vld_idx = train_raw["fold"] == fold

                trn_cat = train_pair.loc[trn_idx]
                trn_tgt = (train_raw["target"].loc[trn_idx] == cls_idx).astype(float)

                stats = trn_tgt.groupby(trn_cat).agg(["count", "mean"])
                smooth_te = (
                    stats["count"] * stats["mean"] + smoothing_pair * global_prior
                ) / (stats["count"] + smoothing_pair)
                te_dict = smooth_te.to_dict()

                vld_cat = train_pair.loc[vld_idx]
                cat_train_df.loc[vld_idx, te_pair_col_name] = (
                    vld_cat.map(te_dict).fillna(global_prior).astype(np.float32)
                )

            full_stats = (
                (train_raw["target"] == cls_idx)
                .astype(float)
                .groupby(train_pair)
                .agg(["count", "mean"])
            )
            full_smooth_te = (
                full_stats["count"] * full_stats["mean"]
                + smoothing_pair * global_prior
            ) / (full_stats["count"] + smoothing_pair)
            full_te_dict = full_smooth_te.to_dict()
            cat_test_df[te_pair_col_name] = (
                test_pair.map(full_te_dict).fillna(global_prior).astype(np.float32)
            )

    # 6. Group Aggregations between Top Categoricals and Numerics
    agg_train_dfs = []
    agg_test_dfs = []

    if len(num_cols) > 0 and len(cat_cols) > 0:
        top_cats = sorted(cat_cols, key=lambda c: train_raw[c].nunique())[:3]
        for ccol in top_cats:
            for ncol in num_cols[:3]:
                train_grp = (
                    train_raw.groupby(train_raw[ccol].fillna("MISSING").astype(str))[
                        ncol
                    ]
                    .agg(["mean", "std"])
                    .reset_index()
                )
                train_grp.columns = [
                    ccol,
                    f"{ccol}_{ncol}_grp_mean",
                    f"{ccol}_{ncol}_grp_std",
                ]

                m_trn = (
                    train_raw[[ccol]]
                    .fillna("MISSING")
                    .astype(str)
                    .merge(train_grp, on=ccol, how="left")
                )
                m_tst = (
                    test_raw[[ccol]]
                    .fillna("MISSING")
                    .astype(str)
                    .merge(train_grp, on=ccol, how="left")
                )

                agg_trn_df = pd.DataFrame(index=train_raw.index)
                agg_tst_df = pd.DataFrame(index=test_raw.index)

                agg_trn_df[f"{ccol}_{ncol}_grp_mean"] = (
                    m_trn[f"{ccol}_{ncol}_grp_mean"].fillna(0.0).astype(np.float32)
                )
                agg_trn_df[f"{ccol}_{ncol}_grp_std"] = (
                    m_trn[f"{ccol}_{ncol}_grp_std"].fillna(0.0).astype(np.float32)
                )
                agg_tst_df[f"{ccol}_{ncol}_grp_mean"] = (
                    m_tst[f"{ccol}_{ncol}_grp_mean"].fillna(0.0).astype(np.float32)
                )
                agg_tst_df[f"{ccol}_{ncol}_grp_std"] = (
                    m_tst[f"{ccol}_{ncol}_grp_std"].fillna(0.0).astype(np.float32)
                )

                agg_train_dfs.append(agg_trn_df)
                agg_test_dfs.append(agg_tst_df)

    # 7. Assemble Feature Matrices
    X_train = pd.concat(
        [train_base] + text_feature_dfs_train + [cat_train_df] + agg_train_dfs, axis=1
    )
    X_test = pd.concat(
        [test_base] + text_feature_dfs_test + [cat_test_df] + agg_test_dfs, axis=1
    )

    X_train = X_train.loc[:, ~X_train.columns.duplicated()].fillna(0.0)
    X_test = X_test.loc[:, ~X_test.columns.duplicated()].fillna(0.0)

    for col in X_train.columns:
        if col not in X_test.columns:
            X_test[col] = 0.0
    X_test = X_test[X_train.columns].fillna(0.0)

    import re
    clean_cols = [re.sub(r'[\[\]<>,: "\']', '_', str(c)) for c in X_train.columns]
    X_train.columns = clean_cols
    X_test.columns = clean_cols

    final_feature_names = list(X_train.columns)
    with open("./working/feature_names.json", "w") as f:
        json.dump(final_feature_names, f, indent=2)

    y = train_raw["target"].values
    folds = train_raw["fold"].values

    n_train = len(X_train)
    n_test = len(X_test)
    print(
        f"Engineered feature set ready: {len(final_feature_names)} features for {n_train} train and {n_test} test instances."
    )

    # 8. Model Training & Evaluation
    oof_lgb = np.zeros((n_train, n_classes), dtype=np.float32)
    oof_cb = np.zeros((n_train, n_classes), dtype=np.float32)
    oof_xgb = np.zeros((n_train, n_classes), dtype=np.float32)

    test_lgb = np.zeros((n_test, n_classes), dtype=np.float32)
    test_cb = np.zeros((n_test, n_classes), dtype=np.float32)
    test_xgb = np.zeros((n_test, n_classes), dtype=np.float32)

    lgb_params = {
        "objective": "multiclass",
        "num_class": n_classes,
        "metric": "multi_logloss",
        "boosting_type": "gbdt",
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": -1,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l1": 0.1,
        "lambda_l2": 1.0,
        "min_child_samples": 20,
        "verbosity": -1,
        "random_state": 42,
        "n_jobs": -1,
    }

    for fold in range(n_splits):
        trn_mask = folds != fold
        val_mask = folds == fold

        X_tr, y_tr = X_train.iloc[trn_mask], y[trn_mask]
        X_va, y_va = X_train.iloc[val_mask], y[val_mask]

        # LightGBM
        dtrain = lgb.Dataset(X_tr, label=y_tr)
        dval = lgb.Dataset(X_va, label=y_va, reference=dtrain)

        lgb_model = lgb.train(
            lgb_params,
            dtrain,
            num_boost_round=1500,
            valid_sets=[dtrain, dval],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )

        val_preds_lgb = lgb_model.predict(X_va, num_iteration=lgb_model.best_iteration)
        oof_lgb[val_mask] = val_preds_lgb
        test_lgb += (
            lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration) / n_splits
        )
        joblib.dump(lgb_model, f"./working/lgb_fold_{fold}.joblib")

        # CatBoost
        cb_model = CatBoostClassifier(
            iterations=1200,
            learning_rate=0.04,
            depth=6,
            loss_function="MultiClass",
            eval_metric="MultiClass",
            random_seed=42 + fold,
            early_stopping_rounds=50,
            verbose=False,
            thread_count=-1,
        )
        cb_model.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=False)

        val_preds_cb = cb_model.predict_proba(X_va)
        oof_cb[val_mask] = val_preds_cb
        test_cb += cb_model.predict_proba(X_test) / n_splits
        joblib.dump(cb_model, f"./working/cb_fold_{fold}.joblib")

        # XGBoost
        xgb_model = xgb.XGBClassifier(
            n_estimators=1000,
            learning_rate=0.03,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            objective="multi:softprob",
            eval_metric="mlogloss",
            early_stopping_rounds=50,
            random_state=42 + fold,
            tree_method="hist",
            n_jobs=-1,
        )
        xgb_model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

        val_preds_xgb = xgb_model.predict_proba(X_va)
        oof_xgb[val_mask] = val_preds_xgb
        test_xgb += xgb_model.predict_proba(X_test) / n_splits
        joblib.dump(xgb_model, f"./working/xgb_fold_{fold}.joblib")

        del dtrain, dval, lgb_model, cb_model, xgb_model
        gc.collect()

    acc_lgb = accuracy_score(y, np.argmax(oof_lgb, axis=1))
    acc_cb = accuracy_score(y, np.argmax(oof_cb, axis=1))
    acc_xgb = accuracy_score(y, np.argmax(oof_xgb, axis=1))
    print(
        f"Individual OOF Accuracies -> LightGBM: {acc_lgb:.5f} | CatBoost: {acc_cb:.5f} | XGBoost: {acc_xgb:.5f}"
    )

    # 9. Optimal Ensemble Blending via SLSQP Optimization
    def blend_objective(weights):
        w = np.array(weights, dtype=np.float64)
        w_sum = np.sum(w)
        if w_sum > 0:
            w = w / w_sum
        blend = w[0] * oof_lgb + w[1] * oof_cb + w[2] * oof_xgb
        blend = np.clip(blend, 1e-15, 1.0 - 1e-15)
        blend = blend / np.sum(blend, axis=1, keepdims=True)
        return log_loss(y, blend)

    cons = {"type": "eq", "fun": lambda w: 1.0 - np.sum(w)}
    bounds = [(0.0, 1.0) for _ in range(3)]
    init_weights = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]

    opt_res = minimize(
        blend_objective,
        init_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
        options={"ftol": 1e-9, "maxiter": 500},
    )

    opt_weights = np.maximum(0.0, opt_res.x)
    opt_weights = opt_weights / np.sum(opt_weights)
    w_lgb, w_cb, w_xgb = opt_weights[0], opt_weights[1], opt_weights[2]
    print(
        f"Optimized Ensemble Weights -> LightGBM: {w_lgb:.4f}, CatBoost: {w_cb:.4f}, XGBoost: {w_xgb:.4f}"
    )

    oof_ensemble = w_lgb * oof_lgb + w_cb * oof_cb + w_xgb * oof_xgb
    test_ensemble = w_lgb * test_lgb + w_cb * test_cb + w_xgb * test_xgb

    final_oof_predictions = np.argmax(oof_ensemble, axis=1)
    final_score = accuracy_score(y, final_oof_predictions)

    # 10. Generate Final Submission
    test_pred_indices = np.argmax(test_ensemble, axis=1)
    test_pred_labels = target_encoder.inverse_transform(test_pred_indices)

    submission_df = pd.DataFrame(
        {"id": np.arange(len(test_pred_labels)), "outcomes": test_pred_labels}
    )
    submission_path = "./submission/submission.csv"
    submission_df.to_csv(submission_path, index=False)

    print(f"Submission generated at {submission_path} with {len(submission_df)} rows.")
    print(f"Final Validation Score: {final_score}")


if __name__ == "__main__":
    main()
