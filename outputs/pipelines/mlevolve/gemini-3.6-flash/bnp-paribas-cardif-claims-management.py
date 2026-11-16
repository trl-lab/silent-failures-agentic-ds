import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
import xgboost as xgb
import catboost as cb


def evaluate_log_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes bounded binary log-loss matching official evaluation metric."""
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1.0 - eps)
    return float(log_loss(y_true, y_pred))


def main():
    print("Starting pipeline execution...")
    os.makedirs("./working", exist_ok=True)
    os.makedirs("./submission", exist_ok=True)

    # 1. Load Parquet Datasets
    train_df = pd.read_parquet("./input/train.parquet")
    test_df = pd.read_parquet("./input/test.parquet")

    target_col = "target"
    id_cols = [c for c in ["id", "ID"] if c in train_df.columns]

    y_train = train_df[target_col].values.astype(np.float32)
    n_train = len(train_df)
    n_test = len(test_df)

    # 2. Categorical and Numerical Column Identification
    cat_cols = [
        c
        for c in train_df.columns
        if c not in [target_col] + id_cols
        and (
            train_df[c].dtype == "object"
            or str(train_df[c].dtype) in ["category", "string"]
        )
    ]
    num_cols = [
        c
        for c in train_df.columns
        if c not in [target_col] + id_cols and c not in cat_cols
    ]

    # 3. Missingness & Row Summary Features
    train_df["num_nulls"] = (
        train_df[num_cols + cat_cols].isnull().sum(axis=1).astype(np.int32)
    )
    test_df["num_nulls"] = (
        test_df[num_cols + cat_cols].isnull().sum(axis=1).astype(np.int32)
    )

    # Missingness Fingerprint
    all_null_cols = num_cols + cat_cols
    missing_str_tr = train_df[all_null_cols].isnull().astype(int).astype(str).agg("".join, axis=1)
    missing_str_te = test_df[all_null_cols].isnull().astype(int).astype(str).agg("".join, axis=1)
    combined_missing = pd.concat([missing_str_tr, missing_str_te], axis=0)
    missing_codes, _ = pd.factorize(combined_missing)
    train_df["missing_fingerprint"] = missing_codes[:n_train].astype(np.int32)
    test_df["missing_fingerprint"] = missing_codes[n_train:].astype(np.int32)

    high_null_cols = [
        c for c in num_cols + cat_cols if train_df[c].isnull().mean() > 0.2
    ]
    for col in high_null_cols[:15]:
        train_df[f"{col}_isna"] = train_df[col].isnull().astype(np.uint8)
        test_df[f"{col}_isna"] = test_df[col].isnull().astype(np.uint8)

    train_df["num_mean"] = train_df[num_cols].mean(axis=1).astype(np.float32)
    test_df["num_mean"] = test_df[num_cols].mean(axis=1).astype(np.float32)
    train_df["num_std"] = train_df[num_cols].std(axis=1).astype(np.float32)
    test_df["num_std"] = test_df[num_cols].std(axis=1).astype(np.float32)

    # 4. Collinear Feature Interactions
    sample_num = train_df[num_cols].fillna(train_df[num_cols].median())
    corr_matrix = sample_num.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    collinear_pairs = []
    for col in upper.columns:
        high_corr_targets = upper.index[upper[col] > 0.92].tolist()
        for target_pair in high_corr_targets:
            collinear_pairs.append((col, target_pair))
            if len(collinear_pairs) >= 10:
                break
        if len(collinear_pairs) >= 10:
            break

    for col1, col2 in collinear_pairs:
        diff_col = f"{col1}_minus_{col2}"
        ratio_col = f"{col1}_ratio_{col2}"

        train_df[diff_col] = (train_df[col1] - train_df[col2]).astype(np.float32)
        test_df[diff_col] = (test_df[col1] - test_df[col2]).astype(np.float32)

        train_df[ratio_col] = (
            (train_df[col1] + 1e-5) / (train_df[col2] + 1e-5)
        ).astype(np.float32)
        test_df[ratio_col] = ((test_df[col1] + 1e-5) / (test_df[col2] + 1e-5)).astype(
            np.float32
        )

    # 5. Categorical Processing & Frequency Encodings & Ordinal Encoding
    combined_cat = pd.concat([train_df[cat_cols], test_df[cat_cols]], axis=0)

    for col in cat_cols:
        train_df[col] = train_df[col].fillna("MISSING").astype(str)
        test_df[col] = test_df[col].fillna("MISSING").astype(str)

        freq = combined_cat[col].value_counts(normalize=True).to_dict()
        train_df[f"{col}_freq"] = train_df[col].map(freq).fillna(0).astype(np.float32)
        test_df[f"{col}_freq"] = test_df[col].map(freq).fillna(0).astype(np.float32)

        # Integer Ordinal Encoding
        col_combined = pd.concat([train_df[col], test_df[col]], axis=0)
        ord_codes, _ = pd.factorize(col_combined)
        train_df[f"{col}_ord"] = ord_codes[:n_train].astype(np.int32)
        test_df[f"{col}_ord"] = ord_codes[n_train:].astype(np.int32)

    high_card_cats = [c for c in cat_cols if train_df[c].nunique() > 50]
    for col in high_card_cats:
        train_df[f"{col}_len"] = train_df[col].apply(len).astype(np.int32)
        test_df[f"{col}_len"] = test_df[col].apply(len).astype(np.int32)

        train_df[f"{col}_char0"] = train_df[col].apply(
            lambda x: str(x)[0] if len(str(x)) > 0 else "X"
        )
        test_df[f"{col}_char0"] = test_df[col].apply(
            lambda x: str(x)[0] if len(str(x)) > 0 else "X"
        )

    if len(cat_cols) >= 2:
        c1, c2 = cat_cols[0], cat_cols[1]
        pair_train = train_df[c1] + "_" + train_df[c2]
        pair_test = test_df[c1] + "_" + test_df[c2]
        pair_combined = pd.concat([pair_train, pair_test], axis=0)
        pair_freq = pair_combined.value_counts(normalize=True).to_dict()

        train_df[f"{c1}_{c2}_pair_freq"] = (
            pair_train.map(pair_freq).fillna(0).astype(np.float32)
        )
        test_df[f"{c1}_{c2}_pair_freq"] = (
            pair_test.map(pair_freq).fillna(0).astype(np.float32)
        )

    key_cat = cat_cols[0] if len(cat_cols) > 0 else None
    if key_cat and len(num_cols) >= 2:
        for num_feat in num_cols[:4]:
            grp_mean = train_df.groupby(key_cat)[num_feat].mean().to_dict()
            grp_std = train_df.groupby(key_cat)[num_feat].std().to_dict()

            train_df[f"{num_feat}_by_{key_cat}_mean"] = (
                train_df[key_cat].map(grp_mean).fillna(0).astype(np.float32)
            )
            test_df[f"{num_feat}_by_{key_cat}_mean"] = (
                test_df[key_cat].map(grp_mean).fillna(0).astype(np.float32)
            )

            train_df[f"{num_feat}_by_{key_cat}_std"] = (
                train_df[key_cat].map(grp_std).fillna(0).astype(np.float32)
            )
            test_df[f"{num_feat}_by_{key_cat}_std"] = (
                test_df[key_cat].map(grp_std).fillna(0).astype(np.float32)
            )

    # 6. Stratified K-Fold Split & Out-Of-Fold Multi-Smoothing Target Encoding
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    folds = list(skf.split(train_df, y_train))

    prior = float(y_train.mean())
    smoothings = [5.0, 25.0]

    for col in cat_cols:
        for smoothing in smoothings:
            s_int = int(smoothing)
            oof_te = np.zeros(n_train, dtype=np.float32)
            for tr_idx, val_idx in folds:
                tr_col = train_df.iloc[tr_idx][col]
                y_tr_sub = y_train[tr_idx]

                df_tr_sub = pd.DataFrame({"cat": tr_col, "target": y_tr_sub})
                grp = df_tr_sub.groupby("cat")["target"].agg(["count", "sum"])
                smooth_vals = (grp["sum"] + smoothing * prior) / (grp["count"] + smoothing)
                stats = smooth_vals.to_dict()

                val_col = train_df.iloc[val_idx][col]
                oof_te[val_idx] = val_col.map(stats).fillna(prior).values.astype(np.float32)

            train_df[f"{col}_te_s{s_int}"] = oof_te

            df_full = pd.DataFrame({"cat": train_df[col], "target": y_train})
            grp_full = df_full.groupby("cat")["target"].agg(["count", "sum"])
            full_smooth = (grp_full["sum"] + smoothing * prior) / (grp_full["count"] + smoothing)
            full_stats = full_smooth.to_dict()

            test_df[f"{col}_te_s{s_int}"] = (
                test_df[col].map(full_stats).fillna(prior).astype(np.float32)
            )

    # 7. Prepare Feature Columns and Dtypes
    drop_cols = [target_col] + id_cols
    feature_cols = [c for c in train_df.columns if c not in drop_cols]

    cat_cols_all = [
        c
        for c in feature_cols
        if train_df[c].dtype == "object"
        or str(train_df[c].dtype) in ["category", "string"]
    ]

    train_cat_str = train_df[feature_cols].copy()
    test_cat_str = test_df[feature_cols].copy()
    for col in cat_cols_all:
        train_cat_str[col] = train_cat_str[col].astype(str)
        test_cat_str[col] = test_cat_str[col].astype(str)

    # Convert string/categorical columns to numeric integer codes for LightGBM & XGBoost
    for col in cat_cols_all:
        train_df[col] = train_df[col].astype("category").cat.codes.astype(np.int32)
        test_df[col] = test_df[col].astype("category").cat.codes.astype(np.int32)

    # 8. Cross-Validation Model Training
    oof_lgb = np.zeros(n_train, dtype=np.float32)
    oof_xgb = np.zeros(n_train, dtype=np.float32)
    oof_cat = np.zeros(n_train, dtype=np.float32)

    test_preds_lgb = np.zeros(n_test, dtype=np.float32)
    test_preds_xgb = np.zeros(n_test, dtype=np.float32)
    test_preds_cat = np.zeros(n_test, dtype=np.float32)

    for fold, (train_idx, val_idx) in enumerate(folds):
        X_tr, y_tr = train_df.iloc[train_idx][feature_cols], y_train[train_idx]
        X_val, y_val = train_df.iloc[val_idx][feature_cols], y_train[val_idx]
        X_test = test_df[feature_cols]

        # Feature subset for LightGBM and XGBoost excluding raw categorical integer codes
        num_feature_cols = [c for c in feature_cols if c not in cat_cols]

        X_tr_num = train_df.iloc[train_idx][num_feature_cols]
        X_val_num = train_df.iloc[val_idx][num_feature_cols]
        X_test_num = test_df[num_feature_cols]

        # --- A. LightGBM ---
        lgb_params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "learning_rate": 0.02,
            "num_leaves": 63,
            "max_depth": 6,
            "feature_fraction": 0.65,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "min_child_samples": 25,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "random_state": 42 + fold,
            "n_jobs": -1,
            "verbose": -1,
        }

        dtrain = lgb.Dataset(X_tr_num, label=y_tr)
        dval = lgb.Dataset(X_val_num, label=y_val, reference=dtrain)
        callbacks = [
            lgb.early_stopping(stopping_rounds=100, verbose=False),
            lgb.log_evaluation(period=-1),
        ]

        model_lgb = lgb.train(
            lgb_params,
            dtrain,
            num_boost_round=3000,
            valid_sets=[dval],
            callbacks=callbacks,
        )

        val_p_lgb = model_lgb.predict(X_val_num)
        oof_lgb[val_idx] = val_p_lgb
        test_preds_lgb += model_lgb.predict(X_test_num) / len(folds)

        # --- B. XGBoost ---
        model_xgb = xgb.XGBClassifier(
            n_estimators=3000,
            learning_rate=0.02,
            max_depth=6,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.65,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42 + fold,
            tree_method="hist",
            early_stopping_rounds=100,
            eval_metric="logloss",
            n_jobs=-1,
        )

        model_xgb.fit(X_tr_num, y_tr, eval_set=[(X_val_num, y_val)], verbose=False)
        val_p_xgb = model_xgb.predict_proba(X_val_num)[:, 1]
        oof_xgb[val_idx] = val_p_xgb
        test_preds_xgb += model_xgb.predict_proba(X_test_num)[:, 1] / len(folds)

        # --- C. CatBoost ---
        X_tr_cb = train_cat_str.iloc[train_idx]
        X_val_cb = train_cat_str.iloc[val_idx]
        X_test_cb = test_cat_str

        model_cat = cb.CatBoostClassifier(
            iterations=3000,
            learning_rate=0.02,
            depth=6,
            rsm=0.65,
            l2_leaf_reg=3.0,
            random_seed=42 + fold,
            eval_metric="Logloss",
            loss_function="Logloss",
            early_stopping_rounds=100,
            verbose=0,
            thread_count=-1,
        )

        model_cat.fit(
            X_tr_cb, y_tr, eval_set=(X_val_cb, y_val), cat_features=cat_cols_all
        )
        val_p_cat = model_cat.predict_proba(X_val_cb)[:, 1]
        oof_cat[val_idx] = val_p_cat
        test_preds_cat += model_cat.predict_proba(X_test_cb)[:, 1] / len(folds)

        fold_loss_lgb = evaluate_log_loss(y_val, val_p_lgb)
        fold_loss_xgb = evaluate_log_loss(y_val, val_p_xgb)
        fold_loss_cat = evaluate_log_loss(y_val, val_p_cat)
        print(
            f"Fold {fold + 1} LogLoss | LightGBM: {fold_loss_lgb:.5f} | XGBoost: {fold_loss_xgb:.5f} | CatBoost: {fold_loss_cat:.5f}"
        )

    # 9. Ensemble Weight Optimization with Nelder-Mead and Logit-Space Ensembling
    from scipy.special import logit, expit

    eps = 1e-15
    oof_lgb_c = np.clip(oof_lgb, eps, 1.0 - eps)
    oof_xgb_c = np.clip(oof_xgb, eps, 1.0 - eps)
    oof_cat_c = np.clip(oof_cat, eps, 1.0 - eps)

    logit_lgb = logit(oof_lgb_c)
    logit_xgb = logit(oof_xgb_c)
    logit_cat = logit(oof_cat_c)

    # Prob-space Nelder-Mead optimization
    def prob_objective(weights):
        w = np.array(weights)
        s = np.sum(np.abs(w))
        if s < 1e-8:
            return 1e9
        w = w / s
        blend = w[0] * oof_lgb_c + w[1] * oof_xgb_c + w[2] * oof_cat_c
        return evaluate_log_loss(y_train, blend)

    # Logit-space Nelder-Mead optimization
    def logit_objective(weights):
        w = np.array(weights)
        s = np.sum(np.sum(np.abs(w)))
        if s < 1e-8:
            return 1e9
        w = w / s
        blend_logit = w[0] * logit_lgb + w[1] * logit_xgb + w[2] * logit_cat
        blend = expit(blend_logit)
        return evaluate_log_loss(y_train, blend)

    res_prob = minimize(
        prob_objective,
        [0.2, 0.2, 0.6],
        method="Nelder-Mead",
        options={"maxiter": 1000, "xatol": 1e-5, "fatol": 1e-5},
    )

    res_logit = minimize(
        logit_objective,
        [0.2, 0.2, 0.6],
        method="Nelder-Mead",
        options={"maxiter": 1000, "xatol": 1e-5, "fatol": 1e-5},
    )

    score_prob = prob_objective(res_prob.x)
    score_logit = logit_objective(res_logit.x)

    test_lgb_c = np.clip(test_preds_lgb, eps, 1.0 - eps)
    test_xgb_c = np.clip(test_preds_xgb, eps, 1.0 - eps)
    test_cat_c = np.clip(test_preds_cat, eps, 1.0 - eps)

    if score_logit < score_prob:
        w = res_logit.x / np.sum(np.abs(res_logit.x))
        oof_ensemble = expit(w[0] * logit_lgb + w[1] * logit_xgb + w[2] * logit_cat)
        final_test_preds = expit(
            w[0] * logit(test_lgb_c)
            + w[1] * logit(test_xgb_c)
            + w[2] * logit(test_cat_c)
        )
        print(f"Logit-space ensembling selected with CV LogLoss: {score_logit:.5f}")
    else:
        w = res_prob.x / np.sum(np.abs(res_prob.x))
        oof_ensemble = w[0] * oof_lgb_c + w[1] * oof_xgb_c + w[2] * oof_cat_c
        final_test_preds = w[0] * test_lgb_c + w[1] * test_xgb_c + w[2] * test_cat_c
        print(f"Prob-space ensembling selected with CV LogLoss: {score_prob:.5f}")

    final_score = evaluate_log_loss(y_train, oof_ensemble)

    eps = 1e-15
    final_test_preds = np.clip(final_test_preds, eps, 1.0 - eps)

    submission_path = "./submission/submission.csv"
    sub_df = pd.DataFrame(
        {"id": np.arange(len(final_test_preds), dtype=int), "target": final_test_preds}
    )
    sub_df.to_csv(submission_path, index=False)

    print(f"Final Validation Score: {final_score}")


if __name__ == "__main__":
    main()