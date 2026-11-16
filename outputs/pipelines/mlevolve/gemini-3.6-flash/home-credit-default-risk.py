import os
import warnings
import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb

warnings.filterwarnings("ignore")


def main():
    # ---------------------------------------------------------
    # 1. Load Raw Parquet Data
    # ---------------------------------------------------------
    train_path = "./input/train.parquet"
    test_path = "./input/test.parquet"

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError("Input files train.parquet or test.parquet not found.")

    print("Loading raw parquet data...")
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    if "TARGET" not in train_df.columns:
        raise ValueError("TARGET column missing in training data.")

    target = train_df["TARGET"].values
    n_train = len(train_df)
    n_test = len(test_df)

    train_ids = (
        train_df["SK_ID_CURR"].values
        if "SK_ID_CURR" in train_df.columns
        else np.arange(n_train)
    )
    test_ids = (
        test_df["SK_ID_CURR"].values
        if "SK_ID_CURR" in test_df.columns
        else np.arange(n_test)
    )

    # Combine feature matrices temporarily for uniform transformation
    df = pd.concat(
        [
            train_df.drop(columns=["TARGET"], errors="ignore"),
            test_df.drop(columns=["TARGET"], errors="ignore"),
        ],
        axis=0,
        ignore_index=True,
    )

    # ---------------------------------------------------------
    # 2. Data Cleaning & Anomaly Correction
    # ---------------------------------------------------------
    if "DAYS_EMPLOYED" in df.columns:
        df["DAYS_EMPLOYED_ANOM"] = (df["DAYS_EMPLOYED"] == 365243).astype(int)
        df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(365243, np.nan)

    for col in [
        "DAYS_BIRTH",
        "DAYS_EMPLOYED",
        "DAYS_REGISTRATION",
        "DAYS_ID_PUBLISH",
        "DAYS_LAST_PHONE_CHANGE",
    ]:
        if col in df.columns:
            df[f"{col}_ABS"] = df[col].abs()

    # ---------------------------------------------------------
    # 3. Domain Financial Ratios & Liquidity Metrics
    # ---------------------------------------------------------
    if all(col in df.columns for col in ["AMT_CREDIT", "AMT_INCOME_TOTAL"]):
        df["CREDIT_INCOME_PERCENT"] = df["AMT_CREDIT"] / (df["AMT_INCOME_TOTAL"] + 1e-5)
    if all(col in df.columns for col in ["AMT_ANNUITY", "AMT_INCOME_TOTAL"]):
        df["ANNUITY_INCOME_PERCENT"] = df["AMT_ANNUITY"] / (
            df["AMT_INCOME_TOTAL"] + 1e-5
        )
    if all(col in df.columns for col in ["AMT_ANNUITY", "AMT_CREDIT"]):
        df["CREDIT_TERM"] = df["AMT_ANNUITY"] / (df["AMT_CREDIT"] + 1e-5)
    if all(col in df.columns for col in ["AMT_GOODS_PRICE", "AMT_CREDIT"]):
        df["DOWNPAYMENT_DIFF"] = df["AMT_CREDIT"] - df["AMT_GOODS_PRICE"]
        df["GOODS_CREDIT_PERCENT"] = df["AMT_GOODS_PRICE"] / (df["AMT_CREDIT"] + 1e-5)
    if all(col in df.columns for col in ["AMT_GOODS_PRICE", "AMT_INCOME_TOTAL"]):
        df["GOODS_INCOME_PERCENT"] = df["AMT_GOODS_PRICE"] / (
            df["AMT_INCOME_TOTAL"] + 1e-5
        )
    if all(col in df.columns for col in ["DAYS_EMPLOYED", "DAYS_BIRTH"]):
        df["DAYS_EMPLOYED_PERCENT"] = df["DAYS_EMPLOYED"] / (df["DAYS_BIRTH"] + 1e-5)
        df["ADULT_WORKING_LIFE_PROP"] = df["DAYS_EMPLOYED"] / (
            df["DAYS_BIRTH"] + 18 * 365.25 - 1e-5
        )
    if all(col in df.columns for col in ["AMT_ANNUITY", "AMT_GOODS_PRICE"]):
        df["ANNUITY_GOODS_RATIO"] = df["AMT_ANNUITY"] / (
            df["AMT_GOODS_PRICE"] + 1e-5
        )
    if all(
        col in df.columns
        for col in ["REGION_RATING_CLIENT", "REGION_RATING_CLIENT_W_CITY"]
    ):
        df["REGION_RATING_DISCREPANCY"] = (
            df["REGION_RATING_CLIENT"] - df["REGION_RATING_CLIENT_W_CITY"]
        )
    if all(col in df.columns for col in ["DAYS_REGISTRATION", "DAYS_BIRTH"]):
        df["DAYS_REGISTRATION_PERCENT"] = df["DAYS_REGISTRATION"] / (
            df["DAYS_BIRTH"] + 1e-5
        )
    if all(col in df.columns for col in ["DAYS_ID_PUBLISH", "DAYS_BIRTH"]):
        df["DAYS_ID_PUBLISH_PERCENT"] = df["DAYS_ID_PUBLISH"] / (
            df["DAYS_BIRTH"] + 1e-5
        )
    if all(col in df.columns for col in ["DAYS_LAST_PHONE_CHANGE", "DAYS_BIRTH"]):
        df["DAYS_LAST_PHONE_CHANGE_PERCENT"] = df["DAYS_LAST_PHONE_CHANGE"] / (
            df["DAYS_BIRTH"] + 1e-5
        )
    if all(col in df.columns for col in ["AMT_INCOME_TOTAL", "CNT_FAM_MEMBERS"]):
        df["INCOME_PER_PERSON"] = df["AMT_INCOME_TOTAL"] / (
            df["CNT_FAM_MEMBERS"].fillna(1) + 1e-5
        )
    if all(col in df.columns for col in ["AMT_INCOME_TOTAL", "CNT_CHILDREN"]):
        df["INCOME_PER_CHILD"] = df["AMT_INCOME_TOTAL"] / (df["CNT_CHILDREN"] + 1)

    if "DAYS_BIRTH" in df.columns:
        df["AGE_YEARS"] = df["DAYS_BIRTH"].abs() / 365.25
    if "DAYS_EMPLOYED" in df.columns:
        df["EMPLOYED_YEARS"] = df["DAYS_EMPLOYED"].abs() / 365.25

    # Document & Contact Completeness Scores
    doc_cols = [c for c in df.columns if c.startswith("FLAG_DOCUMENT_")]
    if doc_cols:
        df["DOCUMENT_COMPLETENESS_SUM"] = df[doc_cols].sum(axis=1)

    contact_cols = [
        c
        for c in [
            "FLAG_MOBIL",
            "FLAG_EMP_PHONE",
            "FLAG_WORK_PHONE",
            "FLAG_CONT_MOBILE",
            "FLAG_PHONE",
            "FLAG_EMAIL",
        ]
        if c in df.columns
    ]
    if contact_cols:
        df["CONTACT_COMPLETENESS_SUM"] = df[contact_cols].sum(axis=1)

    # Credit Bureau Query Dynamics
    bureau_cols = [
        c
        for c in [
            "AMT_REQ_CREDIT_BUREAU_HOUR",
            "AMT_REQ_CREDIT_BUREAU_DAY",
            "AMT_REQ_CREDIT_BUREAU_WEEK",
            "AMT_REQ_CREDIT_BUREAU_MON",
            "AMT_REQ_CREDIT_BUREAU_QRT",
            "AMT_REQ_CREDIT_BUREAU_YEAR",
        ]
        if c in df.columns
    ]
    if "AMT_REQ_CREDIT_BUREAU_YEAR" in df.columns:
        year_denom = df["AMT_REQ_CREDIT_BUREAU_YEAR"] + 1e-5
        for col in [
            "AMT_REQ_CREDIT_BUREAU_HOUR",
            "AMT_REQ_CREDIT_BUREAU_DAY",
            "AMT_REQ_CREDIT_BUREAU_WEEK",
            "AMT_REQ_CREDIT_BUREAU_MON",
            "AMT_REQ_CREDIT_BUREAU_QRT",
        ]:
            if col in df.columns:
                df[f"{col}_TO_YEAR_RATIO"] = df[col] / year_denom
    if all(
        c in df.columns
        for c in [
            "AMT_REQ_CREDIT_BUREAU_HOUR",
            "AMT_REQ_CREDIT_BUREAU_DAY",
            "AMT_REQ_CREDIT_BUREAU_WEEK",
        ]
    ):
        short_term_sum = (
            df["AMT_REQ_CREDIT_BUREAU_HOUR"]
            + df["AMT_REQ_CREDIT_BUREAU_DAY"]
            + df["AMT_REQ_CREDIT_BUREAU_WEEK"]
        )
        if "AMT_REQ_CREDIT_BUREAU_YEAR" in df.columns:
            df["BUREAU_SHORT_VS_YEAR_RATIO"] = short_term_sum / (
                df["AMT_REQ_CREDIT_BUREAU_YEAR"] + 1e-5
            )
        if bureau_cols:
            total_bureau_sum = df[bureau_cols].sum(axis=1)
            df["BUREAU_SHORT_VS_TOTAL_RATIO"] = short_term_sum / (
                total_bureau_sum + 1e-5
            )

    # Social Circle Delinquency Ratios
    if "DEF_30_CNT_SOCIAL_CIRCLE" in df.columns and "OBS_30_CNT_SOCIAL_CIRCLE" in df.columns:
        df["DEF_30_RATIO_SOCIAL_CIRCLE"] = df["DEF_30_CNT_SOCIAL_CIRCLE"] / (
            df["OBS_30_CNT_SOCIAL_CIRCLE"] + 1e-5
        )
    if "DEF_60_CNT_SOCIAL_CIRCLE" in df.columns and "OBS_60_CNT_SOCIAL_CIRCLE" in df.columns:
        df["DEF_60_RATIO_SOCIAL_CIRCLE"] = df["DEF_60_CNT_SOCIAL_CIRCLE"] / (
            df["OBS_60_CNT_SOCIAL_CIRCLE"] + 1e-5
        )

    # Interaction Categoricals
    if "CODE_GENDER" in df.columns and "NAME_EDUCATION_TYPE" in df.columns:
        df["GENDER_EDUCATION"] = (
            df["CODE_GENDER"].astype(str) + "_" + df["NAME_EDUCATION_TYPE"].astype(str)
        )
    if "NAME_INCOME_TYPE" in df.columns and "OCCUPATION_TYPE" in df.columns:
        df["INCOME_OCCUPATION"] = (
            df["NAME_INCOME_TYPE"].astype(str) + "_" + df["OCCUPATION_TYPE"].astype(str)
        )

    # ---------------------------------------------------------
    # 4. External Source Interactions & Statistics
    # ---------------------------------------------------------
    ext_cols = [
        c for c in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"] if c in df.columns
    ]
    if len(ext_cols) > 0:
        df["EXT_SOURCE_MEAN"] = df[ext_cols].mean(axis=1)
        df["EXT_SOURCE_STD"] = df[ext_cols].std(axis=1)
        df["EXT_SOURCE_MIN"] = df[ext_cols].min(axis=1)
        df["EXT_SOURCE_MAX"] = df[ext_cols].max(axis=1)
        df["EXT_SOURCE_MEDIAN"] = df[ext_cols].median(axis=1)
        df["EXT_SOURCES_COUNT_NA"] = df[ext_cols].isna().sum(axis=1)

        if "EXT_SOURCE_1" in ext_cols and "EXT_SOURCE_2" in ext_cols:
            df["EXT_1_X_EXT_2"] = df["EXT_SOURCE_1"] * df["EXT_SOURCE_2"]
            df["EXT_1_DIV_EXT_2"] = df["EXT_SOURCE_1"] / (df["EXT_SOURCE_2"] + 1e-5)
            df["EXT_2_DIV_EXT_1"] = df["EXT_SOURCE_2"] / (df["EXT_SOURCE_1"] + 1e-5)
            df["EXT_1_PLUS_EXT_2"] = df["EXT_SOURCE_1"] + df["EXT_SOURCE_2"]
            df["EXT_1_MINUS_EXT_2"] = df["EXT_SOURCE_1"] - df["EXT_SOURCE_2"]
            df["EXT_1_SQ_X_EXT_2"] = (df["EXT_SOURCE_1"] ** 2) * df["EXT_SOURCE_2"]
            df["EXT_1_X_EXT_2_SQ"] = df["EXT_SOURCE_1"] * (df["EXT_SOURCE_2"] ** 2)
        if "EXT_SOURCE_2" in ext_cols and "EXT_SOURCE_3" in ext_cols:
            df["EXT_2_X_EXT_3"] = df["EXT_SOURCE_2"] * df["EXT_SOURCE_3"]
            df["EXT_2_DIV_EXT_3"] = df["EXT_SOURCE_2"] / (df["EXT_SOURCE_3"] + 1e-5)
            df["EXT_3_DIV_EXT_2"] = df["EXT_SOURCE_3"] / (df["EXT_SOURCE_2"] + 1e-5)
            df["EXT_2_PLUS_EXT_3"] = df["EXT_SOURCE_2"] + df["EXT_SOURCE_3"]
            df["EXT_2_MINUS_EXT_3"] = df["EXT_SOURCE_2"] - df["EXT_SOURCE_3"]
            df["EXT_2_SQ_X_EXT_3"] = (df["EXT_SOURCE_2"] ** 2) * df["EXT_SOURCE_3"]
            df["EXT_2_X_EXT_3_SQ"] = df["EXT_SOURCE_2"] * (df["EXT_SOURCE_3"] ** 2)
        if "EXT_SOURCE_1" in ext_cols and "EXT_SOURCE_3" in ext_cols:
            df["EXT_1_X_EXT_3"] = df["EXT_SOURCE_1"] * df["EXT_SOURCE_3"]
            df["EXT_1_DIV_EXT_3"] = df["EXT_SOURCE_1"] / (df["EXT_SOURCE_3"] + 1e-5)
            df["EXT_3_DIV_EXT_1"] = df["EXT_SOURCE_3"] / (df["EXT_SOURCE_1"] + 1e-5)
            df["EXT_1_PLUS_EXT_3"] = df["EXT_SOURCE_1"] + df["EXT_SOURCE_3"]
            df["EXT_1_MINUS_EXT_3"] = df["EXT_SOURCE_1"] - df["EXT_SOURCE_3"]
            df["EXT_1_SQ_X_EXT_3"] = (df["EXT_SOURCE_1"] ** 2) * df["EXT_SOURCE_3"]
            df["EXT_1_X_EXT_3_SQ"] = df["EXT_SOURCE_1"] * (df["EXT_SOURCE_3"] ** 2)
        if all(col in ext_cols for col in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]):
            df["EXT_1_X_2_X_3"] = (
                df["EXT_SOURCE_1"] * df["EXT_SOURCE_2"] * df["EXT_SOURCE_3"]
            )
            df["EXT_SUM_ALL"] = (
                df["EXT_SOURCE_1"] + df["EXT_SOURCE_2"] + df["EXT_SOURCE_3"]
            )
        for col in ext_cols:
            df[f"{col}_SQR"] = df[col] ** 2

        if "CREDIT_INCOME_PERCENT" in df.columns:
            df["EXT_SOURCE_MEAN_TO_CREDIT_INCOME"] = df["EXT_SOURCE_MEAN"] / (
                df["CREDIT_INCOME_PERCENT"] + 1e-5
            )
            if "EXT_SOURCE_2" in ext_cols:
                df["EXT_SOURCE_2_TO_CREDIT_INCOME"] = df["EXT_SOURCE_2"] / (
                    df["CREDIT_INCOME_PERCENT"] + 1e-5
                )
            if "EXT_SOURCE_3" in ext_cols:
                df["EXT_SOURCE_3_TO_CREDIT_INCOME"] = df["EXT_SOURCE_3"] / (
                    df["CREDIT_INCOME_PERCENT"] + 1e-5
                )
        if "CREDIT_TERM" in df.columns:
            df["EXT_SOURCE_MEAN_TO_CREDIT_TERM"] = df["EXT_SOURCE_MEAN"] / (
                df["CREDIT_TERM"] + 1e-5
            )

        if all(col in ext_cols for col in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]):
            e1_m = df["EXT_SOURCE_1"].iloc[:n_train].mean()
            e2_m = df["EXT_SOURCE_2"].iloc[:n_train].mean()
            e3_m = df["EXT_SOURCE_3"].iloc[:n_train].mean()
            df["EXT_SOURCE_WEIGHTED"] = (
                0.1 * df["EXT_SOURCE_1"].fillna(e1_m)
                + 0.5 * df["EXT_SOURCE_2"].fillna(e2_m)
                + 0.4 * df["EXT_SOURCE_3"].fillna(e3_m)
            )
        ext_mean_tr = df["EXT_SOURCE_MEAN"].iloc[:n_train]
        ext_mean_te = df["EXT_SOURCE_MEAN"].iloc[n_train:]
        m_fill = ext_mean_tr.mean()
        tr_ranks = rankdata(ext_mean_tr.fillna(m_fill)) / n_train
        te_ranks = rankdata(ext_mean_te.fillna(m_fill)) / n_test
        df["EXT_SOURCE_MEAN_RANK"] = np.concatenate([tr_ranks, te_ranks])

    # Missingness profiles
    df["TOTAL_MISSING_COUNT"] = df.isna().sum(axis=1)
    df["TOTAL_MISSING_RATIO"] = df["TOTAL_MISSING_COUNT"] / len(df.columns)

    # ---------------------------------------------------------
    # 5. Grouped Relative Features (Train-Only Stats)
    # ---------------------------------------------------------
    train_subset = df.iloc[:n_train].copy()
    train_subset["TARGET"] = target

    group_cols = [
        c
        for c in [
            "NAME_EDUCATION_TYPE",
            "ORGANIZATION_TYPE",
            "OCCUPATION_TYPE",
            "NAME_INCOME_TYPE",
        ]
        if c in df.columns
    ]
    num_cols_for_grouping = [
        c
        for c in [
            "AMT_INCOME_TOTAL",
            "AMT_CREDIT",
            "AMT_ANNUITY",
            "EXT_SOURCE_1",
            "EXT_SOURCE_2",
            "EXT_SOURCE_3",
            "EXT_SOURCE_MEAN",
        ]
        if c in df.columns
    ]

    for g_col in group_cols:
        for n_col in num_cols_for_grouping:
            g_mean_dict = (
                train_subset.groupby(g_col, observed=False)[n_col].mean().to_dict()
            )
            g_std_dict = (
                train_subset.groupby(g_col, observed=False)[n_col].std().to_dict()
            )
            g_min_dict = (
                train_subset.groupby(g_col, observed=False)[n_col].min().to_dict()
            )
            g_max_dict = (
                train_subset.groupby(g_col, observed=False)[n_col].max().to_dict()
            )

            g_mean = df[g_col].map(g_mean_dict)
            g_std = df[g_col].map(g_std_dict)
            g_min = df[g_col].map(g_min_dict)
            g_max = df[g_col].map(g_max_dict)

            df[f"{n_col}_MEAN_BY_{g_col}"] = g_mean
            df[f"{n_col}_DIFF_FROM_{g_col}_MEAN"] = df[n_col] - g_mean
            df[f"{n_col}_Z_SCORE_BY_{g_col}"] = (df[n_col] - g_mean) / (g_std + 1e-5)
            df[f"{n_col}_MIN_MAX_RATIO_BY_{g_col}"] = (df[n_col] - g_min) / (
                g_max - g_min + 1e-5
            )

    # Frequency mapping for categorical features
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    for c in cat_cols:
        freq = train_subset[c].value_counts().to_dict()
        df[f"{c}_FREQ"] = df[c].map(freq).fillna(0)

    # ---------------------------------------------------------
    # 6. Separate Train and Test Sets & Out-of-Fold Target Encoding
    # ---------------------------------------------------------
    X_train_full = df.iloc[:n_train].copy()
    X_test_full = df.iloc[n_train:].copy()

    X_train_full["TARGET"] = target
    X_train_full["SK_ID_CURR"] = train_ids
    X_test_full["SK_ID_CURR"] = test_ids

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    X_train_full["fold"] = -1
    for fold, (_, val_idx) in enumerate(skf.split(X_train_full, target)):
        X_train_full.loc[val_idx, "fold"] = fold

    target_enc_cols = [
        c
        for c in [
            "ORGANIZATION_TYPE",
            "OCCUPATION_TYPE",
            "NAME_EDUCATION_TYPE",
            "GENDER_EDUCATION",
            "INCOME_OCCUPATION",
        ]
        if c in cat_cols
    ]
    global_mean = target.mean()
    m_smooth = 10

    for c in target_enc_cols:
        X_train_full[f"{c}_TARGET_ENC"] = np.nan
        X_test_full[f"{c}_TARGET_ENC"] = np.nan

        for fold in range(5):
            tr_mask = X_train_full["fold"] != fold
            val_mask = X_train_full["fold"] == fold

            tr_data = X_train_full[tr_mask]
            stats = tr_data.groupby(c, observed=False)["TARGET"].agg(["count", "sum"])
            smoothed = (stats["sum"] + m_smooth * global_mean) / (
                stats["count"] + m_smooth
            )
            smoothed_dict = smoothed.to_dict()

            X_train_full.loc[val_mask, f"{c}_TARGET_ENC"] = (
                X_train_full.loc[val_mask, c].map(smoothed_dict).fillna(global_mean)
            )

        overall_stats = X_train_full.groupby(c, observed=False)["TARGET"].agg(
            ["count", "sum"]
        )
        overall_smoothed = (overall_stats["sum"] + m_smooth * global_mean) / (
            overall_stats["count"] + m_smooth
        )
        overall_smoothed_dict = overall_smoothed.to_dict()

        X_test_full[f"{c}_TARGET_ENC"] = (
            X_test_full[c].map(overall_smoothed_dict).fillna(global_mean)
        )

    # Save processed working copies
    os.makedirs("./working", exist_ok=True)
    X_train_full.to_parquet("./working/train_processed.parquet", index=False)
    X_test_full.to_parquet("./working/test_processed.parquet", index=False)

    # Prepare feature sets for modeling
    drop_cols = ["TARGET", "SK_ID_CURR", "fold"]
    feature_cols = [c for c in X_train_full.columns if c not in drop_cols]

    X_tr_df = X_train_full[feature_cols].copy()
    X_te_df = X_test_full[feature_cols].copy()

    # Categorical encoding consistency between train & test
    for col in cat_cols:
        combined = pd.concat([X_tr_df[col], X_te_df[col]], axis=0).astype("category")
        X_tr_df[col] = combined.iloc[:n_train].cat.codes
        X_te_df[col] = combined.iloc[n_train:].cat.codes

    folds = X_train_full["fold"].values
    n_folds = 5

    oof_lgb = np.zeros(n_train)
    oof_cb = np.zeros(n_train)
    oof_xgb = np.zeros(n_train)

    test_preds_lgb = np.zeros(n_test)
    test_preds_cb = np.zeros(n_test)
    test_preds_xgb = np.zeros(n_test)

    # ---------------------------------------------------------
    # 7. Model Training & Validation Loop (LGB, CB, XGB)
    # ---------------------------------------------------------
    print(
        f"Training LightGBM, CatBoost, and XGBoost on {len(feature_cols)} features..."
    )

    for fold in range(n_folds):
        tr_mask = folds != fold
        val_mask = folds == fold

        X_tr, y_tr = X_tr_df.iloc[tr_mask], target[tr_mask]
        X_va, y_va = X_tr_df.iloc[val_mask], target[val_mask]

        # 1. LightGBM
        lgb_clf = lgb.LGBMClassifier(
            objective="binary",
            metric="auc",
            boosting_type="gbdt",
            n_estimators=3500,
            learning_rate=0.02,
            num_leaves=31,
            max_depth=6,
            subsample=0.85,
            colsample_bytree=0.6,
            reg_alpha=0.5,
            reg_lambda=15.0,
            min_child_samples=50,
            random_state=42 + fold,
            n_jobs=-1,
            verbose=-1,
        )
        lgb_clf.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )

        val_pred_lgb = lgb_clf.predict_proba(X_va)[:, 1]
        oof_lgb[val_mask] = val_pred_lgb
        test_preds_lgb += lgb_clf.predict_proba(X_te_df)[:, 1] / n_folds

        # 2. CatBoost
        cb_clf = cb.CatBoostClassifier(
            loss_function="Logloss",
            eval_metric="AUC",
            iterations=3500,
            learning_rate=0.02,
            depth=6,
            subsample=0.85,
            l2_leaf_reg=15.0,
            random_seed=42 + fold,
            verbose=False,
            thread_count=-1,
        )
        cb_clf.fit(
            X_tr,
            y_tr,
            eval_set=(X_va, y_va),
            early_stopping_rounds=100,
            verbose=False,
        )

        val_pred_cb = cb_clf.predict_proba(X_va)[:, 1]
        oof_cb[val_mask] = val_pred_cb
        test_preds_cb += cb_clf.predict_proba(X_te_df)[:, 1] / n_folds

        # 3. XGBoost
        xgb_clf = xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="auc",
            tree_method="hist",
            n_estimators=3500,
            learning_rate=0.02,
            max_depth=6,
            subsample=0.85,
            colsample_bytree=0.6,
            reg_alpha=0.5,
            reg_lambda=15.0,
            min_child_weight=50,
            random_state=42 + fold,
            n_jobs=-1,
            early_stopping_rounds=100,
        )
        xgb_clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

        val_pred_xgb = xgb_clf.predict_proba(X_va)[:, 1]
        oof_xgb[val_mask] = val_pred_xgb
        test_preds_xgb += xgb_clf.predict_proba(X_te_df)[:, 1] / n_folds

        rank_val_lgb = rankdata(val_pred_lgb) / len(val_pred_lgb)
        rank_val_cb = rankdata(val_pred_cb) / len(val_pred_cb)
        rank_val_xgb = rankdata(val_pred_xgb) / len(val_pred_xgb)

        blend_fold = 0.45 * rank_val_lgb + 0.35 * rank_val_cb + 0.20 * rank_val_xgb
        fold_auc = roc_auc_score(y_va, blend_fold)
        print(f"Fold {fold} - Blend ROC AUC: {fold_auc:.5f}")

    # ---------------------------------------------------------
    # 8. Out-Of-Fold Blend & Final Validation Output
    # ---------------------------------------------------------
    rank_oof_lgb = rankdata(oof_lgb) / len(oof_lgb)
    rank_oof_cb = rankdata(oof_cb) / len(oof_cb)
    rank_oof_xgb = rankdata(oof_xgb) / len(oof_xgb)

    oof_blend = 0.45 * rank_oof_lgb + 0.35 * rank_oof_cb + 0.20 * rank_oof_xgb
    overall_auc = roc_auc_score(target, oof_blend)

    # ---------------------------------------------------------
    # 9. Create Submission File
    # ---------------------------------------------------------
    os.makedirs("./submission", exist_ok=True)
    submission_path = "./submission/submission.csv"

    rank_test_lgb = rankdata(test_preds_lgb) / len(test_preds_lgb)
    rank_test_cb = rankdata(test_preds_cb) / len(test_preds_cb)
    rank_test_xgb = rankdata(test_preds_xgb) / len(test_preds_xgb)

    final_test_preds = (
        0.45 * rank_test_lgb + 0.35 * rank_test_cb + 0.20 * rank_test_xgb
    )

    submission_df = pd.DataFrame(
        {"id": np.arange(len(X_te_df)), "TARGET": final_test_preds}
    )
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path} with shape {submission_df.shape}")

    print(f"Final Validation Score: {overall_auc}")


if __name__ == "__main__":
    main()