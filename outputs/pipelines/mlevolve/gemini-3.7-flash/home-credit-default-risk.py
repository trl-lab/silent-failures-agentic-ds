import os
import re
from typing import Any, Dict, List, Optional, Tuple
from catboost import CatBoostClassifier
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Sanitize column names to remove special characters for GBDT compatibility."""
    df.columns = [re.sub(r"[ ,:\(\)\[\]\{\}\"\']", "_", str(col)) for col in df.columns]
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer domain-specific financial, demographic, and interaction features in a vectorized unfragmented batch."""
    df = df.copy()
    new_cols: Dict[str, Any] = {}

    # Missing value count indicator
    new_cols["MISSING_VALS_COUNT"] = df.isnull().sum(axis=1).astype(np.int16)

    # 1. Handle anomalous values
    if "DAYS_EMPLOYED" in df.columns:
        new_cols["DAYS_EMPLOYED_ANOM"] = (df["DAYS_EMPLOYED"] == 365243).astype(np.int8)
        df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace({365243: np.nan})

    # 2. Domain Financial Ratios
    if "AMT_ANNUITY" in df.columns and "AMT_CREDIT" in df.columns:
        new_cols["PAYMENT_RATE"] = (df["AMT_ANNUITY"] / (df["AMT_CREDIT"] + 1e-6)).astype(np.float32)
        new_cols["ANNUITY_LENGTH"] = (df["AMT_CREDIT"] / (df["AMT_ANNUITY"] + 1e-6)).astype(np.float32)

    if "AMT_INCOME_TOTAL" in df.columns and "AMT_CREDIT" in df.columns:
        new_cols["INCOME_CREDIT_PERC"] = (df["AMT_INCOME_TOTAL"] / (df["AMT_CREDIT"] + 1e-6)).astype(np.float32)
        new_cols["CREDIT_INCOME_PERC"] = (df["AMT_CREDIT"] / (df["AMT_INCOME_TOTAL"] + 1e-6)).astype(np.float32)

    if "AMT_INCOME_TOTAL" in df.columns and "AMT_ANNUITY" in df.columns:
        new_cols["ANNUITY_INCOME_PERC"] = (df["AMT_ANNUITY"] / (df["AMT_INCOME_TOTAL"] + 1e-6)).astype(np.float32)
        new_cols["INCOME_ANNUITY_DIFF"] = (df["AMT_INCOME_TOTAL"] - df["AMT_ANNUITY"]).astype(np.float32)

    if "AMT_GOODS_PRICE" in df.columns and "AMT_CREDIT" in df.columns:
        new_cols["GOODS_CREDIT_PERC"] = (df["AMT_GOODS_PRICE"] / (df["AMT_CREDIT"] + 1e-6)).astype(np.float32)
        new_cols["CREDIT_GOODS_DIFF"] = (df["AMT_CREDIT"] - df["AMT_GOODS_PRICE"]).astype(np.float32)

    if "AMT_INCOME_TOTAL" in df.columns and "CNT_FAM_MEMBERS" in df.columns:
        new_cols["INCOME_PER_PERSON"] = (
            df["AMT_INCOME_TOTAL"] / (df["CNT_FAM_MEMBERS"].fillna(1) + 1e-6)
        ).astype(np.float32)

    if "CNT_CHILDREN" in df.columns and "CNT_FAM_MEMBERS" in df.columns:
        new_cols["CHILDREN_RATIO"] = (
            df["CNT_CHILDREN"] / (df["CNT_FAM_MEMBERS"].fillna(1) + 1e-6)
        ).astype(np.float32)

    if "AMT_INCOME_TOTAL" in df.columns and "CNT_CHILDREN" in df.columns:
        new_cols["INCOME_PER_CHILD"] = (
            df["AMT_INCOME_TOTAL"] / (df["CNT_CHILDREN"].fillna(0) + 1.0)
        ).astype(np.float32)

    # 3. Demographic and Temporal Interactions
    if "DAYS_EMPLOYED" in df.columns and "DAYS_BIRTH" in df.columns:
        new_cols["DAYS_EMPLOYED_PERC"] = (df["DAYS_EMPLOYED"] / (df["DAYS_BIRTH"] + 1e-6)).astype(np.float32)

    if "DAYS_REGISTRATION" in df.columns and "DAYS_BIRTH" in df.columns:
        new_cols["DAYS_REGISTRATION_PERC"] = (
            df["DAYS_REGISTRATION"] / (df["DAYS_BIRTH"] + 1e-6)
        ).astype(np.float32)

    if "DAYS_ID_PUBLISH" in df.columns and "DAYS_BIRTH" in df.columns:
        new_cols["DAYS_ID_PUBLISH_PERC"] = (
            df["DAYS_ID_PUBLISH"] / (df["DAYS_BIRTH"] + 1e-6)
        ).astype(np.float32)

    if "OWN_CAR_AGE" in df.columns and "DAYS_BIRTH" in df.columns:
        new_cols["CAR_TO_BIRTH_RATIO"] = (
            df["OWN_CAR_AGE"] / (df["DAYS_BIRTH"].abs() / 365.25 + 1e-6)
        ).astype(np.float32)

    if "OWN_CAR_AGE" in df.columns and "DAYS_EMPLOYED" in df.columns:
        new_cols["CAR_TO_EMPLOYED_RATIO"] = (
            df["OWN_CAR_AGE"] / (df["DAYS_EMPLOYED"].abs() / 365.25 + 1e-6)
        ).astype(np.float32)

    if "DAYS_LAST_PHONE_CHANGE" in df.columns and "DAYS_BIRTH" in df.columns:
        new_cols["PHONE_TO_BIRTH_RATIO"] = (
            df["DAYS_LAST_PHONE_CHANGE"] / (df["DAYS_BIRTH"] + 1e-6)
        ).astype(np.float32)

    if "DAYS_LAST_PHONE_CHANGE" in df.columns and "DAYS_EMPLOYED" in df.columns:
        new_cols["PHONE_TO_EMPLOYED_RATIO"] = (
            df["DAYS_LAST_PHONE_CHANGE"] / (df["DAYS_EMPLOYED"] + 1e-6)
        ).astype(np.float32)

    # 4. Social Circle Default Rates
    if "DEF_30_CNT_SOCIAL_CIRCLE" in df.columns and "OBS_30_CNT_SOCIAL_CIRCLE" in df.columns:
        new_cols["DEF_OBS_30_RATIO"] = (
            df["DEF_30_CNT_SOCIAL_CIRCLE"] / (df["OBS_30_CNT_SOCIAL_CIRCLE"] + 1e-6)
        ).astype(np.float32)

    if "DEF_60_CNT_SOCIAL_CIRCLE" in df.columns and "OBS_60_CNT_SOCIAL_CIRCLE" in df.columns:
        new_cols["DEF_OBS_60_RATIO"] = (
            df["DEF_60_CNT_SOCIAL_CIRCLE"] / (df["OBS_60_CNT_SOCIAL_CIRCLE"] + 1e-6)
        ).astype(np.float32)

    if "OBS_60_CNT_SOCIAL_CIRCLE" in df.columns and "OBS_30_CNT_SOCIAL_CIRCLE" in df.columns:
        new_cols["OBS_60_30_DIFF"] = (
            df["OBS_60_CNT_SOCIAL_CIRCLE"] - df["OBS_30_CNT_SOCIAL_CIRCLE"]
        ).astype(np.float32)

    if "DEF_60_CNT_SOCIAL_CIRCLE" in df.columns and "DEF_30_CNT_SOCIAL_CIRCLE" in df.columns:
        new_cols["DEF_60_30_DIFF"] = (
            df["DEF_60_CNT_SOCIAL_CIRCLE"] - df["DEF_30_CNT_SOCIAL_CIRCLE"]
        ).astype(np.float32)

    # 5. Credit Bureau Enquiry Velocity Ratios
    bureau_cols = [
        "AMT_REQ_CREDIT_BUREAU_HOUR",
        "AMT_REQ_CREDIT_BUREAU_DAY",
        "AMT_REQ_CREDIT_BUREAU_WEEK",
        "AMT_REQ_CREDIT_BUREAU_MON",
        "AMT_REQ_CREDIT_BUREAU_QRT",
        "AMT_REQ_CREDIT_BUREAU_YEAR",
    ]
    if all(col in df.columns for col in bureau_cols):
        short_term_enq = (
            df["AMT_REQ_CREDIT_BUREAU_HOUR"].fillna(0)
            + df["AMT_REQ_CREDIT_BUREAU_DAY"].fillna(0)
            + df["AMT_REQ_CREDIT_BUREAU_WEEK"].fillna(0)
        )
        long_term_enq = (
            df["AMT_REQ_CREDIT_BUREAU_MON"].fillna(0)
            + df["AMT_REQ_CREDIT_BUREAU_QRT"].fillna(0)
            + df["AMT_REQ_CREDIT_BUREAU_YEAR"].fillna(0)
        )
        new_cols["BUREAU_ENQ_SHORT_TERM"] = short_term_enq.astype(np.float32)
        new_cols["BUREAU_ENQ_LONG_TERM"] = long_term_enq.astype(np.float32)
        new_cols["BUREAU_ENQ_VELOCITY_RATIO"] = (
            short_term_enq / (long_term_enq + 1e-6)
        ).astype(np.float32)
        new_cols["BUREAU_ENQ_MON_TO_YEAR"] = (
            df["AMT_REQ_CREDIT_BUREAU_MON"] / (df["AMT_REQ_CREDIT_BUREAU_YEAR"] + 1e-6)
        ).astype(np.float32)
        new_cols["BUREAU_ENQ_QRT_TO_YEAR"] = (
            df["AMT_REQ_CREDIT_BUREAU_QRT"] / (df["AMT_REQ_CREDIT_BUREAU_YEAR"] + 1e-6)
        ).astype(np.float32)

    # 6. External Sources Interactions and Non-linear Aggregations
    ext_cols = [
        col
        for col in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
        if col in df.columns
    ]
    if len(ext_cols) >= 2:
        new_cols["EXT_SOURCES_MEAN"] = df[ext_cols].mean(axis=1).astype(np.float32)
        new_cols["EXT_SOURCES_STD"] = df[ext_cols].std(axis=1).astype(np.float32)
        new_cols["EXT_SOURCES_MIN"] = df[ext_cols].min(axis=1).astype(np.float32)
        new_cols["EXT_SOURCES_MAX"] = df[ext_cols].max(axis=1).astype(np.float32)
        new_cols["EXT_SOURCES_SUM"] = df[ext_cols].sum(axis=1).astype(np.float32)
        new_cols["EXT_SOURCES_PROD"] = df[ext_cols].prod(axis=1).astype(np.float32)

        # Harmonic and geometric external score aggregates
        if len(ext_cols) == 3:
            s1 = df["EXT_SOURCE_1"]
            s2 = df["EXT_SOURCE_2"]
            s3 = df["EXT_SOURCE_3"]
            new_cols["EXT_SOURCES_HARMONIC"] = (
                3.0 / (1.0 / (s1.fillna(0.5) + 1e-6) + 1.0 / (s2.fillna(0.5) + 1e-6) + 1.0 / (s3.fillna(0.5) + 1e-6))
            ).astype(np.float32)
            new_cols["EXT_SOURCES_GEOMETRIC"] = (
                (s1.fillna(0.5).clip(lower=1e-5) * s2.fillna(0.5).clip(lower=1e-5) * s3.fillna(0.5).clip(lower=1e-5))
                ** (1.0 / 3.0)
            ).astype(np.float32)

        if "EXT_SOURCE_1" in df.columns and "EXT_SOURCE_2" in df.columns:
            new_cols["EXT_1_TIMES_2"] = (df["EXT_SOURCE_1"] * df["EXT_SOURCE_2"]).astype(np.float32)
            new_cols["EXT_1_DIV_2"] = (df["EXT_SOURCE_1"] / (df["EXT_SOURCE_2"] + 1e-6)).astype(np.float32)
        if "EXT_SOURCE_2" in df.columns and "EXT_SOURCE_3" in df.columns:
            new_cols["EXT_2_TIMES_3"] = (df["EXT_SOURCE_2"] * df["EXT_SOURCE_3"]).astype(np.float32)
            new_cols["EXT_2_DIV_3"] = (df["EXT_SOURCE_2"] / (df["EXT_SOURCE_3"] + 1e-6)).astype(np.float32)
        if "EXT_SOURCE_1" in df.columns and "EXT_SOURCE_3" in df.columns:
            new_cols["EXT_1_TIMES_3"] = (df["EXT_SOURCE_1"] * df["EXT_SOURCE_3"]).astype(np.float32)
            new_cols["EXT_1_DIV_3"] = (df["EXT_SOURCE_1"] / (df["EXT_SOURCE_3"] + 1e-6)).astype(np.float32)

        if "PAYMENT_RATE" in new_cols:
            new_cols["EXT_MEAN_DIV_PAYMENT_RATE"] = (
                new_cols["EXT_SOURCES_MEAN"] / (new_cols["PAYMENT_RATE"] + 1e-6)
            ).astype(np.float32)
            new_cols["EXT_MEAN_TIMES_PAYMENT_RATE"] = (
                new_cols["EXT_SOURCES_MEAN"] * new_cols["PAYMENT_RATE"]
            ).astype(np.float32)

        # Weighted & polynomial external source features
        for ext in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]:
            if ext in df.columns:
                new_cols[f"{ext}_SQ"] = (df[ext] ** 2).astype(np.float32)

        s1_val = df["EXT_SOURCE_1"].fillna(0.5) if "EXT_SOURCE_1" in df.columns else 0.5
        s2_val = df["EXT_SOURCE_2"].fillna(0.5) if "EXT_SOURCE_2" in df.columns else 0.5
        s3_val = df["EXT_SOURCE_3"].fillna(0.5) if "EXT_SOURCE_3" in df.columns else 0.5
        new_cols["EXT_SOURCES_WEIGHTED"] = (s1_val * 0.2 + s2_val * 0.4 + s3_val * 0.4).astype(np.float32)

    # 7. Document & Contact Aggregations
    doc_cols = [c for c in df.columns if c.startswith("FLAG_DOCUMENT_")]
    if doc_cols:
        new_cols["TOTAL_DOCUMENTS_PRESENT"] = df[doc_cols].sum(axis=1).astype(np.int16)

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
        new_cols["TOTAL_CONTACTS_PROVIDED"] = df[contact_cols].sum(axis=1).astype(np.int16)

    # Unfragmented single-pass concatenation
    new_features_df = pd.DataFrame(new_cols, index=df.index)
    return pd.concat([df, new_features_df], axis=1)


def add_cohort_and_frequency_features(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute group-level cohort benchmarks and categorical frequency encodings in a single batch."""
    train_new_cols: Dict[str, Any] = {}
    test_new_cols: Dict[str, Any] = {}

    group_cols = [
        "OCCUPATION_TYPE",
        "ORGANIZATION_TYPE",
        "NAME_EDUCATION_TYPE",
        "NAME_INCOME_TYPE",
    ]
    target_num_cols = [
        "AMT_CREDIT",
        "AMT_ANNUITY",
        "AMT_INCOME_TOTAL",
        "EXT_SOURCES_MEAN",
    ]

    # Frequency encoding for categorical columns
    cat_cols_for_freq = [
        col
        for col in [
            "OCCUPATION_TYPE",
            "ORGANIZATION_TYPE",
            "NAME_EDUCATION_TYPE",
            "NAME_INCOME_TYPE",
            "NAME_FAMILY_STATUS",
            "NAME_HOUSING_TYPE",
        ]
        if col in train_df.columns
    ]
    for col in cat_cols_for_freq:
        freq = train_df[col].value_counts(normalize=True)
        train_new_cols[f"{col}_FREQ"] = train_df[col].map(freq).fillna(0).astype(np.float32)
        test_new_cols[f"{col}_FREQ"] = test_df[col].map(freq).fillna(0).astype(np.float32)

    # Group cohort aggregations
    for g_col in group_cols:
        if g_col not in train_df.columns:
            continue
        for num_col in target_num_cols:
            if num_col not in train_df.columns:
                continue
            group_mean = train_df.groupby(g_col)[num_col].mean()
            group_std = train_df.groupby(g_col)[num_col].std()
            group_min = train_df.groupby(g_col)[num_col].min()
            group_max = train_df.groupby(g_col)[num_col].max()

            mean_col = f"GRP_{g_col}_{num_col}_MEAN"
            std_col = f"GRP_{g_col}_{num_col}_STD"
            min_col = f"GRP_{g_col}_{num_col}_MIN"
            max_col = f"GRP_{g_col}_{num_col}_MAX"

            train_mean = train_df[g_col].map(group_mean).astype(np.float32)
            test_mean = test_df[g_col].map(group_mean).astype(np.float32)

            train_new_cols[mean_col] = train_mean
            test_new_cols[mean_col] = test_mean

            train_new_cols[std_col] = train_df[g_col].map(group_std).astype(np.float32)
            test_new_cols[std_col] = test_df[g_col].map(group_std).astype(np.float32)

            train_new_cols[min_col] = train_df[g_col].map(group_min).astype(np.float32)
            test_new_cols[min_col] = test_df[g_col].map(group_min).astype(np.float32)

            train_new_cols[max_col] = train_df[g_col].map(group_max).astype(np.float32)
            test_new_cols[max_col] = test_df[g_col].map(group_max).astype(np.float32)

            train_new_cols[f"{num_col}_TO_{g_col}_MEAN_RATIO"] = (
                train_df[num_col] / (train_mean + 1e-6)
            ).astype(np.float32)
            test_new_cols[f"{num_col}_TO_{g_col}_MEAN_RATIO"] = (
                test_df[num_col] / (test_mean + 1e-6)
            ).astype(np.float32)

            train_new_cols[f"{num_col}_DIFF_{g_col}_MEAN"] = (
                train_df[num_col] - train_mean
            ).astype(np.float32)
            test_new_cols[f"{num_col}_DIFF_{g_col}_MEAN"] = (
                test_df[num_col] - test_mean
            ).astype(np.float32)

    train_df = pd.concat([train_df, pd.DataFrame(train_new_cols, index=train_df.index)], axis=1)
    test_df = pd.concat([test_df, pd.DataFrame(test_new_cols, index=test_df.index)], axis=1)
    return train_df, test_df


class CreditDefaultModelFactory:
    """Model architecture factory for Home Credit Default Risk prediction."""

    @staticmethod
    def get_lightgbm_params(
        custom_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        base_params = {
            "objective": "binary",
            "metric": "auc",
            "boosting_type": "gbdt",
            "learning_rate": 0.02,
            "n_estimators": 3000,
            "num_leaves": 31,
            "max_depth": 7,
            "min_child_samples": 50,
            "subsample": 0.8,
            "subsample_freq": 1,
            "colsample_bytree": 0.6,
            "reg_alpha": 2.0,
            "reg_lambda": 8.0,
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
        }
        if custom_params:
            base_params.update(custom_params)
        return base_params

    @staticmethod
    def create_lgbm_model(
        custom_params: Optional[Dict[str, Any]] = None
    ) -> lgb.LGBMClassifier:
        params = CreditDefaultModelFactory.get_lightgbm_params(custom_params)
        return lgb.LGBMClassifier(**params)

    @staticmethod
    def get_lgbm_callbacks(early_stopping_rounds: int = 100) -> List[Any]:
        return [
            lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=0),
        ]

    @staticmethod
    def get_xgboost_params(
        custom_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        base_params = {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "learning_rate": 0.02,
            "n_estimators": 3000,
            "max_depth": 6,
            "min_child_weight": 30,
            "subsample": 0.8,
            "colsample_bytree": 0.6,
            "reg_alpha": 2.0,
            "reg_lambda": 8.0,
            "tree_method": "hist",
            "random_state": 42,
            "n_jobs": -1,
            "early_stopping_rounds": 100,
        }
        if custom_params:
            base_params.update(custom_params)
        return base_params

    @staticmethod
    def create_xgb_model(
        custom_params: Optional[Dict[str, Any]] = None
    ) -> xgb.XGBClassifier:
        params = CreditDefaultModelFactory.get_xgboost_params(custom_params)
        return xgb.XGBClassifier(**params)

    @staticmethod
    def get_catboost_params(
        custom_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        base_params = {
            "loss_function": "Logloss",
            "eval_metric": "AUC",
            "iterations": 2500,
            "learning_rate": 0.03,
            "depth": 6,
            "l2_leaf_reg": 5.0,
            "random_seed": 42,
            "thread_count": -1,
            "verbose": False,
            "early_stopping_rounds": 100,
        }
        if custom_params:
            base_params.update(custom_params)
        return base_params

    @staticmethod
    def create_catboost_model(
        custom_params: Optional[Dict[str, Any]] = None
    ) -> CatBoostClassifier:
        params = CreditDefaultModelFactory.get_catboost_params(custom_params)
        return CatBoostClassifier(**params)


def optimize_blend_weights(oof_matrix: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """Optimize ensemble blending weights on out-of-fold predictions to directly maximize ROC-AUC."""
    n_models = oof_matrix.shape[1]

    def objective(weights: np.ndarray) -> float:
        weights = np.array(weights)
        weights = weights / (np.sum(weights) + 1e-12)
        blend = np.dot(oof_matrix, weights)
        return -roc_auc_score(y_true, blend)

    init_weights = np.ones(n_models) / n_models
    bounds = [(0.0, 1.0) for _ in range(n_models)]
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

    res = minimize(
        objective,
        init_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 200, "ftol": 1e-7},
    )
    if res.success:
        opt_weights = res.x / np.sum(res.x)
    else:
        opt_weights = init_weights
    return opt_weights


def main():
    train_path = "./input/train.parquet"
    test_path = "./input/test.parquet"

    print("Loading raw training and test data...")
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    # 1. Feature Engineering
    print("Engineering domain features...")
    train_df = engineer_features(train_df)
    test_df = engineer_features(test_df)

    # 2. Cohort Benchmarks & Frequency Encodings
    print("Computing cohort benchmarks and frequency encodings...")
    train_df, test_df = add_cohort_and_frequency_features(train_df, test_df)

    # 3. Categorical Encoding (Fit strictly on train)
    categorical_cols = train_df.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    for col in categorical_cols:
        train_df[col] = train_df[col].astype(str)
        test_df[col] = test_df[col].astype(str)

        le = LabelEncoder()
        train_df[col] = le.fit_transform(train_df[col])
        test_df[col] = test_df[col].map(
            lambda s: le.transform([s])[0] if s in le.classes_ else -1
        )

    # 3. Clean Column Names for GBDT compatibility
    train_df = clean_column_names(train_df)
    test_df = clean_column_names(test_df)

    # 4. Define Features and Target
    target_col = "TARGET"
    ignore_cols = {target_col, "SK_ID_CURR", "id", "fold"}
    feature_cols = [col for col in train_df.columns if col not in ignore_cols]
    y = train_df[target_col].values

    # 5. Stratified 5-Fold Cross-Validation Setup
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    lgb_oof = np.zeros(len(train_df), dtype=np.float64)
    xgb_oof = np.zeros(len(train_df), dtype=np.float64)
    cb_oof = np.zeros(len(train_df), dtype=np.float64)

    test_preds_lgb = np.zeros(len(test_df), dtype=np.float64)
    test_preds_xgb = np.zeros(len(test_df), dtype=np.float64)
    test_preds_cb = np.zeros(len(test_df), dtype=np.float64)

    print(
        f"Training LightGBM + XGBoost + CatBoost ensemble with {len(feature_cols)} features on {len(train_df)} samples..."
    )

    for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, y)):
        X_train, y_train = (
            train_df.iloc[train_idx][feature_cols],
            y[train_idx],
        )
        X_val, y_val = (
            train_df.iloc[val_idx][feature_cols],
            y[val_idx],
        )

        # 1. Train LightGBM model
        lgb_model = CreditDefaultModelFactory.create_lgbm_model(
            custom_params={"random_state": 42 + fold}
        )
        callbacks = CreditDefaultModelFactory.get_lgbm_callbacks(
            early_stopping_rounds=100
        )
        lgb_model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=callbacks,
        )

        # 2. Train XGBoost model
        xgb_model = CreditDefaultModelFactory.create_xgb_model(
            custom_params={"random_state": 42 + fold}
        )
        xgb_model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        # 3. Train CatBoost model
        cb_model = CreditDefaultModelFactory.create_catboost_model(
            custom_params={"random_seed": 42 + fold}
        )
        cb_model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        # Predict fold validation probabilities
        val_preds_lgb = lgb_model.predict_proba(X_val)[:, 1]
        val_preds_xgb = xgb_model.predict_proba(X_val)[:, 1]
        val_preds_cb = cb_model.predict_proba(X_val)[:, 1]

        lgb_oof[val_idx] = val_preds_lgb
        xgb_oof[val_idx] = val_preds_xgb
        cb_oof[val_idx] = val_preds_cb

        # Accumulate test predictions per model
        test_preds_lgb += lgb_model.predict_proba(test_df[feature_cols])[:, 1] / n_splits
        test_preds_xgb += xgb_model.predict_proba(test_df[feature_cols])[:, 1] / n_splits
        test_preds_cb += cb_model.predict_proba(test_df[feature_cols])[:, 1] / n_splits

        fold_auc_lgb = roc_auc_score(y_val, val_preds_lgb)
        fold_auc_xgb = roc_auc_score(y_val, val_preds_xgb)
        fold_auc_cb = roc_auc_score(y_val, val_preds_cb)
        fold_auc_blend = roc_auc_score(
            y_val, (val_preds_lgb + val_preds_xgb + val_preds_cb) / 3.0
        )
        print(
            f"Fold {fold + 1}/{n_splits} - LightGBM AUC: {fold_auc_lgb:.5f}, "
            f"XGBoost AUC: {fold_auc_xgb:.5f}, CatBoost AUC: {fold_auc_cb:.5f}, Equal Blend AUC: {fold_auc_blend:.5f}"
        )

    # 6. Overall OOF Evaluation & Optimization
    lgb_oof_auc = float(roc_auc_score(y, lgb_oof))
    xgb_oof_auc = float(roc_auc_score(y, xgb_oof))
    cb_oof_auc = float(roc_auc_score(y, cb_oof))
    print(f"Overall LightGBM OOF ROC-AUC: {lgb_oof_auc:.5f}")
    print(f"Overall XGBoost OOF ROC-AUC: {xgb_oof_auc:.5f}")
    print(f"Overall CatBoost OOF ROC-AUC: {cb_oof_auc:.5f}")

    oof_matrix = np.column_stack([lgb_oof, xgb_oof, cb_oof])
    optimal_weights = optimize_blend_weights(oof_matrix, y)
    print(
        f"Optimal Blend Weights: LGB={optimal_weights[0]:.4f}, "
        f"XGB={optimal_weights[1]:.4f}, CB={optimal_weights[2]:.4f}"
    )

    oof_predictions = np.dot(oof_matrix, optimal_weights)
    overall_oof_auc = float(roc_auc_score(y, oof_predictions))
    print(f"Overall Optimized Ensemble OOF ROC-AUC: {overall_oof_auc:.5f}")

    # 7. Generate Submission File
    test_matrix = np.column_stack([test_preds_lgb, test_preds_xgb, test_preds_cb])
    test_predictions = np.dot(test_matrix, optimal_weights)

    os.makedirs("./submission", exist_ok=True)
    submission_path = "./submission/submission.csv"

    submission_df = pd.DataFrame(
        {
            "id": np.arange(len(test_df), dtype=int),
            "TARGET": test_predictions,
        }
    )
    submission_df.to_csv(submission_path, index=False)
    print(f"Saved submission to {submission_path} with shape {submission_df.shape}")

    # 8. Output exact final metric
    print(f"Final Validation Score: {overall_oof_auc}")


if __name__ == "__main__":
    main()
