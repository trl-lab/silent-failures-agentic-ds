import json
import os
from catboost import CatBoostClassifier
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import rankdata
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

# Set random seeds for deterministic execution
np.random.seed(42)


# ==============================================================================
# 1. DATA PROCESSING & FEATURE ENGINEERING
# ==============================================================================
def process_and_engineer_features():
    input_dir = "./input"
    working_dir = "./working"
    os.makedirs(working_dir, exist_ok=True)

    train_df = pd.read_parquet(os.path.join(input_dir, "train.parquet"))
    test_df = pd.read_parquet(os.path.join(input_dir, "test.parquet"))

    # Extract target column
    target_col = "class"
    if target_col not in train_df.columns:
        possible_targets = [
            c
            for c in train_df.columns
            if c.lower() in ["class", "target", "risk", "default"]
        ]
        target_col = possible_targets[0] if possible_targets else train_df.columns[-1]

    y_raw = train_df[target_col].copy()

    # Standardize binary target: 1 for positive class matching sklearn LabelBinarizer/roc_auc ordering
    le = LabelEncoder()
    y = le.fit_transform(y_raw.astype(str))

    train_features = train_df.drop(columns=[target_col]).copy()
    test_features = test_df.copy()
    if target_col in test_features.columns:
        test_features = test_features.drop(columns=[target_col])

    n_train = len(train_features)
    combined = pd.concat([train_features, test_features], axis=0).reset_index(drop=True)

    # Standardize column naming
    col_mapping = {
        col: str(col).lower().replace(" ", "_").replace("-", "_")
        for col in combined.columns
    }
    combined = combined.rename(columns=col_mapping)

    def find_col(candidates):
        for c in candidates:
            for col in combined.columns:
                if c in col:
                    return col
        return None

    duration_col = find_col(["duration", "month", "credit_duration"])
    amount_col = find_col(["amount", "credit_amount", "sum"])
    age_col = find_col(["age", "years"])
    installment_col = find_col(
        ["installment_rate", "installment_commitment", "installment"]
    )
    residence_col = find_col(["residence", "residence_since", "present_residence"])
    credits_col = find_col(
        ["existing_credits", "credits_at_this_bank", "number_of_existing_credits"]
    )
    dependents_col = find_col(["dependents", "liable", "maintenance", "people_liable"])
    checking_col = find_col(
        ["checking", "status_of_existing_checking", "checking_status"]
    )
    savings_col = find_col(["savings", "status_of_savings", "savings_status"])
    employment_col = find_col(["employment", "present_employment"])
    purpose_col = find_col(["purpose", "credit_purpose"])
    housing_col = find_col(["housing", "home_ownership"])
    history_col = find_col(["credit_history", "history"])
    property_col = find_col(["property", "property_magnitude"])
    job_col = find_col(["job", "occupation"])
    debtors_col = find_col(["debtor", "guarantor", "other_parties", "other_debtor"])
    personal_col = find_col(["personal", "sex", "status_sex", "marital"])
    foreign_col = find_col(["foreign", "worker"])

    numeric_candidates = [
        duration_col,
        amount_col,
        age_col,
        installment_col,
        residence_col,
        credits_col,
        dependents_col,
    ]
    numeric_cols = [c for c in numeric_candidates if c is not None]
    for c in numeric_cols:
        combined[c] = pd.to_numeric(combined[c], errors="coerce").fillna(
            combined[c].median() if not pd.isna(combined[c].median()) else 0
        )

    # Domain-aware ordinal risk mapping functions
    def map_checking_risk(val):
        s = str(val).lower()
        if "<0" in s or "< 0" in s:
            return 3.0
        elif "0<=" in s or "0 <=" in s or "<200" in s or "< 200" in s:
            return 2.0
        elif ">=200" in s or ">= 200" in s:
            return 1.0
        elif "no" in s or "none" in s:
            return 0.0
        return 1.5

    def map_savings_risk(val):
        s = str(val).lower()
        if "<100" in s or "< 100" in s:
            return 4.0
        elif "100<=" in s or "100 <=" in s or "<500" in s or "< 500" in s:
            return 3.0
        elif "500<=" in s or "500 <=" in s or "<1000" in s or "< 1000" in s:
            return 2.0
        elif ">=1000" in s or ">= 1000" in s:
            return 1.0
        elif "no" in s or "unknown" in s:
            return 2.5
        return 2.5

    def map_employment_risk(val):
        s = str(val).lower()
        if "unemployed" in s:
            return 4.0
        elif "<1" in s or "< 1" in s:
            return 3.0
        elif "1<=" in s or "1 <=" in s or "<4" in s or "< 4" in s:
            return 2.0
        elif "4<=" in s or "4 <=" in s or "<7" in s or "< 7" in s:
            return 1.0
        elif ">=7" in s or ">= 7" in s:
            return 0.0
        return 2.0

    def map_property_risk(val):
        s = str(val).lower()
        if "real estate" in s:
            return 0.0
        elif "building" in s or "society" in s or "insurance" in s:
            return 1.0
        elif "car" in s or "other" in s:
            return 2.0
        elif "no" in s or "unknown" in s:
            return 3.0
        return 1.5

    def map_history_risk(val):
        s = str(val).lower()
        if "critical" in s:
            return 4.0
        elif "delay" in s:
            return 3.0
        elif "existing" in s:
            return 2.0
        elif "all paid" in s:
            return 1.0
        elif "no credit" in s:
            return 0.0
        return 2.0

    # Domain Feature Engineering
    engineered = pd.DataFrame(index=combined.index)

    # Ordinal risk scores and composite index
    risk_sum = np.zeros(len(combined), dtype=np.float32)
    risk_count = 0
    if checking_col:
        engineered["fe_ord_checking"] = np.array(
            [map_checking_risk(x) for x in combined[checking_col]], dtype=np.float32
        )
        risk_sum += engineered["fe_ord_checking"].to_numpy(dtype=np.float32)
        risk_count += 1
    if savings_col:
        engineered["fe_ord_savings"] = np.array(
            [map_savings_risk(x) for x in combined[savings_col]], dtype=np.float32
        )
        risk_sum += engineered["fe_ord_savings"].to_numpy(dtype=np.float32)
        risk_count += 1
    if employment_col:
        engineered["fe_ord_employment"] = np.array(
            [map_employment_risk(x) for x in combined[employment_col]], dtype=np.float32
        )
        risk_sum += engineered["fe_ord_employment"].to_numpy(dtype=np.float32)
        risk_count += 1
    if property_col:
        engineered["fe_ord_property"] = np.array(
            [map_property_risk(x) for x in combined[property_col]], dtype=np.float32
        )
        risk_sum += engineered["fe_ord_property"].to_numpy(dtype=np.float32)
        risk_count += 1
    if history_col:
        engineered["fe_ord_history"] = np.array(
            [map_history_risk(x) for x in combined[history_col]], dtype=np.float32
        )
        risk_sum += engineered["fe_ord_history"].to_numpy(dtype=np.float32)
        risk_count += 1
    if risk_count > 0:
        engineered["fe_risk_composite"] = (risk_sum / risk_count).astype(np.float32)

    if duration_col and amount_col:
        engineered["fe_monthly_payment"] = combined[amount_col] / (
            combined[duration_col] + 1.0
        )
        engineered["fe_log_amount"] = np.log1p(np.maximum(combined[amount_col], 0))
        engineered["fe_log_duration"] = np.log1p(np.maximum(combined[duration_col], 0))
        engineered["fe_amount_duration_ratio"] = combined[amount_col] / (
            combined[duration_col] ** 0.5 + 1.0
        )

    if age_col and duration_col:
        engineered["fe_duration_to_age_ratio"] = combined[duration_col] / (
            combined[age_col] + 1e-5
        )
        engineered["fe_age_times_duration"] = combined[age_col] * combined[duration_col]

    if age_col and amount_col:
        engineered["fe_amount_per_year_age"] = combined[amount_col] / (
            combined[age_col] + 1e-5
        )

    if installment_col and amount_col:
        if "fe_monthly_payment" in engineered.columns:
            income_proxy = engineered["fe_monthly_payment"] / (
                combined[installment_col] / 100.0 + 1e-4
            )
            engineered["fe_estimated_income_proxy"] = income_proxy
            engineered["fe_debt_to_income_proxy"] = combined[amount_col] / (
                income_proxy + 1.0
            )
        engineered["fe_installment_burden"] = combined[amount_col] * (
            combined[installment_col] / 100.0
        )

    if installment_col and age_col:
        engineered["fe_installment_to_age"] = combined[installment_col] / (
            combined[age_col] + 1e-5
        )

    if duration_col and installment_col:
        engineered["fe_total_rate_commitment"] = (
            combined[duration_col] * combined[installment_col]
        )

    if credits_col and amount_col:
        engineered["fe_amount_per_credit"] = combined[amount_col] / (
            combined[credits_col] + 1.0
        )

    if credits_col and age_col:
        engineered["fe_credits_per_age"] = combined[credits_col] / (
            combined[age_col] + 1e-5
        )

    if dependents_col and amount_col:
        engineered["fe_amount_per_dependent"] = combined[amount_col] / (
            combined[dependents_col] + 1.0
        )

    if "fe_ord_checking" in engineered.columns and "fe_ord_savings" in engineered.columns:
        engineered["fe_liquidity_score"] = (
            (4.0 - engineered["fe_ord_checking"]) + (4.0 - engineered["fe_ord_savings"])
        ).astype(np.float32)

    if (
        "fe_risk_composite" in engineered.columns
        and "fe_installment_burden" in engineered.columns
        and "fe_log_amount" in engineered.columns
    ):
        engineered["fe_compound_solvency_risk"] = (
            engineered["fe_risk_composite"]
            * engineered["fe_installment_burden"]
            / (engineered["fe_log_amount"] + 1.0)
        ).astype(np.float32)

    # Domain binary indicators
    if personal_col:
        engineered["fe_is_single_male"] = combined[personal_col].apply(
            lambda x: 1.0 if ("male" in str(x).lower() and "single" in str(x).lower()) else 0.0
        ).astype(np.float32)
    if debtors_col:
        engineered["fe_has_guarantor"] = combined[debtors_col].apply(
            lambda x: 1.0 if ("guarantor" in str(x).lower() or "co-applicant" in str(x).lower() or "coapplicant" in str(x).lower()) else 0.0
        ).astype(np.float32)
    if housing_col:
        engineered["fe_owns_home"] = combined[housing_col].apply(
            lambda x: 1.0 if "own" in str(x).lower() else 0.0
        ).astype(np.float32)
    if history_col:
        engineered["fe_is_critical_history"] = combined[history_col].apply(
            lambda x: 1.0 if ("critical" in str(x).lower() or "delay" in str(x).lower()) else 0.0
        ).astype(np.float32)
    if foreign_col:
        engineered["fe_is_foreign_worker"] = combined[foreign_col].apply(
            lambda x: 1.0 if "yes" in str(x).lower() else 0.0
        ).astype(np.float32)

    full_df = pd.concat([combined, engineered], axis=1)

    # Group aggregations computed leak-free using training split only
    train_part = full_df.iloc[:n_train].copy()
    test_part = full_df.iloc[n_train:].copy()

    agg_group_cols = [
        c for c in [purpose_col, employment_col, job_col, housing_col] if c is not None
    ]
    agg_target_cols = [c for c in [amount_col, duration_col] if c is not None]

    for g_col in agg_group_cols:
        for t_col in agg_target_cols:
            g_train = train_part[g_col].astype(str)
            g_test = test_part[g_col].astype(str)
            t_train = train_part[t_col].astype(float)
            t_test = test_part[t_col].astype(float)

            grp_mean = t_train.groupby(g_train).mean()
            grp_std = t_train.groupby(g_train).std().fillna(0)
            global_mean = float(t_train.mean())

            mean_col = f"fe_grp_{g_col}_{t_col}_mean"
            std_col = f"fe_grp_{g_col}_{t_col}_std"
            diff_col = f"fe_diff_{g_col}_{t_col}"

            train_part[mean_col] = g_train.map(grp_mean).fillna(global_mean).astype(np.float32)
            train_part[std_col] = g_train.map(grp_std).fillna(0.0).astype(np.float32)
            train_part[diff_col] = (t_train - train_part[mean_col]).astype(np.float32)

            test_part[mean_col] = g_test.map(grp_mean).fillna(global_mean).astype(np.float32)
            test_part[std_col] = g_test.map(grp_std).fillna(0.0).astype(np.float32)
            test_part[diff_col] = (t_test - test_part[mean_col]).astype(np.float32)

    # Categorical encoding (Frequency, Target & Label Encoding)
    cat_cols = [
        c
        for c in train_part.columns
        if train_part[c].dtype == object
        or isinstance(train_part[c].dtype, pd.CategoricalDtype)
    ]

    # Generate 10-Fold Stratified Split for leakage-free Target Encoding & Cross-Validation
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    folds = np.zeros(n_train, dtype=int)
    for fold, (_, val_idx) in enumerate(skf.split(train_part, y)):
        folds[val_idx] = fold

    # Out-of-fold smoothed Target Encoding strictly fit on training folds
    smooth_m = 10.0
    prior_mean = float(np.mean(y))

    for c in cat_cols:
        c_train_str = train_part[c].astype(str).fillna("missing")
        c_test_str = test_part[c].astype(str).fillna("missing")
        train_part[f"fe_te_{c}"] = np.nan

        for fold in range(10):
            fold_val_mask = folds == fold
            fold_tr_mask = folds != fold

            tr_c = c_train_str.iloc[fold_tr_mask]
            tr_y = y[fold_tr_mask]

            counts = tr_c.value_counts()
            means = pd.Series(tr_y).groupby(tr_c.values).mean()

            smoothed = (counts * means + smooth_m * prior_mean) / (counts + smooth_m)
            te_dict = smoothed.to_dict()

            train_part.loc[fold_val_mask, f"fe_te_{c}"] = (
                c_train_str.iloc[fold_val_mask].map(te_dict).fillna(prior_mean).values
            )

        full_counts = c_train_str.value_counts()
        full_means = pd.Series(y).groupby(c_train_str.values).mean()
        full_smoothed = (full_counts * full_means + smooth_m * prior_mean) / (
            full_counts + smooth_m
        )
        test_part[f"fe_te_{c}"] = (
            c_test_str.map(full_smoothed.to_dict()).fillna(prior_mean).values
        )

        freq_map = c_train_str.value_counts(normalize=True).to_dict()
        train_part[f"fe_freq_{c}"] = c_train_str.map(freq_map).fillna(0).astype(float)
        test_part[f"fe_freq_{c}"] = c_test_str.map(freq_map).fillna(0).astype(float)

        le = LabelEncoder()
        train_part[c] = le.fit_transform(c_train_str)
        known_classes = set(le.classes_)
        test_part[c] = c_test_str.map(
            lambda s: le.transform([s])[0] if s in known_classes else -1
        )

    train_clean = train_part.fillna(0).astype(np.float32)
    test_clean = test_part.fillna(0).astype(np.float32)

    train_clean["fold"] = folds
    train_clean["target"] = y

    train_clean.to_parquet(
        os.path.join(working_dir, "train_processed.parquet"), index=False
    )
    test_clean.to_parquet(
        os.path.join(working_dir, "test_processed.parquet"), index=False
    )

    return train_clean, test_clean


# ==============================================================================
# 2. MODEL ARCHITECTURES & BUILDERS
# ==============================================================================
def get_catboost_model(
    random_state: int = 42,
    iterations: int = 400,
    learning_rate: float = 0.03,
    depth: int = 4,
    l2_leaf_reg: float = 5.0,
) -> CatBoostClassifier:
    return CatBoostClassifier(
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        l2_leaf_reg=l2_leaf_reg,
        random_seed=random_state,
        eval_metric="AUC",
        verbose=False,
    )


def get_extratrees_model(
    random_state: int = 42,
    n_estimators: int = 300,
    max_depth: int = 8,
    min_samples_split: int = 6,
    min_samples_leaf: int = 3,
) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features="sqrt",
        random_state=random_state,
        n_jobs=-1,
    )


def get_rf_model(
    random_state: int = 42,
    n_estimators: int = 300,
    max_depth: int = 8,
    min_samples_split: int = 6,
    min_samples_leaf: int = 3,
) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features="sqrt",
        random_state=random_state,
        n_jobs=-1,
    )


def get_hgb_model(
    random_state: int = 42,
    max_iter: int = 250,
    learning_rate: float = 0.03,
    max_depth: int = 4,
    min_samples_leaf: int = 15,
    l2_regularization: float = 2.0,
) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=max_iter,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        l2_regularization=l2_regularization,
        random_state=random_state,
    )


def get_lgbm_model(
    random_state: int = 42,
    n_estimators: int = 350,
    learning_rate: float = 0.03,
    num_leaves: int = 15,
    max_depth: int = 4,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    reg_alpha: float = 0.2,
    reg_lambda: float = 2.0,
    min_child_samples: int = 15,
) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        max_depth=max_depth,
        subsample=subsample,
        subsample_freq=1,
        colsample_bytree=colsample_bytree,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        min_child_samples=min_child_samples,
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
        importance_type="gain",
    )


def get_xgb_model(
    random_state: int = 42,
    n_estimators: int = 350,
    learning_rate: float = 0.03,
    max_depth: int = 4,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    reg_alpha: float = 0.2,
    reg_lambda: float = 2.0,
    min_child_weight: int = 3,
) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        min_child_weight=min_child_weight,
        random_state=random_state,
        eval_metric="auc",
        early_stopping_rounds=30,
        n_jobs=-1,
        tree_method="hist",
    )


def get_linear_model(
    C: float = 0.1, l1_ratio: float = 0.3, random_state: int = 42
) -> LogisticRegression:
    return LogisticRegression(
        penalty="elasticnet",
        solver="saga",
        C=C,
        l1_ratio=l1_ratio,
        max_iter=1000,
        random_state=random_state,
    )


# ==============================================================================
# 3. CROSS-VALIDATED TRAINING & EVALUATION
# ==============================================================================
def main():
    train_df, test_df = process_and_engineer_features()

    feature_cols = [c for c in train_df.columns if c not in ["fold", "target"]]
    n_train = len(train_df)
    n_test = len(test_df)
    y_true = train_df["target"].values.astype(int)

    seeds = [42, 100, 2024]
    n_seeds = len(seeds)
    n_folds = 10
    model_names = ["lgbm", "xgb", "cb", "et", "rf", "hgb", "lr"]
    n_models = len(model_names)

    # Accumulators for multi-seed OOF and test predictions
    oof_accum = {m: np.zeros(n_train, dtype=np.float64) for m in model_names}
    test_accum = {m: np.zeros(n_test, dtype=np.float64) for m in model_names}

    for seed in seeds:
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

        for fold, (tr_idx, val_idx) in enumerate(skf.split(train_df, y_true)):
            X_tr = train_df.iloc[tr_idx][feature_cols].values
            y_tr = y_true[tr_idx]
            X_val = train_df.iloc[val_idx][feature_cols].values
            y_val = y_true[val_idx]
            X_test_raw = test_df[feature_cols].values

            # Standardize features strictly on the training split of the current fold
            scaler = StandardScaler()
            X_tr_scaled = scaler.fit_transform(X_tr)
            X_val_scaled = scaler.transform(X_val)
            X_test_scaled = scaler.transform(X_test_raw)

            # 1. LightGBM
            lgb_clf = get_lgbm_model(random_state=seed * 10 + fold)
            lgb_callbacks = [
                lgb.early_stopping(stopping_rounds=30, verbose=False),
                lgb.log_evaluation(period=0),
            ]
            lgb_clf.fit(
                X_tr,
                y_tr,
                eval_set=[(X_val, y_val)],
                eval_metric="auc",
                callbacks=lgb_callbacks,
            )
            oof_accum["lgbm"][val_idx] += lgb_clf.predict_proba(X_val)[:, 1] / n_seeds
            test_accum["lgbm"] += lgb_clf.predict_proba(X_test_raw)[:, 1] / (n_seeds * n_folds)

            # 2. XGBoost
            xgb_clf = get_xgb_model(random_state=seed * 10 + fold)
            xgb_clf.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
            oof_accum["xgb"][val_idx] += xgb_clf.predict_proba(X_val)[:, 1] / n_seeds
            test_accum["xgb"] += xgb_clf.predict_proba(X_test_raw)[:, 1] / (n_seeds * n_folds)

            # 3. CatBoost
            cb_clf = get_catboost_model(random_state=seed * 10 + fold)
            cb_clf.fit(
                X_tr,
                y_tr,
                eval_set=(X_val, y_val),
                early_stopping_rounds=30,
                verbose=False,
            )
            oof_accum["cb"][val_idx] += cb_clf.predict_proba(X_val)[:, 1] / n_seeds
            test_accum["cb"] += cb_clf.predict_proba(X_test_raw)[:, 1] / (n_seeds * n_folds)

            # 4. ExtraTrees
            et_clf = get_extratrees_model(random_state=seed * 10 + fold)
            et_clf.fit(X_tr, y_tr)
            oof_accum["et"][val_idx] += et_clf.predict_proba(X_val)[:, 1] / n_seeds
            test_accum["et"] += et_clf.predict_proba(X_test_raw)[:, 1] / (n_seeds * n_folds)

            # 5. RandomForest
            rf_clf = get_rf_model(random_state=seed * 10 + fold)
            rf_clf.fit(X_tr, y_tr)
            oof_accum["rf"][val_idx] += rf_clf.predict_proba(X_val)[:, 1] / n_seeds
            test_accum["rf"] += rf_clf.predict_proba(X_test_raw)[:, 1] / (n_seeds * n_folds)

            # 6. HistGradientBoosting
            hgb_clf = get_hgb_model(random_state=seed * 10 + fold)
            hgb_clf.fit(X_tr, y_tr)
            oof_accum["hgb"][val_idx] += hgb_clf.predict_proba(X_val)[:, 1] / n_seeds
            test_accum["hgb"] += hgb_clf.predict_proba(X_test_raw)[:, 1] / (n_seeds * n_folds)

            # 7. ElasticNet Logistic Regression
            lr_clf = get_linear_model(C=0.1, l1_ratio=0.3, random_state=seed * 10 + fold)
            lr_clf.fit(X_tr_scaled, y_tr)
            oof_accum["lr"][val_idx] += lr_clf.predict_proba(X_val_scaled)[:, 1] / n_seeds
            test_accum["lr"] += lr_clf.predict_proba(X_test_scaled)[:, 1] / (n_seeds * n_folds)

    # Apply rank-normalization to OOF and Test distributions for calibrated AUC blending
    oof_ranks = np.zeros((n_train, n_models), dtype=np.float64)
    test_ranks = np.zeros((n_test, n_models), dtype=np.float64)

    for i, m in enumerate(model_names):
        oof_ranks[:, i] = (rankdata(oof_accum[m]) - 0.5) / n_train
        test_ranks[:, i] = (rankdata(test_accum[m]) - 0.5) / n_test

    # Optimize rank blending weights directly for ROC AUC using SLSQP
    def objective(weights):
        w = np.maximum(weights, 0)
        w_sum = np.sum(w)
        if w_sum == 0:
            return 0.0
        w = w / w_sum
        blend = oof_ranks @ w
        return -roc_auc_score(y_true, blend)

    init_weights = [1.0 / n_models] * n_models
    bounds = [(0.0, 1.0) for _ in range(n_models)]
    res = minimize(
        objective,
        init_weights,
        method="SLSQP",
        bounds=bounds,
        constraints={"type": "eq", "fun": lambda w: 1.0 - sum(w)},
    )
    optimal_weights = np.maximum(res.x, 0)
    optimal_weights = optimal_weights / np.sum(optimal_weights)

    oof_ensemble = oof_ranks @ optimal_weights
    final_validation_score = roc_auc_score(y_true, oof_ensemble)

    final_test_predictions = test_ranks @ optimal_weights

    # Export Submission
    submission_dir = "./submission"
    os.makedirs(submission_dir, exist_ok=True)
    submission_df = pd.DataFrame(
        {
            "id": np.arange(len(final_test_predictions), dtype=int),
            "class": final_test_predictions,
        }
    )
    submission_df.to_csv(os.path.join(submission_dir, "submission.csv"), index=False)

    print(f"Final Validation Score: {final_validation_score}")


if __name__ == "__main__":
    main()
