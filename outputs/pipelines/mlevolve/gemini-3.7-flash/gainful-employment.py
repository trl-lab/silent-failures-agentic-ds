import os
import re
from typing import Any, Dict, Tuple
from catboost import CatBoostRegressor
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold
import xgboost as xgb

# ---------------------------------------------------------
# 1. Configuration and Paths
# ---------------------------------------------------------
INPUT_DIR = "./input"
SUBMISSION_DIR = "./submission"
WORKING_DIR = "./working"
os.makedirs(SUBMISSION_DIR, exist_ok=True)
os.makedirs(WORKING_DIR, exist_ok=True)

TRAIN_PATH = os.path.join(INPUT_DIR, "train.parquet")
TEST_PATH = os.path.join(INPUT_DIR, "test.parquet")
TARGET_COL = "Debt-to-Earnings Annual Rate"

# ---------------------------------------------------------
# 2. Data Cleaning & String Parsing Helpers
# ---------------------------------------------------------


def clean_numeric_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(np.float64)

    s_str = s.astype(str).str.strip()
    missing_mask = s_str.str.lower().isin(
        [
            "nan",
            "none",
            "null",
            "privacy suppressed",
            "privacy_suppressed",
            "suppressed",
            "n/a",
            "na",
            "-",
            "",
            ".",
            "unknown",
            "nd",
            "ps",
        ]
    )

    cleaned = (
        s_str.str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.replace(" ", "", regex=False)
    )

    num_s = pd.to_numeric(cleaned, errors="coerce")
    num_s[missing_mask] = np.nan
    return num_s


def preprocess_raw_data(train_df: pd.DataFrame, test_df: pd.DataFrame, target_col: str):
    train = train_df.copy()
    test = test_df.copy()

    # Separate target
    y = None
    if target_col in train.columns:
        y = clean_numeric_series(train[target_col])
        valid_y_mask = y.notna()
        train = train.loc[valid_y_mask].reset_index(drop=True)
        y = y.loc[valid_y_mask].reset_index(drop=True)
        feature_cols = [c for c in train.columns if c != target_col]
    else:
        feature_cols = list(train.columns)

    cols_to_process = [c for c in feature_cols if c in test.columns]
    numeric_cols = []
    categorical_cols = []

    for col in cols_to_process:
        combined_sample = (
            pd.concat([train[col], test[col]]).dropna().astype(str).str.strip()
        )
        cleaned_sample = pd.to_numeric(
            combined_sample.str.replace("$", "", regex=False)
            .str.replace(",", "", regex=False)
            .str.replace("%", "", regex=False)
            .str.replace(" ", "", regex=False),
            errors="coerce",
        )
        valid_num_ratio = (
            cleaned_sample.notna().mean() if len(cleaned_sample) > 0 else 0
        )

        if valid_num_ratio > 0.5:
            train[col] = clean_numeric_series(train[col])
            test[col] = clean_numeric_series(test[col])
            numeric_cols.append(col)
        else:
            train[col] = (
                train[col]
                .astype(str)
                .str.strip()
                .fillna("Missing")
                .replace({"nan": "Missing", "None": "Missing", "": "Missing"})
            )
            test[col] = (
                test[col]
                .astype(str)
                .str.strip()
                .fillna("Missing")
                .replace({"nan": "Missing", "None": "Missing", "": "Missing"})
            )
            categorical_cols.append(col)

    return train, test, y, numeric_cols, categorical_cols


def engineer_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1. CIP Taxonomy Features & Interaction Keys
    cip_candidates = [c for c in df.columns if "cip" in c.lower()]
    cred_candidates = [
        c
        for c in df.columns
        if any(k in c.lower() for k in ["cred", "degree", "level"])
    ]
    control_candidates = [
        c
        for c in df.columns
        if any(k in c.lower() for k in ["control", "type", "sector"])
    ]
    state_candidates = [c for c in df.columns if "state" in c.lower()]

    for col in cip_candidates:
        s = df[col].astype(str).str.strip()
        df[f"{col}_2digit"] = s.apply(
            lambda x: (
                re.findall(r"^\d{2}", str(x))[0]
                if re.findall(r"^\d{2}", str(x))
                else "Unknown"
            )
        )
        df[f"{col}_4digit"] = s.apply(
            lambda x: (
                re.findall(r"^\d{2}\.\d{2}|^\d{4}", str(x))[0]
                if re.findall(r"^\d{2}\.\d{2}|^\d{4}", str(x))
                else "Unknown"
            )
        )

    # Credential Level Ordinal Mapping and Statutory Repayment Schedule Period
    def map_cred_ordinal(val: Any) -> float:
        val_str = str(val).lower()
        if "undergrad" in val_str and "cert" in val_str:
            return 1.0
        if "cert" in val_str or "diploma" in val_str:
            return 1.5
        if "assoc" in val_str:
            return 2.0
        if "post-bac" in val_str or "post bac" in val_str:
            return 3.5
        if "bach" in val_str:
            return 3.0
        if "mast" in val_str:
            return 5.0
        if "doct" in val_str or "first prof" in val_str or "prof" in val_str:
            return 6.0
        num_m = re.findall(r"\d+", val_str)
        if num_m:
            return float(num_m[0])
        return 2.0

    for cred_c in cred_candidates:
        df[f"{cred_c}_ordinal"] = df[cred_c].apply(map_cred_ordinal)

    # Statutory Repayment Term (in years) based on federal Gainful Employment regulations:
    # Undergrad Cert / Associate: 10 yrs, Bachelor / Post-Bac: 15 yrs, Master / Doctoral: 20 yrs
    if len(cred_candidates) > 0:
        primary_cred = cred_candidates[0]
        ord_series = df[f"{primary_cred}_ordinal"]
        statutory_term_years = np.where(
            ord_series <= 2.0, 10.0, np.where(ord_series <= 3.5, 15.0, 20.0)
        )
        df["statutory_term_years"] = statutory_term_years
    else:
        df["statutory_term_years"] = 10.0

    # Composite categorical interaction keys
    cip_derived = [
        c
        for c in df.columns
        if "2digit" in c or "4digit" in c or "cip" in c.lower()
    ]
    for cip_c in cip_derived:
        for cred_c in cred_candidates:
            if cred_c in df.columns and cip_c in df.columns:
                df[f"inter_{cip_c}_x_{cred_c}"] = (
                    df[cip_c].astype(str) + "_" + df[cred_c].astype(str)
                )
        for ctrl_c in control_candidates:
            if ctrl_c in df.columns and cip_c in df.columns:
                df[f"inter_{cip_c}_x_{ctrl_c}"] = (
                    df[cip_c].astype(str) + "_" + df[ctrl_c].astype(str)
                )
        for st_c in state_candidates:
            if st_c in df.columns and cip_c in df.columns:
                df[f"inter_{cip_c}_x_{st_c}"] = (
                    df[cip_c].astype(str) + "_" + df[st_c].astype(str)
                )

    for cred_c in cred_candidates:
        for ctrl_c in control_candidates:
            if cred_c in df.columns and ctrl_c in df.columns:
                df[f"inter_{cred_c}_x_{ctrl_c}"] = (
                    df[cred_c].astype(str) + "_" + df[ctrl_c].astype(str)
                )
        for st_c in state_candidates:
            if cred_c in df.columns and st_c in df.columns:
                df[f"inter_{cred_c}_x_{st_c}"] = (
                    df[cred_c].astype(str) + "_" + df[st_c].astype(str)
                )

    # 2. Financial Ratios & Statutory Gainful Employment Formulas
    debt_cols = [
        c
        for c in df.columns
        if any(k in c.lower() for k in ["debt", "loan", "borrowed", "payment"])
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    earnings_cols = [
        c
        for c in df.columns
        if any(k in c.lower() for k in ["earn", "income", "wage", "salary"])
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    mean_earn_cols = [c for c in earnings_cols if "mean" in c.lower()]
    median_earn_cols = [c for c in earnings_cols if "median" in c.lower()]

    # Statutory Amortization Annuity Factors: f(r, n) = r * (1 + r)^n / ((1 + r)^n - 1)
    statutory_rates = [0.068, 0.050, 0.045, 0.0386]
    statutory_terms = [10, 15, 20]

    def annuity_factor(rate: float, n_years: float) -> float:
        return (rate * ((1.0 + rate) ** n_years)) / (((1.0 + rate) ** n_years) - 1.0)

    # Gainful Employment Rule: Denominator = max(Mean Earnings, Median Earnings)
    if len(mean_earn_cols) > 0 and len(median_earn_cols) > 0:
        for m_col in mean_earn_cols:
            for med_col in median_earn_cols:
                max_earn = np.maximum(df[m_col].fillna(0), df[med_col].fillna(0))
                min_earn = np.minimum(df[m_col].fillna(0), df[med_col].fillna(0))
                df[f"max_earnings_{m_col}_{med_col}"] = max_earn
                df[f"min_earnings_{m_col}_{med_col}"] = min_earn
                df[f"spread_earnings_{m_col}_{med_col}"] = df[m_col] - df[med_col]
                df[f"ratio_earnings_{m_col}_{med_col}"] = df[m_col] / np.maximum(100.0, df[med_col])

                for d_col in debt_cols:
                    # Dynamic credential-specific statutory amortization rate
                    for r in statutory_rates:
                        r_key = int(round(r * 10000))
                        dyn_ann_factor = (r * ((1.0 + r) ** df["statutory_term_years"])) / (
                            ((1.0 + r) ** df["statutory_term_years"]) - 1.0
                        )
                        dyn_ann_pmt = df[d_col] * dyn_ann_factor
                        df[f"statutory_pmt_{d_col}_r{r_key}"] = dyn_ann_pmt
                        df[f"ge_statutory_rate_{d_col}_r{r_key}"] = (
                            dyn_ann_pmt / np.maximum(100.0, max_earn)
                        ) * 100.0
                        df[f"ge_statutory_rate_mean_{d_col}_r{r_key}"] = (
                            dyn_ann_pmt / np.maximum(100.0, df[m_col])
                        ) * 100.0
                        df[f"ge_statutory_rate_med_{d_col}_r{r_key}"] = (
                            dyn_ann_pmt / np.maximum(100.0, df[med_col])
                        ) * 100.0

                    # Standard grid amortization schedules
                    for n in statutory_terms:
                        for r in [0.068, 0.050]:
                            factor = annuity_factor(r, n)
                            ann_pmt = df[d_col] * factor
                            r_lbl = int(round(r * 100))
                            df[f"ann_pmt_{d_col}_n{n}_r{r_lbl}"] = ann_pmt
                            df[f"ge_rate_{d_col}_n{n}_r{r_lbl}_max"] = (
                                ann_pmt / np.maximum(100.0, max_earn)
                            ) * 100.0

                    # Raw ratios
                    df[f"ge_rate_{d_col}_div_max_{m_col}_{med_col}"] = (
                        df[d_col] / np.maximum(100.0, max_earn)
                    ) * 100.0
                    df[f"ge_ratio_{d_col}_div_max_{m_col}_{med_col}"] = (
                        df[d_col] / np.maximum(100.0, max_earn)
                    )
                    df[f"ge_ratio_{d_col}_div_min_{m_col}_{med_col}"] = (
                        df[d_col] / np.maximum(100.0, min_earn)
                    )

    for d_col in debt_cols:
        df[f"{d_col}_log1p"] = np.log1p(np.maximum(0, df[d_col]))
        for e_col in earnings_cols:
            df[f"ratio_{d_col}_div_{e_col}"] = df[d_col] / np.maximum(100.0, df[e_col])
            df[f"ratio_{d_col}_div_{e_col}_pct"] = (
                df[d_col] / np.maximum(100.0, df[e_col])
            ) * 100.0
            df[f"diff_{d_col}_sub_{e_col}"] = df[d_col] - df[e_col]
            df[f"log_diff_{d_col}_{e_col}"] = np.log1p(np.maximum(0, df[d_col])) - np.log1p(
                np.maximum(0, df[e_col])
            )

    for e_col in earnings_cols:
        df[f"{e_col}_log1p"] = np.log1p(np.maximum(0, df[e_col]))

    # 3. Missingness Indicators
    for col in list(df.columns):
        if df[col].isna().sum() > 0:
            df[f"{col}_isna"] = df[col].isna().astype(np.float32)

    # Clean numerical anomalies
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)

    return df


# ---------------------------------------------------------
# 3. Model Architecture & Hyperparameter Definitions
# ---------------------------------------------------------


def get_lightgbm_params(seed: int = 42) -> Dict[str, Any]:
    return {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "n_estimators": 2500,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 20,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": seed,
        "n_jobs": -1,
        "verbose": -1,
    }


def get_xgboost_params(seed: int = 42) -> Dict[str, Any]:
    return {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "n_estimators": 2500,
        "learning_rate": 0.03,
        "max_depth": 6,
        "min_child_weight": 3,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": seed,
        "n_jobs": -1,
    }


def get_catboost_params(seed: int = 42) -> Dict[str, Any]:
    return {
        "iterations": 2500,
        "learning_rate": 0.03,
        "depth": 6,
        "l2_leaf_reg": 3.0,
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "random_seed": seed,
        "verbose": False,
        "early_stopping_rounds": 50,
    }


# ---------------------------------------------------------
# 4. Main Pipeline Execution
# ---------------------------------------------------------
def main():
    print(f"Loading data from {INPUT_DIR}...")
    train_raw = pd.read_parquet(TRAIN_PATH)
    test_raw = pd.read_parquet(TEST_PATH)

    print(f"Train raw shape: {train_raw.shape}, Test raw shape: {test_raw.shape}")

    train_clean, test_clean, y_train, num_cols, cat_cols = preprocess_raw_data(
        train_raw, test_raw, TARGET_COL
    )

    print("Engineering domain features...")
    train_feat = engineer_domain_features(train_clean)
    test_feat = engineer_domain_features(test_clean)

    engineered_cols = [
        c for c in train_feat.columns if c in test_feat.columns and c != TARGET_COL
    ]
    train_feat = train_feat[engineered_cols].copy()
    test_feat = test_feat[engineered_cols].copy()

    # 10-Fold Stratification/K-Fold Setup
    N_SPLITS = 10
    RANDOM_STATE = 42
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    folds = np.zeros(len(train_feat), dtype=np.int32)

    for fold, (train_idx, val_idx) in enumerate(kf.split(train_feat, y_train)):
        folds[val_idx] = fold

    # Categorical Frequency and Out-Of-Fold Multi-Statistic Target Encodings (Mean & Std)
    current_cat_cols = [
        c
        for c in train_feat.columns
        if train_feat[c].dtype == "object"
        or isinstance(train_feat[c].dtype, pd.CategoricalDtype)
    ]

    for cat_col in current_cat_cols:
        freq_map = train_feat[cat_col].value_counts(normalize=True).to_dict()
        train_feat[f"{cat_col}_freq"] = (
            train_feat[cat_col].map(freq_map).fillna(0.0).astype(np.float32)
        )
        test_feat[f"{cat_col}_freq"] = (
            test_feat[cat_col].map(freq_map).fillna(0.0).astype(np.float32)
        )

        oof_te_mean = np.zeros(len(train_feat), dtype=np.float32)
        oof_te_std = np.zeros(len(train_feat), dtype=np.float32)
        test_te_mean_accum = np.zeros(len(test_feat), dtype=np.float32)
        test_te_std_accum = np.zeros(len(test_feat), dtype=np.float32)

        smooth_weight = 10.0
        global_target_mean = float(y_train.mean())
        global_target_std = float(y_train.std()) if not np.isnan(y_train.std()) else 0.0

        for fold in range(N_SPLITS):
            tr_idx = folds != fold
            va_idx = folds == fold

            tr_cat = train_feat.loc[tr_idx, cat_col]
            tr_y = y_train.iloc[tr_idx]

            stats = tr_y.groupby(tr_cat).agg(["count", "mean", "std"])
            smoothed_means = (
                stats["count"] * stats["mean"] + smooth_weight * global_target_mean
            ) / (stats["count"] + smooth_weight)
            smoothed_stds = (
                stats["count"] * stats["std"].fillna(global_target_std) + smooth_weight * global_target_std
            ) / (stats["count"] + smooth_weight)

            mean_map = smoothed_means.to_dict()
            std_map = smoothed_stds.to_dict()

            oof_te_mean[va_idx] = (
                train_feat.loc[va_idx, cat_col].map(mean_map).fillna(global_target_mean)
            )
            oof_te_std[va_idx] = (
                train_feat.loc[va_idx, cat_col].map(std_map).fillna(global_target_std)
            )

            test_te_mean_accum += (
                test_feat[cat_col].map(mean_map).fillna(global_target_mean).to_numpy()
                / N_SPLITS
            )
            test_te_std_accum += (
                test_feat[cat_col].map(std_map).fillna(global_target_std).to_numpy()
                / N_SPLITS
            )

        train_feat[f"{cat_col}_target_mean_enc"] = oof_te_mean
        test_feat[f"{cat_col}_target_mean_enc"] = test_te_mean_accum.astype(np.float32)
        train_feat[f"{cat_col}_target_std_enc"] = oof_te_std
        test_feat[f"{cat_col}_target_std_enc"] = test_te_std_accum.astype(np.float32)

    # Label encode remaining categorical columns
    for cat_col in current_cat_cols:
        cat_type = pd.Categorical(
            pd.concat([train_feat[cat_col], test_feat[cat_col]]).astype(str)
        )
        train_feat[f"{cat_col}_code"] = pd.Categorical(
            train_feat[cat_col].astype(str), categories=cat_type.categories
        ).codes.astype(np.int32)
        test_feat[f"{cat_col}_code"] = pd.Categorical(
            test_feat[cat_col].astype(str), categories=cat_type.categories
        ).codes.astype(np.int32)

        train_feat.drop(columns=[cat_col], inplace=True)
        test_feat.drop(columns=[cat_col], inplace=True)

    # Group Aggregation Statistics
    group_key_candidates = [
        c
        for c in train_feat.columns
        if any(k in c.lower() for k in ["cip", "cred", "state", "control"])
        and c.endswith("_code")
    ]
    num_target_candidates = [
        c
        for c in train_feat.columns
        if any(k in c.lower() for k in ["debt", "earn", "loan"])
        and not c.endswith("_isna")
        and not c.endswith("_freq")
        and not c.endswith("_enc")
    ]

    for g_col in group_key_candidates[:3]:
        for n_col in num_target_candidates[:3]:
            grp_stats = (
                train_feat.groupby(g_col)[n_col]
                .agg(["mean", "std"])
                .rename(
                    columns={
                        "mean": f"grp_{g_col}_{n_col}_mean",
                        "std": f"grp_{g_col}_{n_col}_std",
                    }
                )
            )
            train_feat = train_feat.join(grp_stats, on=g_col)
            test_feat = test_feat.join(grp_stats, on=g_col)

    # Clean feature names for boosters
    clean_cols = [re.sub(r"[^A-Za-z0-9_]+", "_", str(c)) for c in train_feat.columns]
    train_feat.columns = clean_cols
    test_feat.columns = clean_cols

    feature_cols = list(train_feat.columns)
    n_train = len(train_feat)
    n_test = len(test_feat)
    y_all = y_train.to_numpy(dtype=np.float64)
    y_all_log = np.log1p(np.maximum(0.0, y_all))

    # Model predictions: 3 architectures x 2 target scales (raw & log1p)
    model_names = [
        "lgb_raw", "lgb_log",
        "xgb_raw", "xgb_log",
        "cb_raw", "cb_log"
    ]
    oof_preds = {m: np.zeros(n_train, dtype=np.float64) for m in model_names}
    test_preds = {m: np.zeros(n_test, dtype=np.float64) for m in model_names}

    print(
        f"Starting {N_SPLITS}-fold CV on {n_train} samples with {len(feature_cols)} features and dual-scale training..."
    )

    for fold in range(N_SPLITS):
        tr_mask = folds != fold
        va_mask = folds == fold

        X_train_fold = train_feat.loc[tr_mask].values
        y_train_raw = y_all[tr_mask]
        y_train_log = y_all_log[tr_mask]

        X_val_fold = train_feat.loc[va_mask].values
        y_val_raw = y_all[va_mask]
        y_val_log = y_all_log[va_mask]

        X_test_all = test_feat.values
        val_indices = np.where(va_mask)[0]

        # 1. LightGBM (Raw Target)
        lgb_raw_params = get_lightgbm_params(seed=RANDOM_STATE + fold)
        lgb_raw = lgb.LGBMRegressor(**lgb_raw_params)
        lgb_raw.fit(
            X_train_fold,
            y_train_raw,
            eval_set=[(X_val_fold, y_val_raw)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )
        pred_val_lgb_raw = np.maximum(0.0, lgb_raw.predict(X_val_fold))
        oof_preds["lgb_raw"][val_indices] = pred_val_lgb_raw
        test_preds["lgb_raw"] += np.maximum(0.0, lgb_raw.predict(X_test_all)) / N_SPLITS

        # 2. LightGBM (Log1p Target)
        lgb_log_params = get_lightgbm_params(seed=RANDOM_STATE + 100 + fold)
        lgb_log = lgb.LGBMRegressor(**lgb_log_params)
        lgb_log.fit(
            X_train_fold,
            y_train_log,
            eval_set=[(X_val_fold, y_val_log)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )
        pred_val_lgb_log = np.maximum(0.0, np.expm1(lgb_log.predict(X_val_fold)))
        oof_preds["lgb_log"][val_indices] = pred_val_lgb_log
        test_preds["lgb_log"] += np.maximum(0.0, np.expm1(lgb_log.predict(X_test_all))) / N_SPLITS

        # 3. XGBoost (Raw Target)
        xgb_raw_params = get_xgboost_params(seed=RANDOM_STATE + fold)
        xgb_raw = xgb.XGBRegressor(early_stopping_rounds=50, **xgb_raw_params)
        xgb_raw.fit(
            X_train_fold,
            y_train_raw,
            eval_set=[(X_val_fold, y_val_raw)],
            verbose=False,
        )
        pred_val_xgb_raw = np.maximum(0.0, xgb_raw.predict(X_val_fold))
        oof_preds["xgb_raw"][val_indices] = pred_val_xgb_raw
        test_preds["xgb_raw"] += np.maximum(0.0, xgb_raw.predict(X_test_all)) / N_SPLITS

        # 4. XGBoost (Log1p Target)
        xgb_log_params = get_xgboost_params(seed=RANDOM_STATE + 100 + fold)
        xgb_log = xgb.XGBRegressor(early_stopping_rounds=50, **xgb_log_params)
        xgb_log.fit(
            X_train_fold,
            y_train_log,
            eval_set=[(X_val_fold, y_val_log)],
            verbose=False,
        )
        pred_val_xgb_log = np.maximum(0.0, np.expm1(xgb_log.predict(X_val_fold)))
        oof_preds["xgb_log"][val_indices] = pred_val_xgb_log
        test_preds["xgb_log"] += np.maximum(0.0, np.expm1(xgb_log.predict(X_test_all))) / N_SPLITS

        # 5. CatBoost (Raw Target)
        cb_raw_params = get_catboost_params(seed=RANDOM_STATE + fold)
        cb_raw = CatBoostRegressor(**cb_raw_params)
        cb_raw.fit(
            X_train_fold,
            y_train_raw,
            eval_set=(X_val_fold, y_val_raw),
            verbose=False,
        )
        pred_val_cb_raw = np.maximum(0.0, cb_raw.predict(X_val_fold))
        oof_preds["cb_raw"][val_indices] = pred_val_cb_raw
        test_preds["cb_raw"] += np.maximum(0.0, cb_raw.predict(X_test_all)) / N_SPLITS

        # 6. CatBoost (Log1p Target)
        cb_log_params = get_catboost_params(seed=RANDOM_STATE + 100 + fold)
        cb_log = CatBoostRegressor(**cb_log_params)
        cb_log.fit(
            X_train_fold,
            y_train_log,
            eval_set=(X_val_fold, y_val_log),
            verbose=False,
        )
        pred_val_cb_log = np.maximum(0.0, np.expm1(cb_log.predict(X_val_fold)))
        oof_preds["cb_log"][val_indices] = pred_val_cb_log
        test_preds["cb_log"] += np.maximum(0.0, np.expm1(cb_log.predict(X_test_all))) / N_SPLITS

        fold_rmse_lgb = root_mean_squared_error(y_val_raw, pred_val_lgb_raw)
        fold_rmse_lgb_log = root_mean_squared_error(y_val_raw, pred_val_lgb_log)
        fold_rmse_xgb = root_mean_squared_error(y_val_raw, pred_val_xgb_raw)
        fold_rmse_xgb_log = root_mean_squared_error(y_val_raw, pred_val_xgb_log)
        fold_rmse_cb = root_mean_squared_error(y_val_raw, pred_val_cb_raw)
        fold_rmse_cb_log = root_mean_squared_error(y_val_raw, pred_val_cb_log)

        print(
            f"Fold {fold+1:02d}/{N_SPLITS:02d} -> "
            f"LGB: {fold_rmse_lgb:.4f} (log: {fold_rmse_lgb_log:.4f}) | "
            f"XGB: {fold_rmse_xgb:.4f} (log: {fold_rmse_xgb_log:.4f}) | "
            f"CB: {fold_rmse_cb:.4f} (log: {fold_rmse_cb_log:.4f})"
        )

    # ---------------------------------------------------------
    # 5. Out-of-Fold Multi-Model Ensemble Optimization
    # ---------------------------------------------------------
    oof_matrix = np.column_stack([oof_preds[m] for m in model_names])
    test_matrix = np.column_stack([test_preds[m] for m in model_names])

    def ensemble_loss(weights: np.ndarray) -> float:
        w = weights / np.sum(weights)
        blend_pred = np.maximum(0.0, oof_matrix @ w)
        return root_mean_squared_error(y_all, blend_pred)

    num_models = len(model_names)
    initial_weights = np.ones(num_models, dtype=np.float64) / num_models
    bounds = [(0.0, 1.0)] * num_models
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

    opt_res = minimize(
        ensemble_loss,
        initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    best_weights = opt_res.x / np.sum(opt_res.x)

    print("Optimal Ensemble Weights:")
    for name, w in zip(model_names, best_weights):
        print(f"  {name}: {w:.4f}")

    oof_ensemble = np.maximum(0.0, oof_matrix @ best_weights)
    final_score = float(root_mean_squared_error(y_all, oof_ensemble))

    # ---------------------------------------------------------
    # 6. Generate Submission
    # ---------------------------------------------------------
    final_test_preds = np.maximum(0.0, test_matrix @ best_weights)

    submission_df = pd.DataFrame(
        {"id": np.arange(n_test, dtype=int), TARGET_COL: final_test_preds}
    )
    sub_file_path = os.path.join(SUBMISSION_DIR, "submission.csv")
    submission_df.to_csv(sub_file_path, index=False)

    print(f"Final Validation Score: {final_score}")


if __name__ == "__main__":
    main()
