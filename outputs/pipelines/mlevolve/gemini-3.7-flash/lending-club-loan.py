import os
import re
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from scipy.optimize import minimize
from scipy.stats import rankdata

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from catboost import CatBoostClassifier
    HAS_CAT = True
except ImportError:
    HAS_CAT = False


def parse_emp_length(series: pd.Series) -> pd.Series:
    """Convert employment length strings to numeric years."""
    mapping = {
        "< 1 year": 0.5,
        "1 year": 1.0,
        "2 years": 2.0,
        "3 years": 3.0,
        "4 years": 4.0,
        "5 years": 5.0,
        "6 years": 6.0,
        "7 years": 7.0,
        "8 years": 8.0,
        "9 years": 9.0,
        "10+ years": 10.0,
        "10+": 10.0,
        "< 1": 0.5,
        "n/a": np.nan,
    }

    def extract_val(x):
        if pd.isna(x):
            return np.nan
        x_str = str(x).strip()
        if x_str in mapping:
            return mapping[x_str]
        digits = re.findall(r"\d+", x_str)
        if digits:
            return float(digits[0])
        return np.nan

    return series.map(extract_val)


def parse_term(series: pd.Series) -> pd.Series:
    """Convert loan term strings to numeric months."""

    def extract_months(x):
        if pd.isna(x):
            return np.nan
        x_str = str(x).strip()
        digits = re.findall(r"\d+", x_str)
        if digits:
            return float(digits[0])
        return np.nan

    return series.map(extract_months)


def parse_percentage(series: pd.Series) -> pd.Series:
    """Convert string percentage values to numeric floats."""

    def extract_pct(x):
        if pd.isna(x):
            return np.nan
        if isinstance(x, (int, float)):
            return float(x)
        x_str = str(x).replace("%", "").strip()
        try:
            return float(x_str)
        except ValueError:
            return np.nan

    return series.map(extract_pct)


def parse_date_to_months(series: pd.Series, ref_year: int = 2020) -> pd.Series:
    """Convert date strings to elapsed months from reference year."""

    def to_months(x):
        if pd.isna(x):
            return np.nan
        try:
            dt = pd.to_datetime(x, format="mixed")
            return (ref_year - dt.year) * 12 + (12 - dt.month)
        except Exception:
            return np.nan

    return series.map(to_months)


def encode_grades(df: pd.DataFrame) -> pd.DataFrame:
    """Encode loan grade and sub_grade into ordered numerical values."""
    grade_map = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7}
    if "grade" in df.columns:
        df["grade_num"] = df["grade"].map(grade_map).fillna(np.nan)

    if "sub_grade" in df.columns:

        def sub_grade_to_num(x):
            if pd.isna(x):
                return np.nan
            x_str = str(x).strip()
            if len(x_str) >= 2 and x_str[0] in grade_map and x_str[1:].isdigit():
                return (grade_map[x_str[0]] - 1) * 5 + int(x_str[1:])
            return np.nan

        df["sub_grade_num"] = df["sub_grade"].map(sub_grade_to_num)

    return df


def engineer_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract financial, credit history, and debt burden domain features."""
    df = df.copy()

    # Numeric cleaning
    if "term" in df.columns:
        df["term_num"] = parse_term(df["term"])
    if "emp_length" in df.columns:
        df["emp_length_num"] = parse_emp_length(df["emp_length"])
    if "int_rate" in df.columns:
        df["int_rate_clean"] = parse_percentage(df["int_rate"])
    if "revol_util" in df.columns:
        df["revol_util_clean"] = parse_percentage(df["revol_util"])

    df = encode_grades(df)

    # Date parsing
    if "earliest_cr_line" in df.columns:
        df["earliest_cr_line_months"] = parse_date_to_months(df["earliest_cr_line"])
    if "issue_d" in df.columns:
        df["issue_d_months"] = parse_date_to_months(df["issue_d"])
    if "earliest_cr_line_months" in df.columns and "issue_d_months" in df.columns:
        df["credit_history_length_months"] = (
            df["earliest_cr_line_months"] - df["issue_d_months"]
        )

    # Financial Ratios
    ann_inc = df["annual_inc"].clip(lower=1.0) if "annual_inc" in df.columns else None
    monthly_inc = (ann_inc / 12.0) if ann_inc is not None else None
    loan_amt = (
        df["loan_amnt"]
        if "loan_amnt" in df.columns
        else (df["funded_amnt"] if "funded_amnt" in df.columns else None)
    )
    installment = df["installment"] if "installment" in df.columns else None
    term_num = df["term_num"] if "term_num" in df.columns else 36.0

    if monthly_inc is not None and installment is not None:
        df["installment_to_monthly_inc"] = installment / monthly_inc
        df["monthly_disposable_inc"] = monthly_inc - installment

    if ann_inc is not None and loan_amt is not None:
        df["loan_to_annual_inc"] = loan_amt / ann_inc

    if installment is not None and loan_amt is not None and term_num is not None:
        df["total_expected_payment"] = installment * term_num
        df["total_interest_burden"] = df["total_expected_payment"] - loan_amt
        df["interest_to_loan_ratio"] = df["total_interest_burden"] / (
            loan_amt.clip(lower=1.0)
        )

    # FICO Score Features
    if "fico_range_low" in df.columns and "fico_range_high" in df.columns:
        df["fico_mid"] = (df["fico_range_low"] + df["fico_range_high"]) / 2.0
        df["fico_spread"] = df["fico_range_high"] - df["fico_range_low"]

    if "last_fico_range_low" in df.columns and "last_fico_range_high" in df.columns:
        df["last_fico_mid"] = (
            df["last_fico_range_low"] + df["last_fico_range_high"]
        ) / 2.0
        df["last_fico_spread"] = df["last_fico_range_high"] - df["last_fico_range_low"]
        if "fico_mid" in df.columns:
            df["fico_change"] = df["last_fico_mid"] - df["fico_mid"]

    # Credit Balance, Capacity & Utilization Ratios
    if "revol_bal" in df.columns and ann_inc is not None:
        df["revol_bal_to_annual_inc"] = df["revol_bal"] / ann_inc

    if "total_rev_hi_lim" in df.columns and "revol_bal" in df.columns:
        df["available_credit"] = (df["total_rev_hi_lim"] - df["revol_bal"]).clip(
            lower=0.0
        )
        df["revol_util_calc"] = df["revol_bal"] / (
            df["total_rev_hi_lim"].clip(lower=1.0)
        )

    if "tot_cur_bal" in df.columns and "tot_hi_cred_lim" in df.columns:
        df["tot_cur_bal_to_hi_lim"] = df["tot_cur_bal"] / (
            df["tot_hi_cred_lim"].clip(lower=1.0)
        )
        df["tot_cur_bal_unused_lim"] = (
            df["tot_hi_cred_lim"] - df["tot_cur_bal"]
        ).clip(lower=0.0)

    if "bc_open_to_buy" in df.columns and "total_bc_limit" in df.columns:
        df["bc_open_to_limit_ratio"] = df["bc_open_to_buy"] / (
            df["total_bc_limit"].clip(lower=1.0)
        )

    if "total_bal_ex_mort" in df.columns and "tot_hi_cred_lim" in df.columns:
        df["bal_ex_mort_to_limit_ratio"] = df["total_bal_ex_mort"] / (
            df["tot_hi_cred_lim"].clip(lower=1.0)
        )

    # Credit Velocity and Line Saturation Metrics
    if "open_acc" in df.columns and "total_acc" in df.columns:
        df["open_to_total_acc_ratio"] = df["open_acc"] / (
            df["total_acc"].clip(lower=1.0)
        )

    if "acc_open_past_24mths" in df.columns and "total_acc" in df.columns:
        df["acc_open_24m_to_total_ratio"] = df["acc_open_past_24mths"] / (
            df["total_acc"].clip(lower=1.0)
        )

    if "num_tl_op_past_12m" in df.columns and "total_acc" in df.columns:
        df["tl_op_12m_to_total_ratio"] = df["num_tl_op_past_12m"] / (
            df["total_acc"].clip(lower=1.0)
        )

    if "num_actv_rev_tl" in df.columns and "num_rev_accts" in df.columns:
        df["actv_rev_to_rev_accts_ratio"] = df["num_actv_rev_tl"] / (
            df["num_rev_accts"].clip(lower=1.0)
        )

    if "num_actv_bc_tl" in df.columns and "num_bc_tl" in df.columns:
        df["actv_bc_to_bc_tl_ratio"] = df["num_actv_bc_tl"] / (
            df["num_bc_tl"].clip(lower=1.0)
        )

    if "num_op_rev_tl" in df.columns and "total_acc" in df.columns:
        df["op_rev_to_total_acc_ratio"] = df["num_op_rev_tl"] / (
            df["total_acc"].clip(lower=1.0)
        )

    if "num_sats" in df.columns and "total_acc" in df.columns:
        df["sats_to_total_acc_ratio"] = df["num_sats"] / (
            df["total_acc"].clip(lower=1.0)
        )

    if "num_bc_sats" in df.columns and "num_bc_tl" in df.columns:
        df["bc_sats_to_bc_tl_ratio"] = df["num_bc_sats"] / (
            df["num_bc_tl"].clip(lower=1.0)
        )

    if "mort_acc" in df.columns and "total_acc" in df.columns:
        df["mort_to_total_acc_ratio"] = df["mort_acc"] / (
            df["total_acc"].clip(lower=1.0)
        )

    if "delinq_2yrs" in df.columns and "total_acc" in df.columns:
        df["delinq_to_total_acc_ratio"] = df["delinq_2yrs"] / (
            df["total_acc"].clip(lower=1.0)
        )

    if "pub_rec" in df.columns and "total_acc" in df.columns:
        df["pub_rec_to_total_acc_ratio"] = df["pub_rec"] / (
            df["total_acc"].clip(lower=1.0)
        )

    if "inq_last_6mths" in df.columns and "open_acc" in df.columns:
        df["inq_last_6mths_to_open_acc"] = df["inq_last_6mths"] / (
            df["open_acc"].clip(lower=1.0)
        )

    # Advanced Net Disposable Income & Debt Burden
    if monthly_inc is not None and "dti" in df.columns and installment is not None:
        estimated_debt_obligations = monthly_inc * (df["dti"].clip(lower=0.0, upper=100.0) / 100.0)
        df["net_disposable_inc_post_all_debt"] = monthly_inc - estimated_debt_obligations - installment
        df["disposable_inc_ratio"] = df["net_disposable_inc_post_all_debt"] / monthly_inc

    # Composite Delinquency Severity Index
    delinq_components = []
    if "delinq_2yrs" in df.columns:
        delinq_components.append(df["delinq_2yrs"].fillna(0.0) * 2.0)
    if "pub_rec" in df.columns:
        delinq_components.append(df["pub_rec"].fillna(0.0) * 2.5)
    if "pub_rec_bankruptcies" in df.columns:
        delinq_components.append(df["pub_rec_bankruptcies"].fillna(0.0) * 3.0)
    if "tax_liens" in df.columns:
        delinq_components.append(df["tax_liens"].fillna(0.0) * 2.0)
    if "collections_12_mths_ex_med" in df.columns:
        delinq_components.append(df["collections_12_mths_ex_med"].fillna(0.0) * 2.5)
    if "acc_now_delinq" in df.columns:
        delinq_components.append(df["acc_now_delinq"].fillna(0.0) * 4.0)
    if delinq_components:
        df["composite_delinquency_severity"] = sum(delinq_components)

    # Bankcard and Credit Saturation Ratios
    if "total_bc_limit" in df.columns and "tot_hi_cred_lim" in df.columns:
        df["bc_limit_to_total_lim"] = df["total_bc_limit"] / df["tot_hi_cred_lim"].clip(lower=1.0)
    if "revol_bal" in df.columns and "total_bc_limit" in df.columns:
        df["revol_bal_to_bc_limit"] = df["revol_bal"] / df["total_bc_limit"].clip(lower=1.0)
    if "total_bal_ex_mort" in df.columns and "total_rev_hi_lim" in df.columns:
        df["bal_ex_mort_to_rev_lim"] = df["total_bal_ex_mort"] / df["total_rev_hi_lim"].clip(lower=1.0)

    # Multi-way Cohort Categorical Interactions
    if "sub_grade" in df.columns and "term" in df.columns:
        df["sub_grade_x_term"] = df["sub_grade"].astype(str) + "_" + df["term"].astype(str)
    if "purpose" in df.columns and "home_ownership" in df.columns:
        df["purpose_x_home_ownership"] = df["purpose"].astype(str) + "_" + df["home_ownership"].astype(str)

    # Text length & word count statistics for emp_title and title
    if "emp_title" in df.columns:
        emp_title_clean = df["emp_title"].fillna("").astype(str)
        df["emp_title_char_len"] = emp_title_clean.str.len()
        df["emp_title_word_cnt"] = emp_title_clean.str.split().str.len()
        df["emp_title_is_missing"] = df["emp_title"].isna().astype(np.int8)

    if "title" in df.columns:
        title_clean = df["title"].fillna("").astype(str)
        df["title_char_len"] = title_clean.str.len()
        df["title_word_cnt"] = title_clean.str.split().str.len()
        df["title_is_missing"] = df["title"].isna().astype(np.int8)

    # Debt and Interest Rate Burden Interactions
    if "dti" in df.columns and "int_rate_clean" in df.columns:
        df["dti_x_int_rate"] = df["dti"] * df["int_rate_clean"]

    if "loan_to_annual_inc" in df.columns and "int_rate_clean" in df.columns:
        df["loan_inc_x_int_rate"] = df["loan_to_annual_inc"] * df["int_rate_clean"]

    # Per-account balance / limit and credit line capacity features
    if "revol_bal" in df.columns and "num_actv_rev_tl" in df.columns:
        df["revol_bal_per_actv_rev_tl"] = df["revol_bal"] / df["num_actv_rev_tl"].clip(lower=1.0)
    if "revol_bal" in df.columns and "num_rev_accts" in df.columns:
        df["revol_bal_per_rev_acct"] = df["revol_bal"] / df["num_rev_accts"].clip(lower=1.0)
    if "tot_cur_bal" in df.columns and "total_acc" in df.columns:
        df["tot_cur_bal_per_total_acc"] = df["tot_cur_bal"] / df["total_acc"].clip(lower=1.0)
    if "tot_cur_bal" in df.columns and "open_acc" in df.columns:
        df["tot_cur_bal_per_open_acc"] = df["tot_cur_bal"] / df["open_acc"].clip(lower=1.0)
    if "total_bc_limit" in df.columns and "num_bc_tl" in df.columns:
        df["bc_limit_per_bc_tl"] = df["total_bc_limit"] / df["num_bc_tl"].clip(lower=1.0)
    if "total_rev_hi_lim" in df.columns and "num_rev_accts" in df.columns:
        df["rev_hi_lim_per_rev_acct"] = df["total_rev_hi_lim"] / df["num_rev_accts"].clip(lower=1.0)
    if "tot_hi_cred_lim" in df.columns and "total_acc" in df.columns:
        df["hi_cred_lim_per_total_acc"] = df["tot_hi_cred_lim"] / df["total_acc"].clip(lower=1.0)

    # Exponential recency decay transforms for delinquency and inquiry variables
    recency_cols = [
        "mths_since_last_delinq",
        "mths_since_recent_inq",
        "mths_since_last_record",
        "mths_since_last_major_derog",
        "mths_since_recent_bc",
        "mths_since_recent_revol_delinq",
    ]
    for col in recency_cols:
        if col in df.columns:
            val_clean = df[col].fillna(999.0).clip(lower=0.0)
            df[f"{col}_exp_decay_6m"] = np.exp(-val_clean / 6.0).astype(np.float32)
            df[f"{col}_exp_decay_12m"] = np.exp(-val_clean / 12.0).astype(np.float32)
            df[f"{col}_exp_decay_24m"] = np.exp(-val_clean / 24.0).astype(np.float32)

    # Missingness indicator features
    for col in [
        "mths_since_last_delinq",
        "mths_since_last_record",
        "mths_since_last_major_derog",
        "mths_since_recent_inq",
        "mths_since_recent_bc",
        "mths_since_recent_revol_delinq",
    ]:
        if col in df.columns:
            df[f"{col}_is_missing"] = df[col].isna().astype(np.int8)

    return df


def add_fold_contextual_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_y: pd.Series,
    smoothing: float = 10.0,
):
    """Compute group aggregations and out-of-fold target encodings strictly on train fold."""
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    # Text SVD feature extraction strictly fitted on train fold
    for text_col in ["emp_title", "title"]:
        if text_col in train_df.columns:
            tr_text = train_df[text_col].fillna("").astype(str)
            va_text = val_df[text_col].fillna("").astype(str)
            te_text = test_df[text_col].fillna("").astype(str)

            tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=3, token_pattern=r"(?u)\b\w+\b")
            tr_tfidf = tfidf.fit_transform(tr_text)
            va_tfidf = tfidf.transform(va_text)
            te_tfidf = tfidf.transform(te_text)

            n_comps = 8
            svd = TruncatedSVD(n_components=n_comps, random_state=42)
            tr_svd = svd.fit_transform(tr_tfidf)
            va_svd = svd.transform(va_tfidf)
            te_svd = svd.transform(te_tfidf)

            for comp_idx in range(n_comps):
                comp_name = f"{text_col}_svd_{comp_idx}"
                train_df[comp_name] = tr_svd[:, comp_idx].astype(np.float32)
                val_df[comp_name] = va_svd[:, comp_idx].astype(np.float32)
                test_df[comp_name] = te_svd[:, comp_idx].astype(np.float32)

    # 1. Group aggregations (mean, std, min, max)
    group_cols = [
        col
        for col in [
            "sub_grade_num",
            "grade_num",
            "purpose",
            "home_ownership",
            "addr_state",
            "sub_grade_x_term",
            "purpose_x_home_ownership",
        ]
        if col in train_df.columns
    ]
    target_num_cols = [
        col
        for col in [
            "int_rate_clean",
            "dti",
            "loan_amnt",
            "installment",
            "installment_to_monthly_inc",
            "net_disposable_inc_post_all_debt",
            "revol_bal_per_actv_rev_tl",
        ]
        if col in train_df.columns
    ]

    for g_col in group_cols:
        for n_col in target_num_cols:
            stat_df = (
                train_df.groupby(g_col, observed=False)[n_col]
                .agg(["mean", "std", "min", "max"])
                .reset_index()
            )
            mean_col = f"{g_col}_{n_col}_mean"
            std_col = f"{g_col}_{n_col}_std"
            min_col = f"{g_col}_{n_col}_min"
            max_col = f"{g_col}_{n_col}_max"
            stat_df.columns = [g_col, mean_col, std_col, min_col, max_col]

            train_df = train_df.merge(stat_df, on=g_col, how="left")
            val_df = val_df.merge(stat_df, on=g_col, how="left")
            test_df = test_df.merge(stat_df, on=g_col, how="left")

            diff_col = f"{n_col}_diff_{g_col}_mean"
            train_df[diff_col] = train_df[n_col] - train_df[mean_col]
            val_df[diff_col] = val_df[n_col] - val_df[mean_col]
            test_df[diff_col] = test_df[n_col] - test_df[mean_col]

            z_col = f"{n_col}_z_{g_col}"
            train_df[z_col] = (train_df[n_col] - train_df[mean_col]) / (train_df[std_col].fillna(1.0) + 1e-5)
            val_df[z_col] = (val_df[n_col] - val_df[mean_col]) / (val_df[std_col].fillna(1.0) + 1e-5)
            test_df[z_col] = (test_df[n_col] - test_df[mean_col]) / (test_df[std_col].fillna(1.0) + 1e-5)

    # 2. Out-of-fold target encoding for high-cardinality & key categoricals
    te_cols = [
        col
        for col in [
            "sub_grade",
            "purpose",
            "addr_state",
            "home_ownership",
            "grade",
            "zip_code",
            "sub_grade_x_term",
            "purpose_x_home_ownership",
        ]
        if col in train_df.columns
    ]
    global_mean = float(train_y.mean())
    inner_kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for c in te_cols:
        train_te = np.full(len(train_df), np.nan, dtype=np.float32)
        train_c = train_df[c].astype(str)
        val_c = val_df[c].astype(str)
        test_c = test_df[c].astype(str)

        # Internal cross-fitting for train target encoding
        for in_tr_idx, in_val_idx in inner_kf.split(train_df, train_y):
            sub_x, sub_y = train_c.iloc[in_tr_idx], train_y.iloc[in_tr_idx]
            sub_val_x = train_c.iloc[in_val_idx]
            counts = sub_y.groupby(sub_x, observed=False).count()
            means = sub_y.groupby(sub_x, observed=False).mean()
            smooth = (counts * means + smoothing * global_mean) / (counts + smoothing)
            train_te[in_val_idx] = sub_val_x.map(smooth).fillna(global_mean).values.astype(np.float32)

        train_df[f"{c}_target_enc"] = train_te

        # Overall fold mapping for val and test
        counts = train_y.groupby(train_c, observed=False).count()
        means = train_y.groupby(train_c, observed=False).mean()
        smooth_map = (counts * means + smoothing * global_mean) / (counts + smoothing)
        val_df[f"{c}_target_enc"] = (
            val_c.map(smooth_map).fillna(global_mean).values.astype(np.float32)
        )
        test_df[f"{c}_target_enc"] = (
            test_c.map(smooth_map).fillna(global_mean).values.astype(np.float32)
        )

    # 3. Frequency encoding & Label encoding for categorical columns
    cat_cols = train_df.select_dtypes(include=["object", "category"]).columns.tolist()
    cols_to_drop = [c for c in ["emp_title", "title", "url", "desc", "id"] if c in cat_cols]
    train_df.drop(columns=cols_to_drop, errors="ignore", inplace=True)
    val_df.drop(columns=cols_to_drop, errors="ignore", inplace=True)
    test_df.drop(columns=cols_to_drop, errors="ignore", inplace=True)

    rem_cats = train_df.select_dtypes(include=["object", "category"]).columns.tolist()
    for col in rem_cats:
        freq_map = train_df[col].value_counts(normalize=True).to_dict()
        train_df[f"{col}_freq"] = train_df[col].map(freq_map).fillna(0.0).astype(np.float32)
        val_df[f"{col}_freq"] = val_df[col].map(freq_map).fillna(0.0).astype(np.float32)
        test_df[f"{col}_freq"] = test_df[col].map(freq_map).fillna(0.0).astype(np.float32)

        # Robust label encoding
        unique_vals = {v: i for i, v in enumerate(train_df[col].dropna().unique())}
        train_df[col] = train_df[col].map(unique_vals).fillna(-1).astype(np.int32)
        val_df[col] = val_df[col].map(unique_vals).fillna(-1).astype(np.int32)
        test_df[col] = test_df[col].map(unique_vals).fillna(-1).astype(np.int32)

    return train_df, val_df, test_df


def parse_target_column(y_raw: pd.Series) -> pd.Series:
    """Clean and binary-encode loan_status target."""
    if pd.api.types.is_numeric_dtype(y_raw):
        return y_raw.astype(int)

    pos_labels = {
        "charged off",
        "default",
        "late (31-120 days)",
        "does not meet the credit policy. status:charged off",
        "1",
    }
    neg_labels = {
        "fully paid",
        "current",
        "in grace period",
        "late (16-30 days)",
        "does not meet the credit policy. status:fully paid",
        "0",
    }

    def map_status(val):
        if pd.isna(val):
            return np.nan
        val_lower = str(val).strip().lower()
        if val_lower in pos_labels:
            return 1
        elif val_lower in neg_labels:
            return 0
        return 1 if "charge" in val_lower or "default" in val_lower else 0

    return y_raw.map(map_status).astype(int)


def main():
    os.makedirs("./submission", exist_ok=True)
    os.makedirs("./working", exist_ok=True)

    # 1. Load Data
    train_raw = pd.read_parquet("./input/train.parquet")
    test_raw = pd.read_parquet("./input/test.parquet")

    target_col = "loan_status"
    y = parse_target_column(train_raw[target_col]).values
    X = train_raw.drop(columns=[target_col])
    X_test_raw = test_raw.copy()

    test_ids = (
        X_test_raw["id"].values
        if "id" in X_test_raw.columns
        else np.arange(len(X_test_raw), dtype=int)
    )

    # 2. Compute stateless domain features once
    X_fe = engineer_domain_features(X)
    X_test_fe = engineer_domain_features(X_test_raw)

    n_splits = 10
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    oof_preds = np.zeros(len(y), dtype=np.float64)
    oof_lgb = np.zeros(len(y), dtype=np.float64)
    oof_xgb = np.zeros(len(y), dtype=np.float64)
    oof_cat = np.zeros(len(y), dtype=np.float64)

    test_preds_lgb = np.zeros(len(X_test_raw), dtype=np.float64)
    test_preds_xgb = np.zeros(len(X_test_raw), dtype=np.float64)
    test_preds_cat = np.zeros(len(X_test_raw), dtype=np.float64)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_fe, y)):
        X_tr = X_fe.iloc[train_idx].copy().reset_index(drop=True)
        y_tr = pd.Series(y[train_idx])
        X_va = X_fe.iloc[val_idx].copy().reset_index(drop=True)
        y_va = pd.Series(y[val_idx])
        X_te = X_test_fe.copy().reset_index(drop=True)

        # Contextual feature engineering & target encoding strictly fitted on fold train
        X_tr_proc, X_va_proc, X_te_proc = add_fold_contextual_features(
            X_tr, X_va, X_te, y_tr
        )

        drop_cols = [c for c in ["id", target_col] if c in X_tr_proc.columns]
        feature_cols = [c for c in X_tr_proc.columns if c not in drop_cols]

        X_tr_f = X_tr_proc[feature_cols].copy()
        X_va_f = X_va_proc[feature_cols].copy()
        X_te_f = X_te_proc[feature_cols].copy()

        # --- 1. LightGBM ---
        lgb_params = {
            "objective": "binary",
            "metric": "auc",
            "boosting_type": "gbdt",
            "learning_rate": 0.03,
            "num_leaves": 63,
            "max_depth": -1,
            "min_child_samples": 35,
            "subsample": 0.8,
            "subsample_freq": 1,
            "colsample_bytree": 0.75,
            "reg_alpha": 0.1,
            "reg_lambda": 2.0,
            "n_estimators": 3000,
            "random_state": 42 + fold,
            "n_jobs": -1,
            "verbose": -1,
        }
        lgb_model = lgb.LGBMClassifier(**lgb_params)
        lgb_callbacks = [
            lgb.early_stopping(stopping_rounds=60, verbose=False),
            lgb.log_evaluation(period=0),
        ]
        lgb_model.fit(
            X_tr_f,
            y_tr,
            eval_set=[(X_va_f, y_va)],
            eval_metric="auc",
            callbacks=lgb_callbacks,
        )
        val_p_lgb = lgb_model.predict_proba(X_va_f)[:, 1]
        oof_lgb[val_idx] = val_p_lgb
        test_preds_lgb += lgb_model.predict_proba(X_te_f)[:, 1] / n_splits

        # --- 2. XGBoost ---
        if HAS_XGB:
            xgb_model = xgb.XGBClassifier(
                objective="binary:logistic",
                eval_metric="auc",
                learning_rate=0.03,
                max_depth=6,
                min_child_weight=15,
                subsample=0.85,
                colsample_bytree=0.65,
                reg_alpha=0.2,
                reg_lambda=2.0,
                n_estimators=3000,
                tree_method="hist",
                random_state=42 + fold,
                n_jobs=-1,
                early_stopping_rounds=60,
            )
            xgb_model.fit(
                X_tr_f,
                y_tr,
                eval_set=[(X_va_f, y_va)],
                verbose=False,
            )
            val_p_xgb = xgb_model.predict_proba(X_va_f)[:, 1]
            oof_xgb[val_idx] = val_p_xgb
            test_preds_xgb += xgb_model.predict_proba(X_te_f)[:, 1] / n_splits
        else:
            val_p_xgb = val_p_lgb
            oof_xgb[val_idx] = val_p_lgb
            test_preds_xgb += lgb_model.predict_proba(X_te_f)[:, 1] / n_splits

        # --- 3. CatBoost ---
        if HAS_CAT:
            cat_model = CatBoostClassifier(
                loss_function="Logloss",
                eval_metric="AUC",
                learning_rate=0.04,
                depth=6,
                l2_leaf_reg=5.0,
                iterations=2000,
                random_seed=42 + fold,
                thread_count=-1,
                verbose=False,
                early_stopping_rounds=60,
            )
            # Fill NaN for CatBoost numeric compatibility
            X_tr_cat = X_tr_f.fillna(-999.0)
            X_va_cat = X_va_f.fillna(-999.0)
            X_te_cat = X_te_f.fillna(-999.0)

            cat_model.fit(
                X_tr_cat,
                y_tr,
                eval_set=(X_va_cat, y_va),
                verbose=False,
            )
            val_p_cat = cat_model.predict_proba(X_va_cat)[:, 1]
            oof_cat[val_idx] = val_p_cat
            test_preds_cat += cat_model.predict_proba(X_te_cat)[:, 1] / n_splits
        else:
            val_p_cat = val_p_lgb
            oof_cat[val_idx] = val_p_lgb
            test_preds_cat += lgb_model.predict_proba(X_te_f)[:, 1] / n_splits

    # 3. Out-Of-Fold Rank-Based Weight Optimization with Nelder-Mead
    models_oof_ranked = [
        rankdata(oof_lgb) / len(oof_lgb),
        rankdata(oof_xgb) / len(oof_xgb),
        rankdata(oof_cat) / len(oof_cat),
    ]

    def objective_fn(weights):
        w = np.array(weights)
        w_sum = np.sum(np.abs(w))
        if w_sum == 0:
            return 0.0
        w_norm = np.abs(w) / w_sum
        blend_oof = (
            w_norm[0] * models_oof_ranked[0]
            + w_norm[1] * models_oof_ranked[1]
            + w_norm[2] * models_oof_ranked[2]
        )
        return -roc_auc_score(y, blend_oof)

    init_w = [0.40, 0.35, 0.25]
    opt_res = minimize(
        objective_fn,
        init_w,
        method="Nelder-Mead",
        options={"maxiter": 500, "xatol": 1e-5, "fatol": 1e-6},
    )
    best_w = np.abs(opt_res.x)
    best_w = best_w / np.sum(best_w)

    oof_preds = (
        best_w[0] * models_oof_ranked[0]
        + best_w[1] * models_oof_ranked[1]
        + best_w[2] * models_oof_ranked[2]
    )
    val_auc = float(roc_auc_score(y, oof_preds))

    # 4. Final Blended Test Inference and Submission Export
    test_ranked_lgb = rankdata(test_preds_lgb) / len(test_preds_lgb)
    test_ranked_xgb = rankdata(test_preds_xgb) / len(test_preds_xgb)
    test_ranked_cat = rankdata(test_preds_cat) / len(test_preds_cat)

    final_test_preds = (
        best_w[0] * test_ranked_lgb
        + best_w[1] * test_ranked_xgb
        + best_w[2] * test_ranked_cat
    )
    final_test_preds = np.clip(final_test_preds, 0.0, 1.0).astype(np.float64)

    submission_df = pd.DataFrame(
        {
            "id": np.asarray(test_ids, dtype=int),
            "loan_status": final_test_preds,
        }
    )
    submission_df.to_csv("./submission/submission.csv", index=False)

    # Required final output metric
    print(f"Final Validation Score: {val_auc}")


if __name__ == "__main__":
    main()