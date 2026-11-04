import os
import warnings
import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder
import xgboost as xgb

warnings.filterwarnings("ignore")

os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# 1. Load Datasets
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Extract test identifiers
if "id" in test_df.columns:
    test_ids = test_df["id"].values
elif test_df.index.name == "id":
    test_ids = test_df.index.values
else:
    test_ids = np.arange(len(test_df))

# 2. Target Extraction & Standardization
target_col = "loan_status"
if target_col in train_df.columns:
    y_raw = train_df[target_col]
    if y_raw.dtype == "object" or str(y_raw.dtype) == "category":
        bad_terms = ["charged off", "default", "late", "does not meet"]
        y = (
            y_raw.astype(str)
            .apply(lambda val: 1 if any(b in val.lower() for b in bad_terms) else 0)
            .values
        )
    else:
        y = y_raw.astype(int).values
else:
    raise KeyError(f"Target column '{target_col}' not found in train.parquet")

train_raw = train_df.drop(columns=[target_col], errors="ignore").copy()
test_raw = test_df.copy()

n_train = len(train_raw)
n_test = len(test_raw)
combined = pd.concat([train_raw, test_raw], axis=0, ignore_index=True)


# 3. Text and Numerical Parsing
def clean_numeric_str(series):
    if series.dtype == "object" or str(series.dtype) == "category":
        return (
            series.astype(str)
            .str.replace("%", "")
            .str.replace("$", "")
            .str.strip()
            .apply(
                lambda x: (
                    float(x) if x not in ["nan", "None", "", "N/A", "n/a"] else np.nan
                )
            )
        )
    return series.astype(float)


def parse_term(series):
    if series.dtype == "object" or str(series.dtype) == "category":
        return series.astype(str).str.extract(r"(\d+)")[0].astype(float)
    return series.astype(float)


def parse_emp_length(series):
    if series.dtype == "object" or str(series.dtype) == "category":
        mapping = {
            "< 1 year": 0,
            "1 year": 1,
            "2 years": 2,
            "3 years": 3,
            "4 years": 4,
            "5 years": 5,
            "6 years": 6,
            "7 years": 7,
            "8 years": 8,
            "9 years": 9,
            "10+ years": 10,
        }
        res = series.map(mapping)
        unmapped = res.isna()
        if unmapped.any():
            extracted = series[unmapped].astype(str).str.extract(r"(\d+)")[0]
            res.update(extracted.astype(float))
        return res
    return series.astype(float)


if "term" in combined.columns:
    combined["term_months"] = parse_term(combined["term"])

if "int_rate" in combined.columns:
    combined["int_rate_clean"] = clean_numeric_str(combined["int_rate"])

if "dti" in combined.columns:
    combined["dti_clean"] = clean_numeric_str(combined["dti"])

if "revol_util" in combined.columns:
    combined["revol_util_clean"] = clean_numeric_str(combined["revol_util"])

if "emp_length" in combined.columns:
    combined["emp_length_years"] = parse_emp_length(combined["emp_length"])

# Ordinal Risk Encodings
grade_map = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7}
if "grade" in combined.columns:
    combined["grade_code"] = combined["grade"].map(grade_map).fillna(-1)

if "sub_grade" in combined.columns:
    sub_grades = [
        f"{g}{i}" for g in ["A", "B", "C", "D", "E", "F", "G"] for i in range(1, 6)
    ]
    sub_grade_map = {sg: idx + 1 for idx, sg in enumerate(sub_grades)}
    combined["sub_grade_code"] = combined["sub_grade"].map(sub_grade_map).fillna(-1)

# Datetime & Tenure Features
date_cols = [
    c
    for c in combined.columns
    if "date" in c.lower() or c.endswith("_d") or c == "earliest_cr_line"
]
for dcol in date_cols:
    dt_series = pd.to_datetime(combined[dcol], errors="coerce")
    if dt_series.notna().sum() > 0:
        combined[f"{dcol}_year"] = dt_series.dt.year
        combined[f"{dcol}_month"] = dt_series.dt.month

if "issue_d" in combined.columns and "earliest_cr_line" in combined.columns:
    issue_dt = pd.to_datetime(combined["issue_d"], errors="coerce")
    cr_dt = pd.to_datetime(combined["earliest_cr_line"], errors="coerce")
    combined["credit_hist_months"] = ((issue_dt - cr_dt).dt.days / 30.4375).fillna(-1)

# Domain Financial Ratios
annual_inc = (
    combined["annual_inc"].fillna(combined["annual_inc"].median())
    if "annual_inc" in combined.columns
    else pd.Series(1.0, index=combined.index)
)
loan_amnt = (
    combined["loan_amnt"].fillna(combined["loan_amnt"].median())
    if "loan_amnt" in combined.columns
    else pd.Series(0.0, index=combined.index)
)
installment = (
    combined["installment"].fillna(combined["installment"].median())
    if "installment" in combined.columns
    else pd.Series(0.0, index=combined.index)
)
term_m = (
    combined["term_months"].fillna(36)
    if "term_months" in combined.columns
    else pd.Series(36, index=combined.index)
)

combined["installment_to_inc"] = installment / (annual_inc / 12.0 + 1e-5)
combined["loan_to_inc"] = loan_amnt / (annual_inc + 1e-5)
combined["total_payment_est"] = installment * term_m
combined["expected_interest"] = combined["total_payment_est"] - loan_amnt
combined["interest_loan_ratio"] = combined["expected_interest"] / (loan_amnt + 1e-5)

if "revol_bal" in combined.columns:
    revol_bal = combined["revol_bal"].fillna(0)
    combined["revol_to_inc"] = revol_bal / (annual_inc + 1e-5)
    combined["revol_to_loan"] = revol_bal / (loan_amnt + 1e-5)

if "open_acc" in combined.columns and "total_acc" in combined.columns:
    combined["open_acc_ratio"] = combined["open_acc"] / (combined["total_acc"] + 1e-5)

if "tot_cur_bal" in combined.columns and "tot_hi_cred_lim" in combined.columns:
    combined["tot_bal_to_lim"] = combined["tot_cur_bal"] / (
        combined["tot_hi_cred_lim"] + 1e-5
    )

if "bc_util" in combined.columns and "revol_util_clean" in combined.columns:
    combined["bc_util_diff"] = combined["bc_util"] - combined["revol_util_clean"]

# Additional Compound Debt Burden Ratios & Interactions
if "dti_clean" in combined.columns and "int_rate_clean" in combined.columns:
    combined["dti_int_rate_mult"] = combined["dti_clean"] * combined["int_rate_clean"]
    combined["dti_to_int_rate"] = combined["dti_clean"] / (combined["int_rate_clean"] + 1e-5)

if "revol_bal" in combined.columns and "tot_hi_cred_lim" in combined.columns:
    revol_bal = combined["revol_bal"].fillna(0)
    tot_lim = combined["tot_hi_cred_lim"].fillna(combined["tot_hi_cred_lim"].median())
    combined["revol_to_hi_lim"] = revol_bal / (tot_lim + 1e-5)

# Missingness Indicators
mths_cols = [c for c in combined.columns if "mths_since" in c]
for mcol in mths_cols:
    combined[f"{mcol}_isnan"] = combined[mcol].isna().astype(int)
    combined[mcol] = combined[mcol].fillna(-999)

# Categorical Frequency Encoding
cat_cols = combined.select_dtypes(include=["object", "category"]).columns.tolist()
for col in cat_cols:
    freq = combined.iloc[:n_train][col].value_counts()
    combined[f"{col}_freq"] = combined[col].map(freq).fillna(0)

# Separate Features
X = combined.iloc[:n_train].copy().reset_index(drop=True)
X_test = combined.iloc[n_train:].copy().reset_index(drop=True)

if "id" in X.columns:
    X = X.drop(columns=["id"])
if "id" in X_test.columns:
    X_test = X_test.drop(columns=["id"])

# Save Engineered Datasets for reproducibility
X_save = X.copy()
X_save["target"] = y
X_save.to_parquet("./working/train_engineered.parquet")
X_test.to_parquet("./working/test_engineered.parquet")

# Identify remaining categorical columns
cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

# 4. Stratified Cross-Validation & Modeling
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_lgb = np.zeros(n_train)
oof_xgb = np.zeros(n_train)
oof_cat = np.zeros(n_train)

test_lgb = np.zeros(n_test)
test_xgb = np.zeros(n_test)
test_cat = np.zeros(n_test)

# Model Hyperparameters
lgb_params = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "learning_rate": 0.03,
    "num_leaves": 31,
    "max_depth": 6,
    "min_child_samples": 40,
    "subsample": 0.8,
    "colsample_bytree": 0.6,
    "reg_alpha": 0.5,
    "reg_lambda": 2.0,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}

xgb_params = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "learning_rate": 0.03,
    "max_depth": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.6,
    "min_child_weight": 8,
    "gamma": 0.1,
    "reg_alpha": 0.5,
    "reg_lambda": 2.0,
    "random_state": 42,
    "n_jobs": -1,
    "tree_method": "hist",
}

cat_params = {
    "loss_function": "Logloss",
    "eval_metric": "AUC",
    "learning_rate": 0.03,
    "depth": 6,
    "l2_leaf_reg": 5.0,
    "random_seed": 42,
    "verbose": 0,
    "thread_count": -1,
}

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_tr = X.iloc[train_idx].copy()
    y_tr = y[train_idx]
    X_va = X.iloc[val_idx].copy()
    y_va = y[val_idx]
    X_te = X_test.copy()

    # Create separate copies for CatBoost (raw string categoricals) and LightGBM/XGBoost (numeric encodings)
    X_tr_cat = X_tr.copy()
    X_va_cat = X_va.copy()
    X_te_cat = X_te.copy()

    for col in cat_cols:
        X_tr_cat[col] = X_tr_cat[col].fillna("missing").astype(str)
        X_va_cat[col] = X_va_cat[col].fillna("missing").astype(str)
        X_te_cat[col] = X_te_cat[col].fillna("missing").astype(str)

    # Fold-safe group relational aggregations (group mean, std, z-scores)
    group_cols = [c for c in ["sub_grade", "purpose", "addr_state"] if c in X_tr.columns]
    num_cols_grp = [c for c in ["dti_clean", "int_rate_clean", "annual_inc", "loan_amnt"] if c in X_tr.columns]

    X_tr_num = X_tr.copy()
    X_va_num = X_va.copy()
    X_te_num = X_te.copy()

    for g_col in group_cols:
        for n_col in num_cols_grp:
            grp_stats = X_tr_num.groupby(g_col, observed=False)[n_col].agg(["mean", "std"])
            grp_mean = grp_stats["mean"]
            grp_std = grp_stats["std"].replace(0, np.nan).fillna(1.0)

            feat_mean = f"{g_col}_{n_col}_mean"
            feat_z = f"{g_col}_{n_col}_zscore"

            for df_n, df_c in [(X_tr_num, X_tr_cat), (X_va_num, X_va_cat), (X_te_num, X_te_cat)]:
                m = df_n[g_col].map(grp_mean).fillna(X_tr_num[n_col].mean())
                s = df_n[g_col].map(grp_std).fillna(1.0)
                z = (df_n[n_col] - m) / (s + 1e-5)
                df_n[feat_mean] = m
                df_n[feat_z] = z
                df_c[feat_mean] = m
                df_c[feat_z] = z

    # Smoothed fold-safe target encoding and ordinal encoding for LightGBM/XGBoost
    global_mean = float(y_tr.mean())
    m_smooth = 10.0
    for col in cat_cols:
        stats = pd.Series(y_tr, index=X_tr_num.index).groupby(X_tr_num[col], observed=False).agg(["count", "mean"])
        smooth_te = (stats["count"] * stats["mean"] + m_smooth * global_mean) / (stats["count"] + m_smooth)

        X_tr_num[f"{col}_te"] = X_tr_num[col].map(smooth_te).fillna(global_mean).astype(np.float32)
        X_va_num[f"{col}_te"] = X_va_num[col].map(smooth_te).fillna(global_mean).astype(np.float32)
        X_te_num[f"{col}_te"] = X_te_num[col].map(smooth_te).fillna(global_mean).astype(np.float32)

        oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X_tr_num[col] = oe.fit_transform(X_tr_num[[col]].astype(str)).ravel().astype(np.float32)
        X_va_num[col] = oe.transform(X_va_num[[col]].astype(str)).ravel().astype(np.float32)
        X_te_num[col] = oe.transform(X_te_num[[col]].astype(str)).ravel().astype(np.float32)

    X_tr_num = X_tr_num.astype(np.float32)
    X_va_num = X_va_num.astype(np.float32)
    X_te_num = X_te_num.astype(np.float32)

    # Train LightGBM
    dtrain = lgb.Dataset(X_tr_num, label=y_tr)
    dval = lgb.Dataset(X_va_num, label=y_va, reference=dtrain)
    model_lgb = lgb.train(
        lgb_params,
        dtrain,
        num_boost_round=1200,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )
    oof_lgb[val_idx] = model_lgb.predict(X_va_num, num_iteration=model_lgb.best_iteration)
    test_lgb += (
        model_lgb.predict(X_te_num, num_iteration=model_lgb.best_iteration) / n_splits
    )

    # Train XGBoost
    model_xgb = xgb.XGBClassifier(
        **xgb_params, n_estimators=1200, early_stopping_rounds=50
    )
    model_xgb.fit(
        X_tr_num,
        y_tr,
        eval_set=[(X_va_num, y_va)],
        verbose=False,
    )
    oof_xgb[val_idx] = model_xgb.predict_proba(X_va_num)[:, 1]
    test_xgb += model_xgb.predict_proba(X_te_num)[:, 1] / n_splits

    # Train CatBoost with Native Categorical Features
    model_cat = cb.CatBoostClassifier(
        **cat_params, iterations=1200, cat_features=cat_cols
    )
    model_cat.fit(
        X_tr_cat,
        y_tr,
        eval_set=(X_va_cat, y_va),
        early_stopping_rounds=50,
        verbose=False,
    )
    oof_cat[val_idx] = model_cat.predict_proba(X_va_cat)[:, 1]
    test_cat += model_cat.predict_proba(X_te_cat)[:, 1] / n_splits

    fold_lgb_auc = roc_auc_score(y_va, oof_lgb[val_idx])
    fold_xgb_auc = roc_auc_score(y_va, oof_xgb[val_idx])
    fold_cat_auc = roc_auc_score(y_va, oof_cat[val_idx])
    print(
        f"Fold {fold + 1} - LightGBM AUC: {fold_lgb_auc:.5f} | XGBoost AUC: {fold_xgb_auc:.5f} | CatBoost AUC: {fold_cat_auc:.5f}"
    )


# 5. Optimize Ensemble Blending Weights
def loss_func(weights):
    w = np.array(weights)
    w = np.maximum(w, 0)
    if np.sum(w) > 0:
        w = w / np.sum(w)
    else:
        w = np.ones_like(w) / len(w)
    blend_oof = w[0] * oof_lgb + w[1] * oof_xgb + w[2] * oof_cat
    return -roc_auc_score(y, blend_oof)


res = minimize(
    loss_func,
    [0.34, 0.33, 0.33],
    method="Nelder-Mead",
    bounds=[(0, 1), (0, 1), (0, 1)],
)
weights = np.maximum(res.x, 0)
weights = weights / np.sum(weights)

oof_final = weights[0] * oof_lgb + weights[1] * oof_xgb + weights[2] * oof_cat
test_final = weights[0] * test_lgb + weights[1] * test_xgb + weights[2] * test_cat

final_auc = roc_auc_score(y, oof_final)

# 6. Save Predictions & Print Metric
sub_df = pd.DataFrame({"id": test_ids, "loan_status": test_final})
sub_df.to_csv("./submission/submission.csv", index=False)

print(f"Final Validation Score: {final_auc:.5f}")