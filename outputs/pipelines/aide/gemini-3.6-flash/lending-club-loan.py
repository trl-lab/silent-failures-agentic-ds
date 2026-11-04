import numpy as np
import pandas as pd
import re
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import lightgbm as lgb
import warnings

warnings.filterwarnings("ignore")

# 1. Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# 2. Extract target
target_col = "loan_status"
if target_col in train_df.columns:
    y_raw = train_df[target_col]
    if y_raw.dtype == "object" or str(y_raw.dtype) == "category":
        y = y_raw.apply(
            lambda x: (
                1
                if any(
                    b in str(x).lower()
                    for b in ["charged off", "default", "late", "does not meet"]
                )
                else 0
            )
        ).values
    else:
        y = y_raw.astype(int).values
    train_df = train_df.drop(columns=[target_col])
else:
    raise ValueError("Target column loan_status not found in train data.")

n_train = len(train_df)
combined_df = pd.concat([train_df, test_df], ignore_index=True)


# 3. Clean numeric and string features
def parse_percentage(val):
    if pd.isna(val):
        return np.nan
    val_str = str(val).replace("%", "").strip()
    try:
        return float(val_str)
    except:
        return np.nan


def parse_term(val):
    if pd.isna(val):
        return np.nan
    match = re.search(r"(\d+)", str(val))
    return float(match.group(1)) if match else np.nan


def parse_emp_length(val):
    if pd.isna(val):
        return np.nan
    val_str = str(val)
    if "<" in val_str:
        return 0.5
    match = re.search(r"(\d+)", val_str)
    return float(match.group(1)) if match else np.nan


if "int_rate" in combined_df.columns and combined_df["int_rate"].dtype == "object":
    combined_df["int_rate"] = combined_df["int_rate"].apply(parse_percentage)

if "revol_util" in combined_df.columns and combined_df["revol_util"].dtype == "object":
    combined_df["revol_util"] = combined_df["revol_util"].apply(parse_percentage)

if "term" in combined_df.columns:
    combined_df["term_num"] = combined_df["term"].apply(parse_term)
else:
    combined_df["term_num"] = np.nan

if "emp_length" in combined_df.columns:
    combined_df["emp_length_num"] = combined_df["emp_length"].apply(parse_emp_length)

# Parse date columns and compute temporal deltas
date_cols = [
    "issue_d",
    "earliest_cr_line",
    "last_pymnt_d",
    "next_pymnt_d",
    "last_credit_pull_d",
]
for date_col in date_cols:
    if date_col in combined_df.columns:
        dt_series = pd.to_datetime(combined_df[date_col], errors="coerce")
        combined_df[f"{date_col}_year"] = dt_series.dt.year
        combined_df[f"{date_col}_month"] = dt_series.dt.month

if (
    "issue_d_year" in combined_df.columns
    and "earliest_cr_line_year" in combined_df.columns
):
    combined_df["credit_hist_years"] = (
        combined_df["issue_d_year"] - combined_df["earliest_cr_line_year"]
    ) + (
        combined_df["issue_d_month"].fillna(1)
        - combined_df["earliest_cr_line_month"].fillna(1)
    ) / 12.0

if "issue_d" in combined_df.columns and "last_pymnt_d" in combined_df.columns:
    issue_dt = pd.to_datetime(combined_df["issue_d"], errors="coerce")
    last_pymnt_dt = pd.to_datetime(combined_df["last_pymnt_d"], errors="coerce")
    combined_df["pymnt_elapsed_months"] = (
        last_pymnt_dt.dt.year - issue_dt.dt.year
    ) * 12 + (last_pymnt_dt.dt.month - issue_dt.dt.month)

if "issue_d" in combined_df.columns and "last_credit_pull_d" in combined_df.columns:
    issue_dt = pd.to_datetime(combined_df["issue_d"], errors="coerce")
    pull_dt = pd.to_datetime(combined_df["last_credit_pull_d"], errors="coerce")
    combined_df["credit_pull_elapsed_months"] = (
        pull_dt.dt.year - issue_dt.dt.year
    ) * 12 + (pull_dt.dt.month - issue_dt.dt.month)

# 4. Domain Ratio & Payment Progress Features
ann_inc = (
    combined_df["annual_inc"].fillna(0)
    if "annual_inc" in combined_df.columns
    else pd.Series(0, index=combined_df.index)
)
loan_amt = (
    combined_df["loan_amnt"].fillna(0)
    if "loan_amnt" in combined_df.columns
    else pd.Series(0, index=combined_df.index)
)
installment = (
    combined_df["installment"].fillna(0)
    if "installment" in combined_df.columns
    else pd.Series(0, index=combined_df.index)
)
revol_bal = (
    combined_df["revol_bal"].fillna(0)
    if "revol_bal" in combined_df.columns
    else pd.Series(0, index=combined_df.index)
)
open_acc = (
    combined_df["open_acc"].fillna(0)
    if "open_acc" in combined_df.columns
    else pd.Series(0, index=combined_df.index)
)
total_acc = (
    combined_df["total_acc"].fillna(0)
    if "total_acc" in combined_df.columns
    else pd.Series(0, index=combined_df.index)
)

combined_df["loan_to_inc"] = loan_amt / (ann_inc + 1.0)
combined_df["installment_to_inc"] = installment / ((ann_inc / 12.0) + 1.0)
combined_df["revol_bal_to_inc"] = revol_bal / (ann_inc + 1.0)
combined_df["revol_bal_per_acc"] = revol_bal / (open_acc + 1.0)
combined_df["open_acc_ratio"] = open_acc / (total_acc + 1.0)

if "term_num" in combined_df.columns:
    tot_pymnt_est = installment * combined_df["term_num"].fillna(36)
    combined_df["est_interest_ratio"] = (tot_pymnt_est - loan_amt) / (loan_amt + 1.0)

# Payment progress & credit risk interactions
if "total_pymnt" in combined_df.columns:
    combined_df["pymnt_progress"] = combined_df["total_pymnt"].fillna(0) / (
        loan_amt + 1.0
    )
if "total_rec_prncp" in combined_df.columns:
    combined_df["prncp_progress"] = combined_df["total_rec_prncp"].fillna(0) / (
        loan_amt + 1.0
    )
if "last_pymnt_amnt" in combined_df.columns:
    combined_df["last_pymnt_ratio"] = combined_df["last_pymnt_amnt"].fillna(0) / (
        installment + 1.0
    )
if "inq_last_6mths" in combined_df.columns:
    combined_df["inq_to_open_acc"] = combined_df["inq_last_6mths"].fillna(0) / (
        open_acc + 1.0
    )
if "delinq_2yrs" in combined_df.columns:
    combined_df["delinq_to_open_acc"] = combined_df["delinq_2yrs"].fillna(0) / (
        open_acc + 1.0
    )

# Outstanding debt and recovery ratios
if "out_prncp" in combined_df.columns:
    combined_df["out_prncp_ratio"] = combined_df["out_prncp"].fillna(0) / (
        loan_amt + 1.0
    )
if "total_rec_int" in combined_df.columns:
    combined_df["rec_int_ratio"] = combined_df["total_rec_int"].fillna(0) / (
        loan_amt + 1.0
    )
if "recoveries" in combined_df.columns:
    combined_df["recoveries_ratio"] = combined_df["recoveries"].fillna(0) / (
        loan_amt + 1.0
    )
if "total_rec_late_fee" in combined_df.columns:
    combined_df["late_fee_ratio"] = combined_df["total_rec_late_fee"].fillna(0) / (
        installment + 1.0
    )

# Ordinal grade and sub_grade mapping
grades = ["A", "B", "C", "D", "E", "F", "G"]
grade_map = {g: i for i, g in enumerate(grades)}
if "grade" in combined_df.columns:
    combined_df["grade_code"] = combined_df["grade"].map(grade_map)

sub_grades = [f"{g}{i}" for g in grades for i in range(1, 6)]
sub_grade_map = {sg: idx for idx, sg in enumerate(sub_grades)}
if "sub_grade" in combined_df.columns:
    combined_df["sub_grade_code"] = combined_df["sub_grade"].map(sub_grade_map)

# 5. Process categorical variables
cat_cols = combined_df.select_dtypes(include=["object", "category"]).columns.tolist()
for col in cat_cols:
    freq = combined_df[col].value_counts().to_dict()
    combined_df[f"{col}_freq"] = combined_df[col].map(freq)
    combined_df[col] = pd.factorize(combined_df[col].astype(str))[0]

# Split back to train and test sets
X = combined_df.iloc[:n_train].copy()
X_test = combined_df.iloc[n_train:].copy()

X = X.select_dtypes(include=[np.number])
X_test = X_test[X.columns]

id_cols = [c for c in X.columns if c.lower() in ["id", "member_id", "index"]]
X = X.drop(columns=id_cols)
X_test = X_test.drop(columns=id_cols)

# 6. Stratified 5-Fold Cross-Validation with XGBoost + LightGBM Blending
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train, y_train = X.iloc[train_idx], y[train_idx]
    X_val, y_val = X.iloc[val_idx], y[val_idx]

    # Model 1: XGBoost (depth-wise)
    xgb_model = xgb.XGBClassifier(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        random_state=42 + fold,
        eval_metric="auc",
        early_stopping_rounds=50,
        n_jobs=-1,
    )
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    val_pred_xgb = xgb_model.predict_proba(X_val)[:, 1]
    test_pred_xgb = xgb_model.predict_proba(X_test)[:, 1]

    # Model 2: LightGBM (leaf-wise)
    lgb_model = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        n_jobs=-1,
        verbosity=-1,
    )
    lgb_model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )

    val_pred_lgb = lgb_model.predict_proba(X_val)[:, 1]
    test_pred_lgb = lgb_model.predict_proba(X_test)[:, 1]

    # Blend predictions
    val_pred_blend = 0.5 * val_pred_xgb + 0.5 * val_pred_lgb
    oof_preds[val_idx] = val_pred_blend
    test_preds += (0.5 * test_pred_xgb + 0.5 * test_pred_lgb) / skf.n_splits

oof_auc = roc_auc_score(y, oof_preds)
print(f"Validation Metric (ROC AUC): {oof_auc:.5f}")

# 7. Output submission file
submission = pd.DataFrame({"id": np.arange(len(test_preds)), "loan_status": test_preds})
submission.to_csv("./working/submission.csv", index=False)
