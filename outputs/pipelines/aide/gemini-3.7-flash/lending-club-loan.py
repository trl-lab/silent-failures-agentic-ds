import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import warnings

warnings.filterwarnings("ignore")

# 1. Load Data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "loan_status"

# Ensure target is binary integer
if (
    train_df[target_col].dtype == object
    or train_df[target_col].dtype.name == "category"
):
    unique_vals = train_df[target_col].dropna().unique()
    if set(unique_vals).issubset({"Fully Paid", "Charged Off", "Default"}):
        target_mapping = {"Fully Paid": 0, "Charged Off": 1, "Default": 1}
        y = train_df[target_col].map(target_mapping).astype(int).values
    else:
        y = pd.factorize(train_df[target_col])[0]
else:
    y = train_df[target_col].astype(int).values

X = train_df.drop(columns=[target_col]).copy()
X_test = test_df.copy()

# 2. Preprocessing and Feature Engineering
combined = pd.concat([X, X_test], axis=0, ignore_index=True)

# Drop any ID-like columns if present to prevent leakage/overfitting
drop_cols = [col for col in ["id", "member_id", "url"] if col in combined.columns]
if drop_cols:
    combined.drop(columns=drop_cols, inplace=True)

# Process object and category columns robustly
for col in combined.columns:
    if combined[col].dtype == object or combined[col].dtype.name == "category":
        sample_non_null = combined[col].dropna().astype(str)
        if len(sample_non_null) > 0 and sample_non_null.str.endswith("%").mean() > 0.5:
            # Strip '%' and convert safely with coerce for NaNs
            combined[col] = pd.to_numeric(
                combined[col].astype(str).str.rstrip("%").str.strip(),
                errors="coerce",
            )
        elif (
            len(sample_non_null) > 0
            and sample_non_null.str.contains("months", case=False).mean() > 0.5
        ):
            # Handle terms like '36 months'
            combined[col] = pd.to_numeric(
                combined[col].astype(str).str.extract(r"(\d+)", expand=False),
                errors="coerce",
            )
        elif (
            len(sample_non_null) > 0
            and sample_non_null.str.contains("year", case=False).mean() > 0.5
        ):
            # Handle emp_length like '< 1 year', '10+ years'
            emp_map = {
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
            }
            cleaned_emp = combined[col].astype(str).str.strip().map(emp_map)
            combined[col] = cleaned_emp
        else:
            # Format as categorical for LightGBM
            combined[col] = combined[col].astype("category")

# Feature engineering: key financial ratios
if "loan_amnt" in combined.columns and "annual_inc" in combined.columns:
    combined["loan_to_income"] = combined["loan_amnt"] / (combined["annual_inc"] + 1.0)

if "installment" in combined.columns and "annual_inc" in combined.columns:
    combined["installment_to_income"] = (combined["installment"] * 12.0) / (
        combined["annual_inc"] + 1.0
    )

if "revol_bal" in combined.columns and "annual_inc" in combined.columns:
    combined["revol_bal_to_income"] = combined["revol_bal"] / (
        combined["annual_inc"] + 1.0
    )

if "total_acc" in combined.columns and "open_acc" in combined.columns:
    combined["open_to_total_acc"] = combined["open_acc"] / (combined["total_acc"] + 1.0)

n_train = len(X)
X_processed = combined.iloc[:n_train].reset_index(drop=True)
X_test_processed = combined.iloc[n_train:].reset_index(drop=True)

cat_features = [
    col for col in X_processed.columns if X_processed[col].dtype.name == "category"
]

# 3. Stratified 5-Fold Cross-Validation
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_preds = np.zeros(n_train)
test_preds = np.zeros(len(X_test_processed))

params = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "n_jobs": -1,
    "random_state": 42,
    "verbose": -1,
}

for fold, (train_idx, val_idx) in enumerate(skf.split(X_processed, y)):
    X_train_fold, y_train_fold = X_processed.iloc[train_idx], y[train_idx]
    X_val_fold, y_val_fold = X_processed.iloc[val_idx], y[val_idx]

    trn_data = lgb.Dataset(
        X_train_fold, label=y_train_fold, categorical_feature=cat_features
    )
    val_data = lgb.Dataset(
        X_val_fold,
        label=y_val_fold,
        reference=trn_data,
        categorical_feature=cat_features,
    )

    model = lgb.train(
        params,
        trn_data,
        num_boost_round=1500,
        valid_sets=[val_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )

    val_pred = model.predict(X_val_fold, num_iteration=model.best_iteration)
    oof_preds[val_idx] = val_pred
    test_preds += (
        model.predict(X_test_processed, num_iteration=model.best_iteration) / n_splits
    )

    fold_auc = roc_auc_score(y_val_fold, val_pred)
    print(f"Fold {fold + 1} ROC-AUC: {fold_auc:.5f}")

overall_auc = roc_auc_score(y, oof_preds)
print(f"Overall OOF ROC-AUC: {overall_auc:.5f}")

# 4. Save Submission
os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame({"id": np.arange(len(test_preds)), "loan_status": test_preds})
submission.to_csv("./working/submission.csv", index=False)
print("Submission successfully saved to ./working/submission.csv")
