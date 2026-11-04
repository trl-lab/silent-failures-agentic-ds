import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder, TargetEncoder
import lightgbm as lgb
import warnings

warnings.filterwarnings("ignore")

# 1. Load data
train_path = "./input/train.parquet"
test_path = "./input/test.parquet"

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = "marketing_status"

# 2. Encode target variable
if train_df[target_col].dtype == "object" or isinstance(
    train_df[target_col].iloc[0], (str, bool)
):
    le = LabelEncoder()
    y = le.fit_transform(train_df[target_col].astype(str))
else:
    unique_vals = np.sort(train_df[target_col].dropna().unique())
    if not np.array_equal(unique_vals, [0, 1]) and len(unique_vals) == 2:
        val_map = {unique_vals[0]: 0, unique_vals[1]: 1}
        y = train_df[target_col].map(val_map).values
    else:
        y = train_df[target_col].astype(int).values

X = train_df.drop(columns=[target_col]).copy()
X_test = test_df.copy()
if target_col in X_test.columns:
    X_test = X_test.drop(columns=[target_col])

n_train = len(X)
combined = pd.concat([X, X_test], axis=0).reset_index(drop=True)

# 3. Feature Engineering & Preprocessing
cat_cols = []
date_cols_processed = []

for col in list(combined.columns):
    # Try parsing date columns
    if combined[col].dtype == "object":
        sample = combined[col].dropna().head(50)
        if len(sample) > 0:
            try:
                pd.to_datetime(sample, errors="raise", format="mixed")
                full_dates = pd.to_datetime(combined[col], errors="coerce")
                if full_dates.notna().sum() > len(combined) * 0.3:
                    min_date = full_dates.min()
                    combined[f"{col}_elapsed_days"] = (full_dates - min_date).dt.days
                    combined[f"{col}_year"] = full_dates.dt.year
                    combined[f"{col}_month"] = full_dates.dt.month
                    combined[f"{col}_dayofweek"] = full_dates.dt.dayofweek
                    combined.drop(columns=[col], inplace=True)
                    date_cols_processed.append(col)
                    continue
            except Exception:
                pass

    # Process categorical / string columns
    if combined[col].dtype == "object" or pd.api.types.is_categorical_dtype(
        combined[col]
    ):
        combined[f"{col}_str_len"] = combined[col].astype(str).str.len().fillna(0)
        combined[f"{col}_freq"] = combined[col].map(
            combined[col].value_counts(dropna=False)
        )
        combined[col] = combined[col].fillna("MISSING").astype(str)
        cat_cols.append(col)

# Convert categorical columns to category dtype for LightGBM
for col in cat_cols:
    combined[col] = combined[col].astype("category")

# Separate train and test sets
X_proc = combined.iloc[:n_train].reset_index(drop=True)
X_test_proc = combined.iloc[n_train:].reset_index(drop=True)

# 4. Stratified 5-Fold Cross-Validation Setup with Target Encoding & Multi-Seed LGBM
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
seeds = [42, 123, 2024]

oof_preds = np.zeros(n_train)
test_preds = np.zeros(len(X_test_proc))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_proc, y)):
    X_train_fold = X_proc.iloc[train_idx].copy()
    y_train_fold = y[train_idx]
    X_val_fold = X_proc.iloc[val_idx].copy()
    y_val_fold = y[val_idx]
    X_test_fold = X_test_proc.copy()

    # Out-of-fold Target Encoding on categorical columns
    if len(cat_cols) > 0:
        te = TargetEncoder(smooth="auto", cv=5, random_state=42)
        train_te = te.fit_transform(X_train_fold[cat_cols].astype(str), y_train_fold)
        val_te = te.transform(X_val_fold[cat_cols].astype(str))
        test_te = te.transform(X_test_fold[cat_cols].astype(str))

        for i, c in enumerate(cat_cols):
            X_train_fold[f"{c}_te"] = train_te[:, i]
            X_val_fold[f"{c}_te"] = val_te[:, i]
            X_test_fold[f"{c}_te"] = test_te[:, i]

    fold_val_preds = np.zeros(len(val_idx))
    fold_test_preds = np.zeros(len(X_test_proc))

    for seed in seeds:
        lgb_params = {
            "objective": "binary",
            "metric": "auc",
            "boosting_type": "gbdt",
            "learning_rate": 0.03,
            "num_leaves": 31,
            "max_depth": -1,
            "min_child_samples": 20,
            "subsample": 0.75,
            "colsample_bytree": 0.75,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "random_state": seed,
            "n_estimators": 2000,
            "n_jobs": -1,
            "verbose": -1,
        }

        model = lgb.LGBMClassifier(**lgb_params)
        callbacks = [
            lgb.early_stopping(stopping_rounds=60, verbose=False),
            lgb.log_evaluation(period=0),
        ]

        model.fit(
            X_train_fold,
            y_train_fold,
            eval_set=[(X_val_fold, y_val_fold)],
            callbacks=callbacks,
        )

        val_p = model.predict_proba(X_val_fold)[:, 1]
        test_p = model.predict_proba(X_test_fold)[:, 1]

        fold_val_preds += val_p / len(seeds)
        fold_test_preds += test_p / len(seeds)

    oof_preds[val_idx] = fold_val_preds
    test_preds += fold_test_preds / n_splits

    fold_auc = roc_auc_score(y_val_fold, fold_val_preds)
    print(f"Fold {fold + 1} ROC-AUC: {fold_auc:.5f}")

overall_auc = roc_auc_score(y, oof_preds)
print(f"\nOverall Out-of-Fold ROC-AUC: {overall_auc:.5f}")

# 5. Save Predictions to working directory
os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame(
    {"id": np.arange(len(test_preds)), "marketing_status": test_preds}
)

submission.to_csv("./working/submission.csv", index=False)
print("Submission saved successfully to ./working/submission.csv")
