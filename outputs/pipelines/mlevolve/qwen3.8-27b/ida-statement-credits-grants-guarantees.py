import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
import lightgbm as lgb
import xgboost as xgb
import json
import os

# Create working directory
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# ============================================================
# 1. LOAD DATA
# ============================================================
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")
print(f"Columns: {list(train.columns)}")
print(f"\nTarget stats:\n{train['Original Principal Amount (US$)'].describe()}")

# ============================================================
# 2. IDENTIFY COLUMN TYPES
# ============================================================
target_col = "Original Principal Amount (US$)"
id_col = "id" if "id" in train.columns else None

# Separate target
train_target = train[target_col].copy()
train = train.drop(columns=[target_col])

# Identify feature columns (exclude id if present)
if id_col:
    test_ids = test[id_col].copy()
    train = train.drop(columns=[id_col])
    test = test.drop(columns=[id_col])

feature_cols = train.columns.tolist()

# Classify columns
numeric_cols = train.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = train.select_dtypes(include=["object", "category"]).columns.tolist()

# Also check for low-cardinality numeric columns that might be categorical
for col in numeric_cols[:]:
    if train[col].nunique() < 20:
        categorical_cols.append(col)
        numeric_cols.remove(col)

print(f"\nNumeric features: {len(numeric_cols)}")
print(f"Categorical features: {len(categorical_cols)}")

# ============================================================
# 3. CREATE TRAIN/VALIDATION SPLIT (BEFORE any stateful transforms)
# ============================================================
train_indices, val_indices = train_test_split(
    range(len(train)), test_size=0.15, random_state=42
)

train_part = train.iloc[train_indices].copy()
val_part = train.iloc[val_indices].copy()
train_target_part = train_target.iloc[train_indices].copy()
val_target_part = train_target.iloc[val_indices].copy()

print(f"\nTrain part: {len(train_part)}, Val part: {len(val_part)}")

# ============================================================
# 4. FEATURE ENGINEERING - Date-based features for LightGBM
# ============================================================

# Define date columns to parse
date_cols = [
    'End of Period',
    'First Repayment Date',
    'Last Repayment Date',
    'Agreement Signing Date',
    'Board Approval Date',
    'Effective Date (Most Recent)',
    'Closed Date (Most Recent)',
    'Last Disbursement Date'
]

# Categorical columns to keep as-is for LightGBM native handling
cat_feature_cols = [
    'Credit Number',
    'Region',
    'Country / Economy Code',
    'Country / Economy',
    'Borrower',
    'Credit Status',
    'Currency of Commitment',
    'Project ID',
    'Project Name'
]


def engineer_date_features(df):
    """Extract date-based features from date columns."""
    df = df.copy()

    # Parse all date columns
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    # Extract year, month, day, day-of-week from key dates
    for col in date_cols:
        if col in df.columns:
            safe_name = col.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_')
            df[f'{safe_name}_year'] = df[col].dt.year
            df[f'{safe_name}_month'] = df[col].dt.month
            df[f'{safe_name}_day'] = df[col].dt.day
            df[f'{safe_name}_dayofweek'] = df[col].dt.dayofweek

    # Time differences between key date pairs (in days)
    date_pairs = [
        ('Last Repayment Date', 'First Repayment Date', 'loan_term_days'),
        ('Agreement Signing Date', 'Board Approval Date', 'approval_to_signing_days'),
        ('First Repayment Date', 'Agreement Signing Date', 'signing_to_first_repayment_days'),
        ('Last Disbursement Date', 'Agreement Signing Date', 'signing_to_last_disbursement_days'),
        ('Closed Date (Most Recent)', 'First Repayment Date', 'first_repayment_to_close_days'),
        ('Effective Date (Most Recent)', 'Board Approval Date', 'approval_to_effective_days'),
        ('Last Repayment Date', 'Agreement Signing Date', 'signing_to_last_repayment_days'),
        ('End of Period', 'Agreement Signing Date', 'signing_to_eop_days'),
    ]

    for col1, col2, feature_name in date_pairs:
        if col1 in df.columns and col2 in df.columns:
            df[feature_name] = (df[col1] - df[col2]).dt.days

    # End of Period year as numeric feature
    if 'End of Period' in df.columns:
        df['eop_year'] = df['End of Period'].dt.year

    # Drop original date columns (we've extracted features from them)
    for col in date_cols:
        if col in df.columns:
            df = df.drop(columns=[col])

    return df


# Apply date feature engineering
train_part = engineer_date_features(train_part)
val_part = engineer_date_features(val_part)
test_processed = engineer_date_features(test)

# Fill NaN in numeric features with median (from train)
numeric_feature_cols = [c for c in train_part.columns if c not in cat_feature_cols]
for col in numeric_feature_cols:
    median_val = train_part[col].median()
    train_part[col] = train_part[col].fillna(median_val)
    val_part[col] = val_part[col].fillna(median_val)
    test_processed[col] = test_processed[col].fillna(median_val)

# Fill NaN in categorical features with 'Unknown'
for col in cat_feature_cols:
    train_part[col] = train_part[col].fillna('Unknown').astype(str)
    val_part[col] = val_part[col].fillna('Unknown').astype(str)
    test_processed[col] = test_processed[col].fillna('Unknown').astype(str)

# Define final feature columns
final_feature_cols = numeric_feature_cols + cat_feature_cols

# Identify categorical feature indices for LightGBM
cat_feature_indices = [final_feature_cols.index(c) for c in cat_feature_cols if c in final_feature_cols]

print(f"\nFinal feature count: {len(final_feature_cols)}")
print(f"Categorical features: {len(cat_feature_indices)}")
print(f"Numeric features: {len(numeric_feature_cols)}")
print(f"Train features shape: {len(train_part)}")
print(f"Val features shape: {len(val_part)}")
print(f"Test features shape: {len(test_processed)}")

# ============================================================
# 5. TARGET - Use original scale (skew is only 0.94, no log needed)
# ============================================================
target_skew = train_target_part.skew()
print(f"\nTarget skewness: {target_skew:.2f}")
print("Using raw target (no log transform needed)")

train_y = train_target_part.values.astype(np.float64)
val_y = val_target_part.values.astype(np.float64)

# ============================================================
# 6. PREPARE DATA FOR LIGHTGBM/XGBOOST
# ============================================================
# For LightGBM, categorical features should be passed as strings
# and indicated via categorical_feature parameter

train_X = train_part[final_feature_cols].copy()
val_X = val_part[final_feature_cols].copy()
test_X = test_processed[final_feature_cols].copy()

# Encode categorical columns as integers for LightGBM
cat_mappings = {}
for col in cat_feature_cols:
    cats = sorted(train_X[col].astype(str).unique().tolist())
    cat_mappings[col] = {c: i for i, c in enumerate(cats)}
    train_X[col] = train_X[col].astype(str).map(cat_mappings[col]).fillna(-1).astype(int)
    val_X[col] = val_X[col].astype(str).map(cat_mappings[col]).fillna(-1).astype(int)
    test_X[col] = test_X[col].astype(str).map(cat_mappings[col]).fillna(-1).astype(int)

# Replace inf values in numeric columns
for col in numeric_feature_cols:
    train_X[col] = train_X[col].replace([np.inf, -np.inf], 0).fillna(0)
    val_X[col] = val_X[col].replace([np.inf, -np.inf], 0).fillna(0)
    test_X[col] = test_X[col].replace([np.inf, -np.inf], 0).fillna(0)

print(f"\nData ready for training:")
print(f"  - Train: {train_X.shape[0]} samples, {train_X.shape[1]} features")
print(f"  - Val: {val_X.shape[0]} samples")
print(f"  - Test: {test_X.shape[0]} samples")
print(f"  - Categorical feature indices: {cat_feature_indices}")

# ============================================================
# 8. TRAIN LIGHTGBM MODEL
# ============================================================
print("\n" + "=" * 60)
print("Training LightGBM Model")
print("=" * 60)

lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'num_leaves': 127,
    'learning_rate': 0.03,
    'feature_fraction': 0.8,
    'feature_fraction_bynode': 0.7,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'min_child_samples': 20,
    'min_data_in_leaf': 20,
    'verbose': -1,
    'n_estimators': 5000,
    'random_state': 42,
}

# ============================================================
# 8. 5-FOLD CROSS-VALIDATION WITH LIGHTGBM
# ============================================================
from sklearn.model_selection import KFold

n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

cv_scores = []
cv_test_preds = np.zeros(len(test_X))
all_data_X = pd.concat([train_X, val_X], axis=0).reset_index(drop=True)
all_data_y = np.concatenate([train_y, val_y])

for fold, (train_idx, val_idx) in enumerate(kf.split(all_data_X)):
    print(f"\n--- Fold {fold+1}/{n_splits} ---")
    fold_train_X = all_data_X.iloc[train_idx]
    fold_val_X = all_data_X.iloc[val_idx]
    fold_train_y = all_data_y[train_idx]
    fold_val_y = all_data_y[val_idx]

    fold_model = lgb.LGBMRegressor(**lgb_params)
    fold_model.fit(
        fold_train_X, fold_train_y,
        eval_set=[(fold_val_X, fold_val_y)],
        categorical_feature=cat_feature_indices,
        callbacks=[
            lgb.early_stopping(50),
            lgb.log_evaluation(200),
        ],
    )

    fold_val_preds = fold_model.predict(fold_val_X, num_iteration=fold_model.best_iteration_)
    fold_rmse = root_mean_squared_error(fold_val_y, fold_val_preds)
    cv_scores.append(fold_rmse)
    print(f"Fold {fold+1} RMSE: {fold_rmse:.4f} (best iter: {fold_model.best_iteration_})")

    cv_test_preds += fold_model.predict(test_X, num_iteration=fold_model.best_iteration_) / n_splits

print(f"\nCV Mean RMSE: {np.mean(cv_scores):.4f} +/- {np.std(cv_scores):.4f}")

# ============================================================
# 9. TRAIN FINAL MODEL ON ALL DATA
# ============================================================
print("\n" + "=" * 60)
print("Training Final Model on All Data")
print("=" * 60)

# Store best iterations from all CV folds
cv_best_iterations = []
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(kf.split(all_data_X)):
    print(f"\n--- Fold {fold+1}/{n_splits} ---")
    fold_train_X = all_data_X.iloc[train_idx]
    fold_val_X = all_data_X.iloc[val_idx]
    fold_train_y = all_data_y[train_idx]
    fold_val_y = all_data_y[val_idx]

    fold_model = lgb.LGBMRegressor(**lgb_params)
    fold_model.fit(
        fold_train_X, fold_train_y,
        eval_set=[(fold_val_X, fold_val_y)],
        categorical_feature=cat_feature_indices,
        callbacks=[
            lgb.early_stopping(50),
            lgb.log_evaluation(200),
        ],
    )
    cv_best_iterations.append(fold_model.best_iteration_)

mean_best_iter = int(np.mean(cv_best_iterations))
print(f"Mean best iteration across folds: {mean_best_iter}")

final_n_estimators = mean_best_iter

final_model = lgb.LGBMRegressor(**{**lgb_params, 'n_estimators': final_n_estimators})
final_model.fit(
    all_data_X, all_data_y,
    categorical_feature=cat_feature_indices,
    callbacks=[lgb.log_evaluation(500)],
)

# ============================================================
# 10. GENERATE FINAL TEST PREDICTIONS
# ============================================================
print("\n" + "=" * 60)
print("Generating Final Test Predictions")
print("=" * 60)

# Use CV-averaged test predictions (standard practice for best generalization)
test_preds_original = cv_test_preds

# Clip negative predictions (loan amounts can't be negative)
test_preds_original = np.clip(test_preds_original, 0, None)

print(f"Test prediction stats: mean={test_preds_original.mean():.2f}, min={test_preds_original.min():.2f}, max={test_preds_original.max():.2f}")

# ============================================================
# 11. SAVE SUBMISSION
# ============================================================
os.makedirs("./submission", exist_ok=True)

if id_col:
    test_ids = test_ids.values
else:
    test_ids = np.arange(len(test_preds_original))

submission = pd.DataFrame(
    {"id": test_ids.astype(int), "Original Principal Amount (US$)": test_preds_original}
)
submission.to_csv("./submission/submission.csv", index=False)
print(f"\nSubmission saved: {submission.shape[0]} rows")

# ============================================================
# 12. FINAL VALIDATION SCORE
# ============================================================
score = np.mean(cv_scores)
print(f"Final Validation Score (CV Mean RMSE): {score:.4f}")