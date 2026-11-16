import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
import os
import warnings

warnings.filterwarnings("ignore")

# Create directories
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# ============ LOAD DATA ============
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")
print(f"Train columns: {list(train.columns)}")

# Identify target
target_col = "Size of Purchase (MW)"
y_full = train[target_col].values
train = train.drop(columns=[target_col])

print(
    f"Target stats: mean={y_full.mean():.4f}, std={y_full.std():.4f}, skew={pd.Series(y_full).skew():.4f}"
)

# ============ SPLIT TRAIN INTO TRAIN/VAL FIRST ============
indices = np.arange(len(train))
train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)

X_train_raw = train.iloc[train_idx].reset_index(drop=True)
X_val_raw = train.iloc[val_idx].reset_index(drop=True)
y_train = y_full[train_idx]
y_val = y_full[val_idx]

n_train = len(X_train_raw)
n_val = len(X_val_raw)
n_test = len(test)

print(
    f"Train split: {X_train_raw.shape}, Val split: {X_val_raw.shape}, Test: {test.shape}"
)

# ============ FEATURE ENGINEERING ============
# Combine all data for consistent feature engineering
X_all = pd.concat([X_train_raw, X_val_raw, test], axis=0, ignore_index=True)

# 1. Parse 'Announcement Date' as datetime and extract date features
if "Announcement Date" in X_all.columns:
    X_all["Announcement Date"] = pd.to_datetime(X_all["Announcement Date"], errors="coerce")
    X_all["Announcement_Date_year"] = X_all["Announcement Date"].dt.year
    X_all["Announcement_Date_month"] = X_all["Announcement Date"].dt.month
    X_all["Announcement_Date_quarter"] = X_all["Announcement Date"].dt.quarter
    X_all.drop(columns=["Announcement Date"], inplace=True)

# 2. Convert columns that should be numeric
numeric_conversion_cols = [
    "Operation Start Year",
    "Contract Length (Year)",
    "Estimated Annual Generation (MWh)",
]
for c in numeric_conversion_cols:
    if c in X_all.columns:
        X_all[c] = pd.to_numeric(X_all[c], errors="coerce")

# Re-identify columns after date processing and numeric conversion
cat_cols = X_all.select_dtypes(include=["object", "category"]).columns.tolist()
num_cols = X_all.select_dtypes(include=["number"]).columns.tolist()

print(f"After date processing - Cat: {len(cat_cols)}, Num: {len(num_cols)}")
print(f"Numeric cols: {num_cols}")
print(f"Categorical cols: {cat_cols}")

# 3. Handle missing values in numerical columns (fit on train only)
num_median = X_all.iloc[:n_train][num_cols].median()
for c in num_cols:
    X_all[c] = X_all[c].fillna(num_median[c])

# 4. Handle missing values in categorical columns
for c in cat_cols:
    X_all[c] = X_all[c].fillna("__MISSING__")
    X_all[c] = X_all[c].astype(str)

# 5. Identify high vs low cardinality
high_card_cols = []
low_card_cols = []
for c in cat_cols:
    n_unique = X_all.iloc[:n_train][c].nunique()
    if n_unique > 10:
        high_card_cols.append(c)
    else:
        low_card_cols.append(c)

print(f"High cardinality cats: {high_card_cols}")
print(f"Low cardinality cats: {low_card_cols}")

# 6. Frequency encoding for high-cardinality categoricals (fit on train only)
for c in high_card_cols:
    train_freq = X_all.iloc[:n_train][c].value_counts(normalize=True)
    X_all[f"{c}_freq"] = X_all[c].map(train_freq).fillna(0)
    X_all.drop(columns=[c], inplace=True)

# 7. One-hot encode low-cardinality categoricals (categories from train only)
ohe_cols = []
for c in low_card_cols:
    train_cats = X_all.iloc[:n_train][c].unique()
    for cat in train_cats:
        col_name = f"{c}_{cat}"
        X_all[col_name] = (X_all[c] == cat).astype(int)
        ohe_cols.append(col_name)
    X_all.drop(columns=[c], inplace=True)

# 8. Scale numerical features (fit on train only)
final_num_cols = X_all.select_dtypes(include=["number"]).columns.tolist()
scaler = StandardScaler()
scaler.fit(X_all.iloc[:n_train][final_num_cols])
X_all[final_num_cols] = scaler.transform(X_all[final_num_cols])

# ============ SPLIT BACK ============
X_train_final = X_all.iloc[:n_train].reset_index(drop=True)
X_val_final = X_all.iloc[n_train : n_train + n_val].reset_index(drop=True)
X_test_final = X_all.iloc[n_train + n_val :].reset_index(drop=True)

# Fill any remaining NaNs
X_train_final = X_train_final.fillna(0)
X_val_final = X_val_final.fillna(0)
X_test_final = X_test_final.fillna(0)

print(
    f"Final shapes - Train: {X_train_final.shape}, Val: {X_val_final.shape}, Test: {X_test_final.shape}"
)
print(f"Features: {len(X_train_final.columns)}")

# ============ MODEL PARAMETERS ============
lgb_params = {
    "objective": "huber",
    "alpha": 0.9,
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.005,
    "num_leaves": 15,
    "max_depth": -1,
    "min_data_in_leaf": 10,
    "feature_fraction": 0.8,
    "feature_fraction_bynode": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "lambda_l1": 0.1,
    "lambda_l2": 1.0,
    "min_gain_to_split": 0.01,
    "verbose": -1,
    "n_jobs": -1,
    "seed": 42,
}

# ============ 5-FOLD CROSS-VALIDATION ============
# Use the full training set (train + val) for CV
X_cv = pd.concat([X_train_final, X_val_final], axis=0, ignore_index=True).reset_index(drop=True)
y_cv = np.concatenate([y_train, y_val])

kf = KFold(n_splits=5, shuffle=True, random_state=42)

oof_predictions = np.zeros(len(X_cv))
test_predictions = np.zeros(len(X_test_final))
fold_scores = []

for fold, (tr_idx, va_idx) in enumerate(kf.split(X_cv)):
    X_tr = X_cv.iloc[tr_idx].reset_index(drop=True)
    X_va = X_cv.iloc[va_idx].reset_index(drop=True)
    y_tr = y_cv[tr_idx]
    y_va = y_cv[va_idx]

    train_data = lgb.Dataset(X_tr, label=y_tr, free_raw_data=False)
    val_data = lgb.Dataset(X_va, label=y_va, free_raw_data=False, reference=train_data)

    callbacks = [
        lgb.early_stopping(stopping_rounds=100, verbose=False),
        lgb.log_evaluation(period=0),
    ]

    model = lgb.train(
        lgb_params,
        train_data,
        num_boost_round=5000,
        valid_sets=[val_data],
        valid_names=["val"],
        callbacks=callbacks,
    )

    best_iter = model.best_iteration
    oof_predictions[va_idx] = model.predict(X_va, num_iteration=best_iter)
    test_predictions += model.predict(X_test_final, num_iteration=best_iter) / 5

    fold_rmse = np.sqrt(mean_squared_error(y_va, oof_predictions[va_idx]))
    fold_scores.append(fold_rmse)
    print(f"  Fold {fold + 1}: best_iter={best_iter}, val_rmse={fold_rmse:.6f}")

# ============ FINAL EVALUATION (official metric: RMSE in original space) ============
# Clip predictions to non-negative
oof_predictions = np.clip(oof_predictions, 0, None)
test_predictions = np.clip(test_predictions, 0, None)

score = np.sqrt(mean_squared_error(y_cv, oof_predictions))
baseline_rmse = np.sqrt(mean_squared_error(y_cv, np.full(len(y_cv), y_cv.mean())))
print(f"\nValidation RMSE (OOF): {score:.6f}")
print(f"Baseline (mean) RMSE: {baseline_rmse:.6f}")
print(f"Improvement over baseline: {(1 - score / baseline_rmse) * 100:.2f}%")
print(f"Mean fold RMSE: {np.mean(fold_scores):.6f} (+/- {np.std(fold_scores):.6f})")

# Feature importance from last fold's model
importance = model.feature_importance(importance_type="gain")
feat_imp = pd.DataFrame(
    {"feature": X_cv.columns, "importance": importance}
).sort_values("importance", ascending=False)
print(f"\nTop 10 features by gain:")
print(feat_imp.head(10).to_string(index=False))

# ============ GENERATE SUBMISSION ============
test_ids = np.arange(len(X_test_final))
submission = pd.DataFrame(
    {"id": test_ids.astype(int), "Size of Purchase (MW)": test_predictions}
)
submission.to_csv("./submission/submission.csv", index=False)
print(f"\nSubmission saved: {submission.shape}")
print(submission.head())

print(f"Final Validation Score: {score:.6f}")
