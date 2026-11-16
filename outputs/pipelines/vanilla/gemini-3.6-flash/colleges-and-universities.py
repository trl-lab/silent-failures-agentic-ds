import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import KFold
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

# 1. Load Data
train_path = "/path/to/user/workspace/ds_failures/datasets/strable/colleges-and-universities/data/train.parquet"
test_path = "/path/to/user/workspace/ds_failures/datasets/strable/colleges-and-universities/data/test.parquet"

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = "RATIO_ENROLL_EMP"

# Handle target missing/invalid values in train
train_df = train_df.dropna(subset=[target_col]).reset_index(drop=True)
train_df = train_df[np.isfinite(train_df[target_col])].reset_index(drop=True)

y = train_df[target_col].values
min_target = y.min() if len(y) > 0 else 0

X_train_raw = train_df.drop(columns=[target_col])
X_test_raw = test_df.copy()

# Combine for uniform preprocessing
n_train = len(X_train_raw)
combined = pd.concat([X_train_raw, X_test_raw], ignore_index=True)

# Drop pure ID columns if present
id_col_candidates = ["OBJECTID", "FID", "GLOBALID", "ID", "id"]
drop_cols = [c for c in id_col_candidates if c in combined.columns]
if drop_cols:
    combined = combined.drop(columns=drop_cols)

# Identify column types
num_cols = combined.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = combined.select_dtypes(include=["object", "category"]).columns.tolist()

# Impute numeric features
for c in num_cols:
    combined[c] = combined[c].replace([np.inf, -np.inf], np.nan)
    med = combined[c].median()
    if pd.isna(med):
        med = 0
    combined[c] = combined[c].fillna(med)

# Feature engineering: Pairwise ratios and differences for numeric features
if len(num_cols) >= 2:
    selected_num = num_cols[:15]
    for i in range(len(selected_num)):
        for j in range(i + 1, len(selected_num)):
            c1, c2 = selected_num[i], selected_num[j]
            combined[f"{c1}_div_{c2}"] = combined[c1] / (np.abs(combined[c2]) + 1e-4)
            combined[f"{c2}_div_{c1}"] = combined[c2] / (np.abs(combined[c1]) + 1e-4)
            combined[f"{c1}_minus_{c2}"] = combined[c1] - combined[c2]

# Process categorical features
for c in cat_cols:
    combined[c] = combined[c].fillna("Missing").astype(str)
    freq = combined[c].value_counts().to_dict()
    combined[f"{c}_freq"] = combined[c].map(freq)
    combined[c] = combined[c].astype("category")

# Split back to train and test
X_train = combined.iloc[:n_train].copy()
X_test = combined.iloc[n_train:].copy()

# Prepare datasets for CatBoost
X_train_cat = X_train.copy()
X_test_cat = X_test.copy()
cat_cols_str = cat_cols.copy()
for c in cat_cols_str:
    X_train_cat[c] = X_train_cat[c].astype(str)
    X_test_cat[c] = X_test_cat[c].astype(str)

# Cross-Validation Setup
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_lgb = np.zeros(n_train)
test_lgb = np.zeros(len(X_test))

oof_xgb = np.zeros(n_train)
test_xgb = np.zeros(len(X_test))

oof_cat = np.zeros(n_train)
test_cat = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y)):
    X_tr, y_tr = X_train.iloc[train_idx], y[train_idx]
    X_va, y_va = X_train.iloc[val_idx], y[val_idx]

    # LightGBM
    model_lgb = lgb.LGBMRegressor(
        n_estimators=1500,
        learning_rate=0.03,
        num_leaves=31,
        random_state=42 + fold,
        n_jobs=-1,
        verbose=-1
    )
    model_lgb.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )
    oof_lgb[val_idx] = model_lgb.predict(X_va)
    test_lgb += model_lgb.predict(X_test) / n_splits

    # XGBoost
    model_xgb = xgb.XGBRegressor(
        n_estimators=1500,
        learning_rate=0.03,
        max_depth=6,
        tree_method="hist",
        enable_categorical=True,
        early_stopping_rounds=50,
        random_state=42 + fold,
        n_jobs=-1
    )
    model_xgb.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        verbose=False
    )
    oof_xgb[val_idx] = model_xgb.predict(X_va)
    test_xgb += model_xgb.predict(X_test) / n_splits

    # CatBoost
    X_tr_cat, X_va_cat = X_train_cat.iloc[train_idx], X_train_cat.iloc[val_idx]
    model_cat = CatBoostRegressor(
        iterations=1500,
        learning_rate=0.03,
        depth=6,
        cat_features=cat_cols_str,
        random_seed=42 + fold,
        verbose=0
    )
    model_cat.fit(
        X_tr_cat, y_tr,
        eval_set=(X_va_cat, y_va),
        early_stopping_rounds=50,
        verbose=False
    )
    oof_cat[val_idx] = model_cat.predict(X_va_cat)
    test_cat += model_cat.predict(X_test_cat) / n_splits

# Find Optimal Blending Weights
def loss_func(weights):
    w1, w2, w3 = weights
    pred = w1 * oof_lgb + w2 * oof_xgb + w3 * oof_cat
    return np.sqrt(np.mean((