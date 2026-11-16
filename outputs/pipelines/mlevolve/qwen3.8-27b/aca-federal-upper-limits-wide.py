import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
import os
import json
import warnings

warnings.filterwarnings("ignore")

# Create directories
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# ============================================================
# 1. LOAD DATA
# ============================================================
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")

# ============================================================
# 2. IDENTIFY FEATURE TYPES
# ============================================================
target_col = "ACA FUL"
id_col = "id" if "id" in train_df.columns else None

if id_col:
    train_ids = train_df[id_col].copy()
    test_ids = (
        test_df[id_col].copy()
        if id_col in test_df.columns
        else pd.Series(range(len(test_df)))
    )
    train_df = train_df.drop(columns=[id_col])
    if id_col in test_df.columns:
        test_df = test_df.drop(columns=[id_col])
else:
    test_ids = pd.Series(range(len(test_df)))

y_full = train_df[target_col].copy()
train_df = train_df.drop(columns=[target_col])

numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = train_df.select_dtypes(
    include=["object", "category"]
).columns.tolist()

for col in numeric_cols[:]:
    if train_df[col].nunique() < 20:
        categorical_cols.append(col)
        numeric_cols.remove(col)

print(f"Numeric features ({len(numeric_cols)}): {numeric_cols}")
print(f"Categorical features ({len(categorical_cols)}): {categorical_cols}")

# ============================================================
# 3. 5-FOLD CROSS-VALIDATION SETUP
# ============================================================
from sklearn.model_selection import KFold

n_folds = 5
kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

test_df = test_df.copy()

print(f"Using {n_folds}-fold cross-validation")

# ============================================================
# 4. FEATURE ENGINEERING HELPER (applied per fold)
# ============================================================

def build_features(X_train_fold, X_val_fold, X_test, y_train_fold, categorical_cols, numeric_cols):
    """Build features for a given fold. Returns feature arrays and cat feature indices."""
    X_tr = X_train_fold.copy()
    X_vl = X_val_fold.copy()
    X_te = X_test.copy()

    # --- Missingness indicators BEFORE any imputation ---
    missing_cols = X_tr.columns[X_tr.isnull().any()].tolist()
    for col in missing_cols:
        miss_col = f"{col}_missing"
        X_tr[miss_col] = X_tr[col].isnull().astype(float)
        X_vl[miss_col] = X_vl[col].isnull().astype(float)
        X_te[miss_col] = X_te[col].isnull().astype(float)

    # --- Label encode categoricals (ordinal) for LGBM/XGB ---
    cat_columns_to_encode = [
        col
        for col in categorical_cols
        if col in X_tr.columns and col in X_vl.columns
    ]

    if cat_columns_to_encode:
        oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        oe.fit(X_tr[cat_columns_to_encode])
        X_tr[cat_columns_to_encode] = oe.transform(X_tr[cat_columns_to_encode])
        X_vl[cat_columns_to_encode] = oe.transform(X_vl[cat_columns_to_encode])
        if all(c in X_te.columns for c in cat_columns_to_encode):
            X_te[cat_columns_to_encode] = oe.transform(X_te[cat_columns_to_encode])

    # --- Impute missing values ---
    if missing_cols:
        imputer = SimpleImputer(strategy="median")
        imputer.fit(X_tr[missing_cols])
        X_tr[missing_cols] = imputer.transform(X_tr[missing_cols])
        X_vl[missing_cols] = imputer.transform(X_vl[missing_cols])
        X_te[missing_cols] = imputer.transform(X_te[missing_cols])

    # --- Log transform ONLY for strictly positive features (min > 0) ---
    all_numeric_cols = X_tr.select_dtypes(include=[np.number]).columns.tolist()
    for col in all_numeric_cols:
        if X_tr[col].min() > 0:
            X_tr[f"{col}_log"] = np.log1p(X_tr[col])
            X_vl[f"{col}_log"] = np.log1p(X_vl[col])
            X_te[f"{col}_log"] = np.log1p(X_te[col])

    # --- Interaction features between key categorical variables ---
    interaction_pairs = [
        ('NDC', 'Ingredient'),
        ('NDC', 'Strength'),
        ('NDC', 'Dosage'),
        ('Ingredient', 'Strength'),
        ('Ingredient', 'Dosage'),
        ('Strength', 'Dosage'),
    ]
    for col1, col2 in interaction_pairs:
        if col1 in X_tr.columns and col2 in X_tr.columns:
            inter_col = f'{col1}_{col2}_inter'
            X_tr[inter_col] = X_tr[col1].astype(str) + '_' + X_tr[col2].astype(str)
            X_vl[inter_col] = X_vl[col1].astype(str) + '_' + X_vl[col2].astype(str)
            X_te[inter_col] = X_te[col1].astype(str) + '_' + X_te[col2].astype(str)
            # Ordinal encode the interaction features
            oe_inter = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
            X_tr[inter_col] = oe_inter.fit_transform(X_tr[[inter_col]]).ravel()
            X_vl[inter_col] = oe_inter.transform(X_vl[[inter_col]]).ravel()
            X_te[inter_col] = oe_inter.transform(X_te[[inter_col]]).ravel()

    # --- Final feature list ---
    final_feature_cols = X_tr.columns.tolist()

    # Ensure all columns exist in val and test
    for col in final_feature_cols:
        if col not in X_vl.columns:
            X_vl[col] = 0.0
        if col not in X_te.columns:
            X_te[col] = 0.0

    # Convert to numpy arrays
    X_tr_arr = X_tr[final_feature_cols].values.astype(np.float64)
    X_vl_arr = X_vl[final_feature_cols].values.astype(np.float64)
    X_te_arr = X_te[final_feature_cols].values.astype(np.float64)

    # Replace any remaining NaN/Inf
    X_tr_arr = np.nan_to_num(X_tr_arr, nan=0.0, posinf=0.0, neginf=0.0)
    X_vl_arr = np.nan_to_num(X_vl_arr, nan=0.0, posinf=0.0, neginf=0.0)
    X_te_arr = np.nan_to_num(X_te_arr, nan=0.0, posinf=0.0, neginf=0.0)

    return X_tr_arr, X_vl_arr, X_te_arr, final_feature_cols

# ============================================================
# 5. MODEL PARAMETERS
# ============================================================
lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.03,
    "num_leaves": 31,
    "max_depth": 5,
    "min_child_samples": 5,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "n_estimators": 5000,
    "verbose": -1,
    "n_jobs": -1,
    "random_state": 42,
}

xgb_params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "learning_rate": 0.03,
    "max_depth": 6,
    "min_child_weight": 5,
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "n_estimators": 5000,
    "verbosity": 0,
    "random_state": 42,
    "n_jobs": -1,
    "early_stopping_rounds": 100,
}

cat_params = {
    "loss_function": "RMSE",
    "learning_rate": 0.03,
    "depth": 6,
    "l2_leaf_reg": 1.0,
    "iterations": 5000,
    "verbose": 0,
    "random_seed": 42,
    "early_stopping_rounds": 100,
}

# ============================================================
# 6. 5-FOLD CROSS-VALIDATION TRAINING
# ============================================================
# Identify categorical columns for CatBoost native handling
cat_feature_names = [
    col for col in categorical_cols
    if col in train_df.columns
]

# Initialize prediction arrays
val_preds_lgb = np.zeros(len(train_df))
val_preds_xgb = np.zeros(len(train_df))
val_preds_cat = np.zeros(len(train_df))
test_preds_lgb = np.zeros(len(test_df))
test_preds_xgb = np.zeros(len(test_df))
test_preds_cat = np.zeros(len(test_df))

for fold_idx, (train_idx, val_idx) in enumerate(kf.split(train_df)):
    print(f"\n{'='*50}")
    print(f"Fold {fold_idx + 1}/{n_folds}")
    print(f"{'='*50}")

    X_tr_fold = train_df.iloc[train_idx].copy()
    y_tr_fold = y_full.iloc[train_idx].copy()
    X_vl_fold = train_df.iloc[val_idx].copy()
    y_vl_fold = y_full.iloc[val_idx].copy()

    # Build features for this fold
    X_tr_arr, X_vl_arr, X_te_arr, final_feature_cols = build_features(
        X_tr_fold, X_vl_fold, test_df, y_tr_fold, categorical_cols, numeric_cols
    )

    y_tr = y_tr_fold.values.astype(np.float64)
    y_vl = y_vl_fold.values.astype(np.float64)

    # --- Train LightGBM ---
    lgb_model = lgb.LGBMRegressor(**lgb_params)
    lgb_callbacks = [
        lgb.early_stopping(stopping_rounds=100, verbose=False),
        lgb.log_evaluation(period=1000),
    ]
    lgb_model.fit(
        X_tr_arr, y_tr,
        eval_set=[(X_vl_arr, y_vl)],
        callbacks=lgb_callbacks,
    )
    lgb_best_it = lgb_model.best_iteration_
    val_preds_lgb[val_idx] = lgb_model.predict(X_vl_arr, num_iteration=lgb_best_it)
    test_preds_lgb += lgb_model.predict(X_te_arr, num_iteration=lgb_best_it) / n_folds
    print(f"  LGBM best iteration: {lgb_best_it}")

    # --- Train XGBoost ---
    xgb_model = xgb.XGBRegressor(**xgb_params)
    xgb_model.fit(
        X_tr_arr, y_tr,
        eval_set=[(X_vl_arr, y_vl)],
        verbose=False,
    )
    xgb_best_it = xgb_model.best_iteration
    val_preds_xgb[val_idx] = xgb_model.predict(X_vl_arr, iteration_range=(0, xgb_best_it + 1))
    test_preds_xgb += xgb_model.predict(X_te_arr, iteration_range=(0, xgb_best_it + 1)) / n_folds
    print(f"  XGB best iteration: {xgb_best_it}")

    # --- Train CatBoost with native categorical handling ---
    # For CatBoost, use the original DataFrame with categorical columns
    X_tr_cb = X_tr_fold.copy()
    X_vl_cb = X_vl_fold.copy()
    X_te_cb = test_df.copy()

    # Add missingness indicators for CatBoost
    missing_cols_cb = X_tr_cb.columns[X_tr_cb.isnull().any()].tolist()
    for col in missing_cols_cb:
        miss_col = f"{col}_missing"
        X_tr_cb[miss_col] = X_tr_cb[col].isnull().astype(float)
        X_vl_cb[miss_col] = X_vl_cb[col].isnull().astype(float)
        X_te_cb[miss_col] = X_te_cb[col].isnull().astype(float)

    # Add log transforms for CatBoost
    all_numeric_cb = X_tr_cb.select_dtypes(include=[np.number]).columns.tolist()
    for col in all_numeric_cb:
        if X_tr_cb[col].min() > 0:
            X_tr_cb[f"{col}_log"] = np.log1p(X_tr_cb[col])
            X_vl_cb[f"{col}_log"] = np.log1p(X_vl_cb[col])
            X_te_cb[f"{col}_log"] = np.log1p(X_te_cb[col])

    # Add interaction features for CatBoost
    interaction_pairs = [
        ('NDC', 'Ingredient'),
        ('NDC', 'Strength'),
        ('NDC', 'Dosage'),
        ('Ingredient', 'Strength'),
        ('Ingredient', 'Dosage'),
        ('Strength', 'Dosage'),
    ]
    for col1, col2 in interaction_pairs:
        if col1 in X_tr_cb.columns and col2 in X_tr_cb.columns:
            inter_col = f'{col1}_{col2}_inter'
            X_tr_cb[inter_col] = X_tr_cb[col1].astype(str) + '_' + X_tr_cb[col2].astype(str)
            X_vl_cb[inter_col] = X_vl_cb[col1].astype(str) + '_' + X_vl_cb[col2].astype(str)
            X_te_cb[inter_col] = X_te_cb[col1].astype(str) + '_' + X_te_cb[col2].astype(str)

    # Identify categorical columns for CatBoost (original cats + interaction cols)
    cat_feature_names_cb = [col for col in cat_feature_names if col in X_tr_cb.columns]
    interaction_col_names = [f'{c1}_{c2}_inter' for c1, c2 in interaction_pairs if c1 in X_tr_cb.columns and c2 in X_tr_cb.columns]
    all_cat_cols_cb = cat_feature_names_cb + interaction_col_names

    # Get column indices for categorical features
    cat_indices = [X_tr_cb.columns.tolist().index(c) for c in all_cat_cols_cb if c in X_tr_cb.columns]

    # Fill NaN in categorical columns with a sentinel value for CatBoost
    for c in all_cat_cols_cb:
        if c in X_tr_cb.columns:
            X_tr_cb[c] = X_tr_cb[c].fillna("__MISSING__").astype(str)
            X_vl_cb[c] = X_vl_cb[c].fillna("__MISSING__").astype(str)
            X_te_cb[c] = X_te_cb[c].fillna("__MISSING__").astype(str)

    # Fill NaN in numeric columns
    num_cols_cb = [c for c in X_tr_cb.columns if c not in all_cat_cols_cb]
    for c in num_cols_cb:
        if X_tr_cb[c].isnull().any():
            median_val = X_tr_cb[c].median()
            X_tr_cb[c] = X_tr_cb[c].fillna(median_val)
            X_vl_cb[c] = X_vl_cb[c].fillna(median_val)
            X_te_cb[c] = X_te_cb[c].fillna(median_val)

    cat_model = CatBoostRegressor(**cat_params)
    cat_model.fit(
        X_tr_cb, y_tr,
        eval_set=(X_vl_cb, y_vl),
        cat_features=cat_indices,
        verbose=0,
    )
    cat_best_it = cat_model.get_best_iteration()
    val_preds_cat[val_idx] = cat_model.predict(X_vl_cb)
    test_preds_cat += cat_model.predict(X_te_cb) / n_folds
    print(f"  CatBoost best iteration: {cat_best_it}")

# ============================================================
# 7. VALIDATION PREDICTIONS (Weighted Ensemble)
# ============================================================
# Weighted ensemble: 0.4*LGB + 0.3*XGB + 0.3*CatBoost
y_val_pred = 0.4 * val_preds_lgb + 0.3 * val_preds_xgb + 0.3 * val_preds_cat

# Compute RMSE in ORIGINAL space (official metric)
val_rmse = np.sqrt(mean_squared_error(y_full.values, y_val_pred))

print(f"\n--- 5-Fold CV Validation ---")
print(f"  CV RMSE (original space): {val_rmse:.4f}")
print(f"  True mean: {y_full.mean():.2f}, Pred mean: {y_val_pred.mean():.2f}")
print(f"  True std: {y_full.std():.2f}, Pred std: {y_val_pred.std():.2f}")

# ============================================================
# 8. TEST PREDICTIONS (Weighted Ensemble)
# ============================================================
y_test_pred = 0.4 * test_preds_lgb + 0.3 * test_preds_xgb + 0.3 * test_preds_cat

print(f"\n--- Test Predictions ---")
print(f"  Test pred range: [{y_test_pred.min():.2f}, {y_test_pred.max():.2f}]")
print(f"  Test pred mean: {y_test_pred.mean():.2f}")

# ============================================================
# 9. SAVE SUBMISSION
# ============================================================
submission = pd.DataFrame({"id": test_ids.values, "ACA FUL": y_test_pred})
submission.to_csv("./submission/submission.csv", index=False)
print(f"\nSubmission saved: {submission.shape}")

# ============================================================
# FINAL SCORE
# ============================================================
score = val_rmse
print(f"Final Validation Score: {score:.4f}")
