import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from scipy.optimize import minimize
import os
import warnings
import json
import pickle

warnings.filterwarnings("ignore")
os.makedirs("./submission", exist_ok=True)
os.makedirs("./working", exist_ok=True)

# ============================================================
# 1. LOAD DATA
# ============================================================
print("Loading data...")
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")
print(f"Target distribution:\n{train_df['TARGET'].value_counts(normalize=True)}")

# ============================================================
# 2. SPLIT FIRST (before any processing to avoid leakage)
# ============================================================
train_part, val_part = train_test_split(
    train_df, test_size=0.2, random_state=42, stratify=train_df["TARGET"]
)

print(f"Train part shape: {train_part.shape}, Val part shape: {val_part.shape}")
print(f"Val target rate: {val_part['TARGET'].mean():.4f}")

# ============================================================
# 3. FEATURE ENGINEERING
# ============================================================


def engineer_features(df, is_train=True, le_dict=None):
    """Apply domain-informed feature engineering. Fit encoders only on train."""
    df = df.copy()

    # --- Handle special missing value codes ---
    if "DAYS_EMPLOYMENT" in df.columns:
        df["DAYS_EMPLOYMENT"] = df["DAYS_EMPLOYMENT"].where(
            df["DAYS_EMPLOYMENT"] > -70000, np.nan
        )

    if "AMT_INCOME_TOTAL" in df.columns:
        df["AMT_INCOME_TOTAL"] = df["AMT_INCOME_TOTAL"].where(
            df["AMT_INCOME_TOTAL"] > 0, np.nan
        )

    if "AMT_ANNUITY" in df.columns:
        df["AMT_ANNUITY"] = df["AMT_ANNUITY"].where(df["AMT_ANNUITY"] > 0, np.nan)

    if "AMT_GOODS_PRICE" in df.columns:
        df["AMT_GOODS_PRICE"] = df["AMT_GOODS_PRICE"].where(
            df["AMT_GOODS_PRICE"] > 0, np.nan
        )

    # --- Convert negative DAYS to positive durations ---
    days_cols = [
        c for c in df.columns if c.startswith("DAYS_") and c != "DAYS_EMPLOYMENT"
    ]
    for col in days_cols:
        if col in df.columns:
            df[col + "_YEARS"] = -df[col] / 365.25

    # Age
    if "DAYS_BIRTH" in df.columns:
        df["AGE"] = -df["DAYS_BIRTH"] / 365.25

    # Employment years (positive)
    if "DAYS_EMPLOYMENT" in df.columns:
        df["EMPLOYMENT_YEARS"] = -df["DAYS_EMPLOYMENT"] / 365.25
        df["EMPLOYMENT_YEARS"] = df["EMPLOYMENT_YEARS"].where(
            df["EMPLOYMENT_YEARS"] > 0, np.nan
        )

    # --- Financial ratio features ---
    if "AMT_CREDIT" in df.columns and "AMT_INCOME_TOTAL" in df.columns:
        df["CREDIT_TO_INCOME"] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"]

    if "AMT_ANNUITY" in df.columns and "AMT_INCOME_TOTAL" in df.columns:
        df["ANNUITY_TO_INCOME"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]

    if "AMT_CREDIT" in df.columns and "AMT_GOODS_PRICE" in df.columns:
        df["CREDIT_TO_GOODS"] = df["AMT_CREDIT"] / df["AMT_GOODS_PRICE"]

    if "AMT_ANNUITY" in df.columns and "AMT_CREDIT" in df.columns:
        df["ANNUITY_TO_CREDIT"] = df["AMT_ANNUITY"] / df["AMT_CREDIT"]

    # Per capita income
    if "AMT_INCOME_TOTAL" in df.columns and "CNT_FAM_MEMBERS" in df.columns:
        df["INCOME_PER_CAPITA"] = df["AMT_INCOME_TOTAL"] / (df["CNT_FAM_MEMBERS"] + 1)

    if "AMT_INCOME_TOTAL" in df.columns and "CNT_CHILDREN" in df.columns:
        df["INCOME_PER_CHILD"] = df["AMT_INCOME_TOTAL"] / (df["CNT_CHILDREN"] + 1)

    # --- Log transforms for skewed monetary features ---
    for col in ["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE"]:
        if col in df.columns:
            df[col + "_LOG"] = np.log1p(df[col].clip(lower=0))

    # --- Interaction features ---
    if "AGE" in df.columns and "EMPLOYMENT_YEARS" in df.columns:
        df["AGE_EMPLOYMENT_RATIO"] = df["EMPLOYMENT_YEARS"] / (df["AGE"] + 1)

    if "CNT_CHILDREN" in df.columns and "CNT_FAM_MEMBERS" in df.columns:
        df["CHILDREN_RATIO"] = df["CNT_CHILDREN"] / (df["CNT_FAM_MEMBERS"] + 1)

    # --- Bureau / Credit History Feature Engineering ---
    # Identify columns that encode credit history signals
    # These are columns related to previous applications, credit bureaus, and financial history

    # Previous application aggregation features
    prev_cols = [c for c in df.columns if "PREV" in c.upper()]
    bureau_cols = [c for c in df.columns if "BUREAU" in c.upper()]
    credit_cols = [c for c in df.columns if "CREDIT" in c.upper()]
    debt_cols = [c for c in df.columns if "DEBT" in c.upper() or "OBLIG" in c.upper()]

    # Previous application count and statistics
    prev_numeric_cols = []
    prev_days_cols = []
    if prev_cols:
        # Count of previous applications (non-null indicators)
        df["CNT_PREV_APPLICATIONS"] = df[prev_cols].notna().sum(axis=1)

        # Numeric previous application columns - aggregate stats
        prev_numeric_cols = [c for c in prev_cols if df[c].dtype in ["int64", "float64"]]
        if prev_numeric_cols:
            df["CNT_PREV_NUMERIC"] = df[prev_numeric_cols].notna().sum(axis=1)
            df["PREV_AMT_CREDIT_SUM"] = df[prev_numeric_cols].sum(axis=1)
            df["PREV_AMT_CREDIT_MEAN"] = df[prev_numeric_cols].mean(axis=1)
            df["PREV_AMT_CREDIT_MAX"] = df[prev_numeric_cols].max(axis=1)
            df["PREV_AMT_CREDIT_MIN"] = df[prev_numeric_cols].min(axis=1)

        # Previous application days statistics
        prev_days_cols = [c for c in prev_cols if c.startswith("DAYS_")]
        if prev_days_cols:
            df["DAYS_DECISION_PREV_MIN"] = df[prev_days_cols].min(axis=1)
            df["DAYS_DECISION_PREV_MAX"] = df[prev_days_cols].max(axis=1)
            df["DAYS_DECISION_PREV_MEAN"] = df[prev_days_cols].mean(axis=1)
            # Convert to years
            df["DAYS_DECISION_PREV_MIN_YEARS"] = -df["DAYS_DECISION_PREV_MIN"] / 365.25
            df["DAYS_DECISION_PREV_MAX_YEARS"] = -df["DAYS_DECISION_PREV_MAX"] / 365.25

    # Credit bureau features
    if bureau_cols:
        df["CNT_CREDIT_BUREAU"] = df[bureau_cols].notna().sum(axis=1)
        bureau_numeric_cols = [c for c in bureau_cols if df[c].dtype in ["int64", "float64"]]
        if bureau_numeric_cols:
            df["BUREAU_SUM"] = df[bureau_numeric_cols].sum(axis=1)
            df["BUREAU_MEAN"] = df[bureau_numeric_cols].mean(axis=1)
            df["BUREAU_MAX"] = df[bureau_numeric_cols].max(axis=1)

    # Credit active counts (from credit-related columns)
    if credit_cols:
        credit_numeric_cols = [c for c in credit_cols if df[c].dtype in ["int64", "float64"]]
        if credit_numeric_cols:
            df["CNT_CREDIT_NUMERIC"] = df[credit_numeric_cols].notna().sum(axis=1)
            df["CREDIT_AMT_SUM"] = df[credit_numeric_cols].sum(axis=1)
            df["CREDIT_AMT_MEAN"] = df[credit_numeric_cols].mean(axis=1)

    # Debt-to-income ratio from bureau
    if "AMT_INCOME_TOTAL" in df.columns:
        if debt_cols:
            debt_numeric_cols = [c for c in debt_cols if df[c].dtype in ["int64", "float64"]]
            if debt_numeric_cols:
                df["DEBT_TOTAL"] = df[debt_numeric_cols].sum(axis=1)
                df["DEBT_INC_RATIO"] = df["DEBT_TOTAL"] / (df["AMT_INCOME_TOTAL"] + 1)

    # Previous application outcome features
    # Look for columns that indicate approval/outcome of previous applications
    outcome_cols = [c for c in df.columns if "APPROVED" in c.upper() or "APPROVE" in c.upper()]
    if outcome_cols:
        df["CNT_PREV_APPROVED"] = df[outcome_cols].sum(axis=1)

    # Interaction between current and previous application features
    if "AMT_CREDIT" in df.columns:
        if prev_numeric_cols:
            df["AMT_CREDIT_PREV_RATIO"] = df["AMT_CREDIT"] / (df["PREV_AMT_CREDIT_SUM"] + 1)

    if "DAYS_BIRTH" in df.columns:
        if prev_days_cols:
            df["BIRTH_TO_PREV_DAYS"] = df["DAYS_BIRTH"] - df["DAYS_DECISION_PREV_MAX"]

    # Additional credit history interaction features
    if "AMT_CREDIT" in df.columns and "AMT_INCOME_TOTAL" in df.columns:
        if "CNT_CREDIT_NUMERIC" in df.columns:
            df["CREDIT_PER_INCOME_PER_APP"] = df["AMT_CREDIT"] / (
                df["AMT_INCOME_TOTAL"] * (df["CNT_CREDIT_NUMERIC"] + 1)
            )

    # Financial stability indicators
    if "AMT_INCOME_TOTAL" in df.columns and "AMT_CREDIT" in df.columns:
        if "CNT_PREV_APPLICATIONS" in df.columns:
            df["CREDIT_PER_PREV_APP"] = df["AMT_CREDIT"] / (df["CNT_PREV_APPLICATIONS"] + 1)
            df["INCOME_PER_PREV_APP"] = df["AMT_INCOME_TOTAL"] / (df["CNT_PREV_APPLICATIONS"] + 1)

    # Region and population interaction with credit
    if "REGION_POPULATION_RELATIVE" in df.columns and "AMT_CREDIT" in df.columns:
        df["CREDIT_PER_POP"] = df["AMT_CREDIT"] * df["REGION_POPULATION_RELATIVE"]
        df["INCOME_PER_POP"] = df["AMT_INCOME_TOTAL"] * df["REGION_POPULATION_RELATIVE"]

    # --- Categorical encoding ---
    cat_cols = [c for c in df.columns if df[c].dtype == "object"]

    if le_dict is None:
        # Fit on train
        le_dict = {}
        for col in cat_cols:
            le = LabelEncoder()
            le_dict[col] = le.fit(df[col].astype(str))
        for col in cat_cols:
            df[col] = le_dict[col].transform(df[col].astype(str))
    else:
        # Transform using fitted encoders
        for col in cat_cols:
            if col in le_dict:
                df[col] = df[col].astype(str)
                unseen = ~df[col].isin(le_dict[col].classes_)
                df[col] = np.where(
                    unseen,
                    -1,
                    le_dict[col].transform(df[col].where(~unseen, "__UNSEEN__")),
                )

    return df, le_dict


# Fit encoders on train_part only
train_part_eng, le_dict = engineer_features(train_part, is_train=True, le_dict=None)
val_part_eng, _ = engineer_features(val_part, is_train=False, le_dict=le_dict)
test_eng, _ = engineer_features(test_df, is_train=False, le_dict=le_dict)

print(
    f"After feature engineering - Train: {train_part_eng.shape}, Val: {val_part_eng.shape}, Test: {test_eng.shape}"
)

# ============================================================
# 4. PREPARE MODELLING DATA
# ============================================================
target_col = "TARGET"
id_col = "SK_ID_CURR"

feature_cols = [c for c in train_part_eng.columns if c not in [target_col, id_col]]

X_train = train_part_eng[feature_cols].apply(pd.to_numeric, errors="coerce")
y_train = train_part_eng[target_col].values
X_val = val_part_eng[feature_cols].apply(pd.to_numeric, errors="coerce")
y_val = val_part_eng[target_col].values
X_test = test_eng[feature_cols].apply(pd.to_numeric, errors="coerce")

print(f"Number of features: {len(feature_cols)}")
print(
    f"Train missing: {X_train.isna().sum().sum()}, Val missing: {X_val.isna().sum().sum()}"
)

# ============================================================
# 5. MODEL FACTORY FUNCTIONS
# ============================================================


def get_lightgbm_model():
    return lgb.LGBMClassifier(
        n_estimators=2000,
        learning_rate=0.03,
        max_depth=7,
        num_leaves=63,
        min_child_samples=40,
        subsample=0.75,
        subsample_freq=1,
        colsample_bytree=0.7,
        reg_alpha=1.0,
        reg_lambda=1.0,
        min_split_gain=0.01,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )


def get_lightgbm_model2():
    return lgb.LGBMClassifier(
        n_estimators=3000,
        learning_rate=0.01,
        max_depth=9,
        num_leaves=127,
        min_child_samples=30,
        subsample=0.7,
        subsample_freq=1,
        colsample_bytree=0.6,
        reg_alpha=0.5,
        reg_lambda=2.0,
        min_split_gain=0.005,
        random_state=123,
        n_jobs=-1,
        verbose=-1,
    )


def get_xgboost_model():
    return xgb.XGBClassifier(
        n_estimators=3000,
        learning_rate=0.03,
        max_depth=6,
        min_child_weight=5,
        subsample=0.75,
        colsample_bytree=0.7,
        reg_alpha=0.5,
        reg_lambda=2.0,
        gamma=0.1,
        random_state=42,
        n_jobs=-1,
        eval_metric="auc",
        tree_method="hist",
        early_stopping_rounds=100,
    )


def get_catboost_model():
    return CatBoostClassifier(
        iterations=3000,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=5.0,
        subsample=0.75,
        colsample_bylevel=0.7,
        random_seed=42,
        verbose=0,
        early_stopping_rounds=100,
        auto_class_weights="Balanced",
    )


# ============================================================
# 6. PHASE 1: 5-FOLD CV FOR OOF PREDICTIONS
# ============================================================
print("\n" + "=" * 60)
print("PHASE 1: 5-Fold CV for OOF predictions")
print("=" * 60)

n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

oof_lgb = np.zeros(len(X_train))
oof_lgb2 = np.zeros(len(X_train))
oof_xgb = np.zeros(len(X_train))
oof_cat = np.zeros(len(X_train))
best_iters_lgb = []
best_iters_lgb2 = []
best_iters_xgb = []
best_iters_cat = []

for fold in range(n_folds):
    tr_idx, va_idx = list(skf.split(X_train, y_train))[fold]
    X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
    y_tr, y_va = y_train[tr_idx], y_train[va_idx]

    # --- LightGBM ---
    m_lgb = get_lightgbm_model()
    m_lgb.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(200, verbose=False)],
    )
    oof_lgb[va_idx] = m_lgb.predict_proba(X_va)[:, 1]
    best_iters_lgb.append(m_lgb.best_iteration_)

    # --- LightGBM 2 (deeper trees, lower learning rate) ---
    m_lgb2 = get_lightgbm_model2()
    m_lgb2.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(200, verbose=False)],
    )
    oof_lgb2[va_idx] = m_lgb2.predict_proba(X_va)[:, 1]
    best_iters_lgb2.append(m_lgb2.best_iteration_)

    # --- XGBoost ---
    m_xgb = get_xgboost_model()
    m_xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    oof_xgb[va_idx] = m_xgb.predict_proba(X_va)[:, 1]
    best_iters_xgb.append(m_xgb.best_iteration)

    # --- CatBoost ---
    m_cat = get_catboost_model()
    m_cat.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=0)
    oof_cat[va_idx] = m_cat.predict_proba(X_va)[:, 1]
    best_iters_cat.append(m_cat.get_best_iteration())

    fold_auc_lgb = roc_auc_score(y_va, oof_lgb[va_idx])
    fold_auc_lgb2 = roc_auc_score(y_va, oof_lgb2[va_idx])
    fold_auc_xgb = roc_auc_score(y_va, oof_xgb[va_idx])
    fold_auc_cat = roc_auc_score(y_va, oof_cat[va_idx])
    print(
        f"  Fold {fold+1}: LGB={fold_auc_lgb:.4f} (iter={m_lgb.best_iteration_}), "
        f"LGB2={fold_auc_lgb2:.4f} (iter={m_lgb2.best_iteration_}), "
        f"XGB={fold_auc_xgb:.4f} (iter={m_xgb.best_iteration}), "
        f"CAT={fold_auc_cat:.4f} (iter={m_cat.get_best_iteration()})"
    )

oof_lgb_auc = roc_auc_score(y_train, oof_lgb)
oof_lgb2_auc = roc_auc_score(y_train, oof_lgb2)
oof_xgb_auc = roc_auc_score(y_train, oof_xgb)
oof_cat_auc = roc_auc_score(y_train, oof_cat)
oof_mean_auc = roc_auc_score(y_train, (oof_lgb + oof_lgb2 + oof_xgb + oof_cat) / 4.0)

print(f"\nOOF AUC: LGB={oof_lgb_auc:.6f}, LGB2={oof_lgb2_auc:.6f}, XGB={oof_xgb_auc:.6f}, CAT={oof_cat_auc:.6f}")
print(f"OOF Mean AUC: {oof_mean_auc:.6f}")

# ============================================================
# 7. PHASE 2: WEIGHT OPTIMIZATION VIA GRID SEARCH ON OOF
# ============================================================
print("\n" + "=" * 60)
print("PHASE 2: Weight optimization on OOF predictions")
print("=" * 60)


def optimize_weights(oof_preds, y_true, n_models=4):
    """Optimize ensemble weights using scipy.optimize.minimize."""

    def neg_auc(weights):
        # Normalize weights
        w = np.abs(weights)
        w = w / w.sum()
        ensemble_pred = np.zeros(len(y_true))
        for i in range(n_models):
            ensemble_pred += w[i] * oof_preds[:, i]
        return -roc_auc_score(y_true, ensemble_pred)

    # Initial weights (equal)
    x0 = np.ones(n_models) / n_models

    # Bounds: each weight between 0 and 1
    bounds = [(0, 1)] * n_models

    result = minimize(neg_auc, x0, method="Nelder-Mead", bounds=bounds,
                      options={"maxiter": 10000, "xatol": 1e-8, "fatol": 1e-10})

    optimal_weights = np.abs(result.x)
    optimal_weights = optimal_weights / optimal_weights.sum()
    return optimal_weights, -result.fun


oof_all = np.column_stack([oof_lgb, oof_lgb2, oof_xgb, oof_cat])
optimal_weights, optimal_auc = optimize_weights(oof_all, y_train, n_models=4)

print(f"Optimized weights: LGB={optimal_weights[0]:.4f}, LGB2={optimal_weights[1]:.4f}, "
      f"XGB={optimal_weights[2]:.4f}, CAT={optimal_weights[3]:.4f}")
print(f"Optimized OOF AUC: {optimal_auc:.6f}")

# ============================================================
# 8. PHASE 3: TRAIN FULL MODELS ON ALL TRAINING DATA
# ============================================================
print("\n" + "=" * 60)
print("PHASE 3: Training full models on all training data")
print("=" * 60)

avg_iter_lgb = int(np.mean(best_iters_lgb))
avg_iter_lgb2 = int(np.mean(best_iters_lgb2))
avg_iter_xgb = int(np.mean(best_iters_xgb))
avg_iter_cat = int(np.mean(best_iters_cat))
print(
    f"Average best iterations: LGB={avg_iter_lgb}, LGB2={avg_iter_lgb2}, XGB={avg_iter_xgb}, CAT={avg_iter_cat}"
)

# --- Full LightGBM ---
m_lgb_full = lgb.LGBMClassifier(
    n_estimators=avg_iter_lgb,
    learning_rate=0.03,
    max_depth=7,
    num_leaves=63,
    min_child_samples=40,
    subsample=0.75,
    subsample_freq=1,
    colsample_bytree=0.7,
    reg_alpha=1.0,
    reg_lambda=1.0,
    min_split_gain=0.01,
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)
m_lgb_full.fit(X_train, y_train)
val_lgb = m_lgb_full.predict_proba(X_val)[:, 1]
test_lgb = m_lgb_full.predict_proba(X_test)[:, 1]
print(f"  LGB full model trained (n_estimators={avg_iter_lgb})")

# --- Full LightGBM 2 (deeper trees, lower learning rate) ---
m_lgb2_full = lgb.LGBMClassifier(
    n_estimators=avg_iter_lgb2,
    learning_rate=0.01,
    max_depth=9,
    num_leaves=127,
    min_child_samples=30,
    subsample=0.7,
    subsample_freq=1,
    colsample_bytree=0.6,
    reg_alpha=0.5,
    reg_lambda=2.0,
    min_split_gain=0.005,
    random_state=123,
    n_jobs=-1,
    verbose=-1,
)
m_lgb2_full.fit(X_train, y_train)
val_lgb2 = m_lgb2_full.predict_proba(X_val)[:, 1]
test_lgb2 = m_lgb2_full.predict_proba(X_test)[:, 1]
print(f"  LGB2 full model trained (n_estimators={avg_iter_lgb2})")

# --- Full XGBoost ---
m_xgb_full = xgb.XGBClassifier(
    n_estimators=avg_iter_xgb,
    learning_rate=0.03,
    max_depth=6,
    min_child_weight=5,
    subsample=0.75,
    colsample_bytree=0.7,
    reg_alpha=0.5,
    reg_lambda=2.0,
    gamma=0.1,
    random_state=42,
    n_jobs=-1,
    eval_metric="auc",
    tree_method="hist",
)
m_xgb_full.fit(X_train, y_train, verbose=False)
val_xgb = m_xgb_full.predict_proba(X_val)[:, 1]
test_xgb = m_xgb_full.predict_proba(X_test)[:, 1]
print(f"  XGB full model trained (n_estimators={avg_iter_xgb})")

# --- Full CatBoost ---
m_cat_full = CatBoostClassifier(
    iterations=avg_iter_cat,
    learning_rate=0.03,
    depth=6,
    l2_leaf_reg=5.0,
    subsample=0.75,
    colsample_bylevel=0.7,
    random_seed=42,
    verbose=0,
    auto_class_weights="Balanced",
)
m_cat_full.fit(X_train, y_train, verbose=0)
val_cat = m_cat_full.predict_proba(X_val)[:, 1]
test_cat = m_cat_full.predict_proba(X_test)[:, 1]
print(f"  CAT full model trained (iterations={avg_iter_cat})")

# ============================================================
# 9. PHASE 4: VALIDATION EVALUATION (Official Metric: ROC-AUC)
# ============================================================
print("\n" + "=" * 60)
print("PHASE 4: Validation Evaluation")
print("=" * 60)

val_lgb_auc = roc_auc_score(y_val, val_lgb)
val_lgb2_auc = roc_auc_score(y_val, val_lgb2)
val_xgb_auc = roc_auc_score(y_val, val_xgb)
val_cat_auc = roc_auc_score(y_val, val_cat)
val_mean_auc = roc_auc_score(y_val, (val_lgb + val_lgb2 + val_xgb + val_cat) / 4.0)

print(f"Val AUC - LGB: {val_lgb_auc:.6f}")
print(f"Val AUC - LGB2: {val_lgb2_auc:.6f}")
print(f"Val AUC - XGB: {val_xgb_auc:.6f}")
print(f"Val AUC - CAT: {val_cat_auc:.6f}")
print(f"Val AUC - Simple Mean: {val_mean_auc:.6f}")

# Weighted ensemble using optimized weights
val_weighted = (
    optimal_weights[0] * val_lgb
    + optimal_weights[1] * val_lgb2
    + optimal_weights[2] * val_xgb
    + optimal_weights[3] * val_cat
)
val_weighted_auc = roc_auc_score(y_val, val_weighted)
print(f"Val AUC - Weighted Ensemble: {val_weighted_auc:.6f}")

# ============================================================
# 10. PHASE 5: TEST PREDICTIONS & SUBMISSION
# ============================================================
print("\n" + "=" * 60)
print("PHASE 5: Generating test predictions")
print("=" * 60)

test_pred = (
    optimal_weights[0] * test_lgb
    + optimal_weights[1] * test_lgb2
    + optimal_weights[2] * test_xgb
    + optimal_weights[3] * test_cat
)

# Ensure predictions are valid probabilities
assert test_pred.min() >= 0.0 and test_pred.max() <= 1.0
assert not np.isnan(test_pred).any()

submission = pd.DataFrame({"id": range(len(test_pred)), "TARGET": test_pred})
submission.to_csv("./submission/submission.csv", index=False)
print(f"Submission saved: {submission.shape}")
print(submission.head())
print(
    f"Test pred stats: mean={test_pred.mean():.4f}, std={test_pred.std():.4f}, "
    f"min={test_pred.min():.6f}, max={test_pred.max():.6f}"
)

# ============================================================
# 11. SAVE MODELS
# ============================================================
print("\n" + "=" * 60)
print("PHASE 6: Saving models")
print("=" * 60)

with open("./working/m_lgb_full.pkl", "wb") as f:
    pickle.dump(m_lgb_full, f)
with open("./working/m_lgb2_full.pkl", "wb") as f:
    pickle.dump(m_lgb2_full, f)
with open("./working/m_xgb_full.pkl", "wb") as f:
    pickle.dump(m_xgb_full, f)
with open("./working/m_cat_full.pkl", "wb") as f:
    pickle.dump(m_cat_full, f)
with open("./working/optimal_weights.json", "w") as f:
    json.dump({"lgb": optimal_weights[0], "lgb2": optimal_weights[1],
               "xgb": optimal_weights[2], "cat": optimal_weights[3]}, f)

print("Models saved to ./working/")

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(
    f"  OOF AUC (train CV):   LGB={oof_lgb_auc:.6f}, LGB2={oof_lgb2_auc:.6f}, XGB={oof_xgb_auc:.6f}, CAT={oof_cat_auc:.6f}"
)
print(
    f"  Val AUC (held-out):   LGB={val_lgb_auc:.6f}, LGB2={val_lgb2_auc:.6f}, XGB={val_xgb_auc:.6f}, CAT={val_cat_auc:.6f}"
)
print(f"  Weighted Ensemble Val AUC: {val_weighted_auc:.6f}")
print(
    f"  Ensemble improvement over best single: {val_weighted_auc - max(val_lgb_auc, val_lgb2_auc, val_xgb_auc, val_cat_auc):+.6f}"
)

score = val_weighted_auc
print(f"Final Validation Score: {score}")
