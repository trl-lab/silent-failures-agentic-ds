import os
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.ensemble import HistGradientBoostingRegressor, ExtraTreesRegressor, RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge, RidgeCV
from scipy.optimize import minimize
import lightgbm as lgb
import xgboost as xgb

try:
    import catboost as cb

    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

warnings.filterwarnings("ignore")

# File paths and configuration
TRAIN_PATH = "./input/train.parquet"
TEST_PATH = "./input/test.parquet"
SUBMISSION_PATH = "./submission/submission.csv"
WORKING_DIR = "./working"
TARGET_COL = "Original Principal Amount (US$)"
SEED = 42

os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# 1. Load Datasets
train_df = pd.read_parquet(TRAIN_PATH)
test_df = pd.read_parquet(TEST_PATH)

# Handle ID columns
if "id" not in test_df.columns:
    test_df["id"] = np.arange(len(test_df))
if "id" not in train_df.columns:
    train_df["id"] = np.arange(len(train_df))

# 2. Cross-Validation Split Creation (Strict Isolation)
kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
train_df["fold"] = -1
for fold, (tr_idx, va_idx) in enumerate(kf.split(train_df)):
    train_df.loc[va_idx, "fold"] = fold

# Identify non-target feature columns
feature_cols = [c for c in train_df.columns if c not in [TARGET_COL, "fold", "id"]]
n_train = len(train_df)
n_test = len(test_df)
n_folds = 5

# Combine features for structure alignment
combined = pd.concat(
    [train_df[feature_cols], test_df[feature_cols]], axis=0, ignore_index=True
)

# 3. Datetime & Temporal Duration Feature Engineering
date_cols = []
for c in feature_cols:
    if pd.api.types.is_datetime64_any_dtype(combined[c]):
        date_cols.append(c)
    elif "date" in c.lower() or "time" in c.lower():
        converted = pd.to_datetime(combined[c], errors="coerce")
        if converted.notnull().sum() > 0.1 * len(combined):
            combined[c] = converted
            date_cols.append(c)

for c in date_cols:
    combined[f"{c}_year"] = combined[c].dt.year
    combined[f"{c}_month"] = combined[c].dt.month
    combined[f"{c}_day"] = combined[c].dt.day
    combined[f"{c}_dayofweek"] = combined[c].dt.dayofweek
    combined[f"{c}_quarter"] = combined[c].dt.quarter
    combined[f"{c}_dayofyear"] = combined[c].dt.dayofyear

# Compute pairwise temporal duration differences in days
for i in range(len(date_cols)):
    for j in range(i + 1, len(date_cols)):
        c1, c2 = date_cols[i], date_cols[j]
        combined[f"date_diff_{c1}_vs_{c2}"] = (combined[c1] - combined[c2]).dt.days

# Drop raw datetime objects
combined.drop(columns=date_cols, inplace=True)

# 4. Categorical Identification, Frequency Encoding & Group Aggregations
cat_cols = []
for c in combined.columns:
    if not pd.api.types.is_numeric_dtype(combined[c]):
        cat_cols.append(c)

# Construct compound interaction features from top categorical variables
compound_cats = []
top_cats = cat_cols[:4] if len(cat_cols) >= 4 else cat_cols
for i in range(len(top_cats)):
    for j in range(i + 1, len(top_cats)):
        c1, c2 = top_cats[i], top_cats[j]
        comp_col = f"{c1}_x_{c2}"
        combined[comp_col] = combined[c1].astype(str) + "_" + combined[c2].astype(str)
        compound_cats.append(comp_col)

cat_cols = cat_cols + compound_cats

for c in cat_cols:
    combined[c] = combined[c].astype(str).fillna("Missing")
    train_counts = combined.iloc[:n_train][c].value_counts(normalize=True)
    combined[f"{c}_freq"] = combined[c].map(train_counts).fillna(0.0)

# Compute group aggregation statistics for numeric features by primary categoricals
num_cols = [c for c in combined.columns if pd.api.types.is_numeric_dtype(combined[c])]
group_candidates = [c for c in cat_cols if combined[c].nunique() > 1 and combined[c].nunique() <= 500]
if not group_candidates:
    group_candidates = cat_cols[:3]
else:
    group_candidates = group_candidates[:5]

for g_col in group_candidates:
    for n_col in num_cols[:10]:  # aggregate key numeric/temporal features
        grouped = combined.groupby(g_col)[n_col]
        combined[f"{g_col}_{n_col}_mean"] = grouped.transform("mean")
        combined[f"{g_col}_{n_col}_std"] = grouped.transform("std").fillna(0)
        combined[f"{g_col}_{n_col}_min"] = grouped.transform("min")
        combined[f"{g_col}_{n_col}_max"] = grouped.transform("max")

# Separate back into train and test
X_train = combined.iloc[:n_train].copy()
X_test = combined.iloc[n_train:].copy()

X_train[TARGET_COL] = train_df[TARGET_COL].values
X_train["fold"] = train_df["fold"].values
X_train["id"] = train_df["id"].values
X_test["id"] = test_df["id"].values

# 5. Out-of-Fold Target Encoding (m-estimate smoothed, strict isolation principle)
m_factor = 10.0
for c in cat_cols:
    target_enc_col = f"{c}_target_enc"
    X_train[target_enc_col] = np.nan
    X_test[target_enc_col] = 0.0

    global_mean = X_train[TARGET_COL].mean()

    for fold_idx in range(n_folds):
        tr_mask = X_train["fold"] != fold_idx
        va_mask = X_train["fold"] == fold_idx

        tr_stats = X_train[tr_mask].groupby(c, observed=False)[TARGET_COL].agg(["count", "mean"])
        smoothed = (tr_stats["count"] * tr_stats["mean"] + m_factor * global_mean) / (tr_stats["count"] + m_factor)
        X_train.loc[va_mask, target_enc_col] = (
            X_train.loc[va_mask, c].map(smoothed).fillna(global_mean)
        )

    full_stats = X_train.groupby(c, observed=False)[TARGET_COL].agg(["count", "mean"])
    smoothed_full = (full_stats["count"] * full_stats["mean"] + m_factor * global_mean) / (full_stats["count"] + m_factor)
    X_test[target_enc_col] = X_test[c].map(smoothed_full).fillna(global_mean)

# 6. Format Categorical Features for Model Families
model_feature_cols = [c for c in X_train.columns if c not in [TARGET_COL, "fold", "id"]]

# LightGBM Feature Matrices
X_lgb_train = X_train[model_feature_cols].copy()
X_lgb_test = X_test[model_feature_cols].copy()
for col in cat_cols:
    X_lgb_train[col] = X_lgb_train[col].astype("category")
    X_lgb_test[col] = X_lgb_test[col].astype("category")

# XGBoost / Numeric Feature Matrices
X_num_train = X_train[model_feature_cols].copy()
X_num_test = X_test[model_feature_cols].copy()

for col in cat_cols:
    categories = pd.concat(
        [X_num_train[col].astype(str), X_num_test[col].astype(str)]
    ).unique()
    cat_map = {cat: idx for idx, cat in enumerate(categories)}
    X_num_train[col] = X_num_train[col].astype(str).map(cat_map).fillna(-1)
    X_num_test[col] = X_num_test[col].astype(str).map(cat_map).fillna(-1)

for col in X_num_train.columns:
    X_num_train[col] = pd.to_numeric(X_num_train[col], errors="coerce").fillna(0.0)
    X_num_test[col] = pd.to_numeric(X_num_test[col], errors="coerce").fillna(0.0)

# 7. Model Training & Validation Prediction Containers
y_true = X_train[TARGET_COL].values

oof_m1 = np.zeros(n_train)  # LightGBM Raw Target (Tuned Regularization)
oof_m2 = np.zeros(n_train)  # XGBoost Raw Target (Tuned Regularization)
oof_m3 = np.zeros(n_train)  # CatBoost / HistGB Raw Target
oof_m4 = np.zeros(n_train)  # LightGBM Log Target
oof_m5 = np.zeros(n_train)  # ExtraTrees Regressor

test_m1 = np.zeros(n_test)
test_m2 = np.zeros(n_test)
test_m3 = np.zeros(n_test)
test_m4 = np.zeros(n_test)
test_m5 = np.zeros(n_test)

# 8. Cross-Validation Model Training Loop
for fold in range(n_folds):
    tr_idx = X_train["fold"] != fold
    va_idx = X_train["fold"] == fold

    y_tr = y_true[tr_idx]
    y_va = y_true[va_idx]

    # Model 1: LightGBM (Raw Scale Target with L1/L2 Regularization)
    m1 = lgb.LGBMRegressor(
        n_estimators=1500,
        learning_rate=0.03,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=SEED + fold,
        n_jobs=-1,
    )
    m1.fit(
        X_lgb_train.loc[tr_idx],
        y_tr,
        eval_set=[(X_lgb_train.loc[va_idx], y_va)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )
    oof_m1[va_idx] = m1.predict(X_lgb_train.loc[va_idx])
    test_m1 += m1.predict(X_lgb_test) / n_folds

    # Model 2: XGBoost (Raw Scale Target with L1/L2 Regularization)
    m2 = xgb.XGBRegressor(
        n_estimators=1500,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        tree_method="hist",
        random_state=SEED + fold,
        early_stopping_rounds=50,
        n_jobs=-1,
    )
    m2.fit(
        X_num_train.loc[tr_idx],
        y_tr,
        eval_set=[(X_num_train.loc[va_idx], y_va)],
        verbose=False,
    )
    oof_m2[va_idx] = m2.predict(X_num_train.loc[va_idx])
    test_m2 += m2.predict(X_num_test) / n_folds

    # Model 3: CatBoost / HistGradientBoosting (Raw Scale Target)
    if HAS_CATBOOST:
        m3 = cb.CatBoostRegressor(
            iterations=1500,
            learning_rate=0.03,
            depth=6,
            l2_leaf_reg=3.0,
            random_seed=SEED + fold,
            early_stopping_rounds=50,
            verbose=0,
        )
        m3.fit(
            X_num_train.loc[tr_idx],
            y_tr,
            eval_set=(X_num_train.loc[va_idx], y_va),
        )
        oof_m3[va_idx] = m3.predict(X_num_train.loc[va_idx])
        test_m3 += m3.predict(X_num_test) / n_folds
    else:
        m3 = HistGradientBoostingRegressor(
            max_iter=500,
            learning_rate=0.03,
            max_depth=8,
            l2_regularization=1.0,
            random_state=SEED + fold,
        )
        m3.fit(X_num_train.loc[tr_idx], y_tr)
        oof_m3[va_idx] = m3.predict(X_num_train.loc[va_idx])
        test_m3 += m3.predict(X_num_test) / n_folds

    # Model 4: LightGBM (Log-Transformed Target)
    y_tr_log = np.log1p(np.maximum(0, y_tr))
    y_va_log = np.log1p(np.maximum(0, y_va))

    m4 = lgb.LGBMRegressor(
        n_estimators=1500,
        learning_rate=0.03,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=SEED + fold + 100,
        n_jobs=-1,
    )
    m4.fit(
        X_lgb_train.loc[tr_idx],
        y_tr_log,
        eval_set=[(X_lgb_train.loc[va_idx], y_va_log)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )
    oof_m4[va_idx] = np.expm1(np.maximum(0, m4.predict(X_lgb_train.loc[va_idx])))
    test_m4 += np.expm1(np.maximum(0, m4.predict(X_lgb_test))) / n_folds

    # Model 5: ExtraTreesRegressor (Diverse Tree Paradigm)
    m5 = ExtraTreesRegressor(
        n_estimators=300,
        max_depth=12,
        min_samples_split=5,
        random_state=SEED + fold,
        n_jobs=-1,
    )
    m5.fit(X_num_train.loc[tr_idx], y_tr)
    oof_m5[va_idx] = m5.predict(X_num_train.loc[va_idx])
    test_m5 += m5.predict(X_num_test) / n_folds

# Non-negativity constraint on raw component outputs
oof_m1, test_m1 = np.maximum(0, oof_m1), np.maximum(0, test_m1)
oof_m2, test_m2 = np.maximum(0, oof_m2), np.maximum(0, test_m2)
oof_m3, test_m3 = np.maximum(0, oof_m3), np.maximum(0, test_m3)
oof_m4, test_m4 = np.maximum(0, oof_m4), np.maximum(0, test_m4)
oof_m5, test_m5 = np.maximum(0, oof_m5), np.maximum(0, test_m5)


# 9. Out-of-Fold Constrained SLSQP Ensemble Optimization
def rmse_objective(weights):
    w = weights / np.sum(weights)
    blend_oof = w[0] * oof_m1 + w[1] * oof_m2 + w[2] * oof_m3 + w[3] * oof_m4 + w[4] * oof_m5
    return np.sqrt(np.mean((y_true - blend_oof) ** 2))


init_weights = [0.2, 0.2, 0.2, 0.2, 0.2]
bounds = [(0, 1)] * 5
opt_res = minimize(
    rmse_objective,
    init_weights,
    bounds=bounds,
    method="SLSQP",
    options={"maxiter": 500},
)
opt_w = opt_res.x / np.sum(opt_res.x)

# 10. Generate Final Predictions and Evaluate Final Score
final_oof = (
    opt_w[0] * oof_m1 + opt_w[1] * oof_m2 + opt_w[2] * oof_m3 + opt_w[3] * oof_m4 + opt_w[4] * oof_m5
)
final_oof = np.maximum(0, final_oof)
final_rmse = np.sqrt(np.mean((y_true - final_oof) ** 2))

final_test = (
    opt_w[0] * test_m1 + opt_w[1] * test_m2 + opt_w[2] * test_m3 + opt_w[3] * test_m4 + opt_w[4] * test_m5
)
final_test = np.maximum(0, final_test)

# 11. Save Submission File
submission_df = pd.DataFrame(
    {"id": test_df["id"].values, "Original Principal Amount (US$)": final_test}
)
submission_df.to_csv(SUBMISSION_PATH, index=False)

print(f"Final Validation Score: {final_rmse}")