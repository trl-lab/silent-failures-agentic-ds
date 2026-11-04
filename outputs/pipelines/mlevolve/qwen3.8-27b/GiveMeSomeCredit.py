import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import os
import joblib

# Create directories
os.makedirs("./submission", exist_ok=True)
os.makedirs("./working", exist_ok=True)

# ============================================================
# 1. LOAD DATA
# ============================================================
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")
print(f"Train columns: {list(train.columns)}")
print(f"Target distribution:\n{train['SeriousDlqin2yrs'].value_counts(normalize=True)}")

# ============================================================
# 2. CLEANING
# ============================================================
# Drop the index column if present
if "Unnamed: 0" in train.columns:
    train = train.drop(columns=["Unnamed: 0"])
if "Unnamed: 0" in test.columns:
    test = test.drop(columns=["Unnamed: 0"])

# Separate target
y_train = train["SeriousDlqin2yrs"].copy()
train = train.drop(columns=["SeriousDlqin2yrs"])
if "SeriousDlqin2yrs" in test.columns:
    test = test.drop(columns=["SeriousDlqin2yrs"])

# Track missing patterns BEFORE imputation
mi_nan_train = train["MonthlyIncome"].isna().values
nd_nan_train = train["NumberOfDependents"].isna().values
mi_nan_test = test["MonthlyIncome"].isna().values
nd_nan_test = test["NumberOfDependents"].isna().values

# Combine for consistent feature engineering
train["is_train"] = 1
test["is_train"] = 0
combined = pd.concat([train, test], axis=0, ignore_index=True)

# ============================================================
# 3. OUTLIER HANDLING (fit parameters from TRAIN only)
# ============================================================
train_mask = combined["is_train"] == 1

# --- DebtRatio: keep log transform only (tree models handle outliers natively) ---
combined["DebtRatio_log"] = np.log1p(combined["DebtRatio"].clip(lower=0))

# --- RevolvingUtilizationOfUnsecuredLines: keep log transform only ---
combined["RevolvingUtil_log"] = np.log1p(
    combined["RevolvingUtilizationOfUnsecuredLines"].clip(lower=0)
)

# --- MonthlyIncome: cap extreme values ---
mi_train = combined.loc[train_mask, "MonthlyIncome"].dropna().values
mi_cap = np.percentile(mi_train, 99)
combined["MonthlyIncome"] = combined["MonthlyIncome"].clip(upper=mi_cap)

# --- Age: fix impossible values ---
age_train = combined.loc[train_mask, "age"].values
age_min = max(18, np.percentile(age_train, 0.5))
age_max = min(100, np.percentile(age_train, 99.5))
combined["age"] = combined["age"].clip(age_min, age_max)

# ============================================================
# 4. MISSING VALUE HANDLING (fit from TRAIN only)
# ============================================================
# MonthlyIncome: median imputation from train
mi_median = combined.loc[train_mask, "MonthlyIncome"].median()
combined["MonthlyIncome"] = combined["MonthlyIncome"].fillna(mi_median)
combined["MonthlyIncome_log"] = np.log1p(combined["MonthlyIncome"])

# NumberOfDependents: median imputation from train
nd_median = combined.loc[train_mask, "NumberOfDependents"].median()
combined["NumberOfDependents"] = combined["NumberOfDependents"].fillna(nd_median)

# Missing indicators (from tracked patterns)
combined["MonthlyIncome_missing"] = np.concatenate([mi_nan_train, mi_nan_test]).astype(
    int
)
combined["NumberOfDependents_missing"] = np.concatenate(
    [nd_nan_train, nd_nan_test]
).astype(int)

# ============================================================
# 5. FEATURE ENGINEERING
# ============================================================

# --- Credit Risk Composite Features ---
combined["TotalPastDue"] = (
    combined["NumberOfTime30-59DaysPastDueNotWorse"]
    + combined["NumberOfTime60-89DaysPastDueNotWorse"]
    + combined["NumberOfTimes90DaysLate"]
)

combined["SeverePastDue"] = (
    combined["NumberOfTime60-89DaysPastDueNotWorse"]
    + combined["NumberOfTimes90DaysLate"]
)

combined["PastDuePerCreditLine"] = combined["TotalPastDue"] / (
    combined["NumberOfOpenCreditLinesAndLoans"] + 1
)

combined["SeverePastDuePerCreditLine"] = combined["SeverePastDue"] / (
    combined["NumberOfOpenCreditLinesAndLoans"] + 1
)

# --- Income-Related Features ---
combined["IncomePerDependent"] = combined["MonthlyIncome"] / (
    combined["NumberOfDependents"] + 1
)

combined["DebtToIncome"] = combined["DebtRatio"] / (
    combined["MonthlyIncome"] / 1000 + 1
)

# --- Credit Utilization Interactions ---
combined["UtilDebtInteraction"] = (
    combined["RevolvingUtilizationOfUnsecuredLines"] * combined["DebtRatio"]
)

combined["UtilPerCreditLine"] = combined["RevolvingUtilizationOfUnsecuredLines"] / (
    combined["NumberOfOpenCreditLinesAndLoans"] + 1
)

# --- Age-Related Features ---
combined["AgePastDue"] = combined["age"] * combined["TotalPastDue"]
combined["AgeIncome"] = combined["age"] * combined["MonthlyIncome_log"]

# --- Credit Line Diversity ---
combined["RealEstateFraction"] = combined["NumberRealEstateLoansOrLines"] / (
    combined["NumberOfOpenCreditLinesAndLoans"] + 1
)

# --- Log transforms for skewed features ---
combined["NumberOfOpenCreditLines_log"] = np.log1p(
    combined["NumberOfOpenCreditLinesAndLoans"]
)
combined["NumberOfTime30-59_log"] = np.log1p(
    combined["NumberOfTime30-59DaysPastDueNotWorse"]
)
combined["NumberOfTime60-89_log"] = np.log1p(
    combined["NumberOfTime60-89DaysPastDueNotWorse"]
)
combined["NumberOfTimes90_log"] = np.log1p(combined["NumberOfTimes90DaysLate"])
combined["NumberRealEstate_log"] = np.log1p(combined["NumberRealEstateLoansOrLines"])
combined["TotalPastDue_log"] = np.log1p(combined["TotalPastDue"])
combined["SeverePastDue_log"] = np.log1p(combined["SeverePastDue"])

# --- Squared features for non-linear patterns ---
combined["DebtRatio_sq"] = combined["DebtRatio"] ** 2
combined["RevolvingUtil_sq"] = combined["RevolvingUtilizationOfUnsecuredLines"] ** 2

# --- Binned features ---
combined["AgeBin"] = pd.cut(
    combined["age"], bins=[17, 25, 35, 45, 55, 65, 100], labels=False
)
# Fit qcut bin edges from TRAIN only to avoid leakage
income_train_vals = combined.loc[train_mask, "MonthlyIncome"]
income_bin_edges = pd.qcut(income_train_vals, q=5, retbins=True, duplicates="drop")[1]
combined["IncomeBin"] = pd.cut(
    combined["MonthlyIncome"], bins=income_bin_edges, labels=False
)

# --- Domain-specific credit risk features ---
# (1) PastDueSeverityRatio: captures severity of delinquency
combined["PastDueSeverityRatio"] = combined["SeverePastDue"] / (
    combined["TotalPastDue"] + 1
)

# (2) IncomeToDebtRatio: cleaner debt-to-income proxy
combined["IncomeToDebtRatio"] = combined["MonthlyIncome"] / (
    combined["DebtRatio"] * combined["MonthlyIncome"] / 100 + 1
)

# (3) CreditLineAgeInteraction: credit history depth
combined["CreditLineAgeInteraction"] = (
    combined["NumberOfOpenCreditLinesAndLoans"] * combined["age"] / 100
)

# (4) RevolvingUtilBucketed: non-linear risk jump in utilization
combined["RevolvingUtilBucketed"] = pd.cut(
    combined["RevolvingUtilizationOfUnsecuredLines"],
    bins=[-0.001, 0.1, 0.3, 0.5, 1.0, 3.0, np.inf],
    labels=False,
)

# (5) PastDueVelocity: delinquency rate over life
combined["PastDueVelocity"] = combined["NumberOfTimes90DaysLate"] / (
    combined["age"] - 18 + 1
)

# ============================================================
# 6. SPLIT BACK INTO TRAIN / VAL / TEST
# ============================================================
combined = combined.drop(columns=["is_train"])

n_train = len(y_train)
train_processed = combined.iloc[:n_train].copy()
test_processed = combined.iloc[n_train:].copy()

# Add target back to train
train_processed["SeriousDlqin2yrs"] = y_train.values

# Stratified train/val split
X_all = train_processed.drop(columns=["SeriousDlqin2yrs"])
y_all = train_processed["SeriousDlqin2yrs"]

X_train, X_val, y_train_split, y_val_split = train_test_split(
    X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
)

print(f"\nX_train shape: {X_train.shape}")
print(f"X_val shape: {X_val.shape}")
print(f"X_test shape: {test_processed.shape}")
print(f"Train target rate: {y_train_split.mean():.4f}")
print(f"Val target rate: {y_val_split.mean():.4f}")

# ============================================================
# 7. MODEL PARAMETERS (Ensemble of 4 diverse models)
# ============================================================
n_pos = int(y_train_split.sum())
n_neg = len(y_train_split) - n_pos
print(f"Class balance: {n_pos} pos / {n_neg} neg (ratio {n_neg/n_pos:.2f})")

# Model 1: LightGBM - regularized (num_leaves=31, lr=0.01, min_child_samples=200)
lgb_params_1 = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "learning_rate": 0.01,
    "num_leaves": 31,
    "max_depth": 5,
    "min_child_samples": 200,
    "feature_fraction": 0.6,
    "bagging_fraction": 0.6,
    "bagging_freq": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 10.0,
    "verbose": -1,
    "random_state": 42,
    "n_jobs": -1,
}

# Model 2: LightGBM - moderate (num_leaves=63, lr=0.01, min_child_samples=100)
lgb_params_2 = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "learning_rate": 0.01,
    "num_leaves": 63,
    "max_depth": 7,
    "min_child_samples": 100,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.7,
    "bagging_freq": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 5.0,
    "verbose": -1,
    "random_state": 42,
    "n_jobs": -1,
}

# Model 3: XGBoost - regularized (max_depth=4, lr=0.01, subsample=0.7)
xgb_params = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "max_depth": 4,
    "subsample": 0.7,
    "colsample_bytree": 0.6,
    "learning_rate": 0.01,
    "reg_alpha": 0.1,
    "reg_lambda": 5.0,
    "random_state": 42,
    "nthread": -1,
    "tree_method": "hist",
}

# Model 4: CatBoost - different inductive bias for diversity
cat_params = {
    "loss_function": "Logloss",
    "learning_rate": 0.01,
    "depth": 6,
    "l2_leaf_reg": 5.0,
    "random_seed": 42,
    "verbose": 0,
}

# ============================================================
# 8. TRAINING ENSEMBLE WITH EARLY STOPPING ON VALIDATION AUC
# ============================================================
# --- Model 1: LightGBM (num_leaves=31, lr=0.01, min_child_samples=200) ---
train_data_1 = lgb.Dataset(X_train.values, label=y_train_split.values)
val_data_1 = lgb.Dataset(X_val.values, label=y_val_split.values, reference=train_data_1)

model_1 = lgb.train(
    lgb_params_1,
    train_data_1,
    num_boost_round=5000,
    valid_sets=[val_data_1],
    valid_names=["validation"],
    callbacks=[
        lgb.early_stopping(stopping_rounds=200, verbose=False),
        lgb.log_evaluation(period=500),
    ],
)
best_iter_1 = model_1.best_iteration
val_auc_1 = model_1.best_score["validation"]["auc"]
print(f"\nModel 1 (LGB leaves=31, lr=0.01): Best iter={best_iter_1}, Val AUC={val_auc_1:.6f}")

# --- Model 2: LightGBM (num_leaves=63, lr=0.01, min_child_samples=100) ---
train_data_2 = lgb.Dataset(X_train.values, label=y_train_split.values)
val_data_2 = lgb.Dataset(X_val.values, label=y_val_split.values, reference=train_data_2)

model_2 = lgb.train(
    lgb_params_2,
    train_data_2,
    num_boost_round=5000,
    valid_sets=[val_data_2],
    valid_names=["validation"],
    callbacks=[
        lgb.early_stopping(stopping_rounds=200, verbose=False),
        lgb.log_evaluation(period=500),
    ],
)
best_iter_2 = model_2.best_iteration
val_auc_2 = model_2.best_score["validation"]["auc"]
print(f"Model 2 (LGB leaves=63, lr=0.01): Best iter={best_iter_2}, Val AUC={val_auc_2:.6f}")

# --- Model 3: XGBoost (max_depth=4, lr=0.01, subsample=0.7) ---
dtrain = xgb.DMatrix(X_train.values, label=y_train_split.values)
dval = xgb.DMatrix(X_val.values, label=y_val_split.values)

evals_result = {}
model_3 = xgb.train(
    xgb_params,
    dtrain,
    num_boost_round=5000,
    evals=[(dval, "validation")],
    early_stopping_rounds=200,
    verbose_eval=500,
    evals_result=evals_result,
)
best_iter_3 = model_3.best_iteration
val_auc_3 = min(evals_result["validation"]["auc"])
print(f"Model 3 (XGB depth=4, lr=0.01): Best iter={best_iter_3}, Val AUC={val_auc_3:.6f}")

# --- Model 4: CatBoost (depth=6, lr=0.01) ---
model_4 = CatBoostClassifier(**cat_params)
model_4.fit(
    X_train,
    y_train_split,
    eval_set=(X_val, y_val_split),
    early_stopping_rounds=200,
    verbose=500,
)
best_iter_4 = model_4.get_best_iteration()
val_preds_4_tmp = model_4.predict_proba(X_val)[:, 1]
val_auc_4 = roc_auc_score(y_val_split.values, val_preds_4_tmp)
print(f"Model 4 (CatBoost depth=6, lr=0.01): Best iter={best_iter_4}, Val AUC={val_auc_4:.6f}")

# ============================================================
# 9. WEIGHTED ENSEMBLE PREDICTIONS (Validation)
# ============================================================
# Compute validation AUC for each model individually
val_preds_1 = model_1.predict(X_val.values, num_iteration=best_iter_1)
val_preds_2 = model_2.predict(X_val.values, num_iteration=best_iter_2)
val_preds_3 = model_3.predict(dval, iteration_range=(0, best_iter_3))
val_preds_4 = model_4.predict_proba(X_val)[:, 1]

val_auc_1_check = roc_auc_score(y_val_split.values, val_preds_1)
val_auc_2_check = roc_auc_score(y_val_split.values, val_preds_2)
val_auc_3_check = roc_auc_score(y_val_split.values, val_preds_3)
val_auc_4_check = roc_auc_score(y_val_split.values, val_preds_4)
print(f"\nIndividual Val AUCs: M1={val_auc_1_check:.6f}, M2={val_auc_2_check:.6f}, M3={val_auc_3_check:.6f}, M4={val_auc_4_check:.6f}")

# Weights proportional to validation AUC
aucs = np.array([val_auc_1_check, val_auc_2_check, val_auc_3_check, val_auc_4_check])
weights = aucs / aucs.sum()
print(f"Ensemble weights: {weights[0]:.4f}, {weights[1]:.4f}, {weights[2]:.4f}, {weights[3]:.4f}")

# Weighted ensemble validation prediction
val_preds_ensemble = (
    weights[0] * val_preds_1
    + weights[1] * val_preds_2
    + weights[2] * val_preds_3
    + weights[3] * val_preds_4
)
val_auc = roc_auc_score(y_val_split.values, val_preds_ensemble)
print(f"Ensemble Validation ROC-AUC: {val_auc:.6f}")

# ============================================================
# 10. RETRAIN ON FULL DATA AND PREDICT TEST
# ============================================================
# Use average of best iterations (capped at 5000)
avg_iter = int(np.mean([best_iter_1, best_iter_2, best_iter_3, best_iter_4]))
print(f"\nAverage best iteration for full-data retraining: {avg_iter}")

# Full training data
X_full = X_all.values
y_full = y_all.values

# --- Retrain Model 1 on full data ---
train_data_full_1 = lgb.Dataset(X_full, label=y_full)
model_1_full = lgb.train(
    lgb_params_1,
    train_data_full_1,
    num_boost_round=avg_iter,
    callbacks=[lgb.log_evaluation(period=500)],
)

# --- Retrain Model 2 on full data ---
train_data_full_2 = lgb.Dataset(X_full, label=y_full)
model_2_full = lgb.train(
    lgb_params_2,
    train_data_full_2,
    num_boost_round=avg_iter,
    callbacks=[lgb.log_evaluation(period=500)],
)

# --- Retrain Model 3 on full data ---
dtrain_full = xgb.DMatrix(X_full, label=y_full)
model_3_full = xgb.train(
    xgb_params,
    dtrain_full,
    num_boost_round=avg_iter,
    verbose_eval=500,
)

# --- Retrain Model 4 (CatBoost) on full data ---
model_4_full = CatBoostClassifier(**cat_params)
model_4_full.fit(X_full, y_full, verbose=500)

# Test predictions from full-data models
test_preds_1 = model_1_full.predict(test_processed.values)
test_preds_2 = model_2_full.predict(test_processed.values)
test_preds_3 = model_3_full.predict(xgb.DMatrix(test_processed.values))
test_preds_4 = model_4_full.predict_proba(test_processed)[:, 1]

# Weighted ensemble test prediction
test_preds = (
    weights[0] * test_preds_1
    + weights[1] * test_preds_2
    + weights[2] * test_preds_3
    + weights[3] * test_preds_4
)

# ============================================================
# 11. FEATURE IMPORTANCE (top 15) from Model 1
# ============================================================
importance = pd.DataFrame(
    {
        "feature": X_train.columns,
        "importance": model_1.feature_importance(importance_type="gain"),
    }
).sort_values("importance", ascending=False)

print("\nTop 15 features by gain (Model 1):")
for i, row in importance.head(15).iterrows():
    print(f"  {row['feature']}: {row['importance']:.1f}")

# Also show CatBoost feature importance
cat_imp = pd.DataFrame(
    {
        "feature": X_train.columns,
        "importance": model_4.get_feature_importance(),
    }
).sort_values("importance", ascending=False)

print("\nTop 15 features by importance (CatBoost):")
for i, row in cat_imp.head(15).iterrows():
    print(f"  {row['feature']}: {row['importance']:.1f}")

# ============================================================
# 12. SAVE SUBMISSION
# ============================================================
submission = pd.DataFrame(
    {"id": range(len(test_preds)), "SeriousDlqin2yrs": test_preds}
)
submission.to_csv("./submission/submission.csv", index=False)
print(f"\nSubmission saved: {submission.shape}")
print(
    f"Prediction stats: mean={test_preds.mean():.4f}, min={test_preds.min():.6f}, max={test_preds.max():.6f}"
)

# ============================================================
# 13. FINAL SCORE
# ============================================================
score = val_auc
print(f"Final Validation Score: {score}")