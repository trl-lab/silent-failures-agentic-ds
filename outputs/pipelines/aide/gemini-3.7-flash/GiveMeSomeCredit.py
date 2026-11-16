import os
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb

warnings.filterwarnings("ignore")

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "SeriousDlqin2yrs"


def engineer_features(df):
    df = df.copy()

    # Clean age anomaly (0 years old)
    median_age = df["age"].replace(0, np.nan).median()
    df["age"] = df["age"].replace(0, median_age)

    # Missingness indicators
    df["MonthlyIncome_isna"] = df["MonthlyIncome"].isna().astype(int)
    df["NumberOfDependents_isna"] = df["NumberOfDependents"].isna().astype(int)

    # Past due 96/98 coding anomalies (represents unknown/unreported)
    past_due_cols = [
        "NumberOfTime30-59DaysPastDueNotWorse",
        "NumberOfTime60-89DaysPastDueNotWorse",
        "NumberOfTimes90DaysLate",
    ]
    for col in past_due_cols:
        df[f"{col}_is_special"] = df[col].isin([96, 98]).astype(int)

    # Clean past due counts (zero out 96/98 for arithmetic features)
    clean_30_59 = df["NumberOfTime30-59DaysPastDueNotWorse"].replace([96, 98], 0)
    clean_60_89 = df["NumberOfTime60-89DaysPastDueNotWorse"].replace([96, 98], 0)
    clean_90 = df["NumberOfTimes90DaysLate"].replace([96, 98], 0)

    # Delinquency aggregate features & weighted severity score
    df["TotalPastDue"] = clean_30_59 + clean_60_89 + clean_90
    df["HasPastDue"] = (df["TotalPastDue"] > 0).astype(int)
    df["WeightedDelinquencyScore"] = (
        1.0 * clean_30_59 + 2.0 * clean_60_89 + 3.0 * clean_90
    )
    df["SevereDelinquencyRatio"] = clean_90 / (df["TotalPastDue"] + 1e-5)

    # Household and Per Capita features
    df["HouseholdSize"] = df["NumberOfDependents"].fillna(0) + 1

    # Debt and Income feature reconstruction
    # In this dataset, when MonthlyIncome is missing, DebtRatio represents total monthly debt payment
    df["MonthlyDebt"] = np.where(
        df["MonthlyIncome"].notna(),
        df["DebtRatio"] * df["MonthlyIncome"],
        df["DebtRatio"],
    )

    df["IncomePerPerson"] = df["MonthlyIncome"] / df["HouseholdSize"]
    df["DebtPerPerson"] = df["MonthlyDebt"] / df["HouseholdSize"]
    df["NetIncome"] = df["MonthlyIncome"] - df["MonthlyDebt"]

    # Log transformations for heavily skewed distributions
    df["Log_MonthlyIncome"] = np.log1p(df["MonthlyIncome"].fillna(0).clip(lower=0))
    df["Log_MonthlyDebt"] = np.log1p(df["MonthlyDebt"].clip(lower=0))
    df["Log_DebtRatio"] = np.log1p(df["DebtRatio"].clip(lower=0))
    df["Log_RevolvingUtilization"] = np.log1p(
        df["RevolvingUtilizationOfUnsecuredLines"].clip(lower=0, upper=100)
    )

    # Utilization status indicators
    df["RevolvingUtil_Over1"] = (
        df["RevolvingUtilizationOfUnsecuredLines"] > 1.0
    ).astype(int)
    df["RevolvingUtil_IsZero"] = (
        df["RevolvingUtilizationOfUnsecuredLines"] == 0
    ).astype(int)

    # Credit lines Breakdown & Ratios
    df["NonRealEstateLines"] = (
        df["NumberOfOpenCreditLinesAndLoans"] - df["NumberRealEstateLoansOrLines"]
    ).clip(lower=0)
    df["RealEstateRatio"] = df["NumberRealEstateLoansOrLines"] / (
        df["NumberOfOpenCreditLinesAndLoans"] + 1e-5
    )
    df["LinesPerAge"] = df["NumberOfOpenCreditLinesAndLoans"] / df["age"]
    df["RealEstatePerAge"] = df["NumberRealEstateLoansOrLines"] / df["age"]

    # Age binnings
    df["IsYoung"] = (df["age"] < 25).astype(int)
    df["IsSenior"] = (df["age"] >= 65).astype(int)

    return df


# Prepare datasets
X_train = engineer_features(train_df.drop(columns=[target_col], errors="ignore"))
y_train = train_df[target_col].values
X_test = engineer_features(test_df.drop(columns=[target_col], errors="ignore"))

features = [c for c in X_train.columns if c not in ["Unnamed: 0", target_col]]

# LightGBM Model Hyperparameters
lgb_params = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "learning_rate": 0.025,
    "num_leaves": 31,
    "max_depth": 6,
    "min_child_samples": 60,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.75,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "n_jobs": -1,
    "verbose": -1,
}

# 5-fold Stratified Cross-Validation with Multi-Seed Averaging
n_splits = 5
seeds = [42, 2024]
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_preds = np.zeros(len(train_df))
test_preds = np.zeros(len(test_df))

for seed in seeds:
    lgb_params["random_state"] = seed
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_tr, y_tr = X_train.iloc[train_idx][features], y_train[train_idx]
        X_va, y_va = X_train.iloc[val_idx][features], y_train[val_idx]

        dtrain = lgb.Dataset(X_tr, label=y_tr)
        dval = lgb.Dataset(X_va, label=y_va, reference=dtrain)

        model = lgb.train(
            lgb_params,
            dtrain,
            num_boost_round=2000,
            valid_sets=[dtrain, dval],
            callbacks=[
                lgb.early_stopping(stopping_rounds=60, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )

        oof_preds[val_idx] += model.predict(
            X_va, num_iteration=model.best_iteration
        ) / len(seeds)
        test_preds += model.predict(
            X_test[features], num_iteration=model.best_iteration
        ) / (n_splits * len(seeds))

# Evaluate CV ROC-AUC
cv_roc_auc = roc_auc_score(y_train, oof_preds)
print(f"5-Fold CV ROC-AUC: {cv_roc_auc:.5f}")

# Save submission
os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame(
    {"id": np.arange(len(test_df)), "SeriousDlqin2yrs": test_preds}
)
submission.to_csv("./working/submission.csv", index=False)
print("Submission successfully saved to ./working/submission.csv")
