import os
import warnings
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

# 1. Load Data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "SeriousDlqin2yrs"


# 2. Feature Engineering
def engineer_features(df):
    data = df.copy()

    # Clean invalid age values
    data["age"] = data["age"].replace(0, np.nan)

    # Flag missing values explicitly
    data["MonthlyIncome_isnull"] = data["MonthlyIncome"].isna().astype(int)
    data["NumberOfDependents_isnull"] = data["NumberOfDependents"].isna().astype(int)

    # Special delinquency code flags (96, 98)
    delinq_cols = [
        "NumberOfTime30-59DaysPastDueNotWorse",
        "NumberOfTime60-89DaysPastDueNotWorse",
        "NumberOfTimes90DaysLate",
    ]
    data["has_delinquency_96_98"] = 0
    for col in delinq_cols:
        data["has_delinquency_96_98"] = data["has_delinquency_96_98"] | (
            data[col].isin([96, 98])
        )
        data[f"{col}_clean"] = data[col].apply(lambda x: np.nan if x >= 96 else x)

    # Weighted delinquency index
    data["total_delinquencies"] = (
        data["NumberOfTime30-59DaysPastDueNotWorse_clean"]
        + data["NumberOfTime60-89DaysPastDueNotWorse_clean"]
        + data["NumberOfTimes90DaysLate_clean"]
    )
    data["weighted_delinquency"] = (
        1 * data["NumberOfTime30-59DaysPastDueNotWorse_clean"]
        + 2 * data["NumberOfTime60-89DaysPastDueNotWorse_clean"]
        + 3 * data["NumberOfTimes90DaysLate_clean"]
    )

    # Debt and Income feature engineering
    # When MonthlyIncome is missing/zero, DebtRatio is absolute monthly debt
    data["EstimatedMonthlyDebt"] = np.where(
        data["MonthlyIncome"].isna() | (data["MonthlyIncome"] == 0),
        data["DebtRatio"],
        data["DebtRatio"] * data["MonthlyIncome"],
    )

    deps = data["NumberOfDependents"].fillna(0)
    data["DisposableIncome"] = data["MonthlyIncome"] - data["EstimatedMonthlyDebt"]
    data["IncomePerPerson"] = data["MonthlyIncome"] / (deps + 1)
    data["DebtPerPerson"] = data["EstimatedMonthlyDebt"] / (deps + 1)

    # Credit lines and utilization
    data["TotalCreditLines"] = (
        data["NumberOfOpenCreditLinesAndLoans"] + data["NumberRealEstateLoansOrLines"]
    )
    data["RealEstateRatio"] = data["NumberRealEstateLoansOrLines"] / (
        data["TotalCreditLines"] + 1e-5
    )
    data["HighUtilizationFlag"] = (
        data["RevolvingUtilizationOfUnsecuredLines"] > 1.0
    ).astype(int)
    data["LogUtilization"] = np.log1p(
        data["RevolvingUtilizationOfUnsecuredLines"].clip(lower=0)
    )

    # Age groups
    data["AgeGroup"] = pd.cut(
        data["age"].fillna(data["age"].median()),
        bins=[0, 30, 45, 60, 75, 120],
        labels=False,
    )

    # Drop non-feature columns if present
    cols_to_drop = [
        col for col in ["Unnamed: 0", "id", target_col] if col in data.columns
    ]
    features = [col for col in data.columns if col not in cols_to_drop]

    return data[features]


X = engineer_features(train_df)
y = train_df[target_col].astype(int).values
X_test = engineer_features(test_df)

# 3. Model Evaluation with 5-Fold Stratified K-Fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train, y_train = X.iloc[train_idx], y[train_idx]
    X_val, y_val = X.iloc[val_idx], y[val_idx]

    model = CatBoostClassifier(
        iterations=1500,
        learning_rate=0.03,
        depth=6,
        eval_metric="AUC",
        random_seed=42 + fold,
        verbose=0,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=(X_val, y_val),
        early_stopping_rounds=100,
        verbose=False,
    )

    oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
    test_preds += model.predict_proba(X_test)[:, 1] / skf.n_splits

cv_auc = roc_auc_score(y, oof_preds)
print(f"Out-of-fold ROC-AUC: {cv_auc:.6f}")

# 4. Save Submission File
os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame(
    {"id": np.arange(len(test_preds)), "SeriousDlqin2yrs": test_preds}
)
submission.to_csv("./working/submission.csv", index=False)
