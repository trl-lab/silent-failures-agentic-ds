import os
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb

warnings.filterwarnings("ignore")

# 1. Load Data
train_path = "./input/train.parquet"
test_path = "./input/test.parquet"

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = "Response"
y = train_df[target_col].values
train_df = train_df.drop(columns=[target_col])

n_train = len(train_df)
combined = pd.concat([train_df, test_df], ignore_index=True)

# 2. Feature Engineering & Preprocessing
drop_cols = [col for col in ["Id", "id"] if col in combined.columns]
if drop_cols:
    combined = combined.drop(columns=drop_cols)

# Process Product_Info_2 categorical variable
if "Product_Info_2" in combined.columns:
    combined["Product_Info_2_char"] = (
        combined["Product_Info_2"].astype(str).str[0].astype("category").cat.codes
    )
    combined["Product_Info_2_num"] = pd.to_numeric(
        combined["Product_Info_2"].astype(str).str[1:], errors="coerce"
    ).fillna(-1)
    combined["Product_Info_2"] = combined["Product_Info_2"].astype("category").cat.codes

# Physiological body metrics & interaction features
eps = 1e-5
combined["BMI_Age"] = combined["BMI"] * combined["Ins_Age"]
combined["Ht_Wt"] = combined["Ht"] * combined["Wt"]
combined["Ht_to_Wt"] = combined["Ht"] / (combined["Wt"] + eps)
combined["BMI_Ht_ratio"] = combined["BMI"] / (combined["Ht"] + eps)
combined["BMI_Wt_ratio"] = combined["BMI"] / (combined["Wt"] + eps)
combined["Age_Ht"] = combined["Ins_Age"] * combined["Ht"]
combined["Age_Wt"] = combined["Ins_Age"] * combined["Wt"]

# Employment & Age interactions
if "Employment_Info_1" in combined.columns and "Ins_Age" in combined.columns:
    combined["Emp1_Age_ratio"] = combined["Employment_Info_1"] / (
        combined["Ins_Age"] + eps
    )
if "Employment_Info_6" in combined.columns and "Ins_Age" in combined.columns:
    combined["Emp6_Age_ratio"] = combined["Employment_Info_6"] / (
        combined["Ins_Age"] + eps
    )

# Medical keyword aggregations
med_cols = [col for col in combined.columns if col.startswith("Medical_Keyword_")]
if med_cols:
    combined["Med_Keywords_Count"] = combined[med_cols].sum(axis=1)
    combined["Med_Keywords_NonZero"] = (combined[med_cols] > 0).sum(axis=1)

# Family History aggregations
fam_cols = [col for col in combined.columns if col.startswith("Family_Hist_")]
if fam_cols:
    combined["Fam_Hist_Mean"] = combined[fam_cols].mean(axis=1)
    combined["Fam_Hist_Max"] = combined[fam_cols].max(axis=1)
    combined["Fam_Hist_Min"] = combined[fam_cols].min(axis=1)
    combined["Fam_Hist_Std"] = combined[fam_cols].std(axis=1).fillna(0)
    combined["Fam_Hist_Null_Count"] = combined[fam_cols].isnull().sum(axis=1)

# Overall missing values count
combined["Total_Missing_Count"] = combined.isnull().sum(axis=1)

# Separate train and test sets
X = combined.iloc[:n_train].reset_index(drop=True)
X_test = combined.iloc[n_train:].reset_index(drop=True)


# 3. Powell Threshold Optimizer for QWK
class QWKThresholdOptimizer:
    def __init__(self):
        self.coef_ = None

    def _loss(self, coef, X, y):
        sorted_coef = np.sort(coef)
        preds = np.digitize(X, sorted_coef) + 1
        return -cohen_kappa_score(y, preds, weights="quadratic")

    def fit(self, X, y):
        initial_coef = [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5]
        res = minimize(
            self._loss,
            initial_coef,
            args=(X, y),
            method="Powell",
            options={"maxiter": 2000, "ftol": 1e-5},
        )
        self.coef_ = np.sort(res.x)

    def predict(self, X):
        return np.digitize(X, self.coef_) + 1


# 4. 5-Fold Stratified Cross-Validation with LightGBM
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_preds = np.zeros(n_train)
test_preds = np.zeros(len(X_test))

lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.03,
    "num_leaves": 63,
    "max_depth": 8,
    "feature_fraction": 0.70,
    "bagging_fraction": 0.80,
    "bagging_freq": 1,
    "min_child_samples": 25,
    "verbosity": -1,
    "n_jobs": -1,
    "random_state": 42,
}

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_tr, y_tr = X.iloc[train_idx], y[train_idx]
    X_va, y_val = X.iloc[val_idx], y[val_idx]

    trn_data = lgb.Dataset(X_tr, label=y_tr)
    val_data = lgb.Dataset(X_va, label=y_val, reference=trn_data)

    model = lgb.train(
        lgb_params,
        trn_data,
        num_boost_round=1500,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
    )

    oof_preds[val_idx] = model.predict(X_va, num_iteration=model.best_iteration)
    test_preds += model.predict(X_test, num_iteration=model.best_iteration) / n_splits

# 5. Optimize Cutoffs and Evaluate
optimizer = QWKThresholdOptimizer()
optimizer.fit(oof_preds, y)
discrete_oof_preds = optimizer.predict(oof_preds)

cv_qwk = cohen_kappa_score(y, discrete_oof_preds, weights="quadratic")
print(f"5-Fold CV Quadratic Weighted Kappa (QWK): {cv_qwk:.5f}")

# 6. Generate Test Submission
discrete_test_preds = optimizer.predict(test_preds)

os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame(
    {
        "id": np.arange(len(discrete_test_preds)),
        "Response": discrete_test_preds.astype(int),
    }
)
submission.to_csv("./working/submission.csv", index=False)
print("Saved submission to ./working/submission.csv successfully.")
