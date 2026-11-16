import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb

# Ensure output directory exists
os.makedirs("./working", exist_ok=True)

# Load dataset
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "DEATH_EVENT"


def engineer_features(df):
    data = df.copy()

    # Skewness corrections
    data["log_cpk"] = np.log1p(data["creatinine_phosphokinase"])
    data["log_serum_creatinine"] = np.log1p(data["serum_creatinine"])

    # Clinical interactions and ratios
    data["ef_creatinine_ratio"] = data["ejection_fraction"] / (
        data["serum_creatinine"] + 1e-5
    )
    data["sodium_creatinine_ratio"] = data["serum_sodium"] / (
        data["serum_creatinine"] + 1e-5
    )
    data["age_time_ratio"] = data["age"] / (data["time"] + 1.0)
    data["platelets_sodium_ratio"] = data["platelets"] / (data["serum_sodium"] + 1e-5)

    # Clinical risk index based on established thresholds
    data["low_ejection_fraction"] = (data["ejection_fraction"] < 30).astype(int)
    data["high_serum_creatinine"] = (data["serum_creatinine"] > 1.5).astype(int)
    data["hyponatremia"] = (data["serum_sodium"] < 135).astype(int)
    data["elderly"] = (data["age"] >= 65).astype(int)

    data["cumulative_risk_count"] = (
        data["low_ejection_fraction"]
        + data["high_serum_creatinine"]
        + data["hyponatremia"]
        + data["elderly"]
        + data["high_blood_pressure"]
    )

    return data


# Prepare feature sets
df_train_feat = engineer_features(train_df)
df_test_feat = engineer_features(test_df)

feature_cols = [c for c in df_train_feat.columns if c not in [target_col, "id"]]

X = df_train_feat[feature_cols]
y = df_train_feat[target_col]
X_test = df_test_feat[feature_cols]

# Stratified 5-Fold Cross-Validation Setup
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(train_df))
test_preds = np.zeros(len(test_df))

xgb_params = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "max_depth": 3,
    "learning_rate": 0.03,
    "n_estimators": 250,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.5,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": -1,
}

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train_fold, y_train_fold = X.iloc[train_idx], y.iloc[train_idx]
    X_val_fold, y_val_fold = X.iloc[val_idx], y.iloc[val_idx]

    model = xgb.XGBClassifier(**xgb_params)
    model.fit(
        X_train_fold, y_train_fold, eval_set=[(X_val_fold, y_val_fold)], verbose=False
    )

    val_pred = model.predict_proba(X_val_fold)[:, 1]
    oof_preds[val_idx] = val_pred
    test_preds += model.predict_proba(X_test)[:, 1] / skf.n_splits

# Compute Out-of-Fold Evaluation Metric
overall_auc = roc_auc_score(y, oof_preds)
print(f"Validation ROC AUC: {overall_auc:.5f}")

# Save Submission File
submission = pd.DataFrame({"id": np.arange(len(test_df)), "DEATH_EVENT": test_preds})

submission.to_csv("./working/submission.csv", index=False)
print("Submission successfully saved to ./working/submission.csv")
