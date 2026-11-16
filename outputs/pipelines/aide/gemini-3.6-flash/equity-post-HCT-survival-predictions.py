import os
import warnings
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

# 1. Load Data
train_path = "./input/train.parquet"
test_path = "./input/test.parquet"

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)


# 2. Feature Engineering
def feature_engineering(df):
    df = df.copy()

    # Age difference
    if "donor_age" in df.columns and "age_at_hct" in df.columns:
        df["age_diff"] = df["donor_age"] - df["age_at_hct"]

    # HLA match aggregates
    high_cols = [c for c in df.columns if "hla_match" in c and "high" in c]
    low_cols = [c for c in df.columns if "hla_match" in c and "low" in c]

    if high_cols:
        df["hla_high_sum"] = df[high_cols].sum(axis=1)
        df["hla_high_mean"] = df[high_cols].mean(axis=1)
    if low_cols:
        df["hla_low_sum"] = df[low_cols].sum(axis=1)
        df["hla_low_mean"] = df[low_cols].mean(axis=1)

    return df


train_df = feature_engineering(train_df)
test_df = feature_engineering(test_df)

# Separate target and feature columns
y_efs = train_df["efs"].values
y_time = train_df["efs_time"].values

drop_cols = ["efs", "efs_time", "ID", "id"]
feature_cols = [c for c in train_df.columns if c not in drop_cols]

# Set categorical column types
for c in feature_cols:
    if train_df[c].dtype == "object" or train_df[c].dtype.name == "category":
        train_df[c] = train_df[c].astype("category")
        test_df[c] = test_df[c].astype("category")

X = train_df[feature_cols]
X_test = test_df[feature_cols]

# Target proxy for risk score learning: shorter survival time with event = higher risk
y_target = np.where(y_efs == 1, -np.log1p(y_time), -np.log1p(y_time) - 0.5)

# 3. Stratified Cross-Validation & Training
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(train_df))
test_preds = np.zeros(len(test_df))

lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "n_estimators": 1000,
    "learning_rate": 0.03,
    "num_leaves": 31,
    "random_state": 42,
    "verbose": -1,
    "n_jobs": -1,
}

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_efs)):
    X_tr, y_tr = X.iloc[train_idx], y_target[train_idx]
    X_va, y_va = X.iloc[val_idx], y_target[val_idx]

    model = lgb.LGBMRegressor(**lgb_params)
    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )

    val_preds = model.predict(X_va)
    oof_preds[val_idx] = val_preds
    test_preds += model.predict(X_test) / skf.n_splits


# 4. Evaluation Metric Calculation (Concordance Index)
def compute_cindex(y_time, y_event, risk_scores):
    try:
        from lifelines.utils import concordance_index

        return concordance_index(y_time, -risk_scores, y_event)
    except ImportError:
        order = np.argsort(y_time)
        y_t = y_time[order]
        y_e = y_event[order]
        r_s = risk_scores[order]

        n = len(y_t)
        concordant = 0.0
        total = 0.0
        for i in range(n):
            if y_e[i] == 1:
                mask = y_t > y_t[i]
                if not np.any(mask):
                    continue
                diff = r_s[i] - r_s[mask]
                concordant += np.sum(diff > 0) + 0.5 * np.sum(diff == 0)
                total += np.sum(mask)
        return concordant / total if total > 0 else 0.5


oof_cindex = compute_cindex(y_time, y_efs, oof_preds)
print(f"OOF Concordance Index: {oof_cindex:.5f}")

# 5. Save Submission File
os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame({"id": np.arange(len(test_preds)), "efs": test_preds})
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved successfully to ./working/submission.csv")
