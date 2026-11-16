import os
import gc
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# 1. Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

orig_path = "./input/original/train.csv"
has_orig = os.path.exists(orig_path)
if has_orig:
    orig_df = pd.read_csv(orig_path)
else:
    orig_df = pd.DataFrame()

# Map target label
target_map = {"Presence": 1, "Absence": 0}
if "Heart Disease" in train_df.columns:
    if train_df["Heart Disease"].dtype == object:
        train_df["Heart Disease"] = train_df["Heart Disease"].map(target_map)

if has_orig and "Heart Disease" in orig_df.columns:
    if orig_df["Heart Disease"].dtype == object:
        orig_df["Heart Disease"] = orig_df["Heart Disease"].map(target_map)


# 2. Feature engineering function
def engineer_features(df):
    df = df.copy()

    # Physiological & heart rate metrics
    max_hr_pred = 220 - df["Age"]
    df["hr_ratio"] = df["Max HR"] / (max_hr_pred + 1e-5)
    df["hr_reserve"] = max_hr_pred - df["Max HR"]
    df["bp_age_ratio"] = df["BP"] / (df["Age"] + 1e-5)
    df["bp_age_prod"] = df["BP"] * df["Age"]
    df["chol_age_ratio"] = df["Cholesterol"] / (df["Age"] + 1e-5)
    df["chol_bp_ratio"] = df["Cholesterol"] / (df["BP"] + 1e-5)

    # Cardiac stress & ischemic factors
    df["st_slope_prod"] = df["ST depression"] * df["Slope of ST"]
    df["st_dep_positive"] = (df["ST depression"] > 0).astype(int)
    df["angina_st_prod"] = df["Exercise angina"] * df["ST depression"]

    # Composite risk category combinations
    cp_str = df["Chest pain type"].astype(str)
    angina_str = df["Exercise angina"].astype(str)
    vessels_str = df["Number of vessels fluro"].astype(str)
    thal_str = df["Thallium"].astype(str)

    df["cp_angina"] = (cp_str + "_" + angina_str).astype("category").cat.codes
    df["vessels_thallium"] = (vessels_str + "_" + thal_str).astype("category").cat.codes
    df["cp_thallium"] = (cp_str + "_" + thal_str).astype("category").cat.codes

    # Simple risk factor count
    df["risk_count"] = (
        (df["Age"] > 55).astype(int)
        + (df["Sex"] == 1).astype(int)
        + (df["BP"] > 130).astype(int)
        + (df["Cholesterol"] > 200).astype(int)
        + (df["Exercise angina"] == 1).astype(int)
        + (df["ST depression"] > 1.0).astype(int)
        + (df["Number of vessels fluro"] > 0).astype(int)
    )

    return df


# Process datasets
train_feat = engineer_features(train_df)
test_feat = engineer_features(test_df)
if has_orig:
    orig_feat = engineer_features(orig_df)

# Separate features and target
target_col = "Heart Disease"
ignore_cols = [target_col, "id"]

feature_cols = [c for c in train_feat.columns if c not in ignore_cols]

X_train_full = train_feat[feature_cols].copy()
y_train_full = train_feat[target_col].values.astype(int)
X_test = test_feat[feature_cols].copy()

if has_orig:
    X_orig = orig_feat[feature_cols].copy()
    y_orig = orig_feat[target_col].values.astype(int)

# 3. Stratified K-Fold Cross-Validation setup
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_preds = np.zeros(len(train_df))
test_preds = np.zeros(len(test_df))

xgb_params = {
    "n_estimators": 1500,
    "learning_rate": 0.03,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "tree_method": "hist",
    "eval_metric": "auc",
    "random_state": 42,
    "early_stopping_rounds": 50,
    "n_jobs": -1,
}

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_full, y_train_full)):
    X_tr = X_train_full.iloc[train_idx]
    y_tr = y_train_full[train_idx]

    if has_orig:
        X_tr = pd.concat([X_tr, X_orig], ignore_index=True)
        y_tr = np.concatenate([y_tr, y_orig])

    X_va = X_train_full.iloc[val_idx]
    y_va = y_train_full[val_idx]

    model = xgb.XGBClassifier(**xgb_params)
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

    val_probs = model.predict_proba(X_va)[:, 1]
    oof_preds[val_idx] = val_probs
    test_preds += model.predict_proba(X_test)[:, 1] / n_splits

# 4. Evaluate OOF Performance
oof_auc = roc_auc_score(y_train_full, oof_preds)
print(f"Out-of-Fold Validation ROC AUC: {oof_auc:.5f}")

# 5. Create submission file
os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame()

if "id" in test_df.columns:
    submission["id"] = test_df["id"]
else:
    submission["id"] = np.arange(len(test_df))

submission["Heart Disease"] = test_preds
submission.to_csv("./working/submission.csv", index=False)
