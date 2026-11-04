import os
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    ExtraTreesClassifier,
    RandomForestClassifier,
)
import lightgbm as lgb

# Ensure working directory exists
os.makedirs("./working", exist_ok=True)

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Separate target
target_col = "Class"
y_raw = train_df[target_col]
le = LabelEncoder()
y = le.fit_transform(y_raw.astype(str))

X_train_raw = train_df.drop(columns=[target_col])
X_test_raw = test_df.copy()


def create_features(df):
    data = df.copy()

    # Map Gender column if present
    gender_col = (
        "V2"
        if "V2" in data.columns
        else ("Gender" if "Gender" in data.columns else None)
    )
    if gender_col:
        data[gender_col] = (
            data[gender_col]
            .astype(str)
            .str.lower()
            .map({"male": 1, "female": 0, "1": 1, "0": 0})
        )
        data[gender_col] = data[gender_col].fillna(0).astype(int)

    # Convert all columns to numeric
    for col in data.columns:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    # Identify clinical features
    c_tb = "V3" if "V3" in data.columns else "Total_Bilirubin"
    c_db = "V4" if "V4" in data.columns else "Direct_Bilirubin"
    c_alp = "V5" if "V5" in data.columns else "Alkphos"
    c_alt = "V6" if "V6" in data.columns else "Sgpt"
    c_ast = "V7" if "V7" in data.columns else "Sgot"
    c_tp = "V8" if "V8" in data.columns else "Total_Proteins"
    c_alb = "V9" if "V9" in data.columns else "Albumin"

    # Derived clinical features
    if c_tb in data.columns and c_db in data.columns:
        data["tb_db_ratio"] = data[c_db] / (data[c_tb] + 1e-5)
        data["indirect_bilirubin"] = data[c_tb] - data[c_db]

    if c_ast in data.columns and c_alt in data.columns:
        data["ast_alt_ratio"] = data[c_ast] / (data[c_alt] + 1e-5)
        data["ast_alt_sum"] = data[c_ast] + data[c_alt]

    if c_tp in data.columns and c_alb in data.columns:
        data["globulin"] = data[c_tp] - data[c_alb]
        data["alb_tp_ratio"] = data[c_alb] / (data[c_tp] + 1e-5)
        data["alb_globulin_ratio_calc"] = data[c_alb] / (data["globulin"] + 1e-5)

    # Log transformations for skewed biochemical markers
    skewed_cols = [c_tb, c_db, c_alp, c_alt, c_ast]
    for sc in skewed_cols:
        if sc in data.columns:
            data[f"{sc}_log"] = np.log1p(np.maximum(0, data[sc]))

    # Impute missing values with column medians
    data = data.fillna(data.median())
    return data


X_train = create_features(X_train_raw)
X_test = create_features(X_test_raw)


def rank_transform(preds):
    """Normalize predictions to percentile ranks in [0, 1]."""
    return rankdata(preds) / len(preds)


# 5-Fold Stratified Cross Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_preds = np.zeros(len(X_train))
test_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y)):
    X_tr, y_tr = X_train.iloc[train_idx], y[train_idx]
    X_va, y_va = X_train.iloc[val_idx], y[val_idx]

    # Model 1: LightGBM Classifier
    model_lgb = lgb.LGBMClassifier(
        n_estimators=150,
        learning_rate=0.03,
        num_leaves=15,
        min_child_samples=15,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        verbosity=-1,
    )
    model_lgb.fit(X_tr, y_tr)
    pred_lgb_val = model_lgb.predict_proba(X_va)[:, 1]
    pred_lgb_test = model_lgb.predict_proba(X_test)[:, 1]

    # Model 2: HistGradientBoosting Classifier
    model_hgb = HistGradientBoostingClassifier(
        max_iter=150,
        learning_rate=0.03,
        max_leaf_nodes=15,
        l2_regularization=1.0,
        random_state=42 + fold,
    )
    model_hgb.fit(X_tr, y_tr)
    pred_hgb_val = model_hgb.predict_proba(X_va)[:, 1]
    pred_hgb_test = model_hgb.predict_proba(X_test)[:, 1]

    # Model 3: Extra Trees Classifier
    model_et = ExtraTreesClassifier(
        n_estimators=200, max_depth=6, min_samples_split=4, random_state=42 + fold
    )
    model_et.fit(X_tr, y_tr)
    pred_et_val = model_et.predict_proba(X_va)[:, 1]
    pred_et_test = model_et.predict_proba(X_test)[:, 1]

    # Model 4: Random Forest Classifier
    model_rf = RandomForestClassifier(
        n_estimators=200, max_depth=6, min_samples_split=4, random_state=42 + fold
    )
    model_rf.fit(X_tr, y_tr)
    pred_rf_val = model_rf.predict_proba(X_va)[:, 1]
    pred_rf_test = model_rf.predict_proba(X_test)[:, 1]

    # Model 5: Logistic Regression Classifier (Linear representation)
    model_lr = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(C=0.1, max_iter=1000, random_state=42 + fold),
    )
    model_lr.fit(X_tr, y_tr)
    pred_lr_val = model_lr.predict_proba(X_va)[:, 1]
    pred_lr_test = model_lr.predict_proba(X_test)[:, 1]

    # Percentile rank transformation before blending
    r_lgb_val, r_lgb_test = rank_transform(pred_lgb_val), rank_transform(pred_lgb_test)
    r_hgb_val, r_hgb_test = rank_transform(pred_hgb_val), rank_transform(pred_hgb_test)
    r_et_val, r_et_test = rank_transform(pred_et_val), rank_transform(pred_et_test)
    r_rf_val, r_rf_test = rank_transform(pred_rf_val), rank_transform(pred_rf_test)
    r_lr_val, r_lr_test = rank_transform(pred_lr_val), rank_transform(pred_lr_test)

    # Weighted Rank Ensemble combining non-linear trees and linear model
    val_blend = (
        r_lgb_val * 0.30
        + r_hgb_val * 0.30
        + r_et_val * 0.15
        + r_rf_val * 0.10
        + r_lr_val * 0.15
    )
    test_blend = (
        r_lgb_test * 0.30
        + r_hgb_test * 0.30
        + r_et_test * 0.15
        + r_rf_test * 0.10
        + r_lr_test * 0.15
    )

    oof_preds[val_idx] = val_blend
    test_preds += test_blend / 5.0

# Calculate overall Out-Of-Fold ROC AUC
final_auc = roc_auc_score(y, oof_preds)
print(f"5-Fold CV ROC AUC Score: {final_auc:.5f}")

# Save test predictions
submission = pd.DataFrame({"id": np.arange(len(test_preds)), "Class": test_preds})
submission.to_csv("./working/submission.csv", index=False)
