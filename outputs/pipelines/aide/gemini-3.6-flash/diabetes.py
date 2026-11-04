import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression

try:
    from catboost import CatBoostClassifier

    USE_CATBOOST = True
except ImportError:
    from sklearn.ensemble import HistGradientBoostingClassifier

    USE_CATBOOST = False

# 1. Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Process target column
target_col = "class"
if target_col in train_df.columns:
    y = train_df[target_col].values
    if y.dtype == object or isinstance(y[0], str):
        y = (y == "tested_positive").astype(int)
    else:
        y = y.astype(int)
    X_train_raw = train_df.drop(columns=[target_col]).copy()
else:
    raise ValueError("Target column not found in training dataset.")

X_test_raw = test_df.copy()
if target_col in X_test_raw.columns:
    X_test_raw = X_test_raw.drop(columns=[target_col])

# Map feature names dynamically for robust access across schema variations
col_map = {}
for col in X_train_raw.columns:
    c_lower = col.lower()
    if "preg" in c_lower:
        col_map["preg"] = col
    elif "plas" in c_lower or "glu" in c_lower:
        col_map["glu"] = col
    elif "pres" in c_lower or "bp" in c_lower or "blood" in c_lower:
        col_map["bp"] = col
    elif "skin" in c_lower or "tri" in c_lower:
        col_map["skin"] = col
    elif "ins" in c_lower:
        col_map["ins"] = col
    elif "mass" in c_lower or "bmi" in c_lower:
        col_map["bmi"] = col
    elif "ped" in c_lower:
        col_map["ped"] = col
    elif "age" in c_lower:
        col_map["age"] = col

# Physiological columns where zero represents missing data
zero_cols = [col_map[k] for k in ["glu", "bp", "skin", "ins", "bmi"] if k in col_map]


def preprocess_and_feature_engineer(df_tr, df_val_or_te, zero_columns):
    df_tr = df_tr.copy()
    df_val_or_te = df_val_or_te.copy()

    # Flag zeros as missing indicators and replace 0 with NaN
    for col in zero_columns:
        df_tr[f"{col}_nan"] = (df_tr[col] == 0).astype(int)
        df_val_or_te[f"{col}_nan"] = (df_val_or_te[col] == 0).astype(int)
        df_tr.loc[df_tr[col] == 0, col] = np.nan
        df_val_or_te.loc[df_val_or_te[col] == 0, col] = np.nan

    # Fit StandardScaler + KNNImputer strictly on training fold split to prevent leakage & scale bias
    cols = df_tr.columns
    scaler = StandardScaler()
    tr_scaled = pd.DataFrame(
        scaler.fit_transform(df_tr), columns=cols, index=df_tr.index
    )
    val_te_scaled = pd.DataFrame(
        scaler.transform(df_val_or_te), columns=cols, index=df_val_or_te.index
    )

    imputer = KNNImputer(n_neighbors=5)
    tr_imp_scaled = imputer.fit_transform(tr_scaled)
    val_te_imp_scaled = imputer.transform(val_te_scaled)

    # Convert back to unscaled physical domain for clinical calculations
    df_tr_imp = pd.DataFrame(
        scaler.inverse_transform(tr_imp_scaled), columns=cols, index=df_tr.index
    )
    df_val_te_imp = pd.DataFrame(
        scaler.inverse_transform(val_te_imp_scaled),
        columns=cols,
        index=df_val_or_te.index,
    )

    # Engineer clinical and metabolic indicators
    for df in [df_tr_imp, df_val_te_imp]:
        glu = df[col_map["glu"]] if "glu" in col_map else 0
        ins = df[col_map["ins"]] if "ins" in col_map else 0
        bmi = df[col_map["bmi"]] if "bmi" in col_map else 0
        age = df[col_map["age"]] if "age" in col_map else 0
        preg = df[col_map["preg"]] if "preg" in col_map else 0
        ped = df[col_map["ped"]] if "ped" in col_map else 0
        bp = df[col_map["bp"]] if "bp" in col_map else 0
        skin = df[col_map["skin"]] if "skin" in col_map else 0

        # Clinical insulin resistance & metabolic indices
        df["homa_ir"] = (glu * ins) / 405.0
        df["quicki"] = 1.0 / (
            np.log10(glu.clip(lower=1.0)) + np.log10(ins.clip(lower=1.0)) + 1e-5
        )
        df["glu_ins_ratio"] = glu / (ins + 1.0)
        df["preg_rate"] = preg / (age - 14.0).clip(lower=1.0)
        df["glu_bmi_ratio"] = glu / (bmi + 1e-5)
        df["glu_age_prod"] = glu * age
        df["ped_glu_prod"] = ped * glu
        df["ins_skin_ratio"] = ins / (skin + 1e-5)
        df["skin_bmi_ratio"] = skin / (bmi + 1e-5)

        # ADA Clinical threshold indicators
        df["prediabetes_flag"] = (glu >= 140).astype(int)
        df["diabetes_flag"] = (glu >= 200).astype(int)
        df["obesity_flag"] = (bmi >= 30).astype(int)

        # Composite metabolic risk indicator
        metabolic_risk = (
            (glu > 100).astype(int)
            + (bmi > 30).astype(int)
            + (bp > 80).astype(int)
            + (age > 40).astype(int)
        )
        df["metabolic_risk_score"] = metabolic_risk

        # Non-linear log transformations
        df["log_ins"] = np.log1p(ins.clip(lower=0))
        df["log_ped"] = np.log1p(ped.clip(lower=0))
        df["log_glu"] = np.log1p(glu.clip(lower=0))
        df["log_bmi"] = np.log1p(bmi.clip(lower=0))

    return df_tr_imp, df_val_te_imp


# Perform 5-fold Stratified Cross-Validation with Multi-Model Diversity Blend
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(X_train_raw))
test_preds = np.zeros(len(X_test_raw))
seeds = [42, 123, 456, 789]

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_raw, y)):
    X_tr_fold = X_train_raw.iloc[train_idx].copy()
    y_tr_fold = y[train_idx]
    X_va_fold = X_train_raw.iloc[val_idx].copy()
    y_va_fold = y[val_idx]
    X_te_fold = X_test_raw.copy()

    # Preprocess splits independently
    X_tr_proc, X_va_proc = preprocess_and_feature_engineer(
        X_tr_fold, X_va_fold, zero_cols
    )
    _, X_te_proc = preprocess_and_feature_engineer(X_tr_fold, X_te_fold, zero_cols)

    # Standardize features for linear model split
    lr_scaler = StandardScaler()
    X_tr_lr = lr_scaler.fit_transform(X_tr_proc)
    X_va_lr = lr_scaler.transform(X_va_proc)
    X_te_lr = lr_scaler.transform(X_te_proc)

    cb_val_preds = np.zeros(len(val_idx))
    cb_te_preds = np.zeros(len(X_test_raw))

    et_val_preds = np.zeros(len(val_idx))
    et_te_preds = np.zeros(len(X_test_raw))

    for s in seeds:
        if USE_CATBOOST:
            cb_model = CatBoostClassifier(
                iterations=600,
                learning_rate=0.025,
                depth=4,
                l2_leaf_reg=4.0,
                eval_metric="AUC",
                random_seed=s + fold * 10,
                verbose=0,
            )
            cb_model.fit(
                X_tr_proc,
                y_tr_fold,
                eval_set=(X_va_proc, y_va_fold),
                early_stopping_rounds=60,
            )
            cb_val_preds += cb_model.predict_proba(X_va_proc)[:, 1] / len(seeds)
            cb_te_preds += cb_model.predict_proba(X_te_proc)[:, 1] / len(seeds)
        else:
            cb_model = HistGradientBoostingClassifier(
                max_iter=300,
                learning_rate=0.03,
                max_depth=4,
                l2_regularization=1.0,
                random_state=s + fold * 10,
            )
            cb_model.fit(X_tr_proc, y_tr_fold)
            cb_val_preds += cb_model.predict_proba(X_va_proc)[:, 1] / len(seeds)
            cb_te_preds += cb_model.predict_proba(X_te_proc)[:, 1] / len(seeds)

        et_model = ExtraTreesClassifier(
            n_estimators=300,
            max_depth=5,
            min_samples_split=4,
            random_state=s + fold * 10,
        )
        et_model.fit(X_tr_proc, y_tr_fold)
        et_val_preds += et_model.predict_proba(X_va_proc)[:, 1] / len(seeds)
        et_te_preds += et_model.predict_proba(X_te_proc)[:, 1] / len(seeds)

    lr_model = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
    lr_model.fit(X_tr_lr, y_tr_fold)
    lr_val_preds = lr_model.predict_proba(X_va_lr)[:, 1]
    lr_te_preds = lr_model.predict_proba(X_te_lr)[:, 1]

    # Weighted ensemble blend (0.70 CatBoost + 0.15 ExtraTrees + 0.15 LogisticRegression)
    fold_val_preds = 0.70 * cb_val_preds + 0.15 * et_val_preds + 0.15 * lr_val_preds
    fold_te_preds = 0.70 * cb_te_preds + 0.15 * et_te_preds + 0.15 * lr_te_preds

    oof_preds[val_idx] = fold_val_preds
    test_preds += fold_te_preds / skf.n_splits

cv_auc = roc_auc_score(y, oof_preds)
print(f"Validation ROC AUC: {cv_auc:.5f}")

# Export test predictions to submission.csv
os.makedirs("./working", exist_ok=True)
sub = pd.DataFrame({"id": np.arange(len(test_preds)), "class": test_preds})
sub.to_csv("./working/submission.csv", index=False)
