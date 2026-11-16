import os
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier
from scipy.stats import rankdata
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

# Ensure output directory exists
os.makedirs("./working", exist_ok=True)

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Standardize target
target_col = "Class"
y_raw = train_df[target_col].copy()

# Target mapping for ILPD: Class 1 (liver patient) vs Class 2 (non-liver patient)
unique_labels = sorted(y_raw.dropna().unique().tolist())
if set(str(x) for x in unique_labels) == {"1", "2"}:
    y = np.where(y_raw.astype(str) == "1", 1, 0).astype(np.int32)
else:
    factorized, _ = pd.factorize(y_raw)
    y = factorized.astype(np.int32)


def engineer_features(df):
    data = df.copy()

    col_names = list(data.columns)
    if "Class" in col_names:
        col_names.remove("Class")

    v_map = {
        "V1": "Age",
        "V2": "Gender",
        "V3": "TB",
        "V4": "DB",
        "V5": "Alkphos",
        "V6": "SGPT",
        "V7": "SGOT",
        "V8": "TP",
        "V9": "ALB",
        "V10": "AG_Ratio",
    }
    rename_dict = {}
    for col in col_names:
        if col in v_map:
            rename_dict[col] = v_map[col]
        else:
            low = col.lower()
            if "age" in low:
                rename_dict[col] = "Age"
            elif "gender" in low or "sex" in low:
                rename_dict[col] = "Gender"
            elif "total" in low and "bil" in low:
                rename_dict[col] = "TB"
            elif "direct" in low and "bil" in low:
                rename_dict[col] = "DB"
            elif "alk" in low:
                rename_dict[col] = "Alkphos"
            elif "sgpt" in low or "alamine" in low or "alanine" in low or "alt" in low:
                rename_dict[col] = "SGPT"
            elif "sgot" in low or "aspartate" in low or "ast" in low:
                rename_dict[col] = "SGOT"
            elif "protein" in low:
                rename_dict[col] = "TP"
            elif "albumin" in low and "ratio" not in low:
                rename_dict[col] = "ALB"
            elif "ratio" in low:
                rename_dict[col] = "AG_Ratio"

    data = data.rename(columns=rename_dict)

    # Encode Gender
    if "Gender" in data.columns:
        gender_str = data["Gender"].astype(str).str.strip().str.lower()
        data["Gender"] = (
            gender_str.map(
                {
                    "male": 1.0,
                    "m": 1.0,
                    "1": 1.0,
                    "female": 0.0,
                    "f": 0.0,
                    "2": 0.0,
                    "0": 0.0,
                }
            )
            .fillna(0.0)
            .astype(np.float32)
        )

    # Convert all feature columns to float
    feat_cols = [c for c in data.columns if c != "Class"]
    for c in feat_cols:
        data[c] = pd.to_numeric(data[c], errors="coerce")

    # Domain features & log-transforms for skewed biomarkers
    if "SGOT" in data.columns and "SGPT" in data.columns:
        data["De_Ritis_Ratio"] = data["SGOT"] / (data["SGPT"] + 1e-5)
        data["Total_Transaminases"] = data["SGOT"] + data["SGPT"]

    if "TB" in data.columns and "DB" in data.columns:
        data["Indirect_Bilirubin"] = (data["TB"] - data["DB"]).clip(lower=0)
        data["DB_TB_Ratio"] = data["DB"] / (data["TB"] + 1e-5)
        data["IB_TB_Ratio"] = data["Indirect_Bilirubin"] / (data["TB"] + 1e-5)

    if "TP" in data.columns and "ALB" in data.columns:
        data["Globulin"] = (data["TP"] - data["ALB"]).clip(lower=0)
        data["Albumin_Globulin_Calc_Ratio"] = data["ALB"] / (data["Globulin"] + 1e-5)

    if "Alkphos" in data.columns and "SGOT" in data.columns and "SGPT" in data.columns:
        data["AST_Alkphos_Ratio"] = data["SGOT"] / (data["Alkphos"] + 1e-5)
        data["ALT_Alkphos_Ratio"] = data["SGPT"] / (data["Alkphos"] + 1e-5)

    # Log-transform skewed features
    skewed_cols = ["TB", "DB", "Alkphos", "SGPT", "SGOT"]
    for col in skewed_cols:
        if col in data.columns:
            data[f"Log_{col}"] = np.log1p(np.clip(data[col], a_min=0, a_max=None))

    if "Class" in data.columns:
        data = data.drop(columns=["Class"])

    return data


X = engineer_features(train_df)
X_test = engineer_features(test_df)

feature_cols = [c for c in X.columns if c in X_test.columns]
X = X[feature_cols].copy()
X_test = X_test[feature_cols].copy()

# Impute missing values with column medians
for col in feature_cols:
    median_val = X[col].median()
    X[col] = X[col].fillna(median_val)
    X_test[col] = X_test[col].fillna(median_val)

n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# Array containers for model predictions
model_names = ["catboost", "lightgbm", "xgboost", "rf", "et"]
oof_ranks = {name: np.zeros(len(X)) for name in model_names}
test_ranks = {name: np.zeros(len(X_test)) for name in model_names}

cb_params = {
    "iterations": 400,
    "learning_rate": 0.03,
    "depth": 4,
    "l2_leaf_reg": 5.0,
    "random_seed": 42,
    "verbose": False,
    "eval_metric": "AUC",
}

lgb_params = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.03,
    "num_leaves": 12,
    "max_depth": 4,
    "min_child_samples": 8,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "verbosity": -1,
}

xgb_params = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "learning_rate": 0.03,
    "max_depth": 3,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.5,
    "reg_lambda": 2.0,
    "random_state": 42,
    "n_estimators": 250,
}

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_tr, y_tr = X.iloc[train_idx].values, y[train_idx]
    X_va, y_va = X.iloc[val_idx].values, y[val_idx]
    X_te = X_test.values

    # 1. CatBoost
    cb_model = CatBoostClassifier(**cb_params)
    cb_model.fit(
        X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=40, verbose=False
    )
    cb_va = cb_model.predict_proba(X_va)[:, 1]
    cb_te = cb_model.predict_proba(X_te)[:, 1]

    # 2. LightGBM
    dtrain = lgb.Dataset(X_tr, label=y_tr)
    dval = lgb.Dataset(X_va, label=y_va, reference=dtrain)
    lgb_model = lgb.train(
        lgb_params,
        dtrain,
        num_boost_round=400,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(0)],
    )
    lgb_va = lgb_model.predict(X_va)
    lgb_te = lgb_model.predict(X_te)

    # 3. XGBoost
    xgb_model = xgb.XGBClassifier(**xgb_params)
    xgb_model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    xgb_va = xgb_model.predict_proba(X_va)[:, 1]
    xgb_te = xgb_model.predict_proba(X_te)[:, 1]

    # 4. Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=250,
        max_depth=5,
        min_samples_leaf=3,
        max_features="sqrt",
        random_state=42 + fold,
        n_jobs=-1,
    )
    rf_model.fit(X_tr, y_tr)
    rf_va = rf_model.predict_proba(X_va)[:, 1]
    rf_te = rf_model.predict_proba(X_te)[:, 1]

    # 5. ExtraTrees
    et_model = ExtraTreesClassifier(
        n_estimators=250,
        max_depth=5,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=42 + fold,
        n_jobs=-1,
    )
    et_model.fit(X_tr, y_tr)
    et_va = et_model.predict_proba(X_va)[:, 1]
    et_te = et_model.predict_proba(X_te)[:, 1]

    # Normalize predictions by rank per fold to optimize ROC-AUC rank invariance
    def rank_norm(arr):
        return (rankdata(arr) - 0.5) / len(arr)

    preds_va = {
        "catboost": rank_norm(cb_va),
        "lightgbm": rank_norm(lgb_va),
        "xgboost": rank_norm(xgb_va),
        "rf": rank_norm(rf_va),
        "et": rank_norm(et_va),
    }

    preds_te = {
        "catboost": rank_norm(cb_te),
        "lightgbm": rank_norm(lgb_te),
        "xgboost": rank_norm(xgb_te),
        "rf": rank_norm(rf_te),
        "et": rank_norm(et_te),
    }

    for name in model_names:
        oof_ranks[name][val_idx] = preds_va[name]
        test_ranks[name] += preds_te[name] / n_splits

# Blend weights across diverse models
weights = {
    "catboost": 0.25,
    "lightgbm": 0.20,
    "xgboost": 0.20,
    "rf": 0.20,
    "et": 0.15,
}

oof_ensemble = np.zeros(len(X))
test_ensemble = np.zeros(len(X_test))

for name, w in weights.items():
    oof_ensemble += w * oof_ranks[name]
    test_ensemble += w * test_ranks[name]

cv_auc = roc_auc_score(y, oof_ensemble)
print(f"5-Fold CV ROC-AUC: {cv_auc:.5f}")

submission = pd.DataFrame({"id": np.arange(len(test_ensemble)), "Class": test_ensemble})
submission.to_csv("./working/submission.csv", index=False)
print("Saved predictions to ./working/submission.csv")
