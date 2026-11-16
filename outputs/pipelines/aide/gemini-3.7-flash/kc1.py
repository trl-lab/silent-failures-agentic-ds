import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
from catboost import CatBoostClassifier
import xgboost as xgb

# 1. Load Data
train_path = "./input/train.parquet"
test_path = "./input/test.parquet"

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

# Handle target column
target_col = "defects" if "defects" in train_df.columns else "problems"
if train_df[target_col].dtype == object or train_df[target_col].dtype == bool:
    y = (
        train_df[target_col]
        .map(
            {
                True: 1,
                False: 0,
                "true": 1,
                "false": 0,
                "True": 1,
                "False": 0,
                1: 1,
                0: 0,
            }
        )
        .values
    )
else:
    y = train_df[target_col].astype(int).values

raw_feature_cols = [c for c in train_df.columns if c != target_col]


# 2. Comprehensive Feature Engineering
def engineer_features(df):
    df_feat = df.copy()
    eps = 1e-6

    # Line proportions
    if "loc" in df_feat.columns:
        loc_safe = df_feat["loc"] + eps
        if "lOCode" in df_feat.columns:
            df_feat["ratio_code_loc"] = df_feat["lOCode"] / loc_safe
        if "lOComment" in df_feat.columns:
            df_feat["ratio_comment_loc"] = df_feat["lOComment"] / loc_safe
        if "lOBlank" in df_feat.columns:
            df_feat["ratio_blank_loc"] = df_feat["lOBlank"] / loc_safe
        if "lOCodeAndComment" in df_feat.columns:
            df_feat["ratio_code_and_comment_loc"] = (
                df_feat["lOCodeAndComment"] / loc_safe
            )
        if "v(g)" in df_feat.columns:
            df_feat["cyclomatic_density"] = df_feat["v(g)"] / loc_safe
        if "branchCount" in df_feat.columns:
            df_feat["branch_density"] = df_feat["branchCount"] / loc_safe

    # McCabe complexity structure ratios
    if "v(g)" in df_feat.columns:
        vg_safe = df_feat["v(g)"] + eps
        if "ev(g)" in df_feat.columns:
            df_feat["essential_complexity_ratio"] = df_feat["ev(g)"] / vg_safe
        if "iv(g)" in df_feat.columns:
            df_feat["design_complexity_ratio"] = df_feat["iv(g)"] / vg_safe

    # Operator and Operand characteristics
    if "uniq_Op" in df_feat.columns and "uniq_Opnd" in df_feat.columns:
        df_feat["ratio_uniq_op_opnd"] = df_feat["uniq_Op"] / (
            df_feat["uniq_Opnd"] + eps
        )
        df_feat["total_vocab"] = df_feat["uniq_Op"] + df_feat["uniq_Opnd"]
    if "total_Op" in df_feat.columns and "total_Opnd" in df_feat.columns:
        df_feat["ratio_total_op_opnd"] = df_feat["total_Op"] / (
            df_feat["total_Opnd"] + eps
        )
        df_feat["total_tokens"] = df_feat["total_Op"] + df_feat["total_Opnd"]

    # Halstead Theoretical Formulations & Discrepancies
    if "uniq_Op" in df_feat.columns and "uniq_Opnd" in df_feat.columns:
        u_op = np.maximum(1, df_feat["uniq_Op"])
        u_opnd = np.maximum(1, df_feat["uniq_Opnd"])
        h_length_est = u_op * np.log2(u_op) + u_opnd * np.log2(u_opnd)
        df_feat["halstead_length_est"] = h_length_est

        if "n" in df_feat.columns:
            df_feat["halstead_length_error"] = df_feat["n"] - h_length_est
            df_feat["halstead_length_ratio"] = df_feat["n"] / (h_length_est + eps)

        h_vocab = u_op + u_opnd
        if "total_Op" in df_feat.columns and "total_Opnd" in df_feat.columns:
            tot_tokens = df_feat["total_Op"] + df_feat["total_Opnd"]
            h_volume_est = tot_tokens * np.log2(np.maximum(1, h_vocab))
            df_feat["halstead_volume_est"] = h_volume_est
            if "v" in df_feat.columns:
                df_feat["halstead_volume_error"] = df_feat["v"] - h_volume_est

            h_diff_est = (u_op / 2.0) * (df_feat["total_Opnd"] / (u_opnd + eps))
            df_feat["halstead_diff_est"] = h_diff_est
            if "d" in df_feat.columns:
                df_feat["halstead_diff_error"] = df_feat["d"] - h_diff_est

    # Log transformations for heavily skewed features
    skewed_candidates = [
        "loc",
        "v",
        "d",
        "e",
        "b",
        "t",
        "lOCode",
        "total_Op",
        "total_Opnd",
    ]
    for col in skewed_candidates:
        if col in df_feat.columns:
            df_feat[f"log_{col}"] = np.log1p(np.maximum(0, df_feat[col]))

    return df_feat


X = engineer_features(train_df[raw_feature_cols])
X_test = engineer_features(test_df[raw_feature_cols])

# Impute any edge NaN/inf values
X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(0)

# 3. Model Training & 5-Fold Stratified Cross Validation
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_lgb = np.zeros(len(train_df))
oof_xgb = np.zeros(len(train_df))
oof_cat = np.zeros(len(train_df))
oof_et = np.zeros(len(train_df))

test_lgb = np.zeros(len(test_df))
test_xgb = np.zeros(len(test_df))
test_cat = np.zeros(len(test_df))
test_et = np.zeros(len(test_df))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_tr, y_tr = X.iloc[train_idx], y[train_idx]
    X_va, y_va = X.iloc[val_idx], y[val_idx]

    # Model 1: LightGBM
    lgb_model = lgb.LGBMClassifier(
        objective="binary",
        metric="auc",
        boosting_type="gbdt",
        learning_rate=0.025,
        num_leaves=24,
        max_depth=5,
        min_child_samples=25,
        subsample=0.8,
        colsample_bytree=0.75,
        random_state=42 + fold,
        n_estimators=1200,
        verbose=-1,
    )
    lgb_model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )
    oof_lgb[val_idx] = lgb_model.predict_proba(X_va)[:, 1]
    test_lgb += lgb_model.predict_proba(X_test)[:, 1] / n_splits

    # Model 2: XGBoost
    xgb_model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        learning_rate=0.025,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.75,
        min_child_weight=3,
        n_estimators=1200,
        random_state=42 + fold,
        early_stopping_rounds=50,
        n_jobs=-1,
    )
    xgb_model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    oof_xgb[val_idx] = xgb_model.predict_proba(X_va)[:, 1]
    test_xgb += xgb_model.predict_proba(X_test)[:, 1] / n_splits

    # Model 3: CatBoost
    cat_model = CatBoostClassifier(
        iterations=1200,
        learning_rate=0.03,
        depth=5,
        eval_metric="AUC",
        random_seed=42 + fold,
        early_stopping_rounds=50,
        verbose=False,
    )
    cat_model.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=False)
    oof_cat[val_idx] = cat_model.predict_proba(X_va)[:, 1]
    test_cat += cat_model.predict_proba(X_test)[:, 1] / n_splits

    # Model 4: ExtraTrees
    et_model = ExtraTreesClassifier(
        n_estimators=400,
        max_depth=9,
        min_samples_split=8,
        min_samples_leaf=4,
        max_features="sqrt",
        random_state=42 + fold,
        n_jobs=-1,
    )
    et_model.fit(X_tr, y_tr)
    oof_et[val_idx] = et_model.predict_proba(X_va)[:, 1]
    test_et += et_model.predict_proba(X_test)[:, 1] / n_splits

print(f"LGBM OOF ROC AUC: {roc_auc_score(y, oof_lgb):.5f}")
print(f"XGB  OOF ROC AUC: {roc_auc_score(y, oof_xgb):.5f}")
print(f"CAT  OOF ROC AUC: {roc_auc_score(y, oof_cat):.5f}")
print(f"ET   OOF ROC AUC: {roc_auc_score(y, oof_et):.5f}")

# 4. Ensemble Weight Optimization
oof_matrix = np.column_stack([oof_lgb, oof_xgb, oof_cat, oof_et])
test_matrix = np.column_stack([test_lgb, test_xgb, test_cat, test_et])


def loss_func(weights):
    w = np.maximum(0, weights)
    if w.sum() == 0:
        return 0.0
    w = w / w.sum()
    blend = np.dot(oof_matrix, w)
    return -roc_auc_score(y, blend)


init_weights = [0.35, 0.25, 0.25, 0.15]
res = minimize(
    loss_func,
    init_weights,
    method="Nelder-Mead",
    options={"maxiter": 500, "disp": False},
)
opt_w = np.maximum(0, res.x)
opt_w /= opt_w.sum()

print(
    f"Optimal Weights: LGB={opt_w[0]:.3f}, XGB={opt_w[1]:.3f}, CAT={opt_w[2]:.3f}, ET={opt_w[3]:.3f}"
)

final_oof_preds = np.dot(oof_matrix, opt_w)
final_test_preds = np.dot(test_matrix, opt_w)

overall_auc = roc_auc_score(y, final_oof_preds)
print(f"Overall 5-Fold Ensemble OOF ROC AUC: {overall_auc:.5f}")

# 5. Save Submission
os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame({"id": np.arange(len(test_df)), "defects": final_test_preds})
submission_path = "./working/submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission successfully saved to {submission_path}")
