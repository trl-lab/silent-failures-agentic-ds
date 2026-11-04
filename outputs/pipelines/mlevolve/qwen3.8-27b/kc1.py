import pandas as pd
import numpy as np
import xgboost as xgb
from catboost import CatBoostClassifier, Pool
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata
import json
import os
import warnings
import joblib

warnings.filterwarnings("ignore")
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# ============================================================
# 1. LOAD DATA
# ============================================================
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")
print(f"Target distribution:\n{train_df['defects'].value_counts()}")
print(f"Target rate: {train_df['defects'].mean():.4f}")

# ============================================================
# 2. CREATE VALIDATION SPLIT (Stratified)
# ============================================================
train_df, val_df = train_test_split(
    train_df, test_size=0.2, random_state=42, stratify=train_df["defects"]
)
print(f"After split - Train: {train_df.shape}, Val: {val_df.shape}")
print(f"Val target rate: {val_df['defects'].mean():.4f}")

# ============================================================
# 3. FEATURE ENGINEERING
# ============================================================
target_col = "defects"
id_col = "id" if "id" in train_df.columns else None

exclude_cols = [target_col] + ([id_col] if id_col else [])
numeric_cols = [
    c
    for c in train_df.columns
    if c not in exclude_cols
    and train_df[c].dtype in ["int64", "float64", "int32", "float32"]
]

print(f"Numeric features: {len(numeric_cols)}")

# --- 3a. LOG TRANSFORM ---
cols_to_log = []
for col in numeric_cols:
    if train_df[col].min() >= 0:
        cols_to_log.append(col)

for df_ref in [train_df, val_df, test_df]:
    for col in cols_to_log:
        if col in df_ref.columns:
            df_ref[col + "_log"] = np.log1p(df_ref[col].clip(lower=0))

print(f"Applied log transform to {len(cols_to_log)} columns")


# --- 3b. RATIO FEATURES ---
def create_ratio_features(df):
    df = df.copy()

    def safe_ratio(numerator, denominator, name):
        denom = denominator.replace(0, np.nan)
        df[name] = (numerator / denom).fillna(0)
        return df

    if "v(g)" in df.columns and "loc" in df.columns:
        df = safe_ratio(df["v(g)"], df["loc"], "vg_per_loc")
    if "ev(g)" in df.columns and "loc" in df.columns:
        df = safe_ratio(df["ev(g)"], df["loc"], "evg_per_loc")
    if "iv(g)" in df.columns and "loc" in df.columns:
        df = safe_ratio(df["iv(g)"], df["loc"], "ivg_per_loc")

    for col in ["v", "l", "d", "i", "e", "b", "t", "n"]:
        if col in df.columns and "loc" in df.columns:
            df = safe_ratio(df[col], df["loc"], f"{col}_per_loc")

    if "lOComment" in df.columns and "lOCode" in df.columns:
        df = safe_ratio(df["lOComment"], df["lOCode"], "comment_to_code")
        df = safe_ratio(df["lOCode"], df["lOComment"], "code_to_comment")
    if "lOBlank" in df.columns and "lOCode" in df.columns:
        df = safe_ratio(df["lOBlank"], df["lOCode"], "blank_to_code")
    if "lOCodeAndComment" in df.columns and "lOCode" in df.columns:
        df = safe_ratio(df["lOCodeAndComment"], df["lOCode"], "codeandcomment_to_code")
    if "lOComment" in df.columns and "lOBlank" in df.columns:
        df = safe_ratio(df["lOComment"], df["lOBlank"], "comment_to_blank")

    if "uniq_Op" in df.columns and "total_Op" in df.columns:
        df = safe_ratio(df["uniq_Op"], df["total_Op"], "op_diversity")
    if "uniq_Opnd" in df.columns and "total_Opnd" in df.columns:
        df = safe_ratio(df["uniq_Opnd"], df["total_Opnd"], "opnd_diversity")
    if "total_Op" in df.columns and "total_Opnd" in df.columns:
        df = safe_ratio(df["total_Op"], df["total_Opnd"], "op_to_opnd")
    if "uniq_Op" in df.columns and "uniq_Opnd" in df.columns:
        df = safe_ratio(df["uniq_Op"], df["uniq_Opnd"], "uniq_op_to_opnd")

    if "branchCount" in df.columns and "loc" in df.columns:
        df = safe_ratio(df["branchCount"], df["loc"], "branch_per_loc")
    if "branchCount" in df.columns and "v(g)" in df.columns:
        df = safe_ratio(df["branchCount"], df["v(g)"], "branch_per_vg")

    if "ev(g)" in df.columns and "v(g)" in df.columns:
        df = safe_ratio(df["ev(g)"], df["v(g)"], "evg_to_vg")
    if "iv(g)" in df.columns and "v(g)" in df.columns:
        df = safe_ratio(df["iv(g)"], df["v(g)"], "ivg_to_vg")

    if "v" in df.columns and "l" in df.columns:
        df = safe_ratio(df["v"], df["l"], "v_to_l")
    if "e" in df.columns and "v" in df.columns:
        df = safe_ratio(df["e"], df["v"], "e_to_v")
    if "t" in df.columns and "e" in df.columns:
        df = safe_ratio(df["t"], df["e"], "t_to_e")
    if "b" in df.columns and "e" in df.columns:
        df = safe_ratio(df["b"], df["e"], "b_to_e")
    if "b" in df.columns and "v" in df.columns:
        df = safe_ratio(df["b"], df["v"], "b_to_v")

    if "v(g)" in df.columns and "v" in df.columns:
        df = safe_ratio(df["v(g)"], df["v"], "vg_to_halstead_v")
    if "v(g)" in df.columns and "e" in df.columns:
        df = safe_ratio(df["v(g)"], df["e"], "vg_to_halstead_e")
    if "n" in df.columns and "loc" in df.columns:
        df = safe_ratio(df["n"], df["loc"], "n_per_loc")

    for col in [
        "vg_per_loc",
        "comment_to_code",
        "op_diversity",
        "opnd_diversity",
        "branch_per_loc",
        "v_per_loc",
        "e_per_loc",
        "d_per_loc",
    ]:
        if col in df.columns:
            df[col + "_log"] = np.log1p(df[col].clip(lower=0))

    return df


train_df = create_ratio_features(train_df)
val_df = create_ratio_features(val_df)
test_df = create_ratio_features(test_df)


# --- 3c. INTERACTION / PRODUCT FEATURES ---
def create_interaction_features(df):
    df = df.copy()

    if "v(g)" in df.columns and "loc" in df.columns:
        df["vg_x_loc"] = df["v(g)"] * df["loc"]
    if "ev(g)" in df.columns and "loc" in df.columns:
        df["evg_x_loc"] = df["ev(g)"] * df["loc"]
    if "v" in df.columns and "d" in df.columns:
        df["v_x_d"] = df["v"] * df["d"]
    if "v(g)" in df.columns and "branchCount" in df.columns:
        df["vg_x_branch"] = df["v(g)"] * df["branchCount"]
    if "lOComment" in df.columns and "lOCode" in df.columns:
        df["comment_x_code"] = df["lOComment"] * df["lOCode"]

    if "v(g)" in df.columns and "ev(g)" in df.columns:
        df["vg_minus_evg"] = df["v(g)"] - df["ev(g)"]
    if "v(g)" in df.columns and "iv(g)" in df.columns:
        df["vg_minus_ivg"] = df["v(g)"] - df["iv(g)"]

    for col in ["vg_x_loc", "v_x_d", "vg_x_branch"]:
        if col in df.columns:
            df[col + "_log"] = np.log1p(df[col].clip(lower=0))

    return df


train_df = create_interaction_features(train_df)
val_df = create_interaction_features(val_df)
test_df = create_interaction_features(test_df)


# --- 3d. AGGREGATE / STATISTICAL FEATURES ---
def create_statistical_features(df):
    df = df.copy()

    if all(c in df.columns for c in ["lOCode", "lOComment", "lOBlank"]):
        df["total_lines"] = df["lOCode"] + df["lOComment"] + df["lOBlank"]
        df["code_pct"] = df["lOCode"] / df["total_lines"].replace(0, np.nan).fillna(1)
        df["comment_pct"] = df["lOComment"] / df["total_lines"].replace(
            0, np.nan
        ).fillna(1)
        df["blank_pct"] = df["lOBlank"] / df["total_lines"].replace(0, np.nan).fillna(1)

    if all(c in df.columns for c in ["total_Op", "total_Opnd"]):
        df["total_tokens"] = df["total_Op"] + df["total_Opnd"]
        df["op_pct"] = df["total_Op"] / df["total_tokens"].replace(0, np.nan).fillna(1)

    if "v(g)" in df.columns and "total_tokens" in df.columns:
        df["vg_per_token"] = df["v(g)"] / df["total_tokens"].replace(0, np.nan).fillna(
            1
        )

    if "e" in df.columns and "loc" in df.columns:
        df["e_per_loc_log"] = np.log1p(
            (df["e"] / df["loc"].replace(0, np.nan)).fillna(0)
        )

    return df


train_df = create_statistical_features(train_df)
val_df = create_statistical_features(val_df)
test_df = create_statistical_features(test_df)

# --- 3e. FINALIZE FEATURE SET ---
all_features = [
    c for c in train_df.columns if c not in [target_col] + ([id_col] if id_col else [])
]
feature_cols = [
    c
    for c in all_features
    if train_df[c].dtype in ["int64", "float64", "int32", "float32"]
]

for df_ref in [train_df, val_df, test_df]:
    for col in feature_cols:
        if col in df_ref.columns:
            df_ref[col] = df_ref[col].replace([np.inf, -np.inf], np.nan).fillna(0)

print(f"Total features after engineering: {len(feature_cols)}")

# Save for reproducibility
train_df.to_parquet("./working/train_processed.parquet", index=False)
val_df.to_parquet("./working/val_processed.parquet", index=False)
test_df.to_parquet("./working/test_processed.parquet", index=False)
with open("./working/feature_cols.json", "w") as f:
    json.dump(feature_cols, f)

# ============================================================
# 4. PREPARE MODEL INPUTS
# ============================================================
X_train = train_df[feature_cols].values.astype(np.float64)
y_train = train_df[target_col].values.astype(np.int64)
X_val = val_df[feature_cols].values.astype(np.float64)
y_val = val_df[target_col].values.astype(np.int64)
X_test = test_df[feature_cols].values.astype(np.float64)

print(f"\nTraining: {X_train.shape}, Validation: {X_val.shape}, Test: {X_test.shape}")
print(f"Train target rate: {y_train.mean():.4f}")
print(f"Val target rate: {y_val.mean():.4f}")

# ============================================================
# 5. MODEL CONFIGURATION
# ============================================================
scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

xgb_params_v1 = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "learning_rate": 0.05,
    "max_depth": 4,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "scale_pos_weight": scale_pos_weight,
    "n_jobs": -1,
    "seed": 42,
    "verbosity": 0,
}

xgb_params_v2 = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "learning_rate": 0.05,
    "max_depth": 5,
    "min_child_weight": 3,
    "subsample": 0.8,
    "colsample_bytree": 0.6,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "scale_pos_weight": scale_pos_weight,
    "n_jobs": -1,
    "seed": 42,
    "verbosity": 0,
}

catboost_params = {
    "iterations": 1000,
    "learning_rate": 0.05,
    "depth": 4,
    "l2_leaf_reg": 3,
    "auto_class_weights": "Balanced",
    "verbose": 0,
    "random_seed": 42,
    "eval_metric": "AUC",
}

lr_params = {
    "C": 0.1,
    "penalty": "l2",
    "max_iter": 1000,
    "random_state": 42,
}

seeds = [42, 123, 456]

# ============================================================
# 6. 5-FOLD CV: Train base models and collect OOF predictions
#    Multi-seed averaging for tree-based models
# ============================================================
n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

# OOF predictions for each model (multi-seed averaged)
xgb1_oof = np.zeros(len(X_train))
xgb2_oof = np.zeros(len(X_train))
cat_oof = np.zeros(len(X_train))
lr_oof = np.zeros(len(X_train))

# Validation predictions for each model (multi-seed averaged)
xgb1_val_cv = np.zeros(len(X_val))
xgb2_val_cv = np.zeros(len(X_val))
cat_val_cv = np.zeros(len(X_val))
lr_val_cv = np.zeros(len(X_val))

print("\n--- 5-Fold CV Training (Multi-Seed) ---")

for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train)):
    X_tr, X_va = X_train[tr_idx], X_train[va_idx]
    y_tr, y_va = y_train[tr_idx], y_train[va_idx]

    # --- XGBoost v1 (multi-seed) ---
    xgb1_fold_oof = np.zeros(len(X_va))
    xgb1_fold_val = np.zeros(len(X_val))
    for seed in seeds:
        params = xgb_params_v1.copy()
        params["seed"] = seed
        dtrain = xgb.DMatrix(X_tr, label=y_tr)
        dval = xgb.DMatrix(X_va, label=y_va)
        model = xgb.train(
            params, dtrain, num_boost_round=1000,
            evals=[(dval, "val")], early_stopping_rounds=50, verbose_eval=False,
        )
        xgb1_fold_oof += model.predict(xgb.DMatrix(X_va)) / len(seeds)
        xgb1_fold_val += model.predict(xgb.DMatrix(X_val)) / len(seeds)
    xgb1_oof[va_idx] = xgb1_fold_oof
    xgb1_val_cv += xgb1_fold_val / n_folds

    # --- XGBoost v2 (multi-seed) ---
    xgb2_fold_oof = np.zeros(len(X_va))
    xgb2_fold_val = np.zeros(len(X_val))
    for seed in seeds:
        params = xgb_params_v2.copy()
        params["seed"] = seed
        dtrain = xgb.DMatrix(X_tr, label=y_tr)
        dval = xgb.DMatrix(X_va, label=y_va)
        model = xgb.train(
            params, dtrain, num_boost_round=1000,
            evals=[(dval, "val")], early_stopping_rounds=50, verbose_eval=False,
        )
        xgb2_fold_oof += model.predict(xgb.DMatrix(X_va)) / len(seeds)
        xgb2_fold_val += model.predict(xgb.DMatrix(X_val)) / len(seeds)
    xgb2_oof[va_idx] = xgb2_fold_oof
    xgb2_val_cv += xgb2_fold_val / n_folds

    # --- CatBoost (multi-seed) ---
    cat_fold_oof = np.zeros(len(X_va))
    cat_fold_val = np.zeros(len(X_val))
    for seed in seeds:
        params = catboost_params.copy()
        params["random_seed"] = seed
        model = CatBoostClassifier(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=(X_va, y_va),
            early_stopping_rounds=50,
            verbose=0,
        )
        cat_fold_oof += model.predict_proba(X_va)[:, 1] / len(seeds)
        cat_fold_val += model.predict_proba(X_val)[:, 1] / len(seeds)
    cat_oof[va_idx] = cat_fold_oof
    cat_val_cv += cat_fold_val / n_folds

    # --- Logistic Regression (single fit, no seed variation needed) ---
    lr_model = LogisticRegression(**lr_params)
    lr_model.fit(X_tr, y_tr)
    lr_oof[va_idx] = lr_model.predict_proba(X_va)[:, 1]
    lr_val_cv += lr_model.predict_proba(X_val)[:, 1] / n_folds

    fold_xgb1_auc = roc_auc_score(y_va, xgb1_oof[va_idx])
    fold_xgb2_auc = roc_auc_score(y_va, xgb2_oof[va_idx])
    fold_cat_auc = roc_auc_score(y_va, cat_oof[va_idx])
    fold_lr_auc = roc_auc_score(y_va, lr_oof[va_idx])
    print(f"  Fold {fold+1}: XGB1={fold_xgb1_auc:.4f}, XGB2={fold_xgb2_auc:.4f}, "
          f"Cat={fold_cat_auc:.4f}, LR={fold_lr_auc:.4f}")

xgb1_oof_auc = roc_auc_score(y_train, xgb1_oof)
xgb2_oof_auc = roc_auc_score(y_train, xgb2_oof)
cat_oof_auc = roc_auc_score(y_train, cat_oof)
lr_oof_auc = roc_auc_score(y_train, lr_oof)
print(f"CV OOF AUC - XGB1: {xgb1_oof_auc:.4f}, XGB2: {xgb2_oof_auc:.4f}, "
      f"CatBoost: {cat_oof_auc:.4f}, LR: {lr_oof_auc:.4f}")

# ============================================================
# 7. RANK-BASED ENSEMBLE on OOF predictions
# ============================================================
def rank_average(*preds):
    """Rank-based averaging of multiple prediction arrays."""
    ranked = np.column_stack([rankdata(p) for p in preds])
    return ranked.mean(axis=1)

# Rank-average OOF predictions
ensemble_oof = rank_average(xgb1_oof, xgb2_oof, cat_oof, lr_oof)
ensemble_oof_auc = roc_auc_score(y_train, ensemble_oof)
print(f"Rank-averaged Ensemble OOF AUC: {ensemble_oof_auc:.4f}")

# Rank-average validation predictions
ensemble_val_cv = rank_average(xgb1_val_cv, xgb2_val_cv, cat_val_cv, lr_val_cv)
ensemble_val_cv_auc = roc_auc_score(y_val, ensemble_val_cv)
print(f"Rank-averaged Ensemble Val AUC: {ensemble_val_cv_auc:.4f}")

# ============================================================
# 8. TRAIN FINAL MODELS ON FULL TRAINING DATA
#    (Multi-seed, early stopping on validation)
# ============================================================
print("\n--- Training Final Models on Full Training Data ---")

# --- XGBoost v1 (multi-seed) ---
xgb1_test_preds = np.zeros(len(X_test))
xgb1_val_preds = np.zeros(len(X_val))
for seed in seeds:
    params = xgb_params_v1.copy()
    params["seed"] = seed
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    model = xgb.train(
        params, dtrain, num_boost_round=1000,
        evals=[(dval, "val")], early_stopping_rounds=50, verbose_eval=False,
    )
    xgb1_test_preds += model.predict(xgb.DMatrix(X_test)) / len(seeds)
    xgb1_val_preds += model.predict(xgb.DMatrix(X_val)) / len(seeds)
print(f"  XGBoost v1 done (best_iter ~{model.best_iteration})")

# --- XGBoost v2 (multi-seed) ---
xgb2_test_preds = np.zeros(len(X_test))
xgb2_val_preds = np.zeros(len(X_val))
for seed in seeds:
    params = xgb_params_v2.copy()
    params["seed"] = seed
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    model = xgb.train(
        params, dtrain, num_boost_round=1000,
        evals=[(dval, "val")], early_stopping_rounds=50, verbose_eval=False,
    )
    xgb2_test_preds += model.predict(xgb.DMatrix(X_test)) / len(seeds)
    xgb2_val_preds += model.predict(xgb.DMatrix(X_val)) / len(seeds)
print(f"  XGBoost v2 done (best_iter ~{model.best_iteration})")

# --- CatBoost (multi-seed) ---
cat_test_preds = np.zeros(len(X_test))
cat_val_preds = np.zeros(len(X_val))
for seed in seeds:
    params = catboost_params.copy()
    params["random_seed"] = seed
    model = CatBoostClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        early_stopping_rounds=50,
        verbose=0,
    )
    cat_test_preds += model.predict_proba(X_test)[:, 1] / len(seeds)
    cat_val_preds += model.predict_proba(X_val)[:, 1] / len(seeds)
print(f"  CatBoost done")

# --- Logistic Regression ---
lr_model = LogisticRegression(**lr_params)
lr_model.fit(X_train, y_train)
lr_test_preds = lr_model.predict_proba(X_test)[:, 1]
lr_val_preds = lr_model.predict_proba(X_val)[:, 1]
print(f"  Logistic Regression done")

# ============================================================
# 9. GENERATE PREDICTIONS (Rank-Based Averaging)
# ============================================================
val_pred = rank_average(xgb1_val_preds, xgb2_val_preds, cat_val_preds, lr_val_preds)
test_pred = rank_average(xgb1_test_preds, xgb2_test_preds, cat_test_preds, lr_test_preds)

# Convert rank-averaged scores back to probability scale for submission
# Normalize to [0, 1] range
test_pred = (test_pred - test_pred.min()) / (test_pred.max() - test_pred.min() + 1e-8)

# ============================================================
# 10. EVALUATE & SELECT BEST APPROACH
# ============================================================
print(f"\n--- Validation Results (OOF AUC) ---")
print(f"  XGBoost v1 OOF AUC:  {xgb1_oof_auc:.4f}")
print(f"  XGBoost v2 OOF AUC:  {xgb2_oof_auc:.4f}")
print(f"  CatBoost OOF AUC:    {cat_oof_auc:.4f}")
print(f"  Logistic Reg OOF AUC:{lr_oof_auc:.4f}")
print(f"  Rank Ensemble OOF AUC: {ensemble_oof_auc:.4f}")

# Use rank-averaged ensemble as final prediction
final_test_pred = test_pred
print("  Selected: Rank-averaged ensemble")

# ============================================================
# 11. SAVE SUBMISSION
# ============================================================
if "id" in test_df.columns:
    test_ids = test_df["id"].values
else:
    test_ids = np.arange(len(final_test_pred))

submission = pd.DataFrame({"id": test_ids.astype(int), "defects": final_test_pred})
submission.to_csv("./submission/submission.csv", index=False)
print(f"\nSubmission saved: {submission.shape}")
print(f"Prediction range: [{final_test_pred.min():.4f}, {final_test_pred.max():.4f}]")
print(f"Prediction mean: {final_test_pred.mean():.4f}")

# Save models
joblib.dump(lr_model, "./working/lr_model.pkl")

# ============================================================
# 12. FINAL VALIDATION SCORE
# ============================================================
score = roc_auc_score(y_val, val_pred)
print(f"Final Validation Score: {score}")