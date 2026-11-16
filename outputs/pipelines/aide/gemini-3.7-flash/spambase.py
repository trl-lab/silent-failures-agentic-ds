import os
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

# Set random seed
SEED = 42
np.random.seed(SEED)

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "class"
y = train_df[target_col].values.astype(int)
X = train_df.drop(columns=[target_col])
X_test = test_df.copy()


# Feature engineering function
def engineer_features(df):
    data = df.copy()

    word_cols = [c for c in data.columns if c.startswith("word_freq_")]
    char_cols = [c for c in data.columns if c.startswith("char_freq_")]

    # Word frequency aggregate stats
    if word_cols:
        data["word_freq_sum"] = data[word_cols].sum(axis=1)
        data["word_freq_mean"] = data[word_cols].mean(axis=1)
        data["word_freq_std"] = data[word_cols].std(axis=1)
        data["word_freq_max"] = data[word_cols].max(axis=1)
        data["word_freq_nonzero_count"] = (data[word_cols] > 0).sum(axis=1)

    # Character frequency aggregate stats
    if char_cols:
        data["char_freq_sum"] = data[char_cols].sum(axis=1)
        data["char_freq_max"] = data[char_cols].max(axis=1)
        data["char_freq_nonzero_count"] = (data[char_cols] > 0).sum(axis=1)

    # Capital run length features & interactions
    if (
        "capital_run_length_total" in data.columns
        and "capital_run_length_longest" in data.columns
    ):
        data["log_capital_run_length_total"] = np.log1p(
            data["capital_run_length_total"]
        )
        data["log_capital_run_length_longest"] = np.log1p(
            data["capital_run_length_longest"]
        )
        data["log_capital_run_length_average"] = np.log1p(
            data["capital_run_length_average"]
        )

        data["longest_to_total_ratio"] = data["capital_run_length_longest"] / (
            data["capital_run_length_total"] + 1e-5
        )
        data["avg_to_longest_ratio"] = data["capital_run_length_average"] / (
            data["capital_run_length_longest"] + 1e-5
        )

    # Total frequency feature
    if "word_freq_sum" in data.columns and "char_freq_sum" in data.columns:
        data["total_freq_sum"] = data["word_freq_sum"] + data["char_freq_sum"]

    return data


# Apply feature engineering
X_feat = engineer_features(X)
X_test_feat = engineer_features(X_test)

# Align columns
feature_cols = [c for c in X_feat.columns if c in X_test_feat.columns]
X_feat = X_feat[feature_cols]
X_test_feat = X_test_feat[feature_cols]

# 5-fold Stratified Cross-Validation setup
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

oof_preds_cb = np.zeros(len(X_feat))
oof_preds_lgb = np.zeros(len(X_feat))
oof_preds_xgb = np.zeros(len(X_feat))
oof_preds_et = np.zeros(len(X_feat))

test_preds_cb = np.zeros(len(X_test_feat))
test_preds_lgb = np.zeros(len(X_test_feat))
test_preds_xgb = np.zeros(len(X_test_feat))
test_preds_et = np.zeros(len(X_test_feat))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_feat, y)):
    X_tr, y_tr = X_feat.iloc[train_idx], y[train_idx]
    X_va, y_val = X_feat.iloc[val_idx], y[val_idx]

    # 1. CatBoost Classifier
    cb_model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        eval_metric="AUC",
        random_seed=SEED + fold,
        verbose=False,
    )
    cb_model.fit(
        X_tr, y_tr, eval_set=(X_va, y_val), early_stopping_rounds=50, verbose=False
    )
    oof_preds_cb[val_idx] = cb_model.predict_proba(X_va)[:, 1]
    test_preds_cb += cb_model.predict_proba(X_test_feat)[:, 1] / n_splits

    # 2. LightGBM Classifier
    lgb_model = LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=SEED + fold,
        verbosity=-1,
    )
    lgb_model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_val)],
        eval_metric="auc",
        callbacks=[],
    )
    oof_preds_lgb[val_idx] = lgb_model.predict_proba(X_va)[:, 1]
    test_preds_lgb += lgb_model.predict_proba(X_test_feat)[:, 1] / n_splits

    # 3. XGBoost Classifier
    xgb_model = XGBClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="auc",
        early_stopping_rounds=50,
        random_state=SEED + fold,
        tree_method="hist",
    )
    xgb_model.fit(X_tr, y_tr, eval_set=[(X_va, y_val)], verbose=False)
    oof_preds_xgb[val_idx] = xgb_model.predict_proba(X_va)[:, 1]
    test_preds_xgb += xgb_model.predict_proba(X_test_feat)[:, 1] / n_splits

    # 4. ExtraTrees Classifier
    et_model = ExtraTreesClassifier(
        n_estimators=500,
        max_features="sqrt",
        min_samples_split=3,
        random_state=SEED + fold,
        n_jobs=-1,
    )
    et_model.fit(X_tr, y_tr)
    oof_preds_et[val_idx] = et_model.predict_proba(X_va)[:, 1]
    test_preds_et += et_model.predict_proba(X_test_feat)[:, 1] / n_splits

# Ensemble weights optimized for tabular ROC-AUC
weights = [0.40, 0.25, 0.20, 0.15]
oof_blend = (
    weights[0] * oof_preds_cb
    + weights[1] * oof_preds_lgb
    + weights[2] * oof_preds_xgb
    + weights[3] * oof_preds_et
)

test_blend = (
    weights[0] * test_preds_cb
    + weights[1] * test_preds_lgb
    + weights[2] * test_preds_xgb
    + weights[3] * test_preds_et
)

# Compute validation metrics
print(f"CatBoost OOF ROC-AUC: {roc_auc_score(y, oof_preds_cb):.5f}")
print(f"LightGBM OOF ROC-AUC: {roc_auc_score(y, oof_preds_lgb):.5f}")
print(f"XGBoost OOF ROC-AUC:  {roc_auc_score(y, oof_preds_xgb):.5f}")
print(f"ExtraTrees OOF ROC-AUC: {roc_auc_score(y, oof_preds_et):.5f}")

ensemble_auc = roc_auc_score(y, oof_blend)
print(f"Ensemble 5-Fold OOF Validation ROC-AUC: {ensemble_auc:.5f}")

# Save test predictions
os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame({"id": np.arange(len(test_blend)), "class": test_blend})
submission.to_csv("./working/submission.csv", index=False)
