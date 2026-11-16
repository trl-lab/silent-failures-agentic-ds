import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from scipy.optimize import minimize
import lightgbm as lgb
import xgboost as xgb

try:
    from catboost import CatBoostClassifier

    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

from sklearn.ensemble import HistGradientBoostingClassifier

# Ensure working directory exists
os.makedirs("./working", exist_ok=True)

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")


# Feature engineering function
def build_features(df):
    X = pd.DataFrame(index=df.index)

    # Base user features list
    base_cols = [c[2:] for c in df.columns if c.startswith("A_")]

    # Extract raw differences and log-differences for each base feature
    for c in base_cols:
        col_A = f"A_{c}"
        col_B = f"B_{c}"
        val_A = df[col_A].values
        val_B = df[col_B].values

        log_A = np.log1p(np.maximum(0, val_A))
        log_B = np.log1p(np.maximum(0, val_B))

        X[f"diff_{c}"] = val_A - val_B
        X[f"log_diff_{c}"] = log_A - log_B
        X[f"rel_diff_{c}"] = (val_A - val_B) / (val_A + val_B + 1e-5)

    # Domain ratios per user
    for prefix in ["A", "B"]:
        f_fol = np.log1p(np.maximum(0, df[f"{prefix}_follower_count"].values))
        f_fow = np.log1p(np.maximum(0, df[f"{prefix}_following_count"].values))
        f_lis = np.log1p(np.maximum(0, df[f"{prefix}_listed_count"].values))
        f_mrec = np.log1p(np.maximum(0, df[f"{prefix}_mentions_received"].values))
        f_msent = np.log1p(np.maximum(0, df[f"{prefix}_mentions_sent"].values))
        f_rrec = np.log1p(np.maximum(0, df[f"{prefix}_retweets_received"].values))
        f_rsent = np.log1p(np.maximum(0, df[f"{prefix}_retweets_sent"].values))
        f_post = np.log1p(np.maximum(0, df[f"{prefix}_posts"].values))
        f_n1 = np.log1p(np.maximum(0, df[f"{prefix}_network_feature_1"].values))
        f_n2 = np.log1p(np.maximum(0, df[f"{prefix}_network_feature_2"].values))
        f_n3 = np.log1p(np.maximum(0, df[f"{prefix}_network_feature_3"].values))

        X[f"{prefix}_fol_fow_ratio"] = f_fol - f_fow
        X[f"{prefix}_listed_fol_ratio"] = f_lis - f_fol
        X[f"{prefix}_retweet_post_ratio"] = f_rrec - f_post
        X[f"{prefix}_mention_post_ratio"] = f_mrec - f_post
        X[f"{prefix}_retweet_fol_ratio"] = f_rrec - f_fol
        X[f"{prefix}_mention_fol_ratio"] = f_mrec - f_fol
        X[f"{prefix}_network_ratio_12"] = f_n1 - f_n2
        X[f"{prefix}_network_ratio_13"] = f_n1 - f_n3
        X[f"{prefix}_total_engagement"] = f_rrec + f_mrec + f_lis

    # Pairwise differences of domain ratios
    ratio_names = [
        "fol_fow_ratio",
        "listed_fol_ratio",
        "retweet_post_ratio",
        "mention_post_ratio",
        "retweet_fol_ratio",
        "mention_fol_ratio",
        "network_ratio_12",
        "network_ratio_13",
        "total_engagement",
    ]
    for r in ratio_names:
        X[f"diff_{r}"] = X[f"A_{r}"] - X[f"B_{r}"]

    return X


def swap_users(df):
    swapped = df.copy()
    a_cols = [c for c in df.columns if c.startswith("A_")]
    b_cols = [c for c in df.columns if c.startswith("B_")]
    rename_dict = {}
    for a, b in zip(a_cols, b_cols):
        rename_dict[a] = b
        rename_dict[b] = a
    swapped = swapped.rename(columns=rename_dict)
    if "Choice" in swapped.columns:
        swapped["Choice"] = 1 - swapped["Choice"]
    return swapped


# Build feature matrices
X_train_orig = build_features(train_df)
X_train_swap = build_features(swap_users(train_df))

X_test_orig = build_features(test_df)
X_test_swap = build_features(swap_users(test_df))

y_train = train_df["Choice"].values

n_samples = len(train_df)
n_test = len(test_df)

# Cross-validation setup
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_preds_lgb = np.zeros(n_samples)
oof_preds_xgb = np.zeros(n_samples)
oof_preds_cat = np.zeros(n_samples)

test_preds_lgb = np.zeros(n_test)
test_preds_xgb = np.zeros(n_test)
test_preds_cat = np.zeros(n_test)

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_orig, y_train)):
    # Train augmentation: Combine original and swapped pairs
    X_tr_orig_fold = X_train_orig.iloc[train_idx]
    X_tr_swap_fold = X_train_swap.iloc[train_idx]
    y_tr_fold = y_train[train_idx]

    X_tr_aug = pd.concat([X_tr_orig_fold, X_tr_swap_fold], axis=0).reset_index(
        drop=True
    )
    y_tr_aug = np.concatenate([y_tr_fold, 1 - y_tr_fold], axis=0)

    # Validation data
    X_va_orig_fold = X_train_orig.iloc[val_idx]
    X_va_swap_fold = X_train_swap.iloc[val_idx]

    # 1. LightGBM
    lgb_model = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        verbose=-1,
    )
    lgb_model.fit(
        X_tr_aug,
        y_tr_aug,
        eval_set=[(X_va_orig_fold, y_train[val_idx])],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )
    p_va_orig_lgb = lgb_model.predict_proba(X_va_orig_fold)[:, 1]
    p_va_swap_lgb = lgb_model.predict_proba(X_va_swap_fold)[:, 1]
    oof_preds_lgb[val_idx] = 0.5 * (p_va_orig_lgb + (1.0 - p_va_swap_lgb))

    p_te_orig_lgb = lgb_model.predict_proba(X_test_orig)[:, 1]
    p_te_swap_lgb = lgb_model.predict_proba(X_test_swap)[:, 1]
    test_preds_lgb += 0.5 * (p_te_orig_lgb + (1.0 - p_te_swap_lgb)) / skf.n_splits

    # 2. XGBoost
    xgb_model = xgb.XGBClassifier(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42 + fold,
        early_stopping_rounds=50,
    )
    xgb_model.fit(
        X_tr_aug, y_tr_aug, eval_set=[(X_va_orig_fold, y_train[val_idx])], verbose=False
    )
    p_va_orig_xgb = xgb_model.predict_proba(X_va_orig_fold)[:, 1]
    p_va_swap_xgb = xgb_model.predict_proba(X_va_swap_fold)[:, 1]
    oof_preds_xgb[val_idx] = 0.5 * (p_va_orig_xgb + (1.0 - p_va_swap_xgb))

    p_te_orig_xgb = xgb_model.predict_proba(X_test_orig)[:, 1]
    p_te_swap_xgb = xgb_model.predict_proba(X_test_swap)[:, 1]
    test_preds_xgb += 0.5 * (p_te_orig_xgb + (1.0 - p_te_swap_xgb)) / skf.n_splits

    # 3. CatBoost or HistGradientBoosting
    if HAS_CATBOOST:
        cat_model = CatBoostClassifier(
            iterations=1000,
            learning_rate=0.03,
            depth=6,
            random_seed=42 + fold,
            early_stopping_rounds=50,
            verbose=False,
        )
        cat_model.fit(X_tr_aug, y_tr_aug, eval_set=(X_va_orig_fold, y_train[val_idx]))
        p_va_orig_cat = cat_model.predict_proba(X_va_orig_fold)[:, 1]
        p_va_swap_cat = cat_model.predict_proba(X_va_swap_fold)[:, 1]
        oof_preds_cat[val_idx] = 0.5 * (p_va_orig_cat + (1.0 - p_va_swap_cat))

        p_te_orig_cat = cat_model.predict_proba(X_test_orig)[:, 1]
        p_te_swap_cat = cat_model.predict_proba(X_test_swap)[:, 1]
        test_preds_cat += 0.5 * (p_te_orig_cat + (1.0 - p_te_swap_cat)) / skf.n_splits
    else:
        hgb_model = HistGradientBoostingClassifier(
            max_iter=500, learning_rate=0.03, max_depth=6, random_state=42 + fold
        )
        hgb_model.fit(X_tr_aug, y_tr_aug)
        p_va_orig_cat = hgb_model.predict_proba(X_va_orig_fold)[:, 1]
        p_va_swap_cat = hgb_model.predict_proba(X_va_swap_fold)[:, 1]
        oof_preds_cat[val_idx] = 0.5 * (p_va_orig_cat + (1.0 - p_va_swap_cat))

        p_te_orig_cat = hgb_model.predict_proba(X_test_orig)[:, 1]
        p_te_swap_cat = hgb_model.predict_proba(X_test_swap)[:, 1]
        test_preds_cat += 0.5 * (p_te_orig_cat + (1.0 - p_te_swap_cat)) / skf.n_splits


# Optimize Ensemble Weights
def loss_func(weights):
    w1, w2, w3 = weights
    pred = w1 * oof_preds_lgb + w2 * oof_preds_xgb + w3 * oof_preds_cat
    acc = accuracy_score(y_train, (pred >= 0.5).astype(int))
    return -acc


res = minimize(
    loss_func, [0.33, 0.33, 0.34], bounds=[(0, 1), (0, 1), (0, 1)], method="SLSQP"
)
w1, w2, w3 = res.x / np.sum(res.x)

oof_ensemble = w1 * oof_preds_lgb + w2 * oof_preds_xgb + w3 * oof_preds_cat
cv_accuracy = accuracy_score(y_train, (oof_ensemble >= 0.5).astype(int))

print(f"5-Fold Cross-Validation Accuracy: {cv_accuracy:.5f}")

# Final Test Predictions
test_ensemble = w1 * test_preds_lgb + w2 * test_preds_xgb + w3 * test_preds_cat
final_test_binary = (test_ensemble >= 0.5).astype(int)

submission = pd.DataFrame(
    {
        "id": test_df["id"] if "id" in test_df.columns else np.arange(len(test_df)),
        "Choice": final_test_binary,
    }
)
submission.to_csv("./working/submission.csv", index=False)
