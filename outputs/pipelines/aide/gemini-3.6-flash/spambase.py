import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from xgboost import XGBClassifier
from scipy.stats import rankdata
from scipy.optimize import minimize
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    ExtraTreesClassifier,
    RandomForestClassifier,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# 1. Load Data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "class"
y = train_df[target_col].astype(int).values

if "id" in test_df.columns:
    test_ids = test_df["id"].values
    test_features_df = test_df.drop(columns=["id"])
else:
    test_ids = np.arange(len(test_df))
    test_features_df = test_df.copy()

if "id" in train_df.columns:
    train_features_df = train_df.drop(columns=["id", target_col])
else:
    train_features_df = train_df.drop(columns=[target_col])

# Domain specific spam and ham words from Spambase dictionary
SPAM_WORDS = [
    "word_freq_make",
    "word_freq_address",
    "word_freq_all",
    "word_freq_3d",
    "word_freq_our",
    "word_freq_over",
    "word_freq_remove",
    "word_freq_internet",
    "word_freq_order",
    "word_freq_mail",
    "word_freq_receive",
    "word_freq_will",
    "word_freq_people",
    "word_freq_report",
    "word_freq_addresses",
    "word_freq_free",
    "word_freq_business",
    "word_freq_email",
    "word_freq_you",
    "word_freq_credit",
    "word_freq_your",
    "word_freq_font",
    "word_freq_000",
    "word_freq_money",
]

HAM_WORDS = [
    "word_freq_hp",
    "word_freq_hpl",
    "word_freq_george",
    "word_freq_650",
    "word_freq_lab",
    "word_freq_labs",
    "word_freq_telnet",
    "word_freq_857",
    "word_freq_415",
    "word_freq_85",
    "word_freq_technology",
    "word_freq_1999",
    "word_freq_parts",
    "word_freq_pm",
    "word_freq_direct",
    "word_freq_cs",
    "word_freq_meeting",
    "word_freq_original",
    "word_freq_project",
    "word_freq_re",
    "word_freq_edu",
    "word_freq_table",
    "word_freq_conference",
]


# 2. Advanced Feature Engineering
def create_features(df):
    data = df.copy()

    word_cols = [c for c in data.columns if c.startswith("word_freq_")]
    char_cols = [c for c in data.columns if c.startswith("char_freq_")]

    # Word aggregations
    if word_cols:
        data["word_freq_sum"] = data[word_cols].sum(axis=1)
        data["word_freq_mean"] = data[word_cols].mean(axis=1)
        data["word_freq_std"] = data[word_cols].std(axis=1)
        data["word_freq_max"] = data[word_cols].max(axis=1)
        data["word_freq_nonzero_cnt"] = (data[word_cols] > 0).sum(axis=1)

        sum_denom = data["word_freq_sum"] + 1e-5
        for col in word_cols:
            data[f"{col}_rel"] = data[col] / sum_denom

    # Domain Ham vs Spam aggregate indicators
    valid_spam_words = [c for c in SPAM_WORDS if c in data.columns]
    valid_ham_words = [c for c in HAM_WORDS if c in data.columns]

    if valid_spam_words and valid_ham_words:
        data["spam_words_sum"] = data[valid_spam_words].sum(axis=1)
        data["ham_words_sum"] = data[valid_ham_words].sum(axis=1)
        data["spam_ham_diff"] = data["spam_words_sum"] - data["ham_words_sum"]
        data["spam_ham_log_ratio"] = np.log1p(data["spam_words_sum"]) - np.log1p(
            data["ham_words_sum"]
        )

    # Character aggregations
    if char_cols:
        data["char_freq_sum"] = data[char_cols].sum(axis=1)
        data["char_freq_mean"] = data[char_cols].mean(axis=1)
        data["char_freq_nonzero_cnt"] = (data[char_cols] > 0).sum(axis=1)

        char_denom = data["char_freq_sum"] + 1e-5
        for col in char_cols:
            data[f"{col}_rel"] = data[col] / char_denom

    # Interaction features
    if "char_freq_!" in data.columns and "char_freq_$" in data.columns:
        data["excl_dollar_prod"] = data["char_freq_!"] * data["char_freq_$"]
    if "word_freq_free" in data.columns and "word_freq_your" in data.columns:
        data["free_your_prod"] = data["word_freq_free"] * data["word_freq_your"]
    if "word_freq_remove" in data.columns and "word_freq_money" in data.columns:
        data["remove_money_prod"] = data["word_freq_remove"] * data["word_freq_money"]

    # Capital run lengths features
    if (
        "capital_run_length_total" in data.columns
        and "capital_run_length_longest" in data.columns
    ):
        data["cap_longest_ratio"] = data["capital_run_length_longest"] / (
            data["capital_run_length_total"] + 1.0
        )
    if (
        "capital_run_length_average" in data.columns
        and "capital_run_length_longest" in data.columns
    ):
        data["cap_avg_longest_ratio"] = data["capital_run_length_average"] / (
            data["capital_run_length_longest"] + 1.0
        )

    # Continuous Log1p transforms
    for col in list(data.columns):
        if "freq" in col or "capital" in col:
            data[f"{col}_log1p"] = np.log1p(np.maximum(0, data[col]))

    return data


X_train = create_features(train_features_df)
X_test = create_features(test_features_df)
X_test = X_test[X_train.columns]

# 3. Stratified 5-Fold Cross Validation Setup
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_preds_lgb = np.zeros(len(X_train))
oof_preds_hgb = np.zeros(len(X_train))
oof_preds_et = np.zeros(len(X_train))
oof_preds_rf = np.zeros(len(X_train))
oof_preds_xgb = np.zeros(len(X_train))

test_preds_lgb = np.zeros(len(X_test))
test_preds_hgb = np.zeros(len(X_test))
test_preds_et = np.zeros(len(X_test))
test_preds_rf = np.zeros(len(X_test))
test_preds_xgb = np.zeros(len(X_test))

lgb_params = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "learning_rate": 0.02,
    "num_leaves": 45,
    "colsample_bytree": 0.8,
    "subsample": 0.8,
    "random_state": 42,
    "verbose": -1,
    "n_estimators": 1200,
}

xgb_params = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "learning_rate": 0.02,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "tree_method": "hist",
    "random_state": 42,
    "n_estimators": 1200,
    "early_stopping_rounds": 50,
    "n_jobs": -1,
}

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y)):
    X_tr, y_tr = X_train.iloc[train_idx], y[train_idx]
    X_va, y_va = X_train.iloc[val_idx], y[val_idx]

    # Model 1: LightGBM
    model_lgb = lgb.LGBMClassifier(**lgb_params)
    model_lgb.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )
    oof_preds_lgb[val_idx] = model_lgb.predict_proba(X_va)[:, 1]
    test_preds_lgb += model_lgb.predict_proba(X_test)[:, 1] / skf.n_splits

    # Model 2: HistGradientBoosting
    model_hgb = HistGradientBoostingClassifier(
        max_iter=400,
        learning_rate=0.03,
        l2_regularization=1.0,
        random_state=42 + fold,
    )
    model_hgb.fit(X_tr, y_tr)
    oof_preds_hgb[val_idx] = model_hgb.predict_proba(X_va)[:, 1]
    test_preds_hgb += model_hgb.predict_proba(X_test)[:, 1] / skf.n_splits

    # Model 3: ExtraTrees
    model_et = ExtraTreesClassifier(
        n_estimators=300, max_features="sqrt", random_state=42 + fold, n_jobs=-1
    )
    model_et.fit(X_tr, y_tr)
    oof_preds_et[val_idx] = model_et.predict_proba(X_va)[:, 1]
    test_preds_et += model_et.predict_proba(X_test)[:, 1] / skf.n_splits

    # Model 4: RandomForest
    model_rf = RandomForestClassifier(
        n_estimators=300, max_features="sqrt", random_state=42 + fold, n_jobs=-1
    )
    model_rf.fit(X_tr, y_tr)
    oof_preds_rf[val_idx] = model_rf.predict_proba(X_va)[:, 1]
    test_preds_rf += model_rf.predict_proba(X_test)[:, 1] / skf.n_splits

    # Model 5: XGBoost
    fold_xgb_params = xgb_params.copy()
    fold_xgb_params["random_state"] = 42 + fold
    model_xgb = XGBClassifier(**fold_xgb_params)
    model_xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    oof_preds_xgb[val_idx] = model_xgb.predict_proba(X_va)[:, 1]
    test_preds_xgb += model_xgb.predict_proba(X_test)[:, 1] / skf.n_splits


# 4. Rank Normalization
def to_ranks(x):
    return rankdata(x) / len(x)


oof_lgb_rank = to_ranks(oof_preds_lgb)
oof_hgb_rank = to_ranks(oof_preds_hgb)
oof_et_rank = to_ranks(oof_preds_et)
oof_rf_rank = to_ranks(oof_preds_rf)
oof_xgb_rank = to_ranks(oof_preds_xgb)

test_lgb_rank = to_ranks(test_preds_lgb)
test_hgb_rank = to_ranks(test_preds_hgb)
test_et_rank = to_ranks(test_preds_et)
test_rf_rank = to_ranks(test_preds_rf)
test_xgb_rank = to_ranks(test_preds_xgb)

OOF_RANKS = np.column_stack(
    [oof_lgb_rank, oof_hgb_rank, oof_et_rank, oof_rf_rank, oof_xgb_rank]
)
TEST_RANKS = np.column_stack(
    [test_lgb_rank, test_hgb_rank, test_et_rank, test_rf_rank, test_xgb_rank]
)


# 5. SciPy Optimization for Max ROC AUC Blending Weights
def loss_func(weights):
    weights = np.array(weights)
    weights = np.maximum(0, weights)
    w_sum = weights.sum()
    if w_sum > 0:
        weights /= w_sum
    blend = OOF_RANKS @ weights
    return -roc_auc_score(y, blend)


init_weights = [0.30, 0.25, 0.15, 0.15, 0.15]
res = minimize(loss_func, init_weights, method="Nelder-Mead", options={"maxiter": 500})
opt_weights = np.maximum(0, res.x)
opt_weights /= opt_weights.sum()

oof_blend = OOF_RANKS @ opt_weights
test_blend = TEST_RANKS @ opt_weights

# 6. Evaluation and Submission
cv_roc_auc = roc_auc_score(y, oof_blend)
print(f"5-Fold Optimized 5-Model Rank-Blended CV ROC AUC Score: {cv_roc_auc:.6f}")

os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame({"id": test_ids, "class": test_blend})
submission.to_csv("./working/submission.csv", index=False)
