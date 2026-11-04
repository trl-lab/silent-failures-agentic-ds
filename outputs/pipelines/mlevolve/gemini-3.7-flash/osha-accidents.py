import os
import re
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score
from scipy.optimize import minimize
from scipy.stats import rankdata
from scipy.sparse import hstack
import lightgbm as lgb
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")

# 0. Setup Directories
os.makedirs("./working/models", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# 1. Load Data
print("Loading train and test datasets...")
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")
print(f"Loaded Train Shape: {train_df.shape} | Test Shape: {test_df.shape}")

target_col = "Task Assigned"
if target_col not in train_df.columns:
    for col in train_df.columns:
        if col.strip().lower() == target_col.lower():
            train_df.rename(columns={col: target_col}, inplace=True)
            break

# Process Target
if train_df[target_col].dtype == object or train_df[target_col].dtype == bool:
    unique_vals = train_df[target_col].dropna().unique()
    mapping = {}
    for val in unique_vals:
        val_str = str(val).strip().lower()
        if val_str in ["1", "true", "yes", "y", "t", "assigned"]:
            mapping[val] = 1
        elif val_str in ["0", "false", "no", "n", "f", "not assigned", "unassigned"]:
            mapping[val] = 0
    if len(mapping) == len(unique_vals):
        y = train_df[target_col].map(mapping).astype(int)
    else:
        y = pd.factorize(train_df[target_col])[0]
else:
    y = train_df[target_col].astype(int)

train_df["target"] = y
test_df["target"] = -1

# Identify ID columns
id_candidates = [
    col
    for col in train_df.columns
    if col.lower() in ["id", "index", "case_id", "rel_id"]
]
id_col = id_candidates[0] if id_candidates else None
feature_cols = [
    c
    for c in train_df.columns
    if c not in [target_col, "target", id_col] and c is not None
]

# 2. Identify Column Modalities
text_cols = []
datetime_cols = []
cat_cols = []
num_cols = []

for col in feature_cols:
    if train_df[col].dtype == "object" or str(train_df[col].dtype).startswith(
        "datetime"
    ):
        sample = train_df[col].dropna().astype(str).head(100)
        is_date = False
        if len(sample) > 0:
            try:
                pd.to_datetime(sample, format="mixed")
                is_date = True
            except Exception:
                is_date = False

        if is_date:
            datetime_cols.append(col)
            continue

        mean_len = sample.apply(len).mean() if len(sample) > 0 else 0
        nunique_ratio = train_df[col].nunique() / max(len(train_df), 1)
        if mean_len > 35 or (nunique_ratio > 0.6 and mean_len > 15):
            text_cols.append(col)
        else:
            cat_cols.append(col)
    elif np.issubdtype(train_df[col].dtype, np.number):
        if train_df[col].nunique() <= 5 and train_df[col].nunique() > 1:
            cat_cols.append(col)
        else:
            num_cols.append(col)
    else:
        cat_cols.append(col)

print(
    f"Detected modalities -> Datetime: {len(datetime_cols)}, Text: {len(text_cols)}, Categorical: {len(cat_cols)}, Numerical: {len(num_cols)}"
)

# 3. Stratified K-Fold Split
n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
train_df["fold"] = -1
for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df["target"])):
    train_df.loc[val_idx, "fold"] = fold

# 4. Feature Engineering Pipeline
train_feat_list = []
test_feat_list = []

# A. Datetime Feature Extraction
for col in datetime_cols:
    for df in [train_df, test_df]:
        dt_series = pd.to_datetime(df[col], errors="coerce", format="mixed")
        df[f"{col}__year"] = dt_series.dt.year.fillna(-1).astype(int)
        df[f"{col}__month"] = dt_series.dt.month.fillna(-1).astype(int)
        df[f"{col}__day"] = dt_series.dt.day.fillna(-1).astype(int)
        df[f"{col}__dayofweek"] = dt_series.dt.dayofweek.fillna(-1).astype(int)
        df[f"{col}__quarter"] = dt_series.dt.quarter.fillna(-1).astype(int)
        df[f"{col}__is_weekend"] = dt_series.dt.dayofweek.isin([5, 6]).astype(int)
        num_cols.extend(
            [
                f"{col}__year",
                f"{col}__month",
                f"{col}__day",
                f"{col}__dayofweek",
                f"{col}__quarter",
                f"{col}__is_weekend",
            ]
        )

# B. Numerical Features & Missingness Indicators
num_cols = list(dict.fromkeys(num_cols))
train_num = pd.DataFrame(index=train_df.index)
test_num = pd.DataFrame(index=test_df.index)

for col in num_cols:
    train_num[f"{col}__isna"] = train_df[col].isna().astype(np.int8)
    test_num[f"{col}__isna"] = test_df[col].isna().astype(np.int8)

    med = train_df[col].median()
    if pd.isna(med):
        med = 0.0
    train_val = train_df[col].fillna(med)
    test_val = test_df[col].fillna(med)

    train_num[col] = train_val.astype(np.float32)
    test_num[col] = test_val.astype(np.float32)

    if (train_val >= 0).all() and train_val.skew() > 2.0:
        train_num[f"{col}__log1p"] = np.log1p(train_val).astype(np.float32)
        test_num[f"{col}__log1p"] = np.log1p(test_val).astype(np.float32)

train_feat_list.append(train_num)
test_feat_list.append(test_num)

# C. Categorical Features, 2-Way Interactions & Group-by Aggregations
cat_cols = list(dict.fromkeys(cat_cols))
if len(cat_cols) > 0:
    train_cat = pd.DataFrame(index=train_df.index)
    test_cat = pd.DataFrame(index=test_df.index)
    global_mean = float(train_df["target"].mean())

    for col in cat_cols:
        train_str = train_df[col].astype(str).fillna("__MISSING__")
        test_str = test_df[col].astype(str).fillna("__MISSING__")

        # Frequency Encoding
        freq_map = train_str.value_counts(normalize=True).to_dict()
        train_cat[f"{col}__freq"] = train_str.map(freq_map).fillna(0).astype(np.float32)
        test_cat[f"{col}__freq"] = test_str.map(freq_map).fillna(0).astype(np.float32)

        # Categorical Index Encoding
        all_cats = pd.Series(
            list(train_str.unique()) + list(test_str.unique())
        ).unique()
        cat_map = {cat: idx for idx, cat in enumerate(all_cats)}
        train_cat[f"{col}__cat"] = train_str.map(cat_map).astype(np.int32)
        test_cat[f"{col}__cat"] = test_str.map(cat_map).astype(np.int32)

        # Leak-Free Out-of-Fold Target Encoding
        oof_te = np.zeros(len(train_df), dtype=np.float32)
        for fold in range(n_folds):
            tr_mask = train_df["fold"] != fold
            va_mask = train_df["fold"] == fold

            counts = train_str[tr_mask].value_counts()
            means = train_df.loc[tr_mask, "target"].groupby(train_str[tr_mask]).mean()
            smooth_te = (counts * means + 10.0 * global_mean) / (counts + 10.0)
            oof_te[va_mask] = (
                train_str[va_mask].map(smooth_te).fillna(global_mean).values
            )

        counts_full = train_str.value_counts()
        means_full = train_df["target"].groupby(train_str).mean()
        smooth_te_full = (counts_full * means_full + 10.0 * global_mean) / (
            counts_full + 10.0
        )

        train_cat[f"{col}__te"] = oof_te
        test_cat[f"{col}__te"] = (
            test_str.map(smooth_te_full).fillna(global_mean).astype(np.float32)
        )

    # 2-Way Categorical Interactions (Top Categorical Pairs)
    top_cats = [c for c in cat_cols if train_df[c].nunique() > 1][:6]
    for i in range(len(top_cats)):
        for j in range(i + 1, len(top_cats)):
            c1, c2 = top_cats[i], top_cats[j]
            pair_name = f"{c1}_x_{c2}"
            tr_pair = train_df[c1].astype(str) + "__" + train_df[c2].astype(str)
            te_pair = test_df[c1].astype(str) + "__" + test_df[c2].astype(str)

            # Interaction Frequency
            pair_freq = tr_pair.value_counts(normalize=True).to_dict()
            train_cat[f"{pair_name}__freq"] = tr_pair.map(pair_freq).fillna(0).astype(np.float32)
            test_cat[f"{pair_name}__freq"] = te_pair.map(pair_freq).fillna(0).astype(np.float32)

            # Interaction OOF Target Encoding
            oof_pair_te = np.zeros(len(train_df), dtype=np.float32)
            for fold in range(n_folds):
                tr_m = train_df["fold"] != fold
                va_m = train_df["fold"] == fold
                cnts = tr_pair[tr_m].value_counts()
                mns = train_df.loc[tr_m, "target"].groupby(tr_pair[tr_m]).mean()
                sm_te = (cnts * mns + 15.0 * global_mean) / (cnts + 15.0)
                oof_pair_te[va_m] = tr_pair[va_m].map(sm_te).fillna(global_mean).values

            cnts_f = tr_pair.value_counts()
            mns_f = train_df["target"].groupby(tr_pair).mean()
            sm_f = (cnts_f * mns_f + 15.0 * global_mean) / (cnts_f + 15.0)
            train_cat[f"{pair_name}__te"] = oof_pair_te
            test_cat[f"{pair_name}__te"] = te_pair.map(sm_f).fillna(global_mean).astype(np.float32)

    # Group-by Statistical Aggregations (mean, std, min, max) of Numerical by Top Categoricals
    base_num_cols = [c for c in num_cols if not c.endswith("__isna") and not c.endswith("__log1p")][:5]
    group_cats = [c for c in cat_cols if 2 <= train_df[c].nunique() <= 50][:4]

    for gcol in group_cats:
        for ncol in base_num_cols:
            grouped = train_df.groupby(gcol)[ncol]
            stats = grouped.agg(["mean", "std", "min", "max"]).reset_index()
            stats.columns = [gcol, f"{gcol}_{ncol}_mean", f"{gcol}_{ncol}_std", f"{gcol}_{ncol}_min", f"{gcol}_{ncol}_max"]

            tr_m = train_df[[gcol]].merge(stats, on=gcol, how="left")
            te_m = test_df[[gcol]].merge(stats, on=gcol, how="left")

            for agg in ["mean", "std", "min", "max"]:
                feat_name = f"{gcol}_{ncol}_{agg}"
                train_cat[feat_name] = tr_m[feat_name].fillna(train_df[ncol].mean()).astype(np.float32)
                test_cat[feat_name] = te_m[feat_name].fillna(train_df[ncol].mean()).astype(np.float32)

    train_feat_list.append(train_cat)
    test_feat_list.append(test_cat)

# D. Text Features (Supervised OOF Ridge Classifier + TF-IDF N-Grams + Truncated SVD)
text_cols = list(dict.fromkeys(text_cols))
if len(text_cols) > 0:
    train_text = pd.DataFrame(index=train_df.index)
    test_text = pd.DataFrame(index=test_df.index)

    for col in text_cols:
        tr_text_series = train_df[col].astype(str).fillna("")
        te_text_series = test_df[col].astype(str).fillna("")

        # Surface statistical length features
        train_text[f"{col}__char_len"] = tr_text_series.apply(len).astype(np.float32)
        test_text[f"{col}__char_len"] = te_text_series.apply(len).astype(np.float32)

        train_text[f"{col}__word_count"] = tr_text_series.apply(
            lambda s: len(s.split())
        ).astype(np.float32)
        test_text[f"{col}__word_count"] = te_text_series.apply(
            lambda s: len(s.split())
        ).astype(np.float32)

        train_text[f"{col}__num_digits"] = tr_text_series.apply(
            lambda s: sum(c.isdigit() for c in s)
        ).astype(np.float32)
        test_text[f"{col}__num_digits"] = te_text_series.apply(
            lambda s: sum(c.isdigit() for c in s)
        ).astype(np.float32)

        # Word & Char N-gram Vectorizers
        word_vec = TfidfVectorizer(
            ngram_range=(1, 3),
            min_df=2,
            max_features=10000,
            sublinear_tf=True,
            stop_words="english",
        )
        char_vec = TfidfVectorizer(
            ngram_range=(3, 5),
            min_df=3,
            max_features=10000,
            analyzer="char_wb",
            sublinear_tf=True,
        )

        # Leak-Free Out-Of-Fold Ridge Risk Predictions
        oof_text_ridge = np.zeros(len(train_df), dtype=np.float32)
        test_text_ridge = np.zeros(len(test_df), dtype=np.float32)

        for fold in range(n_folds):
            tr_idx = train_df["fold"] != fold
            va_idx = train_df["fold"] == fold

            tr_w = word_vec.fit_transform(tr_text_series[tr_idx])
            va_w = word_vec.transform(tr_text_series[va_idx])
            te_w = word_vec.transform(te_text_series)

            tr_c = char_vec.fit_transform(tr_text_series[tr_idx])
            va_c = char_vec.transform(tr_text_series[va_idx])
            te_c = char_vec.transform(te_text_series)

            tr_feat = hstack([tr_w, tr_c])
            va_feat = hstack([va_w, va_c])
            te_feat = hstack([te_w, te_c])

            ridge = RidgeClassifier(alpha=1.0, random_state=42)
            ridge.fit(tr_feat, train_df.loc[tr_idx, "target"])

            oof_text_ridge[va_idx] = ridge.decision_function(va_feat).astype(np.float32)
            test_text_ridge += ridge.decision_function(te_feat).astype(np.float32) / n_folds

        train_text[f"{col}__ridge_risk"] = oof_text_ridge
        test_text[f"{col}__ridge_risk"] = test_text_ridge

        # Dimensionality Reduction (SVD) on Full TF-IDF
        full_word_vec = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=5000,
            sublinear_tf=True,
            stop_words="english",
        )
        tr_full_tfidf = full_word_vec.fit_transform(tr_text_series)
        te_full_tfidf = full_word_vec.transform(te_text_series)

        n_comp = min(15, tr_full_tfidf.shape[1] - 1)
        if n_comp >= 2:
            svd = TruncatedSVD(n_components=n_comp, random_state=42)
            tr_svd = svd.fit_transform(tr_full_tfidf)
            te_svd = svd.transform(te_full_tfidf)

            for i in range(n_comp):
                train_text[f"{col}__svd_{i}"] = tr_svd[:, i].astype(np.float32)
                test_text[f"{col}__svd_{i}"] = te_svd[:, i].astype(np.float32)

    train_feat_list.append(train_text)
    test_feat_list.append(test_text)

# 5. Assemble and Clean Feature Matrices
X_train = pd.concat(train_feat_list, axis=1)
X_test = pd.concat(test_feat_list, axis=1)


def clean_col_names(df):
    clean_cols = [re.sub(r'[\[\]<>{}:,"]+', "_", str(c)) for c in df.columns]
    df.columns = clean_cols
    return df


X_train = clean_col_names(X_train)
X_test = clean_col_names(X_test)

feature_names = list(X_train.columns)
y_all = train_df["target"].values
folds_all = train_df["fold"].values
test_ids = np.arange(len(test_df))

print(
    f"Final Engineered Features: {len(feature_names)} | Train: {len(X_train)} | Test: {len(X_test)}"
)


# 6. Model Configurations
class ModelFactory:
    @staticmethod
    def get_lightgbm_params(random_state: int = 42) -> dict:
        return {
            "objective": "binary",
            "metric": "auc",
            "boosting_type": "gbdt",
            "n_estimators": 1500,
            "learning_rate": 0.03,
            "num_leaves": 31,
            "max_depth": -1,
            "min_child_samples": 20,
            "subsample": 0.8,
            "subsample_freq": 1,
            "colsample_bytree": 0.75,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "random_state": random_state,
            "n_jobs": -1,
            "verbose": -1,
        }

    @staticmethod
    def get_xgboost_params(random_state: int = 42) -> dict:
        return {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "n_estimators": 1500,
            "learning_rate": 0.03,
            "max_depth": 6,
            "min_child_weight": 3,
            "subsample": 0.8,
            "colsample_bytree": 0.75,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "random_state": random_state,
            "n_jobs": -1,
            "tree_method": "hist",
        }

    @staticmethod
    def get_catboost_params(random_state: int = 42) -> dict:
        return {
            "loss_function": "Logloss",
            "eval_metric": "AUC",
            "iterations": 1500,
            "learning_rate": 0.03,
            "depth": 6,
            "l2_leaf_reg": 3.0,
            "random_seed": random_state,
            "thread_count": -1,
            "verbose": False,
        }

    @staticmethod
    def get_extratrees_params(random_state: int = 42) -> dict:
        return {
            "n_estimators": 350,
            "max_depth": 14,
            "min_samples_split": 10,
            "min_samples_leaf": 4,
            "max_features": "sqrt",
            "random_state": random_state,
            "n_jobs": -1,
        }


# 7. Stratified 5-Fold Training Loop
n_samples = len(X_train)
n_test = len(X_test)

oof_lgb = np.zeros(n_samples, dtype=np.float32)
oof_xgb = np.zeros(n_samples, dtype=np.float32)
oof_cb = np.zeros(n_samples, dtype=np.float32)
oof_et = np.zeros(n_samples, dtype=np.float32)

test_preds_lgb = np.zeros(n_test, dtype=np.float32)
test_preds_xgb = np.zeros(n_test, dtype=np.float32)
test_preds_cb = np.zeros(n_test, dtype=np.float32)
test_preds_et = np.zeros(n_test, dtype=np.float32)

print("\nStarting Cross-Validation Model Training...")
for fold in range(n_folds):
    tr_mask = folds_all != fold
    val_mask = folds_all == fold

    X_tr, y_tr = X_train.loc[tr_mask, feature_names], y_all[tr_mask]
    X_va, y_val = X_train.loc[val_mask, feature_names], y_all[val_mask]

    # --- LightGBM ---
    lgb_params = ModelFactory.get_lightgbm_params(random_state=42 + fold)
    lgb_model = lgb.LGBMClassifier(**lgb_params)
    lgb_model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_val)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=-1),
        ],
    )
    val_pred_lgb = lgb_model.predict_proba(X_va)[:, 1]
    oof_lgb[val_mask] = val_pred_lgb
    test_preds_lgb += lgb_model.predict_proba(X_test[feature_names])[:, 1] / n_folds
    joblib.dump(lgb_model, f"./working/models/lgb_fold{fold}.joblib")

    # --- XGBoost ---
    xgb_params = ModelFactory.get_xgboost_params(random_state=42 + fold)
    xgb_model = XGBClassifier(**xgb_params, early_stopping_rounds=50)
    xgb_model.fit(X_tr, y_tr, eval_set=[(X_va, y_val)], verbose=False)
    val_pred_xgb = xgb_model.predict_proba(X_va)[:, 1]
    oof_xgb[val_mask] = val_pred_xgb
    test_preds_xgb += xgb_model.predict_proba(X_test[feature_names])[:, 1] / n_folds
    joblib.dump(xgb_model, f"./working/models/xgb_fold{fold}.joblib")

    # --- CatBoost ---
    cb_params = ModelFactory.get_catboost_params(random_state=42 + fold)
    cb_model = CatBoostClassifier(**cb_params, early_stopping_rounds=50)
    cb_model.fit(X_tr, y_tr, eval_set=[(X_va, y_val)], verbose=False)
    val_pred_cb = cb_model.predict_proba(X_va)[:, 1]
    oof_cb[val_mask] = val_pred_cb
    test_preds_cb += cb_model.predict_proba(X_test[feature_names])[:, 1] / n_folds
    joblib.dump(cb_model, f"./working/models/cb_fold{fold}.joblib")

    # --- ExtraTrees ---
    et_params = ModelFactory.get_extratrees_params(random_state=42 + fold)
    et_model = ExtraTreesClassifier(**et_params)
    X_tr_imp = np.nan_to_num(X_tr.values, nan=0.0)
    X_va_imp = np.nan_to_num(X_va.values, nan=0.0)
    X_te_imp = np.nan_to_num(X_test[feature_names].values, nan=0.0)
    et_model.fit(X_tr_imp, y_tr)
    val_pred_et = et_model.predict_proba(X_va_imp)[:, 1]
    oof_et[val_mask] = val_pred_et
    test_preds_et += et_model.predict_proba(X_te_imp)[:, 1] / n_folds
    joblib.dump(et_model, f"./working/models/et_fold{fold}.joblib")

    fold_blend = (val_pred_lgb + val_pred_xgb + val_pred_cb + val_pred_et) / 4.0
    fold_auc = roc_auc_score(y_val, fold_blend)
    print(
        f"Fold {fold+1}/{n_folds} - LGB: {roc_auc_score(y_val, val_pred_lgb):.5f} | "
        f"XGB: {roc_auc_score(y_val, val_pred_xgb):.5f} | "
        f"CB: {roc_auc_score(y_val, val_pred_cb):.5f} | "
        f"ET: {roc_auc_score(y_val, val_pred_et):.5f} | "
        f"Blend AUC: {fold_auc:.5f}"
    )

# 8. Out-of-Fold Evaluation & Scipy ROC-AUC Optimization
auc_lgb = roc_auc_score(y_all, oof_lgb)
auc_xgb = roc_auc_score(y_all, oof_xgb)
auc_cb = roc_auc_score(y_all, oof_cb)
auc_et = roc_auc_score(y_all, oof_et)

oof_models = [oof_lgb, oof_xgb, oof_cb, oof_et]
test_models = [test_preds_lgb, test_preds_xgb, test_preds_cb, test_preds_et]

# Rank transform predictions for calibration
oof_ranks = np.column_stack([rankdata(oof) / len(oof) for oof in oof_models])
test_ranks = np.column_stack([rankdata(tp) / len(tp) for tp in test_models])

def loss_func(weights):
    w = np.maximum(np.array(weights), 0)
    if w.sum() == 0:
        return 0.0
    w = w / w.sum()
    blend = oof_ranks @ w
    return -roc_auc_score(y_all, blend)

init_weights = np.ones(len(oof_models)) / len(oof_models)
res = minimize(
    loss_func,
    init_weights,
    method="Nelder-Mead",
    options={"maxiter": 1500, "xatol": 1e-4, "fatol": 1e-4},
)
opt_weights = np.maximum(res.x, 0)
opt_weights /= opt_weights.sum()

oof_ensemble = oof_ranks @ opt_weights
final_score = roc_auc_score(y_all, oof_ensemble)

print("\n--- Out-of-Fold Evaluation ---")
print(f"OOF LightGBM AUC : {auc_lgb:.5f}")
print(f"OOF XGBoost AUC  : {auc_xgb:.5f}")
print(f"OOF CatBoost AUC : {auc_cb:.5f}")
print(f"OOF ExtraTrees AUC: {auc_et:.5f}")
print(f"Optimal Weights  : LGB={opt_weights[0]:.4f}, XGB={opt_weights[1]:.4f}, CB={opt_weights[2]:.4f}, ET={opt_weights[3]:.4f}")
print(f"OOF Ensemble AUC : {final_score:.5f}")

# 9. Generate Submission File
final_test_preds = test_ranks @ opt_weights
submission_df = pd.DataFrame(
    {
        "id": test_ids.astype(int),
        "Task Assigned": final_test_preds.astype(np.float64),
    }
)

submission_path = "./submission/submission.csv"
submission_df.to_csv(submission_path, index=False)
print(
    f"\nSubmission successfully saved to {submission_path} with shape {submission_df.shape}"
)

# Required Final Output
print(f"Final Validation Score: {final_score}")
