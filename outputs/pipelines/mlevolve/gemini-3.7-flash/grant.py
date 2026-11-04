import os
import re
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import root_mean_squared_error
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

warnings.filterwarnings("ignore")

os.makedirs("./submission", exist_ok=True)
os.makedirs("./working", exist_ok=True)

# =========================================================================
# 1. Load Data & Identify Column Modalities
# =========================================================================
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "Federal Funds"
id_candidates = ["id", "ID", "Id", "GrantId", "grant_id", "index"]
id_col = None
for col in id_candidates:
    if col in test_df.columns:
        id_col = col
        break

y_train = train_df[target_col].copy().astype(float).values
train_features = train_df.drop(columns=[target_col], errors="ignore")
test_features = test_df.copy()

all_cols = [c for c in train_features.columns if c != id_col]

date_cols = []
text_cols = []
cat_cols = []
num_cols = []

for c in all_cols:
    sample_series = train_features[c].dropna()
    if len(sample_series) == 0:
        continue

    if pd.api.types.is_datetime64_any_dtype(sample_series):
        date_cols.append(c)
        continue

    col_name_lower = c.lower()
    if any(
        k in col_name_lower
        for k in ["date", "time", "start", "end", "award_dt", "grant_period"]
    ):
        try:
            pd.to_datetime(sample_series.iloc[:20])
            date_cols.append(c)
            continue
        except Exception:
            pass

    if pd.api.types.is_numeric_dtype(sample_series):
        num_cols.append(c)
        continue

    str_lengths = sample_series.astype(str).str.len()
    avg_len = str_lengths.mean()
    if avg_len > 40 or any(
        k in col_name_lower
        for k in ["title", "abstract", "description", "narrative", "summary", "scope"]
    ):
        text_cols.append(c)
    else:
        cat_cols.append(c)


# =========================================================================
# 2. Base Feature Extraction
# =========================================================================
def extract_base_features(df):
    feats = pd.DataFrame(index=df.index)

    for c in num_cols:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce")
            feats[c] = s
            if (s > 0).all():
                feats[f"{c}_log1p"] = np.log1p(s)

    parsed_dates = {}
    for c in date_cols:
        if c in df.columns:
            dt_s = pd.to_datetime(df[c], errors="coerce")
            parsed_dates[c] = dt_s
            feats[f"{c}_year"] = dt_s.dt.year
            feats[f"{c}_month"] = dt_s.dt.month
            feats[f"{c}_day"] = dt_s.dt.day
            feats[f"{c}_dayofweek"] = dt_s.dt.dayofweek
            feats[f"{c}_quarter"] = dt_s.dt.quarter
            feats[f"{c}_dayofyear"] = dt_s.dt.dayofyear

    date_keys = list(parsed_dates.keys())
    for i in range(len(date_keys)):
        for j in range(i + 1, len(date_keys)):
            d1_name, d2_name = date_keys[i], date_keys[j]
            diff_days = (
                parsed_dates[d2_name] - parsed_dates[d1_name]
            ).dt.total_seconds() / (24 * 3600)
            feats[f"diff_days_{d2_name}_minus_{d1_name}"] = diff_days
            feats[f"diff_days_{d2_name}_minus_{d1_name}_abs"] = np.abs(diff_days)

    for c in text_cols:
        if c in df.columns:
            text_s = df[c].fillna("").astype(str)
            feats[f"{c}_char_len"] = text_s.str.len()
            feats[f"{c}_word_count"] = text_s.apply(lambda x: len(x.split()))
            feats[f"{c}_upper_count"] = text_s.apply(
                lambda x: sum(1 for ch in x if ch.isupper())
            )
            feats[f"{c}_digit_count"] = text_s.apply(
                lambda x: sum(1 for ch in x if ch.isdigit())
            )
            feats[f"{c}_upper_ratio"] = feats[f"{c}_upper_count"] / (
                feats[f"{c}_char_len"] + 1
            )
            feats[f"{c}_avg_word_len"] = feats[f"{c}_char_len"] / (
                feats[f"{c}_word_count"] + 1
            )

    return feats


X_train_base = extract_base_features(train_features)
X_test_base = extract_base_features(test_features)


# =========================================================================
# 3. Model Architecture Builders
# =========================================================================
def get_lightgbm_model(random_state: int = 42):
    return lgb.LGBMRegressor(
        objective="regression",
        metric="rmse",
        boosting_type="gbdt",
        n_estimators=3000,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.75,
        reg_alpha=0.1,
        reg_lambda=1.0,
        min_child_samples=20,
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
    )


def get_catboost_model(random_state: int = 42):
    return CatBoostRegressor(
        loss_function="RMSE",
        eval_metric="RMSE",
        iterations=3000,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=3.0,
        random_seed=random_state,
        verbose=False,
        thread_count=-1,
    )


def get_xgboost_model(random_state: int = 42):
    return xgb.XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        n_estimators=3000,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.75,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=random_state,
        n_jobs=-1,
        tree_method="hist",
        early_stopping_rounds=50,
    )


# =========================================================================
# 4. K-Fold Cross Validation Setup & Fold Processing
# =========================================================================
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

n_train = len(train_features)
n_test = len(test_features)

oof_lgb = np.zeros(n_train)
oof_cb = np.zeros(n_train)
oof_xgb = np.zeros(n_train)

test_preds_lgb = np.zeros(n_test)
test_preds_cb = np.zeros(n_test)
test_preds_xgb = np.zeros(n_test)

for fold, (train_idx, val_idx) in enumerate(kf.split(train_features)):
    tr_sub = train_features.iloc[train_idx]
    val_sub = train_features.iloc[val_idx]
    te_sub = test_features.copy()

    fold_tr_feat = X_train_base.iloc[train_idx].copy()
    fold_val_feat = X_train_base.iloc[val_idx].copy()
    fold_te_feat = X_test_base.copy()

    # Categorical encodings
    for c in cat_cols:
        freq_map = (
            tr_sub[c].fillna("__MISSING__").value_counts(normalize=True).to_dict()
        )
        fold_tr_feat[f"{c}_freq"] = (
            tr_sub[c].fillna("__MISSING__").map(freq_map).fillna(0)
        )
        fold_val_feat[f"{c}_freq"] = (
            val_sub[c].fillna("__MISSING__").map(freq_map).fillna(0)
        )
        fold_te_feat[f"{c}_freq"] = (
            te_sub[c].fillna("__MISSING__").map(freq_map).fillna(0)
        )

        cats = {val: idx for idx, val in enumerate(tr_sub[c].dropna().unique())}
        fold_tr_feat[f"{c}_catcode"] = tr_sub[c].map(cats).fillna(-1).astype(int)
        fold_val_feat[f"{c}_catcode"] = val_sub[c].map(cats).fillna(-1).astype(int)
        fold_te_feat[f"{c}_catcode"] = te_sub[c].map(cats).fillna(-1).astype(int)

    # TF-IDF + SVD for text columns
    for c in text_cols:
        tr_text = tr_sub[c].fillna("").astype(str)
        val_text = val_sub[c].fillna("").astype(str)
        te_text = te_sub[c].fillna("").astype(str)

        tfidf = TfidfVectorizer(
            max_features=5000, ngram_range=(1, 2), min_df=2, stop_words="english"
        )
        tr_tfidf = tfidf.fit_transform(tr_text)
        val_tfidf = tfidf.transform(val_text)
        te_tfidf = tfidf.transform(te_text)

        n_components = min(16, tr_tfidf.shape[1] - 1, tr_tfidf.shape[0] - 1)
        if n_components >= 2:
            svd = TruncatedSVD(n_components=n_components, random_state=42)
            tr_svd = svd.fit_transform(tr_tfidf)
            val_svd = svd.transform(val_tfidf)
            te_svd = svd.transform(te_tfidf)

            for comp in range(n_components):
                fold_tr_feat[f"{c}_svd_{comp}"] = tr_svd[:, comp]
                fold_val_feat[f"{c}_svd_{comp}"] = val_svd[:, comp]
                fold_te_feat[f"{c}_svd_{comp}"] = te_svd[:, comp]

    # Group aggregations
    key_cat_cols = [
        c for c in cat_cols if tr_sub[c].nunique() > 1 and tr_sub[c].nunique() < 500
    ][:3]
    key_num_cols = [c for c in num_cols if c in fold_tr_feat.columns][:3]

    for cat_c in key_cat_cols:
        for num_c in key_num_cols:
            group_stat = (
                fold_tr_feat.groupby(tr_sub[cat_c].fillna("__MISSING__"))[num_c]
                .agg(["mean", "std", "median"])
                .reset_index()
            )
            group_stat.columns = [
                cat_c,
                f"{cat_c}_{num_c}_mean",
                f"{cat_c}_{num_c}_std",
                f"{cat_c}_{num_c}_median",
            ]

            m_tr = pd.DataFrame(
                {cat_c: tr_sub[cat_c].fillna("__MISSING__")}, index=tr_sub.index
            ).merge(group_stat, on=cat_c, how="left")
            m_val = pd.DataFrame(
                {cat_c: val_sub[cat_c].fillna("__MISSING__")}, index=val_sub.index
            ).merge(group_stat, on=cat_c, how="left")
            m_te = pd.DataFrame(
                {cat_c: te_sub[cat_c].fillna("__MISSING__")}, index=te_sub.index
            ).merge(group_stat, on=cat_c, how="left")

            for stat_col in [
                f"{cat_c}_{num_c}_mean",
                f"{cat_c}_{num_c}_std",
                f"{cat_c}_{num_c}_median",
            ]:
                fold_tr_feat[stat_col] = m_tr[stat_col].values
                fold_val_feat[stat_col] = m_val[stat_col].values
                fold_te_feat[stat_col] = m_te[stat_col].values

    # Clean missing / infinity and sanitize column names for boosting libraries
    fold_tr_feat = fold_tr_feat.replace([np.inf, -np.inf], np.nan).fillna(0)
    fold_val_feat = fold_val_feat.replace([np.inf, -np.inf], np.nan).fillna(0)
    fold_te_feat = fold_te_feat.replace([np.inf, -np.inf], np.nan).fillna(0)

    clean_cols = [
        re.sub(r'[\[\]<:,"\{\}]', "_", str(col)) for col in fold_tr_feat.columns
    ]
    fold_tr_feat.columns = clean_cols
    fold_val_feat.columns = clean_cols
    fold_te_feat.columns = clean_cols

    y_tr = y_train[train_idx]
    y_val = y_train[val_idx]

    # Model 1: LightGBM
    lgb_model = get_lightgbm_model(random_state=42 + fold)
    lgb_model.fit(
        fold_tr_feat,
        y_tr,
        eval_set=[(fold_val_feat, y_val)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    val_pred_lgb = lgb_model.predict(fold_val_feat)
    oof_lgb[val_idx] = val_pred_lgb
    test_preds_lgb += lgb_model.predict(fold_te_feat) / n_splits

    # Model 2: CatBoost
    cb_model = get_catboost_model(random_state=42 + fold)
    cb_model.fit(
        fold_tr_feat,
        y_tr,
        eval_set=(fold_val_feat, y_val),
        early_stopping_rounds=50,
        verbose=False,
    )
    val_pred_cb = cb_model.predict(fold_val_feat)
    oof_cb[val_idx] = val_pred_cb
    test_preds_cb += cb_model.predict(fold_te_feat) / n_splits

    # Model 3: XGBoost
    xgb_model = get_xgboost_model(random_state=42 + fold)
    xgb_model.fit(fold_tr_feat, y_tr, eval_set=[(fold_val_feat, y_val)], verbose=False)
    val_pred_xgb = xgb_model.predict(fold_val_feat)
    oof_xgb[val_idx] = val_pred_xgb
    test_preds_xgb += xgb_model.predict(fold_te_feat) / n_splits

    f_lgb = root_mean_squared_error(y_val, val_pred_lgb)
    f_cb = root_mean_squared_error(y_val, val_pred_cb)
    f_xgb = root_mean_squared_error(y_val, val_pred_xgb)
    print(
        f"Fold {fold+1}/{n_splits} | LGB: {f_lgb:.4f} | CB: {f_cb:.4f} | XGB: {f_xgb:.4f}"
    )


# =========================================================================
# 5. Out-of-Fold Blending & Submission Generation
# =========================================================================
oof_matrix = np.column_stack([oof_lgb, oof_cb, oof_xgb])
test_matrix = np.column_stack([test_preds_lgb, test_preds_cb, test_preds_xgb])


def blend_objective(weights):
    weights = np.array(weights)
    weights = weights / np.sum(weights)
    blended = np.dot(oof_matrix, weights)
    return root_mean_squared_error(y_train, blended)


initial_weights = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
bounds = [(0.0, 1.0)] * 3
constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

opt_res = minimize(
    blend_objective,
    initial_weights,
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
)
best_weights = (
    opt_res.x / np.sum(opt_res.x) if opt_res.success else np.array(initial_weights)
)

oof_ensemble = np.dot(oof_matrix, best_weights)
test_ensemble = np.dot(test_matrix, best_weights)

if (y_train >= 0).all():
    oof_ensemble = np.clip(oof_ensemble, a_min=0.0, a_max=None)
    test_ensemble = np.clip(test_ensemble, a_min=0.0, a_max=None)

final_score = root_mean_squared_error(y_train, oof_ensemble)

submission_df = pd.DataFrame(
    {"id": np.arange(len(test_ensemble), dtype=int), "Federal Funds": test_ensemble}
)
submission_df.to_csv("./submission/submission.csv", index=False)

print(f"Final Validation Score: {final_score}")
