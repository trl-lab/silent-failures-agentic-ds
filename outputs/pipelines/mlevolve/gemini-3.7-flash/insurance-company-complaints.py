import gc
import os
import re
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostRegressor, CatBoostClassifier
from scipy.optimize import nnls, minimize
import lightgbm as lgb
import xgboost as xgb

warnings.filterwarnings("ignore")

# Set random seeds for reproducibility
np.random.seed(42)

os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# ---------------------------------------------------------
# 1. Load Data
# ---------------------------------------------------------
print("Loading datasets from ./input...")
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "Recovery"
if target_col not in train_df.columns:
    for c in train_df.columns:
        if c.lower() == "recovery":
            target_col = c
            break

# Identify ID column if exists
id_col = None
for col in ["id", "ID", "Id", "complaint_id", "Complaint ID"]:
    if col in test_df.columns:
        id_col = col
        break

print(
    f"Train shape: {train_df.shape}, Test shape: {test_df.shape}, Target: {target_col}"
)

# ---------------------------------------------------------
# 2. Stratified 5-Fold Splitting based on zero-inflation & quantiles
# ---------------------------------------------------------
y_raw = train_df[target_col].values.astype(np.float32)
strat_bins = np.zeros(len(y_raw), dtype=int)
positive_mask = y_raw > 0
if positive_mask.sum() > 50:
    q_bins = pd.qcut(y_raw[positive_mask], q=5, labels=False, duplicates="drop")
    strat_bins[positive_mask] = q_bins + 1

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_assignments = np.zeros(len(train_df), dtype=int)
for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, strat_bins)):
    fold_assignments[val_idx] = fold

train_df["fold"] = fold_assignments

# ---------------------------------------------------------
# 3. Column Type Identification
# ---------------------------------------------------------
features_to_drop = [target_col, "fold"]
if id_col and id_col in train_df.columns:
    features_to_drop.append(id_col)

candidate_cols = [
    c for c in train_df.columns if c not in features_to_drop and c in test_df.columns
]

date_cols = []
text_cols = []
cat_cols = []
num_cols = []

for c in candidate_cols:
    if pd.api.types.is_datetime64_any_dtype(train_df[c]):
        date_cols.append(c)
        continue

    sample_vals = train_df[c].dropna().astype(str)
    if len(sample_vals) == 0:
        continue

    date_matches = sample_vals.head(20).apply(
        lambda x: bool(re.match(r"^\d{4}[-/]\d{2}[-/]\d{2}|\d{2}[-/]\d{2}[-/]\d{4}", x))
    )
    if date_matches.mean() > 0.8:
        date_cols.append(c)
        continue

    avg_len = sample_vals.apply(len).mean()
    unique_ratio = train_df[c].nunique() / max(1, len(train_df[c].dropna()))

    if (avg_len > 40 and unique_ratio > 0.2) or any(
        keyword in c.lower()
        for keyword in [
            "narrative",
            "description",
            "text",
            "complaint",
            "summary",
            "detail",
        ]
    ):
        text_cols.append(c)
    elif pd.api.types.is_numeric_dtype(train_df[c]) and train_df[c].nunique() > 20:
        num_cols.append(c)
    else:
        cat_cols.append(c)

print(
    f"Identified feature types: {len(date_cols)} Date, {len(text_cols)} Text, {len(cat_cols)} Categorical, {len(num_cols)} Numerical"
)

# ---------------------------------------------------------
# 4. Feature Engineering
# ---------------------------------------------------------
train_fe = pd.DataFrame(index=train_df.index)
test_fe = pd.DataFrame(index=test_df.index)

# A. Process Date Features with Cyclic Encodings & Durations
for c in date_cols:
    train_dt = pd.to_datetime(train_df[c], errors="coerce")
    test_dt = pd.to_datetime(test_df[c], errors="coerce")

    prefix = f"dt_{c}"
    for df_dt, out_df in [(train_dt, train_fe), (test_dt, test_fe)]:
        out_df[f"{prefix}_year"] = df_dt.dt.year.fillna(-1).astype(int)
        out_df[f"{prefix}_month"] = df_dt.dt.month.fillna(-1).astype(int)
        out_df[f"{prefix}_day"] = df_dt.dt.day.fillna(-1).astype(int)
        out_df[f"{prefix}_dayofweek"] = df_dt.dt.dayofweek.fillna(-1).astype(int)
        out_df[f"{prefix}_quarter"] = df_dt.dt.quarter.fillna(-1).astype(int)
        out_df[f"{prefix}_is_month_end"] = df_dt.dt.is_month_end.fillna(False).astype(int)

        # Cyclic encodings
        m_rad = 2.0 * np.pi * (df_dt.dt.month.fillna(1).astype(float) - 1.0) / 12.0
        d_rad = 2.0 * np.pi * (df_dt.dt.day.fillna(1).astype(float) - 1.0) / 31.0
        dow_rad = 2.0 * np.pi * df_dt.dt.dayofweek.fillna(0).astype(float) / 7.0

        out_df[f"{prefix}_month_sin"] = np.sin(m_rad).astype(float)
        out_df[f"{prefix}_month_cos"] = np.cos(m_rad).astype(float)
        out_df[f"{prefix}_day_sin"] = np.sin(d_rad).astype(float)
        out_df[f"{prefix}_day_cos"] = np.cos(d_rad).astype(float)
        out_df[f"{prefix}_dow_sin"] = np.sin(dow_rad).astype(float)
        out_df[f"{prefix}_dow_cos"] = np.cos(dow_rad).astype(float)

if len(date_cols) >= 2:
    for i in range(len(date_cols)):
        for j in range(i + 1, len(date_cols)):
            c1, c2 = date_cols[i], date_cols[j]
            dt1_tr = pd.to_datetime(train_df[c1], errors="coerce")
            dt2_tr = pd.to_datetime(train_df[c2], errors="coerce")
            dt1_te = pd.to_datetime(test_df[c1], errors="coerce")
            dt2_te = pd.to_datetime(test_df[c2], errors="coerce")

            diff_sec_tr = (dt2_tr - dt1_tr).dt.total_seconds()
            diff_sec_te = (dt2_te - dt1_te).dt.total_seconds()

            diff_name = f"diff_days_{c1}_vs_{c2}"
            train_fe[diff_name] = (diff_sec_tr / 86400.0).fillna(-999.0).astype(float)
            test_fe[diff_name] = (diff_sec_te / 86400.0).fillna(-999.0).astype(float)

            diff_weeks_name = f"diff_weeks_{c1}_vs_{c2}"
            train_fe[diff_weeks_name] = (diff_sec_tr / (86400.0 * 7.0)).fillna(-999.0).astype(float)
            test_fe[diff_weeks_name] = (diff_sec_te / (86400.0 * 7.0)).fillna(-999.0).astype(float)

            train_fe[f"same_day_{c1}_vs_{c2}"] = (train_fe[diff_name] == 0.0).astype(float)
            test_fe[f"same_day_{c1}_vs_{c2}"] = (test_fe[diff_name] == 0.0).astype(float)

            # Business day settlement duration
            def get_busday_diff(s1, s2):
                d1 = pd.to_datetime(s1, errors="coerce").dt.date
                d2 = pd.to_datetime(s2, errors="coerce").dt.date
                res = np.full(len(s1), -999.0, dtype=np.float32)
                valid = d1.notnull() & d2.notnull()
                if valid.sum() > 0:
                    v1 = d1[valid].astype(str).values.astype("datetime64[D]")
                    v2 = d2[valid].astype(str).values.astype("datetime64[D]")
                    bcount = np.zeros(len(v1), dtype=np.float32)
                    fwd = v1 <= v2
                    if fwd.sum() > 0:
                        bcount[fwd] = np.busday_count(v1[fwd], v2[fwd])
                    if (~fwd).sum() > 0:
                        bcount[~fwd] = -np.busday_count(v2[~fwd], v1[~fwd])
                    res[valid] = bcount
                return res

            train_fe[f"busdays_{c1}_vs_{c2}"] = get_busday_diff(train_df[c1], train_df[c2])
            test_fe[f"busdays_{c1}_vs_{c2}"] = get_busday_diff(test_df[c1], test_df[c2])

# Categorical temporal volume aggregations
if len(date_cols) > 0 and len(cat_cols) > 0:
    primary_date = date_cols[0]
    tr_dt = pd.to_datetime(train_df[primary_date], errors="coerce")
    te_dt = pd.to_datetime(test_df[primary_date], errors="coerce")
    tr_ym = tr_dt.dt.year.astype(str) + "_" + tr_dt.dt.month.astype(str)
    te_ym = te_dt.dt.year.astype(str) + "_" + te_dt.dt.month.astype(str)

    for c in cat_cols[:5]:
        comb_cat_ym = pd.concat([
            pd.DataFrame({"cat": train_df[c].astype(str), "ym": tr_ym}),
            pd.DataFrame({"cat": test_df[c].astype(str), "ym": te_ym}),
        ], axis=0)
        ym_counts = comb_cat_ym.groupby(["cat", "ym"]).size().to_dict()

        train_fe[f"vol_ym_{c}"] = [ym_counts.get((cat_v, ym_v), 0) for cat_v, ym_v in zip(train_df[c].astype(str), tr_ym)]
        test_fe[f"vol_ym_{c}"] = [ym_counts.get((cat_v, ym_v), 0) for cat_v, ym_v in zip(test_df[c].astype(str), te_ym)]

# B. Process Categorical Features & Frequency / Count Encoding
for c in cat_cols:
    tr_series = train_df[c].astype(str).fillna("MISSING")
    te_series = test_df[c].astype(str).fillna("MISSING")
    combined_series = pd.concat([tr_series, te_series], axis=0)

    freq_map = combined_series.value_counts(normalize=True).to_dict()
    cnt_map = combined_series.value_counts().to_dict()

    train_fe[f"freq_{c}"] = tr_series.map(freq_map).fillna(0.0).astype(float)
    test_fe[f"freq_{c}"] = te_series.map(freq_map).fillna(0.0).astype(float)
    train_fe[f"cnt_{c}"] = tr_series.map(cnt_map).fillna(0.0).astype(float)
    test_fe[f"cnt_{c}"] = te_series.map(cnt_map).fillna(0.0).astype(float)

    unique_vals = list(combined_series.unique())
    val_to_idx = {val: idx + 1 for idx, val in enumerate(unique_vals)}
    train_fe[f"cat_{c}"] = tr_series.map(val_to_idx).fillna(0).astype(int)
    test_fe[f"cat_{c}"] = te_series.map(val_to_idx).fillna(0).astype(int)

# C. Out-Of-Fold Bayesian-Smoothed Target Statistics
m_smooth = 10.0
y_train_arr = train_df[target_col].values.astype(float)
global_mean = float(np.mean(y_train_arr))
global_std = float(np.std(y_train_arr))
global_prob = float(np.mean(y_train_arr > 0))

for c in cat_cols:
    oof_te_mean = np.zeros(len(train_df), dtype=float)
    oof_te_std = np.zeros(len(train_df), dtype=float)
    oof_te_prob = np.zeros(len(train_df), dtype=float)

    for fold in range(5):
        val_mask = fold_assignments == fold
        trn_mask = ~val_mask

        sub_df = train_df.loc[trn_mask, [c, target_col]]
        grp = sub_df.groupby(c)[target_col]
        counts = grp.count()
        means = grp.mean()
        stds = grp.std().fillna(global_std)
        probs = (sub_df[target_col] > 0).groupby(sub_df[c]).mean()

        sm_mean = ((counts * means + m_smooth * global_mean) / (counts + m_smooth)).to_dict()
        sm_std = ((counts * stds + m_smooth * global_std) / (counts + m_smooth)).to_dict()
        sm_prob = ((counts * probs + m_smooth * global_prob) / (counts + m_smooth)).to_dict()

        oof_te_mean[val_mask] = train_df.loc[val_mask, c].map(sm_mean).fillna(global_mean)
        oof_te_std[val_mask] = train_df.loc[val_mask, c].map(sm_std).fillna(global_std)
        oof_te_prob[val_mask] = train_df.loc[val_mask, c].map(sm_prob).fillna(global_prob)

    # Full train mapping for test set
    full_grp = train_df.groupby(c)[target_col]
    full_counts = full_grp.count()
    full_means = full_grp.mean()
    full_stds = full_grp.std().fillna(global_std)
    full_probs = (train_df[target_col] > 0).groupby(train_df[c]).mean()

    full_sm_mean = ((full_counts * full_means + m_smooth * global_mean) / (full_counts + m_smooth)).to_dict()
    full_sm_std = ((full_counts * full_stds + m_smooth * global_std) / (full_counts + m_smooth)).to_dict()
    full_sm_prob = ((full_counts * full_probs + m_smooth * global_prob) / (full_counts + m_smooth)).to_dict()

    train_fe[f"oof_target_mean_{c}"] = oof_te_mean
    train_fe[f"oof_target_std_{c}"] = oof_te_std
    train_fe[f"oof_target_prob_{c}"] = oof_te_prob

    test_fe[f"oof_target_mean_{c}"] = test_df[c].map(full_sm_mean).fillna(global_mean).astype(float)
    test_fe[f"oof_target_std_{c}"] = test_df[c].map(full_sm_std).fillna(global_std).astype(float)
    test_fe[f"oof_target_prob_{c}"] = test_df[c].map(full_sm_prob).fillna(global_prob).astype(float)

# D. Categorical Pairwise Interaction & Cardinality Aggregations
if len(cat_cols) >= 2:
    top_cats = sorted(cat_cols, key=lambda x: train_df[x].nunique())[:5]
    for i in range(len(top_cats)):
        for j in range(i + 1, len(top_cats)):
            c1, c2 = top_cats[i], top_cats[j]
            inter_tr = train_df[c1].astype(str) + "__" + train_df[c2].astype(str)
            inter_te = test_df[c1].astype(str) + "__" + test_df[c2].astype(str)

            inter_comb = pd.concat([inter_tr, inter_te], axis=0)
            inter_freq = inter_comb.value_counts(normalize=True).to_dict()
            inter_cnt = inter_comb.value_counts().to_dict()

            train_fe[f"freq_inter_{c1}_{c2}"] = inter_tr.map(inter_freq).fillna(0.0).astype(float)
            test_fe[f"freq_inter_{c1}_{c2}"] = inter_te.map(inter_freq).fillna(0.0).astype(float)
            train_fe[f"cnt_inter_{c1}_{c2}"] = inter_tr.map(inter_cnt).fillna(0.0).astype(float)
            test_fe[f"cnt_inter_{c1}_{c2}"] = inter_te.map(inter_cnt).fillna(0.0).astype(float)

    # Group-level cardinality aggregations
    for c1 in top_cats[:3]:
        for c2 in top_cats:
            if c1 != c2:
                comb_c12 = pd.concat([train_df[[c1, c2]], test_df[[c1, c2]]], axis=0)
                nuniq_map = comb_c12.groupby(c1)[c2].nunique().to_dict()
                train_fe[f"nuniq_{c2}_by_{c1}"] = train_df[c1].map(nuniq_map).fillna(0.0).astype(float)
                test_fe[f"nuniq_{c2}_by_{c1}"] = test_df[c1].map(nuniq_map).fillna(0.0).astype(float)

# E. Process Numerical Features
for c in num_cols:
    med = train_df[c].median()
    train_fe[f"num_{c}"] = train_df[c].fillna(med).astype(float)
    test_fe[f"num_{c}"] = test_df[c].fillna(med).astype(float)

    if train_df[c].isnull().sum() > 0 or test_df[c].isnull().sum() > 0:
        train_fe[f"isna_{c}"] = train_df[c].isnull().astype(int)
        test_fe[f"isna_{c}"] = test_df[c].isnull().astype(int)

    if (train_df[c].dropna() >= 0).all():
        train_fe[f"log1p_{c}"] = np.log1p(np.clip(train_fe[f"num_{c}"], 0, None))
        test_fe[f"log1p_{c}"] = np.log1p(np.clip(test_fe[f"num_{c}"], 0, None))

# Group-by Numerical Aggregations
if len(num_cols) > 0 and len(cat_cols) > 0:
    for cat_c in cat_cols[:3]:
        for num_c in num_cols[:3]:
            grp_stats = (
                train_df.groupby(cat_c)[num_c]
                .agg(["mean", "std", "median"])
                .reset_index()
            )
            grp_stats.columns = [
                cat_c,
                f"{cat_c}_{num_c}_mean",
                f"{cat_c}_{num_c}_std",
                f"{cat_c}_{num_c}_median",
            ]

            tr_merged = pd.merge(train_df[[cat_c]], grp_stats, on=cat_c, how="left")
            te_merged = pd.merge(test_df[[cat_c]], grp_stats, on=cat_c, how="left")

            for stat in ["mean", "std", "median"]:
                col_stat_name = f"{cat_c}_{num_c}_{stat}"
                fill_v = grp_stats[col_stat_name].median()
                train_fe[col_stat_name] = (
                    tr_merged[col_stat_name].fillna(fill_v).astype(float)
                )
                test_fe[col_stat_name] = (
                    te_merged[col_stat_name].fillna(fill_v).astype(float)
                )

# F. Process Text Narrative Features
for text_c in text_cols:
    tr_text = train_df[text_c].fillna("").astype(str)
    te_text = test_df[text_c].fillna("").astype(str)

    train_fe[f"txt_len_{text_c}"] = tr_text.apply(len).astype(float)
    test_fe[f"txt_len_{text_c}"] = te_text.apply(len).astype(float)

    train_fe[f"txt_words_{text_c}"] = tr_text.apply(lambda x: len(x.split())).astype(
        float
    )
    test_fe[f"txt_words_{text_c}"] = te_text.apply(lambda x: len(x.split())).astype(
        float
    )

    train_fe[f"txt_digits_{text_c}"] = tr_text.apply(
        lambda x: sum(ch.isdigit() for ch in x)
    ).astype(float)
    test_fe[f"txt_digits_{text_c}"] = te_text.apply(
        lambda x: sum(ch.isdigit() for ch in x)
    ).astype(float)

    train_fe[f"txt_upper_{text_c}"] = tr_text.apply(
        lambda x: sum(ch.isupper() for ch in x)
    ).astype(float)
    test_fe[f"txt_upper_{text_c}"] = te_text.apply(
        lambda x: sum(ch.isupper() for ch in x)
    ).astype(float)

    # TF-IDF + SVD
    tfidf = TfidfVectorizer(
        max_features=2500, stop_words="english", ngram_range=(1, 2), sublinear_tf=True
    )
    tfidf.fit(tr_text)

    n_components = min(16, max(2, int(len(tr_text) ** 0.5)))
    svd = TruncatedSVD(n_components=n_components, random_state=42)

    tr_tfidf = tfidf.transform(tr_text)
    te_tfidf = tfidf.transform(te_text)

    tr_svd = svd.fit_transform(tr_tfidf)
    te_svd = svd.transform(te_tfidf)

    for k in range(n_components):
        train_fe[f"svd_{text_c}_{k}"] = tr_svd[:, k].astype(float)
        test_fe[f"svd_{text_c}_{k}"] = te_svd[:, k].astype(float)

# Clean column names and sanitize values
clean_col_names = [re.sub(r"[^\w_]+", "_", col) for col in train_fe.columns]
train_fe.columns = clean_col_names
test_fe.columns = clean_col_names

train_fe = train_fe.fillna(0.0).replace([np.inf, -np.inf], 0.0)
test_fe = test_fe.fillna(0.0).replace([np.inf, -np.inf], 0.0)

feature_cols = list(train_fe.columns)
print(f"Engineered {len(feature_cols)} total features.")

X = train_fe.values.astype(np.float32)
y = y_raw
X_test = test_fe.values.astype(np.float32)
folds = fold_assignments
n_folds = 5
n_samples = len(X)
n_test = len(X_test)


# ---------------------------------------------------------
# 5. Model Architecture & Hyperparameters Setup
# ---------------------------------------------------------
oof_lgb = np.zeros(n_samples, dtype=np.float32)
oof_cb = np.zeros(n_samples, dtype=np.float32)
oof_xgb = np.zeros(n_samples, dtype=np.float32)

test_lgb = np.zeros(n_test, dtype=np.float32)
test_cb = np.zeros(n_test, dtype=np.float32)
test_xgb = np.zeros(n_test, dtype=np.float32)

lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.03,
    "num_leaves": 31,
    "max_depth": -1,
    "feature_fraction": 0.75,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "min_child_samples": 20,
    "reg_alpha": 0.5,
    "reg_lambda": 1.5,
    "n_estimators": 2000,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}

cb_params = {
    "loss_function": "RMSE",
    "eval_metric": "RMSE",
    "learning_rate": 0.03,
    "depth": 6,
    "l2_leaf_reg": 3.0,
    "iterations": 2000,
    "random_seed": 42,
    "verbose": False,
    "thread_count": -1,
}

xgb_params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "learning_rate": 0.03,
    "max_depth": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.75,
    "min_child_weight": 5,
    "reg_alpha": 0.5,
    "reg_lambda": 2.0,
    "n_estimators": 2000,
    "random_state": 42,
    "n_jobs": -1,
    "tree_method": "hist",
}

# ---------------------------------------------------------
# 6. Model Training & Cross-Validation Pipeline
# ---------------------------------------------------------
print("Executing tree ensemble training across 5 folds...")

for fold in range(n_folds):
    print(f"\n--- Starting Training for Fold {fold + 1}/{n_folds} ---")
    val_mask = folds == fold
    trn_mask = ~val_mask

    X_train_f, y_train_f = X[trn_mask], y[trn_mask]
    X_val_f, y_val_f = X[val_mask], y[val_mask]

    # --- A. LightGBM ---
    lgb_train_data = lgb.Dataset(X_train_f, label=y_train_f)
    lgb_val_data = lgb.Dataset(X_val_f, label=y_val_f, reference=lgb_train_data)

    lgb_callbacks = [
        lgb.early_stopping(stopping_rounds=50, verbose=False),
        lgb.log_evaluation(period=0),
    ]

    lgb_model = lgb.train(
        lgb_params,
        lgb_train_data,
        valid_sets=[lgb_train_data, lgb_val_data],
        callbacks=lgb_callbacks,
    )

    oof_lgb[val_mask] = lgb_model.predict(
        X_val_f, num_iteration=lgb_model.best_iteration
    )
    test_lgb += (
        lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration) / n_folds
    )
    fold_lgb_rmse = np.sqrt(mean_squared_error(y_val_f, oof_lgb[val_mask]))
    print(f"Fold {fold + 1} LightGBM RMSE: {fold_lgb_rmse:.4f}")

    # --- B. CatBoost ---
    cb_model = CatBoostRegressor(
        **cb_params,
        early_stopping_rounds=50,
    )
    cb_model.fit(
        X_train_f,
        y_train_f,
        eval_set=(X_val_f, y_val_f),
        verbose=False,
    )

    oof_cb[val_mask] = cb_model.predict(X_val_f)
    test_cb += cb_model.predict(X_test) / n_folds
    fold_cb_rmse = np.sqrt(mean_squared_error(y_val_f, oof_cb[val_mask]))
    print(f"Fold {fold + 1} CatBoost RMSE: {fold_cb_rmse:.4f}")

    # --- C. XGBoost ---
    xgb_params_fold = xgb_params.copy()
    xgb_params_fold["early_stopping_rounds"] = 50
    xgb_model = xgb.XGBRegressor(
        **xgb_params_fold,
    )
    xgb_model.fit(
        X_train_f,
        y_train_f,
        eval_set=[(X_val_f, y_val_f)],
        verbose=False,
    )

    oof_xgb[val_mask] = xgb_model.predict(X_val_f)
    test_xgb += xgb_model.predict(X_test) / n_folds
    fold_xgb_rmse = np.sqrt(mean_squared_error(y_val_f, oof_xgb[val_mask]))
    print(f"Fold {fold + 1} XGBoost RMSE: {fold_xgb_rmse:.4f}")

    gc.collect()

# ---------------------------------------------------------
# 7. Non-Negative Least Squares & Robust Ensemble Blending
# ---------------------------------------------------------
lgb_total_rmse = np.sqrt(mean_squared_error(y, oof_lgb))
cb_total_rmse = np.sqrt(mean_squared_error(y, oof_cb))
xgb_total_rmse = np.sqrt(mean_squared_error(y, oof_xgb))

print("\n================ OOF MODEL SCORES ================")
print(f"Overall OOF LightGBM RMSE: {lgb_total_rmse:.4f}")
print(f"Overall OOF CatBoost RMSE: {cb_total_rmse:.4f}")
print(f"Overall OOF XGBoost RMSE:  {xgb_total_rmse:.4f}")
print("==================================================")

A = np.column_stack([oof_lgb, oof_cb, oof_xgb])
nnls_w, _ = nnls(A, y)
if np.sum(nnls_w) > 0:
    init_w = nnls_w / np.sum(nnls_w)
else:
    init_w = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])

def ensemble_loss(weights):
    w = np.maximum(0, weights)
    if np.sum(w) > 0:
        w = w / np.sum(w)
    else:
        w = np.ones(3) / 3.0
    pred = np.clip(A @ w, 0, None)
    return np.sqrt(mean_squared_error(y, pred))

opt_res = minimize(ensemble_loss, init_w, method="Nelder-Mead")
best_w = np.maximum(0, opt_res.x)
if np.sum(best_w) > 0:
    best_w = best_w / np.sum(best_w)
else:
    best_w = init_w

print(
    f"\nOptimal Ensemble Weights -> LightGBM: {best_w[0]:.4f}, CatBoost: {best_w[1]:.4f}, XGBoost: {best_w[2]:.4f}"
)

oof_ensemble = np.clip(best_w[0] * oof_lgb + best_w[1] * oof_cb + best_w[2] * oof_xgb, 0, None)
final_test_preds = np.clip(best_w[0] * test_lgb + best_w[1] * test_cb + best_w[2] * test_xgb, 0, None)

final_score = np.sqrt(mean_squared_error(y, oof_ensemble))
print(f"Ensemble Out-of-Fold RMSE: {final_score:.5f}")

# ---------------------------------------------------------
# 8. Generate Submission File
# ---------------------------------------------------------
id_col_name = "id"
for candidate in ["id", "ID", "Id", "complaint_id", "Complaint ID"]:
    if candidate in test_df.columns:
        id_col_name = candidate
        break

if id_col_name in test_df.columns:
    sub_ids = test_df[id_col_name].values
else:
    sub_ids = np.arange(len(test_df))

submission_df = pd.DataFrame({"id": sub_ids, "Recovery": final_test_preds})
submission_path = "./submission/submission.csv"
submission_df.to_csv(submission_path, index=False)
print(
    f"Submission saved successfully to {submission_path} with shape {submission_df.shape}"
)

# ---------------------------------------------------------
# 9. Output Final Validation Score
# ---------------------------------------------------------
print(f"Final Validation Score: {final_score}")