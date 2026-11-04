import itertools
import os
import warnings
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

import lightgbm as lgb
import xgboost as xgb

try:
    from catboost import CatBoostClassifier

    has_catboost = True
except ImportError:
    has_catboost = False
    from sklearn.ensemble import HistGradientBoostingClassifier

warnings.filterwarnings("ignore")

# 1. Ensure working and submission directories exist
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# 2. Read train and test parquet datasets
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "risk_decision"
assert (
    target_col in train_df.columns
), f"Target column '{target_col}' missing from train.parquet"

# Label encode target
le_target = LabelEncoder()
y_train = le_target.fit_transform(train_df[target_col].astype(str))
num_classes = len(le_target.classes_)

# Preserve submission IDs
if "id" in test_df.columns:
    test_id = test_df["id"].values
else:
    test_id = np.arange(len(test_df))

# Separate features from target/ID
X_train = train_df.drop(columns=[target_col], errors="ignore")
X_test = test_df.copy()

if target_col in X_test.columns:
    X_test = X_test.drop(columns=[target_col])
if "id" in X_train.columns:
    X_train = X_train.drop(columns=["id"])
if "id" in X_test.columns:
    X_test = X_test.drop(columns=["id"])

n_train = len(X_train)
n_test = len(X_test)

# Combine for unsupervised feature engineering
combined_df = pd.concat([X_train, X_test], axis=0).reset_index(drop=True)

# 3. Dynamic Column Type Identification
date_cols = []
text_cols = []
cat_cols = []
num_cols = []

for col in combined_df.columns:
    if (
        "date" in col.lower()
        or "time" in col.lower()
        or pd.api.types.is_datetime64_any_dtype(combined_df[col])
    ):
        date_cols.append(col)
    elif pd.api.types.is_numeric_dtype(combined_df[col]):
        num_cols.append(col)
    else:
        sample_vals = combined_df[col].dropna().astype(str)
        if len(sample_vals) > 0:
            avg_words = sample_vals.str.split().str.len().mean()
            avg_len = sample_vals.str.len().mean()
            if avg_words > 3 or avg_len > 30:
                text_cols.append(col)
            else:
                cat_cols.append(col)
        else:
            cat_cols.append(col)

# 4. Feature Engineering Pipeline

# A. Temporal & Cyclical Feature Extraction
for col in date_cols:
    parsed_dates = pd.to_datetime(combined_df[col], errors="coerce")
    combined_df[f"{col}_year"] = parsed_dates.dt.year
    combined_df[f"{col}_month"] = parsed_dates.dt.month
    combined_df[f"{col}_day"] = parsed_dates.dt.day
    combined_df[f"{col}_dayofweek"] = parsed_dates.dt.dayofweek
    combined_df[f"{col}_dayofyear"] = parsed_dates.dt.dayofyear
    combined_df[f"{col}_quarter"] = parsed_dates.dt.quarter

    combined_df[f"{col}_sin_month"] = np.sin(2 * np.pi * parsed_dates.dt.month / 12.0)
    combined_df[f"{col}_cos_month"] = np.cos(2 * np.pi * parsed_dates.dt.month / 12.0)
    combined_df[f"{col}_sin_dow"] = np.sin(2 * np.pi * parsed_dates.dt.dayofweek / 7.0)
    combined_df[f"{col}_cos_dow"] = np.cos(2 * np.pi * parsed_dates.dt.dayofweek / 7.0)

    min_date = parsed_dates.min()
    if pd.notnull(min_date):
        combined_df[f"{col}_days_since_min"] = (parsed_dates - min_date).dt.days

    num_cols.extend(
        [
            f"{col}_year",
            f"{col}_month",
            f"{col}_day",
            f"{col}_dayofweek",
            f"{col}_dayofyear",
            f"{col}_quarter",
            f"{col}_sin_month",
            f"{col}_cos_month",
            f"{col}_sin_dow",
            f"{col}_cos_dow",
            f"{col}_days_since_min",
        ]
    )

# B. Unstructured Text Meta-Features + TF-IDF SVD Embeddings
for col in text_cols:
    text_series = combined_df[col].fillna("").astype(str)

    combined_df[f"{col}_len"] = text_series.str.len()
    combined_df[f"{col}_word_count"] = text_series.str.split().str.len()
    combined_df[f"{col}_upper_count"] = text_series.apply(
        lambda s: sum(1 for c in s if c.isupper())
    )
    combined_df[f"{col}_digit_count"] = text_series.apply(
        lambda s: sum(1 for c in s if c.isdigit())
    )

    num_cols.extend(
        [f"{col}_len", f"{col}_word_count", f"{col}_upper_count", f"{col}_digit_count"]
    )

    train_text = combined_df[col].iloc[:n_train].fillna("").astype(str)
    test_text = combined_df[col].iloc[n_train:].fillna("").astype(str)

    tfidf_word = TfidfVectorizer(max_features=5000, stop_words="english", ngram_range=(1, 2))
    tfidf_char = TfidfVectorizer(max_features=5000, analyzer="char_wb", ngram_range=(3, 5))

    tfidf_train = hstack([tfidf_word.fit_transform(train_text), tfidf_char.fit_transform(train_text)])
    tfidf_test = hstack([tfidf_word.transform(test_text), tfidf_char.transform(test_text)])

    n_comp = min(32, max(2, tfidf_train.shape[1] - 1))
    svd = TruncatedSVD(n_components=n_comp, random_state=42)
    svd_train = svd.fit_transform(tfidf_train)
    svd_test = svd.transform(tfidf_test)
    svd_all = np.vstack([svd_train, svd_test])

    for k in range(n_comp):
        col_name = f"{col}_svd_{k}"
        combined_df[col_name] = svd_all[:, k]
        num_cols.append(col_name)

# C. Categorical Frequency Encoding & Cross-Pair Interactions
for col in cat_cols:
    combined_df[col] = combined_df[col].astype(str).fillna("missing")
    freq = combined_df.iloc[:n_train][col].value_counts(normalize=True)
    combined_df[f"{col}_freq"] = combined_df[col].map(freq).fillna(0.0)
    num_cols.append(f"{col}_freq")

top_cats = [c for c in cat_cols if combined_df[c].nunique() < 500][:4]
interaction_cols = []
for c1, c2 in itertools.combinations(top_cats, 2):
    pair_col = f"{c1}_X_{c2}"
    combined_df[pair_col] = combined_df[c1] + "___" + combined_df[c2]
    freq = combined_df.iloc[:n_train][pair_col].value_counts(normalize=True)
    combined_df[f"{pair_col}_freq"] = combined_df[pair_col].map(freq).fillna(0.0)
    num_cols.append(f"{pair_col}_freq")
    interaction_cols.append(pair_col)

# Group Statistics fit on train
train_comb = combined_df.iloc[:n_train]
key_num_cols = [c for c in num_cols if not c.endswith("_freq") and "svd" not in c][:5]
for cat in top_cats:
    for num in key_num_cols:
        for agg_func in ["mean", "std", "min", "max", "median"]:
            agg_col = f"{cat}_{num}_{agg_func}"
            stats = train_comb.groupby(cat)[num].agg(agg_func)
            combined_df[agg_col] = combined_df[cat].map(stats)
            num_cols.append(agg_col)

all_cat_cols = cat_cols + interaction_cols
for col in all_cat_cols:
    le = LabelEncoder()
    combined_df[f"{col}_encoded"] = le.fit_transform(combined_df[col])

# Split back into train and test sets
processed_train = combined_df.iloc[:n_train].copy()
processed_test = combined_df.iloc[n_train:].copy().reset_index(drop=True)

# D. Strict Out-Of-Fold Smoothed Target Encoding (Multi-Class)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
target_encoded_feature_names = []

te_cats = [
    c
    for c in cat_cols
    if combined_df[c].nunique() > 1 and combined_df[c].nunique() < 1000
][:6]
m_smooth = 10.0
global_priors = np.bincount(y_train, minlength=num_classes) / float(len(y_train))

for col in te_cats:
    for c_idx in range(num_classes):
        te_col_name = f"TE_{col}_class_{c_idx}"
        target_encoded_feature_names.append(te_col_name)
        processed_train[te_col_name] = 0.0
        processed_test[te_col_name] = 0.0

        test_te_sum = np.zeros(n_test)

        for train_idx, val_idx in skf.split(processed_train, y_train):
            tr_sub = processed_train.iloc[train_idx]
            y_sub = pd.Series((y_train[train_idx] == c_idx).astype(float), index=tr_sub.index)

            cat_group = tr_sub.groupby(col)
            counts = cat_group.size()
            target_sums = y_sub.groupby(tr_sub[col]).sum()

            smooth_te = (target_sums + m_smooth * global_priors[c_idx]) / (
                counts + m_smooth
            )

            val_cat = processed_train.iloc[val_idx][col]
            processed_train.loc[val_idx, te_col_name] = (
                val_cat.map(smooth_te).fillna(global_priors[c_idx]).values
            )

            test_cat = processed_test[col]
            test_te_sum += test_cat.map(smooth_te).fillna(global_priors[c_idx]).values

        processed_test[te_col_name] = test_te_sum / 5.0

# Build numeric final feature matrices
final_feature_cols = (
    [f"{col}_encoded" for col in all_cat_cols]
    + [c for c in num_cols if c in processed_train.columns]
    + target_encoded_feature_names
)
final_feature_cols = list(dict.fromkeys(final_feature_cols))

X_tr = processed_train[final_feature_cols].values
X_te = processed_test[final_feature_cols].values

X_tr = np.nan_to_num(X_tr, nan=0.0, posinf=0.0, neginf=0.0)
X_te = np.nan_to_num(X_te, nan=0.0, posinf=0.0, neginf=0.0)

# Save processed datasets to working directory
processed_train_df = pd.DataFrame(X_tr, columns=final_feature_cols)
processed_train_df["target"] = y_train
processed_train_df.to_parquet("./working/processed_train.parquet")

processed_test_df = pd.DataFrame(X_te, columns=final_feature_cols)
processed_test_df.to_parquet("./working/processed_test.parquet")

# 5. Multi-Model Ensembling & Cross-Validation
oof_lgb = np.zeros((n_train, num_classes))
test_lgb = np.zeros((n_test, num_classes))

oof_xgb = np.zeros((n_train, num_classes))
test_xgb = np.zeros((n_test, num_classes))

oof_cat = np.zeros((n_train, num_classes))
test_cat = np.zeros((n_test, num_classes))

oof_et = np.zeros((n_train, num_classes))
test_et = np.zeros((n_test, num_classes))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_tr, y_train)):
    X_train_f, y_train_f = X_tr[train_idx], y_train[train_idx]
    X_val_f, y_val_f = X_tr[val_idx], y_train[val_idx]

    # Model A: LightGBM
    if num_classes > 2:
        lgb_params = {
            "objective": "multiclass",
            "num_class": num_classes,
            "metric": "multi_logloss",
            "boosting_type": "gbdt",
            "learning_rate": 0.03,
            "num_leaves": 31,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "min_child_samples": 20,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "random_state": 42 + fold,
            "n_jobs": -1,
            "verbose": -1,
        }
    else:
        lgb_params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "learning_rate": 0.03,
            "num_leaves": 31,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "min_child_samples": 20,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "random_state": 42 + fold,
            "n_jobs": -1,
            "verbose": -1,
        }

    dtrain = lgb.Dataset(X_train_f, label=y_train_f)
    dval = lgb.Dataset(X_val_f, label=y_val_f, reference=dtrain)

    model_lgb = lgb.train(
        lgb_params,
        dtrain,
        num_boost_round=1000,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
    )

    val_pred_lgb = model_lgb.predict(X_val_f)
    test_pred_lgb = model_lgb.predict(X_te)
    if num_classes == 2:
        val_pred_lgb = np.vstack([1 - val_pred_lgb, val_pred_lgb]).T
        test_pred_lgb = np.vstack([1 - test_pred_lgb, test_pred_lgb]).T

    oof_lgb[val_idx] = val_pred_lgb
    test_lgb += test_pred_lgb / 5.0

    # Model B: XGBoost
    if num_classes > 2:
        model_xgb = xgb.XGBClassifier(
            n_estimators=1000,
            learning_rate=0.03,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42 + fold,
            tree_method="hist",
            early_stopping_rounds=50,
            eval_metric="mlogloss",
            n_jobs=-1,
        )
    else:
        model_xgb = xgb.XGBClassifier(
            n_estimators=1000,
            learning_rate=0.03,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42 + fold,
            tree_method="hist",
            early_stopping_rounds=50,
            eval_metric="logloss",
            n_jobs=-1,
        )

    model_xgb.fit(
        X_train_f,
        y_train_f,
        eval_set=[(X_val_f, y_val_f)],
        verbose=False,
    )

    val_pred_xgb = model_xgb.predict_proba(X_val_f)
    test_pred_xgb = model_xgb.predict_proba(X_te)
    oof_xgb[val_idx] = val_pred_xgb
    test_xgb += test_pred_xgb / 5.0

    # Model C: CatBoost / HistGradientBoosting Fallback
    if has_catboost:
        if num_classes > 2:
            model_cat = CatBoostClassifier(
                iterations=1000,
                learning_rate=0.04,
                depth=6,
                l2_leaf_reg=3.0,
                random_seed=42 + fold,
                early_stopping_rounds=50,
                verbose=False,
                loss_function="MultiClass",
                thread_count=-1,
            )
        else:
            model_cat = CatBoostClassifier(
                iterations=1000,
                learning_rate=0.04,
                depth=6,
                l2_leaf_reg=3.0,
                random_seed=42 + fold,
                early_stopping_rounds=50,
                verbose=False,
                loss_function="Logloss",
                thread_count=-1,
            )

        model_cat.fit(
            X_train_f,
            y_train_f,
            eval_set=(X_val_f, y_val_f),
            verbose=False,
        )
        val_pred_cat = model_cat.predict_proba(X_val_f)
        test_pred_cat = model_cat.predict_proba(X_te)
    else:
        model_cat = HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.05,
            max_depth=8,
            random_state=42 + fold,
        )
        model_cat.fit(X_train_f, y_train_f)
        val_pred_cat = model_cat.predict_proba(X_val_f)
        test_pred_cat = model_cat.predict_proba(X_te)

    oof_cat[val_idx] = val_pred_cat
    test_cat += test_pred_cat / 5.0

    # Model D: ExtraTreesClassifier
    model_et = ExtraTreesClassifier(
        n_estimators=300,
        max_depth=15,
        min_samples_split=5,
        random_state=42 + fold,
        n_jobs=-1,
    )
    model_et.fit(X_train_f, y_train_f)
    val_pred_et = model_et.predict_proba(X_val_f)
    test_pred_et = model_et.predict_proba(X_te)
    oof_et[val_idx] = val_pred_et
    test_et += test_pred_et / 5.0

# 6. Stacking Meta-Classifier
oof_meta = np.hstack([oof_lgb, oof_xgb, oof_cat, oof_et])
test_meta = np.hstack([test_lgb, test_xgb, test_cat, test_et])

oof_final_probs = np.zeros((n_train, num_classes))
for m_tr_idx, m_val_idx in skf.split(oof_meta, y_train):
    meta_sub = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    meta_sub.fit(oof_meta[m_tr_idx], y_train[m_tr_idx])
    oof_final_probs[m_val_idx] = meta_sub.predict_proba(oof_meta[m_val_idx])

meta_model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
meta_model.fit(oof_meta, y_train)

final_oof_labels = np.argmax(oof_final_probs, axis=1)
final_val_score = accuracy_score(y_train, final_oof_labels)

# 7. Compute Final Predictions & Save Submission
test_final_probs = meta_model.predict_proba(test_meta)
final_test_labels = np.argmax(test_final_probs, axis=1)

final_test_preds = le_target.inverse_transform(final_test_labels)

sub_df = pd.DataFrame({"id": test_id, "risk_decision": final_test_preds})
sub_df.to_csv("./submission/submission.csv", index=False)

print(f"Final Validation Score: {final_val_score}")