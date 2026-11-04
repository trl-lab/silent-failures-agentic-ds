import os
import warnings
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")

# 1. Load Data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "Task Assigned"

# Label encode target variable
le = LabelEncoder()
y = le.fit_transform(train_df[target_col].astype(str))

train_df = train_df.drop(columns=[target_col])

n_train = len(train_df)
combined = pd.concat([train_df, test_df], ignore_index=True)

# Identify column types
num_cols = combined.select_dtypes(include=[np.number]).columns.tolist()
if "id" in num_cols:
    num_cols.remove("id")

object_cols = combined.select_dtypes(include=["object", "category"]).columns.tolist()

text_cols = []
cat_cols = []
for col in object_cols:
    non_nulls = combined[col].dropna().astype(str)
    if len(non_nulls) == 0:
        cat_cols.append(col)
        continue
    avg_len = non_nulls.str.len().mean()
    if avg_len > 30 and non_nulls.nunique() > 50:
        text_cols.append(col)
    else:
        cat_cols.append(col)

# 2. Extract Features
features_dict = {}

# Numerical features
for col in num_cols:
    features_dict[col] = combined[col].fillna(combined[col].median()).values
    features_dict[f"{col}_isna"] = combined[col].isna().astype(int).values

# Categorical features: Frequency and Label Encoding
for col in cat_cols:
    s = combined[col].fillna("Missing").astype(str)
    freq_map = s.value_counts().to_dict()
    features_dict[f"{col}_freq"] = s.map(freq_map).values
    features_dict[f"{col}_cat"] = pd.Categorical(s).codes

# Text features: Word & Char TF-IDF SVD, Text Stats, and Logistic Regression OOF
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for col in text_cols:
    s = combined[col].fillna("").astype(str)
    features_dict[f"{col}_len"] = s.str.len().values
    features_dict[f"{col}_word_count"] = s.str.split().str.len().values
    features_dict[f"{col}_cap_ratio"] = s.apply(
        lambda x: sum(1 for c in x if c.isupper()) / (len(x) + 1)
    ).values

    # Word-level TF-IDF + SVD
    tfidf_word = TfidfVectorizer(
        max_features=10000, stop_words="english", ngram_range=(1, 2)
    )
    word_mat = tfidf_word.fit_transform(s)
    svd_word = TruncatedSVD(n_components=25, random_state=42)
    svd_word_feats = svd_word.fit_transform(word_mat)
    for i in range(25):
        features_dict[f"{col}_word_svd_{i}"] = svd_word_feats[:, i]

    # Char-level TF-IDF + SVD
    tfidf_char = TfidfVectorizer(
        max_features=10000, analyzer="char_wb", ngram_range=(3, 5)
    )
    char_mat = tfidf_char.fit_transform(s)
    svd_char = TruncatedSVD(n_components=15, random_state=42)
    svd_char_feats = svd_char.fit_transform(char_mat)
    for i in range(15):
        features_dict[f"{col}_char_svd_{i}"] = svd_char_feats[:, i]

    # Out-of-fold Logistic Regression meta-feature on word TF-IDF
    logreg_oof = np.zeros(n_train)
    test_logreg_preds = np.zeros(len(test_df))
    X_text_train = word_mat[:n_train]
    X_text_test = word_mat[n_train:]

    for train_idx, val_idx in kf.split(X_text_train, y):
        lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        lr.fit(X_text_train[train_idx], y[train_idx])
        logreg_oof[val_idx] = lr.predict_proba(X_text_train[val_idx])[:, 1]
        test_logreg_preds += lr.predict_proba(X_text_test)[:, 1] / kf.n_splits

    features_dict[f"{col}_logreg_oof"] = np.concatenate([logreg_oof, test_logreg_preds])

# Target encoding for categorical variables
for col in cat_cols:
    s_train = combined.iloc[:n_train][col].fillna("Missing").astype(str)
    s_test = combined.iloc[n_train:][col].fillna("Missing").astype(str)

    te_oof = np.zeros(n_train)
    te_test = np.zeros(len(test_df))
    global_mean = y.mean()

    smooth = 10
    full_counts = s_train.value_counts()
    full_sums = (
        pd.DataFrame({"cat": s_train, "target": y}).groupby("cat")["target"].sum()
    )
    smoothed_full = (full_sums + smooth * global_mean) / (full_counts + smooth)

    for train_idx, val_idx in kf.split(s_train, y):
        tr_s, tr_y = s_train.iloc[train_idx], y[train_idx]
        va_s = s_train.iloc[val_idx]

        counts = tr_s.value_counts()
        sums = (
            pd.DataFrame({"cat": tr_s, "target": tr_y}).groupby("cat")["target"].sum()
        )
        smoothed_fold = (sums + smooth * global_mean) / (counts + smooth)

        te_oof[val_idx] = va_s.map(smoothed_fold).fillna(global_mean).values

    te_test = s_test.map(smoothed_full).fillna(global_mean).values
    features_dict[f"{col}_te"] = np.concatenate([te_oof, te_test])

features_df = pd.DataFrame(features_dict)
X_all = features_df.values
X_train = X_all[:n_train]
X_test = X_all[n_train:]

# 3. Model Ensembling with 5-Fold Stratified K-Fold
oof_lgb = np.zeros(n_train)
oof_cat = np.zeros(n_train)
oof_xgb = np.zeros(n_train)

test_lgb = np.zeros(len(test_df))
test_cat = np.zeros(len(test_df))
test_xgb = np.zeros(len(test_df))

lgb_params = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "learning_rate": 0.03,
    "num_leaves": 31,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "verbose": -1,
    "random_state": 42,
    "n_estimators": 1000,
}

xgb_params = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "learning_rate": 0.03,
    "max_depth": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "n_estimators": 1000,
    "n_jobs": -1,
}

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y)):
    X_tr, y_tr = X_train[train_idx], y[train_idx]
    X_va, y_va = X_train[val_idx], y[val_idx]

    # LightGBM
    dtrain = lgb.Dataset(X_tr, label=y_tr)
    dval = lgb.Dataset(X_va, label=y_va, reference=dtrain)
    model_lgb = lgb.train(
        lgb_params,
        dtrain,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )
    oof_lgb[val_idx] = model_lgb.predict(X_va)
    test_lgb += model_lgb.predict(X_test) / kf.n_splits

    # CatBoost
    model_cat = CatBoostClassifier(
        iterations=800,
        learning_rate=0.03,
        depth=5,
        eval_metric="AUC",
        random_seed=42,
        verbose=False,
    )
    model_cat.fit(
        X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=50, verbose=False
    )
    oof_cat[val_idx] = model_cat.predict_proba(X_va)[:, 1]
    test_cat += model_cat.predict_proba(X_test)[:, 1] / kf.n_splits

    # XGBoost
    model_xgb = XGBClassifier(**xgb_params, early_stopping_rounds=50)
    model_xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    oof_xgb[val_idx] = model_xgb.predict_proba(X_va)[:, 1]
    test_xgb += model_xgb.predict_proba(X_test)[:, 1] / kf.n_splits

# Rank Average Ensemble
oof_lgb_rank = rankdata(oof_lgb) / len(oof_lgb)
oof_cat_rank = rankdata(oof_cat) / len(oof_cat)
oof_xgb_rank = rankdata(oof_xgb) / len(oof_xgb)

test_lgb_rank = rankdata(test_lgb) / len(test_lgb)
test_cat_rank = rankdata(test_cat) / len(test_cat)
test_xgb_rank = rankdata(test_xgb) / len(test_xgb)

oof_ensemble = 0.4 * oof_lgb_rank + 0.3 * oof_cat_rank + 0.3 * oof_xgb_rank
test_ensemble = 0.4 * test_lgb_rank + 0.3 * test_cat_rank + 0.3 * test_xgb_rank

oof_auc = roc_auc_score(y, oof_ensemble)
print(f"Validation Out-Of-Fold ROC AUC: {oof_auc:.5f}")

# Save Submission
os.makedirs("./working", exist_ok=True)
sub_id = test_df["id"].values if "id" in test_df.columns else np.arange(len(test_df))
submission = pd.DataFrame(
    {
        "id": sub_id,
        "Task Assigned": test_ensemble,
    }
)
submission.to_csv("./working/submission.csv", index=False)
