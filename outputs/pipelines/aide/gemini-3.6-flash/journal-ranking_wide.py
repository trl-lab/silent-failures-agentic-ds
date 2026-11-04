import os
import warnings
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "H index"
y = train_df[target_col].values

drop_cols = [c for c in ["id", "ID", target_col] if c in train_df.columns]
feature_cols = [c for c in train_df.columns if c not in drop_cols]

# Separate feature types
text_cols = []
cat_cols = []
num_cols = []

for c in feature_cols:
    if train_df[c].dtype == "object" or train_df[c].dtype.name == "category":
        if c.lower() in [
            "title",
            "categories",
            "publisher",
            "description",
            "abstract",
            "journal",
        ]:
            text_cols.append(c)
        elif train_df[c].nunique() > 50:
            text_cols.append(c)
        else:
            cat_cols.append(c)
    elif np.issubdtype(train_df[c].dtype, np.number):
        num_cols.append(c)
    else:
        cat_cols.append(c)

# Unified dataset for global category mapping and aggregations
df_all = pd.concat([train_df, test_df], axis=0, ignore_index=True)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Enhanced Text Features: Word & Char TF-IDF + SVD and OOF Ridge Target Estimates
train_text_dfs = []
test_text_dfs = []

for tc in text_cols:
    full_text = df_all[tc].fillna("").astype(str)

    # 1. Word-level TF-IDF Vectorizer + TruncatedSVD
    word_vec = TfidfVectorizer(
        max_features=25000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        stop_words="english",
    )
    word_tfidf = word_vec.fit_transform(full_text)

    n_comp_w = min(32, max(1, word_tfidf.shape[1] - 1))
    if n_comp_w > 1:
        svd_w = TruncatedSVD(n_components=n_comp_w, random_state=42)
        svd_w_mat = svd_w.fit_transform(word_tfidf)
        train_svd_w = pd.DataFrame(
            svd_w_mat[: len(train_df)],
            columns=[f"{tc}_word_svd_{i}" for i in range(n_comp_w)],
        )
        test_svd_w = pd.DataFrame(
            svd_w_mat[len(train_df) :],
            columns=[f"{tc}_word_svd_{i}" for i in range(n_comp_w)],
        )
        train_text_dfs.append(train_svd_w)
        test_text_dfs.append(test_svd_w)

    # OOF Supervised Ridge Target Estimates for Word TF-IDF
    word_tr = word_tfidf[: len(train_df)]
    word_te = word_tfidf[len(train_df) :]

    oof_ridge_word = np.zeros(len(train_df))
    test_ridge_word = np.zeros(len(test_df))

    for train_idx, val_idx in kf.split(train_df):
        ridge_w = Ridge(alpha=10.0, random_state=42)
        ridge_w.fit(word_tr[train_idx], y[train_idx])
        oof_ridge_word[val_idx] = ridge_w.predict(word_tr[val_idx])
        test_ridge_word += ridge_w.predict(word_te) / kf.n_splits

    train_text_dfs.append(pd.DataFrame({f"{tc}_ridge_word_pred": oof_ridge_word}))
    test_text_dfs.append(pd.DataFrame({f"{tc}_ridge_word_pred": test_ridge_word}))

    # 2. Character-level TF-IDF Vectorizer + TruncatedSVD + Ridge
    char_vec = TfidfVectorizer(
        max_features=25000,
        analyzer="char_wb",
        ngram_range=(3, 5),
        sublinear_tf=True,
    )
    char_tfidf = char_vec.fit_transform(full_text)

    n_comp_c = min(32, max(1, char_tfidf.shape[1] - 1))
    if n_comp_c > 1:
        svd_c = TruncatedSVD(n_components=n_comp_c, random_state=42)
        svd_c_mat = svd_c.fit_transform(char_tfidf)
        train_svd_c = pd.DataFrame(
            svd_c_mat[: len(train_df)],
            columns=[f"{tc}_char_svd_{i}" for i in range(n_comp_c)],
        )
        test_svd_c = pd.DataFrame(
            svd_c_mat[len(train_df) :],
            columns=[f"{tc}_char_svd_{i}" for i in range(n_comp_c)],
        )
        train_text_dfs.append(train_svd_c)
        test_text_dfs.append(test_svd_c)

    # OOF Supervised Ridge Target Estimates for Char TF-IDF
    char_tr = char_tfidf[: len(train_df)]
    char_te = char_tfidf[len(train_df) :]

    oof_ridge_char = np.zeros(len(train_df))
    test_ridge_char = np.zeros(len(test_df))

    for train_idx, val_idx in kf.split(train_df):
        ridge_c = Ridge(alpha=10.0, random_state=42)
        ridge_c.fit(char_tr[train_idx], y[train_idx])
        oof_ridge_char[val_idx] = ridge_c.predict(char_tr[val_idx])
        test_ridge_char += ridge_c.predict(char_te) / kf.n_splits

    train_text_dfs.append(pd.DataFrame({f"{tc}_ridge_char_pred": oof_ridge_char}))
    test_text_dfs.append(pd.DataFrame({f"{tc}_ridge_char_pred": test_ridge_char}))


# Numerical engineering: Ratios and Logs
def build_num_features(df, num_cols):
    feats = pd.DataFrame(index=df.index)
    for col in num_cols:
        feats[col] = df[col]
        val = df[col].dropna()
        if not val.empty and val.min() >= 0 and val.skew() > 1.5:
            feats[f"{col}_log1p"] = np.log1p(df[col].clip(lower=0))

    for i in range(len(num_cols)):
        for j in range(i + 1, min(i + 4, len(num_cols))):
            c1, c2 = num_cols[i], num_cols[j]
            feats[f"{c1}_ratio_{c2}"] = df[c1].fillna(0) / (
                df[c2].fillna(0).abs() + 1e-5
            )
    return feats


train_num_df = build_num_features(train_df, num_cols)
test_num_df = build_num_features(test_df, num_cols)

# Group Aggregations
group_df_all = pd.DataFrame(index=df_all.index)

for cat in cat_cols:
    if df_all[cat].nunique() > 1 and df_all[cat].nunique() < 500:
        for num in num_cols[:5]:
            grouped = df_all.groupby(cat)[num]
            group_df_all[f"{cat}_{num}_mean"] = grouped.transform("mean")
            group_df_all[f"{cat}_{num}_std"] = grouped.transform("std")
            group_df_all[f"{cat}_{num}_median"] = grouped.transform("median")

train_group_df = group_df_all.iloc[: len(train_df)].reset_index(drop=True)
test_group_df = group_df_all.iloc[len(train_df) :].reset_index(drop=True)

# Out-of-fold Target Encoding, Frequency Encoding, and Unified Categorical Dtype
global_mean = y.mean()

train_cat_feats = pd.DataFrame(index=train_df.index)
test_cat_feats = pd.DataFrame(index=test_df.index)

for cat in cat_cols:
    # Frequency Encoding
    freq = df_all[cat].value_counts()
    train_cat_feats[f"{cat}_freq"] = train_df[cat].map(freq).fillna(0)
    test_cat_feats[f"{cat}_freq"] = test_df[cat].map(freq).fillna(0)

    # Target Encoding (OOF for train)
    te_col = f"{cat}_te"
    train_cat_feats[te_col] = np.nan
    test_cat_feats[te_col] = 0.0

    smoothing = 10
    for train_idx, val_idx in kf.split(train_df):
        tr_cat = train_df.iloc[train_idx][cat]
        tr_y = y[train_idx]
        stats = (
            pd.DataFrame({"cat": tr_cat, "y": tr_y})
            .groupby("cat")["y"]
            .agg(["count", "mean"])
        )
        smooth_te = (stats["count"] * stats["mean"] + smoothing * global_mean) / (
            stats["count"] + smoothing
        )
        val_cat = train_df.iloc[val_idx][cat]
        train_cat_feats.loc[train_cat_feats.index[val_idx], te_col] = val_cat.map(
            smooth_te
        ).fillna(global_mean)

    stats_full = (
        pd.DataFrame({"cat": train_df[cat], "y": y})
        .groupby("cat")["y"]
        .agg(["count", "mean"])
    )
    smooth_te_full = (
        stats_full["count"] * stats_full["mean"] + smoothing * global_mean
    ) / (stats_full["count"] + smoothing)
    test_cat_feats[te_col] = test_df[cat].map(smooth_te_full).fillna(global_mean)

    # Consistent Categorical dtype across train and test
    cat_type = pd.CategoricalDtype(categories=df_all[cat].dropna().unique())
    train_cat_feats[cat] = train_df[cat].astype(cat_type)
    test_cat_feats[cat] = test_df[cat].astype(cat_type)

# Combine all engineered features
X_train = pd.concat(
    [train_num_df, train_group_df, train_cat_feats] + train_text_dfs, axis=1
)
X_test = pd.concat([test_num_df, test_group_df, test_cat_feats] + test_text_dfs, axis=1)

# 5-Fold Cross Validation with LightGBM Regressor
lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "n_estimators": 3000,
    "learning_rate": 0.02,
    "num_leaves": 63,
    "max_depth": -1,
    "min_child_samples": 15,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": -1,
}

oof_preds = np.zeros(len(train_df))
test_preds = np.zeros(len(test_df))

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y)):
    X_tr, y_tr = X_train.iloc[train_idx], y[train_idx]
    X_va, y_va = X_train.iloc[val_idx], y[val_idx]

    model = LGBMRegressor(**lgb_params)
    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[
            early_stopping(stopping_rounds=50, verbose=False),
            log_evaluation(period=0),
        ],
    )

    oof_preds[val_idx] = model.predict(X_va)
    test_preds += model.predict(X_test) / kf.n_splits

# Post-process predictions
oof_preds = np.clip(oof_preds, 0, None)
test_preds = np.clip(test_preds, 0, None)

cv_rmse = root_mean_squared_error(y, oof_preds)
print(f"Validation RMSE: {cv_rmse:.5f}")

# Save submission
os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame({"id": np.arange(len(test_preds)), "H index": test_preds})
submission.to_csv("./working/submission.csv", index=False)
