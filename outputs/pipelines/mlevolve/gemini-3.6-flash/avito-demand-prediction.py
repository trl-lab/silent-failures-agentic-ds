import gc
import os
import re
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder

# Create required directories
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

print("Loading raw parquet data...")
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

n_train = len(train_df)
n_test = len(test_df)

# Target vector
y_train = train_df["deal_probability"].values.astype(np.float32)

# Set up 5-fold CV splits on training data
kf = KFold(n_splits=5, shuffle=True, random_state=42)
train_df["fold"] = -1
for fold, (train_idx, val_idx) in enumerate(kf.split(train_df)):
    train_df.iloc[val_idx, train_df.columns.get_loc("fold")] = fold
test_df["fold"] = -1

# Combine datasets for consistent processing
df_all = pd.concat(
    [train_df.drop(columns=["deal_probability"]), test_df], ignore_index=True
)

print("Engineering temporal and price features...")

# Temporal Features
df_all["activation_date"] = pd.to_datetime(df_all["activation_date"])
df_all["dayofweek"] = df_all["activation_date"].dt.dayofweek.astype(np.int8)
df_all["dayofmonth"] = df_all["activation_date"].dt.day.astype(np.int8)
df_all["is_weekend"] = df_all["dayofweek"].isin([5, 6]).astype(np.int8)

# Price features
df_all["price_isnan"] = df_all["price"].isna().astype(np.int8)
df_all["price_log"] = np.log1p(df_all["price"].fillna(0).clip(lower=0)).astype(
    np.float32
)

train_mask = df_all.index < n_train

# Category-level price aggregations (mean, std, median)
cat_stats = df_all[train_mask].groupby("category_name")["price_log"].agg(["mean", "std", "median"])
cat_mean = df_all["category_name"].map(cat_stats["mean"].to_dict()).fillna(0)
cat_std = df_all["category_name"].map(cat_stats["std"].to_dict()).fillna(1.0)
cat_med = df_all["category_name"].map(cat_stats["median"].to_dict()).fillna(0)

df_all["cat_price_mean"] = cat_mean.astype(np.float32)
df_all["cat_price_std"] = cat_std.astype(np.float32)
df_all["cat_price_median"] = cat_med.astype(np.float32)
df_all["price_diff_cat"] = (df_all["price_log"] - cat_med).astype(np.float32)
df_all["price_ratio_cat"] = (df_all["price_log"] / (cat_mean + 1e-5)).astype(np.float32)

# Region-level price aggregations (mean, std, median)
reg_stats = df_all[train_mask].groupby("region")["price_log"].agg(["mean", "std", "median"])
reg_mean = df_all["region"].map(reg_stats["mean"].to_dict()).fillna(0)
reg_std = df_all["region"].map(reg_stats["std"].to_dict()).fillna(1.0)
reg_med = df_all["region"].map(reg_stats["median"].to_dict()).fillna(0)

df_all["reg_price_mean"] = reg_mean.astype(np.float32)
df_all["reg_price_std"] = reg_std.astype(np.float32)
df_all["reg_price_median"] = reg_med.astype(np.float32)
df_all["price_diff_region"] = (df_all["price_log"] - reg_med).astype(np.float32)
df_all["price_ratio_region"] = (df_all["price_log"] / (reg_mean + 1e-5)).astype(np.float32)

# Fine-grained joint hierarchy price aggregations (cat + param_1, region + city, param_1)
df_all["cat_param1"] = df_all["category_name"].astype(str) + "_" + df_all["param_1"].astype(str)
df_all["reg_city"] = df_all["region"].astype(str) + "_" + df_all["city"].astype(str)

cp1_stats = df_all[train_mask].groupby("cat_param1")["price_log"].agg(["mean", "std", "median"])
cp1_mean = df_all["cat_param1"].map(cp1_stats["mean"].to_dict()).fillna(0)
cp1_std = df_all["cat_param1"].map(cp1_stats["std"].to_dict()).fillna(1.0)
cp1_med = df_all["cat_param1"].map(cp1_stats["median"].to_dict()).fillna(0)

df_all["cat_param1_price_mean"] = cp1_mean.astype(np.float32)
df_all["cat_param1_price_std"] = cp1_std.astype(np.float32)
df_all["cat_param1_price_median"] = cp1_med.astype(np.float32)
df_all["price_diff_cat_param1"] = (df_all["price_log"] - cp1_med).astype(np.float32)
df_all["price_ratio_cat_param1"] = (df_all["price_log"] / (cp1_mean + 1e-5)).astype(np.float32)

rc_stats = df_all[train_mask].groupby("reg_city")["price_log"].agg(["mean", "std", "median"])
rc_mean = df_all["reg_city"].map(rc_stats["mean"].to_dict()).fillna(0)
rc_std = df_all["reg_city"].map(rc_stats["std"].to_dict()).fillna(1.0)
rc_med = df_all["reg_city"].map(rc_stats["median"].to_dict()).fillna(0)

df_all["reg_city_price_mean"] = rc_mean.astype(np.float32)
df_all["reg_city_price_std"] = rc_std.astype(np.float32)
df_all["reg_city_price_median"] = rc_med.astype(np.float32)
df_all["price_diff_reg_city"] = (df_all["price_log"] - rc_med).astype(np.float32)
df_all["price_ratio_reg_city"] = (df_all["price_log"] / (rc_mean + 1e-5)).astype(np.float32)

p1_stats = df_all[train_mask].groupby("param_1")["price_log"].agg(["mean", "std", "median"])
p1_mean = df_all["param_1"].map(p1_stats["mean"].to_dict()).fillna(0)
p1_std = df_all["param_1"].map(p1_stats["std"].to_dict()).fillna(1.0)
p1_med = df_all["param_1"].map(p1_stats["median"].to_dict()).fillna(0)

df_all["param1_price_mean"] = p1_mean.astype(np.float32)
df_all["param1_price_std"] = p1_std.astype(np.float32)
df_all["param1_price_median"] = p1_med.astype(np.float32)
df_all["price_diff_param1"] = (df_all["price_log"] - p1_med).astype(np.float32)
df_all["price_ratio_param1"] = (df_all["price_log"] / (p1_mean + 1e-5)).astype(np.float32)

# Raw price ratio feature
cat_raw_mean = df_all["category_name"].map(df_all[train_mask].groupby("category_name")["price"].mean().to_dict()).fillna(1.0)
df_all["price_raw_ratio_cat"] = (df_all["price"].fillna(0) / (cat_raw_mean + 1e-5)).astype(np.float32)

# User activity features
user_counts = df_all[train_mask]["user_id"].value_counts().to_dict()
df_all["user_item_count"] = (
    df_all["user_id"].map(user_counts).fillna(1).astype(np.int32)
)
df_all["item_seq_number_log"] = np.log1p(df_all["item_seq_number"].fillna(0)).astype(
    np.float32
)
df_all["user_seq_ratio"] = (
    df_all["item_seq_number"] / (df_all["user_item_count"] + 1)
).astype(np.float32)

print("Engineering text statistical features...")

for col in ["title", "description"]:
    text_s = df_all[col].fillna("").astype(str)
    df_all[f"{col}_len"] = text_s.str.len().astype(np.int32)
    df_all[f"{col}_word_count"] = text_s.str.split().str.len().astype(np.int32)
    df_all[f"{col}_num_digits"] = text_s.str.count(r"\d").astype(np.int32)
    df_all[f"{col}_num_caps"] = text_s.str.count(r"[A-ZА-Я]").astype(np.int32)
    df_all[f"{col}_caps_ratio"] = (
        df_all[f"{col}_num_caps"] / (df_all[f"{col}_len"] + 1)
    ).astype(np.float32)

df_all["desc_isnan"] = df_all["description"].isna().astype(np.int8)

# Text length ratio and numerical count interaction features
df_all["title_desc_len_ratio"] = (
    df_all["title_len"] / (df_all["description_len"] + 1)
).astype(np.float32)
df_all["title_desc_word_ratio"] = (
    df_all["title_word_count"] / (df_all["description_word_count"] + 1)
).astype(np.float32)
df_all["total_digits"] = (
    df_all["title_num_digits"] + df_all["description_num_digits"]
).astype(np.int32)
df_all["digits_ratio"] = (
    df_all["title_num_digits"] / (df_all["description_num_digits"] + 1)
).astype(np.float32)
df_all["digits_per_char"] = (
    df_all["total_digits"] / (df_all["title_len"] + df_all["description_len"] + 1)
).astype(np.float32)

print("Computing high-dimensional TF-IDF, OOF Ridge predictions, and expanded SVD text features...")
from scipy.sparse import hstack

title_text = df_all["title"].fillna("").astype(str)
desc_text = df_all["description"].fillna("").astype(str)

# Title TF-IDF & Ridge OOF
vec_title = TfidfVectorizer(
    max_features=40000, stop_words=None, sublinear_tf=True, ngram_range=(1, 2)
)
X_title = vec_title.fit_transform(title_text)

oof_ridge_title = np.zeros(n_train, dtype=np.float32)
test_ridge_title = np.zeros(n_test, dtype=np.float32)
X_title_tr = X_title[:n_train]
X_title_te = X_title[n_train:]

for fold in range(5):
    val_m = train_df["fold"].values == fold
    tr_m = train_df["fold"].values != fold
    r_model = Ridge(alpha=5.0, solver="lsqr", random_state=42)
    r_model.fit(X_title_tr[tr_m], y_train[tr_m])
    oof_ridge_title[val_m] = r_model.predict(X_title_tr[val_m])
    test_ridge_title += r_model.predict(X_title_te) / 5.0

df_all["ridge_oof_title"] = np.concatenate([oof_ridge_title, test_ridge_title]).astype(np.float32)

# Description TF-IDF & Ridge OOF
vec_desc = TfidfVectorizer(
    max_features=60000, stop_words=None, sublinear_tf=True, ngram_range=(1, 2)
)
X_desc = vec_desc.fit_transform(desc_text)

oof_ridge_desc = np.zeros(n_train, dtype=np.float32)
test_ridge_desc = np.zeros(n_test, dtype=np.float32)
X_desc_tr = X_desc[:n_train]
X_desc_te = X_desc[n_train:]

for fold in range(5):
    val_m = train_df["fold"].values == fold
    tr_m = train_df["fold"].values != fold
    r_model = Ridge(alpha=5.0, solver="lsqr", random_state=42)
    r_model.fit(X_desc_tr[tr_m], y_train[tr_m])
    oof_ridge_desc[val_m] = r_model.predict(X_desc_tr[val_m])
    test_ridge_desc += r_model.predict(X_desc_te) / 5.0

df_all["ridge_oof_desc"] = np.concatenate([oof_ridge_desc, test_ridge_desc]).astype(np.float32)

# Combined Title + Description TF-IDF sparse matrix & OOF Ridge
X_comb = hstack([X_title, X_desc]).tocsr()
del X_title, X_desc, X_title_tr, X_title_te, X_desc_tr, X_desc_te
gc.collect()

oof_ridge_comb = np.zeros(n_train, dtype=np.float32)
test_ridge_comb = np.zeros(n_test, dtype=np.float32)
X_comb_tr = X_comb[:n_train]
X_comb_te = X_comb[n_train:]

for fold in range(5):
    val_m = train_df["fold"].values == fold
    tr_m = train_df["fold"].values != fold
    r_model = Ridge(alpha=5.0, solver="lsqr", random_state=42)
    r_model.fit(X_comb_tr[tr_m], y_train[tr_m])
    oof_ridge_comb[val_m] = r_model.predict(X_comb_tr[val_m])
    test_ridge_comb += r_model.predict(X_comb_te) / 5.0

df_all["ridge_oof_comb"] = np.concatenate([oof_ridge_comb, test_ridge_comb]).astype(np.float32)

# Expanded Truncated SVD components (25 components)
svd_comb = TruncatedSVD(n_components=25, random_state=42)
svd_comb.fit(X_comb_tr)
tr_comb_svd = svd_comb.transform(X_comb_tr).astype(np.float32)
te_comb_svd = svd_comb.transform(X_comb_te).astype(np.float32)

for i in range(25):
    df_all[f"comb_svd_{i}"] = np.concatenate([tr_comb_svd[:, i], te_comb_svd[:, i]])

del X_comb, X_comb_tr, X_comb_te, svd_comb, tr_comb_svd, te_comb_svd
gc.collect()

print("Encoding categoricals and OOF target encodings...")

cat_cols = [
    "region",
    "city",
    "parent_category_name",
    "category_name",
    "user_type",
    "image_top_1",
    "param_1",
    "param_2",
    "param_3",
    "cat_param1",
    "reg_city",
]

# Frequency & Label encoding
for col in cat_cols:
    df_all[col] = df_all[col].astype(str).fillna("missing")
    freq = df_all[train_mask][col].value_counts().to_dict()
    df_all[f"{col}_freq"] = df_all[col].map(freq).fillna(0).astype(np.int32)

    le = LabelEncoder()
    df_all[f"{col}_encoded"] = le.fit_transform(df_all[col]).astype(np.int32)

# Out-Of-Fold Target Encoding on Training Set (including fine-grained joint hierarchies)
target_enc_cols = [
    "category_name",
    "parent_category_name",
    "user_type",
    "region",
    "image_top_1",
    "param_1",
    "cat_param1",
    "reg_city",
]

df_train_sub = df_all.iloc[:n_train].copy()
df_train_sub["deal_probability"] = y_train
df_train_sub["fold"] = train_df["fold"].values
global_mean = float(y_train.mean())

for col in target_enc_cols:
    target_col = f"{col}_te"
    df_all[target_col] = global_mean

    for fold in range(5):
        val_mask = df_train_sub["fold"] == fold
        tr_mask = df_train_sub["fold"] != fold

        tr_data = df_train_sub[tr_mask]
        group_means = tr_data.groupby(col)["deal_probability"].mean().to_dict()
        group_counts = tr_data.groupby(col)["deal_probability"].count().to_dict()

        smoothed_te = {
            k: (group_counts[k] * group_means[k] + 10 * global_mean)
            / (group_counts[k] + 10)
            for k in group_means
        }

        val_idx_all = df_all.index[:n_train][val_mask]
        df_all.loc[val_idx_all, target_col] = (
            df_all.loc[val_idx_all, col].map(smoothed_te).fillna(global_mean).values
        )

    full_counts = df_train_sub.groupby(col)["deal_probability"].count().to_dict()
    full_means = df_train_sub.groupby(col)["deal_probability"].mean().to_dict()
    full_smoothed = {
        k: (full_counts[k] * full_means[k] + 10 * global_mean) / (full_counts[k] + 10)
        for k in full_means
    }
    df_all.loc[~train_mask, target_col] = (
        df_all.loc[~train_mask, col].map(full_smoothed).fillna(global_mean).values
    )

del df_train_sub
gc.collect()

# Drop raw unneeded string/date columns
drop_cols = [
    "item_id",
    "user_id",
    "title",
    "description",
    "activation_date",
    "image",
] + cat_cols
df_all = df_all.drop(columns=drop_cols)

# Separate train and test feature matrices
folds = train_df["fold"].values
feature_cols = [c for c in df_all.columns if c != "fold"]

X_train_full = df_all.iloc[:n_train][feature_cols].values.astype(np.float32)
X_test = df_all.iloc[n_train:][feature_cols].values.astype(np.float32)

del df_all, train_df, test_df
gc.collect()

X_train_full = np.nan_to_num(X_train_full, nan=0.0, posinf=0.0, neginf=0.0)
X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

num_features = len(feature_cols)
print(f"Features count: {num_features}, Train rows: {n_train}, Test rows: {n_test}")


# ==============================================================================
# MODEL DEFINITIONS & EVALUATION UTILS
# ==============================================================================


def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_pred_clipped = np.clip(y_pred, 0.0, 1.0)
    return float(root_mean_squared_error(y_true, y_pred_clipped))


# ==============================================================================
# CROSS-VALIDATION TRAINING AND INFERENCE (SEED-AVERAGED LIGHTGBM)
# ==============================================================================

oof_preds = np.zeros(n_train, dtype=np.float32)
test_preds = np.zeros(n_test, dtype=np.float32)

seeds = [42, 2024]
n_seeds = len(seeds)

base_lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.04,
    "num_leaves": 255,
    "max_depth": 10,
    "min_child_samples": 40,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.65,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "n_estimators": 2000,
    "n_jobs": -1,
    "verbose": -1,
}

print(f"Starting 5-Fold Seed-Averaged LightGBM training ({n_seeds} seeds)...")

for fold in range(5):
    val_idx = np.where(folds == fold)[0]
    tr_idx = np.where(folds != fold)[0]

    X_tr, y_tr = X_train_full[tr_idx], y_train[tr_idx]
    X_val, y_val = X_train_full[val_idx], y_train[val_idx]

    fold_val_preds = np.zeros(len(val_idx), dtype=np.float32)

    for seed_idx, seed in enumerate(seeds):
        params = base_lgb_params.copy()
        params["random_state"] = seed

        model_lgb = lgb.LGBMRegressor(**params)
        model_lgb.fit(
            X_tr,
            y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
        )

        val_p = model_lgb.predict(X_val)
        test_p = model_lgb.predict(X_test)

        fold_val_preds += val_p / n_seeds
        test_preds += test_p / (5.0 * n_seeds)

        del model_lgb
        gc.collect()

    oof_preds[val_idx] = fold_val_preds
    fold_rmse = calculate_rmse(y_val, fold_val_preds)
    print(f"Fold {fold+1}/5 RMSE: {fold_rmse:.5f}")

    del X_tr, y_tr, X_val, y_val
    gc.collect()

final_score = calculate_rmse(y_train, oof_preds)

sub_df = pd.DataFrame(
    {"id": np.arange(n_test, dtype=np.int64), "deal_probability": np.clip(test_preds, 0.0, 1.0)}
)
sub_df.to_csv("./submission/submission.csv", index=False)

print(f"Final Seed-Averaged Validation Score: {final_score}")