import os
import re
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from scipy.optimize import minimize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error

# 1. Load Data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "Price"
y_train_raw = train_df[target_col].values
y_train_log = np.log1p(np.maximum(0, y_train_raw))

# Combine datasets for consistent feature processing
n_train = len(train_df)
df = pd.concat(
    [train_df.drop(columns=[target_col], errors="ignore"), test_df], ignore_index=True
)

# Ensure ID column exists for final output
if "id" in df.columns:
    test_ids = test_df["id"].values
else:
    test_ids = np.arange(len(test_df))

# 2. Extract Domain Features & Domain Interactions via Regex
text_cols = [
    c for c in df.columns if df[c].dtype == "object" or df[c].dtype == "string"
]
combined_text = (
    df[text_cols].astype(str).fillna("").apply(lambda row: " ".join(row), axis=1)
)

vintages = combined_text.str.extract(r"\b(19\d\d|20\d\d)\b")[0].astype(float)
abv = combined_text.str.extract(r"(\d+(?:\.\d+)?)\s*%\s*(?:abv)?", flags=re.IGNORECASE)[
    0
].astype(float)

# Volume extraction (standardized to ml)
vol_ml = pd.Series(np.nan, index=df.index)
vol_ml = vol_ml.fillna(
    combined_text.str.extract(r"(\d+(?:\.\d+)?)\s*ml", flags=re.IGNORECASE)[0].astype(
        float
    )
)
vol_ml = vol_ml.fillna(
    combined_text.str.extract(r"(\d+(?:\.\d+)?)\s*cl", flags=re.IGNORECASE)[0].astype(
        float
    )
    * 10
)
vol_ml = vol_ml.fillna(
    combined_text.str.extract(r"(\d+(?:\.\d+)?)\s*l\b", flags=re.IGNORECASE)[0].astype(
        float
    )
    * 1000
)

pack_count = (
    combined_text.str.extract(
        r"\b(\d+)\s*(?:x|pack|case|bottle)s?\b", flags=re.IGNORECASE
    )[0]
    .astype(float)
    .fillna(1.0)
)

# Domain interaction features
total_vol_ml = vol_ml * pack_count
vintage_age = 2024.0 - vintages
pure_alcohol_ml = total_vol_ml * (abv.fillna(0.0) / 100.0)

domain_features = pd.DataFrame(
    {
        "extracted_vintage": vintages,
        "extracted_abv": abv,
        "extracted_vol_ml": vol_ml,
        "extracted_pack_count": pack_count,
        "extracted_total_vol_ml": total_vol_ml,
        "extracted_vintage_age": vintage_age,
        "extracted_pure_alcohol_ml": pure_alcohol_ml,
    },
    index=df.index,
)

# 3. Handle Numerical and Categorical Features
num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != "id"]
cat_cols = [c for c in text_cols if df[c].nunique() < 500]

X_num = df[num_cols].copy()
for col in X_num.columns:
    X_num[col] = X_num[col].fillna(X_num[col].median())

# Concatenate numerical and domain extractions
X_numeric = pd.concat([X_num, domain_features], axis=1)
for col in X_numeric.columns:
    X_numeric[col] = X_numeric[col].fillna(X_numeric[col].median())

# Frequency Encoding for Categoricals
X_cat_freq = pd.DataFrame(index=df.index)
for col in cat_cols:
    freq = df[col].astype(str).value_counts()
    X_cat_freq[f"{col}_freq"] = df[col].astype(str).map(freq).fillna(0)

# 4. Text Vectorization: Word & Char TF-IDF, Ridge Stacking, and SVD
word_vec = TfidfVectorizer(
    max_features=25000,
    ngram_range=(1, 2),
    min_df=2,
    stop_words="english",
    sublinear_tf=True,
)
tfidf_word = word_vec.fit_transform(combined_text)

char_vec = TfidfVectorizer(
    max_features=25000, analyzer="char", ngram_range=(3, 5), min_df=2, sublinear_tf=True
)
tfidf_char = char_vec.fit_transform(combined_text)

svd = TruncatedSVD(n_components=30, random_state=42)
svd_feats = pd.DataFrame(
    svd.fit_transform(tfidf_word),
    columns=[f"svd_{i}" for i in range(30)],
    index=df.index,
)

# Ridge Meta-Features via 5-Fold Cross Validation on Word and Char TF-IDF
kf = KFold(n_splits=5, shuffle=True, random_state=42)

oof_ridge_word = np.zeros(n_train)
test_ridge_word_folds = np.zeros((len(test_df), 5))

oof_ridge_char = np.zeros(n_train)
test_ridge_char_folds = np.zeros((len(test_df), 5))

tfidf_word_train, tfidf_word_test = tfidf_word[:n_train], tfidf_word[n_train:]
tfidf_char_train, tfidf_char_test = tfidf_char[:n_train], tfidf_char[n_train:]

for fold, (trn_idx, val_idx) in enumerate(kf.split(tfidf_word_train)):
    # Word Ridge
    ridge_w = Ridge(alpha=1.0)
    ridge_w.fit(tfidf_word_train[trn_idx], y_train_log[trn_idx])
    oof_ridge_word[val_idx] = ridge_w.predict(tfidf_word_train[val_idx])
    test_ridge_word_folds[:, fold] = ridge_w.predict(tfidf_word_test)

    # Char Ridge
    ridge_c = Ridge(alpha=1.0)
    ridge_c.fit(tfidf_char_train[trn_idx], y_train_log[trn_idx])
    oof_ridge_char[val_idx] = ridge_c.predict(tfidf_char_train[val_idx])
    test_ridge_char_folds[:, fold] = ridge_c.predict(tfidf_char_test)

ridge_word_feat = pd.Series(
    np.concatenate([oof_ridge_word, test_ridge_word_folds.mean(axis=1)]),
    index=df.index,
    name="ridge_word_pred",
)
ridge_char_feat = pd.Series(
    np.concatenate([oof_ridge_char, test_ridge_char_folds.mean(axis=1)]),
    index=df.index,
    name="ridge_char_pred",
)

# Assemble full tabular feature dataset
full_features = pd.concat(
    [X_numeric, X_cat_freq, svd_feats, ridge_word_feat, ridge_char_feat], axis=1
)

X_train_full = full_features.iloc[:n_train].copy().reset_index(drop=True)
X_test_full = full_features.iloc[n_train:].copy().reset_index(drop=True)

# Target Encoding for Categorical Features (Out-of-Fold)
for col in cat_cols:
    target_enc_oof = np.zeros(n_train)
    target_enc_test_folds = np.zeros((len(test_df), 5))
    global_mean = y_train_log.mean()

    col_train = df.iloc[:n_train][col].astype(str).values
    col_test = df.iloc[n_train:][col].astype(str).values

    for fold, (trn_idx, val_idx) in enumerate(kf.split(X_train_full)):
        trn_df = pd.DataFrame(
            {"cat": col_train[trn_idx], "target": y_train_log[trn_idx]}
        )
        stats = trn_df.groupby("cat")["target"].agg(["count", "mean"])
        smooth = (stats["count"] * stats["mean"] + 10 * global_mean) / (
            stats["count"] + 10
        )

        target_enc_oof[val_idx] = (
            pd.Series(col_train[val_idx]).map(smooth).fillna(global_mean).values
        )
        target_enc_test_folds[:, fold] = (
            pd.Series(col_test).map(smooth).fillna(global_mean).values
        )

    X_train_full[f"{col}_te"] = target_enc_oof
    X_test_full[f"{col}_te"] = target_enc_test_folds.mean(axis=1)

# Clean column names for tree models
clean_cols = [
    str(col).replace("[", "_").replace("]", "_").replace("<", "_").replace(":", "_")
    for col in X_train_full.columns
]
X_train_full.columns = clean_cols
X_test_full.columns = clean_cols

# 5. Training XGBoost and LightGBM Models
oof_xgb = np.zeros(n_train)
test_xgb = np.zeros(len(test_df))

oof_lgb = np.zeros(n_train)
test_lgb = np.zeros(len(test_df))

xgb_params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "learning_rate": 0.03,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "tree_method": "hist",
    "random_state": 42,
}

lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.03,
    "num_leaves": 31,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "random_state": 42,
    "verbose": -1,
}

for fold, (trn_idx, val_idx) in enumerate(kf.split(X_train_full)):
    X_trn, y_trn = X_train_full.iloc[trn_idx], y_train_log[trn_idx]
    X_val, y_val = X_train_full.iloc[val_idx], y_train_log[val_idx]

    # XGBoost
    model_xgb = xgb.XGBRegressor(
        **xgb_params, n_estimators=1500, early_stopping_rounds=50
    )
    model_xgb.fit(X_trn, y_trn, eval_set=[(X_val, y_val)], verbose=False)
    oof_xgb[val_idx] = model_xgb.predict(X_val)
    test_xgb += model_xgb.predict(X_test_full) / kf.n_splits

    # LightGBM
    model_lgb = lgb.LGBMRegressor(**lgb_params, n_estimators=1500)
    model_lgb.fit(
        X_trn,
        y_trn,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )
    oof_lgb[val_idx] = model_lgb.predict(X_val)
    test_lgb += model_lgb.predict(X_test_full) / kf.n_splits

test_ridge_w = test_ridge_word_folds.mean(axis=1)
test_ridge_c = test_ridge_char_folds.mean(axis=1)


# 6. SLSQP Constrained Optimization for Blend Weights on Raw RMSE
def raw_rmse_objective(weights):
    weights = np.array(weights)
    pred_log = (
        weights[0] * oof_xgb
        + weights[1] * oof_lgb
        + weights[2] * oof_ridge_word
        + weights[3] * oof_ridge_char
    )
    pred_raw = np.maximum(0, np.expm1(pred_log))
    return root_mean_squared_error(y_train_raw, pred_raw)


init_weights = [0.4, 0.4, 0.1, 0.1]
bounds = [(0, 1) for _ in range(4)]
constraints = {"type": "eq", "fun": lambda w: 1.0 - np.sum(w)}

res = minimize(
    raw_rmse_objective,
    init_weights,
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
)
best_w = res.x

# Calculate Final Out-of-Fold Predictions
oof_preds_log = (
    best_w[0] * oof_xgb
    + best_w[1] * oof_lgb
    + best_w[2] * oof_ridge_word
    + best_w[3] * oof_ridge_char
)
oof_preds_raw = np.maximum(0, np.expm1(oof_preds_log))

oof_rmse = root_mean_squared_error(y_train_raw, oof_preds_raw)
print(f"Validation Metric (RMSE): {oof_rmse:.4f}")

# Generate Final Test Predictions
test_preds_log = (
    best_w[0] * test_xgb
    + best_w[1] * test_lgb
    + best_w[2] * test_ridge_w
    + best_w[3] * test_ridge_c
)
test_preds_raw = np.expm1(test_preds_log)
min_train_price = max(0, y_train_raw.min())
test_preds_raw = np.maximum(min_train_price, test_preds_raw)

os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame({"id": test_ids, "Price": test_preds_raw})
submission.to_csv("./working/submission.csv", index=False)
