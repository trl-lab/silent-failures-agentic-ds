import numpy as np
import pandas as pd
import warnings
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings("ignore")

# 1. Load Data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "Supplier Contract Amount (USD)"
y_train = train_df[target_col].values
y_log = np.log1p(np.maximum(0, y_train))

# Concatenate for structural feature engineering
df_all = pd.concat(
    [train_df.drop(columns=[target_col], errors="ignore"), test_df], ignore_index=True
)
n_train = len(train_df)

# Identify column types
date_cols = [c for c in df_all.columns if "date" in c.lower() or "year" in c.lower()]
cat_cols = [
    c for c in df_all.columns if df_all[c].dtype == "object" and c not in date_cols
]

# 2. Date Feature Engineering
for col in date_cols:
    parsed = pd.to_datetime(df_all[col], errors="coerce")
    df_all[f"{col}_year"] = parsed.dt.year
    df_all[f"{col}_month"] = parsed.dt.month
    df_all[f"{col}_day"] = parsed.dt.day
    df_all[f"{col}_dayofweek"] = parsed.dt.dayofweek
    df_all[f"{col}_quarter"] = parsed.dt.quarter
    df_all[f"{col}_timestamp"] = parsed.astype("int64") // 10**9

# 3. Categorical & Text Feature Categorization
text_cols = []
pure_cat_cols = []
for c in cat_cols:
    df_all[c] = df_all[c].fillna("Missing").astype(str)
    avg_words = df_all[c].str.split().str.len().mean()
    if avg_words > 3 and df_all[c].nunique() > 50:
        text_cols.append(c)
    else:
        pure_cat_cols.append(c)

# Text embeddings via TF-IDF + SVD
for c in text_cols:
    tfidf = TfidfVectorizer(max_features=5000, stop_words="english", ngram_range=(1, 2))
    tfidf_mat = tfidf.fit_transform(df_all[c])
    svd = TruncatedSVD(n_components=12, random_state=42)
    svd_mat = svd.fit_transform(tfidf_mat)
    for i in range(svd_mat.shape[1]):
        df_all[f"svd_{c}_{i}"] = svd_mat[:, i]
    df_all[f"{c}_char_len"] = df_all[c].str.len()
    df_all[f"{c}_word_count"] = df_all[c].str.split().str.len()

# Global Categorical Frequency & Category Code Encodings
for c in pure_cat_cols:
    freq = df_all[c].value_counts().to_dict()
    df_all[f"{c}_freq"] = df_all[c].map(freq)
    df_all[f"{c}_cat"] = df_all[c].astype("category").cat.codes

# Base numerical feature matrix preparation
base_features_df = df_all.drop(columns=cat_cols, errors="ignore").select_dtypes(
    include=[np.number]
)
base_features_df = base_features_df.fillna(-999)

X_train_base = base_features_df.iloc[:n_train].copy()
X_test_base = base_features_df.iloc[n_train:].copy()

# Retain pure categorical values for Out-of-Fold Target Encoding
cat_train_df = df_all[pure_cat_cols].iloc[:n_train].copy()
cat_test_df = df_all[pure_cat_cols].iloc[n_train:].copy()

# 4. Cross-Validation with Ensemble (XGBoost + LightGBM)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(n_train)
test_preds = np.zeros(len(test_df))

smoothing_weight = 10.0

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_base)):
    X_tr = X_train_base.iloc[train_idx].copy()
    X_va = X_train_base.iloc[val_idx].copy()
    X_te = X_test_base.copy()

    y_tr = y_log[train_idx]
    y_va = y_log[val_idx]
    global_mean = np.mean(y_tr)

    # Compute smoothed Target Encodings per categorical column on train fold
    for c in pure_cat_cols:
        tr_cats = cat_train_df.iloc[train_idx][c]
        va_cats = cat_train_df.iloc[val_idx][c]
        te_cats = cat_test_df[c]

        stats = pd.DataFrame({"cat": tr_cats, "target": y_tr})
        grp = stats.groupby("cat")["target"].agg(["count", "mean", "std"]).reset_index()

        grp["te_mean"] = (
            grp["count"] * grp["mean"] + smoothing_weight * global_mean
        ) / (grp["count"] + smoothing_weight)
        grp["te_std"] = grp["std"].fillna(0)

        mean_map = grp.set_index("cat")["te_mean"].to_dict()
        std_map = grp.set_index("cat")["te_std"].to_dict()

        X_tr[f"{c}_te_mean"] = tr_cats.map(mean_map).fillna(global_mean)
        X_tr[f"{c}_te_std"] = tr_cats.map(std_map).fillna(0)

        X_va[f"{c}_te_mean"] = va_cats.map(mean_map).fillna(global_mean)
        X_va[f"{c}_te_std"] = va_cats.map(std_map).fillna(0)

        X_te[f"{c}_te_mean"] = te_cats.map(mean_map).fillna(global_mean)
        X_te[f"{c}_te_std"] = te_cats.map(std_map).fillna(0)

    # Train XGBoost Regressor
    model_xgb = xgb.XGBRegressor(
        n_estimators=1500,
        learning_rate=0.03,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        tree_method="hist",
        early_stopping_rounds=50,
    )
    model_xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

    xgb_val = model_xgb.predict(X_va)
    xgb_test = model_xgb.predict(X_te)

    # Train LightGBM Regressor
    model_lgb = lgb.LGBMRegressor(
        n_estimators=1500,
        learning_rate=0.03,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        n_jobs=-1,
        verbose=-1,
    )
    model_lgb.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )

    lgb_val = model_lgb.predict(X_va)
    lgb_test = model_lgb.predict(X_te)

    # Blend predictions (50/50 ensemble)
    fold_val_pred = 0.5 * xgb_val + 0.5 * lgb_val
    fold_test_pred = 0.5 * xgb_test + 0.5 * lgb_test

    oof_preds[val_idx] = fold_val_pred
    test_preds += fold_test_pred / kf.n_splits

# 5. Evaluation Metrics
cv_rmse_log = np.sqrt(mean_squared_error(y_log, oof_preds))
oof_actual_preds = np.maximum(0, np.expm1(oof_preds))
cv_rmse_raw = np.sqrt(mean_squared_error(y_train, oof_actual_preds))

print(f"5-Fold CV RMSE (log scale): {cv_rmse_log:.4f}")

# 6. Save Submission Predictions
sub_preds = np.maximum(0, np.expm1(test_preds))
submission = pd.DataFrame(
    {"id": np.arange(len(test_df)), "Supplier Contract Amount (USD)": sub_preds}
)
submission.to_csv("./working/submission.csv", index=False)
