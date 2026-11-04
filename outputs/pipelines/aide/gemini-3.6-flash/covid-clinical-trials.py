import os
import warnings
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
import lightgbm as lgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")

# 1. Load Data
train_path = "./input/train.parquet"
test_path = "./input/test.parquet"

train = pd.read_parquet(train_path)
test = pd.read_parquet(test_path)

target_col = "days_to_complete"
y_train = train[target_col].values
X_train_raw = train.drop(columns=[target_col])
X_test_raw = test.copy()

n_train = len(train)

# Concatenate for consistent feature engineering
df_all = pd.concat([X_train_raw, X_test_raw], axis=0, ignore_index=True)

# 2. Date Feature Engineering
date_cols = [
    col for col in df_all.columns if "date" in col.lower() or "posted" in col.lower()
]
processed_df = pd.DataFrame(index=df_all.index)

parsed_dates = {}
for col in date_cols:
    try:
        dt_series = pd.to_datetime(df_all[col], errors="coerce")
        parsed_dates[col] = dt_series
        processed_df[f"{col}_year"] = dt_series.dt.year
        processed_df[f"{col}_month"] = dt_series.dt.month
        processed_df[f"{col}_day"] = dt_series.dt.day
        processed_df[f"{col}_dayofweek"] = dt_series.dt.dayofweek
        ts = np.where(dt_series.notnull(), dt_series.astype("int64") // 10**9, np.nan)
        processed_df[f"{col}_timestamp"] = ts
    except Exception:
        pass

# Pairwise date differences in days
date_names = list(parsed_dates.keys())
for i in range(len(date_names)):
    for j in range(i + 1, len(date_names)):
        col1, col2 = date_names[i], date_names[j]
        diff = (parsed_dates[col1] - parsed_dates[col2]).dt.total_seconds() / (
            24 * 3600
        )
        processed_df[f"date_diff_{i}_{j}"] = diff

# 3. Text and Categorical Feature Engineering
text_cols = []
cat_cols = []
num_cols = []

for col in df_all.columns:
    if col in date_cols:
        continue
    if df_all[col].dtype == "object" or df_all[col].dtype == "category":
        sample_str = df_all[col].dropna().astype(str)
        if sample_str.empty:
            continue
        avg_len = sample_str.str.len().mean()
        if avg_len > 30:
            text_cols.append(col)
        else:
            cat_cols.append(col)
    elif pd.api.types.is_numeric_dtype(df_all[col]):
        num_cols.append(col)

# Numerical columns
for col in num_cols:
    processed_df[col] = df_all[col]

# Frequency & label encoding for categoricals
for col in cat_cols:
    freq = df_all[col].value_counts(normalize=True).to_dict()
    processed_df[f"{col}_freq"] = df_all[col].map(freq).fillna(0)
    processed_df[f"{col}_code"] = pd.factorize(df_all[col])[0]

# Text metadata & high-dimensional TF-IDF
text_combined = pd.Series("", index=df_all.index)
for col in text_cols:
    str_series = df_all[col].fillna("").astype(str)
    processed_df[f"{col}_char_len"] = str_series.str.len()
    processed_df[f"{col}_word_count"] = str_series.str.split().str.len()
    text_combined = text_combined + " " + str_series

tfidf = TfidfVectorizer(
    max_features=25000, stop_words="english", ngram_range=(1, 2), sublinear_tf=True
)
text_tfidf = tfidf.fit_transform(text_combined)

svd = TruncatedSVD(n_components=30, random_state=42)
svd_feats = svd.fit_transform(text_tfidf)
for k in range(30):
    processed_df[f"svd_{k}"] = svd_feats[:, k]

# 4. Out-of-Fold Ridge Meta-Feature from TF-IDF
kf_text = KFold(n_splits=5, shuffle=True, random_state=42)
tfidf_tr = text_tfidf[:n_train]
tfidf_te = text_tfidf[n_train:]

ridge_oof = np.zeros(n_train)
ridge_test = np.zeros(len(test))

for tr_idx, va_idx in kf_text.split(tfidf_tr, y_train):
    ridge_model = Ridge(alpha=1.0)
    ridge_model.fit(tfidf_tr[tr_idx], y_train[tr_idx])
    ridge_oof[va_idx] = ridge_model.predict(tfidf_tr[va_idx])
    ridge_test += ridge_model.predict(tfidf_te) / 5

processed_df["tfidf_ridge_pred"] = np.concatenate([ridge_oof, ridge_test])

# 5. Prepare Train / Test Datasets
X_train = processed_df.iloc[:n_train].copy().fillna(-999)
X_test = processed_df.iloc[n_train:].copy().fillna(-999)

# 6. 5-Fold Cross-Validation with Model Ensemble (CatBoost + XGBoost + LightGBM)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(n_train)
test_preds = np.zeros(len(test))

w_cb, w_xgb, w_lgb = 0.60, 0.25, 0.15

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
    X_tr, y_tr = X_train.iloc[train_idx], y_train[train_idx]
    X_va, y_va = X_train.iloc[val_idx], y_train[val_idx]

    # Model 1: CatBoost
    model_cb = CatBoostRegressor(
        iterations=1500,
        learning_rate=0.04,
        depth=6,
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=42 + fold,
        verbose=0,
    )
    model_cb.fit(
        X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=50, verbose=False
    )
    val_pred_cb = model_cb.predict(X_va)
    test_pred_cb = model_cb.predict(X_test)

    # Model 2: XGBoost
    model_xgb = XGBRegressor(
        n_estimators=1500,
        learning_rate=0.04,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        n_jobs=-1,
        early_stopping_rounds=50,
        eval_metric="rmse",
    )
    model_xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    val_pred_xgb = model_xgb.predict(X_va)
    test_pred_xgb = model_xgb.predict(X_test)

    # Model 3: LightGBM
    model_lgb = lgb.LGBMRegressor(
        n_estimators=1500,
        learning_rate=0.04,
        max_depth=6,
        num_leaves=31,
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
    val_pred_lgb = model_lgb.predict(X_va)
    test_pred_lgb = model_lgb.predict(X_test)

    # Weighted Ensemble Blending
    val_pred_blend = w_cb * val_pred_cb + w_xgb * val_pred_xgb + w_lgb * val_pred_lgb
    test_pred_blend = (
        w_cb * test_pred_cb + w_xgb * test_pred_xgb + w_lgb * test_pred_lgb
    )

    oof_preds[val_idx] = np.clip(val_pred_blend, a_min=0, a_max=None)
    test_preds += np.clip(test_pred_blend, a_min=0, a_max=None) / 5

oof_rmse = np.sqrt(mean_squared_error(y_train, oof_preds))
print(f"5-Fold CV RMSE: {oof_rmse:.4f}")

# 7. Save Predictions to Submission File
os.makedirs("./working", exist_ok=True)
sub = pd.DataFrame({"id": np.arange(len(test_preds)), "days_to_complete": test_preds})
sub.to_csv("./working/submission.csv", index=False)
