import os
import warnings
import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import xgboost as xgb

warnings.filterwarnings("ignore")

# Directories
os.makedirs("./submission", exist_ok=True)
os.makedirs("./working", exist_ok=True)

# 1. Load Data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "REVENUE15"
train_df = train_df.dropna(subset=[target_col]).reset_index(drop=True)
y_train = train_df[target_col].values.astype(np.float64)
y_train = np.maximum(0, y_train)

train_df = train_df.drop(columns=[target_col])
if target_col in test_df.columns:
    test_df = test_df.drop(columns=[target_col])

n_train = len(train_df)
n_test = len(test_df)

# Combine for uniform feature preprocessing
df = pd.concat([train_df, test_df], axis=0, ignore_index=True)

# 2. Clean Domain Survey Sentinels (-9, -99, -999, -9999, -1)
for col in df.columns:
    if "LAT" in col.upper() or "LONG" in col.upper():
        df[col] = pd.to_numeric(df[col], errors="coerce")

num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
for col in num_cols:
    if "LONG" in col.upper() or "LAT" in col.upper():
        continue
    df[col] = df[col].replace([-9, -99, -999, -9999, -1], np.nan)

# 3. Spatial & Domain Feature Engineering
if "LATITUDE" in df.columns and "LONGITUDE" in df.columns:
    df["LATITUDE"] = pd.to_numeric(df["LATITUDE"], errors="coerce")
    df["LONGITUDE"] = pd.to_numeric(df["LONGITUDE"], errors="coerce")
    us_lat, us_lon = 39.8283, -98.5795
    df["dist_to_us_center"] = np.sqrt(
        (df["LATITUDE"] - us_lat) ** 2 + (df["LONGITUDE"] - us_lon) ** 2
    )
    df["coord_sum"] = df["LATITUDE"] + df["LONGITUDE"]
    df["coord_diff"] = df["LATITUDE"] - df["LONGITUDE"]

name_col = next((c for c in ["LEGALNAME", "NAME", "ALTNAME"] if c in df.columns), None)
if name_col:
    df["name_len"] = df[name_col].astype(str).str.len()
    df["name_word_count"] = df[name_col].astype(str).apply(lambda x: len(x.split()))
    keywords = [
        "university",
        "college",
        "foundation",
        "society",
        "historical",
        "art",
        "children",
        "science",
        "zoo",
        "botanical",
        "park",
        "memorial",
        "national",
        "association",
        "center",
        "library",
    ]
    for kw in keywords:
        df[f"kw_{kw}"] = (
            df[name_col].astype(str).str.lower().str.contains(kw).astype(int)
        )

url_col = next((c for c in ["WEBURL", "URL", "WEBSITE"] if c in df.columns), None)
if url_col:
    df["has_website"] = df[url_col].notna().astype(int) & (
        df[url_col].astype(str).str.len() > 5
    ).astype(int)
    df["url_is_org"] = (
        df[url_col].astype(str).str.lower().str.contains(r"\.org").astype(int)
    )
    df["url_is_edu"] = (
        df[url_col].astype(str).str.lower().str.contains(r"\.edu").astype(int)
    )
    df["url_is_gov"] = (
        df[url_col].astype(str).str.lower().str.contains(r"\.gov").astype(int)
    )

cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

train_df_temp = df.iloc[:n_train]
for c in cat_cols:
    freq = train_df_temp[c].value_counts(normalize=True).to_dict()
    df[f"{c}_freq"] = df[c].map(freq).fillna(0)
    df[f"{c}_code"] = df[c].astype("category").cat.codes

for col in ["INCOME15", "INCOME"]:
    if col in df.columns:
        df[f"{col}_log1p"] = np.log1p(
            np.maximum(0, pd.to_numeric(df[col], errors="coerce").fillna(0))
        )

name_text_col = "LEGALNAME" if "LEGALNAME" in df.columns else name_col
if name_text_col and name_text_col in df.columns:
    legal_text = df[name_text_col].fillna("").astype(str)
    tfidf = TfidfVectorizer(
        max_features=5000, stop_words="english", ngram_range=(1, 2)
    )
    tfidf_mat_train = tfidf.fit_transform(legal_text.iloc[:n_train])
    tfidf_mat_all = tfidf.transform(legal_text)
    svd = TruncatedSVD(n_components=10, random_state=42)
    svd.fit(tfidf_mat_train)
    svd_feats = svd.transform(tfidf_mat_all)
    for i in range(10):
        df[f"svd_legalname_{i}"] = svd_feats[:, i]

group_keys = [c for c in ["STABR", "DISC3", "REGION", "AAMREG"] if c in df.columns]
agg_num_cols = [
    c for c in ["INCOME15", "LATITUDE", "LONGITUDE", "name_len"] if c in df.columns
]

for ac in agg_num_cols:
    df[ac] = pd.to_numeric(df[ac], errors="coerce")

train_df_temp = df.iloc[:n_train]

if "STABR" in df.columns and "DISC3" in df.columns:
    for ac in agg_num_cols:
        mean_map = train_df_temp.groupby(["STABR", "DISC3"])[ac].mean().to_dict()
        median_map = train_df_temp.groupby(["STABR", "DISC3"])[ac].median().to_dict()
        std_map = train_df_temp.groupby(["STABR", "DISC3"])[ac].std().to_dict()

        pairs = list(zip(df["STABR"], df["DISC3"]))
        df[f"STABR_DISC3_{ac}_mean"] = [mean_map.get(p, np.nan) for p in pairs]
        df[f"STABR_DISC3_{ac}_median"] = [median_map.get(p, np.nan) for p in pairs]
        df[f"STABR_DISC3_{ac}_std"] = [std_map.get(p, np.nan) for p in pairs]

if "DISC3" in df.columns:
    for ac in agg_num_cols:
        grouped_d3 = train_df_temp.groupby("DISC3")[ac]
        df[f"DISC3_{ac}_mean"] = df["DISC3"].map(grouped_d3.mean())
        df[f"DISC3_{ac}_median"] = df["DISC3"].map(grouped_d3.median())
        df[f"DISC3_{ac}_std"] = df["DISC3"].map(grouped_d3.std())

for gk in group_keys:
    for ac in agg_num_cols:
        grouped = train_df_temp.groupby(gk)[ac]
        df[f"{gk}_{ac}_mean"] = df[gk].map(grouped.mean())
        df[f"{gk}_{ac}_std"] = df[gk].map(grouped.std())

df_numeric = df.drop(columns=cat_cols)

X_train = df_numeric.iloc[:n_train].copy().reset_index(drop=True)
X_test = df_numeric.iloc[n_train:].copy().reset_index(drop=True)

# 4. Out-of-Fold Training & Model Stacking Architecture
kf = KFold(n_splits=5, shuffle=True, random_state=42)
y_train_log = np.log1p(y_train)

n_models = 7
oof_matrix = np.zeros((n_train, n_models))
test_matrix = np.zeros((n_test, n_models))

te_cols = [c for c in ["STABR", "DISC3", "COUNTY", "CITY"] if c in train_df.columns]

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
    X_tr, y_tr = X_train.iloc[train_idx].copy(), y_train[train_idx]
    y_tr_log = y_train_log[train_idx]

    X_va, y_va = X_train.iloc[val_idx].copy(), y_train[val_idx]
    y_va_log = y_train_log[val_idx]

    X_te = X_test.copy()

    # Smoothed out-of-fold target encoding
    m_smooth = 10.0
    for te_col in te_cols:
        if te_col in train_df.columns:
            tr_series = train_df.iloc[train_idx][te_col]
            global_mean = float(y_tr.mean())

            stats = pd.DataFrame({"cat": tr_series, "target": y_tr}).groupby("cat")["target"].agg(["count", "mean"])
            smoothed = (stats["count"] * stats["mean"] + m_smooth * global_mean) / (stats["count"] + m_smooth)
            cat_means = smoothed.to_dict()

            tr_cat = train_df.iloc[train_idx][te_col]
            va_cat = train_df.iloc[val_idx][te_col]
            te_cat = test_df[te_col]

            X_tr[f"{te_col}_te_mean"] = tr_cat.map(cat_means).fillna(global_mean)
            X_va[f"{te_col}_te_mean"] = va_cat.map(cat_means).fillna(global_mean)
            X_te[f"{te_col}_te_mean"] = te_cat.map(cat_means).fillna(global_mean)

    # Model 0: LightGBM (Log target)
    model_lgb_log = lgb.LGBMRegressor(
        objective="regression",
        metric="rmse",
        boosting_type="gbdt",
        learning_rate=0.03,
        num_leaves=31,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=1,
        random_state=42 + fold,
        verbose=-1,
        n_estimators=1200,
    )
    model_lgb_log.fit(
        X_tr,
        y_tr_log,
        eval_set=[(X_va, y_va_log)],
        callbacks=[
            lgb.early_stopping(50, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    oof_matrix[val_idx, 0] = np.expm1(np.maximum(0, model_lgb_log.predict(X_va)))
    test_matrix[:, 0] += np.expm1(np.maximum(0, model_lgb_log.predict(X_te))) / 5.0

    # Model 1: XGBoost (Log target)
    model_xgb_log = xgb.XGBRegressor(
        n_estimators=1200,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        early_stopping_rounds=50,
        n_jobs=-1,
        eval_metric="rmse",
    )
    model_xgb_log.fit(X_tr, y_tr_log, eval_set=[(X_va, y_va_log)], verbose=False)
    oof_matrix[val_idx, 1] = np.expm1(np.maximum(0, model_xgb_log.predict(X_va)))
    test_matrix[:, 1] += np.expm1(np.maximum(0, model_xgb_log.predict(X_te))) / 5.0

    # Model 2: CatBoost (Log target)
    model_cb_log = cb.CatBoostRegressor(
        iterations=1200,
        learning_rate=0.03,
        depth=6,
        random_seed=42 + fold,
        early_stopping_rounds=50,
        verbose=0,
        loss_function="RMSE",
    )
    model_cb_log.fit(X_tr, y_tr_log, eval_set=(X_va, y_va_log), verbose=False)
    oof_matrix[val_idx, 2] = np.expm1(np.maximum(0, model_cb_log.predict(X_va)))
    test_matrix[:, 2] += np.expm1(np.maximum(0, model_cb_log.predict(X_te))) / 5.0

    # Model 3: LightGBM (Raw target)
    model_lgb_raw = lgb.LGBMRegressor(
        objective="regression",
        metric="rmse",
        boosting_type="gbdt",
        learning_rate=0.03,
        num_leaves=31,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=1,
        random_state=100 + fold,
        verbose=-1,
        n_estimators=1200,
    )
    model_lgb_raw.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[
            lgb.early_stopping(50, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    oof_matrix[val_idx, 3] = np.maximum(0, model_lgb_raw.predict(X_va))
    test_matrix[:, 3] += np.maximum(0, model_lgb_raw.predict(X_te)) / 5.0

    # Model 4: CatBoost (Raw target)
    model_cb_raw = cb.CatBoostRegressor(
        iterations=1200,
        learning_rate=0.03,
        depth=6,
        random_seed=100 + fold,
        early_stopping_rounds=50,
        verbose=0,
        loss_function="RMSE",
    )
    model_cb_raw.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=False)
    oof_matrix[val_idx, 4] = np.maximum(0, model_cb_raw.predict(X_va))
    test_matrix[:, 4] += np.maximum(0, model_cb_raw.predict(X_te)) / 5.0

    # Model 5: XGBoost (Raw target)
    model_xgb_raw = xgb.XGBRegressor(
        n_estimators=1200,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=100 + fold,
        early_stopping_rounds=50,
        n_jobs=-1,
        eval_metric="rmse",
    )
    model_xgb_raw.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    oof_matrix[val_idx, 5] = np.maximum(0, model_xgb_raw.predict(X_va))
    test_matrix[:, 5] += np.maximum(0, model_xgb_raw.predict(X_te)) / 5.0

    # Model 6: LightGBM (Tweedie regression)
    model_lgb_tweedie = lgb.LGBMRegressor(
        objective="tweedie",
        tweedie_variance_power=1.5,
        metric="rmse",
        boosting_type="gbdt",
        learning_rate=0.03,
        num_leaves=31,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=1,
        random_state=200 + fold,
        verbose=-1,
        n_estimators=1200,
    )
    model_lgb_tweedie.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[
            lgb.early_stopping(50, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    oof_matrix[val_idx, 6] = np.maximum(0, model_lgb_tweedie.predict(X_va))
    test_matrix[:, 6] += np.maximum(0, model_lgb_tweedie.predict(X_te)) / 5.0


# 5. Cross-Validated Ridge Stacking Meta-Learner
meta_kf = KFold(n_splits=5, shuffle=True, random_state=42)
final_oof = np.zeros(n_train)
final_test_preds = np.zeros(n_test)

alphas = np.logspace(-3, 5, 100)

for meta_fold, (m_tr_idx, m_va_idx) in enumerate(meta_kf.split(oof_matrix, y_train)):
    X_m_tr, y_m_tr = oof_matrix[m_tr_idx], y_train[m_tr_idx]
    X_m_va, y_m_va = oof_matrix[m_va_idx], y_train[m_va_idx]

    meta_model = RidgeCV(alphas=alphas, cv=5)
    meta_model.fit(X_m_tr, y_m_tr)

    final_oof[m_va_idx] = meta_model.predict(X_m_va)
    final_test_preds += meta_model.predict(test_matrix) / 5.0

final_oof = np.maximum(0, final_oof)
final_test_preds = np.maximum(0, final_test_preds)

# 6. Save Submission File & Report Official Metric
submission = pd.DataFrame(
    {"id": np.arange(len(final_test_preds)), "REVENUE15": final_test_preds}
)
submission.to_csv("./submission/submission.csv", index=False)

final_rmse = np.sqrt(mean_squared_error(y_train, final_oof))
print(f"Final Validation Score: {final_rmse}")