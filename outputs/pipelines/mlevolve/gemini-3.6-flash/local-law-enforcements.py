import os
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

warnings.filterwarnings("ignore")

os.makedirs("./submission", exist_ok=True)
os.makedirs("./working", exist_ok=True)

# 1. Load Data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "number_of_officers"
if target_col not in train_df.columns:
    raise ValueError(f"Target column {target_col} not found in train data.")

y_train = train_df[target_col].values.astype(np.float32)
train_df = train_df.drop(columns=[target_col])
test_ids = (
    test_df["id"].values
    if "id" in test_df.columns
    else np.arange(len(test_df), dtype=np.int64)
)

# 2. Dynamic Feature Engineering
col_names_lower = {col: col.lower() for col in train_df.columns}
lat_col = next(
    (c for c, l in col_names_lower.items() if l in ["latitude", "lat", "y"]),
    None,
)
lon_col = next(
    (c for c, l in col_names_lower.items() if l in ["longitude", "lon", "x"]),
    None,
)

cat_cols, text_cols, num_cols = [], [], []
for c in train_df.columns:
    if c in [lat_col, lon_col, "id", "ID"]:
        continue
    dtype = train_df[c].dtype
    if dtype == "object" or str(dtype) == "category":
        if train_df[c].nunique(dropna=True) <= 100:
            cat_cols.append(c)
        else:
            text_cols.append(c)
    elif np.issubdtype(dtype, np.number):
        num_cols.append(c)

# Text Processing (TF-IDF + TruncatedSVD)
text_train_features, text_test_features = [], []
if text_cols:
    train_text = train_df[text_cols].fillna("").astype(str).agg(" ".join, axis=1)
    test_text = test_df[text_cols].fillna("").astype(str).agg(" ".join, axis=1)

    tfidf = TfidfVectorizer(
        max_features=5000,
        stop_words="english",
        sublinear_tf=True,
        ngram_range=(1, 2),
    )
    tfidf_train = tfidf.fit_transform(train_text)
    tfidf_test = tfidf.transform(test_text)

    n_components = min(32, tfidf_train.shape[1] - 1) if tfidf_train.shape[1] > 1 else 1
    if n_components > 0:
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        svd_train = svd.fit_transform(tfidf_train)
        svd_test = svd.transform(tfidf_test)
        svd_cols = [f"text_svd_{i}" for i in range(n_components)]
        text_train_features.append(
            pd.DataFrame(svd_train, columns=svd_cols, index=train_df.index)
        )
        text_test_features.append(
            pd.DataFrame(svd_test, columns=svd_cols, index=test_df.index)
        )

# Spatial Processing
spatial_train_list, spatial_test_list = [], []
if lat_col and lon_col:
    lat_tr = train_df[lat_col].fillna(train_df[lat_col].median()).values
    lon_tr = train_df[lon_col].fillna(train_df[lon_col].median()).values
    lat_te = test_df[lat_col].fillna(train_df[lat_col].median()).values
    lon_te = test_df[lon_col].fillna(train_df[lon_col].median()).values

    c_lat, c_lon = np.mean(lat_tr), np.mean(lon_tr)
    coords_tr = np.column_stack([lat_tr, lon_tr])
    coords_te = np.column_stack([lat_te, lon_te])

    kmeans = KMeans(n_clusters=8, random_state=42, n_init=10)
    clusters_tr = kmeans.fit_predict(coords_tr)
    clusters_te = kmeans.predict(coords_te)

    spatial_train_list.append(
        pd.DataFrame(
            {
                "lat": lat_tr,
                "lon": lon_tr,
                "dist_center": np.sqrt((lat_tr - c_lat) ** 2 + (lon_tr - c_lon) ** 2),
                "cluster": clusters_tr,
            },
            index=train_df.index,
        )
    )
    spatial_test_list.append(
        pd.DataFrame(
            {
                "lat": lat_te,
                "lon": lon_te,
                "dist_center": np.sqrt((lat_te - c_lat) ** 2 + (lon_te - c_lon) ** 2),
                "cluster": clusters_te,
            },
            index=test_df.index,
        )
    )

# Categorical Frequency & OOF Target Encoding
cat_train_list, cat_test_list = [], []
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for col in cat_cols:
    freq_map = train_df[col].value_counts(normalize=True).to_dict()
    freq_tr = train_df[col].map(freq_map).fillna(0).values
    freq_te = test_df[col].map(freq_map).fillna(0).values

    oof_target = np.zeros(len(train_df), dtype=np.float32)
    global_mean = np.mean(y_train)

    for tr_idx, val_idx in kf.split(train_df):
        tr_c, tr_y = train_df[col].iloc[tr_idx], y_train[tr_idx]
        val_c = train_df[col].iloc[val_idx]
        means = pd.Series(tr_y).groupby(tr_c).mean()
        oof_target[val_idx] = val_c.map(means).fillna(global_mean).values

    test_means = pd.Series(y_train).groupby(train_df[col]).mean()
    test_target = test_df[col].map(test_means).fillna(global_mean).values

    cat_train_list.append(
        pd.DataFrame(
            {f"{col}_freq": freq_tr, f"{col}_te": oof_target},
            index=train_df.index,
        )
    )
    cat_test_list.append(
        pd.DataFrame(
            {f"{col}_freq": freq_te, f"{col}_te": test_target},
            index=test_df.index,
        )
    )

# Numeric Features
num_train_list, num_test_list = [], []
for col in num_cols:
    med = train_df[col].median()
    tr_val = train_df[col].fillna(med).values
    te_val = test_df[col].fillna(med).values

    tr_log = np.log1p(tr_val) if (tr_val >= 0).all() else tr_val
    te_log = np.log1p(te_val) if (tr_val >= 0).all() else te_val

    num_train_list.append(
        pd.DataFrame(
            {
                col: tr_val,
                f"{col}_log": tr_log,
                f"{col}_null": train_df[col].isnull().astype(int),
            },
            index=train_df.index,
        )
    )
    num_test_list.append(
        pd.DataFrame(
            {
                col: te_val,
                f"{col}_log": te_log,
                f"{col}_null": test_df[col].isnull().astype(int),
            },
            index=test_df.index,
        )
    )

X_tr_full = pd.concat(
    text_train_features + spatial_train_list + cat_train_list + num_train_list,
    axis=1,
).fillna(0)
X_te_full = pd.concat(
    text_test_features + spatial_test_list + cat_test_list + num_test_list,
    axis=1,
).fillna(0)

# Standard Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_tr_full)
X_test = scaler.transform(X_te_full)

# 3. Model Training & Cross-Validation
n_train = len(y_train)
n_test = len(X_test)
y_train_log = np.log1p(y_train)

model_names = [
    "lgb_log_rmse",
    "lgb_raw_huber",
    "xgb_log_rmse",
    "xgb_raw_rmse",
    "cat_log_rmse",
    "ridge_log",
]
n_models = len(model_names)

oof_preds = np.zeros((n_train, n_models), dtype=np.float32)
test_preds = np.zeros((n_test, n_models), dtype=np.float32)

folds = list(kf.split(X_train, y_train))

for fold_idx, (tr_idx, val_idx) in enumerate(folds):
    X_tr, y_tr_raw, y_tr_log = (
        X_train[tr_idx],
        y_train[tr_idx],
        y_train_log[tr_idx],
    )
    X_va, y_va_raw, y_va_log = (
        X_train[val_idx],
        y_train[val_idx],
        y_train_log[val_idx],
    )

    # Model 0: LightGBM Log1p RMSE
    lgb_log = lgb.LGBMRegressor(
        n_estimators=1500,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold_idx,
        verbosity=-1,
        n_jobs=-1,
    )
    lgb_log.fit(
        X_tr,
        y_tr_log,
        eval_set=[(X_va, y_va_log)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    oof_preds[val_idx, 0] = np.expm1(lgb_log.predict(X_va))
    test_preds[:, 0] += np.expm1(lgb_log.predict(X_test)) / len(folds)

    # Model 1: LightGBM Raw Huber
    lgb_huber = lgb.LGBMRegressor(
        objective="huber",
        alpha=10.0,
        n_estimators=1200,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=100 + fold_idx,
        verbosity=-1,
        n_jobs=-1,
    )
    lgb_huber.fit(
        X_tr,
        y_tr_raw,
        eval_set=[(X_va, y_va_raw)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    oof_preds[val_idx, 1] = lgb_huber.predict(X_va)
    test_preds[:, 1] += lgb_huber.predict(X_test) / len(folds)

    # Model 2: XGBoost Log1p RMSE
    xgb_log = xgb.XGBRegressor(
        n_estimators=1500,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold_idx,
        n_jobs=-1,
        early_stopping_rounds=50,
        eval_metric="rmse",
    )
    xgb_log.fit(X_tr, y_tr_log, eval_set=[(X_va, y_va_log)], verbose=False)
    oof_preds[val_idx, 2] = np.expm1(xgb_log.predict(X_va))
    test_preds[:, 2] += np.expm1(xgb_log.predict(X_test)) / len(folds)

    # Model 3: XGBoost Raw RMSE
    xgb_raw = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=200 + fold_idx,
        n_jobs=-1,
        early_stopping_rounds=50,
        eval_metric="rmse",
    )
    xgb_raw.fit(X_tr, y_tr_raw, eval_set=[(X_va, y_va_raw)], verbose=False)
    oof_preds[val_idx, 3] = xgb_raw.predict(X_va)
    test_preds[:, 3] += xgb_raw.predict(X_test) / len(folds)

    # Model 4: CatBoost Log1p RMSE
    cb_log = CatBoostRegressor(
        iterations=1500,
        learning_rate=0.03,
        depth=6,
        random_seed=42 + fold_idx,
        verbose=0,
        early_stopping_rounds=50,
    )
    cb_log.fit(X_tr, y_tr_log, eval_set=(X_va, y_va_log), verbose=False)
    oof_preds[val_idx, 4] = np.expm1(cb_log.predict(X_va))
    test_preds[:, 4] += np.expm1(cb_log.predict(X_test)) / len(folds)

    # Model 5: Ridge Regression on Log Scale
    ridge = Ridge(alpha=100.0)
    ridge.fit(X_tr, y_tr_log)
    oof_preds[val_idx, 5] = np.expm1(ridge.predict(X_va))
    test_preds[:, 5] += np.expm1(ridge.predict(X_test)) / len(folds)

oof_preds = np.clip(oof_preds, 0, None)
test_preds = np.clip(test_preds, 0, None)


# 4. Ensembling via SLSQP Optimization
def rmse_loss(weights):
    weights = weights / np.sum(weights)
    blend_oof = np.dot(oof_preds, weights)
    return np.sqrt(mean_squared_error(y_train, blend_oof))


init_weights = np.ones(n_models) / n_models
bounds = [(0, 1) for _ in range(n_models)]
constraints = {"type": "eq", "fun": lambda w: 1 - np.sum(w)}

opt_res = minimize(
    rmse_loss,
    init_weights,
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
)
best_weights = opt_res.x / np.sum(opt_res.x)

final_oof_preds = np.clip(np.dot(oof_preds, best_weights), 0, None)
final_test_preds = np.clip(np.dot(test_preds, best_weights), 0, None)

final_score = float(np.sqrt(mean_squared_error(y_train, final_oof_preds)))

# 5. Save Test Predictions
submission_df = pd.DataFrame({"id": test_ids, "number_of_officers": final_test_preds})
submission_path = "./submission/submission.csv"
submission_df.to_csv(submission_path, index=False)

print(f"Final Validation Score: {final_score}")
