import os
import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "NAMPLT_CAP"
y = train_df[target_col].values
train_len = len(train_df)

# Combine for uniform feature engineering
df = pd.concat(
    [train_df.drop(columns=[target_col], errors="ignore"), test_df],
    ignore_index=True,
)

id_col = "id" if "id" in df.columns else None

# Spatial features using latitude and longitude
lat_cols = [c for c in df.columns if "lat" in c.lower()]
lon_cols = [c for c in df.columns if "lon" in c.lower() or "lng" in c.lower()]

if lat_cols and lon_cols:
    lat_col, lon_col = lat_cols[0], lon_cols[0]
    coords = df[[lat_col, lon_col]].fillna(df[[lat_col, lon_col]].median())
    kmeans = KMeans(n_clusters=12, random_state=42, n_init=10)
    df["geo_cluster"] = kmeans.fit_predict(coords)

cat_cols = []
for col in df.columns:
    if col == id_col:
        continue
    if df[col].dtype == "object" or df[col].dtype.name == "category":
        df[col] = df[col].fillna("missing").astype(str).astype("category")
        cat_cols.append(col)
    elif pd.api.types.is_datetime64_any_dtype(df[col]):
        df[f"{col}_year"] = df[col].dt.year
        df[f"{col}_month"] = df[col].dt.month
        df.drop(columns=[col], inplace=True)

# Frequency encoding for categorical columns
for col in cat_cols:
    freq = df[col].value_counts().to_dict()
    df[f"{col}_freq"] = df[col].map(freq).astype(float)

feature_cols = [c for c in df.columns if c != id_col]

X_train = df.iloc[:train_len][feature_cols].copy()
X_test = df.iloc[train_len:][feature_cols].copy()

# Log-transform target for skewed positive distribution
y_log = np.log1p(np.maximum(0, y))

kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds_cb = np.zeros(train_len)
oof_preds_lgb = np.zeros(train_len)
test_preds_cb = np.zeros(len(test_df))
test_preds_lgb = np.zeros(len(test_df))

lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.04,
    "max_depth": 6,
    "num_leaves": 31,
    "verbose": -1,
    "n_jobs": -1,
}

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_tr, y_tr = X_train.iloc[train_idx], y_log[train_idx]
    X_val, y_val = X_train.iloc[val_idx], y_log[val_idx]

    # --- CatBoost Training ---
    tr_pool = Pool(X_tr, y_tr, cat_features=cat_cols)
    val_pool = Pool(X_val, y_val, cat_features=cat_cols)
    test_pool = Pool(X_test, cat_features=cat_cols)

    cb_model = CatBoostRegressor(
        iterations=1200,
        learning_rate=0.04,
        depth=6,
        loss_function="RMSE",
        random_seed=42 + fold,
        verbose=0,
    )
    cb_model.fit(tr_pool, eval_set=val_pool, early_stopping_rounds=100)

    oof_preds_cb[val_idx] = cb_model.predict(val_pool)
    test_preds_cb += cb_model.predict(test_pool) / 5.0

    # --- LightGBM Training ---
    dtrain = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat_cols)
    dval = lgb.Dataset(
        X_val, label=y_val, reference=dtrain, categorical_feature=cat_cols
    )

    params = lgb_params.copy()
    params["random_state"] = 42 + fold

    lgb_model = lgb.train(
        params,
        dtrain,
        num_boost_round=1200,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )

    oof_preds_lgb[val_idx] = lgb_model.predict(X_val)
    test_preds_lgb += lgb_model.predict(X_test) / 5.0

# Equal-weight ensemble blending on log scale
oof_preds_log = 0.5 * oof_preds_cb + 0.5 * oof_preds_lgb
test_preds_log = 0.5 * test_preds_cb + 0.5 * test_preds_lgb

oof_preds = np.maximum(0, np.expm1(oof_preds_log))
test_preds = np.maximum(0, np.expm1(test_preds_log))

cv_rmse = np.sqrt(mean_squared_error(y, oof_preds))
print(f"Validation Metric: {cv_rmse:.4f}")

os.makedirs("./working", exist_ok=True)
sub_id = test_df["id"].values if "id" in test_df.columns else np.arange(len(test_df))
submission = pd.DataFrame({"id": sub_id, "NAMPLT_CAP": test_preds})
submission.to_csv("./working/submission.csv", index=False)
