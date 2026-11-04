import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostRegressor, Pool
import lightgbm as lgb
import xgboost as xgb

# 1. Load Data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "NAMPLT_CAP"
y = train_df[target_col].values.astype(np.float64)
X_train = train_df.drop(columns=[target_col]).copy()
X_test = test_df.copy()

n_train = len(train_df)
n_test = len(test_df)
combined = pd.concat([X_train, X_test], axis=0, ignore_index=True)

# 2. Date/Time and Spatial Preprocessing
for col in list(combined.columns):
    if combined[col].dtype == "object" or isinstance(
        combined[col].dtype, pd.DatetimeTZDtype
    ):
        try:
            parsed_dates = pd.to_datetime(combined[col], errors="raise")
            combined[f"{col}_year"] = parsed_dates.dt.year.astype("float32")
            combined[f"{col}_month"] = parsed_dates.dt.month.astype("float32")
            combined[f"{col}_day"] = parsed_dates.dt.day.astype("float32")
            combined = combined.drop(columns=[col])
        except (ValueError, TypeError):
            pass

# Identify categorical and numerical columns
raw_cat_cols = []
raw_num_cols = []
for col in combined.columns:
    if combined[col].dtype == "object" or combined[col].dtype.name == "category":
        combined[col] = combined[col].fillna("Missing").astype(str)
        raw_cat_cols.append(col)
    else:
        combined[col] = combined[col].astype("float32")
        raw_num_cols.append(col)

# Spatial coordinate features if latitude and longitude exist
lat_cols = [c for c in raw_num_cols if "lat" in c.lower()]
lon_cols = [c for c in raw_num_cols if "lon" in c.lower()]
if lat_cols and lon_cols:
    lat_c, lon_c = lat_cols[0], lon_cols[0]
    lat_rad = np.radians(combined[lat_c].fillna(combined[lat_c].median()))
    lon_rad = np.radians(combined[lon_c].fillna(combined[lon_c].median()))
    combined["coord_x"] = (np.cos(lat_rad) * np.cos(lon_rad)).astype("float32")
    combined["coord_y"] = (np.cos(lat_rad) * np.sin(lon_rad)).astype("float32")
    combined["coord_z"] = np.sin(lat_rad).astype("float32")
    combined["coord_dist_center"] = np.sqrt(
        (combined[lat_c] - combined[lat_c].mean()) ** 2
        + (combined[lon_c] - combined[lon_c].mean()) ** 2
    ).astype("float32")

# Frequency encoding for categoricals
for col in raw_cat_cols:
    freq = combined[col].value_counts()
    combined[f"{col}_freq"] = combined[col].map(freq).astype("float32")

# Group aggregations of key numericals by top categoricals
top_cats = [c for c in raw_cat_cols if 1 < combined[c].nunique() < 500][:8]
top_nums = raw_num_cols[:8]
for cat in top_cats:
    for num in top_nums:
        grp_mean = combined.groupby(cat)[num].transform("mean").astype("float32")
        grp_std = (
            combined.groupby(cat)[num].transform("std").fillna(0).astype("float32")
        )
        combined[f"{cat}_{num}_mean"] = grp_mean
        combined[f"{cat}_{num}_std"] = grp_std

# Separate train and test feature sets
X_train_proc = combined.iloc[:n_train].copy().reset_index(drop=True)
X_test_proc = combined.iloc[n_train:].copy().reset_index(drop=True)

# 3. 5-Fold Cross Validation Setup & Out-of-Fold Target Encoding
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

smoothing = 10.0
global_mean = y.mean()
for col in raw_cat_cols:
    oof_te = np.zeros(n_train, dtype=np.float32)

    for tr_idx, va_idx in kf.split(X_train_proc):
        tr_df = pd.DataFrame(
            {"cat": X_train_proc.iloc[tr_idx][col], "target": y[tr_idx]}
        )
        stats = tr_df.groupby("cat")["target"].agg(["count", "mean"])
        smooth_enc = (stats["count"] * stats["mean"] + smoothing * global_mean) / (
            stats["count"] + smoothing
        )
        oof_te[va_idx] = (
            X_train_proc.iloc[va_idx][col].map(smooth_enc).fillna(global_mean)
        )

    full_tr_df = pd.DataFrame({"cat": X_train_proc[col], "target": y})
    stats_full = full_tr_df.groupby("cat")["target"].agg(["count", "mean"])
    smooth_enc_full = (
        stats_full["count"] * stats_full["mean"] + smoothing * global_mean
    ) / (stats_full["count"] + smoothing)
    test_te = (
        X_test_proc[col].map(smooth_enc_full).fillna(global_mean).astype("float32")
    )

    X_train_proc[f"{col}_te"] = oof_te
    X_test_proc[f"{col}_te"] = test_te

# Preprocessing for LightGBM, CatBoost, and XGBoost
feature_cols = list(X_train_proc.columns)
cat_indices = [feature_cols.index(c) for c in raw_cat_cols if c in feature_cols]

# Categorical types for LightGBM
X_train_lgb = X_train_proc.copy()
X_test_lgb = X_test_proc.copy()
for c in raw_cat_cols:
    X_train_lgb[c] = X_train_lgb[c].astype("category")
    X_test_lgb[c] = X_test_lgb[c].astype("category")

# Label encoded features for XGBoost
X_train_xgb = X_train_proc.copy()
X_test_xgb = X_test_proc.copy()
for c in raw_cat_cols:
    le = LabelEncoder()
    full_cat_vals = (
        pd.concat([X_train_proc[c], X_test_proc[c]], axis=0).astype(str).values
    )
    le.fit(full_cat_vals)
    X_train_xgb[c] = le.transform(X_train_proc[c].astype(str)).astype("int32")
    X_test_xgb[c] = le.transform(X_test_proc[c].astype(str)).astype("int32")

# Containers for predictions
oof_cb = np.zeros(n_train, dtype=np.float64)
oof_lgb = np.zeros(n_train, dtype=np.float64)
oof_xgb = np.zeros(n_train, dtype=np.float64)

test_cb = np.zeros(n_test, dtype=np.float64)
test_lgb = np.zeros(n_test, dtype=np.float64)
test_xgb = np.zeros(n_test, dtype=np.float64)

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_proc, y)):
    y_tr, y_va = y[train_idx], y[val_idx]

    # 1) CatBoost
    X_tr_cb, X_va_cb = X_train_proc.iloc[train_idx], X_train_proc.iloc[val_idx]
    train_pool = Pool(X_tr_cb, y_tr, cat_features=cat_indices)
    val_pool = Pool(X_va_cb, y_va, cat_features=cat_indices)
    test_pool = Pool(X_test_proc, cat_features=cat_indices)

    cb_model = CatBoostRegressor(
        iterations=1600,
        learning_rate=0.035,
        depth=6,
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=42 + fold,
        verbose=False,
    )
    cb_model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=50, verbose=False)
    oof_cb[val_idx] = cb_model.predict(val_pool)
    test_cb += cb_model.predict(test_pool) / n_splits

    # 2) LightGBM
    X_tr_lgb, X_va_lgb = X_train_lgb.iloc[train_idx], X_train_lgb.iloc[val_idx]
    lgb_model = lgb.LGBMRegressor(
        n_estimators=1600,
        learning_rate=0.035,
        num_leaves=31,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42 + fold,
        n_jobs=-1,
        verbose=-1,
    )
    lgb_model.fit(
        X_tr_lgb,
        y_tr,
        eval_set=[(X_va_lgb, y_va)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )
    oof_lgb[val_idx] = lgb_model.predict(X_va_lgb)
    test_lgb += lgb_model.predict(X_test_lgb) / n_splits

    # 3) XGBoost
    X_tr_xgb, X_va_xgb = X_train_xgb.iloc[train_idx], X_train_xgb.iloc[val_idx]
    xgb_model = xgb.XGBRegressor(
        n_estimators=1600,
        learning_rate=0.035,
        max_depth=6,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42 + fold,
        n_jobs=-1,
        tree_method="hist",
        early_stopping_rounds=50,
    )
    xgb_model.fit(
        X_tr_xgb,
        y_tr,
        eval_set=[(X_va_xgb, y_va)],
        verbose=False,
    )
    oof_xgb[val_idx] = xgb_model.predict(X_va_xgb)
    test_xgb += xgb_model.predict(X_test_xgb) / n_splits

cb_rmse = root_mean_squared_error(y, oof_cb)
lgb_rmse = root_mean_squared_error(y, oof_lgb)
xgb_rmse = root_mean_squared_error(y, oof_xgb)
print(f"CatBoost OOF RMSE: {cb_rmse:.5f}")
print(f"LightGBM OOF RMSE: {lgb_rmse:.5f}")
print(f"XGBoost  OOF RMSE: {xgb_rmse:.5f}")


# 4. Find optimal blending weights across the 3 diverse models
def blend_obj(weights):
    w1, w2, w3 = weights
    pred = w1 * oof_cb + w2 * oof_lgb + w3 * oof_xgb
    return root_mean_squared_error(y, pred)


res = minimize(
    blend_obj,
    [0.33, 0.34, 0.33],
    bounds=[(0, 1), (0, 1), (0, 1)],
    constraints={"type": "eq", "fun": lambda w: sum(w) - 1},
)
w_cb, w_lgb, w_xgb = res.x
oof_blend = w_cb * oof_cb + w_lgb * oof_lgb + w_xgb * oof_xgb
blend_rmse = root_mean_squared_error(y, oof_blend)
print(
    f"Optimal Weights: CatBoost={w_cb:.3f}, LightGBM={w_lgb:.3f}, XGBoost={w_xgb:.3f}"
)
print(f"Tri-Model Ensemble Out-of-Fold RMSE: {blend_rmse:.5f}")

# 5. Predict on Test and Save Submission
final_test_preds = w_cb * test_cb + w_lgb * test_lgb + w_xgb * test_xgb
final_test_preds = np.clip(final_test_preds, a_min=0.0, a_max=None)

os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame(
    {"id": np.arange(len(final_test_preds)), "NAMPLT_CAP": final_test_preds}
)
submission.to_csv("./working/submission.csv", index=False)
print("Saved submission to ./working/submission.csv successfully.")
