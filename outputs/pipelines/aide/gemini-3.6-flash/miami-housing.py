import os
import warnings
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostRegressor
from scipy.optimize import minimize
from sklearn.cluster import KMeans
from sklearn.metrics import root_mean_squared_error
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")

# 1. Load Data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

train_len = len(train_df)
y_train_orig = train_df["SALE_PRC"].values
y_train_log = np.log1p(y_train_orig)

# Combine for uniform feature engineering
df = pd.concat([train_df.drop(columns=["SALE_PRC"]), test_df], axis=0).reset_index(
    drop=True
)

# 2. Feature Engineering
# Parcel frequency
parcel_counts = df["PARCELNO"].value_counts().to_dict()
df["parcel_freq"] = df["PARCELNO"].map(parcel_counts)

# Spatial rotations and coordinates
lat = df["LATITUDE"].values
lon = df["LONGITUDE"].values

df["lat_lon_sum"] = lat + lon
df["lat_lon_diff"] = lat - lon

for angle in [15, 30, 45, 60, 75]:
    rad = np.radians(angle)
    df[f"rot_{angle}_x"] = lat * np.cos(rad) - lon * np.sin(rad)
    df[f"rot_{angle}_y"] = lat * np.sin(rad) + lon * np.cos(rad)

# Key Landmarks distance (Lat, Lon)
landmarks = {
    "miami_cbd": (25.774, -80.193),
    "miami_beach": (25.782, -80.130),
    "brickell": (25.765, -80.190),
    "key_biscayne": (25.690, -80.160),
    "airport": (25.795, -80.287),
    "coral_gables": (25.721, -80.268),
}

for name, (l_lat, l_lon) in landmarks.items():
    df[f"dist_to_{name}"] = np.sqrt((lat - l_lat) ** 2 + (lon - l_lon) ** 2)

# Multi-scale KMeans spatial clustering
coords = np.column_stack([lat, lon])
for n_clusters in [15, 40, 80]:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df[f"cluster_{n_clusters}"] = kmeans.fit_predict(coords)

# Property structural ratios & interactions
df["living_to_land_ratio"] = df["TOT_LVG_AREA"] / (df["LND_SQFOOT"] + 1.0)
df["land_per_living"] = df["LND_SQFOOT"] / (df["TOT_LVG_AREA"] + 1.0)
df["quality_age_ratio"] = df["structure_quality"] / (df["age"] + 1.0)
df["quality_age_prod"] = df["structure_quality"] * df["age"]
df["spec_feat_per_living"] = df["SPEC_FEAT_VAL"] / (df["TOT_LVG_AREA"] + 1.0)
df["spec_feat_per_land"] = df["SPEC_FEAT_VAL"] / (df["LND_SQFOOT"] + 1.0)
df["has_spec_feat"] = (df["SPEC_FEAT_VAL"] > 0).astype(int)
df["log_spec_feat"] = np.log1p(df["SPEC_FEAT_VAL"])
df["min_water_dist"] = np.minimum(df["OCEAN_DIST"], df["WATER_DIST"])

# Spatial Neighborhood Relative Structural Features (Comparative Market Analysis)
nn_struct = NearestNeighbors(n_neighbors=16, metric="euclidean", n_jobs=-1)
nn_struct.fit(coords)
_, struct_indices = nn_struct.kneighbors(coords)
nbr_indices = struct_indices[:, 1:]  # Exclude self

struct_cols = ["TOT_LVG_AREA", "LND_SQFOOT", "age", "structure_quality"]
for col in struct_cols:
    vals = df[col].values
    nbr_vals = vals[nbr_indices]
    nbr_mean = np.mean(nbr_vals, axis=1)
    df[f"{col}_nbr_mean_15"] = nbr_mean
    df[f"{col}_rel_ratio_15"] = vals / (nbr_mean + 1e-5)
    df[f"{col}_rel_diff_15"] = vals - nbr_mean

# Log distance transforms
dist_cols = [
    "RAIL_DIST",
    "OCEAN_DIST",
    "WATER_DIST",
    "CNTR_DIST",
    "SUBCNTR_DI",
    "HWY_DIST",
    "min_water_dist",
]
for col in dist_cols:
    df[f"log_{col}"] = np.log1p(df[col])

# Month cyclical encoding
df["sin_month"] = np.sin(2 * np.pi * df["month_sold"] / 12.0)
df["cos_month"] = np.cos(2 * np.pi * df["month_sold"] / 12.0)

# Drop raw PARCELNO
df = df.drop(columns=["PARCELNO"])

# Separate back to train and test
X_train_full = df.iloc[:train_len].copy().reset_index(drop=True)
X_test_full = df.iloc[train_len:].copy().reset_index(drop=True)

# 3. K-Fold Setup & Leak-free Spatial KNN Target Statistics
kf = KFold(n_splits=5, shuffle=True, random_state=42)


def compute_knn_features(train_coords, train_target, query_coords, k_list=[5, 15, 30]):
    nn = NearestNeighbors(n_neighbors=max(k_list) + 1, metric="euclidean", n_jobs=-1)
    nn.fit(train_coords)
    distances, indices = nn.kneighbors(query_coords)

    res = {}
    is_self = np.array_equal(train_coords, query_coords)

    for k in k_list:
        if is_self:
            neighbor_idxs = indices[:, 1 : k + 1]
        else:
            neighbor_idxs = indices[:, :k]

        neighbor_targets = train_target[neighbor_idxs]
        res[f"knn_{k}_target_mean"] = np.mean(neighbor_targets, axis=1)
        res[f"knn_{k}_target_std"] = np.std(neighbor_targets, axis=1)
        res[f"knn_{k}_target_median"] = np.median(neighbor_targets, axis=1)
    return pd.DataFrame(res)


# Prepare out-of-fold arrays
oof_lgb = np.zeros(train_len)
oof_xgb = np.zeros(train_len)
oof_cat = np.zeros(train_len)

test_preds_lgb = np.zeros(len(test_df))
test_preds_xgb = np.zeros(len(test_df))
test_preds_cat = np.zeros(len(test_df))

train_spatial_coords = X_train_full[["LATITUDE", "LONGITUDE"]].values
test_spatial_coords = X_test_full[["LATITUDE", "LONGITUDE"]].values

# 4. Model Training Loop
for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_full)):
    X_tr, y_tr = X_train_full.iloc[train_idx].copy(), y_train_log[train_idx]
    X_va, y_va = X_train_full.iloc[val_idx].copy(), y_train_log[val_idx]
    X_te = X_test_full.copy()

    # Spatial KNN features
    tr_coords = train_spatial_coords[train_idx]
    va_coords = train_spatial_coords[val_idx]

    knn_tr = compute_knn_features(tr_coords, y_tr, tr_coords)
    knn_va = compute_knn_features(tr_coords, y_tr, va_coords)
    knn_te = compute_knn_features(tr_coords, y_tr, test_spatial_coords)

    for col in knn_tr.columns:
        X_tr[col] = knn_tr[col].values
        X_va[col] = knn_va[col].values
        X_te[col] = knn_te[col].values

    # LightGBM
    lgb_model = lgb.LGBMRegressor(
        n_estimators=3000,
        learning_rate=0.03,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        n_jobs=-1,
        verbose=-1,
    )
    lgb_model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    oof_lgb[val_idx] = np.expm1(lgb_model.predict(X_va))
    test_preds_lgb += np.expm1(lgb_model.predict(X_te)) / kf.n_splits

    # XGBoost
    xgb_model = xgb.XGBRegressor(
        n_estimators=3000,
        learning_rate=0.03,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        n_jobs=-1,
        early_stopping_rounds=50,
        tree_method="hist",
    )
    xgb_model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    oof_xgb[val_idx] = np.expm1(xgb_model.predict(X_va))
    test_preds_xgb += np.expm1(xgb_model.predict(X_te)) / kf.n_splits

    # CatBoost
    cat_model = CatBoostRegressor(
        iterations=3000,
        learning_rate=0.03,
        depth=7,
        random_seed=42 + fold,
        verbose=0,
        early_stopping_rounds=50,
    )
    cat_model.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=False)
    oof_cat[val_idx] = np.expm1(cat_model.predict(X_va))
    test_preds_cat += np.expm1(cat_model.predict(X_te)) / kf.n_splits


# 5. Optimize Blending Weights
def loss_func(weights):
    w1, w2, w3 = weights
    pred = w1 * oof_lgb + w2 * oof_xgb + w3 * oof_cat
    return root_mean_squared_error(y_train_orig, pred)


res = minimize(
    loss_func,
    x0=[0.333, 0.333, 0.334],
    bounds=[(0, 1), (0, 1), (0, 1)],
    constraints={"type": "eq", "fun": lambda w: sum(w) - 1.0},
)
w1, w2, w3 = res.x

oof_ensemble = w1 * oof_lgb + w2 * oof_xgb + w3 * oof_cat
final_rmse = root_mean_squared_error(y_train_orig, oof_ensemble)

print(f"Validation RMSE: {final_rmse:.4f}")

# 6. Save Submission
final_test_preds = w1 * test_preds_lgb + w2 * test_preds_xgb + w3 * test_preds_cat
final_test_preds = np.clip(final_test_preds, a_min=0, a_max=None)

os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame(
    {"id": np.arange(len(final_test_preds)), "SALE_PRC": final_test_preds}
)
submission.to_csv("./working/submission.csv", index=False)
