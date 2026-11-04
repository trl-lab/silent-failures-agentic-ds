import math
import os
from typing import Any, Dict, List, Optional, Tuple

from catboost import CatBoostRegressor
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.cluster import KMeans
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb

# Set random seed for full reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# =========================================================================
# 1. Load Data
# =========================================================================
print("Loading train and test parquet files...")
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

train_df["PARCELNO"] = train_df["PARCELNO"].astype(str)
test_df["PARCELNO"] = test_df["PARCELNO"].astype(str)

target_col = "SALE_PRC"
train_target = train_df[target_col].values.astype(np.float64)
train_log_target = np.log1p(train_target).astype(np.float32)


# =========================================================================
# 2. Feature Engineering
# =========================================================================
def create_base_features(df: pd.DataFrame) -> pd.DataFrame:
    df_feat = df.copy()

    # --- Physical & Structural Dimensions ---
    df_feat["living_to_land_ratio"] = df_feat["TOT_LVG_AREA"] / (
        df_feat["LND_SQFOOT"] + 1.0
    )
    df_feat["unbuilt_land_area"] = np.maximum(
        0.0, df_feat["LND_SQFOOT"] - df_feat["TOT_LVG_AREA"]
    )
    df_feat["spec_feat_per_sqft"] = df_feat["SPEC_FEAT_VAL"] / (
        df_feat["TOT_LVG_AREA"] + 1.0
    )
    df_feat["has_spec_feat"] = (df_feat["SPEC_FEAT_VAL"] > 0).astype(np.float32)

    # Log transformations of heavily skewed area/value features
    df_feat["log_lnd_sqfoot"] = np.log1p(df_feat["LND_SQFOOT"])
    df_feat["log_tot_lvg_area"] = np.log1p(df_feat["TOT_LVG_AREA"])
    df_feat["log_spec_feat_val"] = np.log1p(df_feat["SPEC_FEAT_VAL"])

    # --- Structure Quality & Age Interactions ---
    df_feat["quality_to_age_ratio"] = df_feat["structure_quality"] / (
        df_feat["age"] + 1.0
    )
    df_feat["quality_x_age"] = df_feat["structure_quality"] * (df_feat["age"] + 1.0)
    df_feat["is_new_construction"] = (df_feat["age"] <= 1).astype(np.float32)
    df_feat["is_historic"] = (df_feat["age"] >= 50).astype(np.float32)
    df_feat["effective_age_sq"] = (df_feat["age"] ** 2).astype(np.float32)

    # --- Amenity & Environmental Noise Distance Features ---
    df_feat["log_rail_dist"] = np.log1p(df_feat["RAIL_DIST"])
    df_feat["log_ocean_dist"] = np.log1p(df_feat["OCEAN_DIST"])
    df_feat["log_water_dist"] = np.log1p(df_feat["WATER_DIST"])
    df_feat["log_cntr_dist"] = np.log1p(df_feat["CNTR_DIST"])
    df_feat["log_subcntr_dist"] = np.log1p(df_feat["SUBCNTR_DI"])
    df_feat["log_hwy_dist"] = np.log1p(df_feat["HWY_DIST"])

    # Proximity indices (higher value = closer to amenity / noise source)
    df_feat["rail_noise_idx"] = 1.0 / (df_feat["RAIL_DIST"] + 100.0)
    df_feat["hwy_noise_idx"] = 1.0 / (df_feat["HWY_DIST"] + 100.0)
    df_feat["combined_noise_idx"] = (
        df_feat["rail_noise_idx"]
        + df_feat["hwy_noise_idx"]
        + (df_feat["avno60plus"] * 0.005)
    )

    df_feat["ocean_prox_idx"] = 1.0 / (df_feat["OCEAN_DIST"] + 500.0)
    df_feat["water_prox_idx"] = 1.0 / (df_feat["WATER_DIST"] + 500.0)
    df_feat["cntr_prox_idx"] = 1.0 / (df_feat["CNTR_DIST"] + 1000.0)
    df_feat["subcntr_prox_idx"] = 1.0 / (df_feat["SUBCNTR_DI"] + 1000.0)

    # Relative distance ratios
    df_feat["ocean_to_water_dist_ratio"] = df_feat["OCEAN_DIST"] / (
        df_feat["WATER_DIST"] + 1.0
    )
    df_feat["cntr_to_subcntr_dist_ratio"] = df_feat["CNTR_DIST"] / (
        df_feat["SUBCNTR_DI"] + 1.0
    )

    # --- Spatial Coordinate Transformations ---
    lat_center = 25.7617
    lon_center = -80.1918
    df_feat["lat_centered"] = df_feat["LATITUDE"] - lat_center
    df_feat["lon_centered"] = df_feat["LONGITUDE"] - lon_center
    df_feat["euclidean_dist_cbd"] = np.sqrt(
        df_feat["lat_centered"] ** 2 + df_feat["lon_centered"] ** 2
    )

    # --- High-Order Physical Interactions ---
    df_feat["living_x_quality"] = df_feat["TOT_LVG_AREA"] * df_feat["structure_quality"]
    df_feat["land_x_quality"] = df_feat["LND_SQFOOT"] * df_feat["structure_quality"]
    df_feat["living_per_age"] = df_feat["TOT_LVG_AREA"] / (df_feat["age"] + 1.0)
    df_feat["land_per_age"] = df_feat["LND_SQFOOT"] / (df_feat["age"] + 1.0)
    df_feat["living_per_quality"] = df_feat["TOT_LVG_AREA"] / (df_feat["structure_quality"] + 0.1)
    df_feat["spec_feat_x_quality"] = df_feat["SPEC_FEAT_VAL"] * df_feat["structure_quality"]

    # --- Miami Hyper-Local Landmark Distances & Bearings ---
    landmarks = {
        "brickell_cbd": (25.7617, -80.1918),
        "south_beach": (25.7825, -80.1340),
        "coral_gables": (25.7215, -80.2684),
        "bal_harbour": (25.8926, -80.1245),
        "key_biscayne": (25.6938, -80.1628),
        "mia_airport": (25.7959, -80.2870),
    }
    for name, (l_lat, l_lon) in landmarks.items():
        d_lat = df_feat["LATITUDE"] - l_lat
        d_lon = df_feat["LONGITUDE"] - l_lon
        df_feat[f"dist_to_{name}"] = np.sqrt(d_lat**2 + d_lon**2)
        df_feat[f"manhattan_to_{name}"] = np.abs(d_lat) + np.abs(d_lon)
        df_feat[f"bearing_to_{name}"] = np.arctan2(d_lat, d_lon)

    # Coordinate rotations
    angles_deg = [15, 30, 45, 60, 75]
    for angle in angles_deg:
        rad = np.radians(angle)
        cos_rad, sin_rad = np.cos(rad), np.sin(rad)
        df_feat[f"rot_coord_{angle}_x"] = (
            df_feat["LONGITUDE"] * cos_rad - df_feat["LATITUDE"] * sin_rad
        )
        df_feat[f"rot_coord_{angle}_y"] = (
            df_feat["LONGITUDE"] * sin_rad + df_feat["LATITUDE"] * cos_rad
        )

    # --- Seasonality / Temporal Features ---
    df_feat["month_sold_sin"] = np.sin(2 * np.pi * df_feat["month_sold"] / 12.0)
    df_feat["month_sold_cos"] = np.cos(2 * np.pi * df_feat["month_sold"] / 12.0)
    df_feat["quarter_sold"] = ((df_feat["month_sold"] - 1) // 3 + 1).astype(np.float32)

    return df_feat


train_feat = create_base_features(train_df)
test_feat = create_base_features(test_df)

# Leak-Free Spatial Multi-Scale Clustering & Density (Fit exclusively on Train)
coords_train = train_feat[["LATITUDE", "LONGITUDE"]].values
coords_test = test_feat[["LATITUDE", "LONGITUDE"]].values

cluster_scales = [10, 25, 50, 100]
for k in cluster_scales:
    kmeans = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
    kmeans.fit(coords_train)

    train_feat[f"spatial_cluster_{k}"] = kmeans.predict(coords_train).astype(np.int32)
    test_feat[f"spatial_cluster_{k}"] = kmeans.predict(coords_test).astype(np.int32)

    train_feat[f"dist_to_centroid_{k}"] = kmeans.transform(coords_train).min(axis=1)
    test_feat[f"dist_to_centroid_{k}"] = kmeans.transform(coords_test).min(axis=1)

knn_scales = [5, 15, 30]
for k in knn_scales:
    knn = NearestNeighbors(n_neighbors=k, algorithm="ball_tree")
    knn.fit(coords_train)

    train_knn_dists, _ = knn.kneighbors(coords_train)
    test_knn_dists, _ = knn.kneighbors(coords_test)

    train_feat[f"knn_mean_dist_{k}"] = train_knn_dists.mean(axis=1)
    test_feat[f"knn_mean_dist_{k}"] = test_knn_dists.mean(axis=1)

# GroupKFold Splitting on PARCELNO
n_splits = 5
gkf = GroupKFold(n_splits=n_splits)
train_feat["fold"] = -1

groups = train_feat["PARCELNO"].values
for fold_idx, (train_idx, val_idx) in enumerate(
    gkf.split(train_feat, train_log_target, groups=groups)
):
    train_feat.loc[val_idx, "fold"] = fold_idx

cat_cardinalities = [10, 25, 50, 100]
cat_cols = [f"spatial_cluster_{k}" for k in cluster_scales]


def compute_spatial_knn_features(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    k_scales: List[int] = [3, 7, 15, 30],
    kernel_bandwidths: List[float] = [0.008, 0.020, 0.045],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Computes leak-free spatial k-NN target encodings, continuous Gaussian kernel smoothing, and comparative features."""
    df_tr = df_train.copy()
    df_va = df_val.copy()
    df_te = df_test.copy()

    coords_tr = df_tr[["LATITUDE", "LONGITUDE"]].values
    coords_va = df_va[["LATITUDE", "LONGITUDE"]].values
    coords_te = df_te[["LATITUDE", "LONGITUDE"]].values

    log_price_tr = np.log1p(df_tr["SALE_PRC"].values.astype(np.float64))
    log_ppsf_tr = np.log1p(
        df_tr["SALE_PRC"].values.astype(np.float64)
        / (df_tr["TOT_LVG_AREA"].values.astype(np.float64) + 1.0)
    )

    comp_cols = ["TOT_LVG_AREA", "LND_SQFOOT", "structure_quality", "age"]

    # 1. Multi-scale k-NN Encodings
    for k in k_scales:
        knn_self = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
        knn_self.fit(coords_tr)
        dists_tr, indices_tr = knn_self.kneighbors(coords_tr)

        dists_tr = dists_tr[:, 1:]
        indices_tr = indices_tr[:, 1:]
        weights_tr = 1.0 / (dists_tr + 1e-4)
        norm_weights_tr = weights_tr / np.sum(weights_tr, axis=1, keepdims=True)

        knn_other = NearestNeighbors(n_neighbors=k, metric="euclidean")
        knn_other.fit(coords_tr)

        dists_va, indices_va = knn_other.kneighbors(coords_va)
        weights_va = 1.0 / (dists_va + 1e-4)
        norm_weights_va = weights_va / np.sum(weights_va, axis=1, keepdims=True)

        dists_te, indices_te = knn_other.kneighbors(coords_te)
        weights_te = 1.0 / (dists_te + 1e-4)
        norm_weights_te = weights_te / np.sum(weights_te, axis=1, keepdims=True)

        # Target-based statistics
        lp_tr_neigh = log_price_tr[indices_tr]
        lppsf_tr_neigh = log_ppsf_tr[indices_tr]
        df_tr[f"knn_{k}_mean_log_price"] = np.sum(norm_weights_tr * lp_tr_neigh, axis=1)
        df_tr[f"knn_{k}_median_log_price"] = np.median(lp_tr_neigh, axis=1)
        df_tr[f"knn_{k}_mean_log_ppsf"] = np.sum(norm_weights_tr * lppsf_tr_neigh, axis=1)
        df_tr[f"knn_{k}_median_log_ppsf"] = np.median(lppsf_tr_neigh, axis=1)

        lp_va_neigh = log_price_tr[indices_va]
        lppsf_va_neigh = log_ppsf_tr[indices_va]
        df_va[f"knn_{k}_mean_log_price"] = np.sum(norm_weights_va * lp_va_neigh, axis=1)
        df_va[f"knn_{k}_median_log_price"] = np.median(lp_va_neigh, axis=1)
        df_va[f"knn_{k}_mean_log_ppsf"] = np.sum(norm_weights_va * lppsf_va_neigh, axis=1)
        df_va[f"knn_{k}_median_log_ppsf"] = np.median(lppsf_va_neigh, axis=1)

        lp_te_neigh = log_price_tr[indices_te]
        lppsf_te_neigh = log_ppsf_tr[indices_te]
        df_te[f"knn_{k}_mean_log_price"] = np.sum(norm_weights_te * lp_te_neigh, axis=1)
        df_te[f"knn_{k}_median_log_price"] = np.median(lp_te_neigh, axis=1)
        df_te[f"knn_{k}_mean_log_ppsf"] = np.sum(norm_weights_te * lppsf_te_neigh, axis=1)
        df_te[f"knn_{k}_median_log_ppsf"] = np.median(lppsf_te_neigh, axis=1)

        # Comparative structural features relative to neighborhood
        for col in comp_cols:
            col_vals_tr = df_tr[col].values.astype(np.float64)

            c_tr_neigh = col_vals_tr[indices_tr]
            c_va_neigh = col_vals_tr[indices_va]
            c_te_neigh = col_vals_tr[indices_te]

            mean_c_tr = np.sum(norm_weights_tr * c_tr_neigh, axis=1)
            mean_c_va = np.sum(norm_weights_va * c_va_neigh, axis=1)
            mean_c_te = np.sum(norm_weights_te * c_te_neigh, axis=1)

            df_tr[f"diff_to_knn_{k}_{col}"] = df_tr[col].values - mean_c_tr
            df_tr[f"ratio_to_knn_{k}_{col}"] = (df_tr[col].values + 1.0) / (mean_c_tr + 1.0)

            df_va[f"diff_to_knn_{k}_{col}"] = df_va[col].values - mean_c_va
            df_va[f"ratio_to_knn_{k}_{col}"] = (df_va[col].values + 1.0) / (mean_c_va + 1.0)

            df_te[f"diff_to_knn_{k}_{col}"] = df_te[col].values - mean_c_te
            df_te[f"ratio_to_knn_{k}_{col}"] = (df_te[col].values + 1.0) / (mean_c_te + 1.0)

    # 2. Continuous Spatial Gaussian Kernel Smoothing
    knn_gauss_self = NearestNeighbors(n_neighbors=80, metric="euclidean").fit(coords_tr)
    dists_tr_g, idxs_tr_g = knn_gauss_self.kneighbors(coords_tr)
    dists_tr_g, idxs_tr_g = dists_tr_g[:, 1:], idxs_tr_g[:, 1:]

    knn_gauss_other = NearestNeighbors(n_neighbors=80, metric="euclidean").fit(coords_tr)
    dists_va_g, idxs_va_g = knn_gauss_other.kneighbors(coords_va)
    dists_te_g, idxs_te_g = knn_gauss_other.kneighbors(coords_te)

    for bw in kernel_bandwidths:
        bw_label = int(bw * 1000)
        # Compute Gaussian weights
        gw_tr = np.exp(-(dists_tr_g**2) / (2.0 * (bw**2)))
        gw_va = np.exp(-(dists_va_g**2) / (2.0 * (bw**2)))
        gw_te = np.exp(-(dists_te_g**2) / (2.0 * (bw**2)))

        sum_gw_tr = np.sum(gw_tr, axis=1, keepdims=True) + 1e-8
        sum_gw_va = np.sum(gw_va, axis=1, keepdims=True) + 1e-8
        sum_gw_te = np.sum(gw_te, axis=1, keepdims=True) + 1e-8

        norm_gw_tr = gw_tr / sum_gw_tr
        norm_gw_va = gw_va / sum_gw_va
        norm_gw_te = gw_te / sum_gw_te

        df_tr[f"gauss_bw_{bw_label}_log_price"] = np.sum(norm_gw_tr * log_price_tr[idxs_tr_g], axis=1)
        df_tr[f"gauss_bw_{bw_label}_log_ppsf"] = np.sum(norm_gw_tr * log_ppsf_tr[idxs_tr_g], axis=1)

        df_va[f"gauss_bw_{bw_label}_log_price"] = np.sum(norm_gw_va * log_price_tr[idxs_va_g], axis=1)
        df_va[f"gauss_bw_{bw_label}_log_ppsf"] = np.sum(norm_gw_va * log_ppsf_tr[idxs_va_g], axis=1)

        df_te[f"gauss_bw_{bw_label}_log_price"] = np.sum(norm_gw_te * log_price_tr[idxs_te_g], axis=1)
        df_te[f"gauss_bw_{bw_label}_log_ppsf"] = np.sum(norm_gw_te * log_ppsf_tr[idxs_te_g], axis=1)

    return df_tr, df_va, df_te


# =========================================================================
# 3. Model Architecture Definitions
# =========================================================================
class SpatialTabularBlock(nn.Module):
    """Residual Block with LayerNorm, Dropout, and SiLU activation."""

    def __init__(self, hidden_dim: int, dropout_rate: float = 0.15):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.block(x))


class SpatialTabularResNet(nn.Module):
    """Deep Tabular Residual Network with Spatial Cluster Embeddings."""

    def __init__(
        self,
        num_continuous_features: int,
        cat_cardinalities: List[int],
        cat_embedding_dims: Optional[List[int]] = None,
        hidden_dim: int = 256,
        num_blocks: int = 3,
        dropout_rate: float = 0.15,
        output_dim: int = 1,
    ):
        super().__init__()

        self.embeddings = nn.ModuleList()
        total_emb_dim = 0

        if cat_embedding_dims is None:
            cat_embedding_dims = [
                max(4, min(32, int(cardinality**0.5) + 4))
                for cardinality in cat_cardinalities
            ]

        for card, emb_dim in zip(cat_cardinalities, cat_embedding_dims):
            self.embeddings.append(nn.Embedding(card + 1, emb_dim, padding_idx=card))
            total_emb_dim += emb_dim

        input_dim = num_continuous_features + total_emb_dim

        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
        )

        self.res_blocks = nn.ModuleList(
            [SpatialTabularBlock(hidden_dim, dropout_rate) for _ in range(num_blocks)]
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout_rate / 2.0),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(
        self,
        x_num: torch.Tensor,
        x_cat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if x_cat is not None and len(self.embeddings) > 0:
            emb_outs = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
            x_cat_emb = torch.cat(emb_outs, dim=-1)
            x_combined = torch.cat([x_num, x_cat_emb], dim=-1)
        else:
            x_combined = x_num

        h = self.input_layer(x_combined)
        for block in self.res_blocks:
            h = block(h)

        out = self.head(h)
        return out.squeeze(-1)


# =========================================================================
# 4. Training & Out-of-Fold Evaluation Loop
# =========================================================================
n_train = len(train_feat)
n_test = len(test_feat)

oof_lgb = np.zeros(n_train, dtype=np.float64)
oof_xgb = np.zeros(n_train, dtype=np.float64)
oof_cat = np.zeros(n_train, dtype=np.float64)
oof_et = np.zeros(n_train, dtype=np.float64)
oof_nn = np.zeros(n_train, dtype=np.float64)

test_preds_lgb = np.zeros(n_test, dtype=np.float64)
test_preds_xgb = np.zeros(n_test, dtype=np.float64)
test_preds_cat = np.zeros(n_test, dtype=np.float64)
test_preds_et = np.zeros(n_test, dtype=np.float64)
test_preds_nn = np.zeros(n_test, dtype=np.float64)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Neural Network Execution Device: {device}")

for fold in range(n_splits):
    print(f"\n--- Training Fold {fold + 1}/{n_splits} ---")

    train_idx = (train_feat["fold"] != fold).values
    val_idx = (train_feat["fold"] == fold).values

    df_tr_fold = train_feat.loc[train_idx].copy()
    df_va_fold = train_feat.loc[val_idx].copy()
    df_te_fold = test_feat.copy()

    df_tr_fold, df_va_fold, df_te_fold = compute_spatial_knn_features(
        df_tr_fold, df_va_fold, df_te_fold
    )

    drop_cols = ["PARCELNO", "SALE_PRC", "fold"]
    feature_cols = [c for c in df_tr_fold.columns if c not in drop_cols]
    num_cols = [c for c in feature_cols if c not in cat_cols]

    X_tr_df = df_tr_fold[feature_cols]
    y_tr_log = train_log_target[train_idx]
    y_tr_raw = train_target[train_idx]

    X_va_df = df_va_fold[feature_cols]
    y_va_log = train_log_target[val_idx]
    y_va_raw = train_target[val_idx]

    X_te_df = df_te_fold[feature_cols]

    # --- A. LightGBM Model ---
    lgb_params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "n_estimators": 3000,
        "learning_rate": 0.025,
        "num_leaves": 47,
        "max_depth": 8,
        "min_child_samples": 16,
        "subsample": 0.85,
        "subsample_freq": 1,
        "colsample_bytree": 0.80,
        "reg_alpha": 0.1,
        "reg_lambda": 1.5,
        "random_state": 42 + fold,
        "n_jobs": -1,
        "verbose": -1,
    }
    lgb_model = lgb.LGBMRegressor(**lgb_params)
    lgb_model.fit(
        X_tr_df,
        y_tr_log,
        eval_set=[(X_va_df, y_va_log)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )

    pred_tr_log_lgb = lgb_model.predict(X_tr_df)
    smear_lgb = np.mean(np.exp(y_tr_log - pred_tr_log_lgb))
    pred_val_log_lgb = lgb_model.predict(X_va_df)
    pred_test_log_lgb = lgb_model.predict(X_te_df)

    oof_lgb[val_idx] = np.maximum(0.0, smear_lgb * np.exp(pred_val_log_lgb) - 1.0)
    test_preds_lgb += np.maximum(0.0, smear_lgb * np.exp(pred_test_log_lgb) - 1.0) / n_splits
    fold_lgb_rmse = root_mean_squared_error(y_va_raw, oof_lgb[val_idx])

    # --- B. XGBoost Model ---
    xgb_params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "n_estimators": 3000,
        "learning_rate": 0.02,
        "max_depth": 7,
        "min_child_weight": 4,
        "subsample": 0.85,
        "colsample_bytree": 0.80,
        "reg_alpha": 0.1,
        "reg_lambda": 2.0,
        "random_state": 42 + fold,
        "n_jobs": -1,
        "tree_method": "hist",
        "early_stopping_rounds": 50,
    }
    xgb_model = xgb.XGBRegressor(**xgb_params)
    xgb_model.fit(
        X_tr_df,
        y_tr_log,
        eval_set=[(X_va_df, y_va_log)],
        verbose=False,
    )

    pred_tr_log_xgb = xgb_model.predict(X_tr_df)
    smear_xgb = np.mean(np.exp(y_tr_log - pred_tr_log_xgb))
    pred_val_log_xgb = xgb_model.predict(X_va_df)
    pred_test_log_xgb = xgb_model.predict(X_te_df)

    oof_xgb[val_idx] = np.maximum(0.0, smear_xgb * np.exp(pred_val_log_xgb) - 1.0)
    test_preds_xgb += np.maximum(0.0, smear_xgb * np.exp(pred_test_log_xgb) - 1.0) / n_splits
    fold_xgb_rmse = root_mean_squared_error(y_va_raw, oof_xgb[val_idx])

    # --- C. CatBoost Model ---
    cb_model = CatBoostRegressor(
        iterations=3000,
        learning_rate=0.03,
        depth=7,
        l2_leaf_reg=3.0,
        random_seed=42 + fold,
        early_stopping_rounds=50,
        verbose=False,
        thread_count=-1,
    )
    cb_model.fit(
        X_tr_df,
        y_tr_log,
        eval_set=(X_va_df, y_va_log),
        verbose=False,
    )

    pred_tr_log_cat = cb_model.predict(X_tr_df)
    smear_cat = np.mean(np.exp(y_tr_log - pred_tr_log_cat))
    pred_val_log_cat = cb_model.predict(X_va_df)
    pred_test_log_cat = cb_model.predict(X_te_df)

    oof_cat[val_idx] = np.maximum(0.0, smear_cat * np.exp(pred_val_log_cat) - 1.0)
    test_preds_cat += np.maximum(0.0, smear_cat * np.exp(pred_test_log_cat) - 1.0) / n_splits
    fold_cat_rmse = root_mean_squared_error(y_va_raw, oof_cat[val_idx])

    # --- D. ExtraTreesRegressor Model ---
    et_model = ExtraTreesRegressor(
        n_estimators=300,
        max_depth=24,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features=0.75,
        random_state=42 + fold,
        n_jobs=-1,
    )
    et_model.fit(X_tr_df, y_tr_log)

    pred_tr_log_et = et_model.predict(X_tr_df)
    smear_et = np.mean(np.exp(y_tr_log - pred_tr_log_et))
    pred_val_log_et = et_model.predict(X_va_df)
    pred_test_log_et = et_model.predict(X_te_df)

    oof_et[val_idx] = np.maximum(0.0, smear_et * np.exp(pred_val_log_et) - 1.0)
    test_preds_et += np.maximum(0.0, smear_et * np.exp(pred_test_log_et) - 1.0) / n_splits
    fold_et_rmse = root_mean_squared_error(y_va_raw, oof_et[val_idx])

    # --- E. PyTorch SpatialTabularResNet Model ---
    scaler = StandardScaler()
    X_tr_num_scaled = scaler.fit_transform(X_tr_df[num_cols].values).astype(np.float32)
    X_va_num_scaled = scaler.transform(X_va_df[num_cols].values).astype(np.float32)
    X_te_num_scaled = scaler.transform(X_te_df[num_cols].values).astype(np.float32)

    X_tr_cat = X_tr_df[cat_cols].values.astype(np.int64)
    X_va_cat = X_va_df[cat_cols].values.astype(np.int64)
    X_te_cat = X_te_df[cat_cols].values.astype(np.int64)

    train_dataset = TensorDataset(
        torch.tensor(X_tr_num_scaled),
        torch.tensor(X_tr_cat),
        torch.tensor(y_tr_log, dtype=torch.float32),
    )
    train_loader = DataLoader(
        train_dataset, batch_size=256, shuffle=True, drop_last=False
    )

    nn_model = SpatialTabularResNet(
        num_continuous_features=len(num_cols),
        cat_cardinalities=cat_cardinalities,
        hidden_dim=256,
        num_blocks=3,
        dropout_rate=0.15,
    ).to(device)

    optimizer = optim.AdamW(nn_model.parameters(), lr=2e-3, weight_decay=1e-4, eps=1e-8)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=55 * len(train_loader), eta_min=2e-5
    )
    criterion = nn.SmoothL1Loss(beta=0.15)

    best_val_rmse = float("inf")
    best_nn_val_preds = None
    best_nn_test_preds = None

    tr_num_tensor = torch.tensor(X_tr_num_scaled).to(device)
    tr_cat_tensor = torch.tensor(X_tr_cat).to(device)
    va_num_tensor = torch.tensor(X_va_num_scaled).to(device)
    va_cat_tensor = torch.tensor(X_va_cat).to(device)
    te_num_tensor = torch.tensor(X_te_num_scaled).to(device)
    te_cat_tensor = torch.tensor(X_te_cat).to(device)

    for epoch in range(55):
        nn_model.train()
        for batch_num, batch_cat, batch_y in train_loader:
            batch_num = batch_num.to(device)
            batch_cat = batch_cat.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            pred = nn_model(batch_num, batch_cat)
            loss = criterion(pred, batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(nn_model.parameters(), max_norm=2.0)
            optimizer.step()
            scheduler.step()

        nn_model.eval()
        with torch.no_grad():
            tr_out_log = nn_model(tr_num_tensor, tr_cat_tensor).cpu().numpy()
            smear_nn = np.mean(np.exp(y_tr_log - tr_out_log))
            val_out_log = nn_model(va_num_tensor, va_cat_tensor).cpu().numpy()
            val_out_dollar = np.maximum(0.0, smear_nn * np.exp(val_out_log) - 1.0)
            current_val_rmse = root_mean_squared_error(y_va_raw, val_out_dollar)

            if current_val_rmse < best_val_rmse:
                best_val_rmse = current_val_rmse
                best_nn_val_preds = val_out_dollar
                test_out_log = nn_model(te_num_tensor, te_cat_tensor).cpu().numpy()
                best_nn_test_preds = np.maximum(0.0, smear_nn * np.exp(test_out_log) - 1.0)

    oof_nn[val_idx] = best_nn_val_preds
    test_preds_nn += best_nn_test_preds / n_splits
    fold_nn_rmse = best_val_rmse

    print(
        f"Fold {fold + 1} RMSE ($): LightGBM: {fold_lgb_rmse:,.1f} | XGBoost: {fold_xgb_rmse:,.1f} | CatBoost: {fold_cat_rmse:,.1f} | ExtraTrees: {fold_et_rmse:,.1f} | SpatialTabularResNet: {fold_nn_rmse:,.1f}"
    )


# =========================================================================
# 5. Out-of-Fold Performance Summary & SLSQP Optimization
# =========================================================================
overall_lgb_rmse = root_mean_squared_error(train_target, oof_lgb)
overall_xgb_rmse = root_mean_squared_error(train_target, oof_xgb)
overall_cat_rmse = root_mean_squared_error(train_target, oof_cat)
overall_et_rmse = root_mean_squared_error(train_target, oof_et)
overall_nn_rmse = root_mean_squared_error(train_target, oof_nn)

print("\n" + "=" * 60)
print("Out-of-Fold Model Performance Comparison (Dollar RMSE):")
print(f"  LightGBM Score:              {overall_lgb_rmse:,.2f}")
print(f"  XGBoost Score:               {overall_xgb_rmse:,.2f}")
print(f"  CatBoost Score:              {overall_cat_rmse:,.2f}")
print(f"  ExtraTrees Score:            {overall_et_rmse:,.2f}")
print(f"  SpatialTabularResNet Score:  {overall_nn_rmse:,.2f}")
print("=" * 60)

oof_matrix = np.column_stack([oof_lgb, oof_xgb, oof_cat, oof_et, oof_nn])
test_matrix = np.column_stack(
    [test_preds_lgb, test_preds_xgb, test_preds_cat, test_preds_et, test_preds_nn]
)


def ensemble_objective(weights: np.ndarray) -> float:
    w = weights / (np.sum(weights) + 1e-12)
    pred_blend = np.dot(oof_matrix, w)
    return root_mean_squared_error(train_target, pred_blend)


init_weights = np.ones(5) / 5.0
bounds = [(0.0, 1.0) for _ in range(5)]
constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

opt_res = minimize(
    ensemble_objective,
    init_weights,
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
    options={"maxiter": 500, "ftol": 1e-9},
)

optimal_weights = opt_res.x / np.sum(opt_res.x)
print(f"\nOptimal Ensemble Weights: {optimal_weights.round(4)}")
print(
    f"  LGBM: {optimal_weights[0]:.4f} | XGB: {optimal_weights[1]:.4f} | CAT: {optimal_weights[2]:.4f} | ET: {optimal_weights[3]:.4f} | NN: {optimal_weights[4]:.4f}"
)

oof_ensemble = np.dot(oof_matrix, optimal_weights)
final_val_score = root_mean_squared_error(train_target, oof_ensemble)

# =========================================================================
# 6. Generate Submission Predictions & Verification
# =========================================================================
test_final_predictions = np.dot(test_matrix, optimal_weights)
test_final_predictions = np.maximum(0.0, test_final_predictions)

submission_df = pd.DataFrame(
    {"id": np.arange(len(test_final_predictions)), "SALE_PRC": test_final_predictions}
)

submission_path = "./submission/submission.csv"
submission_df.to_csv(submission_path, index=False)

assert submission_df["SALE_PRC"].isnull().sum() == 0, "Submission contains null values!"
assert (
    len(submission_df) == n_test
), f"Expected {n_test} rows but got {len(submission_df)}!"

print(f"\nSubmission successfully saved to {submission_path}")
print(f"Submission shape: {submission_df.shape}")
print(f"First 5 predictions:\n{submission_df.head()}")

print(f"Final Validation Score: {final_val_score}")