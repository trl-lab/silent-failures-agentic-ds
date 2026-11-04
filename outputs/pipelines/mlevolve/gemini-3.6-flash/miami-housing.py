import os
import random
import warnings

import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb

warnings.filterwarnings("ignore")


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


seed_everything(42)

# Create submission directory if not exists
os.makedirs("./submission", exist_ok=True)

# 1. Load Data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

y_train_full = train_df["SALE_PRC"].values.astype(np.float64)

# Global bounding box for coordinate normalization
lat_min = min(train_df["LATITUDE"].min(), test_df["LATITUDE"].min())
lat_max = max(train_df["LATITUDE"].max(), test_df["LATITUDE"].max())
lon_min = min(train_df["LONGITUDE"].min(), test_df["LONGITUDE"].min())
lon_max = max(train_df["LONGITUDE"].max(), test_df["LONGITUDE"].max())


# 2. Domain Feature Engineering
def build_domain_features(df):
    df = df.copy()

    # Living & Land Geometry
    df["living_to_land_ratio"] = df["TOT_LVG_AREA"] / (df["LND_SQFOOT"] + 1.0)
    df["unbuilt_land_area"] = df["LND_SQFOOT"] - df["TOT_LVG_AREA"]
    df["log_tot_lvg_area"] = np.log1p(np.maximum(df["TOT_LVG_AREA"], 0))
    df["log_lnd_sqfoot"] = np.log1p(np.maximum(df["LND_SQFOOT"], 0))

    # Quality & Age Interactions
    df["quality_adjusted_living_area"] = df["TOT_LVG_AREA"] * df["structure_quality"]
    df["quality_to_age_ratio"] = df["structure_quality"] / (df["age"] + 1.0)
    df["effective_age"] = df["age"] / (df["structure_quality"] + 0.1)
    df["quality_sq"] = df["structure_quality"] ** 2

    # Special Features
    df["has_spec_feat"] = (df["SPEC_FEAT_VAL"] > 0).astype(np.float32)
    df["log_spec_feat_val"] = np.log1p(np.maximum(df["SPEC_FEAT_VAL"], 0))
    df["spec_feat_per_living_sqft"] = df["SPEC_FEAT_VAL"] / (df["TOT_LVG_AREA"] + 1.0)
    df["spec_feat_per_land_sqft"] = df["SPEC_FEAT_VAL"] / (df["LND_SQFOOT"] + 1.0)

    # Location, Noise & Accessibility
    df["noise_index"] = (
        df["avno60plus"] * 3.0
        + 1000.0 / (df["RAIL_DIST"] + 100.0)
        + 1000.0 / (df["HWY_DIST"] + 100.0)
    )
    df["accessibility_score"] = 10000.0 / (df["CNTR_DIST"] + 1000.0) + 10000.0 / (
        df["SUBCNTR_DI"] + 1000.0
    )
    df["coastal_score"] = 10000.0 / (df["OCEAN_DIST"] + 500.0) + 10000.0 / (
        df["WATER_DIST"] + 500.0
    )
    df["ocean_vs_cntr_ratio"] = df["OCEAN_DIST"] / (df["CNTR_DIST"] + 1.0)
    df["subcntr_vs_cntr_ratio"] = df["SUBCNTR_DI"] / (df["CNTR_DIST"] + 1.0)
    df["hwy_vs_rail_ratio"] = df["HWY_DIST"] / (df["RAIL_DIST"] + 1.0)

    # Temporal Features
    month_rad = 2.0 * np.pi * df["month_sold"] / 12.0
    df["month_sin"] = np.sin(month_rad)
    df["month_cos"] = np.cos(month_rad)

    # Fourier Positional Encodings
    lat_norm = (df["LATITUDE"] - lat_min) / (lat_max - lat_min + 1e-6)
    lon_norm = (df["LONGITUDE"] - lon_min) / (lon_max - lon_min + 1e-6)

    for k in range(6):
        freq = (2**k) * np.pi
        df[f"lat_sin_{k}"] = np.sin(freq * lat_norm)
        df[f"lat_cos_{k}"] = np.cos(freq * lat_norm)
        df[f"lon_sin_{k}"] = np.sin(freq * lon_norm)
        df[f"lon_cos_{k}"] = np.cos(freq * lon_norm)

    # Coordinate Rotations
    cbd_lat, cbd_lon = 25.7617, -80.1918
    df["polar_r_cbd"] = df["CNTR_DIST"]
    df["polar_angle_cbd"] = np.arctan2(
        df["LATITUDE"] - cbd_lat, df["LONGITUDE"] - cbd_lon
    )

    for angle in [15, 30, 45, 60, 75]:
        rad = np.radians(angle)
        df[f"rot_lat_{angle}"] = df["LATITUDE"] * np.cos(rad) - df[
            "LONGITUDE"
        ] * np.sin(rad)
        df[f"rot_lon_{angle}"] = df["LATITUDE"] * np.sin(rad) + df[
            "LONGITUDE"
        ] * np.cos(rad)

    if "PARCELNO" in df.columns:
        df = df.drop(columns=["PARCELNO"])
    if "SALE_PRC" in df.columns:
        df = df.drop(columns=["SALE_PRC"])

    return df


X_tr_base = build_domain_features(train_df)
X_te_base = build_domain_features(test_df)


# 3. Leak-Free Spatial KNN Feature Generator
def add_leak_free_knn_features(X_tr, y_tr, X_va, X_te):
    X_tr = X_tr.copy()
    X_va = X_va.copy()
    X_te = X_te.copy()

    coords_tr = X_tr[["LATITUDE", "LONGITUDE"]].values
    coords_va = X_va[["LATITUDE", "LONGITUDE"]].values
    coords_te = X_te[["LATITUDE", "LONGITUDE"]].values

    price_sqft_tr = y_tr / (X_tr["TOT_LVG_AREA"].values + 1.0)
    price_tr = y_tr

    nn_model = NearestNeighbors(n_neighbors=151, algorithm="kd_tree", n_jobs=-1)
    nn_model.fit(coords_tr)

    # Query neighbors
    dist_tr, idx_tr = nn_model.kneighbors(coords_tr)
    dist_va, idx_va = nn_model.kneighbors(coords_va)
    dist_te, idx_te = nn_model.kneighbors(coords_te)

    k_list = [5, 15, 35, 75, 150]

    for k in k_list:
        # Train set (exclude self neighbor at index 0)
        sub_idx_tr = idx_tr[:, 1 : k + 1]
        sub_dist_tr = dist_tr[:, 1 : k + 1] + 1e-5
        weights_tr = 1.0 / sub_dist_tr
        weights_tr = weights_tr / weights_tr.sum(axis=1, keepdims=True)

        X_tr[f"knn_mean_price_{k}"] = np.mean(price_tr[sub_idx_tr], axis=1)
        X_tr[f"knn_std_price_{k}"] = np.std(price_tr[sub_idx_tr], axis=1)
        X_tr[f"knn_wmean_price_{k}"] = np.sum(price_tr[sub_idx_tr] * weights_tr, axis=1)
        X_tr[f"knn_mean_sqft_price_{k}"] = np.mean(price_sqft_tr[sub_idx_tr], axis=1)

        # Validation set
        sub_idx_va = idx_va[:, :k]
        sub_dist_va = dist_va[:, :k] + 1e-5
        weights_va = 1.0 / sub_dist_va
        weights_va = weights_va / weights_va.sum(axis=1, keepdims=True)

        X_va[f"knn_mean_price_{k}"] = np.mean(price_tr[sub_idx_va], axis=1)
        X_va[f"knn_std_price_{k}"] = np.std(price_tr[sub_idx_va], axis=1)
        X_va[f"knn_wmean_price_{k}"] = np.sum(price_tr[sub_idx_va] * weights_va, axis=1)
        X_va[f"knn_mean_sqft_price_{k}"] = np.mean(price_sqft_tr[sub_idx_va], axis=1)

        # Test set
        sub_idx_te = idx_te[:, :k]
        sub_dist_te = dist_te[:, :k] + 1e-5
        weights_te = 1.0 / sub_dist_te
        weights_te = weights_te / weights_te.sum(axis=1, keepdims=True)

        X_te[f"knn_mean_price_{k}"] = np.mean(price_tr[sub_idx_te], axis=1)
        X_te[f"knn_std_price_{k}"] = np.std(price_tr[sub_idx_te], axis=1)
        X_te[f"knn_wmean_price_{k}"] = np.sum(price_tr[sub_idx_te] * weights_te, axis=1)
        X_te[f"knn_mean_sqft_price_{k}"] = np.mean(price_sqft_tr[sub_idx_te], axis=1)

    return X_tr, X_va, X_te


# 4. PyTorch Spatial RBF Network Architecture
class SpatialRBFEncoding(nn.Module):

    def __init__(self, num_centroids=64, scales=[0.01, 0.03, 0.08, 0.20]):
        super().__init__()
        self.num_centroids = num_centroids
        side = int(np.sqrt(num_centroids))
        gx = torch.linspace(0.0, 1.0, side)
        gy = torch.linspace(0.0, 1.0, side)
        grid_x, grid_y = torch.meshgrid(gx, gy, indexing="ij")
        centroids = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)
        self.register_buffer("centroids", centroids)
        self.register_buffer("scales_tensor", torch.tensor(scales, dtype=torch.float32))

    def forward(self, coords):
        dist_sq = torch.sum(
            (coords.unsqueeze(1) - self.centroids.unsqueeze(0)) ** 2, dim=-1
        )
        rbf_list = [
            torch.exp(-dist_sq / (2.0 * (s**2) + 1e-6)) for s in self.scales_tensor
        ]
        return torch.cat(rbf_list, dim=-1)


class TabularResidualBlock(nn.Module):

    def __init__(self, in_features, hidden_features, dropout_rate=0.15):
        super().__init__()
        self.norm1 = nn.BatchNorm1d(in_features)
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.norm2 = nn.BatchNorm1d(hidden_features)
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        res = x
        out = self.fc1(self.act(self.norm1(x)))
        out = self.fc2(self.dropout(self.act(self.norm2(out))))
        return res + out


class SpatialRBFAttentionNet(nn.Module):

    def __init__(
        self,
        num_tabular_features,
        rbf_centroids=64,
        rbf_scales=[0.01, 0.03, 0.08, 0.20],
        hidden_dim=256,
        num_res_blocks=3,
        dropout_rate=0.15,
    ):
        super().__init__()
        self.rbf_layer = SpatialRBFEncoding(
            num_centroids=rbf_centroids, scales=rbf_scales
        )
        rbf_dim = rbf_centroids * len(rbf_scales)

        self.spatial_proj = nn.Sequential(
            nn.Linear(rbf_dim, 64), nn.GELU(), nn.BatchNorm1d(64)
        )

        self.tab_proj = nn.Sequential(
            nn.BatchNorm1d(num_tabular_features),
            nn.Linear(num_tabular_features, hidden_dim - 64),
            nn.GELU(),
        )

        self.res_blocks = nn.ModuleList(
            [
                TabularResidualBlock(hidden_dim, hidden_dim * 2, dropout_rate)
                for _ in range(num_res_blocks)
            ]
        )

        self.head = nn.Sequential(
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 1),
        )

    def forward(self, x_tab, x_coords):
        rbf_feat = self.rbf_layer(x_coords)
        spatial_emb = self.spatial_proj(rbf_feat)
        tab_emb = self.tab_proj(x_tab)
        x = torch.cat([tab_emb, spatial_emb], dim=-1)
        for block in self.res_blocks:
            x = block(x)
        return self.head(x).squeeze(-1)


# 5. Model Parameters
lgb_log_params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "n_estimators": 3500,
    "learning_rate": 0.025,
    "num_leaves": 63,
    "max_depth": 8,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}

lgb_huber_params = {
    "objective": "huber",
    "alpha": 0.9,
    "metric": "rmse",
    "boosting_type": "gbdt",
    "n_estimators": 3500,
    "learning_rate": 0.025,
    "num_leaves": 127,
    "max_depth": 9,
    "subsample": 0.85,
    "colsample_bytree": 0.65,
    "reg_alpha": 0.5,
    "reg_lambda": 2.0,
    "random_state": 101,
    "n_jobs": -1,
    "verbose": -1,
}

xgb_params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "n_estimators": 3500,
    "learning_rate": 0.02,
    "max_depth": 7,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": -1,
    "tree_method": "hist",
    "early_stopping_rounds": 50,
}

cat_params = {
    "loss_function": "RMSE",
    "eval_metric": "RMSE",
    "iterations": 3500,
    "learning_rate": 0.025,
    "depth": 7,
    "subsample": 0.8,
    "random_seed": 42,
    "verbose": False,
    "thread_count": -1,
    "early_stopping_rounds": 50,
}

# 6. Cross-Validation Loop Setup
kf = KFold(n_splits=5, shuffle=True, random_state=42)

num_train = len(train_df)
num_test = len(test_df)
num_models = 5  # lgb_log, lgb_huber, xgb, cat, nn

oof_preds = np.zeros((num_train, num_models), dtype=np.float64)
test_preds = np.zeros((num_test, num_models), dtype=np.float64)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for fold, (train_idx, val_idx) in enumerate(kf.split(X_tr_base, y_train_full)):
    X_tr_fold, y_tr_fold = X_tr_base.iloc[train_idx], y_train_full[train_idx]
    X_va_fold, y_va_fold = X_tr_base.iloc[val_idx], y_train_full[val_idx]
    X_te_fold = X_te_base.copy()

    # Compute leak-free spatial KNN features
    X_tr_fold, X_va_fold, X_te_fold = add_leak_free_knn_features(
        X_tr_fold, y_tr_fold, X_va_fold, X_te_fold
    )

    # Model 0: LightGBM (Log Target)
    dtrain_log = lgb.Dataset(X_tr_fold, label=np.log1p(y_tr_fold))
    dval_log = lgb.Dataset(X_va_fold, label=np.log1p(y_va_fold), reference=dtrain_log)
    m_lgb_log = lgb.train(
        lgb_log_params,
        dtrain_log,
        valid_sets=[dval_log],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )
    oof_preds[val_idx, 0] = np.expm1(
        m_lgb_log.predict(X_va_fold, num_iteration=m_lgb_log.best_iteration)
    )
    test_preds[:, 0] += (
        np.expm1(m_lgb_log.predict(X_te_fold, num_iteration=m_lgb_log.best_iteration))
        / 5.0
    )

    # Model 1: LightGBM (Huber Loss / Raw Target)
    dtrain_huber = lgb.Dataset(X_tr_fold, label=y_tr_fold)
    dval_huber = lgb.Dataset(X_va_fold, label=y_va_fold, reference=dtrain_huber)
    m_lgb_huber = lgb.train(
        lgb_huber_params,
        dtrain_huber,
        valid_sets=[dval_huber],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )
    oof_preds[val_idx, 1] = m_lgb_huber.predict(
        X_va_fold, num_iteration=m_lgb_huber.best_iteration
    )
    test_preds[:, 1] += (
        m_lgb_huber.predict(X_te_fold, num_iteration=m_lgb_huber.best_iteration) / 5.0
    )

    # Model 2: XGBoost (Log Target)
    m_xgb = xgb.XGBRegressor(**xgb_params)
    m_xgb.fit(
        X_tr_fold,
        np.log1p(y_tr_fold),
        eval_set=[(X_va_fold, np.log1p(y_va_fold))],
        verbose=False,
    )
    oof_preds[val_idx, 2] = np.expm1(m_xgb.predict(X_va_fold))
    test_preds[:, 2] += np.expm1(m_xgb.predict(X_te_fold)) / 5.0

    # Model 3: CatBoost (Raw Target)
    m_cat = cb.CatBoostRegressor(**cat_params)
    m_cat.fit(X_tr_fold, y_tr_fold, eval_set=(X_va_fold, y_va_fold), verbose=False)
    oof_preds[val_idx, 3] = m_cat.predict(X_va_fold)
    test_preds[:, 3] += m_cat.predict(X_te_fold) / 5.0

    # Model 4: PyTorch SpatialRBFAttentionNet
    coords_tr = (X_tr_fold[["LATITUDE", "LONGITUDE"]].values - [lat_min, lon_min]) / [
        lat_max - lat_min + 1e-6,
        lon_max - lon_min + 1e-6,
    ]
    coords_va = (X_va_fold[["LATITUDE", "LONGITUDE"]].values - [lat_min, lon_min]) / [
        lat_max - lat_min + 1e-6,
        lon_max - lon_min + 1e-6,
    ]
    coords_te = (X_te_fold[["LATITUDE", "LONGITUDE"]].values - [lat_min, lon_min]) / [
        lat_max - lat_min + 1e-6,
        lon_max - lon_min + 1e-6,
    ]

    tab_cols = [c for c in X_tr_fold.columns if c not in ["LATITUDE", "LONGITUDE"]]
    scaler = StandardScaler()
    X_tab_tr = scaler.fit_transform(X_tr_fold[tab_cols].fillna(0).values)
    X_tab_va = scaler.transform(X_va_fold[tab_cols].fillna(0).values)
    X_tab_te = scaler.transform(X_te_fold[tab_cols].fillna(0).values)

    log_y_tr = np.log1p(y_tr_fold)
    log_y_va = np.log1p(y_va_fold)
    y_mean, y_std = log_y_tr.mean(), log_y_tr.std()

    norm_y_tr = (log_y_tr - y_mean) / y_std
    norm_y_va = (log_y_va - y_mean) / y_std

    ds_tr = TensorDataset(
        torch.tensor(X_tab_tr, dtype=torch.float32),
        torch.tensor(coords_tr, dtype=torch.float32),
        torch.tensor(norm_y_tr, dtype=torch.float32),
    )
    ds_va = TensorDataset(
        torch.tensor(X_tab_va, dtype=torch.float32),
        torch.tensor(coords_va, dtype=torch.float32),
        torch.tensor(norm_y_va, dtype=torch.float32),
    )

    dl_tr = DataLoader(ds_tr, batch_size=256, shuffle=True)
    dl_va = DataLoader(ds_va, batch_size=512, shuffle=False)

    nn_model = SpatialRBFAttentionNet(num_tabular_features=len(tab_cols)).to(device)
    optimizer = torch.optim.AdamW(nn_model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=40, eta_min=1e-5
    )
    criterion = nn.SmoothL1Loss()

    best_va_loss = float("inf")
    best_weights = None

    for epoch in range(40):
        nn_model.train()
        for b_tab, b_coords, b_y in dl_tr:
            b_tab, b_coords, b_y = (
                b_tab.to(device),
                b_coords.to(device),
                b_y.to(device),
            )
            optimizer.zero_grad()
            pred = nn_model(b_tab, b_coords)
            loss = criterion(pred, b_y)
            loss.backward()
            optimizer.step()
        scheduler.step()

        nn_model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for b_tab, b_coords, b_y in dl_va:
                b_tab, b_coords, b_y = (
                    b_tab.to(device),
                    b_coords.to(device),
                    b_y.to(device),
                )
                pred = nn_model(b_tab, b_coords)
                va_loss += criterion(pred, b_y).item() * len(b_y)
        va_loss /= len(ds_va)

        if va_loss < best_va_loss:
            best_va_loss = va_loss
            best_weights = nn_model.state_dict().copy()

    nn_model.load_state_dict(best_weights)
    nn_model.eval()

    with torch.no_grad():
        t_tab_va = torch.tensor(X_tab_va, dtype=torch.float32).to(device)
        t_coords_va = torch.tensor(coords_va, dtype=torch.float32).to(device)
        nn_pred_va = nn_model(t_tab_va, t_coords_va).cpu().numpy()
        oof_preds[val_idx, 4] = np.expm1(nn_pred_va * y_std + y_mean)

        t_tab_te = torch.tensor(X_tab_te, dtype=torch.float32).to(device)
        t_coords_te = torch.tensor(coords_te, dtype=torch.float32).to(device)
        nn_pred_te = nn_model(t_tab_te, t_coords_te).cpu().numpy()
        test_preds[:, 4] += np.expm1(nn_pred_te * y_std + y_mean) / 5.0


# 7. Optimize SLSQP Ensemble Weights on OOF Predictions
def loss_func(weights):
    weights = weights / np.sum(weights)
    blend_oof = np.dot(oof_preds, weights)
    return root_mean_squared_error(y_train_full, blend_oof)


initial_weights = np.ones(num_models) / num_models
bounds = [(0, 1) for _ in range(num_models)]
res = minimize(loss_func, initial_weights, method="SLSQP", bounds=bounds, tol=1e-6)
opt_weights = res.x / np.sum(res.x)

oof_ensemble = np.dot(oof_preds, opt_weights)
test_ensemble = np.dot(test_preds, opt_weights)


# 8. Optimal Scale Alignment Post-Processing
def scale_loss(s):
    return root_mean_squared_error(y_train_full, oof_ensemble * s[0])


res_scale = minimize(scale_loss, [1.0], method="Nelder-Mead")
opt_scale = res_scale.x[0]

final_oof_preds = oof_ensemble * opt_scale
final_test_preds = test_ensemble * opt_scale

final_score = root_mean_squared_error(y_train_full, final_oof_preds)

# 9. Format & Save Final Submission File
submission_df = pd.DataFrame(
    {"id": np.arange(len(test_df)), "SALE_PRC": final_test_preds}
)
submission_df.to_csv("./submission/submission.csv", index=False)

print(f"Final Validation Score: {final_score}")
