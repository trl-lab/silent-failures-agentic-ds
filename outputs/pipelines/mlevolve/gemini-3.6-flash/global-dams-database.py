import json
import os
import warnings
import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import Ridge
from sklearn.model_selection import StratifiedKFold
from scipy.optimize import minimize
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb

warnings.filterwarnings("ignore")

# 1. Setup Directories
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# 2. Load Raw Datasets
train_path = "./input/train.parquet"
test_path = "./input/test.parquet"

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = "Res_capacityM3"
id_col = "id" if "id" in test_df.columns else None

if id_col is None:
    test_df["id"] = np.arange(len(test_df))
    id_col = "id"


# Helper function to locate column names flexibly
def locate_column(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
        for col in df.columns:
            if col.lower() == c.lower():
                return col
    return None


area_col = locate_column(
    train_df, ["AREA_SKM", "Res_area", "AREA_GEO", "AREA_REP", "area"]
)
height_col = locate_column(train_df, ["DAM_HEIGHT", "HEIGHT_M", "DEPTH_M", "height"])
catch_col = locate_column(
    train_df, ["CATCH_SKM", "CATCH_AREA", "CATCH_REP", "catch_area"]
)
dis_col = locate_column(train_df, ["DIS_AVG_LS", "DIS_AVG", "DIS_AVG_CMS", "discharge"])
lat_col = locate_column(train_df, ["LAT_DEG", "LATITUDE", "lat", "Latitude", "LAT"])
lon_col = locate_column(train_df, ["LON_DEG", "LONGITUDE", "lon", "Longitude", "LON"])
basin_col = locate_column(
    train_df, ["MAIN_BASIN", "main_basin", "BASIN_ID", "sub_basin"]
)
country_col = locate_column(train_df, ["COUNTRY", "country", "CNTY_NAME"])

# Clean target
y_train_raw = train_df[target_col].values.astype(np.float64)
y_train_log = np.log1p(np.maximum(0, y_train_raw)).astype(np.float32)

# Create 5-Fold Split based on log-target quantiles
num_folds = 5
train_df["fold"] = -1
quantile_bins = pd.qcut(y_train_log, q=10, labels=False, duplicates="drop")
skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
for fold, (_, val_idx) in enumerate(skf.split(train_df, quantile_bins)):
    train_df.loc[val_idx, "fold"] = fold


# 3. Domain Physical & Dimensional Feature Generation
def generate_physical_features(df):
    feats = pd.DataFrame(index=df.index)

    area = df[area_col].values if area_col else np.nan
    height = df[height_col].values if height_col else np.nan
    catch = df[catch_col].values if catch_col else np.nan
    dis = df[dis_col].values if dis_col else np.nan

    area = np.where(
        area is not None and not np.all(np.isnan(area)),
        np.maximum(0, np.nan_to_num(area)),
        0,
    )
    height = np.where(
        height is not None and not np.all(np.isnan(height)),
        np.maximum(0, np.nan_to_num(height)),
        0,
    )
    catch = np.where(
        catch is not None and not np.all(np.isnan(catch)),
        np.maximum(0, np.nan_to_num(catch)),
        0,
    )
    dis = np.where(
        dis is not None and not np.all(np.isnan(dis)),
        np.maximum(0, np.nan_to_num(dis)),
        0,
    )

    feats["log_area"] = np.log1p(area)
    feats["log_height"] = np.log1p(height)
    feats["log_catch"] = np.log1p(catch)
    feats["log_dis"] = np.log1p(dis)

    feats["log_area_sq"] = feats["log_area"] ** 2
    feats["log_height_sq"] = feats["log_height"] ** 2
    feats["log_area_x_height"] = feats["log_area"] * feats["log_height"]
    feats["log_area_x_catch"] = feats["log_area"] * feats["log_catch"]
    feats["log_height_x_catch"] = feats["log_height"] * feats["log_catch"]
    feats["log_ratio_height_area"] = feats["log_height"] - feats["log_area"]
    feats["log_ratio_area_catch"] = feats["log_area"] - feats["log_catch"]
    feats["log_ratio_height_catch"] = feats["log_height"] - feats["log_catch"]

    vol_cylinder = area * height
    vol_pyramid = (1.0 / 3.0) * area * height
    vol_power_1_1 = (area**1.1) * height
    vol_power_1_2 = (area**1.2) * height

    feats["log_vol_cylinder"] = np.log1p(vol_cylinder)
    feats["log_vol_pyramid"] = np.log1p(vol_pyramid)
    feats["log_vol_power_1_1"] = np.log1p(vol_power_1_1)
    feats["log_vol_power_1_2"] = np.log1p(vol_power_1_2)

    feats["height_to_area"] = height / (area + 1e-4)
    feats["area_to_catch"] = area / (catch + 1e-4)
    feats["catch_to_area"] = catch / (area + 1e-4)
    feats["runoff_yield_proxy"] = dis / (catch + 1e-4)
    feats["residence_proxy"] = vol_cylinder / (dis + 1e-4)
    feats["dis_per_area"] = dis / (area + 1e-4)

    return feats


train_phys = generate_physical_features(train_df)
test_phys = generate_physical_features(test_df)


# 4. Spatial & Geographic Feature Generation
def generate_spatial_features(train_df, test_df):
    train_sp = pd.DataFrame(index=train_df.index)
    test_sp = pd.DataFrame(index=test_df.index)

    if lat_col and lon_col:
        train_lat = np.nan_to_num(train_df[lat_col].values, nan=0.0)
        train_lon = np.nan_to_num(train_df[lon_col].values, nan=0.0)
        test_lat = np.nan_to_num(test_df[lat_col].values, nan=0.0)
        test_lon = np.nan_to_num(test_df[lon_col].values, nan=0.0)

        lat_rad_train = np.radians(train_lat)
        lon_rad_train = np.radians(train_lon)
        lat_rad_test = np.radians(test_lat)
        lon_rad_test = np.radians(test_lon)

        train_sp["sp_x"] = np.cos(lat_rad_train) * np.cos(lon_rad_train)
        train_sp["sp_y"] = np.cos(lat_rad_train) * np.sin(lon_rad_train)
        train_sp["sp_z"] = np.sin(lat_rad_train)

        test_sp["sp_x"] = np.cos(lat_rad_test) * np.cos(lon_rad_test)
        test_sp["sp_y"] = np.cos(lat_rad_test) * np.sin(lon_rad_test)
        test_sp["sp_z"] = np.sin(lat_rad_test)

        for angle in [30, 60]:
            rad = np.radians(angle)
            train_sp[f"rot_lat_{angle}"] = train_lat * np.cos(rad) - train_lon * np.sin(
                rad
            )
            train_sp[f"rot_lon_{angle}"] = train_lat * np.sin(rad) + train_lon * np.cos(
                rad
            )
            test_sp[f"rot_lat_{angle}"] = test_lat * np.cos(rad) - test_lon * np.sin(
                rad
            )
            test_sp[f"rot_lon_{angle}"] = test_lat * np.sin(rad) + test_lon * np.cos(
                rad
            )

        coords_train = np.column_stack([train_lat, train_lon])
        coords_test = np.column_stack([test_lat, test_lon])

        scaler_geo = StandardScaler().fit(coords_train)
        coords_train_scaled = scaler_geo.transform(coords_train)
        coords_test_scaled = scaler_geo.transform(coords_test)

        # Simple spatial features without global cluster/neighbor leakage across folds
        train_sp["spatial_dist_origin"] = np.sqrt(coords_train[:, 0]**2 + coords_train[:, 1]**2)
        test_sp["spatial_dist_origin"] = np.sqrt(coords_test[:, 0]**2 + coords_test[:, 1]**2)

    return train_sp, test_sp


train_spatial, test_spatial = generate_spatial_features(train_df, test_df)

X_train = pd.concat([train_phys, train_spatial], axis=1)
X_test = pd.concat([test_phys, test_spatial], axis=1)

cat_cols = []
for c in [basin_col, country_col, "spatial_cluster_20"]:
    if c in train_df.columns or c in X_train.columns:
        cat_cols.append(c)

for col in cat_cols:
    src_tr = train_df[col] if col in train_df.columns else X_train[col]
    src_te = test_df[col] if col in test_df.columns else X_test[col]

    freq_map = src_tr.value_counts(normalize=True).to_dict()
    X_train[f"{col}_freq"] = src_tr.map(freq_map).fillna(0).values
    X_test[f"{col}_freq"] = src_te.map(freq_map).fillna(0).values

# 5. Out-Of-Fold Basin/Country Physical Attribute Aggregations & Target Encoding (Leak-Free)
group_cols = [col for col in [basin_col, country_col] if col is not None and col in train_df.columns]
phys_attrs = ["log_area", "log_height", "log_catch", "log_dis"]

for g_col in group_cols:
    for attr in phys_attrs:
        mean_col_name = f"{g_col}_{attr}_mean"
        std_col_name = f"{g_col}_{attr}_std"

        X_train[mean_col_name] = 0.0
        X_train[std_col_name] = 0.0

        for fold in range(num_folds):
            tr_idx = (train_df["fold"] != fold).values
            va_idx = (train_df["fold"] == fold).values

            tr_g = train_df.loc[tr_idx, g_col]
            tr_a = train_phys.loc[tr_idx, attr]

            grouped = pd.DataFrame({"g": tr_g, "a": tr_a}).groupby("g")["a"]
            mean_map = grouped.mean()
            std_map = grouped.std().fillna(0.0)

            overall_mean = tr_a.mean()
            overall_std = tr_a.std() if len(tr_a) > 1 else 0.0

            va_g = train_df.loc[va_idx, g_col]
            X_train.loc[va_idx, mean_col_name] = va_g.map(mean_map).fillna(overall_mean).values
            X_train.loc[va_idx, std_col_name] = va_g.map(std_map).fillna(overall_std).values

        full_g = train_df[g_col]
        full_a = train_phys[attr]
        grouped_full = pd.DataFrame({"g": full_g, "a": full_a}).groupby("g")["a"]
        full_mean_map = grouped_full.mean()
        full_std_map = grouped_full.std().fillna(0.0)

        overall_mean_full = full_a.mean()
        overall_std_full = full_a.std() if len(full_a) > 1 else 0.0

        te_g = test_df[g_col]
        X_test[mean_col_name] = te_g.map(full_mean_map).fillna(overall_mean_full).values
        X_test[std_col_name] = te_g.map(full_std_map).fillna(overall_std_full).values

smoothing = 10
global_mean = y_train_log.mean()

for col in cat_cols:
    src_tr = train_df[col] if col in train_df.columns else X_train[col]
    src_te = test_df[col] if col in test_df.columns else X_test[col]

    oof_te = np.zeros(len(train_df))
    for fold in range(num_folds):
        tr_idx = (train_df["fold"] != fold).values
        va_idx = (train_df["fold"] == fold).values

        tr_cats = src_tr.iloc[np.where(tr_idx)[0]]
        tr_target = y_train_log[tr_idx]

        stats = (
            pd.DataFrame({"cat": tr_cats, "target": tr_target})
            .groupby("cat")["target"]
            .agg(["count", "mean"])
        )
        smooth_map = (stats["count"] * stats["mean"] + smoothing * global_mean) / (
            stats["count"] + smoothing
        )

        oof_te[va_idx] = (
            src_tr.iloc[np.where(va_idx)[0]]
            .map(smooth_map)
            .fillna(global_mean)
            .values
        )

    X_train[f"{col}_te"] = oof_te

    full_stats = (
        pd.DataFrame({"cat": src_tr, "target": y_train_log})
        .groupby("cat")["target"]
        .agg(["count", "mean"])
    )
    full_smooth_map = (
        full_stats["count"] * full_stats["mean"] + smoothing * global_mean
    ) / (full_stats["count"] + smoothing)
    X_test[f"{col}_te"] = src_te.map(full_smooth_map).fillna(global_mean).values

# Missing & Infinite Imputation
feature_names = list(X_train.columns)
for f in feature_names:
    med_val = X_train[f].replace([np.inf, -np.inf], np.nan).median()
    if np.isnan(med_val):
        med_val = 0.0
    X_train[f] = X_train[f].replace([np.inf, -np.inf], np.nan).fillna(med_val).values
    X_test[f] = X_test[f].replace([np.inf, -np.inf], np.nan).fillna(med_val).values


# 6. PyTorch TabularResNet Definition
class TabularResBlock(nn.Module):

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
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class TabularResNet(nn.Module):

    def __init__(
        self,
        num_continuous: int,
        hidden_dim: int = 128,
        num_blocks: int = 3,
        dropout_rate: float = 0.15,
    ):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(num_continuous, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.blocks = nn.ModuleList(
            [
                TabularResBlock(hidden_dim, dropout_rate=dropout_rate)
                for _ in range(num_blocks)
            ]
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout_rate / 2.0),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x_cont: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x_cont)
        for block in self.blocks:
            h = block(h)
        out = self.head(h)
        return out.squeeze(-1)


def train_nn_fold(X_tr, y_tr, X_va, y_va, X_te, epochs=25, batch_size=128, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_va_sc = scaler.transform(X_va)
    X_te_sc = scaler.transform(X_te)

    tr_ds = TensorDataset(
        torch.tensor(X_tr_sc, dtype=torch.float32),
        torch.tensor(y_tr, dtype=torch.float32),
    )
    tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, drop_last=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TabularResNet(
        num_continuous=X_tr.shape[1], hidden_dim=128, num_blocks=3
    ).to(device)
    criterion = nn.HuberLoss(delta=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs * len(tr_loader), eta_min=1e-5
    )

    best_loss = float("inf")
    best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    for epoch in range(epochs):
        model.train()
        for bx, by in tr_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            scheduler.step()

        model.eval()
        with torch.no_grad():
            va_x = torch.tensor(X_va_sc, dtype=torch.float32).to(device)
            va_preds = model(va_x)
            va_loss = criterion(
                va_preds, torch.tensor(y_va, dtype=torch.float32).to(device)
            ).item()

        if va_loss < best_loss:
            best_loss = va_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_weights)
    model.to(device)
    model.eval()
    with torch.no_grad():
        va_oof = (
            model(torch.tensor(X_va_sc, dtype=torch.float32).to(device)).cpu().numpy()
        )
        te_preds = (
            model(torch.tensor(X_te_sc, dtype=torch.float32).to(device)).cpu().numpy()
        )
        tr_preds = (
            model(torch.tensor(X_tr_sc, dtype=torch.float32).to(device)).cpu().numpy()
        )

    return tr_preds, va_oof, te_preds


# 7. Model Training & Cross Validation
X_mat = X_train[feature_names].values.astype(np.float32)
X_test_mat = X_test[feature_names].values.astype(np.float32)
folds = train_df["fold"].values.astype(np.int32)
test_ids = test_df[id_col].values

oof_raw_preds = np.zeros(len(train_df), dtype=np.float64)
test_raw_preds = np.zeros(len(test_df), dtype=np.float64)

for fold in range(num_folds):
    tr_idx = folds != fold
    va_idx = folds == fold

    X_tr, y_tr_log, y_tr_raw = X_mat[tr_idx], y_train_log[tr_idx], y_train_raw[tr_idx]
    X_va, y_va_log, y_va_raw = X_mat[va_idx], y_train_log[va_idx], y_train_raw[va_idx]

    # Model 1: LightGBM
    lgb_model = lgb.LGBMRegressor(
        objective="huber",
        metric="rmse",
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        verbose=-1,
        n_jobs=-1,
    )
    lgb_model.fit(
        X_tr,
        y_tr_log,
        eval_set=[(X_va, y_va_log)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    tr_lgb = lgb_model.predict(X_tr)
    va_lgb = lgb_model.predict(X_va)
    te_lgb = lgb_model.predict(X_test_mat)

    # Model 2: XGBoost
    xgb_model = xgb.XGBRegressor(
        objective="reg:pseudohubererror",
        eval_metric="rmse",
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        n_jobs=-1,
        early_stopping_rounds=50,
        tree_method="hist",
    )
    xgb_model.fit(X_tr, y_tr_log, eval_set=[(X_va, y_va_log)], verbose=False)
    tr_xgb = xgb_model.predict(X_tr)
    va_xgb = xgb_model.predict(X_va)
    te_xgb = xgb_model.predict(X_test_mat)

    # Model 3: CatBoost
    cb_model = cb.CatBoostRegressor(
        loss_function="Huber:delta=1.0",
        eval_metric="RMSE",
        iterations=1000,
        learning_rate=0.03,
        depth=6,
        random_seed=42 + fold,
        verbose=0,
        thread_count=-1,
        early_stopping_rounds=50,
    )
    cb_model.fit(
        X_tr,
        y_tr_log,
        eval_set=(X_va, y_va_log),
        verbose=0,
        early_stopping_rounds=50,
    )
    tr_cb = cb_model.predict(X_tr)
    va_cb = cb_model.predict(X_va)
    te_cb = cb_model.predict(X_test_mat)

    # Model 4: PyTorch TabularResNet
    tr_nn, va_nn, te_nn = train_nn_fold(
        X_tr, y_tr_log, X_va, y_va_log, X_test_mat, epochs=25, seed=42 + fold
    )

    # Linear-scale model predictions
    va_lgb_lin = np.maximum(0, np.expm1(va_lgb))
    va_xgb_lin = np.maximum(0, np.expm1(va_xgb))
    va_cb_lin = np.maximum(0, np.expm1(va_cb))
    va_nn_lin = np.maximum(0, np.expm1(va_nn))

    te_lgb_lin = np.maximum(0, np.expm1(te_lgb))
    te_xgb_lin = np.maximum(0, np.expm1(te_xgb))
    te_cb_lin = np.maximum(0, np.expm1(te_cb))
    te_nn_lin = np.maximum(0, np.expm1(te_nn))

    P_va = np.column_stack([va_lgb_lin, va_xgb_lin, va_cb_lin, va_nn_lin])
    P_te = np.column_stack([te_lgb_lin, te_xgb_lin, te_cb_lin, te_nn_lin])

    # Constrained convex optimization to directly minimize linear RMSE
    def loss_func(w):
        pred = P_va @ w
        return np.sqrt(np.mean((pred - y_va_raw) ** 2))

    bounds = [(0, 1) for _ in range(4)]
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    init_w = np.array([0.25, 0.25, 0.25, 0.25])

    opt_res = minimize(loss_func, init_w, method="SLSQP", bounds=bounds, constraints=constraints)
    best_weights = opt_res.x

    va_lin_blend = np.maximum(0, P_va @ best_weights)
    te_lin_blend = np.maximum(0, P_te @ best_weights)

    oof_raw_preds[va_idx] = va_lin_blend
    test_raw_preds += te_lin_blend / num_folds

# 8. Evaluation & Submission Generation
final_rmse = np.sqrt(np.mean((y_train_raw - oof_raw_preds) ** 2))

sub_df = pd.DataFrame({"id": test_ids, "Res_capacityM3": test_raw_preds})
sub_df.to_csv("./submission/submission.csv", index=False)

print(f"Final Validation Score: {final_rmse}")