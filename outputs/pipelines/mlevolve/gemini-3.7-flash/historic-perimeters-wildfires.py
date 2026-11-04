import copy
import os
import re
import warnings
warnings.filterwarnings("ignore")

from catboost import CatBoostRegressor
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb

# Ensure required directories exist
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# Set random seed for reproducibility
np.random.seed(42)

# =========================================================================
# 1. Data Loading & Feature Engineering
# =========================================================================
train_path = "./input/train.parquet"
test_path = "./input/test.parquet"

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

if "id" in test_df.columns:
    test_ids = test_df["id"].values
else:
    test_ids = np.arange(len(test_df))

if "id" in train_df.columns:
    train_ids = train_df["id"].values
else:
    train_ids = np.arange(len(train_df))

target_col = None
for col in train_df.columns:
    if col.lower() == "gisacres":
        target_col = col
        break

if target_col is None:
    raise ValueError("Target column 'gisacres' not found in training data.")

y_train = train_df[target_col].values.astype(np.float64)
train_features_raw = train_df.drop(columns=[target_col]).copy()
test_features_raw = test_df.copy()


def extract_fire_features(df_train, df_test):
    df_tr = df_train.copy()
    df_te = df_test.copy()

    # Identify date/time columns
    date_cols = []
    for col in df_tr.columns:
        col_lower = col.lower()
        if any(
            k in col_lower for k in ["date", "time", "year_", "created", "perim"]
        ) and not any(k in col_lower for k in ["acres", "id", "length"]):
            date_cols.append(col)

    # Process Datetime Features
    for col in date_cols:
        tr_dt = pd.to_datetime(df_tr[col], errors="coerce")
        te_dt = pd.to_datetime(df_te[col], errors="coerce")

        if tr_dt.notna().sum() > 0:
            for dt_series, target_df in [(tr_dt, df_tr), (te_dt, df_te)]:
                target_df[f"{col}_year"] = dt_series.dt.year.fillna(-1).astype(
                    np.float32
                )
                target_df[f"{col}_month"] = dt_series.dt.month.fillna(-1).astype(
                    np.float32
                )
                target_df[f"{col}_day"] = dt_series.dt.day.fillna(-1).astype(np.float32)
                target_df[f"{col}_dayofweek"] = dt_series.dt.dayofweek.fillna(
                    -1
                ).astype(np.float32)
                target_df[f"{col}_dayofyear"] = dt_series.dt.dayofyear.fillna(
                    -1
                ).astype(np.float32)
                target_df[f"{col}_quarter"] = dt_series.dt.quarter.fillna(-1).astype(
                    np.float32
                )
                target_df[f"{col}_is_weekend"] = (dt_series.dt.dayofweek >= 5).astype(
                    np.float32
                )

                # Cyclical transformations for seasonality
                rad_doy = 2 * np.pi * dt_series.dt.dayofyear.fillna(180).values / 365.25
                target_df[f"{col}_sin_doy"] = np.sin(rad_doy).astype(np.float32)
                target_df[f"{col}_cos_doy"] = np.cos(rad_doy).astype(np.float32)

                rad_month = 2 * np.pi * dt_series.dt.month.fillna(6).values / 12.0
                target_df[f"{col}_sin_month"] = np.sin(rad_month).astype(np.float32)
                target_df[f"{col}_cos_month"] = np.cos(rad_month).astype(np.float32)

    # Identify Spatial Coordinates
    lat_col = None
    lon_col = None
    for col in df_tr.columns:
        cl = col.lower()
        if cl in [
            "lat",
            "latitude",
            "y",
            "cent_lat",
            "centroid_y",
            "perimeter_lat",
        ]:
            lat_col = col
        elif cl in [
            "lon",
            "long",
            "longitude",
            "x",
            "cent_lon",
            "centroid_x",
            "perimeter_lon",
        ]:
            lon_col = col

    if lat_col and lon_col:
        lat_med = df_tr[lat_col].median()
        lon_med = df_tr[lon_col].median()

        for df in [df_tr, df_te]:
            lat_vals = df[lat_col].fillna(lat_med).values
            lon_vals = df[lon_col].fillna(lon_med).values
            lat_rad = np.radians(lat_vals)
            lon_rad = np.radians(lon_vals)

            # 3D Earth spherical coordinates
            df["coord_x_3d"] = (np.cos(lat_rad) * np.cos(lon_rad)).astype(np.float32)
            df["coord_y_3d"] = (np.cos(lat_rad) * np.sin(lon_rad)).astype(np.float32)
            df["coord_z_3d"] = np.sin(lat_rad).astype(np.float32)

            # Rotated 2D coordinates
            for angle_deg in [15, 30, 45, 60, 75]:
                rad = np.radians(angle_deg)
                df[f"coord_rot_{angle_deg}"] = (
                    lat_vals * np.cos(rad) - lon_vals * np.sin(rad)
                ).astype(np.float32)

            # Distance to major wildfire center points
            centers = {
                "ca_center": (37.0, -120.0),
                "pnw_center": (44.0, -120.5),
                "sw_center": (34.0, -111.0),
                "rockies_center": (40.0, -106.0),
                "se_center": (33.0, -84.0),
            }
            for name, (c_lat, c_lon) in centers.items():
                df[f"dist_{name}"] = np.sqrt(
                    (lat_vals - c_lat) ** 2 + (lon_vals - c_lon) ** 2
                ).astype(np.float32)

            # Polynomial spatial interactions
            df["lat_x_lon"] = (lat_vals * lon_vals).astype(np.float32)
            df["lat_sq"] = (lat_vals**2).astype(np.float32)
            df["lon_sq"] = (lon_vals**2).astype(np.float32)
            df["lat_lon_ratio"] = (lat_vals / (np.abs(lon_vals) + 1e-5)).astype(
                np.float32
            )
            df["dist_origin"] = np.sqrt(lat_vals**2 + lon_vals**2).astype(
                np.float32
            )

        # Spatial KMeans Clustering fit purely on training
        km = KMeans(n_clusters=12, random_state=42, n_init=10)
        tr_coords = np.column_stack(
            [
                df_tr[lat_col].fillna(lat_med),
                df_tr[lon_col].fillna(lon_med),
            ]
        )
        te_coords = np.column_stack(
            [
                df_te[lat_col].fillna(lat_med),
                df_te[lon_col].fillna(lon_med),
            ]
        )

        df_tr["spatial_cluster"] = km.fit_predict(tr_coords)
        df_te["spatial_cluster"] = km.predict(te_coords)

        tr_dist_mat = km.transform(tr_coords)
        te_dist_mat = km.transform(te_coords)
        df_tr["min_dist_to_cluster"] = tr_dist_mat.min(axis=1).astype(np.float32)
        df_te["min_dist_to_cluster"] = te_dist_mat.min(axis=1).astype(np.float32)

    # Shape / Perimeter / Geometric features
    shape_length_col = None
    shape_area_col = None
    for col in df_tr.columns:
        cl = col.lower()
        if "length" in cl or "shape_len" in cl or "perim" in cl:
            shape_length_col = col
        if "area" in cl or "shape_area" in cl:
            shape_area_col = col

    for df in [df_tr, df_te]:
        if shape_length_col and shape_area_col:
            sl = np.maximum(df[shape_length_col].fillna(0).values, 0)
            sa = np.maximum(df[shape_area_col].fillna(0).values, 0)
            df["shape_compactness"] = ((4 * np.pi * sa) / (sl**2 + 1e-6)).astype(
                np.float32
            )
            df["shape_len_to_area"] = (sl / (np.sqrt(sa) + 1e-6)).astype(np.float32)
            df["isoperimetric_quotient"] = ((sl**2) / (sa + 1e-6)).astype(
                np.float32
            )
            df["area_to_perim"] = (sa / (sl + 1e-6)).astype(np.float32)
            df["perim_to_area"] = (sl / (sa + 1e-6)).astype(np.float32)
            df["sqrt_area_to_perim"] = (np.sqrt(sa) / (sl + 1e-6)).astype(
                np.float32
            )
            df["log_shape_len"] = np.log1p(sl).astype(np.float32)
            df["log_shape_area"] = np.log1p(sa).astype(np.float32)
            df["log_iso_quotient"] = np.log1p(sl**2 / (sa + 1e-6)).astype(
                np.float32
            )
            df["log_area_perim_ratio"] = np.log1p(sa / (sl + 1e-6)).astype(
                np.float32
            )
        elif shape_length_col:
            sl = np.maximum(df[shape_length_col].fillna(0).values, 0)
            df["log_shape_len"] = np.log1p(sl).astype(np.float32)

    # Spatial-hierarchical group aggregations
    agg_cols = [
        c
        for c in [shape_length_col, shape_area_col, lat_col, lon_col]
        if c is not None and c in df_tr.columns
    ]
    group_keys = [k for k in ["spatial_cluster"] if k in df_tr.columns]
    for col in df_train.columns:
        cl = col.lower()
        if (
            any(
                k in cl
                for k in [
                    "state",
                    "agency",
                    "unit",
                    "region",
                    "forest",
                    "dispatch",
                    "year",
                ]
            )
            and col not in group_keys
            and col in df_tr.columns
        ):
            group_keys.append(col)

    for g_key in group_keys:
        for a_col in agg_cols:
            mean_map = df_tr.groupby(g_key)[a_col].mean().to_dict()
            std_map = df_tr.groupby(g_key)[a_col].std().to_dict()
            min_map = df_tr.groupby(g_key)[a_col].min().to_dict()
            max_map = df_tr.groupby(g_key)[a_col].max().to_dict()

            tr_global_mean = float(df_tr[a_col].mean())
            tr_global_std = (
                float(df_tr[a_col].std()) if df_tr[a_col].std() > 0 else 1.0
            )

            for df in [df_tr, df_te]:
                g_mean = (
                    df[g_key]
                    .map(mean_map)
                    .fillna(tr_global_mean)
                    .astype(np.float32)
                )
                g_std = (
                    df[g_key]
                    .map(std_map)
                    .fillna(tr_global_std)
                    .astype(np.float32)
                )
                g_min = (
                    df[g_key].map(min_map).fillna(df[a_col]).astype(np.float32)
                )
                g_max = (
                    df[g_key].map(max_map).fillna(df[a_col]).astype(np.float32)
                )

                df[f"{g_key}_{a_col}_mean"] = g_mean
                df[f"{g_key}_{a_col}_std"] = g_std
                df[f"{g_key}_{a_col}_min"] = g_min
                df[f"{g_key}_{a_col}_max"] = g_max

                df[f"{a_col}_diff_{g_key}_mean"] = (df[a_col] - g_mean).astype(
                    np.float32
                )
                df[f"{a_col}_ratio_{g_key}_mean"] = (
                    df[a_col] / (g_mean.abs() + 1e-5)
                ).astype(np.float32)

    # Categorical & Text Features
    cat_cols = []
    for col in df_tr.columns:
        if (
            df_tr[col].dtype == "object"
            or df_tr[col].dtype.name == "category"
            or col in ["spatial_cluster"]
        ):
            cat_cols.append(col)

    for col in cat_cols:
        if df_tr[col].dtype == "object":
            df_tr[f"{col}_len"] = (
                df_tr[col].astype(str).str.len().fillna(0).astype(np.float32)
            )
            df_te[f"{col}_len"] = (
                df_te[col].astype(str).str.len().fillna(0).astype(np.float32)
            )

            for kw in ["COMPLEX", "LIGHTNING", "CREEK", "RIDGE", "FIRE", "RX"]:
                df_tr[f"{col}_has_{kw.lower()}"] = (
                    df_tr[col]
                    .astype(str)
                    .str.upper()
                    .str.contains(kw)
                    .astype(np.float32)
                )
                df_te[f"{col}_has_{kw.lower()}"] = (
                    df_te[col]
                    .astype(str)
                    .str.upper()
                    .str.contains(kw)
                    .astype(np.float32)
                )

        freq_map = df_tr[col].value_counts(normalize=True).to_dict()
        df_tr[f"{col}_freq"] = df_tr[col].map(freq_map).fillna(0.0).astype(np.float32)
        df_te[f"{col}_freq"] = df_te[col].map(freq_map).fillna(0.0).astype(np.float32)

        le = LabelEncoder()
        tr_s = df_tr[col].astype(str).fillna("__NA__")
        te_s = df_te[col].astype(str).fillna("__NA__")

        le.fit(tr_s)
        known_labels = set(le.classes_)
        df_tr[col] = le.transform(tr_s).astype(np.float32)
        te_mapped = te_s.apply(
            lambda x: le.transform([x])[0] if x in known_labels else -1
        ).astype(np.float32)
        df_te[col] = te_mapped

    # Drop non-numeric leftover raw date columns
    for col in date_cols:
        if col in df_tr.columns:
            df_tr = df_tr.drop(columns=[col])
        if col in df_te.columns:
            df_te = df_te.drop(columns=[col])

    # Convert all columns to float32 and handle missing values
    for col in df_tr.columns:
        if df_tr[col].dtype == "object":
            df_tr[col] = pd.to_numeric(df_tr[col], errors="coerce").fillna(0)
            df_te[col] = pd.to_numeric(df_te[col], errors="coerce").fillna(0)
        med_val = df_tr[col].median()
        df_tr[col] = df_tr[col].fillna(med_val).astype(np.float32)
        df_te[col] = df_te[col].fillna(med_val).astype(np.float32)

    return df_tr, df_te


X_train_df, X_test_df = extract_fire_features(train_features_raw, test_features_raw)
feature_names = [col for col in X_train_df.columns if col in X_test_df.columns]
X_train_df = X_train_df[feature_names]
X_test_df = X_test_df[feature_names]

X_train = X_train_df.values.astype(np.float32)
X_test = X_test_df.values.astype(np.float32)

# 5-Fold Stratified CV on binned log-target
log_y = np.log1p(np.maximum(y_train, 0))
n_bins = 10
bins = np.quantile(log_y, np.linspace(0, 1, n_bins + 1))
bins[0] -= 1e-5
bins[-1] += 1e-5
binned_target = np.digitize(log_y, bins)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_assignments = np.zeros(len(y_train), dtype=int)
for fold, (_, val_idx) in enumerate(skf.split(X_train, binned_target)):
    fold_assignments[val_idx] = fold


# =========================================================================
# 2. Model Definitions
# =========================================================================
class TabularResidualBlock(nn.Module):

    def __init__(self, hidden_dim: int, dropout_rate: float = 0.15):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class TabularResNet(nn.Module):

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = [256, 128, 64],
        dropout_rate: float = 0.15,
        output_dim: int = 1,
    ):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.SiLU(),
        )

        layers = []
        for i in range(len(hidden_dims) - 1):
            layers.append(
                TabularResidualBlock(hidden_dims[i], dropout_rate=dropout_rate)
            )
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            layers.append(nn.LayerNorm(hidden_dims[i + 1]))
            layers.append(nn.SiLU())

        layers.append(TabularResidualBlock(hidden_dims[-1], dropout_rate=dropout_rate))
        self.residual_stack = nn.Sequential(*layers)

        self.head = nn.Sequential(
            nn.Dropout(dropout_rate / 2),
            nn.Linear(hidden_dims[-1], output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_layer(x)
        h = self.residual_stack(h)
        return self.head(h)


def build_pytorch_components(
    input_dim: int,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    total_epochs: int = 40,
    steps_per_epoch: int = 100,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    model = TabularResNet(
        input_dim=input_dim, hidden_dims=[256, 128, 64], dropout_rate=0.15
    ).to(device)

    criterion = nn.SmoothL1Loss(beta=1.0)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(
        optimizer, T_max=max(total_epochs * steps_per_epoch, 1), eta_min=1e-6
    )

    return {
        "model": model,
        "criterion": criterion,
        "optimizer": optimizer,
        "scheduler": scheduler,
    }


def build_lightgbm_model(custom_params: dict = None) -> lgb.LGBMRegressor:
    params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "n_estimators": 1500,
        "learning_rate": 0.03,
        "num_leaves": 63,
        "max_depth": -1,
        "min_child_samples": 20,
        "subsample": 0.85,
        "subsample_freq": 1,
        "colsample_bytree": 0.80,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }
    if custom_params:
        params.update(custom_params)
    return lgb.LGBMRegressor(**params)


def build_xgboost_model(custom_params: dict = None) -> xgb.XGBRegressor:
    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "n_estimators": 1500,
        "learning_rate": 0.03,
        "max_depth": 7,
        "min_child_weight": 3,
        "subsample": 0.85,
        "colsample_bytree": 0.80,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": -1,
        "early_stopping_rounds": 50,
    }
    if custom_params:
        params.update(custom_params)
    return xgb.XGBRegressor(**params)


def build_catboost_model(custom_params: dict = None) -> CatBoostRegressor:
    params = {
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "iterations": 1500,
        "learning_rate": 0.03,
        "depth": 6,
        "l2_leaf_reg": 3.0,
        "random_seed": 42,
        "verbose": False,
        "early_stopping_rounds": 50,
    }
    if custom_params:
        params.update(custom_params)
    return CatBoostRegressor(**params)


# =========================================================================
# 3. 5-Fold Cross-Validation Training & Inference
# =========================================================================
n_samples = len(y_train)
n_test = len(X_test)
n_folds = int(np.max(fold_assignments) + 1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

oof_lgb = np.zeros(n_samples, dtype=np.float64)
oof_xgb = np.zeros(n_samples, dtype=np.float64)
oof_cat = np.zeros(n_samples, dtype=np.float64)
oof_nn = np.zeros(n_samples, dtype=np.float64)

test_preds_lgb = np.zeros(n_test, dtype=np.float64)
test_preds_xgb = np.zeros(n_test, dtype=np.float64)
test_preds_cat = np.zeros(n_test, dtype=np.float64)
test_preds_nn = np.zeros(n_test, dtype=np.float64)

for fold in range(n_folds):
    tr_idx = fold_assignments != fold
    va_idx = fold_assignments == fold

    X_tr_fold, y_tr_fold = X_train[tr_idx], y_train[tr_idx]
    X_va_fold, y_va_fold = X_train[va_idx], y_train[va_idx]

    # Model A: LightGBM Regressor
    lgb_model = build_lightgbm_model({"random_state": 42 + fold})
    lgb_callbacks = [
        lgb.early_stopping(stopping_rounds=50, verbose=False),
        lgb.log_evaluation(period=0),
    ]
    lgb_model.fit(
        X_tr_fold,
        y_tr_fold,
        eval_set=[(X_va_fold, y_va_fold)],
        callbacks=lgb_callbacks,
    )
    val_pred_lgb = lgb_model.predict(X_va_fold)
    oof_lgb[va_idx] = val_pred_lgb
    test_preds_lgb += lgb_model.predict(X_test) / n_folds

    # Model B: XGBoost Regressor
    xgb_model = build_xgboost_model(
        {"random_state": 42 + fold, "early_stopping_rounds": 50}
    )
    xgb_model.fit(
        X_tr_fold, y_tr_fold, eval_set=[(X_va_fold, y_va_fold)], verbose=False
    )
    val_pred_xgb = xgb_model.predict(X_va_fold)
    oof_xgb[va_idx] = val_pred_xgb
    test_preds_xgb += xgb_model.predict(X_test) / n_folds

    # Model C: CatBoost Regressor
    cat_model = build_catboost_model({"random_seed": 42 + fold})
    cat_model.fit(
        X_tr_fold,
        y_tr_fold,
        eval_set=(X_va_fold, y_va_fold),
        use_best_model=True,
        verbose=False,
    )
    val_pred_cat = cat_model.predict(X_va_fold)
    oof_cat[va_idx] = val_pred_cat
    test_preds_cat += cat_model.predict(X_test) / n_folds

    # Model D: TabularResNet
    scaler_x = StandardScaler()
    X_tr_nn = scaler_x.fit_transform(X_tr_fold)
    X_va_nn = scaler_x.transform(X_va_fold)
    X_te_nn = scaler_x.transform(X_test)

    scaler_y = StandardScaler()
    y_tr_nn = scaler_y.fit_transform(y_tr_fold.reshape(-1, 1)).flatten()
    y_va_nn = scaler_y.transform(y_va_fold.reshape(-1, 1)).flatten()

    batch_size = 256
    train_dataset = TensorDataset(
        torch.tensor(X_tr_nn, dtype=torch.float32),
        torch.tensor(y_tr_nn, dtype=torch.float32).unsqueeze(1),
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=False
    )

    epochs = 35
    steps_per_epoch = len(train_loader)
    nn_components = build_pytorch_components(
        input_dim=X_tr_nn.shape[1],
        lr=1e-3,
        weight_decay=1e-4,
        total_epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        device=device,
    )
    nn_model = nn_components["model"]
    criterion = nn_components["criterion"]
    optimizer = nn_components["optimizer"]
    scheduler = nn_components["scheduler"]

    best_val_loss = float("inf")
    best_weights = copy.deepcopy(nn_model.state_dict())

    X_va_tensor = torch.tensor(X_va_nn, dtype=torch.float32).to(device)
    y_va_tensor = torch.tensor(y_va_nn, dtype=torch.float32).unsqueeze(1).to(device)

    for epoch in range(epochs):
        nn_model.train()
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            preds = nn_model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(nn_model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

        nn_model.eval()
        with torch.no_grad():
            val_preds_scaled = nn_model(X_va_tensor)
            val_loss = criterion(val_preds_scaled, y_va_tensor).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = copy.deepcopy(nn_model.state_dict())

    nn_model.load_state_dict(best_weights)
    nn_model.eval()
    with torch.no_grad():
        val_pred_scaled_best = nn_model(X_va_tensor).cpu().numpy()
        val_pred_nn = scaler_y.inverse_transform(val_pred_scaled_best).flatten()
        oof_nn[va_idx] = val_pred_nn

        X_te_tensor = torch.tensor(X_te_nn, dtype=torch.float32).to(device)
        test_pred_scaled = nn_model(X_te_tensor).cpu().numpy()
        test_pred_fold_nn = scaler_y.inverse_transform(test_pred_scaled).flatten()
        test_preds_nn += test_pred_fold_nn / n_folds

    fold_lgb_rmse = np.sqrt(mean_squared_error(y_va_fold, val_pred_lgb))
    fold_xgb_rmse = np.sqrt(mean_squared_error(y_va_fold, val_pred_xgb))
    fold_cat_rmse = np.sqrt(mean_squared_error(y_va_fold, val_pred_cat))
    fold_nn_rmse = np.sqrt(mean_squared_error(y_va_fold, val_pred_nn))
    print(
        f"Fold {fold+1}/{n_folds} - LGB RMSE: {fold_lgb_rmse:.4f} | XGB RMSE: {fold_xgb_rmse:.4f} | CAT RMSE: {fold_cat_rmse:.4f} | ResNet RMSE: {fold_nn_rmse:.4f}"
    )


# =========================================================================
# 4. Out-Of-Fold Optimization & Final Predictions
# =========================================================================
def ensemble_loss_fn(weights, oof_matrix, y_true):
    weights = np.array(weights)
    blended = np.dot(oof_matrix, weights)
    blended = np.clip(blended, 0.0, None)
    return np.sqrt(mean_squared_error(y_true, blended))


oof_stack = np.column_stack([oof_lgb, oof_xgb, oof_cat, oof_nn])
initial_weights = [0.25, 0.25, 0.25, 0.25]
bounds = [(0.0, 1.0)] * 4
constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

opt_res = minimize(
    ensemble_loss_fn,
    initial_weights,
    args=(oof_stack, y_train),
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
)

best_weights = (
    opt_res.x
    if opt_res.success
    else np.array([0.35, 0.35, 0.25, 0.05], dtype=np.float64)
)
best_weights = best_weights / np.sum(best_weights)

final_oof_preds = np.dot(oof_stack, best_weights)
final_oof_preds = np.clip(final_oof_preds, 0.0, None)
final_validation_rmse = np.sqrt(mean_squared_error(y_train, final_oof_preds))

test_stack = np.column_stack(
    [test_preds_lgb, test_preds_xgb, test_preds_cat, test_preds_nn]
)
final_test_preds = np.dot(test_stack, best_weights)
final_test_preds = np.clip(final_test_preds, 0.0, None)

submission_df = pd.DataFrame({"id": test_ids, "gisacres": final_test_preds})

assert (
    len(submission_df) == n_test
), f"Row mismatch: expected {n_test}, got {len(submission_df)}"
assert not submission_df.isnull().any().any(), "Submission contains null values."
assert (submission_df["gisacres"] >= 0).all(), "Submission contains negative values."

sub_path = "./submission/submission.csv"
submission_df.to_csv(sub_path, index=False)

print(f"Final Validation Score: {final_validation_rmse}")