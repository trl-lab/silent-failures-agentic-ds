import os
import random
import warnings
from catboost import CatBoostRegressor
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.cluster import KMeans
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb

warnings.filterwarnings("ignore")


# ==============================================================================
# 0. Reproducibility
# ==============================================================================
def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ==============================================================================
# 1. Tabular Neural Network Architecture
# ==============================================================================
class TabularResidualBlock(nn.Module):

    def __init__(self, hidden_dim: int, dropout_rate: float = 0.15):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class TabularResNet(nn.Module):

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_blocks: int = 3,
        dropout_rate: float = 0.15,
    ):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
        )
        self.res_blocks = nn.ModuleList(
            [
                TabularResidualBlock(hidden_dim=hidden_dim, dropout_rate=dropout_rate)
                for _ in range(num_blocks)
            ]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout_rate / 2.0),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_layer(x)
        for block in self.res_blocks:
            h = block(h)
        return self.head(h).squeeze(-1)


# ==============================================================================
# 2. Feature Engineering Pipeline
# ==============================================================================
def process_data():
    train_path = "./input/train.parquet"
    test_path = "./input/test.parquet"

    df_train = pd.read_parquet(train_path)
    df_test = pd.read_parquet(test_path)

    target_col = "Res_capacityM3"
    if target_col not in df_train.columns:
        for col in df_train.columns:
            if col.lower() == target_col.lower():
                df_train = df_train.rename(columns={col: target_col})
                break

    if "id" in df_test.columns:
        test_ids = df_test["id"].values
    else:
        test_ids = np.arange(len(df_test))
        df_test["id"] = test_ids

    if "id" not in df_train.columns:
        df_train["id"] = np.arange(len(df_train))

    y_train = df_train[target_col].copy()

    train_cols = [c for c in df_train.columns if c != target_col]
    test_cols = [c for c in df_test.columns]
    common_cols = [c for c in train_cols if c in test_cols and c != "id"]

    cat_cols = []
    num_cols = []

    for c in common_cols:
        s_tr_num = pd.to_numeric(df_train[c], errors="coerce")
        s_te_num = pd.to_numeric(df_test[c], errors="coerce")
        non_null_count = df_train[c].notna().sum()
        valid_ratio = s_tr_num.notna().sum() / max(1, non_null_count)

        if (
            df_train[c].dtype == "object"
            or df_train[c].dtype.name == "category"
            or df_train[c].dtype == "string"
        ):
            if valid_ratio > 0.85 and s_tr_num.notna().sum() > 0:
                df_train[c] = s_tr_num
                df_test[c] = s_te_num
                num_cols.append(c)
            else:
                cat_cols.append(c)
        else:
            if df_train[c].nunique() < 10 and not any(
                k in c.lower()
                for k in [
                    "lat",
                    "lon",
                    "hgt",
                    "area",
                    "elev",
                    "cap",
                    "dis",
                    "flow",
                    "vol",
                    "len",
                    "catch",
                    "year",
                    "yr",
                ]
            ):
                cat_cols.append(c)
            else:
                df_train[c] = s_tr_num
                df_test[c] = s_te_num
                num_cols.append(c)

    lat_cols = [
        c
        for c in num_cols
        if any(k in c.lower() for k in ["latitude", "lat_dd", "lat_deg"])
        or c.lower() in ["lat", "latitude_dd"]
    ]
    lon_cols = [
        c
        for c in num_cols
        if any(
            k in c.lower()
            for k in ["longitude", "long_dd", "lon_dd", "lon_deg", "long_deg"]
        )
        or c.lower() in ["lon", "long", "longitude_dd"]
    ]

    lat_name = lat_cols[0] if lat_cols else None
    lon_name = lon_cols[0] if lon_cols else None

    df_train["is_train"] = 1
    df_test["is_train"] = 0
    combined = pd.concat(
        [df_train[common_cols + ["is_train"]], df_test[common_cols + ["is_train"]]],
        axis=0,
    ).reset_index(drop=True)

    if lat_name and lon_name:
        lat = pd.to_numeric(combined[lat_name], errors="coerce")
        lon = pd.to_numeric(combined[lon_name], errors="coerce")
        lat = lat.fillna(lat.median() if not np.isnan(lat.median()) else 0.0)
        lon = lon.fillna(lon.median() if not np.isnan(lon.median()) else 0.0)
        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)

        combined["geo_x"] = np.cos(lat_rad) * np.cos(lon_rad)
        combined["geo_y"] = np.cos(lat_rad) * np.sin(lon_rad)
        combined["geo_z"] = np.sin(lat_rad)

        for angle in [15, 30, 45, 60, 75]:
            rad = np.radians(angle)
            combined[f"geo_rot_{angle}"] = lat * np.cos(rad) + lon * np.sin(rad)
            combined[f"geo_rot_ortho_{angle}"] = -lat * np.sin(rad) + lon * np.cos(rad)

        combined["dist_equator"] = np.abs(lat) * 111.0
        combined["dist_prime_meridian"] = np.abs(lon) * 111.0 * np.cos(lat_rad)

        coords = np.column_stack([lat.values, lon.values])
        k_neighbors = min(7, len(combined))
        if k_neighbors > 1:
            nn_model = NearestNeighbors(n_neighbors=k_neighbors, metric="euclidean")
            nn_model.fit(coords)
            distances, indices = nn_model.kneighbors(coords)
            combined["spatial_nn_dist_mean"] = distances[:, 1:].mean(axis=1)
            combined["spatial_nn_dist_min"] = distances[:, 1:].min(axis=1)

    area_cols = [
        c
        for c in num_cols
        if any(k in c.lower() for k in ["area", "skm", "res_area", "surf"])
    ]
    hgt_cols = [
        c
        for c in num_cols
        if any(k in c.lower() for k in ["hgt", "height", "dam_hgt", "depth"])
    ]
    len_cols = [
        c for c in num_cols if any(k in c.lower() for k in ["len", "length", "crest"])
    ]
    catch_cols = [
        c
        for c in num_cols
        if any(k in c.lower() for k in ["catch", "basin_area", "catch_skm"])
    ]
    flow_cols = [
        c
        for c in num_cols
        if any(k in c.lower() for k in ["dis", "flow", "discharge", "runoff"])
    ]
    elev_cols = [
        c
        for c in num_cols
        if any(k in c.lower() for k in ["elev", "elevation", "masl"])
    ]
    year_cols = [
        c
        for c in num_cols
        if any(k in c.lower() for k in ["year", "yr", "built", "complete"])
    ]

    if lat_name and lon_name and k_neighbors > 1:
        knn_features = (area_cols + hgt_cols + elev_cols + flow_cols + catch_cols)[:5]
        for kf in knn_features:
            vals = pd.to_numeric(combined[kf], errors="coerce").fillna(0.0).values
            nbr_vals = vals[indices[:, 1:]]
            combined[f"spatial_knn_mean_{kf}"] = np.nanmean(nbr_vals, axis=1)
            combined[f"spatial_knn_std_{kf}"] = np.nanstd(nbr_vals, axis=1)
            combined[f"spatial_knn_diff_{kf}"] = (
                pd.to_numeric(combined[kf], errors="coerce").fillna(0.0)
                - combined[f"spatial_knn_mean_{kf}"]
            )

    for ac in area_cols:
        for hc in hgt_cols:
            ac_clip = combined[ac].clip(lower=0)
            hc_clip = combined[hc].clip(lower=0)
            combined[f"geom_vol_prism_{ac}_{hc}"] = ac_clip * hc_clip
            combined[f"geom_vol_conical_{ac}_{hc}"] = (ac_clip * hc_clip) / 3.0
            combined[f"geom_vol_parabolic_{ac}_{hc}"] = 0.4 * ac_clip * hc_clip
            combined[f"geom_log_vol_proxy_{ac}_{hc}"] = np.log1p(ac_clip) + np.log1p(
                hc_clip
            )
            combined[f"aspect_area_hgt_ratio_{ac}_{hc}"] = combined[ac] / (
                combined[hc] + 1e-4
            )
            for fc in flow_cols:
                fc_clip = combined[fc].clip(lower=0)
                combined[f"hydr_residence_time_{ac}_{hc}_{fc}"] = (
                    ac_clip * hc_clip
                ) / (fc_clip + 1e-4)

    for hc in hgt_cols:
        for lc in len_cols:
            combined[f"aspect_hgt_len_ratio_{hc}_{lc}"] = combined[hc] / (
                combined[lc] + 1e-4
            )
            combined[f"aspect_dam_slenderness_{lc}_{hc}"] = combined[lc] / (
                combined[hc] + 1e-4
            )
            combined[f"geom_dam_face_area_{hc}_{lc}"] = combined[hc].clip(
                lower=0
            ) * combined[lc].clip(lower=0)

    for ac in area_cols:
        for cc in catch_cols:
            combined[f"hydr_area_catch_ratio_{ac}_{cc}"] = combined[ac] / (
                combined[cc] + 1e-4
            )
            combined[f"hydr_log_catch_area_diff_{ac}_{cc}"] = np.log1p(
                combined[cc].clip(lower=0)
            ) - np.log1p(combined[ac].clip(lower=0))

    for fc in flow_cols:
        for cc in catch_cols:
            combined[f"hydr_discharge_yield_{fc}_{cc}"] = combined[fc] / (
                combined[cc] + 1e-4
            )

    for yc in year_cols:
        combined[f"{yc}_age"] = 2025.0 - combined[yc]
        combined[f"{yc}_decade"] = (combined[yc] // 10) * 10
        combined[f"{yc}_era_pre1950"] = (combined[yc] < 1950).astype(float)
        combined[f"{yc}_era_1950_1990"] = (
            (combined[yc] >= 1950) & (combined[yc] < 1990)
        ).astype(float)
        combined[f"{yc}_era_post1990"] = (combined[yc] >= 1990).astype(float)

    numeric_for_log = list(
        set(
            num_cols
            + area_cols
            + hgt_cols
            + len_cols
            + catch_cols
            + flow_cols
            + elev_cols
        )
    )
    for nc in numeric_for_log:
        if nc in combined.columns and pd.api.types.is_numeric_dtype(combined[nc]):
            col_vals = pd.to_numeric(combined[nc], errors="coerce")
            min_val = col_vals.min()
            if pd.isna(min_val):
                min_val = 0.0
            if min_val >= 0:
                combined[f"log1p_{nc}"] = np.log1p(col_vals.clip(lower=0))
            else:
                combined[f"signed_log1p_{nc}"] = np.sign(col_vals) * np.log1p(
                    np.abs(col_vals)
                )
            combined[f"is_na_{nc}"] = col_vals.isna().astype(float)

    for cc in cat_cols:
        combined[cc] = combined[cc].astype(str).fillna("__MISSING__")
        freq = combined[cc].value_counts(normalize=True)
        combined[f"freq_{cc}"] = combined[cc].map(freq).astype(float)

        for agg_num in (area_cols + hgt_cols + catch_cols + elev_cols)[:4]:
            if agg_num in combined.columns and pd.api.types.is_numeric_dtype(combined[agg_num]):
                grp_mean = combined.groupby(cc)[agg_num].transform("mean")
                combined[f"grp_mean_{agg_num}_by_{cc}"] = pd.to_numeric(
                    grp_mean, errors="coerce"
                ).fillna(0.0)
                combined[f"grp_diff_{agg_num}_by_{cc}"] = (
                    combined[agg_num] - combined[f"grp_mean_{agg_num}_by_{cc}"]
                )

    train_proc = (
        combined[combined["is_train"] == 1]
        .drop(columns=["is_train"])
        .reset_index(drop=True)
    )
    test_proc = (
        combined[combined["is_train"] == 0]
        .drop(columns=["is_train"])
        .reset_index(drop=True)
    )

    train_proc[target_col] = y_train.values
    test_proc["id"] = test_ids

    n_splits = 5
    log_target = np.log1p(np.clip(y_train.values, 0, None))
    num_unique = len(np.unique(log_target))
    n_bins = max(2, min(10, num_unique))
    try:
        bins = pd.qcut(
            log_target,
            q=n_bins,
            labels=False,
            duplicates="drop",
        )
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        splits = list(skf.split(train_proc, bins))
    except Exception:
        from sklearn.model_selection import KFold

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        splits = list(kf.split(train_proc))

    train_proc["fold"] = -1
    for fold, (_, val_idx) in enumerate(splits):
        train_proc.loc[val_idx, "fold"] = fold

    high_card_cats = [c for c in cat_cols if train_proc[c].nunique() > 1]
    overall_mean = train_proc[target_col].mean()
    smoothing_m = 10.0
    for cc in high_card_cats:
        target_enc_col = f"oof_target_enc_{cc}"
        train_proc[target_enc_col] = np.nan

        for fold in range(n_splits):
            trn_mask = train_proc["fold"] != fold
            val_mask = train_proc["fold"] == fold
            stats = (
                train_proc.loc[trn_mask]
                .groupby(cc)[target_col]
                .agg(["count", "mean"])
            )
            smooth_enc = (
                stats["count"] * stats["mean"] + smoothing_m * overall_mean
            ) / (stats["count"] + smoothing_m)
            train_proc.loc[val_mask, target_enc_col] = train_proc.loc[val_mask, cc].map(
                smooth_enc
            )

        global_stats = train_proc.groupby(cc)[target_col].agg(["count", "mean"])
        global_smooth = (
            global_stats["count"] * global_stats["mean"] + smoothing_m * overall_mean
        ) / (global_stats["count"] + smoothing_m)
        train_proc[target_enc_col] = train_proc[target_enc_col].fillna(overall_mean)
        test_proc[target_enc_col] = test_proc[cc].map(global_smooth).fillna(overall_mean)

    for cc in cat_cols:
        combined_cats = pd.concat(
            [train_proc[cc].astype(str), test_proc[cc].astype(str)], axis=0
        ).unique()
        train_proc[cc] = pd.Categorical(
            train_proc[cc].astype(str), categories=combined_cats
        )
        test_proc[cc] = pd.Categorical(
            test_proc[cc].astype(str), categories=combined_cats
        )

    ignore_cols = [target_col, "fold", "id", "is_train"]
    feature_cols = [c for c in train_proc.columns if c not in ignore_cols]

    return train_proc, test_proc, feature_cols, cat_cols, target_col, n_splits


# ==============================================================================
# 3. Model Training Routines
# ==============================================================================
def train_lightgbm(train_df, test_df, feature_cols, cat_cols, target_col, n_splits):
    oof_preds = np.zeros(len(train_df))
    test_preds = np.zeros(len(test_df))

    lgb_params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "n_estimators": 2500,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": 6,
        "min_child_samples": 20,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }

    for fold in range(n_splits):
        trn_idx = train_df["fold"] != fold
        val_idx = train_df["fold"] == fold

        X_tr = train_df.loc[trn_idx, feature_cols].copy()
        y_tr = train_df.loc[trn_idx, target_col].values
        X_va = train_df.loc[val_idx, feature_cols].copy()
        y_va = train_df.loc[val_idx, target_col].values
        X_te = test_df[feature_cols].copy()

        for c in cat_cols:
            if c in X_tr.columns:
                X_tr[c] = X_tr[c].astype("category")
                X_va[c] = X_va[c].astype("category")
                X_te[c] = X_te[c].astype("category")

        model = lgb.LGBMRegressor(**lgb_params)
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )

        val_pred = model.predict(X_va)
        oof_preds[val_idx] = val_pred
        test_preds += model.predict(X_te) / n_splits

    return oof_preds, test_preds


def train_catboost(train_df, test_df, feature_cols, cat_cols, target_col, n_splits):
    oof_preds = np.zeros(len(train_df))
    test_preds = np.zeros(len(test_df))

    cb_cat_cols = [c for c in cat_cols if c in feature_cols]

    cb_params = {
        "iterations": 2500,
        "learning_rate": 0.03,
        "depth": 6,
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "random_seed": 42,
        "verbose": False,
        "early_stopping_rounds": 50,
        "task_type": "CPU",
    }

    for fold in range(n_splits):
        trn_idx = train_df["fold"] != fold
        val_idx = train_df["fold"] == fold

        X_tr = train_df.loc[trn_idx, feature_cols].copy()
        y_tr = train_df.loc[trn_idx, target_col].values
        X_va = train_df.loc[val_idx, feature_cols].copy()
        y_va = train_df.loc[val_idx, target_col].values
        X_te = test_df[feature_cols].copy()

        for c in cb_cat_cols:
            X_tr[c] = X_tr[c].astype(str)
            X_va[c] = X_va[c].astype(str)
            X_te[c] = X_te[c].astype(str)

        model = CatBoostRegressor(**cb_params)
        model.fit(
            X_tr,
            y_tr,
            eval_set=(X_va, y_va),
            cat_features=cb_cat_cols if len(cb_cat_cols) > 0 else None,
            verbose=False,
        )

        val_pred = model.predict(X_va)
        oof_preds[val_idx] = val_pred
        test_preds += model.predict(X_te) / n_splits

    return oof_preds, test_preds


def train_xgboost(train_df, test_df, feature_cols, cat_cols, target_col, n_splits):
    oof_preds = np.zeros(len(train_df))
    test_preds = np.zeros(len(test_df))

    xgb_params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "booster": "gbtree",
        "n_estimators": 2500,
        "learning_rate": 0.03,
        "max_depth": 6,
        "min_child_weight": 3,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": -1,
        "tree_method": "hist",
        "enable_categorical": True,
        "early_stopping_rounds": 50,
    }

    for fold in range(n_splits):
        trn_idx = train_df["fold"] != fold
        val_idx = train_df["fold"] == fold

        X_tr = train_df.loc[trn_idx, feature_cols].copy()
        y_tr = train_df.loc[trn_idx, target_col].values
        X_va = train_df.loc[val_idx, feature_cols].copy()
        y_va = train_df.loc[val_idx, target_col].values
        X_te = test_df[feature_cols].copy()

        for c in cat_cols:
            if c in X_tr.columns:
                X_tr[c] = X_tr[c].astype("category")
                X_va[c] = X_va[c].astype("category")
                X_te[c] = X_te[c].astype("category")

        model = xgb.XGBRegressor(**xgb_params)
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

        val_pred = model.predict(X_va)
        oof_preds[val_idx] = val_pred
        test_preds += model.predict(X_te) / n_splits

    return oof_preds, test_preds


def train_tabular_resnet(
    train_df, test_df, feature_cols, cat_cols, target_col, n_splits
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    oof_preds = np.zeros(len(train_df))
    test_preds = np.zeros(len(test_df))

    nn_features = [
        c
        for c in feature_cols
        if c not in cat_cols and train_df[c].dtype.name not in ["object", "category"]
    ]

    for fold in range(n_splits):
        trn_idx = train_df["fold"] != fold
        val_idx = train_df["fold"] == fold

        X_tr = train_df.loc[trn_idx, nn_features].copy()
        y_tr = train_df.loc[trn_idx, target_col].values.astype(np.float32)
        X_va = train_df.loc[val_idx, nn_features].copy()
        y_va = train_df.loc[val_idx, target_col].values.astype(np.float32)
        X_te = test_df[nn_features].copy()

        med = X_tr.median()
        X_tr = X_tr.replace([np.inf, -np.inf], np.nan).fillna(med).fillna(0.0)
        X_va = X_va.replace([np.inf, -np.inf], np.nan).fillna(med).fillna(0.0)
        X_te = X_te.replace([np.inf, -np.inf], np.nan).fillna(med).fillna(0.0)

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_va_s = scaler.transform(X_va)
        X_te_s = scaler.transform(X_te)

        y_scale = float(np.std(y_tr) + 1e-4)
        y_tr_s = y_tr / y_scale
        y_va_s = y_va / y_scale

        train_ds = TensorDataset(
            torch.tensor(X_tr_s, dtype=torch.float32),
            torch.tensor(y_tr_s, dtype=torch.float32),
        )
        val_ds = TensorDataset(
            torch.tensor(X_va_s, dtype=torch.float32),
            torch.tensor(y_va_s, dtype=torch.float32),
        )
        test_ds = TensorDataset(torch.tensor(X_te_s, dtype=torch.float32))

        train_loader = DataLoader(
            train_ds, batch_size=64, shuffle=True, drop_last=False
        )
        val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)

        model = TabularResNet(
            input_dim=len(nn_features), hidden_dim=128, num_blocks=3
        ).to(device)
        criterion = nn.MSELoss()
        optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=35, eta_min=1e-5)

        best_loss = float("inf")
        best_state = None
        patience = 8
        patience_counter = 0

        for epoch in range(35):
            model.train()
            for bx, by in train_loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                pred = model(bx)
                loss = criterion(pred, by)
                loss.backward()
                optimizer.step()
            scheduler.step()

            model.eval()
            val_loss = 0.0
            val_count = 0
            with torch.no_grad():
                for bx, by in val_loader:
                    bx, by = bx.to(device), by.to(device)
                    val_loss += criterion(model(bx), by).item() * len(by)
                    val_count += len(by)
            val_loss /= val_count

            if val_loss < best_loss:
                best_loss = val_loss
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
            model.to(device)

        model.eval()
        val_preds_list = []
        with torch.no_grad():
            for bx, _ in val_loader:
                bx = bx.to(device)
                val_preds_list.extend(model(bx).cpu().numpy().tolist())
        oof_preds[val_idx] = np.array(val_preds_list) * y_scale

        te_preds_list = []
        with torch.no_grad():
            for (bx,) in test_loader:
                bx = bx.to(device)
                te_preds_list.extend(model(bx).cpu().numpy().tolist())
        test_preds += (np.array(te_preds_list) * y_scale) / n_splits

    return oof_preds, test_preds


# ==============================================================================
# 4. Main Execution, Blending & Submission
# ==============================================================================
def main():
    seed_everything(42)

    train_df, test_df, feature_cols, cat_cols, target_col, n_splits = process_data()
    y_true = train_df[target_col].values

    lgb_oof, lgb_test = train_lightgbm(
        train_df, test_df, feature_cols, cat_cols, target_col, n_splits
    )
    cb_oof, cb_test = train_catboost(
        train_df, test_df, feature_cols, cat_cols, target_col, n_splits
    )
    xgb_oof, xgb_test = train_xgboost(
        train_df, test_df, feature_cols, cat_cols, target_col, n_splits
    )
    nn_oof, nn_test = train_tabular_resnet(
        train_df, test_df, feature_cols, cat_cols, target_col, n_splits
    )

    lgb_oof = np.clip(lgb_oof, 0, None)
    lgb_test = np.clip(lgb_test, 0, None)
    cb_oof = np.clip(cb_oof, 0, None)
    cb_test = np.clip(cb_test, 0, None)
    xgb_oof = np.clip(xgb_oof, 0, None)
    xgb_test = np.clip(xgb_test, 0, None)
    nn_oof = np.clip(nn_oof, 0, None)
    nn_test = np.clip(nn_test, 0, None)

    oof_matrix = np.column_stack([lgb_oof, cb_oof, xgb_oof, nn_oof])
    test_matrix = np.column_stack([lgb_test, cb_test, xgb_test, nn_test])

    def loss_func(weights):
        pred = np.dot(oof_matrix, weights)
        return np.sqrt(mean_squared_error(y_true, pred))

    num_models = 4
    init_weights = np.array([0.35, 0.35, 0.25, 0.05])
    bounds = [(0, 1) for _ in range(num_models)]
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

    res = minimize(
        loss_func,
        init_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    if res.success and np.all(res.x >= 0):
        opt_weights = res.x / np.sum(res.x)
    else:
        opt_weights = np.array([0.35, 0.35, 0.30, 0.0])

    final_oof = np.clip(np.dot(oof_matrix, opt_weights), 0, None)
    final_test = np.clip(np.dot(test_matrix, opt_weights), 0, None)

    final_rmse = np.sqrt(mean_squared_error(y_true, final_oof))

    os.makedirs("./submission", exist_ok=True)
    test_ids = (
        test_df["id"].values if "id" in test_df.columns else np.arange(len(test_df))
    )
    submission = pd.DataFrame(
        {"id": test_ids.astype(int), "Res_capacityM3": final_test}
    )
    submission.to_csv("./submission/submission.csv", index=False)

    print(f"Final Validation Score: {final_rmse}")


if __name__ == "__main__":
    main()
