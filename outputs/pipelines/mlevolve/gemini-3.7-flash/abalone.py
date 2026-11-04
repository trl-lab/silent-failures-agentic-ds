import json
import math
import os
import warnings
from typing import Any, Dict, List, Tuple

import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import BayesianRidge, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb

warnings.filterwarnings("ignore")


# ==============================================================================
# 1. Feature Engineering & Preprocessing Utilities
# ==============================================================================
def standardize_col_names(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names to snake_case for consistency."""
    df = df.copy()
    rename_map = {}
    for col in df.columns:
        clean = col.strip().replace(" ", "_").replace("-", "_")
        rename_map[col] = clean
    return df.rename(columns=rename_map)


def identify_target_column(df: pd.DataFrame) -> str:
    """Identify the target column from known naming conventions."""
    for candidate in [
        "Class_number_of_rings",
        "class_number_of_rings",
        "Class_Rings",
        "rings",
        "Rings",
    ]:
        if candidate in df.columns:
            return candidate
    for col in df.columns:
        if "ring" in col.lower():
            return col
    raise ValueError(f"Could not identify target column among: {df.columns.tolist()}")


def clean_anomalies(
    df: pd.DataFrame, height_median: float = None
) -> Tuple[pd.DataFrame, float]:
    """Clean anatomical anomalies such as extreme height typos (> 0.5 mm) and zero heights."""
    df = df.copy()
    h_candidates = [c for c in df.columns if c.lower() == "height"]
    if h_candidates:
        h_col = h_candidates[0]
        df[h_col] = pd.to_numeric(df[h_col], errors="coerce").astype(float)
        # Cap extreme height measurement typos (> 0.5 mm)
        df.loc[df[h_col] > 0.5, h_col] = 0.5
        if height_median is None:
            valid_mask = (df[h_col] > 0) & (df[h_col] <= 0.5)
            height_median = float(df.loc[valid_mask, h_col].median())
        # Impute zero or negative heights
        df.loc[df[h_col] <= 0, h_col] = height_median
    return df, height_median


def engineer_features(
    df: pd.DataFrame, is_train: bool = True, group_stats: dict = None
) -> Tuple[pd.DataFrame, dict]:
    """
    Engineer comprehensive domain-specific morphometric, allometric,
    and interaction features for abalone age prediction.
    """
    data = df.copy()

    col_map = {c.lower(): c for c in data.columns}
    sex_col = col_map.get("sex", "Sex")
    len_col = col_map.get("length", "Length")
    dia_col = col_map.get("diameter", "Diameter")
    hgt_col = col_map.get("height", "Height")
    whole_col = col_map.get("whole_weight", col_map.get("whole weight", "Whole_weight"))
    shucked_col = col_map.get(
        "shucked_weight", col_map.get("shucked weight", "Shucked_weight")
    )
    viscera_col = col_map.get(
        "viscera_weight", col_map.get("viscera weight", "Viscera_weight")
    )
    shell_col = col_map.get("shell_weight", col_map.get("shell weight", "Shell_weight"))

    # Ensure proper data types to prevent Categorical dtype arithmetic errors
    data[sex_col] = data[sex_col].astype(str)
    for col in [len_col, dia_col, hgt_col, whole_col, shucked_col, viscera_col, shell_col]:
        data[col] = pd.to_numeric(data[col], errors="coerce").astype(float)

    # 1. Biological Sex features
    data["is_infant"] = (data[sex_col] == "I").astype(float)
    data["is_male"] = (data[sex_col] == "M").astype(float)
    data["is_female"] = (data[sex_col] == "F").astype(float)
    data["is_adult"] = 1.0 - data["is_infant"]

    # 2. Geometric & Volumetric features
    eps = 1e-6
    data["vol_box"] = data[len_col] * data[dia_col] * data[hgt_col]
    data["vol_ellipsoid"] = (
        (4.0 / 3.0)
        * np.pi
        * (data[len_col] / 2.0)
        * (data[dia_col] / 2.0)
        * (data[hgt_col] / 2.0)
    )
    data["vol_paraboloid"] = 0.5 * np.pi * (data[len_col] / 2.0) * (data[dia_col] / 2.0) * data[hgt_col]
    data["surface_area"] = 2.0 * (
        data[len_col] * data[dia_col]
        + data[len_col] * data[hgt_col]
        + data[dia_col] * data[hgt_col]
    )
    data["projected_area"] = data[len_col] * data[dia_col]

    # Aspect ratios & dome geometry
    data["ratio_len_dia"] = data[len_col] / (data[dia_col] + eps)
    data["ratio_len_hgt"] = data[len_col] / (data[hgt_col] + eps)
    data["ratio_dia_hgt"] = data[dia_col] / (data[hgt_col] + eps)
    data["diff_len_dia"] = data[len_col] - data[dia_col]
    data["dome_aspect_ratio"] = data[hgt_col] / np.sqrt(data[len_col] * data[dia_col] + eps)
    data["sphericity_index"] = (
        (np.pi ** (1.0 / 3.0))
        * ((6.0 * data["vol_ellipsoid"]) ** (2.0 / 3.0))
        / (data["surface_area"] + eps)
    )

    # 3. Mass balance and allometric ratios
    component_weight = data[shucked_col] + data[viscera_col] + data[shell_col]
    data["water_weight"] = np.clip(
        data[whole_col] - component_weight, a_min=0.0, a_max=None
    )
    data["water_ratio"] = data["water_weight"] / (data[whole_col] + eps)
    data["fluid_loss_fraction"] = data["water_weight"] / (data[whole_col] + eps)
    data["meat_yield"] = data[shucked_col] / (data[whole_col] + eps)
    data["shell_ratio"] = data[shell_col] / (data[whole_col] + eps)
    data["viscera_ratio"] = data[viscera_col] / (data[whole_col] + eps)

    # Anatomical cross-ratios
    data["ratio_shell_meat"] = data[shell_col] / (data[shucked_col] + eps)
    data["ratio_viscera_meat"] = data[viscera_col] / (data[shucked_col] + eps)
    data["ratio_viscera_shell"] = data[viscera_col] / (data[shell_col] + eps)
    data["fulton_condition"] = data[whole_col] / (data[len_col] ** 3 + eps)

    # 4. Density, Specific Volume & Shell Thickening Indicators
    data["density_whole"] = data[whole_col] / (data["vol_box"] + eps)
    data["density_shell"] = data[shell_col] / (data["vol_box"] + eps)
    data["density_meat"] = data[shucked_col] / (data["vol_box"] + eps)
    data["specific_vol_box"] = data["vol_box"] / (data[whole_col] + eps)
    data["specific_vol_ellipsoid"] = data["vol_ellipsoid"] / (data[whole_col] + eps)
    data["shell_thickening_index"] = data[shell_col] / (data["surface_area"] + eps)
    data["calcification_thickness_index"] = data[shell_col] / (data["projected_area"] + eps)
    data["shell_to_proj_area"] = data[shell_col] / (data["projected_area"] + eps)

    # Multi-scale polynomial interactions & non-linear power ratios
    data["shell_x_length"] = data[shell_col] * data[len_col]
    data["shell_x_diameter"] = data[shell_col] * data[dia_col]
    data["shell_x_height"] = data[shell_col] * data[hgt_col]
    data["shell_to_len_sq"] = data[shell_col] / (data[len_col] ** 2 + eps)
    data["shell_to_dia_sq"] = data[shell_col] / (data[dia_col] ** 2 + eps)
    data["shell_to_hgt_sq"] = data[shell_col] / (data[hgt_col] ** 2 + eps)

    # 5. Non-linear power law / Log transforms & Allometric power-law residuals
    raw_cols = [
        len_col,
        dia_col,
        hgt_col,
        whole_col,
        shucked_col,
        viscera_col,
        shell_col,
        "vol_box",
    ]
    for col in raw_cols:
        data[f"log_{col}"] = np.log1p(np.clip(data[col], a_min=0.0, a_max=None))
        data[f"sqrt_{col}"] = np.sqrt(np.clip(data[col], a_min=0.0, a_max=None))

    # Isometric scaling baseline: mass ~ dimension^3
    log_l = np.log(np.clip(data[len_col], a_min=eps, a_max=None))
    log_d = np.log(np.clip(data[dia_col], a_min=eps, a_max=None))
    log_w = np.log(np.clip(data[whole_col], a_min=eps, a_max=None))
    log_s = np.log(np.clip(data[shell_col], a_min=eps, a_max=None))
    log_m = np.log(np.clip(data[shucked_col], a_min=eps, a_max=None))
    log_v = np.log(np.clip(data[viscera_col], a_min=eps, a_max=None))

    data["allometric_res_whole_len"] = log_w - 3.0 * log_l
    data["allometric_res_shell_len"] = log_s - 3.0 * log_l
    data["allometric_res_meat_len"] = log_m - 3.0 * log_l
    data["allometric_res_viscera_len"] = log_v - 3.0 * log_l
    data["allometric_res_whole_dia"] = log_w - 3.0 * log_d
    data["allometric_res_shell_dia"] = log_s - 3.0 * log_d

    # 6. Interaction with Infant status
    data["infant_x_shell"] = data["is_infant"] * data[shell_col]
    data["infant_x_length"] = data["is_infant"] * data[len_col]
    data["infant_x_whole"] = data["is_infant"] * data[whole_col]
    data["infant_x_thickening"] = data["is_infant"] * data["shell_thickening_index"]

    # 7. Leak-free Group Aggregations by Sex & PCA decomposition
    agg_cols = [shell_col, len_col, "vol_box", whole_col]
    pca_morpho_cols = [
        f"log_{len_col}",
        f"log_{dia_col}",
        f"log_{hgt_col}",
        f"log_{whole_col}",
        f"log_{shucked_col}",
        f"log_{viscera_col}",
        f"log_{shell_col}",
    ]

    if is_train:
        group_stats = {}
        for col in agg_cols:
            mean_map = data.groupby(sex_col)[col].mean().to_dict()
            std_map = data.groupby(sex_col)[col].std().fillna(1.0).to_dict()
            group_stats[f"{col}_mean"] = mean_map
            group_stats[f"{col}_std"] = std_map

        pca = PCA(n_components=3, random_state=42)
        pca.fit(data[pca_morpho_cols].values)
        group_stats["pca"] = pca

    for col in agg_cols:
        mean_series = data[sex_col].map(group_stats[f"{col}_mean"]).astype(float)
        std_series = (
            data[sex_col].map(group_stats[f"{col}_std"]).astype(float).replace(0.0, 1.0)
        )
        data[f"{col}_sex_diff"] = data[col].astype(float) - mean_series
        data[f"{col}_sex_zscore"] = (data[col].astype(float) - mean_series) / std_series

    # PCA decomposition of log-morphometrics into size vs shape factors
    pca_trans = group_stats["pca"].transform(data[pca_morpho_cols].values)
    data["pca_size_pc1"] = pca_trans[:, 0]
    data["pca_shape_pc2"] = pca_trans[:, 1]
    data["pca_shape_pc3"] = pca_trans[:, 2]

    # Drop raw categorical column and ensure numeric float32 output
    data = data.drop(columns=[sex_col])
    data = data.astype(np.float32)

    return data, group_stats


# ==============================================================================
# 2. Neural Architecture & Loss Function
# ==============================================================================
class TabularResidualBlock(nn.Module):
    """Residual block with LayerNorm, Swish activation, and Dropout."""

    def __init__(self, in_features: int, out_features: int, dropout_rate: float = 0.15):
        super().__init__()
        self.linear1 = nn.Linear(in_features, out_features)
        self.norm1 = nn.LayerNorm(out_features)
        self.act1 = nn.SiLU()
        self.dropout1 = nn.Dropout(dropout_rate)

        self.linear2 = nn.Linear(out_features, out_features)
        self.norm2 = nn.LayerNorm(out_features)
        self.act2 = nn.SiLU()
        self.dropout2 = nn.Dropout(dropout_rate)

        self.shortcut = (
            nn.Linear(in_features, out_features)
            if in_features != out_features
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.dropout1(self.act1(self.norm1(self.linear1(x))))
        out = self.dropout2(self.act2(self.norm2(self.linear2(out))))
        return out + residual


class AbaloneTabularNet(nn.Module):
    """Deep Tabular Residual MLP tailored for morphometric regression."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [256, 128, 64],
        dropout_rate: float = 0.15,
    ):
        super().__init__()
        layers = []
        current_dim = input_dim

        self.input_norm = nn.BatchNorm1d(input_dim)

        for h_dim in hidden_dims:
            layers.append(
                TabularResidualBlock(
                    in_features=current_dim,
                    out_features=h_dim,
                    dropout_rate=dropout_rate,
                )
            )
            current_dim = h_dim

        self.backbone = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Linear(current_dim, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_norm(x)
        x = self.backbone(x)
        out = self.head(x)
        return out.squeeze(-1)


class RMSELoss(nn.Module):
    """Root Mean Squared Error loss."""

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(self.mse(y_pred, y_true) + self.eps)


def create_nn_model_and_optimizer(
    input_dim: int,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    epochs: int = 100,
) -> Tuple[nn.Module, nn.Module, torch.optim.Optimizer, Any]:
    model = AbaloneTabularNet(
        input_dim=input_dim, hidden_dims=[256, 128, 64], dropout_rate=0.15
    )
    criterion = RMSELoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    return model, criterion, optimizer, scheduler


# ==============================================================================
# 3. Model Hyperparameter Factories
# ==============================================================================
def get_lgbm_params(seed: int = 42) -> Dict[str, Any]:
    return {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": 6,
        "min_child_samples": 20,
        "subsample": 0.85,
        "subsample_freq": 1,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": seed,
        "n_jobs": -1,
        "verbose": -1,
    }


def get_xgb_params(seed: int = 42) -> Dict[str, Any]:
    return {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "learning_rate": 0.03,
        "max_depth": 5,
        "min_child_weight": 3,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": seed,
        "n_jobs": -1,
        "tree_method": "hist",
    }


def get_catboost_params(seed: int = 42) -> Dict[str, Any]:
    return {
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "learning_rate": 0.04,
        "depth": 6,
        "l2_leaf_reg": 3.0,
        "random_seed": seed,
        "verbose": False,
        "thread_count": -1,
    }


# ==============================================================================
# 4. Main Execution Pipeline
# ==============================================================================
def main():
    os.makedirs("./working", exist_ok=True)
    os.makedirs("./submission", exist_ok=True)

    # 1. Load Data
    train_path = "./input/train.parquet"
    test_path = "./input/test.parquet"

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    train_df = standardize_col_names(train_df)
    test_df = standardize_col_names(test_df)

    target_col = identify_target_column(train_df)
    y_all = train_df[target_col].copy().values.astype(np.float32)

    X_train_raw = train_df.drop(columns=[target_col]).copy()
    X_test_raw = test_df.copy()

    # 2. Preprocess & Feature Engineering
    X_train_clean, height_median = clean_anomalies(X_train_raw)
    X_test_clean, _ = clean_anomalies(X_test_raw, height_median=height_median)

    X_train_feat, group_stats = engineer_features(X_train_clean, is_train=True)
    X_test_feat, _ = engineer_features(
        X_test_clean, is_train=False, group_stats=group_stats
    )

    feature_cols = X_train_feat.columns.tolist()
    X = X_train_feat.values
    X_test = X_test_feat.values
    n_train = len(X)
    n_test = len(X_test)
    n_features = len(feature_cols)

    # 3. Cross-Validation & Multi-Family Model Zoo Setup
    N_SPLITS = 10
    SEEDS = [42, 2024]
    n_seeds = len(SEEDS)

    oof_lgb = np.zeros(n_train, dtype=np.float32)
    oof_xgb = np.zeros(n_train, dtype=np.float32)
    oof_cat = np.zeros(n_train, dtype=np.float32)
    oof_et = np.zeros(n_train, dtype=np.float32)
    oof_rf = np.zeros(n_train, dtype=np.float32)
    oof_nn = np.zeros(n_train, dtype=np.float32)

    test_preds_lgb = np.zeros(n_test, dtype=np.float32)
    test_preds_xgb = np.zeros(n_test, dtype=np.float32)
    test_preds_cat = np.zeros(n_test, dtype=np.float32)
    test_preds_et = np.zeros(n_test, dtype=np.float32)
    test_preds_rf = np.zeros(n_test, dtype=np.float32)
    test_preds_nn = np.zeros(n_test, dtype=np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 4. Multi-Seed 10-Fold Cross-Validation Loop
    for seed in SEEDS:
        kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
        seed_weight = 1.0 / n_seeds
        fold_weight = seed_weight / N_SPLITS

        for fold, (train_idx, val_idx) in enumerate(kf.split(X, y_all)):
            X_tr, y_tr = X[train_idx], y_all[train_idx]
            X_va, y_va = X[val_idx], y_all[val_idx]
            X_te = X_test

            # 4.1 LightGBM
            lgb_params = get_lgbm_params(seed=seed + fold)
            lgb_train_data = lgb.Dataset(X_tr, label=y_tr)
            lgb_val_data = lgb.Dataset(X_va, label=y_va, reference=lgb_train_data)
            lgb_callbacks = [
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=0),
            ]

            lgb_model = lgb.train(
                params=lgb_params,
                train_set=lgb_train_data,
                num_boost_round=1500,
                valid_sets=[lgb_train_data, lgb_val_data],
                callbacks=lgb_callbacks,
            )

            oof_lgb[val_idx] += (
                lgb_model.predict(X_va, num_iteration=lgb_model.best_iteration)
                * seed_weight
            )
            test_preds_lgb += (
                lgb_model.predict(X_te, num_iteration=lgb_model.best_iteration)
                * fold_weight
            )

            # 4.2 XGBoost
            xgb_params = get_xgb_params(seed=seed + fold)
            xgb_model = xgb.XGBRegressor(
                **xgb_params,
                n_estimators=1500,
                early_stopping_rounds=50,
            )
            xgb_model.fit(
                X_tr,
                y_tr,
                eval_set=[(X_va, y_va)],
                verbose=False,
            )

            oof_xgb[val_idx] += xgb_model.predict(X_va) * seed_weight
            test_preds_xgb += xgb_model.predict(X_te) * fold_weight

            # 4.3 CatBoost
            cat_params = get_catboost_params(seed=seed + fold)
            cat_model = cb.CatBoostRegressor(
                **cat_params,
                iterations=1500,
                early_stopping_rounds=50,
            )
            cat_model.fit(
                X_tr,
                y_tr,
                eval_set=(X_va, y_va),
                verbose=False,
            )

            oof_cat[val_idx] += cat_model.predict(X_va) * seed_weight
            test_preds_cat += cat_model.predict(X_te) * fold_weight

            # 4.4 Extra Trees Regressor (Bagging)
            et_model = ExtraTreesRegressor(
                n_estimators=300,
                max_depth=16,
                min_samples_split=4,
                min_samples_leaf=2,
                max_features=0.75,
                random_state=seed + fold,
                n_jobs=-1,
            )
            et_model.fit(X_tr, y_tr)
            oof_et[val_idx] += et_model.predict(X_va) * seed_weight
            test_preds_et += et_model.predict(X_te) * fold_weight

            # 4.5 Random Forest Regressor (Bagging)
            rf_model = RandomForestRegressor(
                n_estimators=300,
                max_depth=14,
                min_samples_split=4,
                min_samples_leaf=2,
                max_features=0.6,
                random_state=seed + fold,
                n_jobs=-1,
            )
            rf_model.fit(X_tr, y_tr)
            oof_rf[val_idx] += rf_model.predict(X_va) * seed_weight
            test_preds_rf += rf_model.predict(X_te) * fold_weight

            # 4.6 Tabular Residual Neural Network
            scaler = StandardScaler()
            X_tr_sc = scaler.fit_transform(X_tr)
            X_va_sc = scaler.transform(X_va)
            X_te_sc = scaler.transform(X_te)

            tr_dataset = TensorDataset(
                torch.tensor(X_tr_sc, dtype=torch.float32),
                torch.tensor(y_tr, dtype=torch.float32),
            )
            va_dataset = TensorDataset(
                torch.tensor(X_va_sc, dtype=torch.float32),
                torch.tensor(y_va, dtype=torch.float32),
            )

            tr_loader = DataLoader(
                tr_dataset, batch_size=64, shuffle=True, drop_last=False
            )
            va_loader = DataLoader(va_dataset, batch_size=128, shuffle=False)

            nn_model, criterion, optimizer, scheduler = create_nn_model_and_optimizer(
                input_dim=n_features, lr=1e-3, weight_decay=1e-4, epochs=80
            )
            nn_model.to(device)

            best_loss = float("inf")
            best_nn_weights = None
            patience = 15
            patience_counter = 0

            for epoch in range(80):
                nn_model.train()
                for batch_x, batch_y in tr_loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    optimizer.zero_grad()
                    out = nn_model(batch_x)
                    loss = criterion(out, batch_y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(nn_model.parameters(), max_norm=1.0)
                    optimizer.step()
                scheduler.step()

                nn_model.eval()
                val_preds_list = []
                with torch.no_grad():
                    for batch_x, _ in va_loader:
                        batch_x = batch_x.to(device)
                        out = nn_model(batch_x)
                        val_preds_list.append(out.cpu().numpy())

                fold_val_preds = np.concatenate(val_preds_list)
                val_rmse = np.sqrt(mean_squared_error(y_va, fold_val_preds))

                if val_rmse < best_loss:
                    best_loss = val_rmse
                    best_nn_weights = {
                        k: v.cpu().clone() for k, v in nn_model.state_dict().items()
                    }
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        break

            nn_model.load_state_dict(best_nn_weights)
            nn_model.eval()
            with torch.no_grad():
                val_nn_pred = (
                    nn_model(torch.tensor(X_va_sc, dtype=torch.float32).to(device))
                    .cpu()
                    .numpy()
                )
                oof_nn[val_idx] += val_nn_pred * seed_weight

                fold_test_preds_nn = (
                    nn_model(torch.tensor(X_te_sc, dtype=torch.float32).to(device))
                    .cpu()
                    .numpy()
                )
                test_preds_nn += fold_test_preds_nn * fold_weight

    # 5. Out-of-Fold NNLS Optimization and Ensembling
    oof_matrix = np.column_stack(
        [oof_lgb, oof_xgb, oof_cat, oof_et, oof_rf, oof_nn]
    )
    test_matrix = np.column_stack(
        [test_preds_lgb, test_preds_xgb, test_preds_cat, test_preds_et, test_preds_rf, test_preds_nn]
    )

    n_models = oof_matrix.shape[1]

    def objective(weights):
        w = np.array(weights)
        pred = np.clip(np.dot(oof_matrix, w), 1.0, None)
        return np.sqrt(mean_squared_error(y_all, pred))

    init_weights = [1.0 / n_models] * n_models
    bounds = [(0.0, 1.0)] * n_models
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

    opt_res = minimize(
        objective,
        init_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    best_weights = opt_res.x
    best_weights = np.maximum(0, best_weights)
    best_weights /= np.sum(best_weights)

    oof_ensemble = np.clip(np.dot(oof_matrix, best_weights), 1.0, None)
    final_score = float(np.sqrt(mean_squared_error(y_all, oof_ensemble)))

    # 6. Generate Post-Processed Submission
    final_test_preds = np.clip(np.dot(test_matrix, best_weights), 1.0, None)
    submission_target_col = target_col
    id_series = (
        test_df["id"]
        if "id" in test_df.columns
        else np.arange(len(final_test_preds), dtype=int)
    )
    sub_df = pd.DataFrame(
        {
            "id": id_series,
            submission_target_col: final_test_preds,
        }
    )
    sub_path = "./submission/submission.csv"
    sub_df.to_csv(sub_path, index=False)

    print(f"Final Validation Score: {final_score}")


if __name__ == "__main__":
    main()
