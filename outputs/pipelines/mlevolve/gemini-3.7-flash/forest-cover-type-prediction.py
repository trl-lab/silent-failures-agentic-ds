from catboost import CatBoostClassifier
import json
import math
import os
import random
from typing import Any, Callable, Dict, List, Tuple
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb


# =========================================================================
# 0. Reproducibility & Directory Setup
# =========================================================================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


set_seed(42)
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)


# =========================================================================
# 1. Feature Engineering
# =========================================================================
def create_topographical_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    elev_col = (
        "Elevation"
        if "Elevation" in df.columns
        else df.columns[df.columns.str.lower() == "elevation"][0]
    )
    aspect_col = (
        "Aspect"
        if "Aspect" in df.columns
        else df.columns[df.columns.str.lower() == "aspect"][0]
    )
    slope_col = (
        "Slope"
        if "Slope" in df.columns
        else df.columns[df.columns.str.lower() == "slope"][0]
    )
    h_hydro_col = (
        "Horizontal_Distance_To_Hydrology"
        if "Horizontal_Distance_To_Hydrology" in df.columns
        else [
            c for c in df.columns if "horizontal" in c.lower() and "hydro" in c.lower()
        ][0]
    )
    v_hydro_col = (
        "Vertical_Distance_To_Hydrology"
        if "Vertical_Distance_To_Hydrology" in df.columns
        else [
            c for c in df.columns if "vertical" in c.lower() and "hydro" in c.lower()
        ][0]
    )
    road_col = (
        "Horizontal_Distance_To_Roadways"
        if "Horizontal_Distance_To_Roadways" in df.columns
        else [c for c in df.columns if "road" in c.lower()][0]
    )
    fire_col = (
        "Horizontal_Distance_To_Fire_Points"
        if "Horizontal_Distance_To_Fire_Points" in df.columns
        else [c for c in df.columns if "fire" in c.lower()][0]
    )
    hs_9am = (
        "Hillshade_9am"
        if "Hillshade_9am" in df.columns
        else [c for c in df.columns if "9am" in c.lower()][0]
    )
    hs_noon = (
        "Hillshade_Noon"
        if "Hillshade_Noon" in df.columns
        else [c for c in df.columns if "noon" in c.lower()][0]
    )
    hs_3pm = (
        "Hillshade_3pm"
        if "Hillshade_3pm" in df.columns
        else [c for c in df.columns if "3pm" in c.lower()][0]
    )

    # Trigonometric Transformations & Aspect Geometry
    aspect_rad = (df[aspect_col] % 360) * np.pi / 180.0
    slope_rad = df[slope_col] * np.pi / 180.0
    df["Aspect_Sin"] = np.sin(aspect_rad).astype(np.float32)
    df["Aspect_Cos"] = np.cos(aspect_rad).astype(np.float32)
    df["Slope_Sin"] = np.sin(slope_rad).astype(np.float32)
    df["Slope_Cos"] = np.cos(slope_rad).astype(np.float32)
    df["Aspect_Slope_Sin"] = (df["Aspect_Sin"] * df["Slope_Sin"]).astype(np.float32)
    df["Aspect_Slope_Cos"] = (df["Aspect_Cos"] * df["Slope_Cos"]).astype(np.float32)

    # Circular Solar Aspect Deviations & Solar Radiation Approximations
    df["Aspect_Dev_North"] = np.abs(180.0 - np.abs(df[aspect_col] - 180.0)).astype(np.float32)
    df["Aspect_Dev_East"] = np.abs(180.0 - np.abs((df[aspect_col] - 90.0) % 360 - 180.0)).astype(np.float32)
    df["Aspect_Dev_South"] = np.abs(df[aspect_col] - 180.0).astype(np.float32)
    df["Aspect_Dev_West"] = np.abs(180.0 - np.abs((df[aspect_col] - 270.0) % 360 - 180.0)).astype(np.float32)
    lat_rad = 40.0 * np.pi / 180.0
    df["Radiation_Index"] = (
        np.cos(lat_rad) * df["Slope_Cos"] - np.sin(lat_rad) * df["Slope_Sin"] * np.cos(aspect_rad - np.pi)
    ).astype(np.float32)

    # Hydrology Geometric & Slope-Adjusted Features
    df["Euclidean_Distance_To_Hydrology"] = np.sqrt(
        df[h_hydro_col] ** 2 + df[v_hydro_col] ** 2
    ).astype(np.float32)
    df["Manhattan_Distance_To_Hydrology"] = (
        np.abs(df[h_hydro_col]) + np.abs(df[v_hydro_col])
    ).astype(np.float32)
    df["Slope_Adjusted_Hydrology"] = (
        df["Euclidean_Distance_To_Hydrology"] * df["Slope_Cos"]
    ).astype(np.float32)
    df["Slope_Hydrology_Horizontal"] = (df[h_hydro_col] * df["Slope_Cos"]).astype(
        np.float32
    )
    df["Absolute_Elevation_Hydrology"] = (df[elev_col] - df[v_hydro_col]).astype(
        np.float32
    )
    df["Hydrology_Elevation_Plus"] = (df[elev_col] + df[v_hydro_col]).astype(np.float32)
    df["Hydrology_Is_Above"] = (df[v_hydro_col] < 0).astype(np.int8)

    # Spatial Pairwise Euclidean Distances Between Landmarks
    df["Euclidean_Dist_Road_Fire"] = np.sqrt(df[road_col] ** 2 + df[fire_col] ** 2).astype(np.float32)
    df["Euclidean_Dist_Hydro_Road"] = np.sqrt(df[h_hydro_col] ** 2 + df[road_col] ** 2).astype(np.float32)
    df["Euclidean_Dist_Hydro_Fire"] = np.sqrt(df[h_hydro_col] ** 2 + df[fire_col] ** 2).astype(np.float32)
    df["Euclidean_Dist_3D_Road"] = np.sqrt(df[road_col] ** 2 + df[v_hydro_col] ** 2).astype(np.float32)
    df["Euclidean_Dist_3D_Fire"] = np.sqrt(df[fire_col] ** 2 + df[v_hydro_col] ** 2).astype(np.float32)

    # Distance Pairwise Interactions & Relative Ratios
    df["Dist_Road_Plus_Fire"] = (df[road_col] + df[fire_col]).astype(np.float32)
    df["Dist_Road_Minus_Fire"] = (df[road_col] - df[fire_col]).astype(np.float32)
    df["Dist_Road_Minus_Fire_Abs"] = np.abs(df[road_col] - df[fire_col]).astype(
        np.float32
    )

    df["Dist_Hydro_Plus_Road"] = (df[h_hydro_col] + df[road_col]).astype(np.float32)
    df["Dist_Hydro_Minus_Road"] = (df[h_hydro_col] - df[road_col]).astype(np.float32)
    df["Dist_Hydro_Minus_Road_Abs"] = np.abs(df[h_hydro_col] - df[road_col]).astype(
        np.float32
    )

    df["Dist_Hydro_Plus_Fire"] = (df[h_hydro_col] + df[fire_col]).astype(np.float32)
    df["Dist_Hydro_Minus_Fire"] = (df[h_hydro_col] - df[fire_col]).astype(np.float32)
    df["Dist_Hydro_Minus_Fire_Abs"] = np.abs(df[h_hydro_col] - df[fire_col]).astype(
        np.float32
    )

    df["Dist_Hydro_Road_Ratio"] = (df[h_hydro_col] / (df[road_col] + 1.0)).astype(
        np.float32
    )
    df["Dist_Hydro_Fire_Ratio"] = (df[h_hydro_col] / (df[fire_col] + 1.0)).astype(
        np.float32
    )
    df["Dist_Road_Fire_Ratio"] = (df[road_col] / (df[fire_col] + 1.0)).astype(
        np.float32
    )
    df["Total_Distance_Linear"] = (
        df[h_hydro_col] + df[road_col] + df[fire_col]
    ).astype(np.float32)

    # Summary statistics across linear infrastructure distances
    dist_cols = [h_hydro_col, road_col, fire_col]
    df["Dist_Min"] = df[dist_cols].min(axis=1).astype(np.float32)
    df["Dist_Max"] = df[dist_cols].max(axis=1).astype(np.float32)
    df["Dist_Mean"] = df[dist_cols].mean(axis=1).astype(np.float32)
    df["Dist_Std"] = df[dist_cols].std(axis=1).astype(np.float32)

    # Solar Exposure, Hillshade Gradients & Symmetry Indices
    df["Hillshade_9am_minus_3pm"] = (df[hs_9am] - df[hs_3pm]).astype(np.float32)
    df["Hillshade_Noon_minus_3pm"] = (df[hs_noon] - df[hs_3pm]).astype(np.float32)
    df["Hillshade_Noon_minus_9am"] = (df[hs_noon] - df[hs_9am]).astype(np.float32)
    df["Hillshade_Mean"] = ((df[hs_9am] + df[hs_noon] + df[hs_3pm]) / 3.0).astype(
        np.float32
    )
    df["Hillshade_Max"] = df[[hs_9am, hs_noon, hs_3pm]].max(axis=1).astype(np.float32)
    df["Hillshade_Min"] = df[[hs_9am, hs_noon, hs_3pm]].min(axis=1).astype(np.float32)
    df["Hillshade_Range"] = (df["Hillshade_Max"] - df["Hillshade_Min"]).astype(
        np.float32
    )
    df["Hillshade_Symmetry"] = (df[hs_9am] + df[hs_3pm] - df[hs_noon]).astype(
        np.float32
    )
    df["Hillshade_3pm_9am_Ratio"] = (
        (df[hs_3pm] + 1.0) / (df[hs_9am] + 1.0)
    ).astype(np.float32)
    df["Hillshade_9am_3pm_Contrast"] = (
        (df[hs_9am] - df[hs_3pm]) / (df[hs_noon] + 1.0)
    ).astype(np.float32)

    # Directional hillshade gradient contrasts
    df["Hillshade_Gradient_Morning"] = (df[hs_noon] - df[hs_9am]).astype(np.float32)
    df["Hillshade_Gradient_Evening"] = (df[hs_3pm] - df[hs_noon]).astype(np.float32)
    df["Hillshade_Total_Gradient"] = (
        np.abs(df[hs_noon] - df[hs_9am]) + np.abs(df[hs_3pm] - df[hs_noon])
    ).astype(np.float32)

    # Topographical Ratios and Interactions
    df["Elevation_Slope_Ratio"] = (df[elev_col] / (df[slope_col] + 1.0)).astype(
        np.float32
    )
    df["Elevation_Slope_Mult"] = (df[elev_col] * df[slope_col]).astype(np.float32)
    df["Elevation_Dist_Hydro_Ratio"] = (df[elev_col] / (df[h_hydro_col] + 1.0)).astype(
        np.float32
    )
    df["Elevation_Dist_Road_Ratio"] = (df[elev_col] / (df[road_col] + 1.0)).astype(
        np.float32
    )
    df["Elevation_Dist_Fire_Ratio"] = (df[elev_col] / (df[fire_col] + 1.0)).astype(
        np.float32
    )

    # Categorical Compaction (Dense Soil & Wilderness IDs)
    soil_cols = [c for c in df.columns if c.startswith("Soil_Type")]
    wild_cols = [c for c in df.columns if c.startswith("Wilderness_Area")]

    if len(soil_cols) > 0:
        df["Soil_Type_Dense"] = (df[soil_cols].values.argmax(axis=1) + 1).astype(
            np.int16
        )
        df["Soil_Type_Count"] = df[soil_cols].sum(axis=1).astype(np.int8)

    if len(wild_cols) > 0:
        df["Wilderness_Area_Dense"] = (df[wild_cols].values.argmax(axis=1) + 1).astype(
            np.int16
        )

    # USFS ELU Climatic and Geological Zones
    elu_climatic_zone = {
        1: 2,
        2: 2,
        3: 3,
        4: 3,
        5: 4,
        6: 4,
        7: 5,
        8: 6,
        9: 7,
        10: 7,
        11: 7,
        12: 7,
        13: 7,
        14: 7,
        15: 7,
        16: 7,
        17: 7,
        18: 7,
        19: 7,
        20: 7,
        21: 7,
        22: 7,
        23: 7,
        24: 7,
        25: 7,
        26: 7,
        27: 7,
        28: 7,
        29: 7,
        30: 7,
        31: 7,
        32: 7,
        33: 7,
        34: 7,
        35: 7,
        36: 7,
        37: 7,
        38: 7,
        39: 7,
        40: 8,
    }
    elu_geologic_zone = {
        1: 7,
        2: 7,
        3: 7,
        4: 7,
        5: 7,
        6: 7,
        7: 7,
        8: 7,
        9: 2,
        10: 2,
        11: 2,
        12: 2,
        13: 2,
        14: 2,
        15: 2,
        16: 2,
        17: 2,
        18: 2,
        19: 2,
        20: 2,
        21: 2,
        22: 2,
        23: 2,
        24: 7,
        25: 7,
        26: 7,
        27: 7,
        28: 7,
        29: 7,
        30: 7,
        31: 7,
        32: 7,
        33: 7,
        34: 7,
        35: 7,
        36: 7,
        37: 7,
        38: 7,
        39: 7,
        40: 7,
    }

    if len(soil_cols) > 0 and len(wild_cols) > 0:
        df["Wilderness_Soil_Interaction"] = (
            df["Wilderness_Area_Dense"] * 100 + df["Soil_Type_Dense"]
        ).astype(np.int32)

    # USFS Soil Family & Geological Rockiness Indicators
    rocky_soils = {
        1, 2, 3, 4, 5, 6, 10, 11, 13, 24, 25, 26, 27, 28, 31, 32, 33, 34, 38, 39, 40,
    }
    bouldery_soils = {2, 9, 18, 26}
    stony_soils = {
        1, 2, 3, 4, 5, 6, 10, 11, 13, 24, 25, 26, 27, 28, 31, 32, 33, 34, 38, 39, 40,
    }
    rubbly_soils = {3, 4, 5, 10, 11, 13}
    leighcan_soils = {22, 23, 24, 25, 27, 28, 31, 32, 33, 38, 39, 40}
    bullwark_soils = {10, 11, 12, 13}
    vanet_soils = {2, 5, 6}
    catamount_soils = {4, 10, 23, 31, 33}
    hydric_soils = {7, 8, 14, 15, 16, 17, 19, 20, 21}

    if len(soil_cols) > 0:
        df["Soil_Climatic_Zone"] = (
            df["Soil_Type_Dense"].map(elu_climatic_zone).fillna(0).astype(np.int8)
        )
        df["Soil_Geologic_Zone"] = (
            df["Soil_Type_Dense"].map(elu_geologic_zone).fillna(0).astype(np.int8)
        )
        df["Soil_Is_Rocky"] = df["Soil_Type_Dense"].isin(rocky_soils).astype(np.int8)
        df["Soil_Is_Bouldery"] = (
            df["Soil_Type_Dense"].isin(bouldery_soils).astype(np.int8)
        )
        df["Soil_Is_Stony"] = df["Soil_Type_Dense"].isin(stony_soils).astype(np.int8)
        df["Soil_Is_Rubbly"] = df["Soil_Type_Dense"].isin(rubbly_soils).astype(np.int8)
        df["Soil_Is_Leighcan"] = (
            df["Soil_Type_Dense"].isin(leighcan_soils).astype(np.int8)
        )
        df["Soil_Is_Bullwark"] = (
            df["Soil_Type_Dense"].isin(bullwark_soils).astype(np.int8)
        )
        df["Soil_Is_Vanet"] = df["Soil_Type_Dense"].isin(vanet_soils).astype(np.int8)
        df["Soil_Is_Catamount"] = (
            df["Soil_Type_Dense"].isin(catamount_soils).astype(np.int8)
        )
        df["Soil_Is_Hydric"] = df["Soil_Type_Dense"].isin(hydric_soils).astype(np.int8)

    return df


def compute_fold_group_features(
    tr_df: pd.DataFrame, va_df: pd.DataFrame, te_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    group_cols = [
        "Soil_Type_Dense",
        "Wilderness_Area_Dense",
        "Wilderness_Soil_Interaction",
    ]
    num_cols = [
        "Elevation",
        "Slope",
        "Horizontal_Distance_To_Hydrology",
        "Vertical_Distance_To_Hydrology",
        "Horizontal_Distance_To_Roadways",
        "Horizontal_Distance_To_Fire_Points",
    ]

    tr_dict = {}
    va_dict = {}
    te_dict = {}

    for g_col in group_cols:
        if g_col not in tr_df.columns:
            continue
        for n_col in num_cols:
            if n_col not in tr_df.columns:
                continue

            mean_map = tr_df.groupby(g_col)[n_col].mean().to_dict()
            std_map = tr_df.groupby(g_col)[n_col].std().to_dict()
            min_map = tr_df.groupby(g_col)[n_col].min().to_dict()
            max_map = tr_df.groupby(g_col)[n_col].max().to_dict()

            global_mean = float(tr_df[n_col].mean())
            global_std = (
                float(tr_df[n_col].std()) if float(tr_df[n_col].std()) > 0 else 1.0
            )
            global_min = float(tr_df[n_col].min())
            global_max = float(tr_df[n_col].max())

            for df_curr, d_out in [(tr_df, tr_dict), (va_df, va_dict), (te_df, te_dict)]:
                m = df_curr[g_col].map(mean_map).fillna(global_mean).astype(np.float32)
                s = df_curr[g_col].map(std_map).fillna(global_std).astype(np.float32)
                mn = df_curr[g_col].map(min_map).fillna(global_min).astype(np.float32)
                mx = df_curr[g_col].map(max_map).fillna(global_max).astype(np.float32)

                d_out[f"{n_col}_mean_{g_col}"] = m
                d_out[f"{n_col}_std_{g_col}"] = s
                d_out[f"{n_col}_min_{g_col}"] = mn
                d_out[f"{n_col}_max_{g_col}"] = mx
                d_out[f"{n_col}_diff_mean_{g_col}"] = (df_curr[n_col] - m).astype(np.float32)
                d_out[f"{n_col}_ratio_mean_{g_col}"] = (df_curr[n_col] / (m + 1e-4)).astype(np.float32)

    tr = pd.concat([tr_df, pd.DataFrame(tr_dict, index=tr_df.index)], axis=1)
    va = pd.concat([va_df, pd.DataFrame(va_dict, index=va_df.index)], axis=1)
    te = pd.concat([te_df, pd.DataFrame(te_dict, index=te_df.index)], axis=1)

    return tr, va, te


# =========================================================================
# 2. Deep Tabular Neural Network Architecture (TabularResNet)
# =========================================================================
class ResidualBlock(nn.Module):
    def __init__(
        self, hidden_dim: int, dropout_rate: float = 0.15, use_norm: bool = True
    ):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim) if use_norm else nn.Identity()
        self.act1 = nn.SiLU()
        self.dropout1 = nn.Dropout(dropout_rate)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim) if use_norm else nn.Identity()
        self.act2 = nn.SiLU()
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.dropout1(self.act1(self.norm1(self.fc1(x))))
        out = self.dropout2(self.act2(self.norm2(self.fc2(out))))
        return out + residual


class TabularResNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int = 7,
        hidden_dim: int = 256,
        num_blocks: int = 3,
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
        )

        self.res_blocks = nn.ModuleList(
            [
                ResidualBlock(hidden_dim=hidden_dim, dropout_rate=dropout_rate)
                for _ in range(num_blocks)
            ]
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout_rate / 2.0),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_projection(x)
        for block in self.res_blocks:
            h = block(h)
        return self.head(h)


def build_tabular_neural_network(
    input_dim: int,
    num_classes: int = 7,
    hidden_dim: int = 256,
    num_blocks: int = 3,
    dropout_rate: float = 0.2,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    label_smoothing: float = 0.05,
    max_epochs: int = 35,
) -> Tuple[
    nn.Module,
    nn.Module,
    torch.optim.Optimizer,
    Callable[[torch.optim.Optimizer], torch.optim.lr_scheduler._LRScheduler],
]:
    model = TabularResNet(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        num_blocks=num_blocks,
        dropout_rate=dropout_rate,
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    def get_scheduler(opt: torch.optim.Optimizer):
        return CosineAnnealingLR(opt, T_max=max_epochs, eta_min=1e-6)

    return model, criterion, optimizer, get_scheduler


# =========================================================================
# 3. Central Model Design Registry
# =========================================================================
class ModelDesignRegistry:
    @staticmethod
    def create_deep_model(input_dim: int, num_classes: int = 7, **kwargs):
        return build_tabular_neural_network(
            input_dim=input_dim, num_classes=num_classes, **kwargs
        )

    @staticmethod
    def create_lgbm_model(
        num_classes: int = 7, random_state: int = 42
    ) -> lgb.LGBMClassifier:
        params = {
            "objective": "multiclass",
            "num_class": num_classes,
            "metric": "multi_logloss",
            "boosting_type": "gbdt",
            "n_estimators": 1500,
            "learning_rate": 0.04,
            "num_leaves": 95,
            "max_depth": 11,
            "min_child_samples": 10,
            "subsample": 0.85,
            "subsample_freq": 1,
            "colsample_bytree": 0.75,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "random_state": random_state,
            "n_jobs": -1,
            "verbose": -1,
        }
        return lgb.LGBMClassifier(**params)

    @staticmethod
    def create_xgb_model(
        num_classes: int = 7, random_state: int = 42
    ) -> xgb.XGBClassifier:
        return xgb.XGBClassifier(
            objective="multi:softprob",
            num_class=num_classes,
            eval_metric="mlogloss",
            n_estimators=1500,
            learning_rate=0.03,
            max_depth=10,
            min_child_weight=2,
            subsample=0.85,
            colsample_bytree=0.75,
            gamma=0.05,
            reg_alpha=0.1,
            reg_lambda=1.5,
            tree_method="hist",
            early_stopping_rounds=50,
            random_state=random_state,
            n_jobs=-1,
        )

    @staticmethod
    def create_catboost_model(
        num_classes: int = 7, random_state: int = 42
    ) -> CatBoostClassifier:
        return CatBoostClassifier(
            iterations=1500,
            learning_rate=0.04,
            depth=7,
            loss_function="MultiClass",
            eval_metric="MultiClass",
            early_stopping_rounds=50,
            random_seed=random_state,
            verbose=False,
            thread_count=-1,
        )

    @staticmethod
    def create_extratrees_model(random_state: int = 42) -> ExtraTreesClassifier:
        return ExtraTreesClassifier(
            n_estimators=600,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features="sqrt",
            bootstrap=False,
            random_state=random_state,
            n_jobs=-1,
        )

    @staticmethod
    def create_rf_model(random_state: int = 42) -> RandomForestClassifier:
        return RandomForestClassifier(
            n_estimators=600,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features="sqrt",
            bootstrap=True,
            random_state=random_state,
            n_jobs=-1,
        )

    @staticmethod
    def create_knn_model() -> KNeighborsClassifier:
        return KNeighborsClassifier(
            n_neighbors=5,
            weights="distance",
            n_jobs=-1,
        )


# =========================================================================
# 4. Pipeline Execution: Data Loading, Transformation & Stratified Cross-Validation
# =========================================================================
print("Loading train and test parquet datasets...")
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

id_col = "Id" if "Id" in train_df.columns else "id"
target_col = "Cover_Type" if "Cover_Type" in train_df.columns else "cover_type"

print(f"Raw Train Shape: {train_df.shape}, Raw Test Shape: {test_df.shape}")

print("Generating topographical and domain interaction features...")
train_fe = create_topographical_features(train_df)
test_fe = create_topographical_features(test_df)

# Frequency encoding fit strictly on train and mapped to test
soil_freq = train_fe["Soil_Type_Dense"].value_counts(normalize=True).to_dict()
wild_freq = train_fe["Wilderness_Area_Dense"].value_counts(normalize=True).to_dict()

train_fe["Soil_Type_Freq"] = (
    train_fe["Soil_Type_Dense"].map(soil_freq).fillna(0.0).astype(np.float32)
)
test_fe["Soil_Type_Freq"] = (
    test_fe["Soil_Type_Dense"].map(soil_freq).fillna(0.0).astype(np.float32)
)

train_fe["Wilderness_Area_Freq"] = (
    train_fe["Wilderness_Area_Dense"].map(wild_freq).fillna(0.0).astype(np.float32)
)
test_fe["Wilderness_Area_Freq"] = (
    test_fe["Wilderness_Area_Dense"].map(wild_freq).fillna(0.0).astype(np.float32)
)

# Latent TruncatedSVD Soil Representations (5 components)
soil_binary_cols = [c for c in train_df.columns if c.startswith("Soil_Type") and c in train_fe.columns]
if len(soil_binary_cols) > 0:
    svd = TruncatedSVD(n_components=5, random_state=42)
    soil_svd_train = svd.fit_transform(train_fe[soil_binary_cols].values)
    soil_svd_test = svd.transform(test_fe[soil_binary_cols].values)
    for comp_idx in range(5):
        train_fe[f"Soil_SVD_{comp_idx+1}"] = soil_svd_train[:, comp_idx].astype(np.float32)
        test_fe[f"Soil_SVD_{comp_idx+1}"] = soil_svd_test[:, comp_idx].astype(np.float32)

# Identify target, feature columns, and class count
num_classes = int(train_fe[target_col].nunique())
feature_cols = [
    c
    for c in train_fe.columns
    if c not in [id_col, target_col, "fold", "Id", "id", "Cover_Type", "cover_type"]
]

n_samples = len(train_fe)
n_test = len(test_fe)

# Stratified 10-Fold Splits
n_splits = 10
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
train_fe["fold"] = -1
for fold, (train_idx, val_idx) in enumerate(skf.split(train_fe, train_fe[target_col])):
    train_fe.iloc[val_idx, train_fe.columns.get_loc("fold")] = fold

# Map 1-indexed target classes [1..7] to 0-indexed [0..6] for modeling
y_all = train_fe[target_col].values
y_all_0idx = (y_all - 1).astype(np.int64)

oof_lgb = np.zeros((n_samples, num_classes), dtype=np.float32)
test_lgb = np.zeros((n_test, num_classes), dtype=np.float32)

oof_xgb = np.zeros((n_samples, num_classes), dtype=np.float32)
test_xgb = np.zeros((n_test, num_classes), dtype=np.float32)

oof_cb = np.zeros((n_samples, num_classes), dtype=np.float32)
test_cb = np.zeros((n_test, num_classes), dtype=np.float32)

oof_et = np.zeros((n_samples, num_classes), dtype=np.float32)
test_et = np.zeros((n_test, num_classes), dtype=np.float32)

oof_rf = np.zeros((n_samples, num_classes), dtype=np.float32)
test_rf = np.zeros((n_test, num_classes), dtype=np.float32)

oof_knn = np.zeros((n_samples, num_classes), dtype=np.float32)
test_knn = np.zeros((n_test, num_classes), dtype=np.float32)

print(
    f"Training across {n_splits} Stratified Folds with 6 model families and within-fold group aggregations..."
)

for fold in range(n_splits):
    train_mask = train_fe["fold"] != fold
    val_mask = train_fe["fold"] == fold
    val_indices = np.where(val_mask)[0]

    tr_fold_raw = train_fe.loc[train_mask].copy()
    va_fold_raw = train_fe.loc[val_mask].copy()
    te_fold_raw = test_fe.copy()

    # Compute group aggregation features strictly within the fold
    tr_fold, va_fold, te_fold = compute_fold_group_features(
        tr_fold_raw, va_fold_raw, te_fold_raw
    )

    current_feature_cols = [
        c
        for c in tr_fold.columns
        if c not in [id_col, target_col, "fold", "Id", "id", "Cover_Type", "cover_type"]
    ]

    X_tr = tr_fold[current_feature_cols].values.astype(np.float32)
    y_tr = y_all_0idx[train_mask]
    X_va = va_fold[current_feature_cols].values.astype(np.float32)
    y_va = y_all_0idx[val_mask]
    X_te = te_fold[current_feature_cols].values.astype(np.float32)

    # Model 1: LightGBM
    lgb_model = ModelDesignRegistry.create_lgbm_model(
        num_classes=num_classes, random_state=42 + fold
    )
    lgb_callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=False)]
    lgb_model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=lgb_callbacks,
    )
    oof_lgb[val_indices] = lgb_model.predict_proba(X_va)
    test_lgb += lgb_model.predict_proba(X_te) / float(n_splits)
    joblib.dump(lgb_model, f"./working/lgb_model_fold_{fold}.joblib")

    # Model 2: XGBoost
    xgb_model = ModelDesignRegistry.create_xgb_model(
        num_classes=num_classes, random_state=42 + fold
    )
    xgb_model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    oof_xgb[val_indices] = xgb_model.predict_proba(X_va)
    test_xgb += xgb_model.predict_proba(X_te) / float(n_splits)
    joblib.dump(xgb_model, f"./working/xgb_model_fold_{fold}.joblib")

    # Model 3: CatBoost
    cb_model = ModelDesignRegistry.create_catboost_model(
        num_classes=num_classes, random_state=42 + fold
    )
    cb_model.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=False)
    oof_cb[val_indices] = cb_model.predict_proba(X_va)
    test_cb += cb_model.predict_proba(X_te) / float(n_splits)
    joblib.dump(cb_model, f"./working/cb_model_fold_{fold}.joblib")

    # Model 4: ExtraTrees
    et_model = ModelDesignRegistry.create_extratrees_model(random_state=42 + fold)
    et_model.fit(X_tr, y_tr)
    oof_et[val_indices] = et_model.predict_proba(X_va)
    test_et += et_model.predict_proba(X_te) / float(n_splits)
    joblib.dump(et_model, f"./working/et_model_fold_{fold}.joblib")

    # Model 5: RandomForest
    rf_model = ModelDesignRegistry.create_rf_model(random_state=42 + fold)
    rf_model.fit(X_tr, y_tr)
    oof_rf[val_indices] = rf_model.predict_proba(X_va)
    test_rf += rf_model.predict_proba(X_te) / float(n_splits)
    joblib.dump(rf_model, f"./working/rf_model_fold_{fold}.joblib")

    # Model 6: Scaled KNN
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_va_scaled = scaler.transform(X_va)
    X_te_scaled = scaler.transform(X_te)

    knn_model = ModelDesignRegistry.create_knn_model()
    knn_model.fit(X_tr_scaled, y_tr)
    oof_knn[val_indices] = knn_model.predict_proba(X_va_scaled)
    test_knn += knn_model.predict_proba(X_te_scaled) / float(n_splits)
    joblib.dump(knn_model, f"./working/knn_model_fold_{fold}.joblib")

    fold_lgb_acc = accuracy_score(y_va, np.argmax(oof_lgb[val_indices], axis=1))
    fold_xgb_acc = accuracy_score(y_va, np.argmax(oof_xgb[val_indices], axis=1))
    fold_cb_acc = accuracy_score(y_va, np.argmax(oof_cb[val_indices], axis=1))
    fold_et_acc = accuracy_score(y_va, np.argmax(oof_et[val_indices], axis=1))
    fold_rf_acc = accuracy_score(y_va, np.argmax(oof_rf[val_indices], axis=1))
    fold_knn_acc = accuracy_score(y_va, np.argmax(oof_knn[val_indices], axis=1))
    print(
        f"Fold {fold+1}/{n_splits} Accuracies -> LGB: {fold_lgb_acc:.4f} | XGB: {fold_xgb_acc:.4f} | CB: {fold_cb_acc:.4f} | ET: {fold_et_acc:.4f} | RF: {fold_rf_acc:.4f} | KNN: {fold_knn_acc:.4f}"
    )

# =========================================================================
# 5. Ensemble Optimization, Holdout Evaluation & Submission Output
# =========================================================================
model_candidates = [
    ("LGB", "oof_lgb", "test_lgb"),
    ("XGB", "oof_xgb", "test_xgb"),
    ("CatBoost", "oof_cb", "test_cb"),
    ("ExtraTrees", "oof_et", "test_et"),
    ("RandomForest", "oof_rf", "test_rf"),
    ("KNN", "oof_knn", "test_knn"),
]

model_names = []
model_oofs = []
model_tests = []
seen_names = set()

for display_name, oof_var, test_var in model_candidates:
    clean_name = "CatBoost" if "Cat" in display_name or "cb" in display_name else display_name
    if clean_name in seen_names:
        continue
    if oof_var in globals() and test_var in globals():
        seen_names.add(clean_name)
        model_names.append(clean_name)
        model_oofs.append(globals()[oof_var])
        model_tests.append(globals()[test_var])

n_models = len(model_oofs)
print(f"Ensemble models active ({n_models}): {model_names}")

# Print individual model out-of-fold accuracies
best_single_acc = 0.0
best_single_idx = 0
for i, name in enumerate(model_names):
    single_acc = accuracy_score(y_all_0idx, np.argmax(model_oofs[i], axis=1))
    print(f"Model {name} OOF Accuracy: {single_acc:.5f}")
    if single_acc > best_single_acc:
        best_single_acc = single_acc
        best_single_idx = i

# 10-Fold Cross-Validated Multi-Class Stacking Meta-Learner
print("Training L2-regularized Multi-Class Stacking Meta-Learner across OOF probability tensors...")
oof_feature_matrix = np.hstack(model_oofs)
test_feature_matrix = np.hstack(model_tests)

meta_candidates = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
best_meta_acc = -1.0
best_c = 1.0
best_oof_meta = np.zeros((n_samples, num_classes), dtype=np.float32)

meta_skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

for c_val in meta_candidates:
    curr_oof_meta = np.zeros((n_samples, num_classes), dtype=np.float32)
    for m_tr_idx, m_va_idx in meta_skf.split(oof_feature_matrix, y_all_0idx):
        meta_clf = LogisticRegression(
            C=c_val,
            penalty="l2",
            solver="lbfgs",
            max_iter=2000,
            random_state=42,
        )
        meta_clf.fit(oof_feature_matrix[m_tr_idx], y_all_0idx[m_tr_idx])
        curr_oof_meta[m_va_idx] = meta_clf.predict_proba(oof_feature_matrix[m_va_idx])

    curr_acc = accuracy_score(y_all_0idx, np.argmax(curr_oof_meta, axis=1))
    print(f"Stacking Meta-Learner (C={c_val}) CV Accuracy: {curr_acc:.5f}")
    if curr_acc > best_meta_acc:
        best_meta_acc = curr_acc
        best_c = c_val
        best_oof_meta = curr_oof_meta

print(f"Selected Best Meta-Learner C: {best_c} with CV Accuracy: {best_meta_acc:.5f}")

test_ensemble_probs = np.zeros((n_test, num_classes), dtype=np.float32)
for m_tr_idx, m_va_idx in meta_skf.split(oof_feature_matrix, y_all_0idx):
    meta_clf = LogisticRegression(
        C=best_c,
        penalty="l2",
        solver="lbfgs",
        max_iter=2000,
        random_state=42,
    )
    meta_clf.fit(oof_feature_matrix[m_tr_idx], y_all_0idx[m_tr_idx])
    test_ensemble_probs += meta_clf.predict_proba(test_feature_matrix) / 10.0

oof_ensemble_probs = best_oof_meta

oof_preds_0idx = np.argmax(oof_ensemble_probs, axis=1)
oof_preds_cover_type = oof_preds_0idx + 1

test_preds_0idx = np.argmax(test_ensemble_probs, axis=1)
test_preds_cover_type = test_preds_0idx + 1

validation_accuracy = accuracy_score(y_all, oof_preds_cover_type)

test_ids = test_df[id_col].values if id_col in test_df.columns else np.arange(n_test, dtype=np.int64)
submission_df = pd.DataFrame(
    {
        "id": test_ids,
        "Cover_Type": test_preds_cover_type.astype(int),
    }
)
submission_path = "./submission/submission.csv"
submission_df.to_csv(submission_path, index=False)

print(
    f"Submission successfully generated at {submission_path} with {len(submission_df)} rows."
)
print(f"Final Validation Score: {validation_accuracy}")