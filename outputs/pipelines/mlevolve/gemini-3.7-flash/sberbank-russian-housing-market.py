import json
import math
import os
import warnings
from catboost import CatBoostRegressor
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb

warnings.filterwarnings("ignore")

# =========================================================================
# 1. EVALUATION METRIC
# =========================================================================


def compute_rmsle(
    y_true: np.ndarray, y_pred: np.ndarray, is_log_target: bool = False
) -> float:
    """Computes the official Root Mean Squared Logarithmic Error (RMSLE)."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    y_pred = np.clip(y_pred, 0.0, None)

    if is_log_target:
        return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    else:
        log_true = np.log1p(np.clip(y_true, 0.0, None))
        log_pred = np.log1p(np.clip(y_pred, 0.0, None))
        return float(np.sqrt(np.mean((log_pred - log_true) ** 2)))


# =========================================================================
# 2. FEATURE ENGINEERING & DATA PROCESSING
# =========================================================================


def clean_structural_features(df: pd.DataFrame) -> pd.DataFrame:
    """Clean physical property anomalies, impossible years, and contradictory spatial dimensions."""
    df = df.copy()

    # Structural missingness indicator binary flags (identifies new investment deals vs resales)
    structural_cols = [
        "life_sq",
        "build_year",
        "kitch_sq",
        "num_room",
        "state",
        "max_floor",
    ]
    for col in structural_cols:
        if col in df.columns:
            df[f"missing_{col}"] = df[col].isnull().astype(np.float32)

    # 1. full_sq cleaning: full area < 1 or extreme outliers (> 1500) set to NaN
    if "full_sq" in df.columns:
        df.loc[(df["full_sq"] < 5) | (df["full_sq"] > 1500), "full_sq"] = np.nan

    # 2. life_sq cleaning: living area cannot exceed full_sq or be negative
    if "life_sq" in df.columns:
        df.loc[df["life_sq"] < 2, "life_sq"] = np.nan
        if "full_sq" in df.columns:
            invalid_life = df["life_sq"] > df["full_sq"]
            df.loc[invalid_life & (df["life_sq"] > df["full_sq"] * 1.5), "life_sq"] = (
                np.nan
            )
            df.loc[df["life_sq"] > df["full_sq"], "life_sq"] = df.loc[
                df["life_sq"] > df["full_sq"], "full_sq"
            ]

    # 3. kitch_sq cleaning: kitchen cannot exceed full_sq
    if "kitch_sq" in df.columns:
        df.loc[df["kitch_sq"] < 1, "kitch_sq"] = np.nan
        if "full_sq" in df.columns:
            df.loc[df["kitch_sq"] >= df["full_sq"], "kitch_sq"] = np.nan

    # 4. build_year cleaning: Moscow realistic residential building years (1850 to 2020)
    if "build_year" in df.columns:
        df.loc[(df["build_year"] < 1850) | (df["build_year"] > 2020), "build_year"] = (
            np.nan
        )

    # 5. floor and max_floor consistency
    if "floor" in df.columns and "max_floor" in df.columns:
        df.loc[df["floor"] < 0, "floor"] = np.nan
        df.loc[df["max_floor"] < 1, "max_floor"] = np.nan
        df.loc[
            (df["floor"] > df["max_floor"]) & (df["max_floor"].notnull()), "max_floor"
        ] = np.nan

    # 6. state: Moscow state classification is 1-4. Fix known typo (33 -> 3)
    if "state" in df.columns:
        df.loc[df["state"] == 33, "state"] = 3
        df.loc[~df["state"].isin([1, 2, 3, 4]), "state"] = np.nan

    # 7. num_room: room count < 1 set to NaN
    if "num_room" in df.columns:
        df.loc[df["num_room"] < 1, "num_room"] = np.nan

    return df


def engineer_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generate domain-specific interactions, structural ratios, and temporal signals."""
    df = df.copy()

    # Temporal features
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"])
        df["year"] = ts.dt.year
        df["month"] = ts.dt.month
        df["day"] = ts.dt.day
        df["dayofweek"] = ts.dt.dayofweek
        df["quarter"] = ts.dt.quarter
        df["year_month"] = df["year"] * 100 + df["month"]

        epoch = pd.to_datetime("2011-08-01")
        df["days_since_start"] = (ts - epoch).dt.days
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0).astype(np.float32)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0).astype(np.float32)

    # Real estate geometric ratios
    if "full_sq" in df.columns and "life_sq" in df.columns:
        df["ratio_life_full"] = (df["life_sq"] / (df["full_sq"] + 1e-5)).astype(
            np.float32
        )
        df["diff_full_life"] = (df["full_sq"] - df["life_sq"]).astype(np.float32)

    if "full_sq" in df.columns and "kitch_sq" in df.columns:
        df["ratio_kitch_full"] = (df["kitch_sq"] / (df["full_sq"] + 1e-5)).astype(
            np.float32
        )

    if "life_sq" in df.columns and "kitch_sq" in df.columns:
        df["ratio_kitch_life"] = (df["kitch_sq"] / (df["life_sq"] + 1e-5)).astype(
            np.float32
        )

    if "full_sq" in df.columns and "life_sq" in df.columns and "kitch_sq" in df.columns:
        df["extra_usable_sq"] = (
            df["full_sq"] - df["life_sq"] - df["kitch_sq"]
        ).astype(np.float32)

    if "full_sq" in df.columns and "num_room" in df.columns:
        df["avg_room_size"] = (df["full_sq"] / (df["num_room"] + 1e-5)).astype(
            np.float32
        )

    if "life_sq" in df.columns and "num_room" in df.columns:
        df["avg_life_room_size"] = (df["life_sq"] / (df["num_room"] + 1e-5)).astype(
            np.float32
        )

    # Floor position features
    if "floor" in df.columns and "max_floor" in df.columns:
        df["ratio_floor_max"] = (df["floor"] / (df["max_floor"] + 1e-5)).astype(
            np.float32
        )
        df["diff_floor_max"] = (df["max_floor"] - df["floor"]).astype(np.float32)
        df["is_ground_floor"] = (df["floor"] == 1).astype(np.float32)
        df["is_top_floor"] = (
            (df["floor"] == df["max_floor"]) & (df["max_floor"] > 1)
        ).astype(np.float32)

    # Building age and era cohorts
    if "build_year" in df.columns and "year" in df.columns:
        df["building_age"] = (df["year"] - df["build_year"]).astype(np.float32)
        df["is_new_building"] = (df["building_age"] <= 3).astype(np.float32)
        df["is_pre_revolution"] = (df["build_year"] < 1917).astype(np.float32)
        df["is_stalin"] = (
            (df["build_year"] >= 1930) & (df["build_year"] < 1955)
        ).astype(np.float32)
        df["is_khrushchev"] = (
            (df["build_year"] >= 1955) & (df["build_year"] <= 1970)
        ).astype(np.float32)
        df["is_brezhnev"] = (
            (df["build_year"] > 1970) & (df["build_year"] <= 1990)
        ).astype(np.float32)
        df["is_modern"] = (df["build_year"] > 2000).astype(np.float32)

    # Moscow radial and transport spatial interactions
    if "kremlin_km" in df.columns and "mkad_km" in df.columns:
        df["ratio_kremlin_mkad"] = (
            df["kremlin_km"] / (df["mkad_km"] + 1e-5)
        ).astype(np.float32)

    if "ttk_km" in df.columns and "sadovoe_km" in df.columns:
        df["ratio_ttk_sadovoe"] = (
            df["ttk_km"] / (df["sadovoe_km"] + 1e-5)
        ).astype(np.float32)

    if "metro_km_avto" in df.columns and "kremlin_km" in df.columns:
        df["ratio_metro_kremlin"] = (
            df["metro_km_avto"] / (df["kremlin_km"] + 1e-5)
        ).astype(np.float32)

    if "metro_min_walk" in df.columns and "metro_km_walk" in df.columns:
        df["metro_walk_intensity"] = (
            df["metro_min_walk"] * df["metro_km_walk"]
        ).astype(np.float32)

    # Demographic and spatial density features
    if "raion_popul" in df.columns and "area_m" in df.columns:
        df["raion_density"] = (df["raion_popul"] / (df["area_m"] / 1e6 + 1e-5)).astype(
            np.float32
        )

    if "work_all" in df.columns and "raion_popul" in df.columns:
        df["work_share"] = (df["work_all"] / (df["raion_popul"] + 1e-5)).astype(
            np.float32
        )

    if "children_preschool" in df.columns and "preschool_quota" in df.columns:
        df["preschool_load"] = (
            df["children_preschool"] / (df["preschool_quota"] + 1e-5)
        ).astype(np.float32)

    if "children_school" in df.columns and "school_quota" in df.columns:
        df["school_load"] = (
            df["children_school"] / (df["school_quota"] + 1e-5)
        ).astype(np.float32)

    # Ecology ordinal mapping
    if "ecology" in df.columns:
        ecology_map = {
            "no data": 0,
            "poor": 1,
            "satisfactory": 2,
            "good": 3,
            "excellent": 4,
        }
        df["ecology_score"] = (
            df["ecology"].map(ecology_map).fillna(0).astype(np.float32)
        )

    # Binary binary string encoding (yes/no)
    binary_cols = [
        "thermal_power_plant_raion",
        "incineration_raion",
        "oil_chemistry_raion",
        "radiation_raion",
        "railroad_terminal_raion",
        "big_market_raion",
        "nuclear_reactor_raion",
        "detention_facility_raion",
        "water_1line",
        "big_road1_1line",
        "railroad_1line",
        "culture_objects_top_25",
    ]
    for col in binary_cols:
        if col in df.columns:
            df[col] = (df[col].astype(str).str.lower() == "yes").astype(np.float32)

    # Enhanced spatial and infrastructure interaction features
    if "metro_km_avto" in df.columns and "metro_km_walk" in df.columns:
        df["metro_accessibility_idx"] = (
            1.0 / (df["metro_km_avto"] + df["metro_km_walk"] + 1e-3)
        ).astype(np.float32)

    if "kremlin_km" in df.columns:
        df["kremlin_accessibility_idx"] = (1.0 / (df["kremlin_km"] + 1e-3)).astype(
            np.float32
        )

    if "preschool_quota" in df.columns and "raion_popul" in df.columns:
        df["preschool_quota_per_capita"] = (
            df["preschool_quota"] / (df["raion_popul"] + 1e-5)
        ).astype(np.float32)

    if "school_quota" in df.columns and "raion_popul" in df.columns:
        df["school_quota_per_capita"] = (
            df["school_quota"] / (df["raion_popul"] + 1e-5)
        ).astype(np.float32)

    if "sport_count_3000" in df.columns and "area_m" in df.columns:
        df["sport_density_3000"] = (
            df["sport_count_3000"] / (df["area_m"] / 1e6 + 1e-5)
        ).astype(np.float32)

    if "shopping_centers_raion" in df.columns and "area_m" in df.columns:
        df["shopping_density"] = (
            df["shopping_centers_raion"] / (df["area_m"] / 1e6 + 1e-5)
        ).astype(np.float32)

    # Sub_area and product_type composite categorical key
    if "sub_area" in df.columns and "product_type" in df.columns:
        df["sub_area_x_product_type"] = (
            df["sub_area"].astype(str) + "___" + df["product_type"].astype(str)
        )

    # Macroeconomic purchasing power and currency interaction terms
    if "usdrub" in df.columns and "full_sq" in df.columns:
        df["usdrub_x_full_sq"] = (df["usdrub"] * df["full_sq"]).astype(np.float32)
    if "cpi" in df.columns and "full_sq" in df.columns:
        df["cpi_x_full_sq"] = (df["cpi"] * df["full_sq"]).astype(np.float32)
    if "oil_brent" in df.columns and "full_sq" in df.columns:
        df["oil_brent_x_full_sq"] = (df["oil_brent"] * df["full_sq"]).astype(np.float32)
    elif "brent" in df.columns and "full_sq" in df.columns:
        df["oil_brent_x_full_sq"] = (df["brent"] * df["full_sq"]).astype(np.float32)
    if "usdrub" in df.columns and "oil_brent" in df.columns:
        df["oil_ruble_val"] = (df["oil_brent"] * df["usdrub"]).astype(np.float32)

    return df


def load_and_merge_macro(df: pd.DataFrame) -> pd.DataFrame:
    """Attempt to load and merge macroeconomic indicators with rolling dynamics and momentum."""
    df = df.copy()
    macro_paths = [
        "./input/macro.parquet",
        "./input/macro.csv",
        "./input/original/macro.csv",
    ]
    macro_df = None
    for path in macro_paths:
        if os.path.exists(path):
            if path.endswith(".parquet"):
                macro_df = pd.read_parquet(path)
            else:
                macro_df = pd.read_csv(path)
            break

    if (
        macro_df is not None
        and "timestamp" in macro_df.columns
        and "timestamp" in df.columns
    ):
        macro_df["dt"] = pd.to_datetime(macro_df["timestamp"])
        macro_df = macro_df.sort_values("dt").reset_index(drop=True)

        dynamic_macro_cols = [
            "usdrub",
            "eurrub",
            "brent",
            "oil_brent",
            "cpi",
            "ppi",
            "deposits_rate",
            "mortgage_rate",
            "micex",
            "rts",
            "salary",
            "unemployment",
        ]

        for col in dynamic_macro_cols:
            if col in macro_df.columns:
                macro_df[col] = pd.to_numeric(macro_df[col], errors="coerce")
                macro_df[f"{col}_roll30"] = (
                    macro_df[col].rolling(window=30, min_periods=1).mean().astype(np.float32)
                )
                macro_df[f"{col}_roll90"] = (
                    macro_df[col].rolling(window=90, min_periods=1).mean().astype(np.float32)
                )
                macro_df[f"{col}_mom30"] = (
                    (macro_df[col] - macro_df[f"{col}_roll30"])
                    / (macro_df[f"{col}_roll30"] + 1e-5)
                ).astype(np.float32)
                macro_df[f"{col}_mom90"] = (
                    (macro_df[col] - macro_df[f"{col}_roll90"])
                    / (macro_df[f"{col}_roll90"] + 1e-5)
                ).astype(np.float32)

        macro_df["timestamp_str"] = macro_df["dt"].dt.strftime("%Y-%m-%d")
        df["timestamp_str"] = pd.to_datetime(df["timestamp"]).dt.strftime("%Y-%m-%d")

        drop_cols_macro = ["dt", "timestamp"]
        macro_sub = macro_df.drop(
            columns=[c for c in drop_cols_macro if c in macro_df.columns]
        ).drop_duplicates(subset=["timestamp_str"])

        df = df.merge(macro_sub, on="timestamp_str", how="left")
        if "timestamp_str" in df.columns:
            df = df.drop(columns=["timestamp_str"])

    return df


def apply_leak_free_encodings(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """Fit categorical mappings and sub_area statistics strictly on training data."""
    train = train_df.copy()
    test = test_df.copy()

    # 1. sub_area spatial aggregations calculated purely on train
    if "sub_area" in train.columns:
        sub_area_stats = (
            train.groupby("sub_area")
            .agg(
                sub_area_mean_full_sq=("full_sq", "mean"),
                sub_area_median_full_sq=("full_sq", "median"),
                sub_area_std_full_sq=("full_sq", "std"),
                sub_area_mean_build_year=("build_year", "mean"),
                sub_area_mean_life_sq=("life_sq", "mean"),
                sub_area_count=("full_sq", "count"),
            )
            .reset_index()
        )

        train = train.merge(sub_area_stats, on="sub_area", how="left")
        test = test.merge(sub_area_stats, on="sub_area", how="left")

        for d in [train, test]:
            d["ratio_full_sq_to_sub_area_mean"] = (
                d["full_sq"] / (d["sub_area_mean_full_sq"] + 1e-5)
            ).astype(np.float32)
            d["diff_full_sq_to_sub_area_mean"] = (
                d["full_sq"] - d["sub_area_mean_full_sq"]
            ).astype(np.float32)
            d["ratio_full_sq_to_sub_area_median"] = (
                d["full_sq"] / (d["sub_area_median_full_sq"] + 1e-5)
            ).astype(np.float32)
            if "build_year" in d.columns:
                d["diff_build_year_to_sub_area_mean"] = (
                    d["build_year"] - d["sub_area_mean_build_year"]
                ).astype(np.float32)
            if "building_age" in d.columns and "year" in d.columns:
                d["sub_area_mean_age"] = (
                    d["year"] - d["sub_area_mean_build_year"]
                ).astype(np.float32)
                d["diff_age_to_sub_area_mean"] = (
                    d["building_age"] - d["sub_area_mean_age"]
                ).astype(np.float32)

    # Out-of-fold product_type aggregations
    if "product_type" in train.columns:
        prod_stats = (
            train.groupby("product_type")
            .agg(
                prod_mean_full_sq=("full_sq", "mean"),
                prod_median_full_sq=("full_sq", "median"),
                prod_count=("full_sq", "count"),
            )
            .reset_index()
        )
        train = train.merge(prod_stats, on="product_type", how="left")
        test = test.merge(prod_stats, on="product_type", how="left")

    # 2. Categorical label encodings based on train frequencies
    cat_cols = train.select_dtypes(include=["object", "category"]).columns.tolist()
    cat_cols = [c for c in cat_cols if c not in ["timestamp", "id", "price_doc"]]

    for col in cat_cols:
        unique_cats = train[col].dropna().unique().tolist()
        cat_to_int = {val: idx + 1 for idx, val in enumerate(unique_cats)}

        train[col + "_cat"] = train[col].map(cat_to_int).fillna(0).astype(np.int32)
        test[col + "_cat"] = test[col].map(cat_to_int).fillna(0).astype(np.int32)

        train.drop(columns=[col], inplace=True)
        test.drop(columns=[col], inplace=True)

    return train, test


def compute_nested_oof_target_encoding(
    cat_tr: pd.Series,
    target_tr: pd.Series,
    cat_val: pd.Series,
    cat_test: pd.Series,
    m: float = 10.0,
    n_inner_splits: int = 5,
    random_state: int = 42,
):
    """Computes strictly leak-free nested out-of-fold target encoding for train, val, and test."""
    global_mean = float(target_tr.mean())
    train_encoded = np.full(len(cat_tr), global_mean, dtype=np.float32)

    inner_kf = KFold(n_splits=n_inner_splits, shuffle=True, random_state=random_state)
    for in_tr_idx, in_va_idx in inner_kf.split(cat_tr):
        in_cat_tr = cat_tr.iloc[in_tr_idx]
        in_tgt_tr = target_tr.iloc[in_tr_idx]
        in_mean = float(in_tgt_tr.mean())

        stats = (
            pd.DataFrame({"cat": in_cat_tr, "target": in_tgt_tr})
            .groupby("cat")["target"]
            .agg(count="count", mean="mean")
        )
        smooth = (stats["count"] * stats["mean"] + m * in_mean) / (stats["count"] + m)
        mapping = smooth.to_dict()
        train_encoded[in_va_idx] = (
            cat_tr.iloc[in_va_idx].map(mapping).fillna(in_mean).values.astype(np.float32)
        )

    full_stats = (
        pd.DataFrame({"cat": cat_tr, "target": target_tr})
        .groupby("cat")["target"]
        .agg(count="count", mean="mean")
    )
    full_smooth = (full_stats["count"] * full_stats["mean"] + m * global_mean) / (
        full_stats["count"] + m
    )
    full_mapping = full_smooth.to_dict()

    val_encoded = cat_val.map(full_mapping).fillna(global_mean).values.astype(np.float32)
    test_encoded = cat_test.map(full_mapping).fillna(global_mean).values.astype(np.float32)

    return train_encoded, val_encoded, test_encoded


# =========================================================================
# 3. PYTORCH TABULAR RESNET MODEL ARCHITECTURE
# =========================================================================


class TabularResidualBlock(nn.Module):
    """Residual block with Skip Connection, BatchNorm, LeakyReLU, and Dropout."""

    def __init__(self, hidden_dim: int, dropout_rate: float = 0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class TabularResNet(nn.Module):
    """Deep Residual MLP designed for high-dimensional tabular real estate features."""

    def __init__(
        self,
        num_features: int,
        hidden_dims: list = [256, 128],
        num_res_blocks: int = 2,
        dropout_rate: float = 0.2,
        output_dim: int = 1,
    ):
        super().__init__()
        self.input_norm = nn.BatchNorm1d(num_features)

        self.input_projection = nn.Sequential(
            nn.Linear(num_features, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.LeakyReLU(negative_slope=0.01),
        )

        res_layers = []
        for _ in range(num_res_blocks):
            res_layers.append(
                TabularResidualBlock(
                    hidden_dim=hidden_dims[0], dropout_rate=dropout_rate
                )
            )
        self.residual_stack = nn.Sequential(*res_layers)

        down_layers = []
        in_dim = hidden_dims[0]
        for out_dim in hidden_dims[1:]:
            down_layers.extend(
                [
                    nn.Linear(in_dim, out_dim),
                    nn.BatchNorm1d(out_dim),
                    nn.LeakyReLU(negative_slope=0.01),
                    nn.Dropout(p=dropout_rate),
                ]
            )
            in_dim = out_dim
        self.down_projection = (
            nn.Sequential(*down_layers) if down_layers else nn.Identity()
        )

        self.head = nn.Linear(in_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_norm(x)
        x = self.input_projection(x)
        x = self.residual_stack(x)
        x = self.down_projection(x)
        return self.head(x)


def create_optimizer_and_scheduler(
    model: nn.Module,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    total_epochs: int = 25,
):
    """Instantiates AdamW optimizer with Cosine Annealing learning rate schedule."""
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_epochs, eta_min=learning_rate * 0.01
    )
    return optimizer, scheduler


def train_pytorch_resnet(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_te: np.ndarray,
    device: torch.device,
    epochs: int = 25,
    batch_size: int = 256,
    lr: float = 1e-3,
):
    """Trains TabularResNet model on log-target with early stopping."""
    num_features = X_tr.shape[1]
    model = TabularResNet(
        num_features=num_features,
        hidden_dims=[256, 128],
        num_res_blocks=2,
        dropout_rate=0.2,
        output_dim=1,
    ).to(device)

    optimizer, scheduler = create_optimizer_and_scheduler(
        model, learning_rate=lr, weight_decay=1e-4, total_epochs=epochs
    )
    criterion = nn.MSELoss()

    train_dataset = TensorDataset(
        torch.tensor(X_tr, dtype=torch.float32),
        torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1),
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=False
    )

    best_val_loss = float("inf")
    best_weights = None

    val_tensor_x = torch.tensor(X_val, dtype=torch.float32).to(device)
    val_tensor_y = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1).to(device)

    for epoch in range(epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_preds = model(val_tensor_x)
            val_loss = criterion(val_preds, val_tensor_y).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_weights is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_weights.items()})

    model.eval()
    with torch.no_grad():
        oof_pred = model(val_tensor_x).cpu().numpy().flatten()
        test_tensor_x = torch.tensor(X_te, dtype=torch.float32).to(device)
        test_pred = model(test_tensor_x).cpu().numpy().flatten()

    return oof_pred, test_pred


# =========================================================================
# 4. MAIN TRAINING & INFERENCE PIPELINE
# =========================================================================


def main():
    os.makedirs("./working", exist_ok=True)
    os.makedirs("./submission", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Data
    train_raw = pd.read_parquet("./input/train.parquet")
    test_raw = pd.read_parquet("./input/test.parquet")

    # Merge macro indicators if present
    train_macro = load_and_merge_macro(train_raw)
    test_macro = load_and_merge_macro(test_raw)

    # Clean structural anomalies & engineer features
    train_clean = clean_structural_features(train_macro)
    test_clean = clean_structural_features(test_macro)

    train_feat = engineer_domain_features(train_clean)
    test_feat = engineer_domain_features(test_clean)

    # Encode categoricals and sub_area statistics leak-free
    train_proc, test_proc = apply_leak_free_encodings(train_feat, test_feat)

    # Log-transform target for RMSLE metric alignment
    train_proc["target_log"] = np.log1p(
        train_proc["price_doc"].astype(np.float64)
    ).astype(np.float32)

    # Drop non-feature identifiers
    drop_cols = ["timestamp"]
    for d in [train_proc, test_proc]:
        for col in drop_cols:
            if col in d.columns:
                d.drop(columns=[col], inplace=True)

    # Downcast datatypes to optimize memory and execution speed
    for d in [train_proc, test_proc]:
        for col in d.columns:
            if col in ["id", "price_doc", "target_log"]:
                continue
            if d[col].dtype == np.float64:
                d[col] = d[col].astype(np.float32)
            elif d[col].dtype == np.int64:
                d[col] = d[col].astype(np.int32)

    excluded = {"id", "price_doc", "target_log"}
    feature_cols = [c for c in train_proc.columns if c not in excluded]

    X_full = train_proc[feature_cols].copy()
    y_full_raw = train_proc["price_doc"].values.astype(np.float64)
    y_full_log = train_proc["target_log"].values.astype(np.float32)

    X_test = test_proc.reindex(columns=feature_cols, fill_value=0.0).copy()
    n_samples = len(train_proc)
    n_test = len(test_proc)

    # Compute log price-per-square-meter for auxiliary target encoding
    sq_clipped = np.clip(
        train_proc["full_sq"].fillna(train_proc["full_sq"].median()).values,
        5.0,
        1000.0,
    )
    y_full_log_ppsq = np.log1p(y_full_raw / sq_clipped).astype(np.float32)

    # Set up 5-Fold Cross-Validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    kfold_splits = list(kf.split(X_full))

    # GBDT and CatBoost Model Hyperparameters
    lgb_params_1 = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "learning_rate": 0.03,
        "num_leaves": 45,
        "max_depth": 7,
        "min_data_in_leaf": 25,
        "subsample": 0.85,
        "subsample_freq": 1,
        "colsample_bytree": 0.75,
        "reg_alpha": 0.5,
        "reg_lambda": 1.5,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }

    lgb_params_2 = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "learning_rate": 0.02,
        "num_leaves": 63,
        "max_depth": 9,
        "min_data_in_leaf": 35,
        "subsample": 0.75,
        "subsample_freq": 1,
        "colsample_bytree": 0.65,
        "reg_alpha": 1.0,
        "reg_lambda": 3.0,
        "random_state": 2024,
        "n_jobs": -1,
        "verbose": -1,
    }

    callbacks_early = [
        lgb.early_stopping(stopping_rounds=50, verbose=False),
        lgb.log_evaluation(period=0),
    ]

    xgb_params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "learning_rate": 0.03,
        "max_depth": 6,
        "subsample": 0.80,
        "colsample_bytree": 0.70,
        "reg_alpha": 0.5,
        "reg_lambda": 1.5,
        "random_state": 42,
        "n_estimators": 1500,
        "n_jobs": -1,
    }

    cb_params = {
        "iterations": 1500,
        "learning_rate": 0.035,
        "depth": 6,
        "l2_leaf_reg": 4.0,
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "random_seed": 42,
        "verbose": False,
        "thread_count": -1,
    }

    oof_preds_lgb1 = np.zeros(n_samples, dtype=np.float64)
    oof_preds_lgb2 = np.zeros(n_samples, dtype=np.float64)
    oof_preds_xgb = np.zeros(n_samples, dtype=np.float64)
    oof_preds_cb = np.zeros(n_samples, dtype=np.float64)
    oof_preds_nn = np.zeros(n_samples, dtype=np.float64)

    test_preds_lgb1 = np.zeros(n_test, dtype=np.float64)
    test_preds_lgb2 = np.zeros(n_test, dtype=np.float64)
    test_preds_xgb = np.zeros(n_test, dtype=np.float64)
    test_preds_cb = np.zeros(n_test, dtype=np.float64)
    test_preds_nn = np.zeros(n_test, dtype=np.float64)

    # Cross-Validation Loop
    for fold_idx, (train_idx, val_idx) in enumerate(kfold_splits):
        # Leak-free nested out-of-fold target encoding inside fold
        te_features_tr = {}
        te_features_va = {}
        te_features_te = {}

        if "sub_area_cat" in X_full.columns:
            cat_tr = X_full["sub_area_cat"].iloc[train_idx]
            cat_va = X_full["sub_area_cat"].iloc[val_idx]
            cat_te = X_test["sub_area_cat"]

            sub_p_tr, sub_p_va, sub_p_te = compute_nested_oof_target_encoding(
                cat_tr,
                pd.Series(y_full_log[train_idx], index=train_idx),
                cat_va,
                cat_te,
                m=10.0,
            )
            sub_ppsq_tr, sub_ppsq_va, sub_ppsq_te = (
                compute_nested_oof_target_encoding(
                    cat_tr,
                    pd.Series(y_full_log_ppsq[train_idx], index=train_idx),
                    cat_va,
                    cat_te,
                    m=10.0,
                )
            )
            te_features_tr["te_sub_area_price"] = sub_p_tr
            te_features_va["te_sub_area_price"] = sub_p_va
            te_features_te["te_sub_area_price"] = sub_p_te

            te_features_tr["te_sub_area_ppsq"] = sub_ppsq_tr
            te_features_va["te_sub_area_ppsq"] = sub_ppsq_va
            te_features_te["te_sub_area_ppsq"] = sub_ppsq_te

        if "sub_area_x_product_type_cat" in X_full.columns:
            cat_tr = X_full["sub_area_x_product_type_cat"].iloc[train_idx]
            cat_va = X_full["sub_area_x_product_type_cat"].iloc[val_idx]
            cat_te = X_test["sub_area_x_product_type_cat"]

            combo_p_tr, combo_p_va, combo_p_te = (
                compute_nested_oof_target_encoding(
                    cat_tr,
                    pd.Series(y_full_log[train_idx], index=train_idx),
                    cat_va,
                    cat_te,
                    m=15.0,
                )
            )
            combo_ppsq_tr, combo_ppsq_va, combo_ppsq_te = (
                compute_nested_oof_target_encoding(
                    cat_tr,
                    pd.Series(y_full_log_ppsq[train_idx], index=train_idx),
                    cat_va,
                    cat_te,
                    m=15.0,
                )
            )
            te_features_tr["te_combo_price"] = combo_p_tr
            te_features_va["te_combo_price"] = combo_p_va
            te_features_te["te_combo_price"] = combo_p_te

            te_features_tr["te_combo_ppsq"] = combo_ppsq_tr
            te_features_va["te_combo_ppsq"] = combo_ppsq_va
            te_features_te["te_combo_ppsq"] = combo_ppsq_te

        X_tr = X_full.iloc[train_idx].copy()
        X_va = X_full.iloc[val_idx].copy()
        X_te = X_test.copy()

        for col_name, vals in te_features_tr.items():
            X_tr[col_name] = vals
            X_va[col_name] = te_features_va[col_name]
            X_te[col_name] = te_features_te[col_name]

        y_tr_log = y_full_log[train_idx]
        y_va_log = y_full_log[val_idx]

        # Model 1: LightGBM
        dtrain_1 = lgb.Dataset(X_tr, label=y_tr_log)
        dval_1 = lgb.Dataset(X_va, label=y_va_log, reference=dtrain_1)
        booster_1 = lgb.train(
            lgb_params_1,
            train_set=dtrain_1,
            num_boost_round=1500,
            valid_sets=[dval_1],
            callbacks=callbacks_early,
        )
        oof_preds_lgb1[val_idx] = booster_1.predict(
            X_va, num_iteration=booster_1.best_iteration
        )
        test_preds_lgb1 += booster_1.predict(
            X_te, num_iteration=booster_1.best_iteration
        ) / len(kfold_splits)

        # Model 2: LightGBM Deep
        dtrain_2 = lgb.Dataset(X_tr, label=y_tr_log)
        dval_2 = lgb.Dataset(X_va, label=y_va_log, reference=dtrain_2)
        booster_2 = lgb.train(
            lgb_params_2,
            train_set=dtrain_2,
            num_boost_round=1800,
            valid_sets=[dval_2],
            callbacks=callbacks_early,
        )
        oof_preds_lgb2[val_idx] = booster_2.predict(
            X_va, num_iteration=booster_2.best_iteration
        )
        test_preds_lgb2 += booster_2.predict(
            X_te, num_iteration=booster_2.best_iteration
        ) / len(kfold_splits)

        # Model 3: XGBoost Hist Regressor
        xgb_model = xgb.XGBRegressor(
            **xgb_params,
            early_stopping_rounds=50,
        )
        xgb_model.fit(
            X_tr,
            y_tr_log,
            eval_set=[(X_va, y_va_log)],
            verbose=False,
        )
        oof_preds_xgb[val_idx] = xgb_model.predict(X_va)
        test_preds_xgb += xgb_model.predict(X_te) / len(kfold_splits)

        # Model 4: CatBoost Regressor
        cb_model = CatBoostRegressor(**cb_params)
        cb_model.fit(
            X_tr,
            y_tr_log,
            eval_set=(X_va, y_va_log),
            early_stopping_rounds=50,
            verbose=False,
        )
        oof_preds_cb[val_idx] = cb_model.predict(X_va)
        test_preds_cb += cb_model.predict(X_te) / len(kfold_splits)

        # Model 5: PyTorch TabularResNet
        impute_vals = X_tr.median()
        X_tr_imp = X_tr.fillna(impute_vals).values.astype(np.float32)
        X_va_imp = X_va.fillna(impute_vals).values.astype(np.float32)
        X_te_imp = X_te.fillna(impute_vals).values.astype(np.float32)

        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr_imp)
        X_va_sc = scaler.transform(X_va_imp)
        X_te_sc = scaler.transform(X_te_imp)

        nn_val_pred_log, nn_test_pred_log = train_pytorch_resnet(
            X_tr=X_tr_sc,
            y_tr=y_tr_log,
            X_val=X_va_sc,
            y_val=y_va_log,
            X_te=X_te_sc,
            device=device,
            epochs=25,
            batch_size=256,
            lr=1e-3,
        )
        oof_preds_nn[val_idx] = nn_val_pred_log
        test_preds_nn += nn_test_pred_log / len(kfold_splits)

    # Out-of-Fold Ensemble Blending via SLSQP Optimization
    oof_matrix = np.column_stack(
        [
            oof_preds_lgb1,
            oof_preds_lgb2,
            oof_preds_xgb,
            oof_preds_cb,
            oof_preds_nn,
        ]
    )
    test_matrix = np.column_stack(
        [
            test_preds_lgb1,
            test_preds_lgb2,
            test_preds_xgb,
            test_preds_cb,
            test_preds_nn,
        ]
    )

    def ensemble_objective(weights):
        w = np.array(weights)
        pred_log = np.dot(oof_matrix, w)
        pred_raw = np.expm1(pred_log)
        return compute_rmsle(y_full_raw, pred_raw, is_log_target=False)

    num_models = oof_matrix.shape[1]
    init_weights = np.ones(num_models, dtype=np.float64) / num_models
    bounds = [(0.0, 1.0) for _ in range(num_models)]
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

    opt_result = minimize(
        ensemble_objective,
        init_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-7},
    )

    opt_weights = np.maximum(0, opt_result.x)
    opt_weights /= np.sum(opt_weights)

    oof_blend_log = np.dot(oof_matrix, opt_weights)
    oof_blend_raw = np.expm1(oof_blend_log)
    final_oof_rmsle = compute_rmsle(y_full_raw, oof_blend_raw, is_log_target=False)

    # Test Inference & Prediction Assembly
    test_blend_log = np.dot(test_matrix, opt_weights)
    test_blend_raw = np.expm1(test_blend_log)

    median_train_price = float(np.median(y_full_raw))
    test_blend_raw = np.nan_to_num(
        test_blend_raw, nan=median_train_price, posinf=median_train_price
    )
    test_blend_raw = np.clip(test_blend_raw, 100000.0, None)

    # Generate Submission
    if "id" in test_raw.columns:
        test_ids = test_raw["id"].values
    else:
        test_ids = np.arange(len(test_blend_raw))

    submission = pd.DataFrame({"id": test_ids, "price_doc": test_blend_raw})
    submission_path = "./submission/submission.csv"
    submission.to_csv(submission_path, index=False)

    print(f"Final Validation Score: {final_oof_rmsle:.5f}")


if __name__ == "__main__":
    main()