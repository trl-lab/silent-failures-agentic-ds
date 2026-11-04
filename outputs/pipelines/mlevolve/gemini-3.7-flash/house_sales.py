import os
import warnings
from catboost import CatBoostRegressor
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.cluster import KMeans
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb

warnings.filterwarnings("ignore")


# ==============================================================================
# 0. Global Reproducibility
# ==============================================================================
def seed_everything(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ==============================================================================
# 1. Feature Engineering & Preprocessing
# ==============================================================================
def haversine_np(lat1, lon1, lat2, lon2):
    """
    Calculate the great-circle distance between points on the earth in kilometers.
    """
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.clip(np.sqrt(a), -1.0, 1.0))
    return 6371.0 * c


def parse_date_features(df):
    """
    Parse sale date into rich temporal features.
    """
    if df["date"].dtype == "object":
        clean_date_str = df["date"].astype(str).str.replace(r"T.*$", "", regex=True)
        dates = pd.to_datetime(clean_date_str, format="%Y%m%d", errors="coerce")
        if dates.isna().all():
            dates = pd.to_datetime(df["date"], errors="coerce")
    else:
        dates = pd.to_datetime(df["date"], errors="coerce")

    sale_year = dates.dt.year.fillna(2014).astype(int)
    sale_month = dates.dt.month.fillna(6).astype(int)
    sale_day = dates.dt.day.fillna(15).astype(int)
    sale_dayofweek = dates.dt.dayofweek.fillna(0).astype(int)
    sale_dayofyear = dates.dt.dayofyear.fillna(180).astype(int)
    sale_quarter = dates.dt.quarter.fillna(2).astype(int)

    ref_date = pd.Timestamp("2014-05-01")
    days_since_start = (dates - ref_date).dt.days.fillna(0).astype(float)

    return {
        "sale_year": sale_year,
        "sale_month": sale_month,
        "sale_day": sale_day,
        "sale_dayofweek": sale_dayofweek,
        "sale_dayofyear": sale_dayofyear,
        "sale_quarter": sale_quarter,
        "days_since_start": days_since_start,
    }


def engineer_base_features(df):
    """
    Create domain-specific structural, geographic, and architectural features.
    """
    df = df.copy()

    # Ensure numeric columns are properly converted from Categorical/Object dtypes
    for col in df.columns:
        if col not in ["date", "id"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Temporal features
    date_feats = parse_date_features(df)
    for col, val in date_feats.items():
        df[col] = val

    # Structural & Age Features
    df["house_age_at_sale"] = np.maximum(0, df["sale_year"] - df["yr_built"])
    df["is_renovated"] = (df["yr_renovated"] > 0).astype(int)
    effective_yr = np.where(df["yr_renovated"] > 0, df["yr_renovated"], df["yr_built"])
    df["effective_age_at_sale"] = np.maximum(0, df["sale_year"] - effective_yr)
    df["years_since_renovation"] = np.where(
        df["yr_renovated"] > 0, np.maximum(0, df["sale_year"] - df["yr_renovated"]), 0
    )

    # Spatial Hub Distances (Puget Sound Key Regional Centers & Waterfronts)
    hubs = {
        "dist_downtown_seattle": (47.6062, -122.3321),
        "dist_bellevue": (47.6101, -122.2015),
        "dist_redmond_tech": (47.6740, -122.1215),
        "dist_medina": (47.6209, -122.2276),
        "dist_seatac_airport": (47.4502, -122.3088),
        "dist_south_lake_union": (47.6253, -122.3377),
        "dist_university_dist": (47.6553, -122.3035),
        "dist_lake_washington_center": (47.6200, -122.2500),
        "dist_lake_sammamish_center": (47.6100, -122.0800),
        "dist_elliott_bay": (47.6000, -122.3600),
        "dist_kirkland_waterfront": (47.6760, -122.2060),
        "dist_mercer_island": (47.5700, -122.2200),
    }

    for hub_name, (hub_lat, hub_lon) in hubs.items():
        df[hub_name] = haversine_np(
            df["lat"].values, df["long"].values, hub_lat, hub_lon
        )

    # Core and Shoreline Proximities
    df["min_dist_major_core"] = np.minimum(
        df["dist_downtown_seattle"], df["dist_bellevue"]
    )
    df["min_dist_waterfront_proxy"] = np.minimum.reduce(
        [
            df["dist_lake_washington_center"],
            df["dist_lake_sammamish_center"],
            df["dist_elliott_bay"],
            df["dist_kirkland_waterfront"],
            df["dist_mercer_island"],
        ]
    )

    # Coordinate transformations
    df["lat_plus_long"] = df["lat"] + df["long"]
    df["lat_minus_long"] = df["lat"] - df["long"]
    df["radial_dist_center"] = np.sqrt(
        (df["lat"] - 47.6062) ** 2 + (df["long"] - (-122.3321)) ** 2
    )

    # Room and Area Ratios
    df["total_rooms"] = df["bedrooms"] + df["bathrooms"]
    df["sqft_living_per_bed"] = df["sqft_living"] / (df["bedrooms"] + 0.1)
    df["sqft_living_per_bath"] = df["sqft_living"] / (df["bathrooms"] + 0.1)
    df["sqft_living_per_room"] = df["sqft_living"] / (df["total_rooms"] + 0.1)
    df["bed_bath_ratio"] = df["bedrooms"] / (df["bathrooms"] + 0.1)

    # Lot & Structure Proportions
    df["living_lot_ratio"] = df["sqft_living"] / (df["sqft_lot"] + 1.0)
    df["sqft_above_ratio"] = df["sqft_above"] / (df["sqft_living"] + 1.0)
    df["sqft_basement_ratio"] = df["sqft_basement"] / (df["sqft_living"] + 1.0)
    df["sqft_living_ratio_total"] = df["sqft_living"] / (
        df["sqft_living"] + df["sqft_lot"] + 1.0
    )
    df["has_basement"] = (df["sqft_basement"] > 0).astype(int)

    # Neighborhood Comparison Ratios
    df["living_vs_living15_ratio"] = df["sqft_living"] / (df["sqft_living15"] + 1.0)
    df["living_vs_living15_diff"] = df["sqft_living"] - df["sqft_living15"]
    df["lot_vs_lot15_ratio"] = df["sqft_lot"] / (df["sqft_lot15"] + 1.0)
    df["lot_vs_lot15_diff"] = df["sqft_lot"] - df["sqft_lot15"]

    # Quality and Luxury Interaction Indices
    df["quality_condition_interaction"] = df["grade"] * df["condition"]
    df["grade_sqft_interaction"] = df["grade"] * df["sqft_living"]
    df["grade_above_interaction"] = df["grade"] * df["sqft_above"]
    df["view_waterfront_interaction"] = df["view"] * (df["waterfront"] * 3.0 + 1.0)
    df["grade_condition_ratio"] = df["grade"] / (df["condition"] + 0.1)
    df["overall_luxury_score"] = (
        df["grade"] * 2.0 + df["condition"] + df["view"] * 1.5 + df["waterfront"] * 5.0
    )
    df["luxury_waterfront_view_score"] = (
        (df["grade"] ** 2) * (df["view"] + 1.0) * (df["waterfront"] * 2.0 + 1.0)
    )

    # Temporal Market Appreciation Interactions
    df["days_since_start_x_sqft"] = df["days_since_start"] * df["sqft_living"]
    df["days_since_start_x_grade"] = df["days_since_start"] * df["grade"]
    df["days_since_start_x_lat"] = df["days_since_start"] * df["lat"]

    return df


def apply_leak_free_aggregations(train_df, test_df, num_folds=5, random_state=42):
    """
    Compute group aggregations and out-of-fold target statistics strictly preventing leakage.
    """
    train = train_df.copy().reset_index(drop=True)
    test = test_df.copy().reset_index(drop=True)

    group_col = "zipcode"
    num_cols_to_agg = [
        "sqft_living",
        "grade",
        "yr_built",
        "house_age_at_sale",
        "lat",
        "long",
    ]

    # Global non-target group statistics computed strictly on train
    zip_stats = (
        train.groupby(group_col)[num_cols_to_agg]
        .agg(["mean", "std", "median"])
        .fillna(0)
    )
    zip_stats.columns = [f"zip_{col}_{stat}" for col, stat in zip_stats.columns]
    zip_stats = zip_stats.reset_index()

    zip_counts = train[group_col].value_counts().rename("zip_frequency").reset_index()
    zip_counts.columns = [group_col, "zip_frequency"]

    train = train.merge(zip_stats, on=group_col, how="left")
    test = test.merge(zip_stats, on=group_col, how="left")
    train = train.merge(zip_counts, on=group_col, how="left")
    test = test.merge(zip_counts, on=group_col, how="left")

    stat_cols = [c for c in zip_stats.columns if c != group_col] + ["zip_frequency"]
    for col in stat_cols:
        default_val = train[col].median()
        train[col] = train[col].fillna(default_val)
        test[col] = test[col].fillna(default_val)

    # Out-of-fold target encoding, Spatial KNN, and KMeans clustering
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=random_state)
    train["fold"] = -1
    train["oof_target_enc_zip_mean"] = np.nan
    train["oof_target_enc_zip_median"] = np.nan
    train["oof_target_enc_zip_price_sqft"] = np.nan

    knn_ks = [5, 15, 30]
    for k in knn_ks:
        train[f"knn_{k}_price_mean"] = np.nan
        train[f"knn_{k}_price_sqft_mean"] = np.nan
        train[f"knn_{k}_grade_mean"] = np.nan
        train[f"knn_{k}_dist_mean"] = np.nan
        test[f"knn_{k}_price_mean"] = np.nan
        test[f"knn_{k}_price_sqft_mean"] = np.nan
        test[f"knn_{k}_grade_mean"] = np.nan
        test[f"knn_{k}_dist_mean"] = np.nan
    train["knn_nearest_dist"] = np.nan
    test["knn_nearest_dist"] = np.nan

    cluster_resolutions = [25, 60]
    for c_res in cluster_resolutions:
        train[f"cluster_{c_res}_price_mean"] = np.nan
        train[f"cluster_{c_res}_price_sqft_mean"] = np.nan
        train[f"cluster_{c_res}_grade_mean"] = np.nan
        test[f"cluster_{c_res}_price_mean"] = np.nan
        test[f"cluster_{c_res}_price_sqft_mean"] = np.nan
        test[f"cluster_{c_res}_grade_mean"] = np.nan

    global_price_mean = train["price"].mean()
    global_price_median = train["price"].median()
    global_price_sqft_mean = (train["price"] / (train["sqft_living"] + 1.0)).mean()
    global_grade_mean = train["grade"].mean()

    for fold, (trn_idx, val_idx) in enumerate(kf.split(train)):
        train.loc[val_idx, "fold"] = fold
        trn_sub = train.iloc[trn_idx]
        val_sub = train.iloc[val_idx]

        # 1. Zipcode Target Encoding
        mean_s = trn_sub.groupby(group_col)["price"].mean().rename("target_mean")
        med_s = trn_sub.groupby(group_col)["price"].median().rename("target_median")
        psqft_s = (
            (trn_sub["price"] / (trn_sub["sqft_living"] + 1.0))
            .groupby(trn_sub[group_col])
            .mean()
            .rename("target_price_sqft")
        )
        fold_target_stats = pd.concat([mean_s, med_s, psqft_s], axis=1).reset_index()

        val_merged = val_sub[[group_col]].merge(
            fold_target_stats, on=group_col, how="left"
        )
        train.loc[val_idx, "oof_target_enc_zip_mean"] = (
            val_merged["target_mean"].fillna(global_price_mean).values
        )
        train.loc[val_idx, "oof_target_enc_zip_median"] = (
            val_merged["target_median"].fillna(global_price_median).values
        )
        train.loc[val_idx, "oof_target_enc_zip_price_sqft"] = (
            val_merged["target_price_sqft"].fillna(global_price_sqft_mean).values
        )

        # 2. Spatial KNN Comparables (Fitted strictly on train fold)
        coords_trn_rad = np.radians(trn_sub[["lat", "long"]].values)
        coords_val_rad = np.radians(val_sub[["lat", "long"]].values)

        knn_model = NearestNeighbors(n_neighbors=30, metric="haversine", n_jobs=-1)
        knn_model.fit(coords_trn_rad)
        val_dists, val_indices = knn_model.kneighbors(coords_val_rad, n_neighbors=30)
        val_dists_km = val_dists * 6371.0

        trn_prices = trn_sub["price"].values
        trn_psqft = (trn_sub["price"] / (trn_sub["sqft_living"] + 1.0)).values
        trn_grades = trn_sub["grade"].values

        for k in knn_ks:
            k_idx = val_indices[:, :k]
            k_d = val_dists_km[:, :k]
            train.loc[val_idx, f"knn_{k}_price_mean"] = trn_prices[k_idx].mean(axis=1)
            train.loc[val_idx, f"knn_{k}_price_sqft_mean"] = trn_psqft[k_idx].mean(axis=1)
            train.loc[val_idx, f"knn_{k}_grade_mean"] = trn_grades[k_idx].mean(axis=1)
            train.loc[val_idx, f"knn_{k}_dist_mean"] = k_d.mean(axis=1)
        train.loc[val_idx, "knn_nearest_dist"] = val_dists_km[:, 0]

        # 3. Spatial Multi-Resolution Clustering (Fitted strictly on train fold)
        for c_res in cluster_resolutions:
            km = KMeans(n_clusters=c_res, random_state=random_state, n_init=5)
            c_trn = km.fit_predict(trn_sub[["lat", "long"]].values)
            c_val = km.predict(val_sub[["lat", "long"]].values)

            c_stats = (
                pd.DataFrame({
                    "cluster": c_trn,
                    "price": trn_prices,
                    "psqft": trn_psqft,
                    "grade": trn_grades,
                })
                .groupby("cluster")
                .agg(
                    c_price_mean=("price", "mean"),
                    c_psqft_mean=("psqft", "mean"),
                    c_grade_mean=("grade", "mean"),
                )
                .reset_index()
            )

            val_c_df = pd.DataFrame({"cluster": c_val}).merge(c_stats, on="cluster", how="left")
            train.loc[val_idx, f"cluster_{c_res}_price_mean"] = val_c_df["c_price_mean"].fillna(global_price_mean).values
            train.loc[val_idx, f"cluster_{c_res}_price_sqft_mean"] = val_c_df["c_psqft_mean"].fillna(global_price_sqft_mean).values
            train.loc[val_idx, f"cluster_{c_res}_grade_mean"] = val_c_df["c_grade_mean"].fillna(global_grade_mean).values

    # Full Train Statistics applied to Test Set
    mean_full = train.groupby(group_col)["price"].mean().rename("target_mean")
    med_full = train.groupby(group_col)["price"].median().rename("target_median")
    psqft_full = (
        (train["price"] / (train["sqft_living"] + 1.0))
        .groupby(train[group_col])
        .mean()
        .rename("target_price_sqft")
    )
    full_target_stats = pd.concat(
        [mean_full, med_full, psqft_full], axis=1
    ).reset_index()

    test_merged = test[[group_col]].merge(full_target_stats, on=group_col, how="left")
    test["oof_target_enc_zip_mean"] = (
        test_merged["target_mean"].fillna(global_price_mean).values
    )
    test["oof_target_enc_zip_median"] = (
        test_merged["target_median"].fillna(global_price_median).values
    )
    test["oof_target_enc_zip_price_sqft"] = (
        test_merged["target_price_sqft"].fillna(global_price_sqft_mean).values
    )

    # Full Spatial KNN on Test
    coords_full_trn_rad = np.radians(train[["lat", "long"]].values)
    coords_test_rad = np.radians(test[["lat", "long"]].values)

    knn_full = NearestNeighbors(n_neighbors=30, metric="haversine", n_jobs=-1)
    knn_full.fit(coords_full_trn_rad)
    test_dists, test_indices = knn_full.kneighbors(coords_test_rad, n_neighbors=30)
    test_dists_km = test_dists * 6371.0

    full_prices = train["price"].values
    full_psqft = (train["price"] / (train["sqft_living"] + 1.0)).values
    full_grades = train["grade"].values

    for k in knn_ks:
        k_idx = test_indices[:, :k]
        k_d = test_dists_km[:, :k]
        test[f"knn_{k}_price_mean"] = full_prices[k_idx].mean(axis=1)
        test[f"knn_{k}_price_sqft_mean"] = full_psqft[k_idx].mean(axis=1)
        test[f"knn_{k}_grade_mean"] = full_grades[k_idx].mean(axis=1)
        test[f"knn_{k}_dist_mean"] = k_d.mean(axis=1)
    test["knn_nearest_dist"] = test_dists_km[:, 0]

    # Full Spatial Clusters on Test
    for c_res in cluster_resolutions:
        km_full = KMeans(n_clusters=c_res, random_state=random_state, n_init=5)
        c_full_trn = km_full.fit_predict(train[["lat", "long"]].values)
        c_full_test = km_full.predict(test[["lat", "long"]].values)

        c_full_stats = (
            pd.DataFrame({
                "cluster": c_full_trn,
                "price": full_prices,
                "psqft": full_psqft,
                "grade": full_grades,
            })
            .groupby("cluster")
            .agg(
                c_price_mean=("price", "mean"),
                c_psqft_mean=("psqft", "mean"),
                c_grade_mean=("grade", "mean"),
            )
            .reset_index()
        )

        test_c_df = pd.DataFrame({"cluster": c_full_test}).merge(c_full_stats, on="cluster", how="left")
        test[f"cluster_{c_res}_price_mean"] = test_c_df["c_price_mean"].fillna(global_price_mean).values
        test[f"cluster_{c_res}_price_sqft_mean"] = test_c_df["c_psqft_mean"].fillna(global_price_sqft_mean).values
        test[f"cluster_{c_res}_grade_mean"] = test_c_df["c_grade_mean"].fillna(global_grade_mean).values

    train_num_cols = train.select_dtypes(include=[np.number]).columns
    train[train_num_cols] = train[train_num_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    test_num_cols = test.select_dtypes(include=[np.number]).columns
    test[test_num_cols] = test[test_num_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

    return train, test


# ==============================================================================
# 2. Model Architectures & Parameter Configurations
# ==============================================================================
class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout_rate: float = 0.15):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
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
        hidden_dim: int = 256,
        num_blocks: int = 3,
        dropout_rate: float = 0.15,
        output_dim: int = 1,
    ):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()
        )
        self.res_blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim, dropout_rate) for _ in range(num_blocks)]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate / 2.0),
            nn.Linear(hidden_dim // 2, output_dim),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.input_layer(x)
        for block in self.res_blocks:
            out = block(out)
        out = self.head(out)
        return out.squeeze(-1)


def get_lgbm_params():
    return {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "n_estimators": 2500,
        "learning_rate": 0.03,
        "num_leaves": 63,
        "max_depth": -1,
        "min_child_samples": 20,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.75,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }


def get_xgb_params():
    return {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "n_estimators": 2500,
        "learning_rate": 0.03,
        "max_depth": 7,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.75,
        "reg_alpha": 0.1,
        "reg_lambda": 1.5,
        "random_state": 42,
        "n_jobs": -1,
        "early_stopping_rounds": 50,
    }


def get_catboost_params():
    return {
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "iterations": 2500,
        "learning_rate": 0.03,
        "depth": 7,
        "l2_leaf_reg": 3.0,
        "subsample": 0.8,
        "random_seed": 42,
        "verbose": False,
        "thread_count": -1,
        "early_stopping_rounds": 50,
    }


# ==============================================================================
# 3. Model Training & Prediction Handlers
# ==============================================================================
def train_lgbm_fold(X_tr, y_tr, X_va, y_va):
    params = get_lgbm_params()
    model = lgb.LGBMRegressor(**params)
    callbacks = [
        lgb.early_stopping(stopping_rounds=50, verbose=False),
        lgb.log_evaluation(-1),
    ]
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=callbacks)
    return model


def train_xgb_fold(X_tr, y_tr, X_va, y_va):
    import xgboost as xgb

    params = get_xgb_params()
    model = xgb.XGBRegressor(**params)
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    return model


def train_catboost_fold(X_tr, y_tr, X_va, y_va):
    params = get_catboost_params()
    model = CatBoostRegressor(**params)
    model.fit(
        X_tr,
        y_tr,
        eval_set=(X_va, y_va),
        use_best_model=True,
        verbose=False,
    )
    return model


def train_nn_fold(
    X_tr_np,
    y_tr_np,
    X_va_np,
    y_va_np,
    input_dim,
    device,
    epochs=50,
    batch_size=256,
    lr=1e-3,
):
    model = TabularResNet(
        input_dim=input_dim, hidden_dim=256, num_blocks=3, dropout_rate=0.15
    ).to(device)
    criterion = nn.MSELoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4, eps=1e-8)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    target_scale = 100000.0
    y_tr_scaled = y_tr_np / target_scale
    y_va_scaled = y_va_np / target_scale

    train_dataset = TensorDataset(
        torch.tensor(X_tr_np, dtype=torch.float32),
        torch.tensor(y_tr_scaled, dtype=torch.float32),
    )
    val_dataset = TensorDataset(
        torch.tensor(X_va_np, dtype=torch.float32),
        torch.tensor(y_va_scaled, dtype=torch.float32),
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=False
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size * 2, shuffle=False)

    best_val_loss = float("inf")
    best_model_state = None
    patience = 12
    patience_counter = 0

    for _ in range(epochs):
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
        val_losses = []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                preds = model(batch_x)
                val_loss = criterion(preds, batch_y)
                val_losses.append(val_loss.item())

        mean_val_loss = np.mean(val_losses)
        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    model.load_state_dict(best_model_state)
    model.eval()
    return model, target_scale


def predict_nn(model, X_np, target_scale, device, batch_size=512):
    model.eval()
    loader = DataLoader(
        TensorDataset(torch.tensor(X_np, dtype=torch.float32)),
        batch_size=batch_size,
        shuffle=False,
    )
    preds_list = []
    with torch.no_grad():
        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            batch_preds = model(batch_x)
            preds_list.append(batch_preds.cpu().numpy() * target_scale)
    return np.concatenate(preds_list)


# ==============================================================================
# 4. Main Pipeline Execution
# ==============================================================================
def main():
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("./submission", exist_ok=True)
    os.makedirs("./working", exist_ok=True)

    # 1. Load Datasets
    train_path = "./input/train.parquet"
    test_path = "./input/test.parquet"
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    # 2. Base Feature Engineering
    train_feat = engineer_base_features(train_df)
    test_feat = engineer_base_features(test_df)

    # 3. Leakage-Free Aggregations and 5-Fold Setup
    num_folds = 5
    train_proc, test_proc = apply_leak_free_aggregations(
        train_feat, test_feat, num_folds=num_folds, random_state=42
    )

    cols_to_drop = ["date"]
    train_proc = train_proc.drop(
        columns=[c for c in cols_to_drop if c in train_proc.columns]
    )
    test_proc = test_proc.drop(
        columns=[c for c in cols_to_drop if c in test_proc.columns]
    )

    target_col = "price"
    metadata_cols = ["id", "fold", target_col]
    feature_cols = [c for c in train_proc.columns if c not in metadata_cols]

    y_train = train_proc[target_col].values
    folds = train_proc["fold"].values
    n_train = len(train_proc)
    n_test = len(test_proc)

    # 4. Out-of-fold and Test Prediction Allocations
    oof_lgb = np.zeros(n_train, dtype=np.float64)
    oof_xgb = np.zeros(n_train, dtype=np.float64)
    oof_nn = np.zeros(n_train, dtype=np.float64)

    test_preds_lgb = np.zeros(n_test, dtype=np.float64)
    test_preds_xgb = np.zeros(n_test, dtype=np.float64)
    test_preds_nn = np.zeros(n_test, dtype=np.float64)

    # 5. Cross-Validation Training Loop
    for fold in range(num_folds):
        trn_idx = np.where(folds != fold)[0]
        val_idx = np.where(folds == fold)[0]

        X_tr = train_proc.iloc[trn_idx][feature_cols].copy()
        y_tr = y_train[trn_idx]
        X_va = train_proc.iloc[val_idx][feature_cols].copy()
        y_va = y_train[val_idx]
        X_te = test_proc[feature_cols].copy()

        # Neural Network Scaling (strictly on fold train data)
        scaler = RobustScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)
        X_va_scaled = scaler.transform(X_va)
        X_te_scaled = scaler.transform(X_te)

        # A: LightGBM
        lgb_model = train_lgbm_fold(X_tr, y_tr, X_va, y_va)
        oof_lgb[val_idx] = lgb_model.predict(X_va)
        test_preds_lgb += lgb_model.predict(X_te) / num_folds

        # B: XGBoost
        xgb_model = train_xgb_fold(X_tr, y_tr, X_va, y_va)
        oof_xgb[val_idx] = xgb_model.predict(X_va)
        test_preds_xgb += xgb_model.predict(X_te) / num_folds

        # C: TabularResNet
        nn_model, target_scale = train_nn_fold(
            X_tr_scaled, y_tr, X_va_scaled, y_va, len(feature_cols), device
        )
        oof_nn[val_idx] = predict_nn(nn_model, X_va_scaled, target_scale, device)
        test_preds_nn += (
            predict_nn(nn_model, X_te_scaled, target_scale, device) / num_folds
        )

    # 6. Ensemble Weight Optimization (SLSQP minimizing global RMSE)
    oof_matrix = np.column_stack([oof_lgb, oof_xgb, oof_nn])
    test_matrix = np.column_stack([test_preds_lgb, test_preds_xgb, test_preds_nn])

    def rmse_objective(weights):
        w = weights / np.sum(weights)
        blended = np.dot(oof_matrix, w)
        return root_mean_squared_error(y_train, blended)

    init_weights = [0.45, 0.45, 0.10]
    bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]
    constraints = {"type": "eq", "fun": lambda w: 1.0 - np.sum(w)}

    opt_res = minimize(
        rmse_objective,
        init_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    best_weights = (
        opt_res.x / np.sum(opt_res.x)
        if opt_res.success
        else np.array([0.45, 0.45, 0.10])
    )

    final_oof_predictions = np.dot(oof_matrix, best_weights)
    final_score = root_mean_squared_error(y_train, final_oof_predictions)

    # 7. Generate & Format Test Predictions
    final_test_predictions = np.dot(test_matrix, best_weights)
    final_test_predictions = np.maximum(final_test_predictions, 10000.0)

    test_ids = np.arange(len(test_df))
    submission_df = pd.DataFrame({"id": test_ids, "price": final_test_predictions})

    submission_path = "./submission/submission.csv"
    submission_df.to_csv(submission_path, index=False)

    print(f"Final Validation Score: {final_score}")


if __name__ == "__main__":
    main()
