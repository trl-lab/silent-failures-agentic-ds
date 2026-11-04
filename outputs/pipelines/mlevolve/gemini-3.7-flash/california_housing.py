import json
import os
from typing import Any, Dict, Tuple
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.cluster import MiniBatchKMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb
from catboost import CatBoostRegressor


# =========================================================================
# 1. Spatial and Domain Feature Engineering
# =========================================================================
CA_COASTLINE_COORDS = np.array([
    [41.99, -124.21], [41.75, -124.20], [41.48, -124.06], [41.18, -124.16],
    [40.80, -124.16], [40.44, -124.41], [40.26, -124.36], [39.90, -124.02],
    [39.52, -123.79], [39.23, -123.77], [38.91, -123.71], [38.61, -123.43],
    [38.30, -123.07], [38.00, -122.99], [37.75, -122.51], [37.50, -122.52],
    [37.20, -122.41], [36.97, -122.02], [36.60, -121.89], [36.31, -121.90],
    [35.90, -121.48], [35.66, -121.28], [35.40, -120.90], [35.17, -120.74],
    [34.89, -120.66], [34.57, -120.65], [34.45, -120.47], [34.42, -119.83],
    [34.40, -119.54], [34.25, -119.29], [34.15, -119.22], [34.02, -118.82],
    [33.95, -118.45], [33.73, -118.29], [33.56, -117.85], [33.46, -117.72],
    [33.32, -117.58], [33.15, -117.34], [32.88, -117.26], [32.67, -117.18],
    [32.53, -117.12]
])


def haversine_distance_np(lat1, lon1, lat2, lon2):
    """Calculate the great circle distance between two points on Earth in km."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.clip(np.sqrt(a), 0, 1))
    km = 6367.0 * c
    return km


def distance_to_coastline(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """Calculate minimum haversine distance to sampled California coastline coordinates."""
    lats_rad = np.radians(lats)[:, None]
    lons_rad = np.radians(lons)[:, None]
    c_lats_rad = np.radians(CA_COASTLINE_COORDS[:, 0])[None, :]
    c_lons_rad = np.radians(CA_COASTLINE_COORDS[:, 1])[None, :]
    dlat = c_lats_rad - lats_rad
    dlon = c_lons_rad - lons_rad
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lats_rad) * np.cos(c_lats_rad) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.clip(np.sqrt(a), 0, 1))
    dist_km = 6367.0 * c
    return np.min(dist_km, axis=1)


def engineer_base_features(df):
    """Create census domain ratios, interactions, structural features, coordinate rotations, and economic/coastal distances."""
    res = df.copy()

    # 1. Census Domain Ratio & Structural Features
    res["rooms_per_household"] = res["totalRooms"] / (res["households"] + 1e-5)
    res["bedrooms_per_room"] = res["totalBedrooms"] / (res["totalRooms"] + 1e-5)
    res["population_per_household"] = res["population"] / (res["households"] + 1e-5)
    res["bedrooms_per_household"] = res["totalBedrooms"] / (res["households"] + 1e-5)

    # Living area proxies (non-bedroom rooms) and density ratios
    res["non_bed_rooms"] = np.maximum(0, res["totalRooms"] - res["totalBedrooms"])
    res["non_bedrooms_per_household"] = res["non_bed_rooms"] / (res["households"] + 1e-5)
    res["persons_per_room"] = res["population"] / (res["totalRooms"] + 1e-5)
    res["persons_per_bedroom"] = res["population"] / (res["totalBedrooms"] + 1e-5)
    res["rooms_per_person"] = res["totalRooms"] / (res["population"] + 1e-5)
    res["bedrooms_per_person"] = res["totalBedrooms"] / (res["population"] + 1e-5)

    # Housing age interactions
    res["age_x_rooms_ratio"] = res["housingMedianAge"] / (res["rooms_per_household"] + 1e-5)
    res["age_x_bedrooms_ratio"] = res["housingMedianAge"] * res["bedrooms_per_room"]

    # Income interactions
    res["income_per_room"] = res["medianIncome"] / (res["totalRooms"] + 1e-5)
    res["income_per_person"] = res["medianIncome"] / (res["population"] + 1e-5)
    res["income_per_household"] = res["medianIncome"] / (res["households"] + 1e-5)
    res["income_x_age"] = res["medianIncome"] * res["housingMedianAge"]
    res["income_x_rooms_per_hh"] = res["medianIncome"] * res["rooms_per_household"]

    # Skewed count variables log transforms
    for col in ["totalRooms", "totalBedrooms", "population", "households", "non_bed_rooms"]:
        res[f"log_{col}"] = np.log1p(np.maximum(res[col], 0))

    # 2. Geographic Coordinate Rotations
    angles = [15, 30, 45, 60, 75]
    for angle in angles:
        rad = np.radians(angle)
        res[f"rot_{angle}_x"] = (
            np.cos(rad) * res["longitude"] + np.sin(rad) * res["latitude"]
        )
        res[f"rot_{angle}_y"] = (
            -np.sin(rad) * res["longitude"] + np.cos(rad) * res["latitude"]
        )

    # 3. Distances to Major California Economic Centers
    centers = {
        "sf": (37.7749, -122.4194),
        "silicon_valley": (37.3382, -121.8863),
        "la": (34.0522, -118.2437),
        "san_diego": (32.7157, -117.1611),
        "sacramento": (38.5816, -121.4944),
        "fresno": (36.7468, -119.7726),
    }

    dist_cols = []
    for city_name, (city_lat, city_lon) in centers.items():
        col_name = f"dist_to_{city_name}"
        res[col_name] = haversine_distance_np(
            res["latitude"].values, res["longitude"].values, city_lat, city_lon
        )
        dist_cols.append(col_name)

    res["min_dist_to_major_city"] = res[dist_cols].min(axis=1)

    # 4. California Coastline Proximity
    res["dist_to_coastline"] = distance_to_coastline(
        res["latitude"].values, res["longitude"].values
    )
    res["log_dist_to_coastline"] = np.log1p(res["dist_to_coastline"])
    return res


# =========================================================================
# 2. Model Architecture Definitions
# =========================================================================
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
        out_dim: int = 1,
    ):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )
        self.res_blocks = nn.ModuleList(
            [
                ResidualBlock(hidden_dim=hidden_dim, dropout_rate=dropout_rate)
                for _ in range(num_blocks)
            ]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate / 2.0),
            nn.Linear(hidden_dim // 2, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_layer(x)
        for block in self.res_blocks:
            h = block(h)
        out = self.head(h)
        return out.squeeze(-1)


def get_neural_criterion() -> nn.Module:
    return nn.MSELoss()


def build_neural_model_and_optimizer(
    input_dim: int,
    hidden_dim: int = 256,
    num_blocks: int = 3,
    dropout_rate: float = 0.15,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Tuple[nn.Module, torch.optim.Optimizer, Any]:
    model = TabularResNet(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_blocks=num_blocks,
        dropout_rate=dropout_rate,
        out_dim=1,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay, eps=1e-8
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=15, T_mult=2, eta_min=1e-6
    )
    return model, optimizer, scheduler


def get_lgb_config() -> Dict[str, Any]:
    return {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "n_estimators": 3500,
        "learning_rate": 0.03,
        "num_leaves": 63,
        "max_depth": 8,
        "min_child_samples": 20,
        "subsample": 0.85,
        "subsample_freq": 1,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }


def get_xgb_config() -> Dict[str, Any]:
    return {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "n_estimators": 3500,
        "learning_rate": 0.03,
        "max_depth": 7,
        "min_child_weight": 4,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": -1,
    }


def get_catboost_config() -> Dict[str, Any]:
    return {
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "iterations": 3500,
        "learning_rate": 0.035,
        "depth": 7,
        "l2_leaf_reg": 3.0,
        "random_seed": 42,
        "verbose": False,
        "thread_count": -1,
    }


# =========================================================================
# 3. Main End-to-End Pipeline
# =========================================================================
def main():
    print("Initializing California Housing ML pipeline...")
    os.makedirs("./working", exist_ok=True)
    os.makedirs("./submission", exist_ok=True)

    # 1. Load Data
    train_df = pd.read_parquet("./input/train.parquet")
    test_df = pd.read_parquet("./input/test.parquet")
    target_col = "medianHouseValue"

    if "id" not in test_df.columns:
        test_df["id"] = np.arange(len(test_df))
    if "id" not in train_df.columns:
        train_df["id"] = np.arange(len(train_df))

    test_ids = test_df["id"].values.copy()
    y_true = train_df[target_col].values.copy()

    # Handle missing values fitted strictly on training data
    imputer = SimpleImputer(strategy="median")
    base_numeric_cols = [
        "longitude",
        "latitude",
        "housingMedianAge",
        "totalRooms",
        "totalBedrooms",
        "population",
        "households",
        "medianIncome",
    ]
    train_df[base_numeric_cols] = imputer.fit_transform(train_df[base_numeric_cols])
    test_df[base_numeric_cols] = imputer.transform(test_df[base_numeric_cols])

    # Feature Engineering
    train_feat = engineer_base_features(train_df)
    test_feat = engineer_base_features(test_df)

    # Multi-Resolution Spatial Clustering (MiniBatchKMeans)
    for n_clusters in [15, 30, 60, 120]:
        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters, random_state=42, batch_size=2048, n_init=3
        )
        train_feat[f"cluster_{n_clusters}"] = kmeans.fit_predict(
            train_feat[["latitude", "longitude"]]
        )
        test_feat[f"cluster_{n_clusters}"] = kmeans.predict(
            test_feat[["latitude", "longitude"]]
        )

        train_dists = kmeans.transform(train_feat[["latitude", "longitude"]])
        test_dists = kmeans.transform(test_feat[["latitude", "longitude"]])
        train_feat[f"min_dist_cluster_{n_clusters}"] = np.min(train_dists, axis=1)
        test_feat[f"min_dist_cluster_{n_clusters}"] = np.min(test_dists, axis=1)

    # Cross-Validation Setup: 5 Stratified Folds
    n_splits = 5
    num_bins = 15
    y_bins = pd.qcut(
        train_feat[target_col], q=num_bins, labels=False, duplicates="drop"
    )
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    train_feat["fold"] = -1
    for fold, (_, val_idx) in enumerate(skf.split(train_feat, y_bins)):
        train_feat.loc[val_idx, "fold"] = fold

    # Leak-Free Spatial Nearest Neighbor Aggregations for Income and Target
    k_neighbors_list = [5, 15, 30, 60]
    metrics = ["mean", "std", "median", "idw"]
    for target_var in ["medianIncome", target_col]:
        for k in k_neighbors_list:
            for m in metrics:
                train_feat[f"knn_{k}_{target_var}_{m}"] = 0.0
                test_feat[f"knn_{k}_{target_var}_{m}"] = 0.0
            train_feat[f"knn_{k}_dist_mean"] = 0.0
            test_feat[f"knn_{k}_dist_mean"] = 0.0

    max_k = max(k_neighbors_list)
    for fold in range(n_splits):
        trn_mask = train_feat["fold"] != fold
        val_mask = train_feat["fold"] == fold

        trn_coords = train_feat.loc[trn_mask, ["latitude", "longitude"]].values
        val_coords = train_feat.loc[val_mask, ["latitude", "longitude"]].values

        nn_spatial = NearestNeighbors(n_neighbors=max_k, algorithm="ball_tree")
        nn_spatial.fit(trn_coords)
        all_dists, all_indices = nn_spatial.kneighbors(val_coords)

        for target_var in ["medianIncome", target_col]:
            trn_vals = train_feat.loc[trn_mask, target_var].values
            for k in k_neighbors_list:
                k_idx = all_indices[:, :k]
                k_dist = all_dists[:, :k]
                nbr_vals = trn_vals[k_idx]

                weights = 1.0 / (k_dist + 1e-4)
                weights /= np.sum(weights, axis=1, keepdims=True)
                idw_val = np.sum(nbr_vals * weights, axis=1)

                train_feat.loc[val_mask, f"knn_{k}_{target_var}_mean"] = np.mean(
                    nbr_vals, axis=1
                )
                train_feat.loc[val_mask, f"knn_{k}_{target_var}_std"] = np.std(
                    nbr_vals, axis=1
                )
                train_feat.loc[val_mask, f"knn_{k}_{target_var}_median"] = np.median(
                    nbr_vals, axis=1
                )
                train_feat.loc[val_mask, f"knn_{k}_{target_var}_idw"] = idw_val
                if target_var == "medianIncome":
                    train_feat.loc[val_mask, f"knn_{k}_dist_mean"] = np.mean(k_dist, axis=1)

    full_train_coords = train_feat[["latitude", "longitude"]].values
    test_coords = test_feat[["latitude", "longitude"]].values
    nn_spatial_full = NearestNeighbors(n_neighbors=max_k, algorithm="ball_tree")
    nn_spatial_full.fit(full_train_coords)
    test_dists, test_indices = nn_spatial_full.kneighbors(test_coords)

    for target_var in ["medianIncome", target_col]:
        full_trn_vals = train_feat[target_var].values
        for k in k_neighbors_list:
            k_idx = test_indices[:, :k]
            k_dist = test_dists[:, :k]
            nbr_vals = full_trn_vals[k_idx]

            weights = 1.0 / (k_dist + 1e-4)
            weights /= np.sum(weights, axis=1, keepdims=True)
            idw_val = np.sum(nbr_vals * weights, axis=1)

            test_feat[f"knn_{k}_{target_var}_mean"] = np.mean(nbr_vals, axis=1)
            test_feat[f"knn_{k}_{target_var}_std"] = np.std(nbr_vals, axis=1)
            test_feat[f"knn_{k}_{target_var}_median"] = np.median(nbr_vals, axis=1)
            test_feat[f"knn_{k}_{target_var}_idw"] = idw_val
            if target_var == "medianIncome":
                test_feat[f"knn_{k}_dist_mean"] = np.mean(k_dist, axis=1)

    feature_cols = [
        c for c in train_feat.columns if c not in ["id", "fold", target_col]
    ]

    train_feat[feature_cols] = train_feat[feature_cols].replace(
        [np.inf, -np.inf], np.nan
    )
    test_feat[feature_cols] = test_feat[feature_cols].replace([np.inf, -np.inf], np.nan)

    final_imputer = SimpleImputer(strategy="median")
    train_feat[feature_cols] = final_imputer.fit_transform(train_feat[feature_cols])
    test_feat[feature_cols] = final_imputer.transform(test_feat[feature_cols])

    # Out-Of-Fold and Test prediction arrays
    models = ["lgb", "xgb", "cat", "nn"]
    oof_preds = {m: np.zeros(len(train_df)) for m in models}
    test_preds = {m: np.zeros(len(test_df)) for m in models}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Cross-Validated Training Loop
    for fold in range(n_splits):
        trn_idx = train_feat["fold"] != fold
        val_idx = train_feat["fold"] == fold

        X_tr = train_feat.loc[trn_idx, feature_cols].values
        y_tr = train_feat.loc[trn_idx, target_col].values
        X_va = train_feat.loc[val_idx, feature_cols].values
        y_va = train_feat.loc[val_idx, target_col].values
        X_te = test_feat[feature_cols].values

        # 1. LightGBM
        lgb_params = get_lgb_config()
        lgb_model = lgb.LGBMRegressor(**lgb_params)
        lgb_model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )
        oof_preds["lgb"][val_idx] = lgb_model.predict(X_va)
        test_preds["lgb"] += lgb_model.predict(X_te) / n_splits

        # 2. XGBoost
        xgb_params = get_xgb_config()
        xgb_params["early_stopping_rounds"] = 50
        xgb_model = xgb.XGBRegressor(**xgb_params)
        xgb_model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        oof_preds["xgb"][val_idx] = xgb_model.predict(X_va)
        test_preds["xgb"] += xgb_model.predict(X_te) / n_splits

        # 3. CatBoost
        cat_params = get_catboost_config()
        cat_params["early_stopping_rounds"] = 50
        cat_model = CatBoostRegressor(**cat_params)
        cat_model.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=False)
        oof_preds["cat"][val_idx] = cat_model.predict(X_va)
        test_preds["cat"] += cat_model.predict(X_te) / n_splits

        # 4. TabularResNet
        feat_scaler = StandardScaler()
        X_tr_sc = feat_scaler.fit_transform(X_tr)
        X_va_sc = feat_scaler.transform(X_va)
        X_te_sc = feat_scaler.transform(X_te)

        target_scaler = StandardScaler()
        y_tr_sc = target_scaler.fit_transform(y_tr.reshape(-1, 1)).flatten()
        y_va_sc = target_scaler.transform(y_va.reshape(-1, 1)).flatten()

        batch_size = 256
        trn_dataset = TensorDataset(
            torch.tensor(X_tr_sc, dtype=torch.float32),
            torch.tensor(y_tr_sc, dtype=torch.float32),
        )
        trn_loader = DataLoader(
            trn_dataset, batch_size=batch_size, shuffle=True, drop_last=False
        )

        nn_model, optimizer, scheduler = build_neural_model_and_optimizer(
            input_dim=len(feature_cols),
            hidden_dim=256,
            num_blocks=3,
            dropout_rate=0.15,
            lr=1e-3,
            device=str(device),
        )
        criterion = get_neural_criterion()

        best_val_loss = float("inf")
        best_model_weights = None
        patience = 12
        patience_counter = 0
        epochs = 60

        X_va_tensor = torch.tensor(X_va_sc, dtype=torch.float32).to(device)
        y_va_tensor = torch.tensor(y_va_sc, dtype=torch.float32).to(device)

        for epoch in range(epochs):
            nn_model.train()
            for batch_x, batch_y in trn_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                preds = nn_model(batch_x)
                loss = criterion(preds, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(nn_model.parameters(), max_norm=1.0)
                optimizer.step()

            scheduler.step()

            nn_model.eval()
            with torch.no_grad():
                val_preds_sc = nn_model(X_va_tensor)
                val_loss = criterion(val_preds_sc, y_va_tensor).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_weights = {
                    k: v.cpu().clone() for k, v in nn_model.state_dict().items()
                }
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        nn_model.load_state_dict(
            {k: v.to(device) for k, v in best_model_weights.items()}
        )
        nn_model.eval()
        with torch.no_grad():
            oof_nn_sc = nn_model(X_va_tensor).cpu().numpy()
            oof_nn = target_scaler.inverse_transform(oof_nn_sc.reshape(-1, 1)).flatten()
            oof_preds["nn"][val_idx] = oof_nn

            X_te_tensor = torch.tensor(X_te_sc, dtype=torch.float32).to(device)
            test_nn_sc = nn_model(X_te_tensor).cpu().numpy()
            test_nn = target_scaler.inverse_transform(
                test_nn_sc.reshape(-1, 1)
            ).flatten()
            test_preds["nn"] += test_nn / n_splits

    # Metric-Optimized Ensemble Blending (SLSQP optimization directly on official RMSE)
    oof_matrix = np.column_stack([oof_preds[m] for m in models])
    test_matrix = np.column_stack([test_preds[m] for m in models])

    def rmse_objective(weights):
        blended = np.dot(oof_matrix, weights)
        return root_mean_squared_error(y_true, blended)

    init_weights = np.ones(len(models)) / len(models)
    bounds = [(0.0, 1.0) for _ in range(len(models))]
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

    opt_result = minimize(
        rmse_objective,
        init_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    opt_weights = opt_result.x
    oof_ensemble = np.dot(oof_matrix, opt_weights)
    final_score = root_mean_squared_error(y_true, oof_ensemble)

    # Test set inference
    final_test_preds = np.dot(test_matrix, opt_weights)
    final_test_preds = np.maximum(final_test_preds, 0.0)

    # Export Submission
    submission_df = pd.DataFrame(
        {
            "id": test_ids.astype(int),
            "medianHouseValue": final_test_preds.astype(float),
        }
    )
    submission_df.to_csv("./submission/submission.csv", index=False)

    print(f"Final Validation Score: {final_score}")


if __name__ == "__main__":
    main()
