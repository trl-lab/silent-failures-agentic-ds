import os
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
import lightgbm as lgb
import xgboost as xgb
import catboost as cb

warnings.filterwarnings("ignore")

# Ensure required output directories exist
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------
# 1. Spatial & Domain Feature Engineering Definitions
# ---------------------------------------------------------
CITIES = {
    "LA": (34.0522, -118.2437),
    "SF": (37.7749, -122.4194),
    "SJ": (37.3382, -121.8863),
    "SD": (32.7157, -117.1611),
    "SAC": (38.5816, -121.4944),
    "FRESNO": (36.7468, -119.7726),
}

COAST_POINTS = np.array(
    [
        [32.53, -117.12],
        [32.71, -117.17],
        [33.38, -117.58],
        [33.74, -118.27],
        [34.01, -118.50],
        [34.40, -119.69],
        [35.18, -120.64],
        [36.60, -121.89],
        [37.77, -122.51],
        [38.33, -123.05],
        [39.44, -123.81],
        [40.80, -124.16],
        [41.76, -124.20],
    ]
)


def haversine_np(lat1, lon1, lat2, lon2):
    r = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    return r * c


class RandomFourierFeatures:
    def __init__(self, n_components=16, gamma=0.5, random_state=42):
        self.n_components = n_components
        self.gamma = gamma
        self.random_state = random_state
        self.scaler = StandardScaler()

    def fit(self, X):
        X_scaled = self.scaler.fit_transform(X)
        rng = np.random.RandomState(self.random_state)
        self.W = rng.normal(
            loc=0.0,
            scale=np.sqrt(2.0 * self.gamma),
            size=(X.shape[1], self.n_components),
        )
        self.b = rng.uniform(0.0, 2.0 * np.pi, size=self.n_components)
        return self

    def transform(self, X):
        X_scaled = self.scaler.transform(X)
        projection = np.dot(X_scaled, self.W) + self.b
        return np.sqrt(2.0 / self.n_components) * np.cos(projection)


def create_domain_features(df):
    df = df.copy()
    households = np.maximum(df["households"].values, 1.0)
    total_rooms = np.maximum(df["totalRooms"].values, 1.0)
    total_bedrooms = np.maximum(df["totalBedrooms"].values, 1.0)
    population = np.maximum(df["population"].values, 1.0)
    median_income = np.maximum(df["medianIncome"].values, 0.001)

    df["rooms_per_household"] = total_rooms / households
    df["bedrooms_per_room"] = total_bedrooms / total_rooms
    df["population_per_household"] = population / households
    df["bedrooms_per_household"] = total_bedrooms / households
    df["non_bedroom_rooms"] = np.maximum(total_rooms - total_bedrooms, 0.0)
    df["non_bedrooms_per_household"] = df["non_bedroom_rooms"] / households

    df["income_per_person"] = median_income / (df["population_per_household"] + 0.1)
    df["income_per_room"] = median_income / (df["rooms_per_household"] + 0.1)
    df["income_age_prod"] = median_income * df["housingMedianAge"]
    df["income_rooms_prod"] = median_income * df["rooms_per_household"]

    for col in [
        "totalRooms",
        "totalBedrooms",
        "population",
        "households",
        "medianIncome",
    ]:
        df[f"log_{col}"] = np.log1p(np.maximum(df[col].values, 0.0))

    return df


def create_spatial_features(df):
    df = df.copy()
    lats = df["latitude"].values
    lons = df["longitude"].values

    min_dists = np.full(len(df), np.inf)
    for city_name, (city_lat, city_lon) in CITIES.items():
        dist = haversine_np(lats, lons, city_lat, city_lon)
        df[f"dist_{city_name}"] = dist
        min_dists = np.minimum(min_dists, dist)
    df["min_dist_major_city"] = min_dists

    coast_dists = []
    for lat, lon in zip(lats, lons):
        d = haversine_np(lat, lon, COAST_POINTS[:, 0], COAST_POINTS[:, 1])
        coast_dists.append(np.min(d))
    df["dist_to_coast"] = coast_dists

    for angle_deg in [30, 45, 60]:
        rad = np.radians(angle_deg)
        cos_a, sin_a = np.cos(rad), np.sin(rad)
        df[f"rot_x_{angle_deg}"] = lons * cos_a - lats * sin_a
        df[f"rot_y_{angle_deg}"] = lons * sin_a + lats * cos_a

    ca_center_lat, ca_center_lon = 36.7783, -119.4179
    df["polar_r"] = np.sqrt((lats - ca_center_lat) ** 2 + (lons - ca_center_lon) ** 2)
    df["polar_theta"] = np.arctan2(lats - ca_center_lat, lons - ca_center_lon)

    return df


# ---------------------------------------------------------
# 2. PyTorch Tabular ResNet Architecture
# ---------------------------------------------------------
class TabularResBlock(nn.Module):
    def __init__(self, in_features, hidden_features, dropout=0.15):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.LayerNorm(hidden_features),
            nn.Mish(),
            nn.Dropout(dropout),
            nn.Linear(hidden_features, in_features),
            nn.LayerNorm(in_features),
        )
        self.act = nn.Mish()

    def forward(self, x):
        return self.act(x + self.block(x))


class PyTorchTabularResNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_blocks=3, dropout=0.15):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Mish(),
        )
        self.blocks = nn.ModuleList(
            [
                TabularResBlock(hidden_dim, hidden_dim * 2, dropout=dropout)
                for _ in range(num_blocks)
            ]
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Mish(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        x = self.input_layer(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x)


# ---------------------------------------------------------
# 3. Training Loop & Stacking Evaluation
# ---------------------------------------------------------
def main():
    train_raw = pd.read_parquet("./input/train.parquet")
    test_raw = pd.read_parquet("./input/test.parquet")

    test_ids = (
        test_raw["id"].values if "id" in test_raw.columns else np.arange(len(test_raw))
    )

    br_ratio = (train_raw["totalBedrooms"] / train_raw["totalRooms"]).median()
    train_raw["totalBedrooms"] = train_raw["totalBedrooms"].fillna(
        train_raw["totalRooms"] * br_ratio
    )
    test_raw["totalBedrooms"] = test_raw["totalBedrooms"].fillna(
        test_raw["totalRooms"] * br_ratio
    )

    train_base = create_spatial_features(create_domain_features(train_raw))
    test_base = create_spatial_features(create_domain_features(test_raw))

    target_col = "medianHouseValue"
    y = train_base[target_col].values
    y_log = np.log1p(y)

    drop_cols = [target_col, "id"] if "id" in train_base.columns else [target_col]

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    num_models = 5
    oof_matrix = np.zeros((len(train_base), num_models))
    test_matrix = np.zeros((len(test_base), num_models))

    for fold, (train_idx, val_idx) in enumerate(kf.split(train_base)):
        tr_df = train_base.iloc[train_idx].copy()
        va_df = train_base.iloc[val_idx].copy()
        te_df = test_base.copy()

        # Fold Grid aggregations (fitted strictly on training split)
        scales = [0.1, 0.05, 0.02]
        grid_maps = {}
        for scale in scales:
            tr_df[f"grid_lat_{scale}"] = (tr_df["latitude"] / scale).round() * scale
            tr_df[f"grid_lon_{scale}"] = (tr_df["longitude"] / scale).round() * scale

            agg_df = tr_df.groupby([f"grid_lat_{scale}", f"grid_lon_{scale}"]).agg(
                {
                    "medianIncome": ["mean", "std"],
                    "housingMedianAge": ["mean"],
                    "rooms_per_household": ["mean"],
                }
            )
            agg_df.columns = [
                f"grid_{scale}_{col}_{stat}" for col, stat in agg_df.columns
            ]
            grid_maps[scale] = agg_df.reset_index()

        for df_target in [tr_df, va_df, te_df]:
            for scale in scales:
                grid_lat = (df_target["latitude"] / scale).round() * scale
                grid_lon = (df_target["longitude"] / scale).round() * scale
                temp_keys = pd.DataFrame(
                    {f"grid_lat_{scale}": grid_lat, f"grid_lon_{scale}": grid_lon},
                    index=df_target.index,
                )
                merged = temp_keys.merge(
                    grid_maps[scale],
                    on=[f"grid_lat_{scale}", f"grid_lon_{scale}"],
                    how="left",
                )
                merged.index = df_target.index

                for c in grid_maps[scale].columns:
                    if c not in [f"grid_lat_{scale}", f"grid_lon_{scale}"]:
                        df_target[c] = merged[c]

                if f"grid_{scale}_medianIncome_mean" in df_target.columns:
                    df_target[f"rel_income_{scale}"] = df_target["medianIncome"] / (
                        df_target[f"grid_{scale}_medianIncome_mean"] + 1e-5
                    )

        # Spatial Random Fourier Features
        rff = RandomFourierFeatures(n_components=16, gamma=0.5, random_state=42 + fold)
        tr_coords = tr_df[["latitude", "longitude"]].values
        rff.fit(tr_coords)

        for df_target in [tr_df, va_df, te_df]:
            coords = df_target[["latitude", "longitude"]].values
            rff_feats = rff.transform(coords)
            for k in range(16):
                df_target[f"rff_{k}"] = rff_feats[:, k]

        # Spatial KNN Features
        knn = NearestNeighbors(n_neighbors=15, metric="haversine", n_jobs=-1)
        knn.fit(np.radians(tr_coords))

        for df_target in [tr_df, va_df, te_df]:
            rad_coords = np.radians(df_target[["latitude", "longitude"]].values)
            dists, indices = knn.kneighbors(rad_coords)
            dists_km = dists * 6371.0

            df_target["knn_mean_dist_15"] = np.mean(dists_km[:, :15], axis=1)
            neighbor_income = tr_df["medianIncome"].values[indices[:, :15]]
            df_target["knn_income_mean_15"] = np.mean(neighbor_income, axis=1)
            df_target["knn_income_std_15"] = np.std(neighbor_income, axis=1)

        fold_cols = [
            c
            for c in tr_df.columns
            if c not in drop_cols
            and not c.startswith("grid_lat_")
            and not c.startswith("grid_lon_")
        ]

        train_medians = tr_df[fold_cols].median()
        X_tr = tr_df[fold_cols].fillna(train_medians).values
        X_va = va_df[fold_cols].fillna(train_medians).values
        X_te = te_df[fold_cols].fillna(train_medians).values

        y_tr = y_log[train_idx]
        y_va = y_log[val_idx]

        # Model 1: LightGBM (ExtraTrees mode)
        m1 = lgb.LGBMRegressor(
            n_estimators=2500,
            learning_rate=0.02,
            num_leaves=127,
            max_depth=9,
            subsample=0.8,
            colsample_bytree=0.65,
            min_child_samples=15,
            extra_trees=True,
            random_state=42 + fold,
            n_jobs=-1,
        )
        m1.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
        )
        oof_matrix[val_idx, 0] = m1.predict(X_va)
        test_matrix[:, 0] += m1.predict(X_te) / 5.0

        # Model 2: XGBoost (Histogram)
        m2 = xgb.XGBRegressor(
            n_estimators=2500,
            learning_rate=0.02,
            max_depth=8,
            subsample=0.8,
            colsample_bytree=0.6,
            min_child_weight=3,
            tree_method="hist",
            random_state=42 + fold,
            n_jobs=-1,
            early_stopping_rounds=50,
        )
        m2.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        oof_matrix[val_idx, 1] = m2.predict(X_va)
        test_matrix[:, 1] += m2.predict(X_te) / 5.0

        # Model 3: CatBoost Regressor
        m3 = cb.CatBoostRegressor(
            iterations=2500,
            learning_rate=0.03,
            depth=8,
            l2_leaf_reg=5.0,
            random_seed=42 + fold,
            early_stopping_rounds=50,
            verbose=0,
        )
        m3.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=False)
        oof_matrix[val_idx, 2] = m3.predict(X_va)
        test_matrix[:, 2] += m3.predict(X_te) / 5.0

        # Model 4: HistGradientBoostingRegressor
        m4 = HistGradientBoostingRegressor(
            max_iter=800,
            learning_rate=0.03,
            max_leaf_nodes=63,
            min_samples_leaf=20,
            l2_regularization=1.0,
            random_state=42 + fold,
            early_stopping=True,
        )
        m4.fit(X_tr, y_tr)
        oof_matrix[val_idx, 3] = m4.predict(X_va)
        test_matrix[:, 3] += m4.predict(X_te) / 5.0

        # Model 5: PyTorch Tabular ResNet
        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr)
        X_va_sc = scaler.transform(X_va)
        X_te_sc = scaler.transform(X_te)

        train_ds = TensorDataset(
            torch.tensor(X_tr_sc, dtype=torch.float32),
            torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1),
        )
        val_ds = TensorDataset(
            torch.tensor(X_va_sc, dtype=torch.float32),
            torch.tensor(y_va, dtype=torch.float32).unsqueeze(1),
        )

        train_ld = DataLoader(train_ds, batch_size=256, shuffle=True)
        val_ld = DataLoader(val_ds, batch_size=512, shuffle=False)

        net = PyTorchTabularResNet(
            input_dim=X_tr_sc.shape[1], hidden_dim=256, num_blocks=3, dropout=0.15
        ).to(device)
        optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=40, eta_min=1e-5
        )
        criterion = nn.MSELoss()

        best_val_loss = float("inf")
        best_state = None

        for epoch in range(40):
            net.train()
            for bx, by in train_ld:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                loss = criterion(net(bx), by)
                loss.backward()
                optimizer.step()
            scheduler.step()

            net.eval()
            val_loss = 0.0
            with torch.no_grad():
                for bx, by in val_ld:
                    bx, by = bx.to(device), by.to(device)
                    val_loss += criterion(net(bx), by).item() * len(bx)
            val_loss /= len(val_ds)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu() for k, v in net.state_dict().items()}

        net.load_state_dict(best_state)
        net.eval()
        with torch.no_grad():
            nn_val_pred = (
                net(torch.tensor(X_va_sc, dtype=torch.float32).to(device))
                .cpu()
                .numpy()
                .flatten()
            )
            nn_test_pred = (
                net(torch.tensor(X_te_sc, dtype=torch.float32).to(device))
                .cpu()
                .numpy()
                .flatten()
            )

        oof_matrix[val_idx, 4] = nn_val_pred
        test_matrix[:, 4] += nn_test_pred / 5.0

        print(
            f"Fold {fold+1}/5 completed. Best Tabular ResNet Loss: {best_val_loss:.5f}"
        )

    # ---------------------------------------------------------
    # 4. Stacking Meta-Learner Layer
    # ---------------------------------------------------------
    def extract_meta_features(preds):
        mean_p = np.mean(preds, axis=1, keepdims=True)
        std_p = np.std(preds, axis=1, keepdims=True)
        min_p = np.min(preds, axis=1, keepdims=True)
        max_p = np.max(preds, axis=1, keepdims=True)
        diffs = preds - mean_p
        return np.hstack([preds, mean_p, std_p, min_p, max_p, diffs])

    meta_X_tr = extract_meta_features(oof_matrix)
    meta_X_te = extract_meta_features(test_matrix)

    meta_kf = KFold(n_splits=5, shuffle=True, random_state=42)
    meta_oof = np.zeros(len(meta_X_tr))
    meta_test = np.zeros(len(meta_X_te))

    for m_tr_idx, m_val_idx in meta_kf.split(meta_X_tr):
        meta_model = Ridge(alpha=10.0, random_state=42)
        meta_model.fit(meta_X_tr[m_tr_idx], y_log[m_tr_idx])
        meta_oof[m_val_idx] = meta_model.predict(meta_X_tr[m_val_idx])
        meta_test += meta_model.predict(meta_X_te) / 5.0

    final_oof_preds = np.clip(np.expm1(meta_oof), 0.0, 500001.0)
    final_test_preds = np.clip(np.expm1(meta_test), 0.0, 500001.0)

    # Save final predictions to required path
    sub = pd.DataFrame({"id": test_ids, "medianHouseValue": final_test_preds})
    sub.to_csv("./submission/submission.csv", index=False)

    final_score = root_mean_squared_error(y, final_oof_preds)
    print(f"Final Validation Score: {final_score}")


if __name__ == "__main__":
    main()
