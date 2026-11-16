import os
import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb


def calculate_rmsle(y_true, y_pred):
    """Calculate exact Root Mean Squared Logarithmic Error (RMSLE)."""
    y_pred_clipped = np.clip(y_pred, 0, None)
    return np.sqrt(np.mean((np.log1p(y_true) - np.log1p(y_pred_clipped)) ** 2))


def haversine_array(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    AVG_EARTH_RADIUS_KM = 6371.0088
    lat = lat2 - lat1
    lng = lng2 - lng1
    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2
    d = np.clip(d, 0.0, 1.0)
    h = 2 * AVG_EARTH_RADIUS_KM * np.arcsin(np.sqrt(d))
    return h


def dummy_manhattan_distance(lat1, lng1, lat2, lng2):
    a = haversine_array(lat1, lng1, lat1, lng2)
    b = haversine_array(lat1, lng1, lat2, lng1)
    return a + b


def bearing_array(lat1, lng1, lat2, lng2):
    lng_delta_rad = np.radians(lng2 - lng1)
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    y = np.sin(lng_delta_rad) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(
        lng_delta_rad
    )
    return np.degrees(np.arctan2(y, x))


def extract_features(df, pca_model=None, kmeans_model=None, is_train=False):
    df = df.copy()

    # Convert datetime
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])

    # Temporal features
    df["pickup_hour"] = df["pickup_datetime"].dt.hour
    df["pickup_dayofweek"] = df["pickup_datetime"].dt.dayofweek
    df["pickup_month"] = df["pickup_datetime"].dt.month
    df["pickup_day"] = df["pickup_datetime"].dt.day
    df["pickup_minute"] = df["pickup_datetime"].dt.minute
    df["pickup_dayofyear"] = df["pickup_datetime"].dt.dayofyear
    df["is_weekend"] = (df["pickup_dayofweek"] >= 5).astype(int)

    # Cyclical temporal encodings
    df["hour_sin"] = np.sin(2 * np.pi * df["pickup_hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["pickup_hour"] / 24.0)
    df["dayofweek_sin"] = np.sin(2 * np.pi * df["pickup_dayofweek"] / 7.0)
    df["dayofweek_cos"] = np.cos(2 * np.pi * df["pickup_dayofweek"] / 7.0)
    df["month_sin"] = np.sin(2 * np.pi * df["pickup_month"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["pickup_month"] / 12.0)

    # Binary and categorical transformations
    if "store_and_fwd_flag" in df.columns:
        df["store_and_fwd_flag"] = (df["store_and_fwd_flag"] == "Y").astype(int)

    # Distance and directional features
    df["haversine_dist"] = haversine_array(
        df["pickup_latitude"].values,
        df["pickup_longitude"].values,
        df["dropoff_latitude"].values,
        df["dropoff_longitude"].values,
    )
    df["manhattan_dist"] = dummy_manhattan_distance(
        df["pickup_latitude"].values,
        df["pickup_longitude"].values,
        df["dropoff_latitude"].values,
        df["dropoff_longitude"].values,
    )
    df["bearing"] = bearing_array(
        df["pickup_latitude"].values,
        df["pickup_longitude"].values,
        df["dropoff_latitude"].values,
        df["dropoff_longitude"].values,
    )

    # Landmark proximity features (JFK, LGA, EWR, Times Square)
    landmarks = {
        "jfk": (40.6413, -73.7781),
        "lga": (40.7769, -73.8740),
        "ewr": (40.6895, -74.1745),
        "times_square": (40.7580, -73.9855),
    }

    for name, (l_lat, l_lon) in landmarks.items():
        df[f"dist_pickup_{name}"] = haversine_array(
            df["pickup_latitude"].values,
            df["pickup_longitude"].values,
            l_lat,
            l_lon,
        )
        df[f"dist_dropoff_{name}"] = haversine_array(
            df["dropoff_latitude"].values,
            df["dropoff_longitude"].values,
            l_lat,
            l_lon,
        )

    # Coordinate PCA transformation
    coords = np.vstack(
        [
            df[["pickup_latitude", "pickup_longitude"]].values,
            df[["dropoff_latitude", "dropoff_longitude"]].values,
        ]
    )

    if is_train and pca_model is None:
        pca_model = PCA(n_components=2, random_state=42).fit(coords)

    pickup_pca = pca_model.transform(df[["pickup_latitude", "pickup_longitude"]].values)
    dropoff_pca = pca_model.transform(
        df[["dropoff_latitude", "dropoff_longitude"]].values
    )

    df["pca_pickup_0"] = pickup_pca[:, 0]
    df["pca_pickup_1"] = pickup_pca[:, 1]
    df["pca_dropoff_0"] = dropoff_pca[:, 0]
    df["pca_dropoff_1"] = dropoff_pca[:, 1]

    # Spatial clustering (KMeans)
    if is_train and kmeans_model is None:
        kmeans_model = KMeans(n_clusters=20, random_state=42, n_init=5).fit(coords)

    df["pickup_cluster"] = kmeans_model.predict(
        df[["pickup_latitude", "pickup_longitude"]].values
    )
    df["dropoff_cluster"] = kmeans_model.predict(
        df[["dropoff_latitude", "dropoff_longitude"]].values
    )

    return df, pca_model, kmeans_model


class ResidualBlock(nn.Module):
    """Deep residual block for numerical tabular features."""

    def __init__(self, features: int, dropout_rate: float = 0.15):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(features, features),
            nn.BatchNorm1d(features),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(features, features),
            nn.BatchNorm1d(features),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class TabularResNet(nn.Module):
    """PyTorch Residual Network for high-dimensional tabular regression."""

    def __init__(
        self,
        num_features: int,
        hidden_dim: int = 256,
        num_blocks: int = 3,
        dropout_rate: float = 0.15,
    ):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )
        self.res_blocks = nn.ModuleList(
            [
                ResidualBlock(hidden_dim, dropout_rate=dropout_rate)
                for _ in range(num_blocks)
            ]
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(x)
        for block in self.res_blocks:
            x = block(x)
        return self.head(x).squeeze(-1)


def main():
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Data
    train_df = pd.read_parquet("./input/train.parquet")
    test_df = pd.read_parquet("./input/test.parquet")

    # Clean extreme duration and coordinate outliers in training set
    coord_mask = (
        (train_df["pickup_latitude"].between(40.4, 41.2))
        & (train_df["pickup_longitude"].between(-74.4, -73.6))
        & (train_df["dropoff_latitude"].between(40.4, 41.2))
        & (train_df["dropoff_longitude"].between(-74.4, -73.6))
    )
    duration_mask = (train_df["trip_duration"] >= 10) & (
        train_df["trip_duration"] <= 86400
    )
    train_clean = train_df[coord_mask & duration_mask].reset_index(drop=True)

    # Log1p target transformation
    train_clean["log_trip_duration"] = np.log1p(train_clean["trip_duration"])

    # Train / Validation Split
    train_split, val_split = train_test_split(
        train_clean, test_size=0.15, random_state=42
    )
    train_split = train_split.reset_index(drop=True)
    val_split = val_split.reset_index(drop=True)

    # Feature engineering (stateful models fitted ONLY on train_split)
    train_processed, pca_model, kmeans_model = extract_features(
        train_split, is_train=True
    )
    val_processed, _, _ = extract_features(
        val_split, pca_model=pca_model, kmeans_model=kmeans_model, is_train=False
    )
    test_processed, _, _ = extract_features(
        test_df, pca_model=pca_model, kmeans_model=kmeans_model, is_train=False
    )

    exclude_cols = [
        "id",
        "pickup_datetime",
        "dropoff_datetime",
        "trip_duration",
        "log_trip_duration",
    ]
    feature_cols = [c for c in train_processed.columns if c not in exclude_cols]

    X_train = train_processed[feature_cols].values.astype(np.float32)
    y_train_log = train_processed["log_trip_duration"].values.astype(np.float32)

    X_val = val_processed[feature_cols].values.astype(np.float32)
    y_val_log = val_processed["log_trip_duration"].values.astype(np.float32)
    y_val_actual = val_processed["trip_duration"].values.astype(np.float64)

    X_test = test_processed[feature_cols].values.astype(np.float32)

    # 2. LightGBM
    lgb_params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "n_estimators": 2500,
        "learning_rate": 0.05,
        "num_leaves": 127,
        "max_depth": -1,
        "min_child_samples": 50,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }
    lgb_model = lgb.LGBMRegressor(**lgb_params)
    lgb_model.fit(
        X_train,
        y_train_log,
        eval_set=[(X_val, y_val_log)],
        callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)],
    )
    lgb_val_pred_log = lgb_model.predict(X_val)
    lgb_test_pred_log = lgb_model.predict(X_test)

    # 3. XGBoost
    xgb_model = xgb.XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        n_estimators=2500,
        learning_rate=0.05,
        max_depth=9,
        min_child_weight=10,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=0.1,
        reg_lambda=1.0,
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=100,
    )
    xgb_model.fit(X_train, y_train_log, eval_set=[(X_val, y_val_log)], verbose=False)
    xgb_val_pred_log = xgb_model.predict(X_val)
    xgb_test_pred_log = xgb_model.predict(X_test)

    # 4. CatBoost
    cb_model = cb.CatBoostRegressor(
        loss_function="RMSE",
        eval_metric="RMSE",
        iterations=2500,
        learning_rate=0.05,
        depth=8,
        l2_leaf_reg=3.0,
        random_seed=42,
        verbose=False,
        thread_count=-1,
        early_stopping_rounds=100,
    )
    cb_model.fit(X_train, y_train_log, eval_set=(X_val, y_val_log), verbose=False)
    cb_val_pred_log = cb_model.predict(X_val)
    cb_test_pred_log = cb_model.predict(X_test)

    # 5. PyTorch TabularResNet
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    train_dataset = TensorDataset(
        torch.tensor(X_train_scaled, dtype=torch.float32),
        torch.tensor(y_train_log, dtype=torch.float32),
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val_scaled, dtype=torch.float32),
        torch.tensor(y_val_log, dtype=torch.float32),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=2048,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=4096,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    epochs = 20
    nn_model = TabularResNet(num_features=X_train.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(nn_model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-5
    )

    best_val_loss = float("inf")
    best_state_dict = None

    for epoch in range(1, epochs + 1):
        nn_model.train()
        train_loss_sum, n_train_samples = 0.0, 0
        for x_b, y_b in train_loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            preds = nn_model(x_b)
            loss = criterion(preds, y_b)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * x_b.size(0)
            n_train_samples += x_b.size(0)

        scheduler.step()

        nn_model.eval()
        val_loss_sum, n_val_samples = 0.0, 0
        with torch.no_grad():
            for x_b, y_b in val_loader:
                x_b, y_b = x_b.to(device), y_b.to(device)
                preds = nn_model(x_b)
                loss = criterion(preds, y_b)
                val_loss_sum += loss.item() * x_b.size(0)
                n_val_samples += x_b.size(0)

        train_rmse = np.sqrt(train_loss_sum / n_train_samples)
        val_rmse = np.sqrt(val_loss_sum / n_val_samples)
        print(
            f"Epoch {epoch:02d}/{epochs:02d} | Train RMSE: {train_rmse:.4f} | Val"
            f" RMSE: {val_rmse:.4f}"
        )

        if val_rmse < best_val_loss:
            best_val_loss = val_rmse
            best_state_dict = {
                k: v.cpu().clone() for k, v in nn_model.state_dict().items()
            }

    nn_model.load_state_dict(best_state_dict)

    nn_model.eval()
    nn_val_list = []
    with torch.no_grad():
        for x_b, _ in val_loader:
            nn_val_list.append(nn_model(x_b.to(device)).cpu().numpy())
    nn_val_pred_log = np.concatenate(nn_val_list)

    test_dataset = TensorDataset(torch.tensor(X_test_scaled, dtype=torch.float32))
    test_loader = DataLoader(
        test_dataset,
        batch_size=4096,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    nn_test_list = []
    with torch.no_grad():
        for (x_b,) in test_loader:
            nn_test_list.append(nn_model(x_b.to(device)).cpu().numpy())
    nn_test_pred_log = np.concatenate(nn_test_list)

    # 6. SLSQP Ensemble Weight Optimization
    val_preds_matrix = np.column_stack(
        [
            lgb_val_pred_log,
            xgb_val_pred_log,
            cb_val_pred_log,
            nn_val_pred_log,
        ]
    )

    def loss_func(weights):
        weights = weights / np.sum(weights)
        blend_log = val_preds_matrix @ weights
        return np.sqrt(np.mean((y_val_log - blend_log) ** 2))

    init_weights = [0.25, 0.25, 0.25, 0.25]
    bounds = [(0, 1)] * 4
    constraints = {"type": "eq", "fun": lambda w: 1.0 - np.sum(w)}

    res = minimize(
        loss_func,
        init_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    opt_weights = res.x / np.sum(res.x)

    blend_val_log = val_preds_matrix @ opt_weights
    blend_val_actual = np.expm1(np.maximum(0, blend_val_log))

    final_val_rmsle = calculate_rmsle(y_val_actual, blend_val_actual)

    # 7. Test Predictions & Submission Generation
    test_preds_matrix = np.column_stack(
        [
            lgb_test_pred_log,
            xgb_test_pred_log,
            cb_test_pred_log,
            nn_test_pred_log,
        ]
    )
    blend_test_log = test_preds_matrix @ opt_weights
    final_test_predictions = np.expm1(np.maximum(0, blend_test_log))

    os.makedirs("./submission", exist_ok=True)
    submission = pd.DataFrame(
        {
            "id": np.arange(len(final_test_predictions)),
            "trip_duration": final_test_predictions,
        }
    )
    submission.to_csv("./submission/submission.csv", index=False)

    print(f"Final Validation Score: {final_val_rmsle}")


if __name__ == "__main__":
    main()
