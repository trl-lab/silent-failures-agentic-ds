import copy
import math
import os
import re
import warnings
import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from scipy.optimize import minimize
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

# =========================================================================
# 1. Data Loading and Feature Engineering
# =========================================================================


def load_data(input_dir="./input"):
    """Load train and test parquet files from input directory."""
    train_path = os.path.join(input_dir, "train.parquet")
    test_path = os.path.join(input_dir, "test.parquet")

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)
    return train_df, test_df


def detect_coordinate_columns(df):
    """Detect latitude and longitude column names flexibly."""
    lat_col = None
    lon_col = None
    for col in df.columns:
        col_upper = col.upper()
        if col_upper in ["LATITUDE", "LAT", "Y", "SURF_LAT", "SURF_LATITUDE"]:
            lat_col = col
        elif col_upper in [
            "LONGITUDE",
            "LON",
            "LONG",
            "X",
            "SURF_LON",
            "SURF_LONGITUDE",
        ]:
            lon_col = col
    return lat_col, lon_col


def extract_datetime_features(df, col):
    """Extract granular temporal features from a datetime column."""
    features = pd.DataFrame(index=df.index)
    dt_series = pd.to_datetime(df[col], errors="coerce")
    features[f"{col}_year"] = dt_series.dt.year.fillna(-1).astype(np.float32)
    features[f"{col}_month"] = dt_series.dt.month.fillna(-1).astype(np.float32)
    features[f"{col}_day"] = dt_series.dt.day.fillna(-1).astype(np.float32)
    features[f"{col}_dayofyear"] = dt_series.dt.dayofyear.fillna(-1).astype(np.float32)
    features[f"{col}_quarter"] = dt_series.dt.quarter.fillna(-1).astype(np.float32)
    features[f"{col}_is_null"] = dt_series.isna().astype(np.float32)
    return features


def engineer_features(train_df, test_df, target_col="WATERDEPTH"):
    """Clean data and generate spatial, categorical, and temporal features."""
    train = train_df.copy()
    test = test_df.copy()

    # Store target and drop from feature processing
    y_train = None
    if target_col in train.columns:
        y_train = train[target_col].values
        train = train.drop(columns=[target_col])

    # Preserve ID columns if present
    id_cols = [c for c in train.columns if "id" in c.lower() or c.lower() == "index"]

    # Identify coordinate columns
    lat_col, lon_col = detect_coordinate_columns(train)

    # Identify column types
    datetime_cols = []
    categorical_cols = []
    numeric_cols = []

    for c in train.columns:
        if c in id_cols:
            continue
        if (
            pd.api.types.is_datetime64_any_dtype(train[c])
            or "date" in c.lower()
            or "year" in c.lower()
            or "time" in c.lower()
        ):
            try:
                pd.to_datetime(train[c].dropna().head(10))
                datetime_cols.append(c)
                continue
            except Exception:
                pass

        if pd.api.types.is_numeric_dtype(train[c]):
            numeric_cols.append(c)
        else:
            categorical_cols.append(c)

    # Process Datetime columns
    train_dt_feats = []
    test_dt_feats = []
    for c in datetime_cols:
        train_dt_feats.append(extract_datetime_features(train, c))
        test_dt_feats.append(extract_datetime_features(test, c))

    if train_dt_feats:
        train_dt_df = pd.concat(train_dt_feats, axis=1)
        test_dt_df = pd.concat(test_dt_feats, axis=1)
    else:
        train_dt_df = pd.DataFrame(index=train.index)
        test_dt_df = pd.DataFrame(index=test.index)

    # Spatial Engineering if coordinates exist
    train_spatial = pd.DataFrame(index=train.index)
    test_spatial = pd.DataFrame(index=test.index)

    if lat_col and lon_col:
        train_lat = train[lat_col].fillna(train[lat_col].median()).values
        train_lon = train[lon_col].fillna(train[lon_col].median()).values
        test_lat = test[lat_col].fillna(train[lat_col].median()).values
        test_lon = test[lon_col].fillna(train[lon_col].median()).values

        # 3D Cartesian coordinates on unit sphere
        lat_rad_tr = np.radians(train_lat)
        lon_rad_tr = np.radians(train_lon)
        lat_rad_te = np.radians(test_lat)
        lon_rad_te = np.radians(test_lon)

        train_spatial["coord_x_sphere"] = np.cos(lat_rad_tr) * np.cos(lon_rad_tr)
        train_spatial["coord_y_sphere"] = np.cos(lat_rad_tr) * np.sin(lon_rad_tr)
        train_spatial["coord_z_sphere"] = np.sin(lat_rad_tr)

        test_spatial["coord_x_sphere"] = np.cos(lat_rad_te) * np.cos(lon_rad_te)
        test_spatial["coord_y_sphere"] = np.cos(lat_rad_te) * np.sin(lon_rad_te)
        test_spatial["coord_z_sphere"] = np.sin(lat_rad_te)

        # Multi-angle coordinate rotations
        for angle in [15, 30, 45, 60, 75]:
            rad = np.radians(angle)
            cos_a, sin_a = np.cos(rad), np.sin(rad)
            train_spatial[f"rot_coord_{angle}_x"] = (
                train_lat * cos_a - train_lon * sin_a
            )
            train_spatial[f"rot_coord_{angle}_y"] = (
                train_lat * sin_a + train_lon * cos_a
            )
            test_spatial[f"rot_coord_{angle}_x"] = test_lat * cos_a - test_lon * sin_a
            test_spatial[f"rot_coord_{angle}_y"] = test_lat * sin_a + test_lon * cos_a

        # Distance to centroid
        lat_mean, lon_mean = np.mean(train_lat), np.mean(train_lon)
        train_spatial["dist_to_center"] = np.sqrt(
            (train_lat - lat_mean) ** 2 + (train_lon - lon_mean) ** 2
        )
        test_spatial["dist_to_center"] = np.sqrt(
            (test_lat - lat_mean) ** 2 + (test_lon - lon_mean) ** 2
        )

        # Spatial k-NN distance features strictly fitted on training coordinates
        coords_train = np.column_stack([train_lat, train_lon])
        coords_test = np.column_stack([test_lat, test_lon])

        for k in [3, 5, 10, 20]:
            if len(coords_train) > k:
                knn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean", n_jobs=-1)
                knn.fit(coords_train)

                tr_distances, _ = knn.kneighbors(coords_train)
                train_spatial[f"knn_{k}_mean_dist"] = tr_distances[:, 1:].mean(axis=1)
                train_spatial[f"knn_{k}_min_dist"] = tr_distances[:, 1]
                train_spatial[f"knn_{k}_max_dist"] = tr_distances[:, -1]

                te_distances, _ = knn.kneighbors(coords_test)
                test_spatial[f"knn_{k}_mean_dist"] = te_distances[:, :k].mean(axis=1)
                test_spatial[f"knn_{k}_min_dist"] = te_distances[:, 0]
                test_spatial[f"knn_{k}_max_dist"] = te_distances[:, k - 1]

    # Numeric feature processing & missing indicators
    train_num = pd.DataFrame(index=train.index)
    test_num = pd.DataFrame(index=test.index)

    for c in numeric_cols:
        if c in [lat_col, lon_col]:
            train_num[c] = train[c].astype(np.float32)
            test_num[c] = test[c].astype(np.float32)
            continue

        med_val = train[c].median()
        train_num[c] = train[c].fillna(med_val).astype(np.float32)
        test_num[c] = test[c].fillna(med_val).astype(np.float32)

        if train[c].isna().sum() > 0 or test[c].isna().sum() > 0:
            train_num[f"{c}_isna"] = train[c].isna().astype(np.float32)
            test_num[f"{c}_isna"] = test[c].isna().astype(np.float32)

    # Categorical feature processing (Frequency Encoding & Ordinal Encoding)
    train_cat = pd.DataFrame(index=train.index)
    test_cat = pd.DataFrame(index=test.index)

    for c in categorical_cols:
        train_str = train[c].fillna("MISSING").astype(str)
        test_str = test[c].fillna("MISSING").astype(str)

        freq = train_str.value_counts(normalize=True).to_dict()
        train_cat[f"{c}_freq"] = train_str.map(freq).fillna(0.0).astype(np.float32)
        test_cat[f"{c}_freq"] = test_str.map(freq).fillna(0.0).astype(np.float32)

        le = LabelEncoder()
        le.fit(list(train_str.unique()) + ["__UNSEEN__"])

        train_cat[f"{c}_encoded"] = le.transform(train_str).astype(np.int32)
        test_cat[f"{c}_encoded"] = test_str.map(
            lambda s: (
                le.transform([s])[0]
                if s in le.classes_
                else le.transform(["__UNSEEN__"])[0]
            )
        ).astype(np.int32)

    # Categorical interaction pair encoding
    if len(categorical_cols) >= 2:
        top_cats = categorical_cols[:3]
        for i in range(len(top_cats)):
            for j in range(i + 1, len(top_cats)):
                c1, c2 = top_cats[i], top_cats[j]
                inter_tr = (
                    train[c1].fillna("MISSING").astype(str)
                    + "_"
                    + train[c2].fillna("MISSING").astype(str)
                )
                inter_te = (
                    test[c1].fillna("MISSING").astype(str)
                    + "_"
                    + test[c2].fillna("MISSING").astype(str)
                )

                freq_inter = inter_tr.value_counts(normalize=True).to_dict()
                train_cat[f"{c1}_{c2}_inter_freq"] = (
                    inter_tr.map(freq_inter).fillna(0.0).astype(np.float32)
                )
                test_cat[f"{c1}_{c2}_inter_freq"] = (
                    inter_te.map(freq_inter).fillna(0.0).astype(np.float32)
                )

    # Combine all feature subsets
    X_train = pd.concat([train_num, train_spatial, train_dt_df, train_cat], axis=1)
    X_test = pd.concat([test_num, test_spatial, test_dt_df, test_cat], axis=1)

    X_train = X_train.loc[:, ~X_train.columns.duplicated()]
    X_test = X_test.loc[:, ~X_test.columns.duplicated()]

    common_cols = [col for col in X_train.columns if col in X_test.columns]
    X_train = X_train[common_cols]
    X_test = X_test[common_cols]

    # Create Stratified 5-Fold split based on binned depth target
    if y_train is not None:
        y_train_clean = np.maximum(y_train, 0.0)
        num_bins = min(10, len(np.unique(y_train_clean)))
        binned_target = pd.qcut(
            y_train_clean, q=num_bins, labels=False, duplicates="drop"
        )

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        folds = np.zeros(len(X_train), dtype=np.int32)
        for fold, (_, val_idx) in enumerate(skf.split(X_train, binned_target)):
            folds[val_idx] = fold

        X_train["fold"] = folds
        X_train[target_col] = y_train_clean

    return X_train, X_test


# =========================================================================
# 2. PyTorch Architecture and GBDT Model Builders
# =========================================================================


class ResidualBlock(nn.Module):
    """Residual dense block with LayerNorm, GELU activation, and Dropout."""

    def __init__(self, hidden_dim: int, dropout_rate: float = 0.15):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.block(x))


class TabularResNet(nn.Module):
    """Deep Tabular ResNet for continuous depth estimation."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_blocks: int = 3,
        dropout_rate: float = 0.15,
    ):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()
        )
        self.res_blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim, dropout_rate) for _ in range(num_blocks)]
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.input_layer(x)
        for block in self.res_blocks:
            feat = block(feat)
        out = self.head(feat)
        return out.squeeze(-1)


class RMSELoss(nn.Module):
    """Root Mean Squared Error loss criterion for direct metric alignment."""

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.mse = nn.MSELoss(reduction="mean")
        self.eps = eps

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(self.mse(y_pred, y_true) + self.eps)


def get_neural_components(
    input_dim: int,
    lr: float = 2e-3,
    weight_decay: float = 1e-4,
    epochs: int = 40,
):
    """Instantiate TabularResNet model, RMSE criterion, AdamW optimizer, and Cosine scheduler."""
    model = TabularResNet(
        input_dim=input_dim, hidden_dim=256, num_blocks=3, dropout_rate=0.15
    )
    criterion = RMSELoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    return model, criterion, optimizer, scheduler


def create_lightgbm_model(random_state: int = 42) -> LGBMRegressor:
    """Build LightGBM regressor with parameters tuned for depth regression."""
    return LGBMRegressor(
        objective="regression",
        metric="rmse",
        n_estimators=2500,
        learning_rate=0.03,
        num_leaves=63,
        max_depth=8,
        min_child_samples=20,
        subsample=0.85,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
    )


def create_xgboost_model(random_state: int = 42) -> XGBRegressor:
    """Build XGBoost regressor for bathymetric depth estimation."""
    return XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        n_estimators=2500,
        learning_rate=0.03,
        max_depth=7,
        subsample=0.85,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=random_state,
        n_jobs=-1,
        tree_method="hist",
        early_stopping_rounds=50,
    )


def create_catboost_model(random_state: int = 42) -> CatBoostRegressor:
    """Build CatBoost regressor for oblivious tree depth learning."""
    return CatBoostRegressor(
        loss_function="RMSE",
        eval_metric="RMSE",
        iterations=2500,
        learning_rate=0.03,
        depth=7,
        l2_leaf_reg=3.0,
        random_seed=random_state,
        verbose=0,
        thread_count=-1,
    )


# =========================================================================
# 3. Main Training, Ensembling, and Evaluation Workflow
# =========================================================================


def main():
    os.makedirs("./working", exist_ok=True)
    os.makedirs("./submission", exist_ok=True)

    print("Loading data from ./input...")
    train_raw, test_raw = load_data("./input")
    print(f"Raw Train Shape: {train_raw.shape}, Raw Test Shape: {test_raw.shape}")

    print("Engineering features, spatial geometry, and cross-validation folds...")
    target_col = "WATERDEPTH"
    train_processed, test_processed = engineer_features(
        train_raw, test_raw, target_col=target_col
    )

    feature_cols = [c for c in train_processed.columns if c not in ["fold", target_col]]
    y = train_processed[target_col].values.astype(np.float32)
    folds = train_processed["fold"].values
    n_splits = len(np.unique(folds))

    X_train_full = train_processed[feature_cols].copy().fillna(0.0)
    X_test_full = test_processed[feature_cols].copy().fillna(0.0)

    # Initialize out-of-fold and test prediction matrices
    oof_lgb = np.zeros(len(train_processed), dtype=np.float32)
    oof_xgb = np.zeros(len(train_processed), dtype=np.float32)
    oof_cb = np.zeros(len(train_processed), dtype=np.float32)
    oof_nn = np.zeros(len(train_processed), dtype=np.float32)

    test_lgb = np.zeros(len(test_processed), dtype=np.float32)
    test_xgb = np.zeros(len(test_processed), dtype=np.float32)
    test_cb = np.zeros(len(test_processed), dtype=np.float32)
    test_nn = np.zeros(len(test_processed), dtype=np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"Starting 5-Fold Training on {len(feature_cols)} features with 4 heterogeneous models..."
    )

    for fold in range(n_splits):
        train_mask = folds != fold
        val_mask = folds == fold

        X_tr = X_train_full.iloc[train_mask]
        y_tr = y[train_mask]
        X_va = X_train_full.iloc[val_mask]
        y_va = y[val_mask]
        val_indices = np.where(val_mask)[0]

        # 1. Train LightGBM
        lgb_model = create_lightgbm_model(random_state=42 + fold)
        lgb_callbacks = [
            lgb.early_stopping(50, verbose=False),
            lgb.log_evaluation(0),
        ]
        lgb_model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=lgb_callbacks)

        val_pred_lgb = lgb_model.predict(X_va)
        oof_lgb[val_indices] = val_pred_lgb
        test_lgb += lgb_model.predict(X_test_full) / n_splits

        # 2. Train XGBoost
        xgb_model = create_xgboost_model(random_state=42 + fold)
        xgb_model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

        val_pred_xgb = xgb_model.predict(X_va)
        oof_xgb[val_indices] = val_pred_xgb
        test_xgb += xgb_model.predict(X_test_full) / n_splits

        # 3. Train CatBoost
        cb_model = create_catboost_model(random_state=42 + fold)
        cb_model.fit(
            X_tr,
            y_tr,
            eval_set=(X_va, y_va),
            early_stopping_rounds=50,
            verbose=0,
        )

        val_pred_cb = cb_model.predict(X_va)
        oof_cb[val_indices] = val_pred_cb
        test_cb += cb_model.predict(X_test_full) / n_splits

        # 4. Train Deep TabularResNet with Standard Scaling
        scaler = StandardScaler()
        X_tr_sc = np.nan_to_num(scaler.fit_transform(X_tr))
        X_va_sc = np.nan_to_num(scaler.transform(X_va))
        X_te_sc = np.nan_to_num(scaler.transform(X_test_full))

        train_dataset = TensorDataset(
            torch.tensor(X_tr_sc, dtype=torch.float32),
            torch.tensor(y_tr, dtype=torch.float32),
        )
        val_dataset = TensorDataset(
            torch.tensor(X_va_sc, dtype=torch.float32),
            torch.tensor(y_va, dtype=torch.float32),
        )

        train_loader = DataLoader(
            train_dataset, batch_size=128, shuffle=True, drop_last=False
        )
        val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

        torch.manual_seed(42 + fold)
        epochs = 40
        nn_model, criterion, optimizer, scheduler = get_neural_components(
            input_dim=X_tr_sc.shape[1], lr=2e-3, weight_decay=1e-4, epochs=epochs
        )
        nn_model = nn_model.to(device)

        best_val_loss = float("inf")
        best_weights = copy.deepcopy(nn_model.state_dict())
        patience = 10
        patience_counter = 0

        for epoch in range(epochs):
            nn_model.train()
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                preds = nn_model(batch_x)
                loss = criterion(preds, batch_y)
                loss.backward()
                optimizer.step()
            scheduler.step()

            nn_model.eval()
            val_preds_list = []
            with torch.no_grad():
                for batch_x, _ in val_loader:
                    batch_x = batch_x.to(device)
                    val_preds_list.append(nn_model(batch_x).cpu().numpy())
            epoch_val_preds = np.concatenate(val_preds_list)
            epoch_val_loss = np.sqrt(np.mean((y_va - epoch_val_preds) ** 2))

            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                best_weights = copy.deepcopy(nn_model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        nn_model.load_state_dict(best_weights)
        nn_model.eval()

        with torch.no_grad():
            val_preds_list = []
            for batch_x, _ in val_loader:
                batch_x = batch_x.to(device)
                val_preds_list.append(nn_model(batch_x).cpu().numpy())
            val_pred_nn = np.concatenate(val_preds_list)

            test_tensor = torch.tensor(X_te_sc, dtype=torch.float32).to(device)
            test_pred_nn = nn_model(test_tensor).cpu().numpy()

        oof_nn[val_indices] = val_pred_nn
        test_nn += test_pred_nn / n_splits

        fold_rmse = root_mean_squared_error(
            y_va,
            (val_pred_lgb + val_pred_xgb + val_pred_cb + val_pred_nn) / 4.0,
        )
        print(f"Fold {fold+1}/{n_splits} - Ensemble Validation RMSE: {fold_rmse:.4f}")

    # Ensembling via Constrained Quadratic Optimization
    oof_matrix = np.column_stack([oof_lgb, oof_xgb, oof_cb, oof_nn])
    test_matrix = np.column_stack([test_lgb, test_xgb, test_cb, test_nn])

    def ensemble_loss_fn(weights):
        norm_w = weights / np.sum(weights)
        pred = oof_matrix @ norm_w
        return np.sqrt(np.mean((y - pred) ** 2))

    init_w = np.ones(4) / 4.0
    bounds = [(0.0, 1.0) for _ in range(4)]
    constraints = {"type": "eq", "fun": lambda w: 1.0 - np.sum(w)}
    opt_res = minimize(
        ensemble_loss_fn,
        init_w,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    optimal_weights = opt_res.x / np.sum(opt_res.x)

    # Compute final out-of-fold predictions with physical non-negativity constraint
    oof_final = np.maximum(oof_matrix @ optimal_weights, 0.0)
    test_final = np.maximum(test_matrix @ optimal_weights, 0.0)

    final_score = root_mean_squared_error(y, oof_final)

    # Generate submission file
    submission_df = pd.DataFrame(
        {"id": np.arange(len(test_final)), target_col: test_final}
    )
    submission_path = "./submission/submission.csv"
    submission_df.to_csv(submission_path, index=False)

    print(
        f"Optimal Ensemble Weights: LGB={optimal_weights[0]:.3f}, XGB={optimal_weights[1]:.3f}, CB={optimal_weights[2]:.3f}, ResNet={optimal_weights[3]:.3f}"
    )
    print(f"Predictions saved to {submission_path} with shape {submission_df.shape}")
    print(f"Final Validation Score: {final_score}")


if __name__ == "__main__":
    main()
