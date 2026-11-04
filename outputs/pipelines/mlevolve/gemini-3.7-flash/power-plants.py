import os
import re
from typing import Any, Dict, Optional, Tuple
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb


# ==============================================================================
# 1. Tabular Deep Architecture (TabularResNet) & PyTorch Components
# ==============================================================================
class TabularResidualBlock(nn.Module):
    """
    Residual feed-forward block with LayerNorm, Swish (SiLU), and Dropout.
    """

    def __init__(self, in_features: int, out_features: int, dropout_rate: float = 0.15):
        super().__init__()
        self.norm = nn.LayerNorm(in_features)
        self.fc1 = nn.Linear(in_features, out_features)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(out_features, out_features)

        if in_features != out_features:
            self.shortcut = nn.Linear(in_features, out_features)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.norm(x)
        out = self.fc1(out)
        out = self.act(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return self.act(out + residual)


class TabularResNet(nn.Module):
    """
    Deep Tabular Neural Network with residual connections tailored for regression tasks.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Tuple[int, ...] = (256, 256, 128, 64),
        dropout_rate: float = 0.15,
    ):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.SiLU(),
        )

        blocks = []
        for i in range(len(hidden_dims) - 1):
            blocks.append(
                TabularResidualBlock(
                    in_features=hidden_dims[i],
                    out_features=hidden_dims[i + 1],
                    dropout_rate=dropout_rate,
                )
            )
        self.res_blocks = nn.Sequential(*blocks)

        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dims[-1]),
            nn.Linear(hidden_dims[-1], 32),
            nn.SiLU(),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.input_layer(x)
        out = self.res_blocks(out)
        out = self.head(out)
        return out.squeeze(-1)


def get_torch_criterion_and_optimizer(
    model: nn.Module,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    t_max: int = 100,
    eta_min: float = 1e-6,
) -> Tuple[nn.Module, torch.optim.Optimizer, Any]:
    criterion = nn.MSELoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)
    return criterion, optimizer, scheduler


# ==============================================================================
# 2. Gradient Boosted Trees Model Configurations & Factory
# ==============================================================================
def get_lgbm_regressor_params(random_state: int = 42) -> Dict[str, Any]:
    return {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "n_estimators": 2500,
        "learning_rate": 0.03,
        "num_leaves": 63,
        "max_depth": 8,
        "min_child_samples": 20,
        "subsample": 0.85,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": random_state,
        "n_jobs": -1,
        "verbosity": -1,
    }


def get_xgb_regressor_params(random_state: int = 42) -> Dict[str, Any]:
    return {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "n_estimators": 2500,
        "learning_rate": 0.03,
        "max_depth": 7,
        "min_child_weight": 3,
        "subsample": 0.85,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": random_state,
        "n_jobs": -1,
        "tree_method": "hist",
    }


def create_model(
    model_type: str = "lgbm",
    input_dim: Optional[int] = None,
    random_state: int = 42,
    **kwargs,
) -> Any:
    model_type = model_type.lower()
    if model_type == "lgbm":
        params = get_lgbm_regressor_params(random_state=random_state)
        params.update(kwargs)
        return lgb.LGBMRegressor(**params)

    elif model_type in ["xgb", "xgboost"]:
        params = get_xgb_regressor_params(random_state=random_state)
        params.update(kwargs)
        return xgb.XGBRegressor(**params)

    elif model_type in ["nn", "resnet", "tabular_resnet"]:
        if input_dim is None:
            raise ValueError(
                "input_dim must be provided when instantiating TabularResNet."
            )
        return TabularResNet(input_dim=input_dim, **kwargs)

    else:
        raise ValueError(f"Unsupported model_type: '{model_type}'.")


# ==============================================================================
# 3. Main Data Preprocessing, Training, and Blending Pipeline
# ==============================================================================
def main():
    input_dir = "./input"
    submission_dir = "./submission"
    os.makedirs(submission_dir, exist_ok=True)

    print("Loading datasets...")
    train_df = pd.read_parquet(os.path.join(input_dir, "train.parquet"))
    test_df = pd.read_parquet(os.path.join(input_dir, "test.parquet"))

    target_col = "NAMPLT_CAP"
    if target_col not in train_df.columns:
        for col in train_df.columns:
            if col.upper() == target_col:
                target_col = col
                break

    # Extract test id
    if "id" in test_df.columns:
        test_ids = test_df["id"].values
    elif "ID" in test_df.columns:
        test_ids = test_df["ID"].values
    else:
        test_ids = np.arange(len(test_df))

    # Split train into train and validation sets prior to any feature transformations
    train_split, val_split = train_test_split(
        train_df, test_size=0.2, random_state=42, shuffle=True
    )

    y_train = train_split[target_col].values.astype(np.float32)
    y_val = val_split[target_col].values.astype(np.float32)

    X_train_raw = train_split.drop(columns=[target_col]).copy()
    X_val_raw = val_split.drop(columns=[target_col]).copy()
    X_test_raw = test_df.copy()
    if target_col in X_test_raw.columns:
        X_test_raw = X_test_raw.drop(columns=[target_col])

    # Identify column types
    lat_cols = [c for c in X_train_raw.columns if "LAT" in c.upper()]
    lon_cols = [
        c for c in X_train_raw.columns if "LON" in c.upper() or "LONG" in c.upper()
    ]
    lat_col = lat_cols[0] if lat_cols else None
    lon_col = lon_cols[0] if lon_cols else None

    all_cols = [c for c in X_train_raw.columns if c not in ["id", "ID"]]
    cat_cols = [
        c
        for c in all_cols
        if X_train_raw[c].dtype == "object"
        or isinstance(X_train_raw[c].dtype, pd.CategoricalDtype)
    ]
    num_cols = [
        c for c in all_cols if c not in cat_cols and c != lat_col and c != lon_col
    ]

    print(
        f"Detected {len(num_cols)} numerical, {len(cat_cols)} categorical features, "
        f"coordinates: ({lat_col}, {lon_col})"
    )

    # --- 1. Coordinate & Spatial Feature Engineering ---
    if lat_col and lon_col:
        for df in [X_train_raw, X_val_raw, X_test_raw]:
            lat_rad = np.radians(
                pd.to_numeric(df[lat_col], errors="coerce").fillna(0.0)
            )
            lon_rad = np.radians(
                pd.to_numeric(df[lon_col], errors="coerce").fillna(0.0)
            )
            df["coord_x"] = np.cos(lat_rad) * np.cos(lon_rad)
            df["coord_y"] = np.cos(lat_rad) * np.sin(lon_rad)
            df["coord_z"] = np.sin(lat_rad)

        train_coords = X_train_raw[["coord_x", "coord_y", "coord_z"]].values
        kmeans = KMeans(n_clusters=8, random_state=42, n_init=10)
        kmeans.fit(train_coords)

        for df in [X_train_raw, X_val_raw, X_test_raw]:
            coords = df[["coord_x", "coord_y", "coord_z"]].values
            df["spatial_cluster"] = kmeans.predict(coords)
            cluster_dists = kmeans.transform(coords)
            df["dist_to_nearest_cluster"] = np.min(cluster_dists, axis=1)
            df["dist_to_mean_cluster"] = np.mean(cluster_dists, axis=1)

        cat_cols.append("spatial_cluster")
        num_cols.extend(
            [
                "coord_x",
                "coord_y",
                "coord_z",
                "dist_to_nearest_cluster",
                "dist_to_mean_cluster",
            ]
        )
        if lat_col not in num_cols:
            num_cols.append(lat_col)
        if lon_col not in num_cols:
            num_cols.append(lon_col)

    # --- 2. Categorical Crosses & Interaction Features ---
    high_impact_cats = [
        c for c in cat_cols if c != "spatial_cluster" and X_train_raw[c].nunique() < 100
    ]
    if len(high_impact_cats) >= 2:
        top_cats = high_impact_cats[:3]
        for i in range(len(top_cats)):
            for j in range(i + 1, len(top_cats)):
                c1, c2 = top_cats[i], top_cats[j]
                inter_col = f"{c1}_x_{c2}"
                for df in [X_train_raw, X_val_raw, X_test_raw]:
                    df[inter_col] = df[c1].astype(str) + "_" + df[c2].astype(str)
                cat_cols.append(inter_col)

    # --- 3. Missing Value Imputation & Indicator Flags ---
    for c in num_cols:
        med_val = X_train_raw[c].median()
        for df in [X_train_raw, X_val_raw, X_test_raw]:
            df[f"{c}_isna"] = df[c].isna().astype(np.float32)
            df[c] = df[c].fillna(med_val)

    for c in cat_cols:
        for df in [X_train_raw, X_val_raw, X_test_raw]:
            df[c] = df[c].fillna("MISSING").astype(str)

    # --- 4. Group Aggregation Features (Train-Fitted Only) ---
    key_cat_candidates = [c for c in cat_cols if 2 <= X_train_raw[c].nunique() <= 80]
    key_num_candidates = [
        c for c in num_cols if not c.endswith("_isna") and "coord" not in c
    ][:3]

    for cat_c in key_cat_candidates[:2]:
        for num_c in key_num_candidates:
            agg_stats = (
                X_train_raw.groupby(cat_c)[num_c]
                .agg(["mean", "std", "min", "max"])
                .rename(
                    columns={
                        "mean": f"{cat_c}_{num_c}_mean",
                        "std": f"{cat_c}_{num_c}_std",
                        "min": f"{cat_c}_{num_c}_min",
                        "max": f"{cat_c}_{num_c}_max",
                    }
                )
                .reset_index()
            )
            for df in [X_train_raw, X_val_raw, X_test_raw]:
                merged = df[[cat_c]].merge(agg_stats, on=cat_c, how="left")
                for stat in ["mean", "std", "min", "max"]:
                    new_col_name = f"{cat_c}_{num_c}_{stat}"
                    df[new_col_name] = (
                        merged[new_col_name].fillna(X_train_raw[num_c].mean()).values
                    )

    # --- 5. Frequency & Ordinal Encoding for Categoricals ---
    for c in cat_cols:
        freq_map = X_train_raw[c].value_counts(normalize=True).to_dict()
        for df in [X_train_raw, X_val_raw, X_test_raw]:
            df[f"{c}_freq"] = df[c].map(freq_map).fillna(0.0).astype(np.float32)

    ord_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    ord_encoder.fit(X_train_raw[cat_cols])
    X_train_raw[cat_cols] = ord_encoder.transform(X_train_raw[cat_cols])
    X_val_raw[cat_cols] = ord_encoder.transform(X_val_raw[cat_cols])
    X_test_raw[cat_cols] = ord_encoder.transform(X_test_raw[cat_cols])

    # Remove non-feature columns
    feature_cols = [
        c
        for c in X_train_raw.columns
        if c not in ["id", "ID", target_col]
        and not re.search(r"(_name|_id)$", c, re.IGNORECASE)
    ]
    feature_cols = [
        c for c in feature_cols if pd.api.types.is_numeric_dtype(X_train_raw[c])
    ]

    X_train_df = (
        X_train_raw[feature_cols]
        .astype(np.float32)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )
    X_val_df = (
        X_val_raw[feature_cols]
        .astype(np.float32)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )
    X_test_df = (
        X_test_raw[feature_cols]
        .astype(np.float32)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )

    input_dim = len(feature_cols)
    print(
        f"Feature engineering complete: Train={X_train_df.shape}, Val={X_val_df.shape}, Test={X_test_df.shape}"
    )

    val_preds_dict = {}
    test_preds_dict = {}

    # --- Training LightGBM Model ---
    print("\n--- Training LightGBM Model ---")
    model_lgb = create_model("lgbm", random_state=42)
    lgb_callbacks = [
        lgb.early_stopping(stopping_rounds=50, verbose=False),
        lgb.log_evaluation(period=0),
    ]
    model_lgb.fit(
        X_train_df,
        y_train,
        eval_set=[(X_val_df, y_val)],
        callbacks=lgb_callbacks,
    )

    val_preds_lgb = np.clip(model_lgb.predict(X_val_df), 0, None)
    test_preds_lgb = np.clip(model_lgb.predict(X_test_df), 0, None)
    lgb_rmse = np.sqrt(mean_squared_error(y_val, val_preds_lgb))
    print(f"LightGBM Validation RMSE: {lgb_rmse:.4f}")

    val_preds_dict["lgb"] = val_preds_lgb
    test_preds_dict["lgb"] = test_preds_lgb

    # --- Training XGBoost Model ---
    print("\n--- Training XGBoost Model ---")
    model_xgb = create_model("xgb", early_stopping_rounds=50, random_state=42)
    model_xgb.fit(
        X_train_df,
        y_train,
        eval_set=[(X_val_df, y_val)],
        verbose=False,
    )

    val_preds_xgb = np.clip(model_xgb.predict(X_val_df), 0, None)
    test_preds_xgb = np.clip(model_xgb.predict(X_test_df), 0, None)
    xgb_rmse = np.sqrt(mean_squared_error(y_val, val_preds_xgb))
    print(f"XGBoost Validation RMSE: {xgb_rmse:.4f}")

    val_preds_dict["xgb"] = val_preds_xgb
    test_preds_dict["xgb"] = test_preds_xgb

    # --- Training PyTorch TabularResNet ---
    print("\n--- Training PyTorch TabularResNet ---")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_df.values).astype(np.float32).copy()
    X_val_scaled = scaler.transform(X_val_df.values).astype(np.float32).copy()
    X_test_scaled = scaler.transform(X_test_df.values).astype(np.float32).copy()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    train_dataset = TensorDataset(torch.tensor(X_train_scaled), torch.tensor(y_train))
    val_dataset = TensorDataset(torch.tensor(X_val_scaled), torch.tensor(y_val))

    train_loader = DataLoader(
        train_dataset, batch_size=256, shuffle=True, drop_last=False
    )
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)

    model_nn = create_model("nn", input_dim=input_dim, dropout_rate=0.15).to(device)
    epochs = 60
    criterion, optimizer, scheduler = get_torch_criterion_and_optimizer(
        model_nn, lr=2e-3, weight_decay=1e-4, t_max=epochs
    )

    best_nn_rmse = float("inf")
    best_nn_state = None
    patience = 15
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        model_nn.train()
        total_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            preds = model_nn(bx)
            loss = criterion(preds, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_nn.parameters(), max_norm=2.0)
            optimizer.step()
            total_loss += loss.item() * len(by)
        scheduler.step()

        model_nn.eval()
        val_preds_list = []
        with torch.no_grad():
            for bx, _ in val_loader:
                bx = bx.to(device)
                preds = model_nn(bx)
                val_preds_list.extend(preds.cpu().numpy())

        current_val_preds = np.clip(np.array(val_preds_list), 0, None)
        epoch_rmse = np.sqrt(mean_squared_error(y_val, current_val_preds))

        if epoch_rmse < best_nn_rmse:
            best_nn_rmse = epoch_rmse
            best_nn_state = {k: v.cpu() for k, v in model_nn.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 10 == 0 or epoch == epochs:
            print(
                f"Epoch [{epoch:02d}/{epochs}] - Train Loss: {total_loss / len(y_train):.2f} - Val RMSE: {epoch_rmse:.4f}"
            )

        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch}")
            break

    model_nn.load_state_dict(best_nn_state)
    model_nn.eval()
    with torch.no_grad():
        val_preds_nn = model_nn(torch.tensor(X_val_scaled).to(device)).cpu().numpy()
        test_preds_nn = model_nn(torch.tensor(X_test_scaled).to(device)).cpu().numpy()

    val_preds_nn = np.clip(val_preds_nn, 0, None)
    test_preds_nn = np.clip(test_preds_nn, 0, None)
    nn_rmse = np.sqrt(mean_squared_error(y_val, val_preds_nn))
    print(f"PyTorch TabularResNet Best Validation RMSE: {nn_rmse:.4f}")

    val_preds_dict["nn"] = val_preds_nn
    test_preds_dict["nn"] = test_preds_nn

    # --- 5. Optimal Constrained Blending ---
    models_list = ["lgb", "xgb", "nn"]
    val_matrix = np.column_stack([val_preds_dict[m] for m in models_list])
    test_matrix = np.column_stack([test_preds_dict[m] for m in models_list])

    def blend_objective(weights):
        w = np.array(weights)
        w = w / np.sum(w)
        pred_blend = np.clip(val_matrix @ w, 0, None)
        return np.sqrt(mean_squared_error(y_val, pred_blend))

    initial_weights = [1.0 / len(models_list)] * len(models_list)
    bounds = [(0.0, 1.0) for _ in models_list]
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

    res = minimize(
        blend_objective,
        initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    optimal_weights = res.x / np.sum(res.x)

    print(
        f"\nOptimal Model Blend Weights: "
        f"LightGBM={optimal_weights[0]:.3f}, XGBoost={optimal_weights[1]:.3f}, TabularResNet={optimal_weights[2]:.3f}"
    )

    final_val_preds = np.clip(val_matrix @ optimal_weights, 0, None)
    final_test_preds = np.clip(test_matrix @ optimal_weights, 0, None)

    final_val_rmse = np.sqrt(mean_squared_error(y_val, final_val_preds))

    # Fallback check: Ensure ensemble strictly outperforms or matches best single model
    min_single_rmse = min(lgb_rmse, xgb_rmse, nn_rmse)
    if final_val_rmse > min_single_rmse:
        best_single_idx = np.argmin([lgb_rmse, xgb_rmse, nn_rmse])
        best_model_name = models_list[best_single_idx]
        final_val_preds = val_preds_dict[best_model_name]
        final_test_preds = test_preds_dict[best_model_name]
        final_val_rmse = min_single_rmse

    # --- 6. Save Submission & Output Final Metric ---
    clean_test_ids = pd.to_numeric(test_ids, errors="coerce")
    if np.isnan(clean_test_ids).any():
        final_ids = np.arange(len(final_test_preds), dtype=int)
    else:
        final_ids = clean_test_ids.astype(int)

    submission_df = pd.DataFrame(
        {"id": final_ids, "NAMPLT_CAP": np.round(final_test_preds, 4)}
    )

    submission_path = os.path.join(submission_dir, "submission.csv")
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path} with shape {submission_df.shape}")

    print(f"Final Validation Score: {final_val_rmse}")


if __name__ == "__main__":
    main()
