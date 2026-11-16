import os
from typing import Any, Dict, List, Tuple
import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb


# ==============================================================================
# 1. Feature Engineering Functions
# ==============================================================================
def create_atmospheric_and_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Constructs domain-specific atmospheric physics, wind vector decomposition,

    and cyclical temporal features from raw meteorological measurements.
    """
    df = df.copy()

    # --- Temporal Features ---
    df["datetime"] = pd.to_datetime(
        df["year"].astype(str)
        + "-"
        + df["month"].astype(str).str.zfill(2)
        + "-"
        + df["day"].astype(str).str.zfill(2)
        + " "
        + df["hour"].astype(str).str.zfill(2)
        + ":00:00"
    )
    df["dayofweek"] = df["datetime"].dt.dayofweek
    df["dayofyear"] = df["datetime"].dt.dayofyear
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

    # Cyclical encodings (Hour, Month, Day of Year, Day of Week)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["month_sin"] = np.sin(2 * np.pi * (df["month"] - 1) / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * (df["month"] - 1) / 12.0)
    df["doy_sin"] = np.sin(2 * np.pi * (df["dayofyear"] - 1) / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * (df["dayofyear"] - 1) / 365.25)
    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7.0)

    # Beijing heating season: Nov 15 to Mar 15 (high coal emissions)
    df["is_heating_season"] = (
        (df["month"] == 11) & (df["day"] >= 15)
        | (df["month"].isin([12, 1, 2]))
        | (df["month"] == 3) & (df["day"] <= 15)
    ).astype(int)

    # Diurnal activity regimes
    df["is_rush_hour"] = df["hour"].isin([7, 8, 9, 17, 18, 19, 20]).astype(int)
    df["is_night_time"] = df["hour"].isin([22, 23, 0, 1, 2, 3, 4, 5]).astype(int)

    # --- Atmospheric Physics Features ---
    # Dew point depression: T - T_dew
    df["dew_point_depression"] = df["TEMP"] - df["DEWP"]

    # Relative Humidity (%) via Magnus-Tetens approximation
    sat_vp_temp = 6.112 * np.exp((17.67 * df["TEMP"]) / (df["TEMP"] + 243.5))
    act_vp_dewp = 6.112 * np.exp((17.67 * df["DEWP"]) / (df["DEWP"] + 243.5))
    df["relative_humidity"] = np.clip(
        100.0 * (act_vp_dewp / (sat_vp_temp + 1e-6)), 0.0, 100.0
    )

    # Vapor Pressure Deficit (VPD in kPa)
    df["vpd"] = np.maximum(0.0, (sat_vp_temp - act_vp_dewp) / 10.0)

    # Air Density Proxy (rho ~ P / (R_spec * T_Kelvin))
    df["temp_kelvin"] = df["TEMP"] + 273.15
    df["air_density_proxy"] = (df["PRES"] * 100.0) / (287.058 * df["temp_kelvin"])

    # Atmospheric Stagnation Index: High pressure + high humidity + low wind
    df["stagnation_index"] = (
        (df["PRES"] / 1013.25) * (df["relative_humidity"] / 100.0) / (df["Iws"] + 1.0)
    )

    # --- Wind Dynamics & Vector Decomposition ---
    wind_angle_map = {"NE": 45.0, "SE": 135.0, "SW": 225.0, "NW": 315.0, "cv": 0.0}
    wind_angles_deg = df["cbwd"].map(wind_angle_map).fillna(0.0)
    wind_angles_rad = np.radians(wind_angles_deg)

    df["wind_u"] = -df["Iws"] * np.sin(wind_angles_rad)
    df["wind_v"] = -df["Iws"] * np.cos(wind_angles_rad)

    df["is_calm_wind"] = ((df["cbwd"] == "cv") | (df["Iws"] <= 0.5)).astype(int)
    df["is_north_clean_wind"] = df["cbwd"].isin(["NW", "NE"]).astype(int)
    df["is_south_polluted_wind"] = df["cbwd"].isin(["SE", "SW"]).astype(int)

    # --- Precipitation Scavenging Features ---
    df["has_rain"] = (df["Ir"] > 0).astype(int)
    df["has_snow"] = (df["Is"] > 0).astype(int)
    df["total_precip_hours"] = df["Ir"] + df["Is"]
    df["cleansing_scavenging_index"] = (df["Ir"] * 1.5 + df["Is"] * 2.5) * (
        df["Iws"] + 0.5
    )

    # --- Non-linear Interaction Terms ---
    df["temp_pres_ratio"] = df["TEMP"] / (df["PRES"] + 1e-5)
    df["wind_humidity_dispersion"] = (
        df["Iws"] * (100.0 - df["relative_humidity"]) / 100.0
    )

    df = df.drop(columns=["datetime"])
    return df


def create_temporal_dynamics_features(df: pd.DataFrame) -> pd.DataFrame:
    """Computes multi-scale lags, leads, rolling statistics, EMAs, differences,
    and cumulative wind transport on continuous chronological exogenous meteorological variables.
    """
    df = df.copy()

    dynamics_cols = [
        "DEWP", "TEMP", "PRES", "Iws", "wind_u", "wind_v",
        "relative_humidity", "vpd", "dew_point_depression", "stagnation_index"
    ]

    # 1. Multi-scale Lags and Leads (strictly exogenous meteorological variables)
    lag_steps = [1, 2, 3, 4, 5, 6, 12, 24, 48, 72]
    lead_steps = [1, 2, 3, 6, 12, 24]

    for col in dynamics_cols:
        for lag in lag_steps:
            df[f"{col}_lag_{lag}"] = df[col].shift(lag)
        for lead in lead_steps:
            df[f"{col}_lead_{lead}"] = df[col].shift(-lead)

    # 2. Temporal Differences & Gradients
    diff_steps = [1, 2, 3, 6, 12, 24, 48]
    for col in ["PRES", "TEMP", "DEWP", "Iws", "wind_u", "wind_v", "relative_humidity"]:
        for step in diff_steps:
            df[f"{col}_diff_{step}h"] = df[col] - df[col].shift(step)

    # 3. Multi-Window Rolling Aggregations
    rolling_windows = [3, 6, 12, 24, 48, 72]
    for col in ["PRES", "TEMP", "DEWP", "Iws", "wind_u", "wind_v", "relative_humidity", "stagnation_index"]:
        for w in rolling_windows:
            df[f"{col}_roll_mean_{w}"] = df[col].rolling(w, min_periods=1).mean()
            df[f"{col}_roll_std_{w}"] = df[col].rolling(w, min_periods=1).std()
            df[f"{col}_roll_min_{w}"] = df[col].rolling(w, min_periods=1).min()
            df[f"{col}_roll_max_{w}"] = df[col].rolling(w, min_periods=1).max()

    # 4. Exponential Moving Averages (EMA)
    ema_spans = [3, 6, 12, 24, 48, 72]
    for col in ["PRES", "TEMP", "DEWP", "Iws", "wind_u", "wind_v", "relative_humidity"]:
        for span in ema_spans:
            df[f"{col}_ema_{span}"] = df[col].ewm(span=span, adjust=False).mean()

    # 5. Cumulative Wind Transport and Directional Dispersion Memory
    south_flux = np.maximum(0.0, df["wind_v"])
    north_flux = np.maximum(0.0, -df["wind_v"])
    east_flux = np.maximum(0.0, df["wind_u"])
    west_flux = np.maximum(0.0, -df["wind_u"])

    for w in [6, 12, 24, 48, 72]:
        df[f"south_flux_sum_{w}"] = south_flux.rolling(w, min_periods=1).sum()
        df[f"north_flux_sum_{w}"] = north_flux.rolling(w, min_periods=1).sum()
        df[f"east_flux_sum_{w}"] = east_flux.rolling(w, min_periods=1).sum()
        df[f"west_flux_sum_{w}"] = west_flux.rolling(w, min_periods=1).sum()
        df[f"wind_speed_sum_{w}"] = df["Iws"].rolling(w, min_periods=1).sum()
        df[f"precip_sum_{w}"] = (df["Ir"] + df["Is"]).rolling(w, min_periods=1).sum()

    # Clean initial boundary NaNs
    df = df.bfill().ffill()
    return df


# -----------------------------------------------------------------------------
# Data Preparation Pipeline
# -----------------------------------------------------------------------------
def prepare_data(
    train_path: str = "./input/train.parquet",
    test_path: str = "./input/test.parquet",
    n_splits: int = 5,
    seed: int = 42,
):
    train_raw = pd.read_parquet(train_path)
    test_raw = pd.read_parquet(test_path)

    target_col = "pm2.5"
    train_clean = train_raw.dropna(subset=[target_col]).copy()

    train_clean["_is_test"] = 0
    train_clean["_orig_idx"] = np.arange(len(train_clean))

    test_clean = test_raw.copy()
    test_clean["_is_test"] = 1
    test_clean["_orig_idx"] = np.arange(len(test_clean))

    combined = pd.concat([train_clean, test_clean], ignore_index=True)
    sort_cols = ["No"] if "No" in combined.columns else ["year", "month", "day", "hour"]
    combined = combined.sort_values(sort_cols).reset_index(drop=True)

    # 1. Base atmospheric and temporal features
    combined_features = create_atmospheric_and_temporal_features(combined)

    # 2. Continuous time-series dynamic features
    combined_features = create_temporal_dynamics_features(combined_features)

    # Split back cleanly
    train_features = (
        combined_features[combined_features["_is_test"] == 0]
        .sort_values("_orig_idx")
        .reset_index(drop=True)
    )
    test_features = (
        combined_features[combined_features["_is_test"] == 1]
        .sort_values("_orig_idx")
        .reset_index(drop=True)
    )

    # 3. Categorical encoding
    categorical_cols = ["cbwd"]
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    train_features[categorical_cols] = encoder.fit_transform(
        train_features[categorical_cols].astype(str)
    )
    test_features[categorical_cols] = encoder.transform(
        test_features[categorical_cols].astype(str)
    )

    ignore_cols = ["No", target_col, "_is_test", "_orig_idx"]
    feature_cols = [col for col in train_features.columns if col not in ignore_cols]

    numeric_cols = [c for c in feature_cols if c not in categorical_cols]
    medians = train_features[numeric_cols].median()
    train_features[numeric_cols] = train_features[numeric_cols].fillna(medians)
    test_features[numeric_cols] = test_features[numeric_cols].fillna(medians)

    X = train_features[feature_cols].copy()
    y = train_features[target_col].values.astype(np.float64)
    X_test = test_features[feature_cols].copy()

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    cv_splits = list(kf.split(X, y))

    print(
        f"Data Processing Complete: Train shape={X.shape}, Test shape={X_test.shape}, Features={len(feature_cols)}, Folds={n_splits}"
    )
    return X, y, X_test, feature_cols, cv_splits


# ==============================================================================
# 2. PyTorch Deep Tabular Neural Network Architecture
# ==============================================================================
class TabularResidualBlock(nn.Module):

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
        hidden_dims: List[int] = [256, 128, 64],
        num_blocks_per_stage: int = 2,
        dropout_rate: float = 0.15,
        output_dim: int = 1,
    ):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )

        stages = []
        in_dim = hidden_dims[0]
        for h_dim in hidden_dims:
            if in_dim != h_dim:
                stages.append(
                    nn.Sequential(
                        nn.Linear(in_dim, h_dim),
                        nn.LayerNorm(h_dim),
                        nn.GELU(),
                    )
                )
            for _ in range(num_blocks_per_stage):
                stages.append(
                    TabularResidualBlock(hidden_dim=h_dim, dropout_rate=dropout_rate)
                )
            in_dim = h_dim

        self.body = nn.Sequential(*stages)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dims[-1]),
            nn.Linear(hidden_dims[-1], hidden_dims[-1] // 2),
            nn.GELU(),
            nn.Linear(hidden_dims[-1] // 2, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_layer(x)
        h = self.body(h)
        out = self.head(h)
        return out.squeeze(-1)


def create_pytorch_criterion_and_optimizer(
    model: nn.Module,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    scheduler_t_max: int = 100,
) -> Tuple[nn.Module, torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay, eps=1e-8
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=scheduler_t_max, eta_min=1e-6
    )
    return criterion, optimizer, scheduler


# ==============================================================================
# 3. Model Parameter Factory
# ==============================================================================
def get_lgbm_model_params(seed: int = 42) -> Dict[str, Any]:
    return {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "n_estimators": 4000,
        "learning_rate": 0.03,
        "num_leaves": 127,
        "max_depth": 9,
        "min_child_samples": 20,
        "subsample": 0.85,
        "subsample_freq": 1,
        "colsample_bytree": 0.75,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": seed,
        "n_jobs": -1,
        "verbose": -1,
    }


def get_xgboost_model_params(seed: int = 42) -> Dict[str, Any]:
    return {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "n_estimators": 4000,
        "learning_rate": 0.03,
        "max_depth": 8,
        "min_child_weight": 3,
        "subsample": 0.85,
        "colsample_bytree": 0.75,
        "reg_alpha": 0.1,
        "reg_lambda": 1.5,
        "random_state": seed,
        "n_jobs": -1,
    }


def get_catboost_model_params(seed: int = 42) -> Dict[str, Any]:
    return {
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "iterations": 4000,
        "learning_rate": 0.04,
        "depth": 8,
        "l2_leaf_reg": 3.0,
        "subsample": 0.85,
        "random_seed": seed,
        "verbose": False,
        "thread_count": -1,
    }


# ==============================================================================
# 4. Training, Ensembling, and Evaluation Execution
# ==============================================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X, y, X_test, feature_cols, cv_splits = prepare_data()

    n_splits = len(cv_splits)
    n_samples_train = len(y)
    n_samples_test = len(X_test)

    oof_lgb = np.zeros(n_samples_train)
    oof_xgb = np.zeros(n_samples_train)
    oof_cb = np.zeros(n_samples_train)
    oof_nn = np.zeros(n_samples_train)

    test_lgb = np.zeros(n_samples_test)
    test_xgb = np.zeros(n_samples_test)
    test_cb = np.zeros(n_samples_test)
    test_nn = np.zeros(n_samples_test)

    print(
        f"Starting {n_splits}-Fold Training Pipeline on {device} for {len(feature_cols)} features..."
    )

    for fold, (train_idx, val_idx) in enumerate(cv_splits):
        X_tr_df, y_tr = X.iloc[train_idx].copy(), y[train_idx]
        X_va_df, y_val = X.iloc[val_idx].copy(), y[val_idx]
        X_te_df = X_test.copy()

        # A. LightGBM Regressor
        lgb_params = get_lgbm_model_params(seed=42 + fold)
        lgb_model = lgb.LGBMRegressor(**lgb_params)
        lgb_model.fit(
            X_tr_df,
            y_tr,
            eval_set=[(X_va_df, y_val)],
            callbacks=[
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(-1),
            ],
        )
        val_pred_lgb = lgb_model.predict(X_va_df)
        oof_lgb[val_idx] = val_pred_lgb
        test_lgb += lgb_model.predict(X_te_df) / n_splits

        # B. XGBoost Regressor
        xgb_params = get_xgboost_model_params(seed=42 + fold)
        xgb_model = xgb.XGBRegressor(**xgb_params, early_stopping_rounds=50)
        xgb_model.fit(X_tr_df, y_tr, eval_set=[(X_va_df, y_val)], verbose=False)
        val_pred_xgb = xgb_model.predict(X_va_df)
        oof_xgb[val_idx] = val_pred_xgb
        test_xgb += xgb_model.predict(X_te_df) / n_splits

        # C. CatBoost Regressor
        cb_params = get_catboost_model_params(seed=42 + fold)
        cb_model = cb.CatBoostRegressor(**cb_params, early_stopping_rounds=50)
        cb_model.fit(
            X_tr_df,
            y_tr,
            eval_set=(X_va_df, y_val),
            verbose=False,
            use_best_model=True,
        )
        val_pred_cb = cb_model.predict(X_va_df)
        oof_cb[val_idx] = val_pred_cb
        test_cb += cb_model.predict(X_te_df) / n_splits

        # D. Deep Tabular Neural Network (TabularResNet)
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr_df)
        X_va_scaled = scaler.transform(X_va_df)
        X_te_scaled = scaler.transform(X_te_df)

        train_tensor_x = torch.tensor(X_tr_scaled, dtype=torch.float32)
        train_tensor_y = torch.tensor(y_tr, dtype=torch.float32)
        val_tensor_x = torch.tensor(X_va_scaled, dtype=torch.float32)
        val_tensor_y = torch.tensor(y_val, dtype=torch.float32)
        test_tensor_x = torch.tensor(X_te_scaled, dtype=torch.float32).to(device)

        train_loader = DataLoader(
            TensorDataset(train_tensor_x, train_tensor_y),
            batch_size=256,
            shuffle=True,
        )

        nn_model = TabularResNet(
            input_dim=len(feature_cols),
            hidden_dims=[256, 128, 64],
            num_blocks_per_stage=2,
            dropout_rate=0.15,
        ).to(device)
        criterion, optimizer, scheduler = create_pytorch_criterion_and_optimizer(
            nn_model, lr=2e-3, weight_decay=1e-4, scheduler_t_max=50
        )

        best_val_loss = float("inf")
        best_nn_weights = None
        patience, patience_counter = 12, 0

        for epoch in range(50):
            nn_model.train()
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                out = nn_model(batch_x)
                loss = criterion(out, batch_y)
                loss.backward()
                optimizer.step()
            scheduler.step()

            nn_model.eval()
            with torch.no_grad():
                v_out = nn_model(val_tensor_x.to(device))
                v_loss = criterion(v_out, val_tensor_y.to(device)).item()

            if v_loss < best_val_loss:
                best_val_loss = v_loss
                best_nn_weights = {
                    k: v.cpu().clone() for k, v in nn_model.state_dict().items()
                }
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        if best_nn_weights is not None:
            nn_model.load_state_dict(
                {k: v.to(device) for k, v in best_nn_weights.items()}
            )
        nn_model.eval()
        with torch.no_grad():
            val_pred_nn = nn_model(val_tensor_x.to(device)).cpu().numpy()
            test_pred_nn = nn_model(test_tensor_x).cpu().numpy()

        oof_nn[val_idx] = val_pred_nn
        test_nn += test_pred_nn / n_splits

        fold_lgb_rmse = np.sqrt(
            mean_squared_error(y_val, np.clip(val_pred_lgb, 0, None))
        )
        fold_xgb_rmse = np.sqrt(
            mean_squared_error(y_val, np.clip(val_pred_xgb, 0, None))
        )
        fold_cb_rmse = np.sqrt(mean_squared_error(y_val, np.clip(val_pred_cb, 0, None)))
        fold_nn_rmse = np.sqrt(mean_squared_error(y_val, np.clip(val_pred_nn, 0, None)))
        print(
            f"Fold {fold + 1}/{n_splits} | LGB: {fold_lgb_rmse:.4f} | XGB: {fold_xgb_rmse:.4f} | CB: {fold_cb_rmse:.4f} | NN: {fold_nn_rmse:.4f}"
        )

    # Post-Processing & Constrained Ensemble Blending
    oof_lgb_clip = np.clip(oof_lgb, 0, None)
    oof_xgb_clip = np.clip(oof_xgb, 0, None)
    oof_cb_clip = np.clip(oof_cb, 0, None)
    oof_nn_clip = np.clip(oof_nn, 0, None)

    test_lgb_clip = np.clip(test_lgb, 0, None)
    test_xgb_clip = np.clip(test_xgb, 0, None)
    test_cb_clip = np.clip(test_cb, 0, None)
    test_nn_clip = np.clip(test_nn, 0, None)

    oof_matrix = np.column_stack([oof_lgb_clip, oof_xgb_clip, oof_cb_clip, oof_nn_clip])
    test_matrix = np.column_stack(
        [test_lgb_clip, test_xgb_clip, test_cb_clip, test_nn_clip]
    )

    def ensemble_loss(weights: np.ndarray) -> float:
        weights = weights / (np.sum(weights) + 1e-12)
        blend = np.dot(oof_matrix, weights)
        return np.sqrt(mean_squared_error(y, np.clip(blend, 0, None)))

    initial_weights = np.array([0.35, 0.35, 0.20, 0.10])
    bounds = [(0.0, 1.0)] * 4
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    opt_result = minimize(
        ensemble_loss,
        initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    best_weights = opt_result.x / np.sum(opt_result.x)
    print(f"Optimal Ensemble Weights: {best_weights.round(4)}")

    final_oof_predictions = np.clip(np.dot(oof_matrix, best_weights), 0, None)
    final_test_predictions = np.clip(np.dot(test_matrix, best_weights), 0, None)

    # Save Submission
    os.makedirs("./submission", exist_ok=True)
    submission_df = pd.DataFrame(
        {
            "id": np.arange(len(final_test_predictions)),
            "pm2.5": final_test_predictions,
        }
    )
    submission_df.to_csv("./submission/submission.csv", index=False)
    print(
        f"Submission saved successfully to ./submission/submission.csv with shape {submission_df.shape}"
    )

    # Validation Score Evaluation
    final_val_score = float(np.sqrt(mean_squared_error(y, final_oof_predictions)))
    print(f"Final Validation Score: {final_val_score}")


if __name__ == "__main__":
    main()