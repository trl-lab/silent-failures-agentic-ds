import os
import random
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor


def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


# =========================================================================
# 1. Feature Engineering & Psychrometric Transformations
# =========================================================================
def calculate_psychrometric_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    new_cols = {}

    p_hpa = (
        out["Press_mm_hg"] * 1.33322
        if "Press_mm_hg" in out.columns
        else pd.Series(1013.25, index=out.index)
    )

    indoor_zones = [1, 2, 3, 4, 5, 7, 8, 9]
    all_temp_cols = [f"T{i}" for i in indoor_zones]
    all_rh_cols = [f"RH_{i}" for i in indoor_zones]
    enthalpy_cols = []
    vpd_cols = []

    for i in range(1, 10):
        t_col = f"T{i}"
        rh_col = f"RH_{i}"
        if t_col in out.columns and rh_col in out.columns:
            t_val = out[t_col]
            rh_val = np.clip(out[rh_col], 0.0, 100.0)

            # Saturation vapor pressure (hPa)
            e_sat = 6.112 * np.exp((17.67 * t_val) / (t_val + 243.5))
            # Actual vapor pressure (hPa)
            e_act = e_sat * (rh_val / 100.0)
            # Vapor Pressure Deficit (hPa)
            vpd = np.maximum(0.0, e_sat - e_act)
            # Absolute humidity (g/m^3)
            ah = (216.7 * e_act) / (t_val + 273.15)

            # Magnus-Tetens Dewpoint
            alpha = (17.27 * t_val) / (237.7 + t_val) + np.log(
                np.clip(rh_val, 1e-3, 100.0) / 100.0
            )
            tdew = (237.7 * alpha) / (17.27 - alpha)
            tdew_diff = t_val - tdew

            # Humidity ratio w (kg water / kg dry air)
            w = (0.622 * e_act) / np.maximum(10.0, p_hpa - e_act)
            # Specific Enthalpy h (kJ / kg dry air)
            h = 1.006 * t_val + w * (2501.0 + 1.86 * t_val)

            # Humidex & Apparent Temperature
            humidex = t_val + (5.0 / 9.0) * (e_act - 10.0)
            apparent_temp = t_val + 0.33 * e_act - 4.0

            new_cols[f"VP_sat_{i}"] = e_sat.astype(np.float32)
            new_cols[f"VP_act_{i}"] = e_act.astype(np.float32)
            new_cols[f"VPD_{i}"] = vpd.astype(np.float32)
            new_cols[f"AH_{i}"] = ah.astype(np.float32)
            new_cols[f"Tdew_{i}"] = tdew.astype(np.float32)
            new_cols[f"Tdew_diff_{i}"] = tdew_diff.astype(np.float32)
            new_cols[f"Enthalpy_{i}"] = h.astype(np.float32)
            new_cols[f"Humidex_{i}"] = humidex.astype(np.float32)
            new_cols[f"ApparentTemp_{i}"] = apparent_temp.astype(np.float32)

            if i in indoor_zones:
                enthalpy_cols.append(f"Enthalpy_{i}")
                vpd_cols.append(f"VPD_{i}")

    if "T_out" in out.columns and "RH_out" in out.columns:
        t_out = out["T_out"]
        rh_out = np.clip(out["RH_out"], 0.0, 100.0)
        e_sat_out = 6.112 * np.exp((17.67 * t_out) / (t_out + 243.5))
        e_act_out = e_sat_out * (rh_out / 100.0)
        vpd_out = np.maximum(0.0, e_sat_out - e_act_out)
        ah_out = (216.7 * e_act_out) / (t_out + 273.15)
        w_out = (0.622 * e_act_out) / np.maximum(10.0, p_hpa - e_act_out)
        h_out = 1.006 * t_out + w_out * (2501.0 + 1.86 * t_out)

        new_cols["VP_sat_out"] = e_sat_out.astype(np.float32)
        new_cols["VP_act_out"] = e_act_out.astype(np.float32)
        new_cols["VPD_out"] = vpd_out.astype(np.float32)
        new_cols["AH_out"] = ah_out.astype(np.float32)
        new_cols["Enthalpy_out"] = h_out.astype(np.float32)

        if "Tdewpoint" in out.columns:
            new_cols["Tdew_depression_out"] = (out["T_out"] - out["Tdewpoint"]).astype(
                np.float32
            )
        if "Windspeed" in out.columns:
            new_cols["ApparentTemp_out"] = (
                t_out + 0.33 * e_act_out - 0.70 * out["Windspeed"] - 4.0
            ).astype(np.float32)

    valid_t_cols = [c for c in all_temp_cols if c in out.columns]
    valid_rh_cols = [c for c in all_rh_cols if c in out.columns]

    if valid_t_cols:
        new_cols["T_indoor_mean"] = out[valid_t_cols].mean(axis=1).astype(np.float32)
        new_cols["T_indoor_std"] = out[valid_t_cols].std(axis=1).astype(np.float32)
        new_cols["T_indoor_min"] = out[valid_t_cols].min(axis=1).astype(np.float32)
        new_cols["T_indoor_max"] = out[valid_t_cols].max(axis=1).astype(np.float32)
        new_cols["T_indoor_range"] = (
            new_cols["T_indoor_max"] - new_cols["T_indoor_min"]
        ).astype(np.float32)

    if valid_rh_cols:
        new_cols["RH_indoor_mean"] = out[valid_rh_cols].mean(axis=1).astype(np.float32)
        new_cols["RH_indoor_std"] = out[valid_rh_cols].std(axis=1).astype(np.float32)
        new_cols["RH_indoor_min"] = out[valid_rh_cols].min(axis=1).astype(np.float32)
        new_cols["RH_indoor_max"] = out[valid_rh_cols].max(axis=1).astype(np.float32)
        new_cols["RH_indoor_range"] = (
            new_cols["RH_indoor_max"] - new_cols["RH_indoor_min"]
        ).astype(np.float32)

    if enthalpy_cols:
        enthalpy_df = pd.DataFrame(
            {col: new_cols[col] for col in enthalpy_cols}, index=out.index
        )
        new_cols["Enthalpy_indoor_mean"] = (
            enthalpy_df.mean(axis=1).astype(np.float32)
        )
        new_cols["Enthalpy_indoor_std"] = enthalpy_df.std(axis=1).astype(np.float32)
        new_cols["Enthalpy_indoor_range"] = (
            enthalpy_df.max(axis=1) - enthalpy_df.min(axis=1)
        ).astype(np.float32)

    if vpd_cols:
        vpd_df = pd.DataFrame(
            {col: new_cols[col] for col in vpd_cols}, index=out.index
        )
        new_cols["VPD_indoor_mean"] = vpd_df.mean(axis=1).astype(np.float32)
        new_cols["VPD_indoor_std"] = vpd_df.std(axis=1).astype(np.float32)

    if "T_out" in out.columns and "T_indoor_mean" in new_cols:
        new_cols["T_in_out_diff"] = (
            new_cols["T_indoor_mean"] - out["T_out"]
        ).astype(np.float32)
    if "RH_out" in out.columns and "RH_indoor_mean" in new_cols:
        new_cols["RH_in_out_diff"] = (
            new_cols["RH_indoor_mean"] - out["RH_out"]
        ).astype(np.float32)
    if "Enthalpy_out" in new_cols and "Enthalpy_indoor_mean" in new_cols:
        new_cols["Enthalpy_in_out_diff"] = (
            new_cols["Enthalpy_indoor_mean"] - new_cols["Enthalpy_out"]
        ).astype(np.float32)

    # Thermal stratification gradients
    if "T1" in out.columns and "T2" in out.columns:
        new_cols["T_kitchen_living_diff"] = (out["T1"] - out["T2"]).astype(np.float32)
    if "T3" in out.columns and "T5" in out.columns:
        new_cols["T_laundry_bath_diff"] = (out["T3"] - out["T5"]).astype(np.float32)
    if "T8" in out.columns and "T9" in out.columns:
        new_cols["T_teen_parent_diff"] = (out["T8"] - out["T9"]).astype(np.float32)
    if "T6" in out.columns and "T_out" in out.columns:
        new_cols["T6_Tout_diff"] = (out["T6"] - out["T_out"]).astype(np.float32)

    # Vertical gradient: Upper floor (T8, T9) vs Ground floor (T1, T2)
    if all(k in out.columns for k in ["T1", "T2", "T8", "T9"]):
        new_cols["T_vertical_gradient"] = (
            0.5 * (out["T8"] + out["T9"]) - 0.5 * (out["T1"] + out["T2"])
        ).astype(np.float32)
    if all(k in out.columns for k in ["RH_1", "RH_2", "RH_8", "RH_9"]):
        new_cols["RH_vertical_gradient"] = (
            0.5 * (out["RH_8"] + out["RH_9"]) - 0.5 * (out["RH_1"] + out["RH_2"])
        ).astype(np.float32)
    if all(k in new_cols for k in ["Enthalpy_1", "Enthalpy_2", "Enthalpy_8", "Enthalpy_9"]):
        new_cols["Enthalpy_vertical_gradient"] = (
            0.5 * (new_cols["Enthalpy_8"] + new_cols["Enthalpy_9"])
            - 0.5 * (new_cols["Enthalpy_1"] + new_cols["Enthalpy_2"])
        ).astype(np.float32)

    # Convective cooling & atmosphere proxies
    if "Windspeed" in out.columns and "T_in_out_diff" in new_cols:
        new_cols["Wind_Thermal_Loss_Proxy"] = (
            out["Windspeed"] * np.maximum(0.0, new_cols["T_in_out_diff"])
        ).astype(np.float32)
        new_cols["Convective_Cooling_Proxy"] = (
            new_cols["T_in_out_diff"] * np.sqrt(np.maximum(0.0, out["Windspeed"]))
        ).astype(np.float32)

    if "Press_mm_hg" in out.columns and "T_out" in out.columns:
        new_cols["Atmospheric_Density_Proxy"] = (
            out["Press_mm_hg"] / (out["T_out"] + 273.15)
        ).astype(np.float32)

    features_df = pd.DataFrame(new_cols, index=out.index)
    return pd.concat([out, features_df], axis=1)


def calculate_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "date" in out.columns:
        date_str = out["date"].astype(str).str.strip()
        cleaned_dates = date_str.str.replace(
            r"^(\d{4}-\d{2}-\d{2})[T ]?(\d{2}:\d{2}:\d{2})$", r"\1 \2", regex=True
        )
        dates = pd.to_datetime(
            cleaned_dates, format="%Y-%m-%d %H:%M:%S", errors="coerce"
        )
        if dates.isna().any():
            dates = dates.fillna(pd.to_datetime(out["date"], errors="coerce"))

        hour = dates.dt.hour
        minute = dates.dt.minute
        dayofweek = dates.dt.dayofweek
        dayofyear = dates.dt.dayofyear
        month = dates.dt.month

        minute_of_day = hour * 60 + minute

        out["hour"] = hour.astype(np.int32)
        out["minute"] = minute.astype(np.int32)
        out["dayofweek"] = dayofweek.astype(np.int32)
        out["dayofyear"] = dayofyear.astype(np.int32)
        out["month"] = month.astype(np.int32)
        out["is_weekend"] = (dayofweek >= 5).astype(np.int32)

        out["sin_time_of_day"] = np.sin(2.0 * np.pi * minute_of_day / 1440.0).astype(
            np.float32
        )
        out["cos_time_of_day"] = np.cos(2.0 * np.pi * minute_of_day / 1440.0).astype(
            np.float32
        )
        out["sin_day_of_week"] = np.sin(2.0 * np.pi * dayofweek / 7.0).astype(
            np.float32
        )
        out["cos_day_of_week"] = np.cos(2.0 * np.pi * dayofweek / 7.0).astype(
            np.float32
        )
        out["sin_day_of_year"] = np.sin(2.0 * np.pi * dayofyear / 365.25).astype(
            np.float32
        )
        out["cos_day_of_year"] = np.cos(2.0 * np.pi * dayofyear / 365.25).astype(
            np.float32
        )

        out["is_night"] = ((hour >= 23) | (hour <= 6)).astype(np.float32)
        out["is_morning_peak"] = ((hour >= 7) & (hour <= 9)).astype(np.float32)
        out["is_evening_peak"] = ((hour >= 17) & (hour <= 22)).astype(np.float32)
        out["is_cooking_time"] = (
            ((hour >= 11) & (hour <= 13)) | ((hour >= 18) & (hour <= 20))
        ).astype(np.float32)

        # Solar elevation angle proxy (Stambruges, Belgium: lat ~ 50.5 deg N)
        lat_rad = np.radians(50.5)
        solar_declination = np.radians(
            23.45 * np.sin(2.0 * np.pi * (284 + dayofyear) / 365.25)
        )
        hour_angle = np.radians(15.0 * (minute_of_day / 60.0 - 12.0))
        sin_elev = np.sin(lat_rad) * np.sin(solar_declination) + np.cos(
            lat_rad
        ) * np.cos(solar_declination) * np.cos(hour_angle)

        out["solar_elevation_proxy"] = np.maximum(0.0, sin_elev).astype(np.float32)
        out["solar_sin_elev"] = sin_elev.astype(np.float32)

    if "lights" in out.columns:
        out["lights_binary"] = (out["lights"] > 0).astype(np.float32)
        out["lights_log"] = np.log1p(np.maximum(0.0, out["lights"])).astype(np.float32)
        if "T_indoor_mean" in out.columns:
            out["lights_per_indoor_temp"] = (
                out["lights"] / (out["T_indoor_mean"] + 1e-4)
            ).astype(np.float32)

    return out


# =========================================================================
# 2. PyTorch Tabular Deep Residual Neural Network (TabularResNet)
# =========================================================================
class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout_rate: float = 0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
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
        hidden_dim: int = 128,
        num_blocks: int = 3,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
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
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.input_layer(x)
        for block in self.res_blocks:
            out = block(out)
        out = self.head(out)
        return out.squeeze(-1)


def create_tabular_nn(
    input_dim: int,
    hidden_dim: int = 128,
    num_blocks: int = 3,
    dropout_rate: float = 0.1,
    lr: float = 2e-3,
    weight_decay: float = 1e-4,
    device: str = "cpu",
):
    model = TabularResNet(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_blocks=num_blocks,
        dropout_rate=dropout_rate,
    ).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=60, eta_min=1e-5
    )
    return model, criterion, optimizer, scheduler


# =========================================================================
# 3. Model Hyperparameters Factory
# =========================================================================
def get_lightgbm_params(seed: int = 42):
    return {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "learning_rate": 0.03,
        "num_leaves": 45,
        "max_depth": 7,
        "min_child_samples": 20,
        "subsample": 0.85,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": seed,
        "n_estimators": 2000,
        "n_jobs": -1,
        "verbose": -1,
    }


def get_xgboost_params(seed: int = 42):
    return {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "learning_rate": 0.03,
        "max_depth": 6,
        "min_child_weight": 3,
        "subsample": 0.80,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": seed,
        "n_estimators": 2000,
        "n_jobs": -1,
        "tree_method": "hist",
    }


def get_catboost_params(seed: int = 42):
    return {
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "learning_rate": 0.04,
        "depth": 6,
        "l2_leaf_reg": 4.0,
        "random_seed": seed,
        "iterations": 2000,
        "verbose": False,
        "thread_count": -1,
    }


def get_extratrees_params(seed: int = 42):
    return {
        "n_estimators": 350,
        "max_depth": 20,
        "min_samples_split": 4,
        "min_samples_leaf": 2,
        "max_features": 0.75,
        "n_jobs": -1,
        "random_state": seed,
    }


# =========================================================================
# 4. Pipeline Execution
# =========================================================================
def main():
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load raw data
    train_path = "./input/train.parquet"
    test_path = "./input/test.parquet"

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    # Feature Engineering
    train_feat = calculate_psychrometric_features(train_df)
    train_feat = calculate_temporal_features(train_feat)

    test_feat = calculate_psychrometric_features(test_df)
    test_feat = calculate_temporal_features(test_feat)

    # Missing value handling (fit strictly on train)
    features = [
        c
        for c in train_feat.columns
        if c not in ["date", "Appliances", "fold", "rv1", "rv2"]
    ]
    target_col = "Appliances"

    train_medians = train_feat[features].median()
    train_feat[features] = train_feat[features].fillna(train_medians)
    test_feat[features] = test_feat[features].fillna(train_medians)

    # 5-Fold Cross Validation Setup
    n_folds = 5
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    train_feat = train_feat.copy()
    train_feat["fold"] = -1
    for fold_idx, (_, val_idx) in enumerate(kf.split(train_feat)):
        train_feat.loc[val_idx, "fold"] = fold_idx

    n_train = len(train_feat)
    n_test = len(test_feat)
    y_train_full = train_feat[target_col].values.astype(np.float32)
    min_target = float(train_feat[target_col].min())

    oof_lgb = np.zeros(n_train, dtype=np.float32)
    oof_xgb = np.zeros(n_train, dtype=np.float32)
    oof_cat = np.zeros(n_train, dtype=np.float32)
    oof_et = np.zeros(n_train, dtype=np.float32)
    oof_nn = np.zeros(n_train, dtype=np.float32)

    test_lgb = np.zeros(n_test, dtype=np.float32)
    test_xgb = np.zeros(n_test, dtype=np.float32)
    test_cat = np.zeros(n_test, dtype=np.float32)
    test_et = np.zeros(n_test, dtype=np.float32)
    test_nn = np.zeros(n_test, dtype=np.float32)

    X_test_all = test_feat[features].values.astype(np.float32)

    for fold in range(n_folds):
        train_idx = train_feat["fold"] != fold
        val_idx = train_feat["fold"] == fold

        X_tr = train_feat.loc[train_idx, features].values.astype(np.float32)
        y_tr = train_feat.loc[train_idx, target_col].values.astype(np.float32)
        X_va = train_feat.loc[val_idx, features].values.astype(np.float32)
        y_va = train_feat.loc[val_idx, target_col].values.astype(np.float32)

        # LightGBM
        lgb_params = get_lightgbm_params(seed=42 + fold)
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
        oof_lgb[val_idx] = lgb_model.predict(X_va)
        test_lgb += lgb_model.predict(X_test_all) / n_folds

        # XGBoost
        xgb_params = get_xgboost_params(seed=42 + fold)
        xgb_params["early_stopping_rounds"] = 50
        xgb_model = xgb.XGBRegressor(**xgb_params)
        xgb_model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            verbose=False,
        )
        oof_xgb[val_idx] = xgb_model.predict(X_va)
        test_xgb += xgb_model.predict(X_test_all) / n_folds

        # CatBoost
        cat_params = get_catboost_params(seed=42 + fold)
        cat_model = CatBoostRegressor(**cat_params)
        cat_model.fit(
            X_tr,
            y_tr,
            eval_set=(X_va, y_va),
            early_stopping_rounds=50,
            verbose=False,
        )
        oof_cat[val_idx] = cat_model.predict(X_va)
        test_cat += cat_model.predict(X_test_all) / n_folds

        # ExtraTrees
        et_params = get_extratrees_params(seed=42 + fold)
        et_model = ExtraTreesRegressor(**et_params)
        et_model.fit(X_tr, y_tr)
        oof_et[val_idx] = et_model.predict(X_va)
        test_et += et_model.predict(X_test_all) / n_folds

        # PyTorch Tabular ResNet
        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr)
        X_va_sc = scaler.transform(X_va)
        X_te_sc = scaler.transform(X_test_all)

        train_dataset = TensorDataset(
            torch.tensor(X_tr_sc, dtype=torch.float32),
            torch.tensor(y_tr, dtype=torch.float32),
        )
        val_dataset = TensorDataset(
            torch.tensor(X_va_sc, dtype=torch.float32),
            torch.tensor(y_va, dtype=torch.float32),
        )

        train_loader = DataLoader(
            train_dataset, batch_size=256, shuffle=True, drop_last=False
        )
        val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)

        nn_model, criterion, optimizer, scheduler = create_tabular_nn(
            input_dim=len(features),
            hidden_dim=128,
            num_blocks=3,
            dropout_rate=0.1,
            lr=2e-3,
            device=device,
        )

        best_val_loss = float("inf")
        best_nn_weights = None
        patience, patience_counter = 15, 0

        for _ in range(60):
            nn_model.train()
            for bx, by in train_loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                pred = nn_model(bx)
                loss = criterion(pred, by)
                loss.backward()
                optimizer.step()

            scheduler.step()

            nn_model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for bx, by in val_loader:
                    bx, by = bx.to(device), by.to(device)
                    pred = nn_model(bx)
                    val_loss += criterion(pred, by).item() * len(by)
            val_loss /= len(val_dataset)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_nn_weights = {
                    k: v.cpu().clone() for k, v in nn_model.state_dict().items()
                }
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        nn_model.load_state_dict({k: v.to(device) for k, v in best_nn_weights.items()})
        nn_model.eval()
        with torch.no_grad():
            va_t = torch.tensor(X_va_sc, dtype=torch.float32).to(device)
            te_t = torch.tensor(X_te_sc, dtype=torch.float32).to(device)
            oof_nn[val_idx] = nn_model(va_t).cpu().numpy()
            test_nn += nn_model(te_t).cpu().numpy() / n_folds

    # Ensemble optimization via SLSQP
    oof_matrix = np.column_stack([oof_lgb, oof_xgb, oof_cat, oof_et, oof_nn])
    test_matrix = np.column_stack([test_lgb, test_xgb, test_cat, test_et, test_nn])

    def loss_func(weights):
        pred = oof_matrix @ weights
        pred_clipped = np.clip(pred, a_min=min_target, a_max=None)
        return np.sqrt(mean_squared_error(y_train_full, pred_clipped))

    initial_weights = np.ones(5) / 5.0
    bounds = [(0.0, 1.0) for _ in range(5)]
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

    opt_res = minimize(
        loss_func,
        initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    best_weights = (
        opt_res.x if opt_res.success else np.array([0.3, 0.25, 0.2, 0.15, 0.1])
    )
    best_weights = best_weights / np.sum(best_weights)

    # Compute final validation score
    oof_final = np.clip(oof_matrix @ best_weights, a_min=min_target, a_max=None)
    final_rmse = np.sqrt(mean_squared_error(y_train_full, oof_final))

    # Generate submission file
    test_preds = np.clip(test_matrix @ best_weights, a_min=min_target, a_max=None)
    submission_df = pd.DataFrame(
        {"id": np.arange(len(test_preds), dtype=np.int64), "Appliances": test_preds}
    )

    os.makedirs("./submission", exist_ok=True)
    submission_df.to_csv("./submission/submission.csv", index=False)

    print(f"Final Validation Score: {final_rmse}")


if __name__ == "__main__":
    main()
