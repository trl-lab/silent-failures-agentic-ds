import os
import random
from typing import Any, Dict, Tuple
from catboost import CatBoostRegressor
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb


# =========================================================================
# 0. Reproducibility Setup
# =========================================================================
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


seed_everything(42)


# =========================================================================
# 1. Feature Engineering Pipeline
# =========================================================================
def create_feature_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Constructs cyclical, bioclimatic, and temporal interaction features

    for the Bike Sharing dataset without leaking validation/test information.
    """
    df = df.copy()

    # 1. Parse datetime features
    if "dteday" in df.columns:
        dteday = pd.to_datetime(df["dteday"])
        df["day"] = dteday.dt.day
        df["dayofweek"] = dteday.dt.dayofweek
        df["dayofyear"] = dteday.dt.dayofyear
        df["weekofyear"] = dteday.dt.isocalendar().week.astype(int)
        df["quarter"] = dteday.dt.quarter
        df["is_month_start"] = dteday.dt.is_month_start.astype(int)
        df["is_month_end"] = dteday.dt.is_month_end.astype(int)

        # Cumulative timeline feature (days since inception)
        min_date = pd.to_datetime("2011-01-01")
        df["days_since_start"] = (dteday - min_date).dt.days

    # 2. Cyclical Encodings (Harmonic Sine & Cosine)
    if "hr" in df.columns:
        df["sin_hr"] = np.sin(2 * np.pi * df["hr"] / 24.0)
        df["cos_hr"] = np.cos(2 * np.pi * df["hr"] / 24.0)

    if "mnth" in df.columns:
        df["sin_mnth"] = np.sin(2 * np.pi * (df["mnth"] - 1) / 12.0)
        df["cos_mnth"] = np.cos(2 * np.pi * (df["mnth"] - 1) / 12.0)

    weekday_col = "weekday" if "weekday" in df.columns else "dayofweek"
    if weekday_col in df.columns:
        df["sin_weekday"] = np.sin(2 * np.pi * df[weekday_col] / 7.0)
        df["cos_weekday"] = np.cos(2 * np.pi * df[weekday_col] / 7.0)

    if "dayofyear" in df.columns:
        df["sin_doy"] = np.sin(2 * np.pi * df["dayofyear"] / 365.25)
        df["cos_doy"] = np.cos(2 * np.pi * df["dayofyear"] / 365.25)

    # 3. Commute and Diurnal Regimes
    if "hr" in df.columns and "workingday" in df.columns:
        df["is_morning_rush"] = (
            (df["hr"].isin([7, 8, 9])) & (df["workingday"] == 1)
        ).astype(int)
        df["is_evening_rush"] = (
            (df["hr"].isin([16, 17, 18, 19])) & (df["workingday"] == 1)
        ).astype(int)
        df["is_rush_hour"] = df["is_morning_rush"] | df["is_evening_rush"]
        df["is_night"] = (df["hr"].isin([0, 1, 2, 3, 4, 5])).astype(int)
        df["is_midday"] = (df["hr"].isin([10, 11, 12, 13, 14, 15])).astype(int)
        df["hr_workingday_interaction"] = df["hr"] + df["workingday"] * 24

    if "holiday" in df.columns and "workingday" in df.columns:
        df["is_weekend"] = ((df["workingday"] == 0) & (df["holiday"] == 0)).astype(int)

    # 4. Bioclimatic and Meteorological Features
    if "temp" in df.columns and "atemp" in df.columns:
        df["temp_atemp_diff"] = df["atemp"] - df["temp"]
        df["temp_atemp_ratio"] = df["atemp"] / (df["temp"] + 1e-4)

    if "temp" in df.columns and "hum" in df.columns:
        df["heat_discomfort_index"] = df["temp"] * 40.0 + (df["hum"] * 100.0) * 0.1
        df["temp_hum_interaction"] = df["temp"] * df["hum"]

    if "temp" in df.columns and "windspeed" in df.columns:
        df["wind_chill_proxy"] = df["temp"] - (df["windspeed"] * 0.2)
        df["temp_wind_interaction"] = df["temp"] * df["windspeed"]

    if "hum" in df.columns and "windspeed" in df.columns:
        df["hum_wind_interaction"] = df["hum"] * df["windspeed"]

    # Non-linear quadratic terms
    if "temp" in df.columns:
        df["temp_sq"] = df["temp"] ** 2
    if "atemp" in df.columns:
        df["atemp_sq"] = df["atemp"] ** 2
    if "hum" in df.columns:
        df["hum_sq"] = df["hum"] ** 2
    if "windspeed" in df.columns:
        df["windspeed_sq"] = df["windspeed"] ** 2

    # Weather comfort indicators
    if "temp" in df.columns and "hum" in df.columns and "windspeed" in df.columns:
        df["is_ideal_weather"] = (
            (df["temp"].between(0.35, 0.75))
            & (df["hum"] < 0.70)
            & (df["windspeed"] < 0.35)
            & (df["weathersit"].isin([1, 2]) if "weathersit" in df.columns else True)
        ).astype(int)

    if "weathersit" in df.columns:
        df["is_severe_weather"] = (df["weathersit"] >= 3).astype(int)

    return df


# =========================================================================
# 2. PyTorch Deep Tabular Residual MLP Architecture
# =========================================================================
class TabularResBlock(nn.Module):

    def __init__(self, dim: int, dropout_rate: float = 0.15):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim, dim),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class BikeDemandResMLP(nn.Module):

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_blocks: int = 3,
        dropout_rate: float = 0.15,
    ):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
        )

        self.res_blocks = nn.ModuleList(
            [
                TabularResBlock(hidden_dim, dropout_rate=dropout_rate)
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
        h = self.input_layer(x)
        for block in self.res_blocks:
            h = block(h)
        out = self.head(h)
        return out.squeeze(-1)


class LogSpaceMSELoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return self.mse(y_pred, y_true)


def build_pytorch_optimizer_and_scheduler(
    model: nn.Module,
    lr: float = 2e-3,
    weight_decay: float = 1e-4,
    num_epochs: int = 60,
    eta_min: float = 1e-6,
) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=eta_min)
    return optimizer, scheduler


# =========================================================================
# 3. Gradient Boosted Decision Tree Model Factories
# =========================================================================
def get_lightgbm_regressor(
    random_state: int = 42, extra_params: Dict[str, Any] = None
) -> lgb.LGBMRegressor:
    params = {
        "n_estimators": 2000,
        "learning_rate": 0.03,
        "num_leaves": 45,
        "max_depth": 7,
        "min_child_samples": 20,
        "subsample": 0.85,
        "subsample_freq": 1,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "objective": "regression",
        "random_state": random_state,
        "n_jobs": -1,
        "verbose": -1,
    }
    if extra_params:
        params.update(extra_params)
    return lgb.LGBMRegressor(**params)


def get_xgboost_regressor(
    random_state: int = 42, extra_params: Dict[str, Any] = None
) -> xgb.XGBRegressor:
    params = {
        "n_estimators": 2000,
        "learning_rate": 0.03,
        "max_depth": 6,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "objective": "reg:squarederror",
        "random_state": random_state,
        "n_jobs": -1,
        "early_stopping_rounds": 50,
        "tree_method": "hist",
    }
    if extra_params:
        params.update(extra_params)
    return xgb.XGBRegressor(**params)


def get_catboost_regressor(
    random_state: int = 42, extra_params: Dict[str, Any] = None
) -> CatBoostRegressor:
    params = {
        "iterations": 2000,
        "learning_rate": 0.04,
        "depth": 6,
        "l2_leaf_reg": 3.0,
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "random_seed": random_state,
        "verbose": False,
        "early_stopping_rounds": 50,
    }
    if extra_params:
        params.update(extra_params)
    return CatBoostRegressor(**params)


# =========================================================================
# 4. Main Training and Inference Pipeline
# =========================================================================
def main():
    os.makedirs("./submission", exist_ok=True)
    os.makedirs("./working", exist_ok=True)

    # 1. Load Data
    train_raw = pd.read_parquet("./input/train.parquet")
    test_raw = pd.read_parquet("./input/test.parquet")

    # 2. Extract Features
    train_df = create_feature_pipeline(train_raw)
    test_df = create_feature_pipeline(test_raw)

    target_col = "cnt"
    id_cols = ["instant", "dteday"]
    leakage_cols = ["casual", "registered"]
    exclude_cols = set(id_cols + [target_col] + leakage_cols)

    feature_cols = [c for c in train_df.columns if c not in exclude_cols]

    # Convert all feature columns to float32 and impute if necessary
    for col in feature_cols:
        col_median = train_df[col].median()
        train_df[col] = train_df[col].fillna(col_median).astype(np.float32)
        test_df[col] = test_df[col].fillna(col_median).astype(np.float32)

    X_train_full = train_df[feature_cols].copy()
    X_test_full = test_df[feature_cols].copy()
    y_train_full = train_df[target_col].values.astype(np.float32)
    y_train_log = np.log1p(y_train_full)

    n_train = len(train_df)
    n_test = len(test_df)
    n_folds = 5

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    model_names = ["LightGBM", "XGBoost", "CatBoost", "ResMLP"]
    n_models = len(model_names)
    oof_predictions = {
        name: np.zeros(n_train, dtype=np.float32) for name in model_names
    }
    test_predictions = {
        name: np.zeros(n_test, dtype=np.float32) for name in model_names
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_te_np = X_test_full.to_numpy(dtype=np.float32)

    for fold, (train_idx, val_idx) in enumerate(kf.split(train_df)):
        X_tr_np = X_train_full.iloc[train_idx].to_numpy(dtype=np.float32)
        y_tr_log = y_train_log[train_idx]
        X_va_np = X_train_full.iloc[val_idx].to_numpy(dtype=np.float32)
        y_va_log = y_train_log[val_idx]

        # ----------------- LightGBM -----------------
        lgb_model = get_lightgbm_regressor(random_state=42 + fold)
        lgb_model.fit(
            X_tr_np,
            y_tr_log,
            eval_set=[(X_va_np, y_va_log)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )
        val_pred_lgb = np.clip(
            np.expm1(lgb_model.predict(X_va_np)), a_min=0, a_max=None
        )
        te_pred_lgb = np.clip(np.expm1(lgb_model.predict(X_te_np)), a_min=0, a_max=None)
        oof_predictions["LightGBM"][val_idx] = val_pred_lgb
        test_predictions["LightGBM"] += te_pred_lgb / n_folds

        # ----------------- XGBoost -----------------
        xgb_model = get_xgboost_regressor(random_state=42 + fold)
        xgb_model.fit(X_tr_np, y_tr_log, eval_set=[(X_va_np, y_va_log)], verbose=False)
        val_pred_xgb = np.clip(
            np.expm1(xgb_model.predict(X_va_np)), a_min=0, a_max=None
        )
        te_pred_xgb = np.clip(np.expm1(xgb_model.predict(X_te_np)), a_min=0, a_max=None)
        oof_predictions["XGBoost"][val_idx] = val_pred_xgb
        test_predictions["XGBoost"] += te_pred_xgb / n_folds

        # ----------------- CatBoost -----------------
        cat_model = get_catboost_regressor(random_state=42 + fold)
        cat_model.fit(X_tr_np, y_tr_log, eval_set=(X_va_np, y_va_log), verbose=False)
        val_pred_cat = np.clip(
            np.expm1(cat_model.predict(X_va_np)), a_min=0, a_max=None
        )
        te_pred_cat = np.clip(np.expm1(cat_model.predict(X_te_np)), a_min=0, a_max=None)
        oof_predictions["CatBoost"][val_idx] = val_pred_cat
        test_predictions["CatBoost"] += te_pred_cat / n_folds

        # ----------------- Deep Res-MLP -----------------
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr_np)
        X_va_scaled = scaler.transform(X_va_np)
        X_te_scaled = scaler.transform(X_te_np)

        train_dataset = TensorDataset(
            torch.tensor(X_tr_scaled, dtype=torch.float32),
            torch.tensor(y_tr_log, dtype=torch.float32),
        )
        train_loader = DataLoader(
            train_dataset, batch_size=128, shuffle=True, drop_last=False
        )
        val_tensor_x = torch.tensor(X_va_scaled, dtype=torch.float32).to(device)
        val_tensor_y = torch.tensor(y_va_log, dtype=torch.float32).to(device)
        test_tensor = torch.tensor(X_te_scaled, dtype=torch.float32).to(device)

        mlp_model = BikeDemandResMLP(
            input_dim=X_tr_scaled.shape[1],
            hidden_dim=128,
            num_blocks=3,
            dropout_rate=0.15,
        ).to(device)
        criterion = LogSpaceMSELoss()
        optimizer, scheduler = build_pytorch_optimizer_and_scheduler(
            mlp_model, lr=2e-3, weight_decay=1e-4, num_epochs=60
        )

        best_val_loss = float("inf")
        best_weights = None
        patience = 12
        patience_counter = 0

        for epoch in range(60):
            mlp_model.train()
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                preds = mlp_model(batch_x)
                loss = criterion(preds, batch_y)
                loss.backward()
                optimizer.step()
            scheduler.step()

            mlp_model.eval()
            with torch.no_grad():
                val_preds_epoch = mlp_model(val_tensor_x)
                val_loss = criterion(val_preds_epoch, val_tensor_y).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights = {
                    k: v.cpu().clone() for k, v in mlp_model.state_dict().items()
                }
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        mlp_model.load_state_dict(best_weights)
        mlp_model.eval()
        with torch.no_grad():
            val_pred_mlp_log = mlp_model(val_tensor_x).cpu().numpy()
            te_pred_mlp_log = mlp_model(test_tensor).cpu().numpy()

        val_pred_mlp = np.clip(np.expm1(val_pred_mlp_log), a_min=0, a_max=None)
        te_pred_mlp = np.clip(np.expm1(te_pred_mlp_log), a_min=0, a_max=None)
        oof_predictions["ResMLP"][val_idx] = val_pred_mlp
        test_predictions["ResMLP"] += te_pred_mlp / n_folds

    # 5. Out-of-Fold Evaluation & Constrained SLSQP Optimization
    oof_matrix = np.column_stack([oof_predictions[name] for name in model_names])
    test_matrix = np.column_stack([test_predictions[name] for name in model_names])

    def ensemble_objective(weights):
        w = np.array(weights)
        w = w / np.sum(w)
        pred = np.dot(oof_matrix, w)
        return root_mean_squared_error(y_train_full, pred)

    init_weights = np.ones(n_models) / n_models
    bounds = [(0.0, 1.0) for _ in range(n_models)]
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    opt_res = minimize(
        ensemble_objective,
        init_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    opt_weights = opt_res.x / np.sum(opt_res.x)

    final_oof_pred = np.dot(oof_matrix, opt_weights)
    final_test_pred = np.dot(test_matrix, opt_weights)
    final_test_pred = np.clip(final_test_pred, a_min=0.0, a_max=None)

    # 6. Save Submission File
    submission_df = pd.DataFrame(
        {"id": np.arange(len(final_test_pred)), "cnt": final_test_pred}
    )
    submission_path = "./submission/submission.csv"
    submission_df.to_csv(submission_path, index=False)

    # 7. Final Validation Score Output
    final_validation_score = root_mean_squared_error(y_train_full, final_oof_pred)
    print(f"Final Validation Score: {final_validation_score}")


if __name__ == "__main__":
    main()
