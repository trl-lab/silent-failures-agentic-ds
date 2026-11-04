import os
import random
from typing import Tuple
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from catboost import CatBoostRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb


# =========================================================================
# 0. Reproducibility
# =========================================================================
def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =========================================================================
# 1. Data Processing and Feature Engineering
# =========================================================================
def parse_uci_datetime(df: pd.DataFrame) -> pd.Series:
    """Parse Date and Time columns into a unified pandas datetime series."""
    date_series = df["Date"].astype(str).str.strip()
    time_series = df["Time"].astype(str).str.strip()
    time_series = time_series.str.replace(".", ":", regex=False)
    datetime_str = date_series + " " + time_series
    return pd.to_datetime(datetime_str, errors="coerce")


def clean_sentinel_values(df: pd.DataFrame) -> pd.DataFrame:
    """Replace UCI missing value sentinels (-200) with NaN for numeric features."""
    df_clean = df.copy()
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df_clean.loc[df_clean[col] <= -100, col] = np.nan
    return df_clean


def engineer_air_quality_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer domain-specific physical, chemical, and temporal features."""
    df_feat = df.copy()

    # 1. Datetime feature extraction
    dt = parse_uci_datetime(df_feat)
    df_feat["hour"] = dt.dt.hour.fillna(-1).astype(int)
    df_feat["dayofweek"] = dt.dt.dayofweek.fillna(-1).astype(int)
    df_feat["month"] = dt.dt.month.fillna(-1).astype(int)
    df_feat["is_weekend"] = (df_feat["dayofweek"] >= 5).astype(int)

    # Cyclical encodings for periodic temporal behavior
    df_feat["hour_sin"] = np.sin(2 * np.pi * df_feat["hour"] / 24.0)
    df_feat["hour_cos"] = np.cos(2 * np.pi * df_feat["hour"] / 24.0)
    df_feat["month_sin"] = np.sin(2 * np.pi * df_feat["month"] / 12.0)
    df_feat["month_cos"] = np.cos(2 * np.pi * df_feat["month"] / 12.0)

    # Rush hour indicators (typical urban traffic emission peaks)
    df_feat["is_morning_rush"] = (
        (df_feat["hour"] >= 7) & (df_feat["hour"] <= 9)
    ).astype(int)
    df_feat["is_evening_rush"] = (
        (df_feat["hour"] >= 17) & (df_feat["hour"] <= 20)
    ).astype(int)

    # 2. Missingness indicator flags for reference analyzers
    ref_cols = ["NMHC(GT)", "C6H6(GT)", "NOx(GT)", "NO2(GT)"]
    for col in ref_cols:
        if col in df_feat.columns:
            df_feat[f"{col}_isna"] = df_feat[col].isna().astype(float)

    # 3. Atmospheric & Physical Thermodynamic / Arrhenius Features
    if "T" in df_feat.columns:
        t_safe = df_feat["T"].clip(-20, 60)
        t_kelvin = t_safe + 273.15
        df_feat["arrhenius_inv_T"] = 1.0 / t_kelvin

        if "RH" in df_feat.columns:
            rh_safe = df_feat["RH"].clip(0, 100)
            e_sat = 6.1078 * (10.0 ** ((7.5 * t_safe) / (237.3 + t_safe)))
            actual_vapor_pressure = (rh_safe / 100.0) * e_sat
            df_feat["calc_vapor_pressure"] = actual_vapor_pressure

            # Dew point approximation
            a, b = 17.27, 237.7
            alpha = ((a * t_safe) / (b + t_safe)) + np.log((rh_safe + 1e-4) / 100.0)
            df_feat["dew_point"] = (b * alpha) / (a - alpha)
            df_feat["rh_to_t_ratio"] = rh_safe / t_kelvin

        if "AH" in df_feat.columns:
            df_feat["ah_to_t_ratio"] = df_feat["AH"] / t_kelvin

    # 4. Sensor Cross-Interactions, Non-linearities, and Physical MOS Matrices
    s1 = "PT08.S1(CO)"
    s2 = "PT08.S2(NMHC)"
    s3 = "PT08.S3(NOx)"
    s4 = "PT08.S4(NO2)"
    s5 = "PT08.S5(O3)"
    sensor_cols = [s1, s2, s3, s4, s5]

    for col in sensor_cols:
        if col in df_feat.columns:
            if "T" in df_feat.columns:
                df_feat[f"{col}_x_T"] = df_feat[col] * df_feat["T"]
                df_feat[f"{col}_arrhenius"] = df_feat[col] / (df_feat["T"].clip(-20, 60) + 273.15)
            if "AH" in df_feat.columns:
                df_feat[f"{col}_x_AH"] = df_feat[col] * df_feat["AH"]
            if "RH" in df_feat.columns:
                df_feat[f"{col}_x_RH"] = df_feat[col] * df_feat["RH"]
            df_feat[f"log_{col}"] = np.log1p(df_feat[col].clip(lower=0))
            df_feat[f"sqrt_{col}"] = np.sqrt(df_feat[col].clip(lower=0))

    # Complete cross interaction products and oxidizing/reducing ratios
    for i in range(len(sensor_cols)):
        for j in range(i + 1, len(sensor_cols)):
            c1, c2 = sensor_cols[i], sensor_cols[j]
            s_name1 = c1.split(".")[1].split("(")[0]
            s_name2 = c2.split(".")[1].split("(")[0]
            if c1 in df_feat.columns and c2 in df_feat.columns:
                df_feat[f"prod_{s_name1}_{s_name2}"] = df_feat[c1] * df_feat[c2]
                df_feat[f"ratio_{s_name1}_{s_name2}"] = df_feat[c1] / (df_feat[c2] + 1.0)
                df_feat[f"ratio_{s_name2}_{s_name1}"] = df_feat[c2] / (df_feat[c1] + 1.0)

    # 5. Combustion Stoichiometric Ratios & Reference Interactions
    if "C6H6(GT)" in df_feat.columns:
        df_feat["log_C6H6"] = np.log1p(df_feat["C6H6(GT)"].clip(lower=0))
        if s1 in df_feat.columns:
            df_feat["C6H6_x_S1"] = df_feat["C6H6(GT)"] * df_feat[s1]
        if s2 in df_feat.columns:
            df_feat["C6H6_x_S2"] = df_feat["C6H6(GT)"] * df_feat[s2]
    if "NOx(GT)" in df_feat.columns and "NO2(GT)" in df_feat.columns:
        df_feat["NO_proxy"] = (df_feat["NOx(GT)"] - df_feat["NO2(GT)"]).clip(lower=0)
        df_feat["NOx_diff_ratio"] = (df_feat["NOx(GT)"] - df_feat["NO2(GT)"]) / (df_feat["NOx(GT)"] + 1.0)
        df_feat["NO2_to_NOx_ratio"] = df_feat["NO2(GT)"] / (df_feat["NOx(GT)"] + 1.0)
    if "C6H6(GT)" in df_feat.columns and "NOx(GT)" in df_feat.columns:
        df_feat["C6H6_to_NOx"] = df_feat["C6H6(GT)"] / (df_feat["NOx(GT)"] + 1.0)

    df_feat = df_feat.drop(columns=["Date", "Time"], errors="ignore")
    return df_feat


# =========================================================================
# 2. PyTorch Tabular Neural Network Architecture
# =========================================================================
class TabularResidualBlock(nn.Module):
    """Residual block with LayerNorm, GELU, and Dropout."""

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
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.block(x))


class TabularResNet(nn.Module):
    """Deep Tabular Residual Network for non-linear gas sensor calibration."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_blocks: int = 3,
        dropout_rate: float = 0.15,
        output_dim: int = 1,
    ):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )

        self.res_blocks = nn.ModuleList(
            [
                TabularResidualBlock(hidden_dim, dropout_rate=dropout_rate)
                for _ in range(num_blocks)
            ]
        )

        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate / 2.0),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.input_layer(x)
        for block in self.res_blocks:
            feat = block(feat)
        out = self.head(feat)
        return out.squeeze(-1)


def build_optimizer_and_scheduler(
    model: nn.Module,
    lr: float = 1.5e-3,
    weight_decay: float = 1e-4,
    epochs: int = 70,
    steps_per_epoch: int = 10,
) -> Tuple[optim.Optimizer, optim.lr_scheduler._LRScheduler]:
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs * steps_per_epoch, eta_min=lr * 1e-2
    )
    return optimizer, scheduler


# =========================================================================
# 3. Model Factories
# =========================================================================
def get_lightgbm_model(random_state: int = 42) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        objective="regression",
        metric="rmse",
        n_estimators=1500,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=6,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=1.0,
        min_child_samples=20,
        random_state=random_state,
        n_jobs=-1,
        verbosity=-1,
    )


def get_xgboost_model(random_state: int = 42) -> xgb.XGBRegressor:
    return xgb.XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        n_estimators=1500,
        learning_rate=0.03,
        max_depth=5,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.2,
        reg_lambda=2.0,
        tree_method="hist",
        early_stopping_rounds=50,
        random_state=random_state,
        n_jobs=-1,
    )


def get_catboost_model(random_state: int = 42) -> CatBoostRegressor:
    return CatBoostRegressor(
        iterations=1500,
        learning_rate=0.03,
        depth=6,
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=random_state,
        verbose=False,
        early_stopping_rounds=50,
        thread_count=-1,
    )


# =========================================================================
# 4. Pipeline Execution
# =========================================================================
def main():
    seed_everything(42)

    # 1. Load raw datasets
    raw_train = pd.read_parquet("./input/train.parquet")
    raw_test = pd.read_parquet("./input/test.parquet")
    raw_test["id"] = np.arange(len(raw_test))

    target_col = "CO(GT)"
    valid_target_mask = (raw_train[target_col].notna()) & (raw_train[target_col] > -100)
    raw_train = raw_train.loc[valid_target_mask].reset_index(drop=True)

    # 2. Clean sentinels and engineer features
    train_cleaned = clean_sentinel_values(raw_train)
    test_cleaned = clean_sentinel_values(raw_test)

    train_df = engineer_air_quality_features(train_cleaned)
    test_df = engineer_air_quality_features(test_cleaned)

    # 3. Establish 5-Fold Cross Validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    train_df["fold"] = -1
    for fold, (_, val_idx) in enumerate(kf.split(train_df, train_df[target_col])):
        train_df.loc[val_idx, "fold"] = fold

    feature_cols = [c for c in train_df.columns if c not in [target_col, "fold", "id"]]
    n_samples = len(train_df)
    n_test = len(test_df)
    n_folds = 5

    model_names = ["LightGBM", "XGBoost", "CatBoost", "TabularResNet"]
    oof_preds = {name: np.zeros(n_samples, dtype=np.float32) for name in model_names}
    test_preds = {name: np.zeros(n_test, dtype=np.float32) for name in model_names}
    y_true = train_df[target_col].values.astype(np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 4. Model Training & Evaluation Loop
    for fold in range(n_folds):
        train_mask = train_df["fold"] != fold
        val_mask = train_df["fold"] == fold

        X_train = train_df.loc[train_mask, feature_cols].copy()
        y_tr = train_df.loc[train_mask, target_col].values.astype(np.float32)

        X_val = train_df.loc[val_mask, feature_cols].copy()
        y_v = train_df.loc[val_mask, target_col].values.astype(np.float32)

        X_test = test_df[feature_cols].copy()

        # Fit preprocessing exclusively on fold training data
        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()

        X_tr_imp = imputer.fit_transform(X_train)
        X_val_imp = imputer.transform(X_val)
        X_te_imp = imputer.transform(X_test)

        X_tr_scaled = scaler.fit_transform(X_tr_imp)
        X_val_scaled = scaler.transform(X_val_imp)
        X_te_scaled = scaler.transform(X_te_imp)

        # LightGBM
        lgb_model = get_lightgbm_model(random_state=42 + fold)
        lgb_model.fit(
            X_train,
            y_tr,
            eval_set=[(X_val, y_v)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )
        lgb_val_pred = lgb_model.predict(X_val)
        lgb_test_pred = lgb_model.predict(X_test)
        oof_preds["LightGBM"][val_mask] = lgb_val_pred
        test_preds["LightGBM"] += lgb_test_pred / n_folds

        # XGBoost
        xgb_model = get_xgboost_model(random_state=42 + fold)
        xgb_model.fit(X_train, y_tr, eval_set=[(X_val, y_v)], verbose=False)
        xgb_val_pred = xgb_model.predict(X_val)
        xgb_test_pred = xgb_model.predict(X_test)
        oof_preds["XGBoost"][val_mask] = xgb_val_pred
        test_preds["XGBoost"] += xgb_test_pred / n_folds

        # CatBoost
        cb_model = get_catboost_model(random_state=42 + fold)
        cb_model.fit(
            X_train,
            y_tr,
            eval_set=(X_val, y_v),
            verbose=False,
        )
        cb_val_pred = cb_model.predict(X_val)
        cb_test_pred = cb_model.predict(X_test)
        oof_preds["CatBoost"][val_mask] = cb_val_pred
        test_preds["CatBoost"] += cb_test_pred / n_folds

        # Tabular Residual Neural Network
        torch.manual_seed(42 + fold)
        nn_model = TabularResNet(
            input_dim=len(feature_cols),
            hidden_dim=128,
            num_blocks=3,
            dropout_rate=0.15,
        ).to(device)

        criterion = nn.MSELoss()
        steps_per_epoch = (len(X_tr_scaled) // 64) + 1
        optimizer, scheduler = build_optimizer_and_scheduler(
            nn_model,
            lr=1.5e-3,
            weight_decay=1e-4,
            epochs=70,
            steps_per_epoch=steps_per_epoch,
        )

        train_ds = TensorDataset(
            torch.tensor(X_tr_scaled, dtype=torch.float32),
            torch.tensor(y_tr, dtype=torch.float32),
        )
        val_t_x = torch.tensor(X_val_scaled, dtype=torch.float32).to(device)
        test_t_x = torch.tensor(X_te_scaled, dtype=torch.float32).to(device)

        train_loader = DataLoader(
            train_ds, batch_size=64, shuffle=True, drop_last=False
        )

        best_val_loss = float("inf")
        best_state = None

        for epoch in range(70):
            nn_model.train()
            for bx, by in train_loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                out = nn_model(bx)
                loss = criterion(out, by)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(nn_model.parameters(), max_norm=2.0)
                optimizer.step()
                scheduler.step()

            nn_model.eval()
            with torch.no_grad():
                val_out = nn_model(val_t_x).cpu().numpy()
                epoch_val_rmse = root_mean_squared_error(y_v, val_out)
                if epoch_val_rmse < best_val_loss:
                    best_val_loss = epoch_val_rmse
                    best_state = {
                        k: v.cpu().clone() for k, v in nn_model.state_dict().items()
                    }

        if best_state is not None:
            nn_model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        nn_model.eval()
        with torch.no_grad():
            nn_val_pred = nn_model(val_t_x).cpu().numpy()
            nn_test_pred = nn_model(test_t_x).cpu().numpy()

        oof_preds["TabularResNet"][val_mask] = nn_val_pred
        test_preds["TabularResNet"] += nn_test_pred / n_folds

        print(
            f"Fold {fold} | LGBM: {root_mean_squared_error(y_v, lgb_val_pred):.4f} | "
            f"XGB: {root_mean_squared_error(y_v, xgb_val_pred):.4f} | "
            f"CB: {root_mean_squared_error(y_v, cb_val_pred):.4f} | "
            f"NN: {root_mean_squared_error(y_v, nn_val_pred):.4f}"
        )

    # 5. Out-Of-Fold Ensembling via SLSQP Optimization
    oof_matrix = np.column_stack([oof_preds[name] for name in model_names])
    test_matrix = np.column_stack([test_preds[name] for name in model_names])

    def ensemble_objective(weights):
        pred = oof_matrix @ weights
        return root_mean_squared_error(y_true, np.clip(pred, 0, None))

    n_models = len(model_names)
    initial_weights = np.ones(n_models) / n_models
    bounds = [(0.0, 1.0) for _ in range(n_models)]
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

    opt_result = minimize(
        ensemble_objective,
        initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    optimal_weights = opt_result.x
    print(
        f"Optimal Ensemble Weights: {dict(zip(model_names, np.round(optimal_weights, 4)))}"
    )

    # 6. Final Evaluation and Submission
    final_oof_predictions = np.clip(oof_matrix @ optimal_weights, 0, None)
    final_score = root_mean_squared_error(y_true, final_oof_predictions)

    final_test_predictions = np.clip(test_matrix @ optimal_weights, 0, None)

    os.makedirs("./submission", exist_ok=True)
    submission = pd.DataFrame(
        {
            "id": test_df["id"].values.astype(int),
            "CO(GT)": final_test_predictions,
        }
    )
    submission.to_csv("./submission/submission.csv", index=False)

    print(f"Final Validation Score: {final_score}")


if __name__ == "__main__":
    main()
