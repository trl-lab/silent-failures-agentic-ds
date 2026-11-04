import copy
import os
import random
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from catboost import CatBoostRegressor
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import lightgbm as lgb
import xgboost as xgb


# ============================================================================
# 0. Reproducibility & Environment Setup
# ============================================================================
def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


seed_everything(42)
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================================
# 1. Data Loading & Feature Engineering
# ============================================================================
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "heating_load"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct physically meaningful building geometry and thermodynamic features.
    Deterministic per-sample calculations ensuring zero train-test leakage.
    """
    df = df.copy()

    # 1. Estimated Volume and Dimensional Ratios
    df["est_volume"] = df["roof_area"] * df["overall_height"]
    df["surface_to_volume"] = df["surface_area"] / np.maximum(df["est_volume"], 1e-6)

    # 2. Envelope Component Ratios & Perimeter Proxies
    df["wall_to_surface_ratio"] = df["wall_area"] / np.maximum(df["surface_area"], 1e-6)
    df["roof_to_surface_ratio"] = df["roof_area"] / np.maximum(df["surface_area"], 1e-6)
    df["wall_to_roof_ratio"] = df["wall_area"] / np.maximum(df["roof_area"], 1e-6)
    df["building_perimeter_exact"] = df["wall_area"] / np.maximum(df["overall_height"], 1e-6)

    # 3. Shape, Sphericity & Slenderness
    footprint_length = np.sqrt(np.maximum(df["roof_area"], 1e-6))
    df["building_aspect_ratio"] = df["overall_height"] / footprint_length
    df["footprint_perimeter_proxy"] = 4.0 * footprint_length
    df["wall_height_perimeter_ratio"] = df["wall_area"] / np.maximum(
        df["overall_height"] * df["footprint_perimeter_proxy"], 1e-6
    )
    df["sphericity_proxy"] = (6.0 * (df["est_volume"] ** (2.0 / 3.0))) / np.maximum(
        df["surface_area"], 1e-6
    )
    df["isoperimetric_quotient"] = (36.0 * np.pi * (df["est_volume"] ** 2)) / np.maximum(
        df["surface_area"] ** 3, 1e-6
    )

    # 4. Thermal Transmission & Net Opaque Wall Envelope Proxies
    df["net_opaque_wall_area"] = np.maximum(df["wall_area"] - df["glazing_area"], 0.0)
    df["window_to_wall_ratio"] = df["glazing_area"] / np.maximum(df["wall_area"], 1e-6)
    df["net_wall_to_surface_ratio"] = df["net_opaque_wall_area"] / np.maximum(df["surface_area"], 1e-6)
    df["net_wall_to_roof_ratio"] = df["net_opaque_wall_area"] / np.maximum(df["roof_area"], 1e-6)
    df["glazing_x_wall"] = df["glazing_area"] * df["wall_area"]
    df["glazing_x_surface"] = df["glazing_area"] * df["surface_area"]
    df["glazing_to_wall_ratio"] = df["glazing_area"] / np.maximum(df["wall_area"], 1e-6)
    df["glazing_to_surface_ratio"] = df["glazing_area"] / np.maximum(df["surface_area"], 1e-6)
    df["glazing_transmission_proxy"] = df["glazing_area"] * df["overall_height"]

    # Thermal envelope heat loss conductance factoring net opaque wall area
    df["envelope_heat_loss_proxy"] = (
        df["net_opaque_wall_area"] * 1.0 + df["roof_area"] * 0.6 + df["glazing_area"] * 2.8
    )

    # 5. Compactness Interactions & Discrete Shape Signature
    df["compactness_x_height"] = df["relative_compactness"] * df["overall_height"]
    df["compactness_x_surface"] = df["relative_compactness"] * df["surface_area"]
    df["compactness_x_volume"] = df["relative_compactness"] * df["est_volume"]
    df["shape_signature"] = (
        (df["relative_compactness"] * 100).round().astype(int) * 1000
        + (df["overall_height"] * 10).round().astype(int)
    )

    # 6. Orientation Cyclical Encoding & Solar-Glazing Alignment
    if "orientation" in df.columns:
        rad = (df["orientation"] - 2) * (2 * np.pi / 4.0)
        df["orientation_sin"] = np.sin(rad)
        df["orientation_cos"] = np.cos(rad)

    # 7. Facade-Specific Directional Solar Capture & Glazing Distribution
    if "glazing_area_distribution" in df.columns:
        dist = df["glazing_area_distribution"]
        # Ecotect Glazing Distribution mapping: 0=None, 1=Uniform (25% each), 2=North (55%), 3=East (55%), 4=South (55%), 5=West (55%)
        frac_n = np.where(dist == 0, 0.0, np.where(dist == 1, 0.25, np.where(dist == 2, 0.55, 0.15)))
        frac_e = np.where(dist == 0, 0.0, np.where(dist == 1, 0.25, np.where(dist == 3, 0.55, 0.15)))
        frac_s = np.where(dist == 0, 0.0, np.where(dist == 1, 0.25, np.where(dist == 4, 0.55, 0.15)))
        frac_w = np.where(dist == 0, 0.0, np.where(dist == 1, 0.25, np.where(dist == 5, 0.55, 0.15)))

        df["glazing_north_area"] = df["glazing_area"] * frac_n
        df["glazing_east_area"] = df["glazing_area"] * frac_e
        df["glazing_south_area"] = df["glazing_area"] * frac_s
        df["glazing_west_area"] = df["glazing_area"] * frac_w

        # Passive solar heat gain weighted by cardinal solar radiation intensity (South maximum, North diffuse minimal)
        df["directional_solar_gain_proxy"] = (
            df["glazing_north_area"] * 0.20
            + df["glazing_east_area"] * 0.55
            + df["glazing_south_area"] * 1.00
            + df["glazing_west_area"] * 0.55
        )

        # Net thermal energy balance (Envelope conduction loss minus passive solar capture)
        df["net_energy_balance_proxy"] = (
            df["envelope_heat_loss_proxy"] - df["directional_solar_gain_proxy"] * 4.5
        )

        if "orientation" in df.columns:
            df["glazing_dist_is_zero"] = (df["glazing_area_distribution"] == 0).astype(float)
            df["glazing_active_intensity"] = df["glazing_area"] * (
                df["glazing_area_distribution"] > 0
            ).astype(float)
            df["solar_glazing_aligned"] = (
                (df["orientation"] == df["glazing_area_distribution"])
                & (df["glazing_area_distribution"] > 1)
            ).astype(float)
            angle_diff = np.abs(df["orientation"] - df["glazing_area_distribution"]) * (np.pi / 2.0)
            df["solar_glazing_alignment_cos"] = np.where(
                df["glazing_area_distribution"] >= 2,
                np.cos(angle_diff),
                np.where(df["glazing_area_distribution"] == 1, 0.5, 0.0),
            )
            df["solar_gain_directional_proxy"] = (
                df["glazing_area"] * df["solar_glazing_alignment_cos"]
            )

    # 8. Polynomial & Non-linear combinations
    df["log_surface_area"] = np.log1p(df["surface_area"])
    df["log_est_volume"] = np.log1p(df["est_volume"])
    df["compactness_squared"] = df["relative_compactness"] ** 2
    df["height_squared"] = df["overall_height"] ** 2

    return df


train_engineered = engineer_features(train_df)
test_engineered = engineer_features(test_df)

engineered_features = [
    c
    for c in train_engineered.columns
    if c not in [target_col, "cooling_load", "id", "fold"]
]

# 10-Fold Stratified Split based on binned heating_load quantiles
N_SPLITS = 10
target_bins = pd.qcut(
    train_engineered[target_col], q=10, labels=False, duplicates="drop"
)
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
train_engineered["fold"] = -1

for fold, (train_idx, val_idx) in enumerate(skf.split(train_engineered, target_bins)):
    train_engineered.loc[val_idx, "fold"] = fold


# ============================================================================
# 2. Model Architecture Definitions
# ============================================================================
class ResidualTabularBlock(nn.Module):
    """Residual dense block with LayerNorm, GELU, and Dropout for tabular regression."""

    def __init__(self, dim: int, dropout_rate: float = 0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim, dim),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class TabularResNet(nn.Module):
    """Deep Tabular Residual Regressor tailored for continuous physical simulation targets."""

    def __init__(
        self,
        in_features: int,
        hidden_dim: int = 128,
        num_blocks: int = 3,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Linear(in_features, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )

        self.res_blocks = nn.ModuleList(
            [
                ResidualTabularBlock(hidden_dim, dropout_rate=dropout_rate)
                for _ in range(num_blocks)
            ]
        )

        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate / 2.0),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_layer(x)
        for block in self.res_blocks:
            h = block(h)
        out = self.head(h)
        return out.squeeze(-1)


def get_lgbm_params(random_state: int = 42):
    return {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": 6,
        "min_child_samples": 8,
        "subsample": 0.85,
        "subsample_freq": 1,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.05,
        "reg_lambda": 1.0,
        "random_state": random_state,
        "n_estimators": 1500,
        "verbose": -1,
        "n_jobs": -1,
    }


def get_xgb_params(random_state: int = 42):
    return {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "learning_rate": 0.03,
        "max_depth": 5,
        "min_child_weight": 3,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.05,
        "reg_lambda": 1.0,
        "random_state": random_state,
        "n_estimators": 1500,
        "n_jobs": -1,
    }


def get_catboost_params(random_state: int = 42):
    return {
        "iterations": 1500,
        "learning_rate": 0.03,
        "depth": 6,
        "l2_leaf_reg": 3.0,
        "random_seed": random_state,
        "verbose": 0,
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
    }


def get_extratrees_model(random_state: int = 42):
    return ExtraTreesRegressor(
        n_estimators=500,
        max_depth=12,
        min_samples_split=3,
        min_samples_leaf=1,
        max_features=0.85,
        random_state=random_state,
        n_jobs=-1,
    )


def get_randomforest_model(random_state: int = 42):
    return RandomForestRegressor(
        n_estimators=500,
        max_depth=12,
        min_samples_split=3,
        min_samples_leaf=1,
        max_features=0.85,
        random_state=random_state,
        n_jobs=-1,
    )


def get_svr_model():
    return SVR(
        C=35.0,
        epsilon=0.05,
        kernel="rbf",
        gamma="scale",
    )


def get_kernelridge_model():
    return KernelRidge(
        alpha=0.1,
        kernel="rbf",
        gamma=None,
    )


# ============================================================================
# 3. Model Training & Out-of-Fold Evaluation
# ============================================================================
n_samples = len(train_engineered)
n_test = len(test_engineered)
y_true = train_engineered[target_col].values
folds = sorted(train_engineered["fold"].unique())

oof_lgb = np.zeros(n_samples)
oof_xgb = np.zeros(n_samples)
oof_cb = np.zeros(n_samples)
oof_et = np.zeros(n_samples)
oof_rf = np.zeros(n_samples)
oof_svr = np.zeros(n_samples)
oof_kr = np.zeros(n_samples)
oof_nn = np.zeros(n_samples)

test_preds_lgb = np.zeros(n_test)
test_preds_xgb = np.zeros(n_test)
test_preds_cb = np.zeros(n_test)
test_preds_et = np.zeros(n_test)
test_preds_rf = np.zeros(n_test)
test_preds_svr = np.zeros(n_test)
test_preds_kr = np.zeros(n_test)
test_preds_nn = np.zeros(n_test)

X_test_raw = test_engineered[engineered_features].values

for fold_idx in folds:
    train_mask = train_engineered["fold"] != fold_idx
    val_mask = train_engineered["fold"] == fold_idx

    X_train = train_engineered.loc[train_mask, engineered_features].values
    y_train = train_engineered.loc[train_mask, target_col].values

    X_val = train_engineered.loc[val_mask, engineered_features].values
    y_val = train_engineered.loc[val_mask, target_col].values
    val_indices = np.where(val_mask)[0]

    # Standard scaled matrices strictly per-fold
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test_raw)

    # Model A: LightGBM
    lgb_params = get_lgbm_params(random_state=42 + fold_idx)
    lgb_model = lgb.LGBMRegressor(**lgb_params)
    lgb_model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(50, verbose=False),
            lgb.log_evaluation(-1),
        ],
    )
    oof_lgb[val_indices] = lgb_model.predict(X_val)
    test_preds_lgb += lgb_model.predict(X_test_raw) / len(folds)

    # Model B: XGBoost
    xgb_params = get_xgb_params(random_state=42 + fold_idx)
    xgb_model = xgb.XGBRegressor(**xgb_params, early_stopping_rounds=50)
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    oof_xgb[val_indices] = xgb_model.predict(X_val)
    test_preds_xgb += xgb_model.predict(X_test_raw) / len(folds)

    # Model C: CatBoost
    cb_params = get_catboost_params(random_state=42 + fold_idx)
    cb_model = CatBoostRegressor(**cb_params)
    cb_model.fit(
        X_train,
        y_train,
        eval_set=(X_val, y_val),
        early_stopping_rounds=50,
        verbose=False,
    )
    oof_cb[val_indices] = cb_model.predict(X_val)
    test_preds_cb += cb_model.predict(X_test_raw) / len(folds)

    # Model D: ExtraTrees
    et_model = get_extratrees_model(random_state=42 + fold_idx)
    et_model.fit(X_train, y_train)
    oof_et[val_indices] = et_model.predict(X_val)
    test_preds_et += et_model.predict(X_test_raw) / len(folds)

    # Model E: RandomForest
    rf_model = get_randomforest_model(random_state=42 + fold_idx)
    rf_model.fit(X_train, y_train)
    oof_rf[val_indices] = rf_model.predict(X_val)
    test_preds_rf += rf_model.predict(X_test_raw) / len(folds)

    # Model F: Support Vector Regressor (RBF Kernel on Scaled Features)
    svr_model = get_svr_model()
    svr_model.fit(X_train_scaled, y_train)
    oof_svr[val_indices] = svr_model.predict(X_val_scaled)
    test_preds_svr += svr_model.predict(X_test_scaled) / len(folds)

    # Model G: Kernel Ridge Regressor (RBF Kernel on Scaled Features)
    kr_model = get_kernelridge_model()
    kr_model.fit(X_train_scaled, y_train)
    oof_kr[val_indices] = kr_model.predict(X_val_scaled)
    test_preds_kr += kr_model.predict(X_test_scaled) / len(folds)

    # Model H: PyTorch TabularResNet
    train_dataset = TensorDataset(
        torch.tensor(X_train_scaled, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    train_loader = DataLoader(
        train_dataset, batch_size=32, shuffle=True, drop_last=False
    )

    nn_model = TabularResNet(
        in_features=len(engineered_features),
        hidden_dim=128,
        num_blocks=3,
        dropout_rate=0.1,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(nn_model.parameters(), lr=3e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=120, eta_min=1e-5
    )

    best_val_loss = float("inf")
    best_weights = None
    patience = 25
    patience_counter = 0

    val_tensor_x = torch.tensor(X_val_scaled, dtype=torch.float32).to(device)
    val_tensor_y = torch.tensor(y_val, dtype=torch.float32).to(device)

    for epoch in range(120):
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
        with torch.no_grad():
            val_preds_epoch = nn_model(val_tensor_x)
            val_loss = criterion(val_preds_epoch, val_tensor_y).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = copy.deepcopy(nn_model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    nn_model.load_state_dict(best_weights)
    nn_model.eval()
    with torch.no_grad():
        oof_nn[val_indices] = nn_model(val_tensor_x).cpu().numpy()
        test_tensor_x = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
        test_preds_nn += nn_model(test_tensor_x).cpu().numpy().flatten() / len(folds)

# ============================================================================
# 4. Constrained Optimal Blending
# ============================================================================
oof_matrix = np.column_stack(
    [oof_lgb, oof_xgb, oof_cb, oof_et, oof_rf, oof_svr, oof_kr, oof_nn]
)
test_matrix = np.column_stack(
    [
        test_preds_lgb,
        test_preds_xgb,
        test_preds_cb,
        test_preds_et,
        test_preds_rf,
        test_preds_svr,
        test_preds_kr,
        test_preds_nn,
    ]
)

n_models = oof_matrix.shape[1]


def blend_objective(weights):
    w = weights / np.maximum(np.sum(weights), 1e-12)
    blend = np.dot(oof_matrix, w)
    rmse = root_mean_squared_error(y_true, blend)
    l2_reg = 1e-4 * np.sum(weights ** 2)
    return rmse + l2_reg


init_weights = np.ones(n_models) / float(n_models)
bounds = [(0.0, 1.0) for _ in range(n_models)]
constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
opt_res = minimize(
    blend_objective,
    init_weights,
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
)
opt_weights = opt_res.x / np.sum(opt_res.x)

oof_final = np.dot(oof_matrix, opt_weights)
test_final = np.dot(test_matrix, opt_weights)
final_validation_score = float(root_mean_squared_error(y_true, oof_final))

# ============================================================================
# 5. Submission Generation
# ============================================================================
submission_df = pd.DataFrame(
    {"id": np.arange(len(test_final), dtype=int), "heating_load": test_final}
)
submission_df.to_csv("./submission/submission.csv", index=False)

print(f"Final Validation Score: {final_validation_score}")
