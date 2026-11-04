import os
import warnings
import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb

warnings.filterwarnings("ignore")

# Ensure required directories exist
os.makedirs("./submission", exist_ok=True)
os.makedirs("./working", exist_ok=True)

# Set seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------
# 1. Feature Engineering Routine
# ---------------------------------------------------------
def engineer_features(df):
    df = df.copy()
    eps = 1e-6

    properties = [
        "atomic_mass",
        "fie",
        "atomic_radius",
        "Density",
        "ElectronAffinity",
        "FusionHeat",
        "ThermalConductivity",
        "Valence",
    ]

    for prop in properties:
        mean_col = f"mean_{prop}"
        wtd_mean_col = f"wtd_mean_{prop}"
        gmean_col = f"gmean_{prop}"
        wtd_gmean_col = f"wtd_gmean_{prop}"
        entropy_col = f"entropy_{prop}"
        wtd_entropy_col = f"wtd_entropy_{prop}"
        std_col = f"std_{prop}"
        wtd_std_col = f"wtd_std_{prop}"

        # Stoichiometric Ratios (Weighted vs Unweighted)
        if mean_col in df and wtd_mean_col in df:
            df[f"{prop}_mean_ratio"] = (df[wtd_mean_col] + eps) / (df[mean_col] + eps)
        if gmean_col in df and wtd_gmean_col in df:
            df[f"{prop}_gmean_ratio"] = (df[wtd_gmean_col] + eps) / (
                df[gmean_col] + eps
            )
        if std_col in df and wtd_std_col in df:
            df[f"{prop}_std_ratio"] = (df[wtd_std_col] + eps) / (df[std_col] + eps)
        if entropy_col in df and wtd_entropy_col in df:
            df[f"{prop}_entropy_ratio"] = (df[wtd_entropy_col] + eps) / (
                df[entropy_col] + eps
            )

        # Coefficient of Variation
        if std_col in df and mean_col in df:
            df[f"{prop}_cv"] = df[std_col] / (df[mean_col] + eps)
        if wtd_std_col in df and wtd_mean_col in df:
            df[f"{prop}_wtd_cv"] = df[wtd_std_col] / (df[wtd_mean_col] + eps)

        # Normalized Stoichiometric Bias Features: (wtd_stat - stat) / (range + eps)
        range_col = f"range_{prop}"
        if range_col in df:
            denom = df[range_col] + eps
            if mean_col in df and wtd_mean_col in df:
                df[f"{prop}_mean_bias"] = (df[wtd_mean_col] - df[mean_col]) / denom
            if gmean_col in df and wtd_gmean_col in df:
                df[f"{prop}_gmean_bias"] = (df[wtd_gmean_col] - df[gmean_col]) / denom
            if std_col in df and wtd_std_col in df:
                df[f"{prop}_std_bias"] = (df[wtd_std_col] - df[std_col]) / denom
            if entropy_col in df and wtd_entropy_col in df:
                df[f"{prop}_entropy_bias"] = (df[wtd_entropy_col] - df[entropy_col]) / denom

        # Geometric to Arithmetic Mean Disparity Index
        if gmean_col in df and mean_col in df:
            df[f"{prop}_am_gm_ratio"] = (df[gmean_col] + eps) / (df[mean_col] + eps)
        if wtd_gmean_col in df and wtd_mean_col in df:
            df[f"{prop}_wtd_am_gm_ratio"] = (df[wtd_gmean_col] + eps) / (
                df[wtd_mean_col] + eps
            )

        # Normalized Entropy
        if "number_of_elements" in df and entropy_col in df:
            df[f"{prop}_norm_entropy"] = df[entropy_col] / (
                np.log1p(df["number_of_elements"]) + eps
            )

        # Log Transforms
        range_col = f"range_{prop}"
        if range_col in df:
            df[f"{prop}_log_range"] = np.log1p(np.maximum(0, df[range_col]))
        if std_col in df:
            df[f"{prop}_log_std"] = np.log1p(np.maximum(0, df[std_col]))
        if wtd_std_col in df:
            df[f"{prop}_log_wtd_std"] = np.log1p(np.maximum(0, df[wtd_std_col]))

        if prop in [
            "Density",
            "ThermalConductivity",
            "FusionHeat",
            "ElectronAffinity",
        ]:
            if mean_col in df:
                df[f"{prop}_log_mean"] = np.log1p(np.maximum(0, df[mean_col]))
            if wtd_mean_col in df:
                df[f"{prop}_log_wtd_mean"] = np.log1p(np.maximum(0, df[wtd_mean_col]))

    # Interaction Features
    if "wtd_mean_Valence" in df and "wtd_mean_Density" in df:
        df["valence_density_prod"] = df["wtd_mean_Valence"] * df["wtd_mean_Density"]
    if "wtd_mean_ThermalConductivity" in df and "wtd_mean_fie" in df:
        df["thermal_fie_ratio"] = (df["wtd_mean_ThermalConductivity"] + eps) / (
            df["wtd_mean_fie"] + eps
        )
    if "wtd_mean_atomic_mass" in df and "wtd_mean_atomic_radius" in df:
        df["mass_radius_ratio"] = (df["wtd_mean_atomic_mass"] + eps) / (
            df["wtd_mean_atomic_radius"] + eps
        )
    if "wtd_mean_FusionHeat" in df and "wtd_mean_atomic_mass" in df:
        df["fusion_mass_ratio"] = (df["wtd_mean_FusionHeat"] + eps) / (
            df["wtd_mean_atomic_mass"] + eps
        )

    # Physical Proxy Interaction Features
    if "wtd_mean_Valence" in df and "wtd_mean_atomic_radius" in df:
        df["electronic_density"] = df["wtd_mean_Valence"] / ((df["wtd_mean_atomic_radius"] ** 3) + eps)
    if "wtd_mean_ThermalConductivity" in df and "wtd_mean_atomic_mass" in df:
        df["mass_phonon_ratio"] = (df["wtd_mean_ThermalConductivity"] + eps) / (df["wtd_mean_atomic_mass"] + eps)
    if "wtd_mean_Density" in df and "wtd_mean_atomic_mass" in df:
        df["density_mass_imbalance"] = (df["wtd_mean_Density"] + eps) / (df["wtd_mean_atomic_mass"] + eps)

    # Domain-Specific Physical Proxy Features
    if "wtd_mean_ThermalConductivity" in df and "wtd_mean_Density" in df and "wtd_mean_atomic_mass" in df:
        df["debye_temp_proxy"] = np.sqrt(
            np.maximum(0, df["wtd_mean_ThermalConductivity"]) / (df["wtd_mean_Density"] * df["wtd_mean_atomic_mass"] + eps)
        )
    if "wtd_mean_Valence" in df and "wtd_mean_Density" in df and "wtd_mean_atomic_mass" in df:
        df["fermi_energy_proxy"] = (
            (df["wtd_mean_Valence"] * df["wtd_mean_Density"]) / (df["wtd_mean_atomic_mass"] + eps)
        ) ** (2.0 / 3.0)
    if "wtd_mean_Valence" in df and "wtd_mean_fie" in df and "wtd_mean_atomic_mass" in df and "wtd_mean_ThermalConductivity" in df:
        df["electron_phonon_proxy"] = (df["wtd_mean_Valence"] * df["wtd_mean_fie"]) / (
            df["wtd_mean_atomic_mass"] * df["wtd_mean_ThermalConductivity"] + eps
        )

    return df


# ---------------------------------------------------------
# 2. PyTorch Gated Residual Network (GRN) with SE Gating & Trainer
# ---------------------------------------------------------
class SEFeatureGating(nn.Module):

    def __init__(self, dim: int, reduction: int = 4):
        super().__init__()
        reduced_dim = max(4, dim // reduction)
        self.gate = nn.Sequential(
            nn.Linear(dim, reduced_dim),
            nn.Mish(),
            nn.Linear(reduced_dim, dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gate(x)


class GRNBlock(nn.Module):

    def __init__(self, hidden_dim: int, dropout_rate: float = 0.15):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Mish(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.se_gate = SEFeatureGating(hidden_dim)
        self.highway_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.act = nn.Mish()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.block(x)
        res = self.se_gate(res)
        hw_gate = self.highway_gate(x)
        return self.act(hw_gate * res + (1.0 - hw_gate) * x)


class GatedResidualNet(nn.Module):

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_blocks: int = 3,
        dropout_rate: float = 0.15,
    ):
        super().__init__()
        self.input_se = SEFeatureGating(input_dim)
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Mish(),
        )
        self.blocks = nn.ModuleList(
            [GRNBlock(hidden_dim, dropout_rate) for _ in range(num_blocks)]
        )
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_se(x)
        x = self.input_layer(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x)


def train_pytorch_grn(
    X_tr, y_tr, X_va, y_va, X_te, input_dim, epochs=80, batch_size=256
):
    model = GatedResidualNet(
        input_dim=input_dim, hidden_dim=256, num_blocks=3, dropout_rate=0.15
    ).to(device)
    criterion = nn.HuberLoss(delta=5.0)
    optimizer = optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=1, eta_min=1e-6
    )

    tr_ds = TensorDataset(
        torch.tensor(X_tr, dtype=torch.float32),
        torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1),
    )
    va_ds = TensorDataset(
        torch.tensor(X_va, dtype=torch.float32),
        torch.tensor(y_va, dtype=torch.float32).unsqueeze(1),
    )

    tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    va_loader = DataLoader(va_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    best_val_loss = float("inf")
    best_weights = None
    patience = 15
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        for bx, by in tr_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()

        scheduler.step()

        model.eval()
        val_sq_err = 0.0
        val_count = 0
        with torch.no_grad():
            for bx, by in va_loader:
                bx, by = bx.to(device), by.to(device)
                out = model(bx)
                val_sq_err += torch.sum((out - by) ** 2).item()
                val_count += len(by)

        val_rmse = np.sqrt(val_sq_err / val_count)

        if val_rmse < best_val_loss:
            best_val_loss = val_rmse
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_weights is not None:
        model.load_state_dict(best_weights)
    model.eval()

    with torch.no_grad():
        va_tensor = torch.tensor(X_va, dtype=torch.float32).to(device)
        val_preds = model(va_tensor).cpu().numpy().flatten()

        te_tensor = torch.tensor(X_te, dtype=torch.float32).to(device)
        test_preds = model(te_tensor).cpu().numpy().flatten()

    return np.maximum(0, val_preds), np.maximum(0, test_preds)


# ---------------------------------------------------------
# 3. Data Loading & Feature Pipeline Execution
# ---------------------------------------------------------
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "critical_temp"
y_full = train_df[target_col].values
X_train_raw = train_df.drop(columns=[target_col])

if target_col in test_df.columns:
    X_test_raw = test_df.drop(columns=[target_col])
else:
    X_test_raw = test_df.copy()

test_ids = np.arange(len(X_test_raw))

X_full = engineer_features(X_train_raw)
X_test = engineer_features(X_test_raw)

num_samples = len(X_full)
num_test = len(X_test)
num_models = 9  # LightGBM (MSE), LightGBM (Huber), XGBoost, CatBoost, LightGBM (Log), CatBoost (Log), PyTorch GRN, ExtraTrees, RandomForest

# Target quantile stratification for 5-fold CV
num_bins = 10
y_bins = pd.qcut(y_full, q=num_bins, labels=False, duplicates="drop")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_preds = np.zeros((num_samples, num_models))
test_preds_models = np.zeros((num_test, num_models))

# GBDT Hyperparameters
lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.03,
    "num_leaves": 127,
    "max_depth": -1,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}

lgb_huber_params = {
    "objective": "huber",
    "alpha": 10.0,
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.03,
    "num_leaves": 127,
    "max_depth": -1,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}

# ---------------------------------------------------------
# 4. 5-Fold Cross Validation Loop
# ---------------------------------------------------------
for fold, (train_idx, val_idx) in enumerate(skf.split(X_full, y_bins)):
    X_tr = X_full.iloc[train_idx].copy()
    y_tr = y_full[train_idx]
    X_va = X_full.iloc[val_idx].copy()
    y_va = y_full[val_idx]
    X_te = X_test.copy()

    # Clean NaNs/Infs strictly using training set statistics
    medians = X_tr.median()
    X_tr = X_tr.replace([np.inf, -np.inf], np.nan).fillna(medians)
    X_va = X_va.replace([np.inf, -np.inf], np.nan).fillna(medians)
    X_te = X_te.replace([np.inf, -np.inf], np.nan).fillna(medians)

    # Unsupervised PCA embeddings fit strictly on training fold
    central_cols = [c for c in X_tr.columns if "mean" in c or "gmean" in c]
    dispersion_cols = [
        c for c in X_tr.columns if "std" in c or "entropy" in c or "range" in c
    ]

    pca_central = PCA(n_components=5, random_state=42)
    pca_disp = PCA(n_components=5, random_state=42)

    tr_cent_pca = pca_central.fit_transform(X_tr[central_cols])
    va_cent_pca = pca_central.transform(X_va[central_cols])
    te_cent_pca = pca_central.transform(X_te[central_cols])

    tr_disp_pca = pca_disp.fit_transform(X_tr[dispersion_cols])
    va_disp_pca = pca_disp.transform(X_va[dispersion_cols])
    te_disp_pca = pca_disp.transform(X_te[dispersion_cols])

    for i in range(5):
        X_tr[f"pca_central_{i}"] = tr_cent_pca[:, i]
        X_va[f"pca_central_{i}"] = va_cent_pca[:, i]
        X_te[f"pca_central_{i}"] = te_cent_pca[:, i]

        X_tr[f"pca_disp_{i}"] = tr_disp_pca[:, i]
        X_va[f"pca_disp_{i}"] = va_disp_pca[:, i]
        X_te[f"pca_disp_{i}"] = te_disp_pca[:, i]

    # KMeans clustering on core physical property features strictly fitted on training split
    core_prop_cols = [
        c for c in X_tr.columns if any(
            p in c.lower() for p in ["atomic_mass", "atomic_radius", "valence", "fie", "density"]
        )
    ]
    km_scaler = StandardScaler()
    X_tr_core_scaled = km_scaler.fit_transform(X_tr[core_prop_cols])
    X_va_core_scaled = km_scaler.transform(X_va[core_prop_cols])
    X_te_core_scaled = km_scaler.transform(X_te[core_prop_cols])

    kmeans = KMeans(n_clusters=8, random_state=42 + fold, n_init=10)
    tr_km_dist = kmeans.fit_transform(X_tr_core_scaled)
    va_km_dist = kmeans.transform(X_va_core_scaled)
    te_km_dist = kmeans.transform(X_te_core_scaled)

    for k in range(tr_km_dist.shape[1]):
        X_tr[f"kmeans_dist_{k}"] = tr_km_dist[:, k]
        X_va[f"kmeans_dist_{k}"] = va_km_dist[:, k]
        X_te[f"kmeans_dist_{k}"] = te_km_dist[:, k]

    # Scale features strictly on training fold
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_va_s = scaler.transform(X_va)
    X_te_s = scaler.transform(X_te)

    # Model 0: LightGBM (MSE)
    lgb_tr = lgb.Dataset(X_tr_s, label=y_tr)
    lgb_va = lgb.Dataset(X_va_s, label=y_va, reference=lgb_tr)
    lgb_model = lgb.train(
        lgb_params,
        lgb_tr,
        num_boost_round=2500,
        valid_sets=[lgb_va],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )
    oof_preds[val_idx, 0] = np.maximum(0, lgb_model.predict(X_va_s))
    test_preds_models[:, 0] += np.maximum(0, lgb_model.predict(X_te_s)) / 5.0

    # Model 1: LightGBM (Huber)
    lgb_huber_tr = lgb.Dataset(X_tr_s, label=y_tr)
    lgb_huber_va = lgb.Dataset(X_va_s, label=y_va, reference=lgb_huber_tr)
    lgb_huber_model = lgb.train(
        lgb_huber_params,
        lgb_huber_tr,
        num_boost_round=2500,
        valid_sets=[lgb_huber_va],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )
    oof_preds[val_idx, 1] = np.maximum(0, lgb_huber_model.predict(X_va_s))
    test_preds_models[:, 1] += np.maximum(0, lgb_huber_model.predict(X_te_s)) / 5.0

    # Model 2: XGBoost
    xgb_model = xgb.XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        n_estimators=2500,
        learning_rate=0.03,
        max_depth=7,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42 + fold,
        n_jobs=-1,
        tree_method="hist",
        early_stopping_rounds=50,
    )
    xgb_model.fit(X_tr_s, y_tr, eval_set=[(X_va_s, y_va)], verbose=False)
    oof_preds[val_idx, 2] = np.maximum(0, xgb_model.predict(X_va_s))
    test_preds_models[:, 2] += np.maximum(0, xgb_model.predict(X_te_s)) / 5.0

    # Model 3: CatBoost
    cat_model = cb.CatBoostRegressor(
        loss_function="RMSE",
        eval_metric="RMSE",
        iterations=2500,
        learning_rate=0.03,
        depth=7,
        l2_leaf_reg=3.0,
        random_seed=42 + fold,
        verbose=False,
        early_stopping_rounds=50,
    )
    cat_model.fit(X_tr_s, y_tr, eval_set=(X_va_s, y_va), verbose=False)
    oof_preds[val_idx, 3] = np.maximum(0, cat_model.predict(X_va_s))
    test_preds_models[:, 3] += np.maximum(0, cat_model.predict(X_te_s)) / 5.0

    # Model 4: LightGBM (Log target)
    y_tr_log = np.log1p(np.maximum(0, y_tr))
    y_va_log = np.log1p(np.maximum(0, y_va))
    lgb_log_tr = lgb.Dataset(X_tr_s, label=y_tr_log)
    lgb_log_va = lgb.Dataset(X_va_s, label=y_va_log, reference=lgb_log_tr)
    lgb_log_model = lgb.train(
        lgb_params,
        lgb_log_tr,
        num_boost_round=2500,
        valid_sets=[lgb_log_va],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )
    oof_preds[val_idx, 4] = np.maximum(0, np.expm1(lgb_log_model.predict(X_va_s)))
    test_preds_models[:, 4] += np.maximum(0, np.expm1(lgb_log_model.predict(X_te_s))) / 5.0

    # Model 5: CatBoost (Log target)
    cat_log_model = cb.CatBoostRegressor(
        loss_function="RMSE",
        eval_metric="RMSE",
        iterations=2500,
        learning_rate=0.03,
        depth=7,
        l2_leaf_reg=3.0,
        random_seed=42 + fold,
        verbose=False,
        early_stopping_rounds=50,
    )
    cat_log_model.fit(X_tr_s, y_tr_log, eval_set=(X_va_s, y_va_log), verbose=False)
    oof_preds[val_idx, 5] = np.maximum(0, np.expm1(cat_log_model.predict(X_va_s)))
    test_preds_models[:, 5] += np.maximum(0, np.expm1(cat_log_model.predict(X_te_s))) / 5.0

    # Model 6: PyTorch Gated Residual Network (GRN)
    nn_va_pred, nn_te_pred = train_pytorch_grn(
        X_tr_s, y_tr, X_va_s, y_va, X_te_s, input_dim=X_tr_s.shape[1]
    )
    oof_preds[val_idx, 6] = nn_va_pred
    test_preds_models[:, 6] += nn_te_pred / 5.0

    # Model 7: ExtraTreesRegressor
    et_model = ExtraTreesRegressor(
        n_estimators=300,
        max_depth=22,
        min_samples_split=4,
        random_state=42 + fold,
        n_jobs=-1,
    )
    et_model.fit(X_tr_s, y_tr)
    oof_preds[val_idx, 7] = np.maximum(0, et_model.predict(X_va_s))
    test_preds_models[:, 7] += np.maximum(0, et_model.predict(X_te_s)) / 5.0

    # Model 8: RandomForestRegressor
    rf_model = RandomForestRegressor(
        n_estimators=300,
        max_depth=22,
        min_samples_split=4,
        random_state=42 + fold,
        n_jobs=-1,
    )
    rf_model.fit(X_tr_s, y_tr)
    oof_preds[val_idx, 8] = np.maximum(0, rf_model.predict(X_va_s))
    test_preds_models[:, 8] += np.maximum(0, rf_model.predict(X_te_s)) / 5.0

    fold_rmse = root_mean_squared_error(y_va, np.mean(oof_preds[val_idx], axis=1))
    print(f"Fold {fold+1}/5 Completed - Average Fold RMSE: {fold_rmse:.4f}")


# ---------------------------------------------------------
# 5. SLSQP Constrained Optimization for Ensemble Blending
# ---------------------------------------------------------
def ensemble_loss(weights, oof_matrix, y_true):
    w = weights / np.sum(weights)
    pred = np.maximum(0, oof_matrix @ w)
    return root_mean_squared_error(y_true, pred)


init_weights = np.ones(num_models) / num_models
bounds = [(0, 1) for _ in range(num_models)]
constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

opt_res = minimize(
    ensemble_loss,
    init_weights,
    args=(oof_preds, y_full),
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
)

opt_weights = opt_res.x / np.sum(opt_res.x)

# Compute final out-of-fold validation score
final_oof_preds = np.maximum(0, oof_preds @ opt_weights)
score = root_mean_squared_error(y_full, final_oof_preds)

# Generate final test predictions using ensemble weights
final_test_preds = np.maximum(0, test_preds_models @ opt_weights)

# Save submission file
sub_df = pd.DataFrame({"id": test_ids, "critical_temp": final_test_preds})
sub_df.to_csv("./submission/submission.csv", index=False)

print(f"Final Validation Score: {score}")
