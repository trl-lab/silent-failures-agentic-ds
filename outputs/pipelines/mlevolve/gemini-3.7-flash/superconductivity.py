import os
from typing import Any, Dict, List, Tuple
import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb


# =========================================================================
# 1. Feature Engineering
# =========================================================================


def engineer_superconductivity_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineers domain-informed physical, dispersion, and interaction features

    from constituent property moments for superconductivity prediction.
    """
    fe = df.copy()
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

    # 1. Intra-property moment ratios and dispersion metrics
    for prop in properties:
        mean_col = f"mean_{prop}"
        wtd_mean_col = f"wtd_mean_{prop}"
        gmean_col = f"gmean_{prop}"
        wtd_gmean_col = f"wtd_gmean_{prop}"
        std_col = f"std_{prop}"
        wtd_std_col = f"wtd_std_{prop}"
        range_col = f"range_{prop}"
        wtd_range_col = f"wtd_range_{prop}"
        entropy_col = f"entropy_{prop}"
        wtd_entropy_col = f"wtd_entropy_{prop}"

        # Stoichiometric weighting skew
        if mean_col in fe.columns and wtd_mean_col in fe.columns:
            fe[f"{prop}_wtd_to_mean_ratio"] = fe[wtd_mean_col] / (fe[mean_col] + eps)
            fe[f"{prop}_wtd_mean_diff"] = fe[wtd_mean_col] - fe[mean_col]

        # Dispersion / Relative variation (Coefficient of Variation)
        if std_col in fe.columns and mean_col in fe.columns:
            fe[f"{prop}_coef_variation"] = fe[std_col] / (fe[mean_col] + eps)
        if wtd_std_col in fe.columns and wtd_mean_col in fe.columns:
            fe[f"{prop}_wtd_coef_variation"] = fe[wtd_std_col] / (
                fe[wtd_mean_col] + eps
            )

        # Geometric vs Arithmetic mean discrepancy (indicator of constituent disparity)
        if mean_col in fe.columns and gmean_col in fe.columns:
            fe[f"{prop}_mean_gmean_diff"] = fe[mean_col] - fe[gmean_col]
        if wtd_mean_col in fe.columns and wtd_gmean_col in fe.columns:
            fe[f"{prop}_wtd_mean_gmean_diff"] = fe[wtd_mean_col] - fe[wtd_gmean_col]

        # Normalized entropy relative to element multiplicity
        if entropy_col in fe.columns and "number_of_elements" in fe.columns:
            fe[f"{prop}_normalized_entropy"] = fe[entropy_col] / np.log(
                fe["number_of_elements"] + 1.0 + eps
            )
        if wtd_entropy_col in fe.columns and "number_of_elements" in fe.columns:
            fe[f"{prop}_wtd_normalized_entropy"] = fe[wtd_entropy_col] / np.log(
                fe["number_of_elements"] + 1.0 + eps
            )

        # Range to standard deviation ratio (tail/outlier distribution behavior)
        if range_col in fe.columns and std_col in fe.columns:
            fe[f"{prop}_range_to_std"] = fe[range_col] / (fe[std_col] + eps)
        if wtd_range_col in fe.columns and wtd_std_col in fe.columns:
            fe[f"{prop}_wtd_range_to_std"] = fe[wtd_range_col] / (fe[wtd_std_col] + eps)

    # 2. Log-transforms of heavy-tailed transport and thermal properties
    skewed_props = ["ThermalConductivity", "FusionHeat", "Density", "ElectronAffinity"]
    for prop in skewed_props:
        for stat in [
            "mean",
            "wtd_mean",
            "gmean",
            "wtd_gmean",
            "range",
            "wtd_range",
            "std",
            "wtd_std",
        ]:
            col = f"{stat}_{prop}"
            if col in fe.columns:
                fe[f"log1p_{col}"] = np.log1p(np.maximum(0, fe[col]))

    # 3. Inter-property physical interactions (Mulliken, Valence Density, Electron-Phonon proxies)
    if "mean_fie" in fe.columns and "mean_ElectronAffinity" in fe.columns:
        fe["mean_mulliken_electronegativity"] = (
            fe["mean_fie"] + fe["mean_ElectronAffinity"]
        ) / 2.0
    if "wtd_mean_fie" in fe.columns and "wtd_mean_ElectronAffinity" in fe.columns:
        fe["wtd_mean_mulliken_electronegativity"] = (
            fe["wtd_mean_fie"] + fe["wtd_mean_ElectronAffinity"]
        ) / 2.0

    # Valence Electron Volume Density (Valence / Atomic Volume proxy ~ r^3)
    if "mean_Valence" in fe.columns and "mean_atomic_radius" in fe.columns:
        fe["mean_valence_electron_density"] = fe["mean_Valence"] / (
            ((fe["mean_atomic_radius"] / 100.0) ** 3) + eps
        )
    if "wtd_mean_Valence" in fe.columns and "wtd_mean_atomic_radius" in fe.columns:
        fe["wtd_mean_valence_electron_density"] = fe["wtd_mean_Valence"] / (
            ((fe["wtd_mean_atomic_radius"] / 100.0) ** 3) + eps
        )

    # Atomic Packing / Density Index: Density * r^3 / atomic_mass
    if (
        "mean_Density" in fe.columns
        and "mean_atomic_radius" in fe.columns
        and "mean_atomic_mass" in fe.columns
    ):
        fe["mean_packing_index"] = (
            fe["mean_Density"] * ((fe["mean_atomic_radius"] / 100.0) ** 3)
        ) / (fe["mean_atomic_mass"] + eps)
    if (
        "wtd_mean_Density" in fe.columns
        and "wtd_mean_atomic_radius" in fe.columns
        and "wtd_mean_atomic_mass" in fe.columns
    ):
        fe["wtd_mean_packing_index"] = (
            fe["wtd_mean_Density"] * ((fe["wtd_mean_atomic_radius"] / 100.0) ** 3)
        ) / (fe["wtd_mean_atomic_mass"] + eps)

    # Thermal conductivity to mass ratio (thermal transport per atomic inertia)
    if "mean_ThermalConductivity" in fe.columns and "mean_atomic_mass" in fe.columns:
        fe["mean_thermal_to_mass"] = fe["mean_ThermalConductivity"] / (
            fe["mean_atomic_mass"] + eps
        )
    if (
        "wtd_mean_ThermalConductivity" in fe.columns
        and "wtd_mean_atomic_mass" in fe.columns
    ):
        fe["wtd_mean_thermal_to_mass"] = fe["wtd_mean_ThermalConductivity"] / (
            fe["wtd_mean_atomic_mass"] + eps
        )

    # Cohesive energy / Ionization coupling
    if "mean_FusionHeat" in fe.columns and "mean_fie" in fe.columns:
        fe["mean_fusion_to_fie_ratio"] = fe["mean_FusionHeat"] / (fe["mean_fie"] + eps)
    if "wtd_mean_FusionHeat" in fe.columns and "wtd_mean_fie" in fe.columns:
        fe["wtd_mean_fusion_to_fie_ratio"] = fe["wtd_mean_FusionHeat"] / (
            fe["wtd_mean_fie"] + eps
        )

    # Global cross-property chemical disorder (mean across normalized entropies)
    entropy_cols = [c for c in fe.columns if "normalized_entropy" in c]
    if entropy_cols:
        fe["global_chemical_disorder_mean"] = fe[entropy_cols].mean(axis=1)
        fe["global_chemical_disorder_std"] = fe[entropy_cols].std(axis=1)

    # Replace potential infinite values or NaNs resulting from edge-case calculations
    fe.replace([np.inf, -np.inf], np.nan, inplace=True)
    fe.fillna(0.0, inplace=True)

    return fe


# =========================================================================
# 2. Deep Tabular Neural Network Architecture & Model Configurations
# =========================================================================


class GatedResidualBlock(nn.Module):
    """Gated Residual Block with Layer Normalization, GELU activation,

    feature gating, and dropout for robust tabular representation learning.
    """

    def __init__(self, hidden_dim: int, dropout_rate: float = 0.15):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.linear1 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.act = nn.GELU()
        self.linear2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.gate = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.norm1(x)
        out = self.linear1(out)
        out = self.act(out)
        out = self.linear2(out)
        gate_weights = torch.sigmoid(self.gate(residual))
        out = out * gate_weights
        out = self.dropout(out)
        return residual + out


class SuperconductorGatedResNet(nn.Module):
    """Deep Residual Tabular Network tailored for superconductivity prediction."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [256, 256, 128],
        dropout_rate: float = 0.15,
        output_dim: int = 1,
    ):
        super().__init__()
        self.input_norm = nn.BatchNorm1d(input_dim)
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )

        self.blocks = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.blocks.append(
                GatedResidualBlock(hidden_dim=hidden_dims[i], dropout_rate=dropout_rate)
            )
            if hidden_dims[i] != hidden_dims[i + 1]:
                self.blocks.append(
                    nn.Sequential(
                        nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                        nn.LayerNorm(hidden_dims[i + 1]),
                        nn.GELU(),
                        nn.Dropout(dropout_rate),
                    )
                )

        self.blocks.append(
            GatedResidualBlock(hidden_dim=hidden_dims[-1], dropout_rate=dropout_rate)
        )

        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dims[-1]),
            nn.Linear(hidden_dims[-1], hidden_dims[-1] // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate / 2.0),
            nn.Linear(hidden_dims[-1] // 2, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_norm(x)
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        out = self.head(x)
        return out


class RMSELoss(nn.Module):
    """Exact Root Mean Squared Error loss aligned with the competition metric."""

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(self.mse(y_pred.view(-1), y_true.view(-1)) + self.eps)


def create_neural_model_and_optimizer(
    input_dim: int,
    lr: float = 1.5e-3,
    weight_decay: float = 1e-4,
    device: str = "cpu",
) -> Tuple[nn.Module, nn.Module, optim.Optimizer, optim.lr_scheduler._LRScheduler]:
    model = SuperconductorGatedResNet(
        input_dim=input_dim,
        hidden_dims=[256, 256, 128],
        dropout_rate=0.15,
        output_dim=1,
    ).to(device)

    criterion = RMSELoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999)
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=80, eta_min=1e-6)

    return model, criterion, optimizer, scheduler


def get_lightgbm_regressor(seed: int = 42) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        objective="regression",
        metric="rmse",
        n_estimators=3000,
        learning_rate=0.03,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=20,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.80,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )


def get_xgboost_regressor(seed: int = 42) -> xgb.XGBRegressor:
    return xgb.XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        n_estimators=3000,
        learning_rate=0.03,
        max_depth=7,
        min_child_weight=3,
        subsample=0.85,
        colsample_bytree=0.80,
        colsample_bylevel=0.85,
        reg_alpha=0.1,
        reg_lambda=1.5,
        tree_method="hist",
        early_stopping_rounds=50,
        random_state=seed,
        n_jobs=-1,
    )


def get_catboost_regressor(seed: int = 42) -> cb.CatBoostRegressor:
    return cb.CatBoostRegressor(
        loss_function="RMSE",
        eval_metric="RMSE",
        iterations=3000,
        learning_rate=0.04,
        depth=7,
        l2_leaf_reg=4.0,
        random_seed=seed,
        verbose=False,
        thread_count=-1,
    )


def get_extratrees_regressor(seed: int = 42) -> ExtraTreesRegressor:
    return ExtraTreesRegressor(
        n_estimators=300,
        max_depth=25,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features=0.85,
        random_state=seed,
        n_jobs=-1,
    )


# =========================================================================
# 3. Main Training, Cross-Validation, and Submission Pipeline
# =========================================================================


def main():
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("./working", exist_ok=True)
    os.makedirs("./submission", exist_ok=True)

    print("Loading datasets...")
    train_df = pd.read_parquet("./input/train.parquet")
    test_df = pd.read_parquet("./input/test.parquet")

    target_col = "critical_temp"
    target = train_df[target_col].copy()
    train_raw = train_df.drop(columns=[target_col])
    test_raw = test_df.copy()

    print(f"Raw train shape: {train_df.shape}, Raw test shape: {test_df.shape}")

    # Feature Engineering
    print("Engineering domain-specific features...")
    train_fe = engineer_superconductivity_features(train_raw)
    test_fe = engineer_superconductivity_features(test_raw)

    feature_cols = [c for c in train_fe.columns]
    print(f"Total features after engineering: {len(feature_cols)}")

    # 5-Fold Cross-Validation
    num_folds = 5
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=seed)

    folds = np.zeros(len(train_fe), dtype=int)
    for fold_idx, (_, val_idx) in enumerate(kf.split(train_fe)):
        folds[val_idx] = fold_idx

    X = train_fe[feature_cols].values
    y = target.values
    X_test_raw = test_fe[feature_cols].values

    n_samples = len(train_fe)
    n_test = len(test_fe)
    input_dim = len(feature_cols)

    model_names = ["lightgbm", "xgboost", "catboost", "extratrees", "neural_net"]
    oof_predictions = {name: np.zeros(n_samples) for name in model_names}
    test_predictions = {name: np.zeros(n_test) for name in model_names}

    print(f"\nBeginning {num_folds}-Fold Cross-Validation on device: {device}")

    for fold in range(num_folds):
        train_idx = np.where(folds != fold)[0]
        val_idx = np.where(folds == fold)[0]

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        # 1. LightGBM
        lgb_model = get_lightgbm_regressor(seed=seed + fold)
        lgb_model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )
        oof_lgb = lgb_model.predict(X_val)
        oof_predictions["lightgbm"][val_idx] = oof_lgb
        test_predictions["lightgbm"] += lgb_model.predict(X_test_raw) / num_folds

        # 2. XGBoost
        xgb_model = get_xgboost_regressor(seed=seed + fold)
        xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        oof_xgb = xgb_model.predict(X_val)
        oof_predictions["xgboost"][val_idx] = oof_xgb
        test_predictions["xgboost"] += xgb_model.predict(X_test_raw) / num_folds

        # 3. CatBoost
        cb_model = get_catboost_regressor(seed=seed + fold)
        cb_model.fit(
            X_train,
            y_train,
            eval_set=(X_val, y_val),
            early_stopping_rounds=50,
            verbose=False,
        )
        oof_cb = cb_model.predict(X_val)
        oof_predictions["catboost"][val_idx] = oof_cb
        test_predictions["catboost"] += cb_model.predict(X_test_raw) / num_folds

        # 4. ExtraTrees
        et_model = get_extratrees_regressor(seed=seed + fold)
        et_model.fit(X_train, y_train)
        oof_et = et_model.predict(X_val)
        oof_predictions["extratrees"][val_idx] = oof_et
        test_predictions["extratrees"] += et_model.predict(X_test_raw) / num_folds

        # 5. Deep Tabular Neural Network (SuperconductorGatedResNet)
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test_raw)

        train_dataset = TensorDataset(
            torch.tensor(X_tr_scaled, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
        )
        val_dataset = TensorDataset(
            torch.tensor(X_val_scaled, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.float32),
        )
        test_dataset = TensorDataset(torch.tensor(X_test_scaled, dtype=torch.float32))

        train_loader = DataLoader(
            train_dataset, batch_size=256, shuffle=True, drop_last=False
        )
        val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

        nn_model, criterion, optimizer, scheduler = create_neural_model_and_optimizer(
            input_dim=input_dim,
            lr=1.5e-3,
            weight_decay=1e-4,
            device=str(device),
        )

        best_val_loss = float("inf")
        best_state = None
        patience = 12
        patience_counter = 0
        epochs = 60

        for epoch in range(epochs):
            nn_model.train()
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                optimizer.zero_grad()
                pred = nn_model(batch_x)
                loss = criterion(pred, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(nn_model.parameters(), max_norm=1.0)
                optimizer.step()

            scheduler.step()

            nn_model.eval()
            val_preds_list = []
            with torch.no_grad():
                for batch_x, _ in val_loader:
                    batch_x = batch_x.to(device)
                    out = nn_model(batch_x)
                    val_preds_list.append(out.cpu().numpy().flatten())
            val_preds_epoch = np.concatenate(val_preds_list)
            val_rmse_epoch = root_mean_squared_error(y_val, val_preds_epoch)

            if val_rmse_epoch < best_val_loss:
                best_val_loss = val_rmse_epoch
                best_state = {
                    k: v.cpu().clone() for k, v in nn_model.state_dict().items()
                }
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        nn_model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        nn_model.eval()

        val_nn_preds = []
        with torch.no_grad():
            for batch_x, _ in val_loader:
                batch_x = batch_x.to(device)
                val_nn_preds.append(nn_model(batch_x).cpu().numpy().flatten())
        oof_nn = np.concatenate(val_nn_preds)
        oof_predictions["neural_net"][val_idx] = oof_nn

        test_nn_preds = []
        with torch.no_grad():
            for (batch_x,) in test_loader:
                batch_x = batch_x.to(device)
                test_nn_preds.append(nn_model(batch_x).cpu().numpy().flatten())
        test_predictions["neural_net"] += np.concatenate(test_nn_preds) / num_folds

        print(
            f"Fold {fold + 1}/{num_folds} RMSE | "
            f"LGB: {root_mean_squared_error(y_val, oof_lgb):.4f} | "
            f"XGB: {root_mean_squared_error(y_val, oof_xgb):.4f} | "
            f"CB: {root_mean_squared_error(y_val, oof_cb):.4f} | "
            f"ET: {root_mean_squared_error(y_val, oof_et):.4f} | "
            f"NN: {root_mean_squared_error(y_val, oof_nn):.4f}"
        )

    # Report individual full out-of-fold performance
    print("\n--- Out-of-Fold Model Validation Performance ---")
    for name in model_names:
        clipped_oof = np.clip(oof_predictions[name], a_min=0.0, a_max=None)
        score_m = root_mean_squared_error(y, clipped_oof)
        print(f"Model [{name.upper()}] Full OOF RMSE: {score_m:.4f}")

    # Constrained Ensemble Optimization (SLSQP over bounded non-negative weights)
    oof_matrix = np.column_stack([oof_predictions[name] for name in model_names])
    test_matrix = np.column_stack([test_predictions[name] for name in model_names])

    def loss_func(weights):
        pred = np.clip(oof_matrix @ weights, a_min=0.0, a_max=None)
        return root_mean_squared_error(y, pred)

    initial_weights = np.ones(len(model_names)) / len(model_names)
    bounds = [(0.0, 1.0) for _ in range(len(model_names))]
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

    opt_res = minimize(
        loss_func,
        initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    best_weights = (
        opt_res.x if opt_res.success else np.ones(len(model_names)) / len(model_names)
    )
    best_weights = np.maximum(0.0, best_weights)
    best_weights /= np.sum(best_weights)

    weight_str = ", ".join(
        [f"{name}: {weight:.4f}" for name, weight in zip(model_names, best_weights)]
    )
    print(f"\nOptimized Ensemble Weights: {weight_str}")

    # Generate final ensemble predictions with non-negativity physical constraint
    final_oof_preds = np.clip(oof_matrix @ best_weights, a_min=0.0, a_max=None)
    final_test_preds = np.clip(test_matrix @ best_weights, a_min=0.0, a_max=None)

    final_score = root_mean_squared_error(y, final_oof_preds)

    # Save formatted submission
    submission_path = "./submission/submission.csv"
    submission_df = pd.DataFrame(
        {
            "id": np.arange(len(final_test_preds), dtype=int),
            "critical_temp": final_test_preds,
        }
    )
    submission_df.to_csv(submission_path, index=False)

    print(f"Submission saved: {submission_df.shape} written to {submission_path}")
    print(
        f"Prediction summary - Min: {final_test_preds.min():.2f}, Median: {np.median(final_test_preds):.2f}, Mean: {final_test_preds.mean():.2f}, Max: {final_test_preds.max():.2f}"
    )

    print(f"Final Validation Score: {final_score}")


if __name__ == "__main__":
    main()