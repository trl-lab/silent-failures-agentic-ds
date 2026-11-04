import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from catboost import CatBoostRegressor
import lightgbm as lgb
import xgboost as xgb


def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ==============================================================================
# 1. DOMAIN-INFORMED FEATURE ENGINEERING
# ==============================================================================
def create_concrete_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies domain-specific civil engineering feature engineering for concrete compressive strength.
    Incorporates Abram's Law, binder hydration kinetics, aggregate packing ratios, and non-linear age dynamics.
    """
    feat_df = df.copy()
    eps = 1e-6

    # 1. Total Binder & Effective Binder Formulations
    feat_df["total_binder"] = (
        feat_df["cement"] + feat_df["blast_furnace_slag"] + feat_df["fly_ash"]
    )
    feat_df["cement_plus_slag"] = feat_df["cement"] + feat_df["blast_furnace_slag"]
    feat_df["cement_plus_flyash"] = feat_df["cement"] + feat_df["fly_ash"]

    # Effective binder reactivity factors (k-value concept)
    feat_df["effective_binder_standard"] = (
        feat_df["cement"]
        + 0.8 * feat_df["blast_furnace_slag"]
        + 0.5 * feat_df["fly_ash"]
    )
    feat_df["effective_binder_conservative"] = (
        feat_df["cement"]
        + 0.6 * feat_df["blast_furnace_slag"]
        + 0.3 * feat_df["fly_ash"]
    )

    # 2. Water-to-Binder & Abram's Law Ratios
    feat_df["water_to_cement"] = feat_df["water"] / (feat_df["cement"] + eps)
    feat_df["water_to_binder"] = feat_df["water"] / (feat_df["total_binder"] + eps)
    feat_df["water_to_effective_binder"] = feat_df["water"] / (
        feat_df["effective_binder_standard"] + eps
    )
    feat_df["binder_to_water"] = feat_df["total_binder"] / (feat_df["water"] + eps)
    feat_df["cement_to_water"] = feat_df["cement"] / (feat_df["water"] + eps)
    feat_df["inv_water_to_binder_sq"] = (feat_df["binder_to_water"]) ** 2
    feat_df["abrams_power_ratio"] = feat_df["total_binder"] / (
        (feat_df["water"] + eps) ** 1.5
    )

    # 3. Aggregate Proportions & Packing Densities
    feat_df["total_aggregate"] = feat_df["coarse_aggregate"] + feat_df["fine_aggregate"]
    feat_df["fine_to_total_aggregate"] = feat_df["fine_aggregate"] / (
        feat_df["total_aggregate"] + eps
    )
    feat_df["coarse_to_fine_aggregate"] = feat_df["coarse_aggregate"] / (
        feat_df["fine_aggregate"] + eps
    )
    feat_df["aggregate_to_binder"] = feat_df["total_aggregate"] / (
        feat_df["total_binder"] + eps
    )
    feat_df["coarse_to_binder"] = feat_df["coarse_aggregate"] / (
        feat_df["total_binder"] + eps
    )
    feat_df["fine_to_binder"] = feat_df["fine_aggregate"] / (
        feat_df["total_binder"] + eps
    )

    # 4. Total Mass / Volumetric Proportions
    feat_df["total_mass"] = (
        feat_df["total_binder"]
        + feat_df["water"]
        + feat_df["superplasticizer"]
        + feat_df["total_aggregate"]
    )
    feat_df["cement_mass_fraction"] = feat_df["cement"] / (feat_df["total_mass"] + eps)
    feat_df["slag_mass_fraction"] = feat_df["blast_furnace_slag"] / (
        feat_df["total_mass"] + eps
    )
    feat_df["flyash_mass_fraction"] = feat_df["fly_ash"] / (feat_df["total_mass"] + eps)
    feat_df["binder_mass_fraction"] = feat_df["total_binder"] / (
        feat_df["total_mass"] + eps
    )
    feat_df["water_mass_fraction"] = feat_df["water"] / (feat_df["total_mass"] + eps)
    feat_df["aggregate_mass_fraction"] = feat_df["total_aggregate"] / (
        feat_df["total_mass"] + eps
    )
    feat_df["water_to_solid_ratio"] = feat_df["water"] / (
        feat_df["total_mass"] - feat_df["water"] + eps
    )

    # Binder component fractions relative to total binder
    feat_df["cement_in_binder"] = feat_df["cement"] / (feat_df["total_binder"] + eps)
    feat_df["slag_in_binder"] = feat_df["blast_furnace_slag"] / (
        feat_df["total_binder"] + eps
    )
    feat_df["flyash_in_binder"] = feat_df["fly_ash"] / (feat_df["total_binder"] + eps)
    feat_df["scms_in_binder"] = (feat_df["blast_furnace_slag"] + feat_df["fly_ash"]) / (
        feat_df["total_binder"] + eps
    )

    # 5. Chemical Admixture / Superplasticizer Interactions
    feat_df["sp_to_binder"] = feat_df["superplasticizer"] / (
        feat_df["total_binder"] + eps
    )
    feat_df["sp_to_cement"] = feat_df["superplasticizer"] / (feat_df["cement"] + eps)
    feat_df["sp_to_water"] = feat_df["superplasticizer"] / (feat_df["water"] + eps)
    feat_df["sp_percentage"] = feat_df["sp_to_binder"] * 100.0

    # 6. Binary Composition Flags
    feat_df["has_slag"] = (feat_df["blast_furnace_slag"] > 0.0).astype(np.float32)
    feat_df["has_flyash"] = (feat_df["fly_ash"] > 0.0).astype(np.float32)
    feat_df["has_superplasticizer"] = (feat_df["superplasticizer"] > 0.0).astype(
        np.float32
    )
    feat_df["is_pure_cement_binder"] = (
        (feat_df["blast_furnace_slag"] == 0.0) & (feat_df["fly_ash"] == 0.0)
    ).astype(np.float32)
    feat_df["is_ternary_binder"] = (
        (feat_df["cement"] > 0.0)
        & (feat_df["blast_furnace_slag"] > 0.0)
        & (feat_df["fly_ash"] > 0.0)
    ).astype(np.float32)

    # 7. Hydration Kinetics & Age Dynamics
    feat_df["log_age"] = np.log1p(feat_df["age"])
    feat_df["log10_age"] = np.log10(feat_df["age"] + 1.0)
    feat_df["sqrt_age"] = np.sqrt(feat_df["age"])
    feat_df["inv_age"] = 1.0 / (feat_df["age"] + 1.0)
    feat_df["hyperbolic_age"] = feat_df["age"] / (feat_df["age"] + 28.0)

    feat_df["is_early_age"] = (feat_df["age"] <= 7).astype(np.float32)
    feat_df["is_standard_28d"] = (feat_df["age"] == 28).astype(np.float32)
    feat_df["is_mature_age"] = (feat_df["age"] >= 56).astype(np.float32)

    # 8. Kinetics-Composition Interaction Terms
    feat_df["log_age_x_cement"] = feat_df["log_age"] * feat_df["cement"]
    feat_df["log_age_x_binder"] = feat_df["log_age"] * feat_df["total_binder"]
    feat_df["log_age_x_binder_to_water"] = (
        feat_df["log_age"] * feat_df["binder_to_water"]
    )
    feat_df["log_age_x_water_to_binder"] = (
        feat_df["log_age"] * feat_df["water_to_binder"]
    )
    feat_df["log_age_x_effective_binder"] = (
        feat_df["log_age"] * feat_df["effective_binder_standard"]
    )
    feat_df["log_age_x_scms_fraction"] = feat_df["log_age"] * feat_df["scms_in_binder"]
    feat_df["age_x_flyash"] = feat_df["log_age"] * feat_df["fly_ash"]
    feat_df["age_x_slag"] = feat_df["log_age"] * feat_df["blast_furnace_slag"]
    feat_df["sp_x_water_to_binder"] = (
        feat_df["superplasticizer"] * feat_df["water_to_binder"]
    )

    # 9. ACI 209 Maturation Functions & Hydration Equations
    feat_df["aci209_age_factor"] = feat_df["age"] / (4.0 + 0.85 * feat_df["age"])
    feat_df["aci209_age_factor_rapid"] = feat_df["age"] / (2.3 + 0.92 * feat_df["age"])
    feat_df["aci209_x_cement"] = feat_df["aci209_age_factor"] * feat_df["cement"]
    feat_df["aci209_x_binder"] = feat_df["aci209_age_factor"] * feat_df["total_binder"]
    feat_df["aci209_x_binder_to_water"] = (
        feat_df["aci209_age_factor"] * feat_df["binder_to_water"]
    )

    # 10. Powers' Gel-Space Ratio Kinetics & Strength Potential
    alpha_28 = feat_df["age"] / (feat_df["age"] + 28.0)
    alpha_14 = feat_df["age"] / (feat_df["age"] + 14.0)
    alpha_7 = feat_df["age"] / (feat_df["age"] + 7.0)

    powers_x_cement = (0.68 * alpha_28) / (feat_df["water_to_cement"] + 0.32 * alpha_28 + eps)
    powers_x_binder = (0.68 * alpha_28) / (feat_df["water_to_binder"] + 0.32 * alpha_28 + eps)
    feat_df["powers_gel_space_ratio_cement"] = powers_x_cement
    feat_df["powers_gel_space_ratio_binder"] = powers_x_binder
    feat_df["powers_strength_potential_c3"] = powers_x_cement ** 3
    feat_df["powers_strength_potential_b3"] = powers_x_binder ** 3
    feat_df["powers_alpha14_b3"] = (
        (0.68 * alpha_14) / (feat_df["water_to_binder"] + 0.32 * alpha_14 + eps)
    ) ** 3
    feat_df["powers_alpha7_b3"] = (
        (0.68 * alpha_7) / (feat_df["water_to_binder"] + 0.32 * alpha_7 + eps)
    ) ** 3

    # 11. Abram's Exponential & Logarithmic Kinetics
    feat_df["abrams_log_c_w"] = np.log((feat_df["cement"] + 1.0) / (feat_df["water"] + 1.0))
    feat_df["abrams_log_b_w"] = np.log((feat_df["total_binder"] + 1.0) / (feat_df["water"] + 1.0))
    feat_df["abrams_exp_neg_wb"] = np.exp(-feat_df["water_to_binder"])
    feat_df["abrams_exp_neg_wc"] = np.exp(-feat_df["water_to_cement"])
    feat_df["abrams_ratio_kinetic"] = feat_df["log_age"] * feat_df["abrams_exp_neg_wb"]

    # 12. Bolomey / Feret Volumetric & De Larrard Packing Models
    vol_cement = feat_df["cement"] / 3.15
    vol_slag = feat_df["blast_furnace_slag"] / 2.90
    vol_flyash = feat_df["fly_ash"] / 2.30
    vol_water = feat_df["water"] / 1.00
    vol_sp = feat_df["superplasticizer"] / 1.10
    vol_coarse = feat_df["coarse_aggregate"] / 2.68
    vol_fine = feat_df["fine_aggregate"] / 2.62

    vol_binder = vol_cement + vol_slag + vol_flyash
    vol_paste = vol_binder + vol_water + vol_sp
    vol_mortar = vol_paste + vol_fine
    vol_agg = vol_coarse + vol_fine
    vol_total = vol_paste + vol_agg

    feat_df["feret_binder_ratio"] = vol_binder / (vol_binder + vol_water + eps)
    feat_df["feret_binder_ratio_sq"] = (feat_df["feret_binder_ratio"]) ** 2
    feat_df["feret_cement_ratio"] = vol_cement / (vol_cement + vol_water + eps)
    feat_df["feret_cement_ratio_sq"] = (feat_df["feret_cement_ratio"]) ** 2

    feat_df["bolomey_index"] = feat_df["cement"] / (feat_df["water"] + eps) - 0.5
    feat_df["bolomey_effective_index"] = (
        feat_df["effective_binder_standard"] / (feat_df["water"] + eps) - 0.5
    )
    feat_df["paste_to_agg_vol_ratio"] = vol_paste / (vol_agg + eps)
    feat_df["paste_volume_fraction"] = vol_paste / (vol_total + eps)
    feat_df["fine_to_paste_vol_ratio"] = vol_fine / (vol_paste + eps)
    feat_df["mortar_volume_fraction"] = vol_mortar / (vol_total + eps)
    feat_df["coarse_to_mortar_vol_ratio"] = vol_coarse / (vol_mortar + eps)
    feat_df["sand_ratio_volumetric"] = vol_fine / (vol_agg + eps)

    # De Larrard packing approximation & excess paste
    phi_coarse = 0.60
    phi_fine = 0.65
    feat_df["delarrard_packing_index"] = (
        vol_coarse * phi_coarse + vol_fine * phi_fine
    ) / (vol_agg + eps)
    feat_df["excess_paste_layer"] = (vol_paste - (1.0 - 0.62) * vol_agg) / (vol_agg + eps)

    # 13. Pozzolanic Late-Age & Slag Reaction Dynamics
    feat_df["flyash_late_activation"] = feat_df["fly_ash"] * (
        feat_df["age"] / (28.0 + feat_df["age"])
    )
    feat_df["slag_intermediate_maturity"] = feat_df["blast_furnace_slag"] * (
        feat_df["age"] / (14.0 + feat_df["age"])
    )

    dynamic_k_slag = 0.5 + 0.4 * (feat_df["age"] / (14.0 + feat_df["age"]))
    dynamic_k_flyash = 0.2 + 0.6 * (feat_df["age"] / (28.0 + feat_df["age"]))
    time_dependent_active_binder = (
        feat_df["cement"]
        + dynamic_k_slag * feat_df["blast_furnace_slag"]
        + dynamic_k_flyash * feat_df["fly_ash"]
    )
    feat_df["time_dependent_w_b_ratio"] = feat_df["water"] / (
        time_dependent_active_binder + eps
    )
    feat_df["time_dependent_b_w_ratio"] = time_dependent_active_binder / (
        feat_df["water"] + eps
    )

    return feat_df


# ==============================================================================
# 2. DEEP TABULAR RESIDUAL NETWORK ARCHITECTURE
# ==============================================================================
class TabularResidualBlock(nn.Module):
    """Residual block with LayerNorm, Mish activation, and Dropout."""

    def __init__(self, hidden_dim: int, dropout_rate: float = 0.15):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Mish(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Mish(),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class ConcreteTabularResNet(nn.Module):
    """Deep Tabular ResNet model tailored for concrete compressive strength regression."""

    def __init__(
        self,
        in_features: int,
        hidden_dim: int = 128,
        num_blocks: int = 3,
        dropout_rate: float = 0.15,
        out_features: int = 1,
    ):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Mish(),
            nn.Dropout(dropout_rate / 2.0),
        )

        self.res_blocks = nn.ModuleList(
            [
                TabularResidualBlock(hidden_dim, dropout_rate=dropout_rate)
                for _ in range(num_blocks)
            ]
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.Mish(),
            nn.Dropout(dropout_rate / 2.0),
            nn.Linear(hidden_dim // 2, out_features),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_in", nonlinearity="leaky_relu"
                )
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_layer(x)
        for block in self.res_blocks:
            h = block(h)
        out = self.head(h)
        return out.squeeze(-1)


# ==============================================================================
# 3. PIPELINE EXECUTION & ENSEMBLE TRAINING
# ==============================================================================
def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("./submission", exist_ok=True)

    train_path = "./input/train.parquet"
    test_path = "./input/test.parquet"

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)
    target_col = "strength"

    # Feature Engineering
    train_feat = create_concrete_features(train_df)
    test_feat = create_concrete_features(test_df)

    # 10-Fold Stratified Cross-Validation on target strength quantiles
    num_folds = 10
    num_bins = 10
    target_bins = pd.qcut(
        train_feat[target_col], q=num_bins, labels=False, duplicates="drop"
    )

    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
    train_feat["fold"] = -1
    for fold, (_, val_idx) in enumerate(skf.split(train_feat, target_bins)):
        train_feat.loc[val_idx, "fold"] = fold

    feature_cols = [c for c in train_feat.columns if c not in [target_col, "fold"]]
    num_features = len(feature_cols)

    oof_preds = {
        "lgbm": np.zeros(len(train_feat)),
        "xgb": np.zeros(len(train_feat)),
        "catboost": np.zeros(len(train_feat)),
        "extratrees": np.zeros(len(train_feat)),
        "randomforest": np.zeros(len(train_feat)),
        "resnet": np.zeros(len(train_feat)),
    }

    test_preds = {
        "lgbm": np.zeros(len(test_feat)),
        "xgb": np.zeros(len(test_feat)),
        "catboost": np.zeros(len(test_feat)),
        "extratrees": np.zeros(len(test_feat)),
        "randomforest": np.zeros(len(test_feat)),
        "resnet": np.zeros(len(test_feat)),
    }

    for fold in range(num_folds):
        train_idx = train_feat["fold"] != fold
        val_idx = train_feat["fold"] == fold

        X_train, y_train = (
            train_feat.loc[train_idx, feature_cols],
            train_feat.loc[train_idx, target_col],
        )
        X_val, y_val = (
            train_feat.loc[val_idx, feature_cols],
            train_feat.loc[val_idx, target_col],
        )
        X_test = test_feat[feature_cols]

        # 1. LightGBM
        lgb_model = lgb.LGBMRegressor(
            objective="regression",
            metric="rmse",
            boosting_type="gbdt",
            n_estimators=2000,
            learning_rate=0.03,
            num_leaves=31,
            max_depth=6,
            min_child_samples=15,
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42 + fold,
            n_jobs=-1,
            verbose=-1,
        )
        lgb_model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )
        oof_preds["lgbm"][val_idx] = lgb_model.predict(X_val)
        test_preds["lgbm"] += lgb_model.predict(X_test) / num_folds

        # 2. XGBoost
        xgb_model = xgb.XGBRegressor(
            objective="reg:squarederror",
            eval_metric="rmse",
            n_estimators=2000,
            learning_rate=0.03,
            max_depth=6,
            min_child_weight=2.0,
            subsample=0.85,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42 + fold,
            n_jobs=-1,
            tree_method="hist",
            early_stopping_rounds=50,
        )
        xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        oof_preds["xgb"][val_idx] = xgb_model.predict(X_val)
        test_preds["xgb"] += xgb_model.predict(X_test) / num_folds

        # 3. CatBoost
        cat_model = CatBoostRegressor(
            iterations=2000,
            learning_rate=0.03,
            depth=6,
            l2_leaf_reg=3.0,
            random_seed=42 + fold,
            verbose=False,
            early_stopping_rounds=50,
        )
        cat_model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
        oof_preds["catboost"][val_idx] = cat_model.predict(X_val)
        test_preds["catboost"] += cat_model.predict(X_test) / num_folds

        # 4. ExtraTrees
        et_model = ExtraTreesRegressor(
            n_estimators=600,
            max_depth=16,
            min_samples_split=4,
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=42 + fold,
            n_jobs=-1,
        )
        et_model.fit(X_train, y_train)
        oof_preds["extratrees"][val_idx] = et_model.predict(X_val)
        test_preds["extratrees"] += et_model.predict(X_test) / num_folds

        # 5. RandomForest
        rf_model = RandomForestRegressor(
            n_estimators=500,
            max_depth=16,
            min_samples_split=4,
            min_samples_leaf=2,
            max_features=0.8,
            random_state=42 + fold,
            n_jobs=-1,
        )
        rf_model.fit(X_train, y_train)
        oof_preds["randomforest"][val_idx] = rf_model.predict(X_val)
        test_preds["randomforest"] += rf_model.predict(X_test) / num_folds

        # 6. Deep Tabular ResNet
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)
        X_test_s = scaler.transform(X_test)

        train_ds = TensorDataset(
            torch.tensor(X_train_s, dtype=torch.float32),
            torch.tensor(y_train.values, dtype=torch.float32),
        )
        train_loader = DataLoader(
            train_ds, batch_size=64, shuffle=True, drop_last=False
        )

        resnet = ConcreteTabularResNet(
            in_features=num_features,
            hidden_dim=128,
            num_blocks=3,
            dropout_rate=0.15,
        ).to(device)

        optimizer = torch.optim.AdamW(
            resnet.parameters(), lr=2e-3, weight_decay=1e-4, eps=1e-8
        )
        epochs = 180
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-5
        )
        criterion = nn.MSELoss()

        best_val_rmse = float("inf")
        best_resnet_weights = None
        patience = 35
        patience_counter = 0

        X_val_tensor = torch.tensor(X_val_s, dtype=torch.float32).to(device)
        y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).to(device)

        for epoch in range(epochs):
            resnet.train()
            for bx, by in train_loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                pred = resnet(bx)
                loss = criterion(pred, by)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(resnet.parameters(), max_norm=2.0)
                optimizer.step()

            scheduler.step()

            resnet.eval()
            with torch.no_grad():
                val_preds_nn = resnet(X_val_tensor)
                val_rmse_nn = torch.sqrt(criterion(val_preds_nn, y_val_tensor)).item()

            if val_rmse_nn < best_val_rmse:
                best_val_rmse = val_rmse_nn
                best_resnet_weights = {
                    k: v.cpu() for k, v in resnet.state_dict().items()
                }
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        resnet.load_state_dict(best_resnet_weights)
        resnet.eval()
        with torch.no_grad():
            nn_val_pred = resnet(X_val_tensor).cpu().numpy()
            X_test_tensor = torch.tensor(X_test_s, dtype=torch.float32).to(device)
            nn_test_pred = resnet(X_test_tensor).cpu().numpy()

        oof_preds["resnet"][val_idx] = nn_val_pred
        test_preds["resnet"] += nn_test_pred / num_folds

    y_true = train_feat[target_col].values
    model_names = ["lgbm", "xgb", "catboost", "extratrees", "randomforest", "resnet"]

    oof_matrix = np.column_stack([oof_preds[m] for m in model_names])
    test_matrix = np.column_stack([test_preds[m] for m in model_names])

    # Feature-Informed Meta-Stacking
    physical_meta_cols = [
        "log_age",
        "water_to_binder",
        "total_binder",
        "powers_strength_potential_b3",
    ]
    meta_scaler = StandardScaler()
    train_phys_s = meta_scaler.fit_transform(train_feat[physical_meta_cols])
    test_phys_s = meta_scaler.transform(test_feat[physical_meta_cols])

    X_meta_train = np.hstack([oof_matrix, train_phys_s])
    X_meta_test = np.hstack([test_matrix, test_phys_s])

    meta_learner = RidgeCV(alphas=np.logspace(-3, 4, 100), cv=10)
    meta_learner.fit(X_meta_train, y_true)

    final_oof_predictions = meta_learner.predict(X_meta_train)
    final_score = root_mean_squared_error(y_true, final_oof_predictions)
    final_test_predictions = meta_learner.predict(X_meta_test)

    # Save final verified submission
    submission_path = "./submission/submission.csv"
    sub = pd.DataFrame(
        {
            "id": np.arange(len(final_test_predictions)),
            "strength": final_test_predictions,
        }
    )

    assert len(sub) == len(test_df), "Submission row count mismatch"
    assert not sub["strength"].isnull().any(), "Submission contains null values"
    assert not np.isinf(sub["strength"].values).any(), "Submission contains inf values"

    sub.to_csv(submission_path, index=False)
    print(f"Final Validation Score: {final_score}")


if __name__ == "__main__":
    main()
