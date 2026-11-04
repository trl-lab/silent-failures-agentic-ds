import os
import random
import numpy as np
import pandas as pd
import scipy.optimize as opt
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import lightgbm as lgb
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

# =========================================================================
# 0. Global Setup & Seed Initializations
# =========================================================================
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)


# =========================================================================
# 1. Domain-Specific Crystallographic Feature Engineering
# =========================================================================
def engineer_crystal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes physical, geometric, and chemical descriptors for
    transparent conducting sesquioxides (Al_x Ga_y In_z)_2 O_3.
    """
    feats = pd.DataFrame(index=df.index)

    if "id" in df.columns:
        feats["id"] = df["id"]

    def _get_col(candidates):
        for cand in candidates:
            if cand in df.columns:
                return df[cand]
        for cand in candidates:
            for col in df.columns:
                if cand.lower() == col.lower() or cand.lower() in col.lower():
                    return df[col]
        raise KeyError(f"None of {candidates} found in columns: {list(df.columns)}")

    # --- Composition Fractions ---
    al = _get_col(["percent_atom_al", "al"]).astype(np.float64)
    ga = _get_col(["percent_atom_ga", "ga"]).astype(np.float64)
    in_ = _get_col(["percent_atom_in", "in"]).astype(np.float64)

    feats["percent_atom_al"] = al
    feats["percent_atom_ga"] = ga
    feats["percent_atom_in"] = in_

    # Stoichiometry interaction terms
    feats["al_ga_prod"] = al * ga
    feats["al_in_prod"] = al * in_
    feats["ga_in_prod"] = ga * in_
    feats["al_ga_in_prod"] = al * ga * in_

    feats["al_ga_diff"] = (al - ga).abs()
    feats["al_in_diff"] = (al - in_).abs()
    feats["ga_in_diff"] = (ga - in_).abs()

    # Ideal configurational entropy of mixing: S_mix = -R * sum(x_i * ln(x_i))
    eps = 1e-12
    s_mix = -(
        al * np.log(np.maximum(al, eps))
        + ga * np.log(np.maximum(ga, eps))
        + in_ * np.log(np.maximum(in_, eps))
    )
    feats["entropy_of_mixing"] = np.where(s_mix < 1e-6, 0.0, s_mix)

    # Number of cations present (> 1% concentration)
    feats["num_cation_species"] = (
        (al > 0.01).astype(int) + (ga > 0.01).astype(int) + (in_ > 0.01).astype(int)
    )
    feats["is_binary"] = (feats["num_cation_species"] == 1).astype(int)
    feats["is_ternary"] = (feats["num_cation_species"] == 2).astype(int)
    feats["is_quaternary"] = (feats["num_cation_species"] == 3).astype(int)

    # --- Elemental & Chemical Properties ---
    # Shannon 6-fold ionic radii (Å): Al3+=0.535, Ga3+=0.620, In3+=0.800, O2-=1.400
    r_al, r_ga, r_in, r_o = 0.535, 0.620, 0.800, 1.400
    mean_r_cat = al * r_al + ga * r_ga + in_ * r_in
    feats["mean_cation_radius"] = mean_r_cat
    feats["radius_ratio_cat_anion"] = mean_r_cat / r_o

    # Cation size variance (lattice strain proxy)
    feats["cation_radius_variance"] = (
        al * (r_al - mean_r_cat) ** 2
        + ga * (r_ga - mean_r_cat) ** 2
        + in_ * (r_in - mean_r_cat) ** 2
    )
    feats["cation_radius_std"] = np.sqrt(feats["cation_radius_variance"])

    # Pauling Electronegativities: Al=1.61, Ga=1.81, In=1.78, O=3.44
    chi_al, chi_ga, chi_in, chi_o = 1.61, 1.81, 1.78, 3.44
    mean_chi_cat = al * chi_al + ga * chi_ga + in_ * chi_in
    feats["mean_cation_electronegativity"] = mean_chi_cat
    feats["electronegativity_diff_anion"] = chi_o - mean_chi_cat
    feats["cation_electronegativity_variance"] = (
        al * (chi_al - mean_chi_cat) ** 2
        + ga * (chi_ga - mean_chi_cat) ** 2
        + in_ * (chi_in - mean_chi_cat) ** 2
    )

    # Atomic Masses (g/mol): Al=26.9815, Ga=69.723, In=114.818, O=15.9994
    m_al, m_ga, m_in, m_o = 26.9815, 69.723, 114.818, 15.9994
    mean_m_cat = al * m_al + ga * m_ga + in_ * m_in
    feats["mean_cation_mass"] = mean_m_cat
    mw_fu = 2.0 * mean_m_cat + 3.0 * m_o
    feats["molar_mass_formula_unit"] = mw_fu

    # Sum of 1st-3rd Ionization Energies (eV): Al=53.263, Ga=57.223, In=52.686
    ie_al, ie_ga, ie_in = 53.263, 57.223, 52.686
    feats["mean_cation_ionization_energy"] = al * ie_al + ga * ie_ga + in_ * ie_in

    # d-electron count per cation: Al=0, Ga=10, In=10
    feats["mean_d_electrons"] = al * 0.0 + ga * 10.0 + in_ * 10.0

    # --- Lattice Geometry & Unit Cell Physics ---
    a = _get_col(["lattice_vector_1_ang", "lattice_vector_1"]).astype(np.float64)
    b = _get_col(["lattice_vector_2_ang", "lattice_vector_2"]).astype(np.float64)
    c = _get_col(["lattice_vector_3_ang", "lattice_vector_3"]).astype(np.float64)

    deg_alpha = _get_col(["lattice_angle_alpha_degree", "lattice_angle_alpha_deg", "alpha"]).astype(np.float64)
    deg_beta = _get_col(["lattice_angle_beta_degree", "lattice_angle_beta_deg", "beta"]).astype(np.float64)
    deg_gamma = _get_col(["lattice_angle_gamma_degree", "lattice_angle_gamma_deg", "gamma"]).astype(np.float64)

    alpha = np.radians(deg_alpha)
    beta = np.radians(deg_beta)
    gamma = np.radians(deg_gamma)

    feats["lattice_vector_1_ang"] = a
    feats["lattice_vector_2_ang"] = b
    feats["lattice_vector_3_ang"] = c
    feats["lattice_angle_alpha_deg"] = deg_alpha
    feats["lattice_angle_beta_deg"] = deg_beta
    feats["lattice_angle_gamma_deg"] = deg_gamma

    # Exact unit cell volume: V = a*b*c * sqrt(1 - cos^2 a - cos^2 b - cos^2 g + 2*cos a*cos b*cos g)
    cos_a, cos_b, cos_g = np.cos(alpha), np.cos(beta), np.cos(gamma)
    vol_term = 1.0 - cos_a**2 - cos_b**2 - cos_g**2 + 2.0 * cos_a * cos_b * cos_g
    unit_cell_vol = a * b * c * np.sqrt(np.maximum(vol_term, 1e-8))
    feats["unit_cell_volume"] = unit_cell_vol

    # Number of atoms and formula units
    n_atoms = _get_col(["number_of_total_atoms", "total_atoms", "n_atoms"]).astype(np.float64)
    feats["number_of_total_atoms"] = n_atoms
    z_fu = n_atoms / 5.0
    feats["num_formula_units"] = z_fu

    # Density Descriptors
    feats["atomic_volume"] = unit_cell_vol / n_atoms
    feats["atomic_density"] = n_atoms / unit_cell_vol
    feats["mass_density"] = (z_fu * mw_fu * 1.66053907) / unit_cell_vol

    # Theoretical sphere volumes of ions
    vol_cations_sphere = (
        2.0 * z_fu * (4.0 / 3.0 * np.pi) * (al * r_al**3 + ga * r_ga**3 + in_ * r_in**3)
    )
    vol_anions_sphere = 3.0 * z_fu * (4.0 / 3.0 * np.pi) * (r_o**3)
    total_sphere_vol = vol_cations_sphere + vol_anions_sphere
    feats["packing_fraction"] = total_sphere_vol / unit_cell_vol
    feats["free_volume_per_atom"] = (unit_cell_vol - total_sphere_vol) / n_atoms

    # Aspect ratios and anisotropic distortion
    feats["lattice_ratio_a_b"] = a / b
    feats["lattice_ratio_b_c"] = b / c
    feats["lattice_ratio_c_a"] = c / a
    feats["lattice_ratio_max_min"] = np.maximum.reduce([a, b, c]) / np.minimum.reduce(
        [a, b, c]
    )

    feats["angle_ortho_deviation"] = (
        (deg_alpha - 90.0).abs() + (deg_beta - 90.0).abs() + (deg_gamma - 90.0).abs()
    )

    # --- Space Group & Symmetry Descriptors ---
    sg = _get_col(["spacegroup", "space_group", "sg"]).astype(int)
    feats["spacegroup"] = sg

    feats["is_monoclinic"] = (sg == 12).astype(int)
    feats["is_orthorhombic"] = (sg == 33).astype(int)
    feats["is_trigonal"] = (sg == 167).astype(int)
    feats["is_hexagonal"] = (sg == 194).astype(int)
    feats["is_cubic"] = ((sg == 206) | (sg == 227)).astype(int)
    feats["is_centrosymmetric"] = (
        (sg == 12) | (sg == 167) | (sg == 194) | (sg == 206) | (sg == 227)
    ).astype(int)

    sg_counts = {12: 440, 33: 440, 167: 400, 194: 400, 206: 360, 227: 360}
    feats["spacegroup_frequency"] = sg.map(sg_counts).fillna(100).astype(np.float64)

    return feats


# =========================================================================
# 2. Metric Definition & Model Architectures
# =========================================================================
def calc_rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes Root Mean Squared Logarithmic Error (RMSLE)."""
    y_true = np.clip(y_true, a_min=0.0, a_max=None)
    y_pred = np.clip(y_pred, a_min=0.0, a_max=None)
    return float(root_mean_squared_error(np.log1p(y_true), np.log1p(y_pred)))


class ResidualBlock(nn.Module):
    """Residual Block with LayerNorm, SiLU, and Dropout for tabular features."""

    def __init__(self, hidden_dim: int, dropout_rate: float = 0.15):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class CrystalResNet(nn.Module):
    """Deep Tabular Residual Network tailored for crystallographic features."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = [256, 128, 64],
        num_res_blocks: int = 2,
        dropout_rate: float = 0.15,
    ):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
        )

        res_layers = []
        for _ in range(num_res_blocks):
            res_layers.append(ResidualBlock(hidden_dims[0], dropout_rate=dropout_rate))
        self.res_backbone = nn.Sequential(*res_layers)

        transition_layers = []
        prev_dim = hidden_dims[0]
        for h_dim in hidden_dims[1:]:
            transition_layers.extend(
                [
                    nn.Linear(prev_dim, h_dim),
                    nn.LayerNorm(h_dim),
                    nn.SiLU(),
                    nn.Dropout(dropout_rate),
                ]
            )
            prev_dim = h_dim
        self.transition = nn.Sequential(*transition_layers)

        self.head = nn.Linear(prev_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(x)
        x = self.res_backbone(x)
        x = self.transition(x)
        return self.head(x).squeeze(-1)


def build_lgb_model(seed: int = 42) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        n_estimators=1500,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=6,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=1.0,
        min_child_samples=15,
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )


def build_catboost_model(seed: int = 42) -> CatBoostRegressor:
    return CatBoostRegressor(
        iterations=1500,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=3.0,
        subsample=0.85,
        random_seed=seed,
        loss_function="RMSE",
        eval_metric="RMSE",
        verbose=0,
    )


def build_xgb_model(seed: int = 42) -> XGBRegressor:
    return XGBRegressor(
        n_estimators=1500,
        learning_rate=0.03,
        max_depth=5,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=1.0,
        early_stopping_rounds=50,
        random_state=seed,
        n_jobs=-1,
        verbosity=0,
    )


# =========================================================================
# 3. Data Ingestion & Cross-Validation Pipeline
# =========================================================================
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "formation_energy_ev_natom"
y_train = train_df[target_col].values.astype(np.float64)

X_train_df = engineer_crystal_features(train_df)
X_test_df = engineer_crystal_features(test_df)

feature_cols = [c for c in X_train_df.columns if c not in ["id", target_col]]

X_tr_all = X_train_df[feature_cols].values.astype(np.float32)
X_te_all = X_test_df[feature_cols].values.astype(np.float32)
y_tr_all = y_train.astype(np.float32)
y_tr_log = np.log1p(y_tr_all)

n_samples = len(X_tr_all)
n_test = len(X_te_all)
n_splits = 5

# Stratified Fold Assignment based on Space Group & Target Quantile Bins
energy_bins = pd.qcut(y_train, q=10, labels=False, duplicates="drop")
strat_key = X_train_df["spacegroup"].astype(str) + "_" + energy_bins.astype(str)
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
fold_assignments = np.zeros(n_samples, dtype=int)
for fold_idx, (_, val_idx) in enumerate(skf.split(X_tr_all, strat_key)):
    fold_assignments[val_idx] = fold_idx

# Prediction storage
oof_lgb = np.zeros(n_samples, dtype=np.float64)
oof_cat = np.zeros(n_samples, dtype=np.float64)
oof_xgb = np.zeros(n_samples, dtype=np.float64)
oof_nn = np.zeros(n_samples, dtype=np.float64)

test_preds_lgb = np.zeros(n_test, dtype=np.float64)
test_preds_cat = np.zeros(n_test, dtype=np.float64)
test_preds_xgb = np.zeros(n_test, dtype=np.float64)
test_preds_nn = np.zeros(n_test, dtype=np.float64)

# =========================================================================
# 4. Model Training & Cross-Validation Execution
# =========================================================================
for fold in range(n_splits):
    tr_idx = np.where(fold_assignments != fold)[0]
    va_idx = np.where(fold_assignments == fold)[0]

    X_tr, y_tr, y_tr_l = X_tr_all[tr_idx], y_tr_all[tr_idx], y_tr_log[tr_idx]
    X_va, y_va, y_va_l = X_tr_all[va_idx], y_tr_all[va_idx], y_tr_log[va_idx]

    # --- A. LightGBM ---
    lgb_model = build_lgb_model(seed=SEED + fold)
    lgb_model.fit(
        X_tr,
        y_tr_l,
        eval_set=[(X_va, y_va_l)],
        callbacks=[
            lgb.early_stopping(50, verbose=False),
            lgb.log_evaluation(-1),
        ],
    )
    val_lgb_log = lgb_model.predict(X_va)
    oof_lgb[va_idx] = np.clip(np.expm1(val_lgb_log), 0.0, None)
    test_preds_lgb += (
        np.clip(np.expm1(lgb_model.predict(X_te_all)), 0.0, None) / n_splits
    )

    # --- B. CatBoost ---
    cat_model = build_catboost_model(seed=SEED + fold)
    cat_model.fit(
        X_tr,
        y_tr_l,
        eval_set=[(X_va, y_va_l)],
        early_stopping_rounds=50,
        verbose=False,
    )
    val_cat_log = cat_model.predict(X_va)
    oof_cat[va_idx] = np.clip(np.expm1(val_cat_log), 0.0, None)
    test_preds_cat += (
        np.clip(np.expm1(cat_model.predict(X_te_all)), 0.0, None) / n_splits
    )

    # --- C. XGBoost ---
    xgb_model = build_xgb_model(seed=SEED + fold)
    xgb_model.fit(X_tr, y_tr_l, eval_set=[(X_va, y_va_l)], verbose=False)
    val_xgb_log = xgb_model.predict(X_va)
    oof_xgb[va_idx] = np.clip(np.expm1(val_xgb_log), 0.0, None)
    test_preds_xgb += (
        np.clip(np.expm1(xgb_model.predict(X_te_all)), 0.0, None) / n_splits
    )

    # --- D. CrystalResNet (PyTorch) ---
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_va_scaled = scaler.transform(X_va)
    X_te_scaled = scaler.transform(X_te_all)

    torch.manual_seed(SEED + fold)
    nn_model = CrystalResNet(
        input_dim=X_tr.shape[1],
        hidden_dims=[256, 128, 64],
        num_res_blocks=2,
        dropout_rate=0.15,
    ).to(device)

    num_epochs = 120
    optimizer = AdamW(nn_model.parameters(), lr=1.5e-3, weight_decay=1e-4, eps=1e-8)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    criterion = nn.MSELoss()

    tr_dataset = TensorDataset(
        torch.tensor(X_tr_scaled, dtype=torch.float32),
        torch.tensor(y_tr_l, dtype=torch.float32),
    )
    tr_loader = DataLoader(
        tr_dataset, batch_size=32, shuffle=True, drop_last=False, num_workers=2
    )

    val_x_t = torch.tensor(X_va_scaled, dtype=torch.float32).to(device)
    val_y_t = torch.tensor(y_va_l, dtype=torch.float32).to(device)
    test_x_t = torch.tensor(X_te_scaled, dtype=torch.float32).to(device)

    best_val_loss = float("inf")
    best_weights = None

    for epoch in range(num_epochs):
        nn_model.train()
        for bx, by in tr_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = nn_model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
        scheduler.step()

        nn_model.eval()
        with torch.no_grad():
            val_out = nn_model(val_x_t)
            val_loss = criterion(val_out, val_y_t).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = {
                k: v.cpu().clone() for k, v in nn_model.state_dict().items()
            }

    nn_model.load_state_dict(best_weights)
    nn_model.eval()
    with torch.no_grad():
        val_nn_log = nn_model(val_x_t).cpu().numpy()
        te_nn_log = nn_model(test_x_t).cpu().numpy()

    oof_nn[va_idx] = np.clip(np.expm1(val_nn_log), 0.0, None)
    test_preds_nn += np.clip(np.expm1(te_nn_log), 0.0, None) / n_splits

    fold_lgb = calc_rmsle(y_va, oof_lgb[va_idx])
    fold_cat = calc_rmsle(y_va, oof_cat[va_idx])
    fold_xgb = calc_rmsle(y_va, oof_xgb[va_idx])
    fold_nn = calc_rmsle(y_va, oof_nn[va_idx])
    print(
        f"Fold {fold} | LGB RMSLE: {fold_lgb:.5f} | CAT RMSLE: {fold_cat:.5f} | "
        f"XGB RMSLE: {fold_xgb:.5f} | NN RMSLE: {fold_nn:.5f}"
    )


# =========================================================================
# 5. Optimal Blending & Submission Generation
# =========================================================================
def blend_objective(weights):
    w = np.array(weights)
    w = w / np.sum(w)
    pred = w[0] * oof_lgb + w[1] * oof_cat + w[2] * oof_xgb + w[3] * oof_nn
    return calc_rmsle(y_tr_all, pred)


init_weights = [0.35, 0.35, 0.20, 0.10]
bounds = [(0.0, 1.0)] * 4
constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

opt_res = opt.minimize(
    blend_objective,
    init_weights,
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
)
opt_w = (
    opt_res.x / np.sum(opt_res.x)
    if opt_res.success
    else np.array([0.35, 0.35, 0.20, 0.10])
)

blended_oof = (
    opt_w[0] * oof_lgb + opt_w[1] * oof_cat + opt_w[2] * oof_xgb + opt_w[3] * oof_nn
)
final_val_score = calc_rmsle(y_tr_all, blended_oof)

blended_test_preds = (
    opt_w[0] * test_preds_lgb
    + opt_w[1] * test_preds_cat
    + opt_w[2] * test_preds_xgb
    + opt_w[3] * test_preds_nn
)
blended_test_preds = np.clip(blended_test_preds, 0.0, None)

if "id" in test_df.columns:
    test_ids = test_df["id"].values
else:
    test_ids = np.arange(len(blended_test_preds), dtype=int)

submission_df = pd.DataFrame(
    {"id": test_ids, "formation_energy_ev_natom": blended_test_preds}
)
submission_df.to_csv("./submission/submission.csv", index=False)

print(f"Final Validation Score: {final_val_score}")
