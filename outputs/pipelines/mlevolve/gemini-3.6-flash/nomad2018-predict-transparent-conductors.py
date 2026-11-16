import os
import math
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Set seed for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Ensure required directories exist
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# 1. Load Data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "formation_energy_ev_natom"
id_col = "id"

test_ids = test_df[id_col].values


def extract_physical_features(df):
    """
    Extract physics-based domain features for transparent conductors:
    - Exact 3D unit cell volume & volume per atom
    - Shannon ionic/atomic radius expectations and variance (lattice strain proxy)
    - Electronegativity mean and variance (chemical bond polarity)
    - Mass density and packing efficiency proxies
    - Crystallographic aspect ratios and angular projections
    - Spacegroup symmetry properties
    """
    data = df.copy()

    al_col = "percent_atom_al" if "percent_atom_al" in data.columns else None
    ga_col = "percent_atom_ga" if "percent_atom_ga" in data.columns else None
    in_col = "percent_atom_in" if "percent_atom_in" in data.columns else None

    a_col = "lattice_vector_1_ang"
    b_col = "lattice_vector_2_ang"
    c_col = "lattice_vector_3_ang"

    alpha_col = "lattice_angle_alpha_degree"
    beta_col = "lattice_angle_beta_degree"
    gamma_col = "lattice_angle_gamma_degree"

    atoms_col = "number_of_total_atoms"
    sg_col = "spacegroup"

    x_al = data[al_col].values if al_col else np.zeros(len(data))
    x_ga = data[ga_col].values if ga_col else np.zeros(len(data))
    x_in = data[in_col].values if in_col else np.zeros(len(data))

    r_al, r_ga, r_in, r_o = 0.535, 0.620, 0.800, 0.600
    e_al, e_ga, e_in, e_o = 1.61, 1.81, 1.78, 3.44
    m_al, m_ga, m_in, m_o = 26.9815, 69.723, 114.818, 15.999
    z_al, z_ga, z_in = 13, 31, 49

    avg_radius = x_al * r_al + x_ga * r_ga + x_in * r_in
    radius_var = (
        x_al * (r_al - avg_radius) ** 2
        + x_ga * (r_ga - avg_radius) ** 2
        + x_in * (r_in - avg_radius) ** 2
    )

    avg_electroneg = x_al * e_al + x_ga * e_ga + x_in * e_in
    electroneg_var = (
        x_al * (e_al - avg_electroneg) ** 2
        + x_ga * (e_ga - avg_electroneg) ** 2
        + x_in * (e_in - avg_electroneg) ** 2
    )
    electroneg_diff_o = e_o - avg_electroneg

    avg_mass = x_al * m_al + x_ga * m_ga + x_in * m_in
    avg_z = x_al * z_al + x_ga * z_ga + x_in * z_in

    al_ga_ratio = x_al / (x_ga + 1e-5)
    in_ga_ratio = x_in / (x_ga + 1e-5)
    in_al_ratio = x_in / (x_al + 1e-5)

    data["avg_cation_radius"] = avg_radius
    data["cation_radius_var"] = radius_var
    data["avg_electronegativity"] = avg_electroneg
    data["electronegativity_var"] = electroneg_var
    data["electronegativity_diff_O"] = electroneg_diff_o
    data["avg_cation_mass"] = avg_mass
    data["avg_cation_z"] = avg_z

    data["al_ga_ratio"] = al_ga_ratio
    data["in_ga_ratio"] = in_ga_ratio
    data["in_al_ratio"] = in_al_ratio

    data["al_ga_prod"] = x_al * x_ga
    data["al_in_prod"] = x_al * x_in
    data["ga_in_prod"] = x_ga * x_in

    misfit_radius = 2.0 * (
        x_al * x_ga * (r_al - r_ga) ** 2
        + x_al * x_in * (r_al - r_in) ** 2
        + x_ga * x_in * (r_ga - r_in) ** 2
    )
    misfit_electroneg = 2.0 * (
        x_al * x_ga * (e_al - e_ga) ** 2
        + x_al * x_in * (e_al - e_in) ** 2
        + x_ga * x_in * (e_ga - e_in) ** 2
    )
    goldschmidt_tf = (avg_radius + r_o) / (np.sqrt(2.0) * r_o)

    data["misfit_enthalpy_radius"] = misfit_radius
    data["misfit_enthalpy_electroneg"] = misfit_electroneg
    data["goldschmidt_tolerance_factor"] = goldschmidt_tf

    a = data[a_col].values
    b = data[b_col].values
    c = data[c_col].values

    alpha_rad = np.radians(data[alpha_col].values)
    beta_rad = np.radians(data[beta_col].values)
    gamma_rad = np.radians(data[gamma_col].values)

    cos_a, cos_b, cos_g = np.cos(alpha_rad), np.cos(beta_rad), np.cos(gamma_rad)
    sin_a, sin_b, sin_g = np.sin(alpha_rad), np.sin(beta_rad), np.sin(gamma_rad)

    n_samples = len(data)
    G = np.zeros((n_samples, 3, 3))
    G[:, 0, 0] = a**2
    G[:, 1, 1] = b**2
    G[:, 2, 2] = c**2
    G[:, 0, 1] = G[:, 1, 0] = a * b * cos_g
    G[:, 0, 2] = G[:, 2, 0] = a * c * cos_b
    G[:, 1, 2] = G[:, 2, 1] = b * c * cos_a

    eigvals = np.linalg.eigvalsh(G)

    data["metric_tensor_g11"] = G[:, 0, 0]
    data["metric_tensor_g22"] = G[:, 1, 1]
    data["metric_tensor_g33"] = G[:, 2, 2]
    data["metric_tensor_g12"] = G[:, 0, 1]
    data["metric_tensor_g13"] = G[:, 0, 2]
    data["metric_tensor_g23"] = G[:, 1, 2]
    data["metric_tensor_trace"] = np.trace(G, axis1=1, axis2=2)
    data["metric_tensor_det"] = np.linalg.det(G)
    data["metric_tensor_min_eig"] = eigvals[:, 0]
    data["metric_tensor_max_eig"] = eigvals[:, 2]

    vol_term = 1.0 - cos_a**2 - cos_b**2 - cos_g**2 + 2.0 * cos_a * cos_b * cos_g
    vol_term = np.maximum(vol_term, 1e-8)
    unit_cell_vol = a * b * c * np.sqrt(vol_term)
    data["unit_cell_volume"] = unit_cell_vol

    n_atoms = data[atoms_col].values
    vol_per_atom = unit_cell_vol / np.maximum(n_atoms, 1.0)
    data["volume_per_atom"] = vol_per_atom

    unit_cell_mass = (n_atoms / 5.0) * (2.0 * avg_mass + 3.0 * m_o)
    density_proxy = unit_cell_mass / np.maximum(unit_cell_vol, 1e-5)
    data["density_proxy"] = density_proxy

    cation_vol_sphere = (
        (2.0 / 5.0)
        * n_atoms
        * (4.0 / 3.0)
        * np.pi
        * (x_al * r_al**3 + x_ga * r_ga**3 + x_in * r_in**3)
    )
    oxygen_vol_sphere = (3.0 / 5.0) * n_atoms * (4.0 / 3.0) * np.pi * (r_o**3)
    packing_fraction = (cation_vol_sphere + oxygen_vol_sphere) / np.maximum(
        unit_cell_vol, 1e-5
    )
    data["packing_fraction"] = packing_fraction

    data["aspect_a_b"] = a / np.maximum(b, 1e-5)
    data["aspect_b_c"] = b / np.maximum(c, 1e-5)
    data["aspect_c_a"] = c / np.maximum(a, 1e-5)

    data["diff_a_b"] = np.abs(a - b)
    data["diff_b_c"] = np.abs(b - c)
    data["diff_c_a"] = np.abs(c - a)

    data["surface_area_proxy"] = 2.0 * (a * b * sin_g + b * c * sin_a + c * a * sin_b)

    data["cos_alpha"] = cos_a
    data["cos_beta"] = cos_b
    data["cos_gamma"] = cos_g

    data["dev_alpha_90"] = np.abs(data[alpha_col].values - 90.0)
    data["dev_beta_90"] = np.abs(data[beta_col].values - 90.0)
    data["dev_gamma_90"] = np.abs(data[gamma_col].values - 90.0)

    data["strain_density"] = radius_var / np.maximum(vol_per_atom, 1e-5)
    data["vol_per_atom_x_radius"] = vol_per_atom * avg_radius

    # Crystallographic stability & strain features
    cation_std = np.sqrt(np.maximum(0, radius_var))
    data["ionic_mismatch_factor"] = cation_std / np.maximum(avg_radius, 1e-5)
    data["cation_anion_radius_ratio"] = avg_radius / r_o
    data["cell_diag_ab"] = np.sqrt(np.maximum(0, a**2 + b**2 + 2 * a * b * cos_g))
    data["cell_diag_bc"] = np.sqrt(np.maximum(0, b**2 + c**2 + 2 * b * c * cos_a))
    data["cell_diag_ca"] = np.sqrt(np.maximum(0, c**2 + a**2 + 2 * c * a * cos_b))
    data["comp_vol_interaction"] = avg_electroneg * vol_per_atom

    if sg_col in data.columns:
        sg = data[sg_col].values
        data["is_monoclinic"] = (sg == 12).astype(float)
        data["is_orthorhombic"] = (sg == 33).astype(float)
        data["is_trigonal"] = (sg == 167).astype(float)
        data["is_hexagonal"] = (sg == 194).astype(float)
        data["is_cubic"] = np.isin(sg, [206, 227]).astype(float)

    return data


# Apply feature extraction
train_processed = extract_physical_features(train_df)
test_processed = extract_physical_features(test_df)

drop_cols = [id_col, target_col, "bandgap_energy_ev"]
feature_cols = [c for c in train_processed.columns if c not in drop_cols]

X = train_processed[feature_cols].copy()
y = train_processed[target_col].values
X_test = test_processed[feature_cols].copy()

# Stratified 10-Fold Cross-Validation Setup
target_bins = pd.qcut(y, q=10, labels=False, duplicates="drop")
n_splits = 10
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

oof_lgb = np.zeros(len(X))
oof_xgb = np.zeros(len(X))
oof_cat = np.zeros(len(X))
oof_nn = np.zeros(len(X))

test_lgb = np.zeros(len(X_test))
test_xgb = np.zeros(len(X_test))
test_cat = np.zeros(len(X_test))
test_nn = np.zeros(len(X_test))


# PyTorch Deep Residual MLP Architecture
class TabularResNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, dropout_rate=0.1):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
        )
        self.block1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.act1 = nn.SiLU()
        self.block2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.act2 = nn.SiLU()
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h = self.input_layer(x)
        h = self.act1(h + self.block1(h))
        h = self.act2(h + self.block2(h))
        return self.head(h).squeeze(-1)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

y_log_all = np.log1p(np.maximum(0, y))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, target_bins)):
    X_tr, y_tr = X.iloc[train_idx].copy(), y[train_idx]
    X_va, y_va = X.iloc[val_idx].copy(), y[val_idx]
    X_te = X_test.copy()

    # Fill missing values with fold-isolated median
    fold_median = X_tr.median()
    X_tr = X_tr.fillna(fold_median)
    X_va = X_va.fillna(fold_median)
    X_te = X_te.fillna(fold_median)

    # Target Encoding for Spacegroup (fold-isolated)
    if "spacegroup" in X_tr.columns:
        sg_map = (
            pd.DataFrame({"spacegroup": X_tr["spacegroup"], "target": y_tr})
            .groupby("spacegroup")["target"]
            .mean()
            .to_dict()
        )
        g_mean = y_tr.mean()
        X_tr["sg_target_enc"] = X_tr["spacegroup"].map(lambda k: sg_map.get(k, g_mean))
        X_va["sg_target_enc"] = X_va["spacegroup"].map(lambda k: sg_map.get(k, g_mean))
        X_te["sg_target_enc"] = X_te["spacegroup"].map(lambda k: sg_map.get(k, g_mean))

    # Standard Scaling (fold-isolated)
    scaler = StandardScaler()
    X_tr_sc = pd.DataFrame(scaler.fit_transform(X_tr), columns=X_tr.columns)
    X_va_sc = pd.DataFrame(scaler.transform(X_va), columns=X_va.columns)
    X_te_sc = pd.DataFrame(scaler.transform(X_te), columns=X_te.columns)

    y_tr_log = np.log1p(np.maximum(0, y_tr))
    y_va_log = np.log1p(np.maximum(0, y_va))

    # 1. LightGBM
    lgb_model = lgb.LGBMRegressor(
        n_estimators=1200,
        learning_rate=0.025,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=10,
        random_state=SEED + fold,
        verbosity=-1,
    )
    lgb_model.fit(
        X_tr_sc,
        y_tr_log,
        eval_set=[(X_va_sc, y_va_log)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )
    oof_lgb[val_idx] = lgb_model.predict(X_va_sc)
    test_lgb += lgb_model.predict(X_te_sc) / n_splits

    # 2. XGBoost
    xgb_model = xgb.XGBRegressor(
        n_estimators=1200,
        learning_rate=0.025,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        early_stopping_rounds=50,
        random_state=SEED + fold,
        verbosity=0,
    )
    xgb_model.fit(
        X_tr_sc, y_tr_log, eval_set=[(X_va_sc, y_va_log)], verbose=False
    )
    oof_xgb[val_idx] = xgb_model.predict(X_va_sc)
    test_xgb += xgb_model.predict(X_te_sc) / n_splits

    # 3. CatBoost
    cat_model = CatBoostRegressor(
        iterations=1200,
        learning_rate=0.025,
        depth=6,
        l2_leaf_reg=3.0,
        early_stopping_rounds=50,
        random_seed=SEED + fold,
        verbose=0,
    )
    cat_model.fit(
        X_tr_sc, y_tr_log, eval_set=(X_va_sc, y_va_log), verbose=False
    )
    oof_cat[val_idx] = cat_model.predict(X_va_sc)
    test_cat += cat_model.predict(X_te_sc) / n_splits

    # 4. TabularResNet
    tr_ds = TensorDataset(
        torch.tensor(X_tr_sc.values, dtype=torch.float32),
        torch.tensor(y_tr_log, dtype=torch.float32),
    )
    va_tens = torch.tensor(X_va_sc.values, dtype=torch.float32)
    te_tens = torch.tensor(X_te_sc.values, dtype=torch.float32)

    tr_loader = DataLoader(tr_ds, batch_size=64, shuffle=True)
    nn_m = TabularResNet(input_dim=X_tr_sc.shape[1]).to(device)
    opt = torch.optim.AdamW(nn_m.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=10)
    crit = nn.MSELoss()

    best_v_loss = float("inf")
    best_va_preds = None
    best_te_preds = None

    for ep in range(120):
        nn_m.train()
        for bx, by in tr_loader:
            bx, by = bx.to(device), by.to(device)
            opt.zero_grad()
            l = crit(nn_m(bx), by)
            l.backward()
            opt.step()

        nn_m.eval()
        with torch.no_grad():
            vp = nn_m(va_tens.to(device)).cpu().numpy().flatten()
            vl = mean_squared_error(y_va_log, vp)
            sched.step(vl)

            if vl < best_v_loss:
                best_v_loss = vl
                best_va_preds = vp
                best_te_preds = nn_m(te_tens.to(device)).cpu().numpy().flatten()

    oof_nn[val_idx] = best_va_preds
    test_nn += best_te_preds / n_splits

# Optimal SLSQP Blending Strategy on Complete Out-Of-Fold Predictions
oof_preds_list = [oof_lgb, oof_xgb, oof_cat, oof_nn]
test_preds_list = [test_lgb, test_xgb, test_cat, test_nn]


def blend_objective(weights):
    weights = np.array(weights)
    weights = weights / np.sum(weights)
    blend_log = np.zeros_like(y_log_all)
    for w, pred in zip(weights, oof_preds_list):
        blend_log += w * pred
    pred_real = np.maximum(0, np.expm1(blend_log))
    return np.sqrt(
        mean_squared_error(np.log1p(y), np.log1p(np.maximum(0, pred_real)))
    )


initial_weights = [0.25, 0.25, 0.25, 0.25]
bounds = [(0, 1)] * 4
opt_res = minimize(blend_objective, initial_weights, method="SLSQP", bounds=bounds)
opt_weights = opt_res.x / np.sum(opt_res.x)

# Construct final OOF and test predictions
final_oof_log = np.zeros_like(y_log_all)
final_test_log = np.zeros(len(X_test))

for w, oof_p, test_p in zip(opt_weights, oof_preds_list, test_preds_list):
    final_oof_log += w * oof_p
    final_test_log += w * test_p

final_oof_preds = np.maximum(0.0, np.expm1(final_oof_log))
final_test_preds = np.maximum(0.0, np.expm1(final_test_log))

# Calculate official metric (RMSLE)
final_rmsle = np.sqrt(
    mean_squared_error(np.log1p(y), np.log1p(np.maximum(0, final_oof_preds)))
)

# Export official submission file
submission_df = pd.DataFrame(
    {"id": test_ids.astype(int), "formation_energy_ev_natom": final_test_preds}
)
submission_df.to_csv("./submission/submission.csv", index=False)

print(f"Final Validation Score: {final_rmsle:.5f}")