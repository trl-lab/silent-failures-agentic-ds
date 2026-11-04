import os
import random
from typing import Any, Dict, Tuple
from catboost import CatBoostClassifier
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import rankdata
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb

# ==============================================================================
# 1. Reproducibility & Environment Setup
# ==============================================================================
SEED = 42


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


seed_everything(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs("./working/checkpoints", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# ==============================================================================
# 2. Data Ingestion & Target Formatting
# ==============================================================================
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "class"
if target_col in train_df.columns:
    if train_df[target_col].dtype == object or isinstance(
        train_df[target_col].iloc[0], str
    ):
        target_map = {
            "g": 1,
            "gamma": 1,
            "h": 0,
            "hadron": 0,
            "1": 1,
            "0": 0,
            "True": 1,
            "False": 0,
        }
        y = (
            train_df[target_col]
            .astype(str)
            .str.strip()
            .str.lower()
            .map(target_map)
            .astype(int)
            .values
        )
    else:
        y = train_df[target_col].astype(int).values
else:
    raise ValueError("Target column 'class' not found in training data.")

train_features = train_df.drop(columns=[target_col, "target"], errors="ignore")
test_features = test_df.drop(columns=[target_col, "id"], errors="ignore")
test_ids = test_df["id"].values if "id" in test_df.columns else np.arange(len(test_df))


# ==============================================================================
# 3. Physics-Informed Feature Engineering
# ==============================================================================
def extract_physics_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extracts geometric, optical, and directional shower features for MAGIC Cherenkov data."""
    feat = df.copy()
    eps = 1e-7

    # Core raw features
    length = feat["fLength"].clip(lower=0.0)
    width = feat["fWidth"].clip(lower=0.0)
    size = feat["fSize"]
    conc = feat["fConc"].clip(lower=0.0)
    conc1 = feat["fConc1"].clip(lower=0.0)
    asym = feat["fAsym"]
    m3long = feat["fM3Long"]
    m3trans = feat["fM3Trans"]
    alpha = feat["fAlpha"].clip(lower=0.0, upper=180.0)
    dist = feat["fDist"].clip(lower=0.0)

    # Convert alpha to radians
    alpha_rad = np.radians(alpha)

    # Geometric properties of the Cherenkov shower ellipse
    feat["elongation"] = length / (width + eps)
    feat["eccentricity"] = np.sqrt(np.maximum(0.0, length**2 - width**2)) / (
        length + eps
    )
    feat["ellipse_area"] = np.pi * (length / 2.0) * (width / 2.0)
    feat["ellipse_perimeter"] = np.pi * (
        3.0 * (length + width)
        - np.sqrt(np.maximum(0.0, (3.0 * length + width) * (length + 3.0 * width)))
    )
    feat["compactness"] = feat["ellipse_area"] / (feat["ellipse_perimeter"] ** 2 + eps)
    feat["aspect_ratio_inv"] = width / (length + eps)

    # Optical intensity & density features
    feat["log_density"] = size / (feat["ellipse_area"] + eps)
    feat["size_to_length"] = size / (length + eps)
    feat["size_to_width"] = size / (width + eps)
    feat["conc_diff"] = conc - conc1
    feat["conc_ratio"] = conc1 / (conc + eps)
    feat["energy_weighted_conc"] = size * conc
    feat["energy_weighted_conc1"] = size * conc1

    # Physical linear photon flux and peak energy intensities
    linear_flux = np.power(10.0, size)
    feat["linear_photon_flux"] = linear_flux
    feat["linear_peak_intensity"] = linear_flux * conc
    feat["linear_peak_intensity1"] = linear_flux * conc1
    feat["linear_flux_density"] = linear_flux / (feat["ellipse_area"] + eps)
    feat["linear_peak_contrast"] = linear_flux * (conc - conc1)

    # Normalized shape index & log aspect ratio
    feat["normalized_shape_index"] = (length - width) / (length + width + eps)
    feat["log_aspect_ratio"] = np.log((length + eps) / (width + eps))

    # Directional and impact geometry & harmonic terms
    feat["sin_alpha"] = np.sin(alpha_rad)
    feat["cos_alpha"] = np.cos(alpha_rad)
    feat["sin_2alpha"] = np.sin(2.0 * alpha_rad)
    feat["cos_2alpha"] = np.cos(2.0 * alpha_rad)
    feat["x_src"] = dist * np.cos(alpha_rad)
    feat["y_src"] = dist * np.sin(alpha_rad)
    feat["miss_distance"] = dist * np.sin(alpha_rad)
    feat["projected_disp"] = dist * np.cos(alpha_rad)
    feat["impact_parameter"] = np.sqrt(
        feat["miss_distance"] ** 2
        + (feat["projected_disp"] * (1.0 - feat["eccentricity"])) ** 2
    )

    # Asymmetry & Higher Order Moments (normalized skewness)
    feat["asym_abs"] = np.abs(asym)
    feat["asym_to_length"] = asym / (length + eps)
    feat["asym_sign"] = np.sign(asym)
    feat["m3_norm"] = np.sqrt(m3long**2 + m3trans**2)
    feat["m3_ratio"] = m3long / (np.abs(m3trans) + eps)
    feat["m3_long_to_length"] = np.abs(m3long) / (length + eps)
    feat["m3_trans_to_width"] = np.abs(m3trans) / (width + eps)
    feat["norm_m3_long_cubed"] = (m3long / (length + eps)) ** 3
    feat["norm_m3_trans_cubed"] = (m3trans / (width + eps)) ** 3

    # Non-linear transformations
    feat["log_length"] = np.log1p(length)
    feat["log_width"] = np.log1p(width)
    feat["log_dist"] = np.log1p(dist)
    feat["log_miss_dist"] = np.log1p(np.maximum(0.0, feat["miss_distance"]))

    # Key interaction terms
    feat["length_x_dist"] = length * dist
    feat["width_x_dist"] = width * dist
    feat["alpha_x_dist"] = alpha * dist
    feat["elongation_x_conc"] = feat["elongation"] * conc

    # Standard Cherenkov DISP method & theta-squared source reconstruction
    xi = 1.35
    disp = xi * (1.0 - width / (length + eps))
    disp_scaled = (1.0 - width / (length + eps)) * (1.2 + 0.06 * size)
    feat["disp"] = disp
    feat["disp_scaled"] = disp_scaled

    # Reconstructed source distance squared (theta^2) and angular offset
    theta2_minus = dist**2 + disp**2 - 2.0 * dist * disp * feat["cos_alpha"]
    theta2_plus = dist**2 + disp**2 + 2.0 * dist * disp * feat["cos_alpha"]
    feat["theta2_min"] = np.maximum(0.0, np.minimum(theta2_minus, theta2_plus))
    feat["theta_min"] = np.sqrt(feat["theta2_min"])

    # Disambiguate head-tail orientation using fAsym and fM3Long signs
    feat["theta2_asym"] = np.maximum(0.0, np.where(asym >= 0, theta2_minus, theta2_plus))
    feat["theta2_m3"] = np.maximum(0.0, np.where(m3long >= 0, theta2_minus, theta2_plus))
    feat["theta_asym"] = np.sqrt(feat["theta2_asym"])
    feat["theta_m3"] = np.sqrt(feat["theta2_m3"])

    # Energy-scaled Hillas parameters and photon distribution scaling
    feat["scaled_width"] = width / (size + eps)
    feat["scaled_length"] = length / (size + eps)
    feat["log_scaled_width"] = np.log1p(width) / (size + eps)
    feat["log_scaled_length"] = np.log1p(length) / (size + eps)
    feat["core_to_tail_ratio"] = conc / (1.0 - conc + eps)
    feat["core_peak_fraction"] = conc1 / (conc + eps)
    feat["photon_density_scale"] = size / (length * width + eps)
    feat["energy_normalized_compactness"] = (length * width) / (size**2 + eps)
    feat["disp_to_dist_ratio"] = disp / (dist + eps)

    # Camera containment, outer boundaries, and edge truncation proxies
    feat["shower_outer_edge"] = dist + 0.5 * length
    feat["shower_inner_edge"] = np.maximum(0.0, dist - 0.5 * length)
    feat["shower_lateral_edge"] = dist + 0.5 * width
    feat["containment_ratio_outer"] = dist / (dist + 0.5 * length + eps)
    feat["containment_ratio_width"] = dist / (dist + 0.5 * width + eps)
    feat["radial_extent"] = np.sqrt(dist**2 + (0.5 * length) ** 2)

    # Square-root energy scaling for lateral and longitudinal shower development
    sqrt_size = np.sqrt(np.maximum(0.0, size))
    feat["width_to_sqrt_size"] = width / (sqrt_size + eps)
    feat["length_to_sqrt_size"] = length / (sqrt_size + eps)
    feat["area_to_sqrt_size"] = (np.pi * length * width / 4.0) / (sqrt_size + eps)
    feat["area_to_size"] = (length * width) / (size + eps)

    # Normalized skewness moments & coupling
    feat["m3long_norm_1p5"] = m3long / (np.power(length + eps, 1.5))
    feat["m3trans_norm_1p5"] = m3trans / (np.power(width + eps, 1.5))
    feat["m3_asym_coupling"] = (m3long * asym) / (length**2 + eps)
    feat["disp_over_length"] = disp / (length + eps)
    feat["disp_over_width"] = disp / (width + eps)
    feat["impact_to_dist"] = feat["impact_parameter"] / (dist + eps)

    # Additional Cherenkov physics features: Azimuthal width, miss-to-width ratio, multi-scale DISP
    feat["azwidth"] = length * feat["sin_alpha"] + width * feat["cos_alpha"]
    feat["azwidth_ratio"] = feat["azwidth"] / (width + eps)
    feat["miss_to_width"] = feat["miss_distance"] / (width + eps)
    feat["miss_to_length"] = feat["miss_distance"] / (length + eps)

    # Multi-scale energy-dependent DISP formulations
    feat["disp_energy_multiscale"] = (1.0 - width / (length + eps)) * (1.15 + 0.075 * size + 0.01 * (size**2))
    feat["disp_asym_weighted"] = disp * np.sign(asym)
    feat["disp_m3_weighted"] = disp * np.sign(m3long)

    # Higher order moment tensor couplings
    feat["m3_tensor_coupling"] = (m3long * m3trans) / ((length * width) + eps)
    feat["asym_trans_coupling"] = (asym * m3trans) / ((length * width) + eps)

    return feat


X_train = (
    extract_physics_features(train_features)
    .replace([np.inf, -np.inf], np.nan)
    .fillna(0.0)
)
X_test = (
    extract_physics_features(test_features)
    .replace([np.inf, -np.inf], np.nan)
    .fillna(0.0)
)
feature_cols = list(X_train.columns)


# ==============================================================================
# 4. Neural Architecture Definition (Tabular Gated SE Residual Network)
# ==============================================================================
class SqueezeExcitation(nn.Module):

    def __init__(self, dim: int, reduction: int = 4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim, max(8, dim // reduction), bias=False),
            nn.GELU(),
            nn.Linear(max(8, dim // reduction), dim, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(x)


class ResNetSEBlock(nn.Module):

    def __init__(self, hidden_dim: int, dropout_rate: float = 0.15):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.linear1 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.act1 = nn.GELU()
        self.norm2 = nn.LayerNorm(hidden_dim * 2)
        self.linear2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.se = SqueezeExcitation(hidden_dim, reduction=4)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.norm1(x)
        out = self.act1(self.linear1(out))
        out = self.norm2(out)
        out = self.linear2(out)
        out = self.se(out)
        out = self.dropout(out)
        return residual + out


class TabularResNetSE(nn.Module):

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_blocks: int = 4,
        dropout_rate: float = 0.15,
        output_dim: int = 1,
    ):
        super().__init__()
        self.input_gate = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, input_dim),
            nn.Sigmoid(),
        )
        self.input_proj = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.5),
        )

        self.blocks = nn.ModuleList(
            [
                ResNetSEBlock(hidden_dim=hidden_dim, dropout_rate=dropout_rate)
                for _ in range(num_blocks)
            ]
        )

        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gated_x = x * self.input_gate(x)
        h = self.input_proj(gated_x)
        for block in self.blocks:
            h = block(h)
        return self.head(h).squeeze(-1)


# ==============================================================================
# 5. Cross-Validation Setup & Model Hyperparameters
# ==============================================================================
n_splits = 10
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

lgb_params = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "learning_rate": 0.03,
    "num_leaves": 47,
    "max_depth": 7,
    "min_child_samples": 25,
    "subsample": 0.85,
    "subsample_freq": 1,
    "colsample_bytree": 0.85,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "n_estimators": 2500,
    "random_state": SEED,
    "n_jobs": -1,
    "verbose": -1,
}

xgb_params = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "tree_method": "hist",
    "learning_rate": 0.03,
    "max_depth": 7,
    "min_child_weight": 3,
    "subsample": 0.85,
    "colsample_bytree": 0.75,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "n_estimators": 2500,
    "random_state": SEED,
    "n_jobs": -1,
}

catboost_params1 = {
    "loss_function": "Logloss",
    "eval_metric": "AUC",
    "learning_rate": 0.04,
    "depth": 6,
    "l2_leaf_reg": 3.0,
    "iterations": 2500,
    "random_seed": SEED,
    "verbose": False,
}

catboost_params2 = {
    "loss_function": "Logloss",
    "eval_metric": "AUC",
    "learning_rate": 0.03,
    "depth": 8,
    "l2_leaf_reg": 7.0,
    "iterations": 2500,
    "random_seed": SEED + 1,
    "verbose": False,
}

hgb_params = {
    "loss": "log_loss",
    "learning_rate": 0.035,
    "max_iter": 600,
    "max_leaf_nodes": 47,
    "min_samples_leaf": 20,
    "l2_regularization": 1.0,
    "early_stopping": True,
    "n_iter_no_change": 30,
    "random_state": SEED,
}

# Prediction containers
oof_lgb = np.zeros(len(X_train))
oof_xgb = np.zeros(len(X_train))
oof_cat1 = np.zeros(len(X_train))
oof_cat2 = np.zeros(len(X_train))
oof_hgb = np.zeros(len(X_train))
oof_et = np.zeros(len(X_train))
oof_rf = np.zeros(len(X_train))
oof_nn = np.zeros(len(X_train))

test_preds_lgb = np.zeros(len(X_test))
test_preds_xgb = np.zeros(len(X_test))
test_preds_cat1 = np.zeros(len(X_test))
test_preds_cat2 = np.zeros(len(X_test))
test_preds_hgb = np.zeros(len(X_test))
test_preds_et = np.zeros(len(X_test))
test_preds_rf = np.zeros(len(X_test))
test_preds_nn = np.zeros(len(X_test))

# ==============================================================================
# 6. Stratified Cross-Validation Execution
# ==============================================================================
for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train, y)):
    X_tr_raw, y_tr_f = X_train.iloc[train_idx], y[train_idx]
    X_va_raw, y_va_f = X_train.iloc[val_idx], y[val_idx]
    X_te_raw = X_test.copy()

    # Leak-free Unsupervised Feature Extraction (KMeans & PCA) per split
    scaler_us = StandardScaler()
    tr_scaled_us = scaler_us.fit_transform(X_tr_raw)
    va_scaled_us = scaler_us.transform(X_va_raw)
    te_scaled_us = scaler_us.transform(X_te_raw)

    kmeans = MiniBatchKMeans(n_clusters=8, random_state=SEED + fold_idx, batch_size=256, n_init=3)
    tr_km = kmeans.fit_transform(tr_scaled_us)
    va_km = kmeans.transform(va_scaled_us)
    te_km = kmeans.transform(te_scaled_us)

    pca = PCA(n_components=6, random_state=SEED + fold_idx)
    tr_pca = pca.fit_transform(tr_scaled_us)
    va_pca = pca.transform(va_scaled_us)
    te_pca = pca.transform(te_scaled_us)

    # Leak-free k-NN manifold density representation
    knn = NearestNeighbors(n_neighbors=16, metric="euclidean", n_jobs=-1)
    knn.fit(tr_scaled_us)
    tr_knn_dists, _ = knn.kneighbors(tr_scaled_us)
    tr_knn_feats = np.column_stack([
        tr_knn_dists[:, 1:6].mean(axis=1),
        tr_knn_dists[:, 1:16].mean(axis=1),
        tr_knn_dists[:, 15] / (tr_knn_dists[:, 1] + 1e-7),
    ])
    va_knn_dists, _ = knn.kneighbors(va_scaled_us, n_neighbors=15)
    va_knn_feats = np.column_stack([
        va_knn_dists[:, :5].mean(axis=1),
        va_knn_dists[:, :15].mean(axis=1),
        va_knn_dists[:, 14] / (va_knn_dists[:, 0] + 1e-7),
    ])
    te_knn_dists, _ = knn.kneighbors(te_scaled_us, n_neighbors=15)
    te_knn_feats = np.column_stack([
        te_knn_dists[:, :5].mean(axis=1),
        te_knn_dists[:, :15].mean(axis=1),
        te_knn_dists[:, 14] / (te_knn_dists[:, 0] + 1e-7),
    ])

    X_tr_f = np.hstack([X_tr_raw.values, tr_km, tr_pca, tr_knn_feats])
    X_va_f = np.hstack([X_va_raw.values, va_km, va_pca, va_knn_feats])
    X_te_f = np.hstack([X_te_raw.values, te_km, te_pca, te_knn_feats])

    # --- A. LightGBM ---
    lgb_clf = lgb.LGBMClassifier(**lgb_params)
    lgb_clf.fit(
        X_tr_f,
        y_tr_f,
        eval_set=[(X_va_f, y_va_f)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
    )
    oof_lgb[val_idx] = lgb_clf.predict_proba(X_va_f)[:, 1]
    test_preds_lgb += lgb_clf.predict_proba(X_te_f)[:, 1] / n_splits

    # --- B. XGBoost ---
    xgb_clf = xgb.XGBClassifier(**xgb_params, early_stopping_rounds=50)
    xgb_clf.fit(X_tr_f, y_tr_f, eval_set=[(X_va_f, y_va_f)], verbose=False)
    oof_xgb[val_idx] = xgb_clf.predict_proba(X_va_f)[:, 1]
    test_preds_xgb += xgb_clf.predict_proba(X_te_f)[:, 1] / n_splits

    # --- C. CatBoost Model 1 (Depth 6) ---
    cat_clf1 = CatBoostClassifier(**catboost_params1, early_stopping_rounds=50)
    cat_clf1.fit(X_tr_f, y_tr_f, eval_set=(X_va_f, y_va_f), verbose=False)
    oof_cat1[val_idx] = cat_clf1.predict_proba(X_va_f)[:, 1]
    test_preds_cat1 += cat_clf1.predict_proba(X_te_f)[:, 1] / n_splits

    # --- D. CatBoost Model 2 (Depth 8) ---
    cat_clf2 = CatBoostClassifier(**catboost_params2, early_stopping_rounds=50)
    cat_clf2.fit(X_tr_f, y_tr_f, eval_set=(X_va_f, y_va_f), verbose=False)
    oof_cat2[val_idx] = cat_clf2.predict_proba(X_va_f)[:, 1]
    test_preds_cat2 += cat_clf2.predict_proba(X_te_f)[:, 1] / n_splits

    # --- E. HistGradientBoosting ---
    hgb_clf = HistGradientBoostingClassifier(**hgb_params)
    hgb_clf.fit(X_tr_f, y_tr_f)
    oof_hgb[val_idx] = hgb_clf.predict_proba(X_va_f)[:, 1]
    test_preds_hgb += hgb_clf.predict_proba(X_te_f)[:, 1] / n_splits

    # --- F. ExtraTrees ---
    et_clf = ExtraTreesClassifier(
        n_estimators=350,
        max_depth=24,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=SEED,
        n_jobs=-1,
    )
    et_clf.fit(X_tr_f, y_tr_f)
    oof_et[val_idx] = et_clf.predict_proba(X_va_f)[:, 1]
    test_preds_et += et_clf.predict_proba(X_te_f)[:, 1] / n_splits

    # --- G. RandomForest ---
    rf_clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=SEED,
        n_jobs=-1,
    )
    rf_clf.fit(X_tr_f, y_tr_f)
    oof_rf[val_idx] = rf_clf.predict_proba(X_va_f)[:, 1]
    test_preds_rf += rf_clf.predict_proba(X_te_f)[:, 1] / n_splits

    # --- H. Tabular ResNet SE ---
    scaler_nn = StandardScaler()
    X_tr_scaled = scaler_nn.fit_transform(X_tr_f)
    X_va_scaled = scaler_nn.transform(X_va_f)
    X_te_scaled = scaler_nn.transform(X_te_f)

    train_ds = TensorDataset(
        torch.tensor(X_tr_scaled, dtype=torch.float32),
        torch.tensor(y_tr_f, dtype=torch.float32),
    )
    val_ds = TensorDataset(
        torch.tensor(X_va_scaled, dtype=torch.float32),
        torch.tensor(y_va_f, dtype=torch.float32),
    )

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, drop_last=False)

    nn_model = TabularResNetSE(
        input_dim=X_tr_scaled.shape[1],
        hidden_dim=256,
        num_blocks=4,
        dropout_rate=0.15,
    ).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(nn_model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=35, eta_min=1e-6
    )

    best_val_loss = float("inf")
    best_weights_path = f"./working/checkpoints/nn_best_fold_{fold_idx}.pt"

    for epoch in range(35):
        nn_model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = nn_model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
        scheduler.step()

        nn_model.eval()
        val_loss_epoch = 0.0
        val_samples = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                logits = nn_model(batch_x)
                loss = criterion(logits, batch_y)
                val_loss_epoch += loss.item() * len(batch_y)
                val_samples += len(batch_y)
        avg_val_loss = val_loss_epoch / val_samples

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(nn_model.state_dict(), best_weights_path)

    nn_model.load_state_dict(torch.load(best_weights_path))
    nn_model.eval()
    with torch.no_grad():
        val_tensor = torch.tensor(X_va_scaled, dtype=torch.float32).to(device)
        oof_nn[val_idx] = torch.sigmoid(nn_model(val_tensor)).cpu().numpy().flatten()

        test_tensor = torch.tensor(X_te_scaled, dtype=torch.float32).to(device)
        test_preds_nn += (
            torch.sigmoid(nn_model(test_tensor)).cpu().numpy().flatten() / n_splits
        )

# ==============================================================================
# 7. Ensemble Weight Optimization & Metric Evaluation
# ==============================================================================
oof_list = [
    oof_lgb,
    oof_xgb,
    oof_cat1,
    oof_cat2,
    oof_hgb,
    oof_et,
    oof_rf,
    oof_nn,
]
test_list = [
    test_preds_lgb,
    test_preds_xgb,
    test_preds_cat1,
    test_preds_cat2,
    test_preds_hgb,
    test_preds_et,
    test_preds_rf,
    test_preds_nn,
]

# Apply percentile rank transformation to align calibration across model families
oof_rank_matrix = np.column_stack(
    [rankdata(pred) / len(pred) for pred in oof_list]
)
test_rank_matrix = np.column_stack(
    [rankdata(pred) / len(pred) for pred in test_list]
)


def roc_objective(weights: np.ndarray) -> float:
    w = np.maximum(0, weights)
    w_sum = np.sum(w)
    if w_sum == 0:
        return 1.0
    w = w / w_sum
    blend_oof = np.dot(oof_rank_matrix, w)
    return -roc_auc_score(y, blend_oof)


init_weights = np.ones(len(oof_list)) / len(oof_list)
opt_res = minimize(
    roc_objective,
    init_weights,
    method="Nelder-Mead",
    options={"maxiter": 1000, "disp": False},
)

optimal_weights = np.maximum(0, opt_res.x)
optimal_weights /= np.sum(optimal_weights)

final_oof_predictions = np.dot(oof_rank_matrix, optimal_weights)
final_test_predictions = np.dot(test_rank_matrix, optimal_weights)

final_score = roc_auc_score(y, final_oof_predictions)

# ==============================================================================
# 8. Generate Submission File
# ==============================================================================
submission_df = pd.DataFrame(
    {"id": test_ids.astype(int), "class": final_test_predictions}
)
submission_df.to_csv("./submission/submission.csv", index=False)

print(f"Final Validation Score: {final_score}")
