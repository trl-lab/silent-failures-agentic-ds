import itertools
import json
import math
import os
from typing import Any, Dict, List, Tuple

from catboost import CatBoostRegressor
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.cluster import KMeans
from sklearn.decomposition import FactorAnalysis, PCA, TruncatedSVD
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import BayesianRidge, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb


def set_seed(seed: int = 42):
    """Set random seed across all libraries for deterministic reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names to snake_case for consistent processing."""
    df = df.copy()
    rename_dict = {}
    for col in df.columns:
        clean_col = col.strip().replace(" ", "_").lower()
        rename_dict[col] = clean_col
    df.rename(columns=rename_dict, inplace=True)
    return df


def engineer_domain_features(df: pd.DataFrame, binary_cols: list) -> pd.DataFrame:
    """
    Engineer domain-specific photographic interaction hierarchies, aesthetic scores,
    and pair-wise combinations from compositional binary attributes.
    """
    df = df.copy()

    focus_col = "subject_focus" if "subject_focus" in df.columns else "focus"
    blur_col = "blur"
    occlusion_col = "occlusion"
    eyes_col = "eyes"
    face_col = "face"
    near_col = "near"
    group_col = "group"
    collage_col = "collage"
    human_col = "human"
    accessory_col = "accessory"
    info_col = "info"
    action_col = "action"

    # Total active attributes
    df["total_active_features"] = df[binary_cols].sum(axis=1).astype(np.float32)

    # Portrait prominence components
    prominence_components = [
        c for c in [eyes_col, face_col, near_col, focus_col] if c in df.columns
    ]
    df["portrait_prominence_score"] = (
        df[prominence_components].sum(axis=1)
        if prominence_components
        else np.zeros(len(df))
    ).astype(np.float32)

    # Visual defect score: blur, occlusion, text overlays
    defect_components = [
        c for c in [blur_col, occlusion_col, info_col] if c in df.columns
    ]
    df["visual_defect_score"] = (
        df[defect_components].sum(axis=1) if defect_components else np.zeros(len(df))
    ).astype(np.float32)

    # Contextual clutter score: accessories, humans, group photos, collages
    clutter_components = [
        c for c in [accessory_col, human_col, group_col, collage_col] if c in df.columns
    ]
    df["contextual_clutter_score"] = (
        df[clutter_components].sum(axis=1) if clutter_components else np.zeros(len(df))
    ).astype(np.float32)

    # 1. Aesthetic Quality Score: high focus and facial features penalized by defects and clutter
    df["aesthetic_quality_score"] = (
        df["portrait_prominence_score"] * 1.5
        - df["visual_defect_score"] * 1.2
        - df["contextual_clutter_score"] * 0.6
    ).astype(np.float32)

    # 2. Social Interaction Index: pet + human bonding, group dynamics, and activity
    social_components = [
        c for c in [human_col, group_col, action_col, accessory_col] if c in df.columns
    ]
    df["social_interaction_index"] = (
        df[social_components].sum(axis=1) if social_components else np.zeros(len(df))
    ).astype(np.float32)

    # 3. Compositional Complexity: ratio and interactions of active elements vs clutter
    df["composition_complexity"] = (
        (df["total_active_features"] * (df["visual_defect_score"] + 1.0))
        / (df["portrait_prominence_score"] + 1.0)
    ).astype(np.float32)

    # 4. Visual Clarity Ratio
    df["visual_clarity_ratio"] = (
        (df["portrait_prominence_score"] + 1.0) / (df["visual_defect_score"] + 1.0)
    ).astype(np.float32)

    # Net aesthetic score
    df["net_aesthetic_score"] = (
        df["portrait_prominence_score"]
        - (df["visual_defect_score"] + 0.5 * df["contextual_clutter_score"])
    ).astype(np.float32)

    # Domain indicators
    if eyes_col in df.columns and face_col in df.columns:
        solo_cond = (df[eyes_col] == 1) & (df[face_col] == 1)
        if group_col in df.columns:
            solo_cond = solo_cond & (df[group_col] == 0)
        if human_col in df.columns:
            solo_cond = solo_cond & (df[human_col] == 0)
        df["is_clean_solo_portrait"] = solo_cond.astype(np.float32)

    if focus_col in df.columns and eyes_col in df.columns and face_col in df.columns:
        pro_cond = (df[focus_col] == 1) & (df[eyes_col] == 1) & (df[face_col] == 1)
        if blur_col in df.columns:
            pro_cond = pro_cond & (df[blur_col] == 0)
        if info_col in df.columns:
            pro_cond = pro_cond & (df[info_col] == 0)
        df["is_pro_style_photo"] = pro_cond.astype(np.float32)

    if action_col in df.columns and group_col in df.columns:
        df["active_solo_pet"] = ((df[action_col] == 1) & (df[group_col] == 0)).astype(
            np.float32
        )

    if human_col in df.columns and face_col in df.columns:
        df["human_bonding_photo"] = ((df[human_col] == 1) & (df[face_col] == 1)).astype(
            np.float32
        )

    if face_col in df.columns and eyes_col in df.columns:
        df["face_without_eyes"] = ((df[face_col] == 1) & (df[eyes_col] == 0)).astype(
            np.float32
        )
    if action_col in df.columns and blur_col in df.columns:
        df["action_with_blur"] = ((df[action_col] == 1) & (df[blur_col] == 1)).astype(
            np.float32
        )

    # Multiplicative 2-way interactions
    for col1, col2 in itertools.combinations(binary_cols, 2):
        inter_name = f"inter_{col1}_x_{col2}"
        df[inter_name] = (df[col1] * df[col2]).astype(np.float32)

    # Binary configuration signature string
    df["binary_signature"] = df[binary_cols].astype(str).agg("".join, axis=1)

    return df


def compute_multi_prior_target_encoding(
    train_sig: pd.Series,
    train_y: np.ndarray,
    val_sig: pd.Series,
    test_sig: pd.Series,
    priors: List[float] = [5.0, 15.0, 30.0],
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Compute out-of-fold multi-prior smoothed Empirical Bayes target encodings on binary signatures.
    """
    global_mean = float(np.mean(train_y))
    df_temp = pd.DataFrame({"sig": train_sig.values, "target": train_y})
    stats = df_temp.groupby("sig")["target"].agg(["count", "mean"])

    tr_enc_dict = {}
    va_enc_dict = {}
    te_enc_dict = {}

    for m in priors:
        smoothed = (stats["count"] * stats["mean"] + m * global_mean) / (
            stats["count"] + m
        )
        smooth_dict = smoothed.to_dict()
        col_name = f"signature_target_enc_m{int(m)}"
        tr_enc_dict[col_name] = (
            train_sig.map(smooth_dict).fillna(global_mean).values.astype(np.float32)
        )
        va_enc_dict[col_name] = (
            val_sig.map(smooth_dict).fillna(global_mean).values.astype(np.float32)
        )
        te_enc_dict[col_name] = (
            test_sig.map(smooth_dict).fillna(global_mean).values.astype(np.float32)
        )

    return tr_enc_dict, va_enc_dict, te_enc_dict


class GatedResidualDenseBlock(nn.Module):
    """
    Gated Residual Dense Block with Feature Gating, SiLU, and LayerNorm.
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.15):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, hidden_dim * 2)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.fc(self.norm(x))
        val, gate = torch.chunk(h, 2, dim=-1)
        gated = val * torch.sigmoid(gate)
        gated = self.dropout(gated)
        out = self.proj(gated)
        return residual + out


class PawpularityGatedNet(nn.Module):
    """
    Deep Gated Feature-Interaction Neural Network for Pawpularity.
    Outputs logit representations trained via BCEWithLogitsLoss for bounded predictions.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_blocks: int = 3,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.in_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.blocks = nn.ModuleList(
            [
                GatedResidualDenseBlock(hidden_dim=hidden_dim, dropout=dropout)
                for _ in range(num_blocks)
            ]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout / 2.0),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x)
        for block in self.blocks:
            h = block(h)
        logits = self.head(h).squeeze(-1)
        return logits


def get_optimizer_and_scheduler(
    model: nn.Module,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    total_steps: int = 100,
) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)
    return optimizer, scheduler


def get_lgbm_regressor(random_state: int = 42) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        objective="regression",
        metric="rmse",
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=15,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.5,
        reg_lambda=3.0,
        min_child_samples=20,
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
    )


def get_lgbm_bce_regressor(random_state: int = 42) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        objective="cross_entropy",
        metric="cross_entropy",
        n_estimators=1000,
        learning_rate=0.025,
        num_leaves=15,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=2.0,
        reg_lambda=4.0,
        min_child_samples=20,
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
    )


def get_xgb_regressor(random_state: int = 42) -> xgb.XGBRegressor:
    return xgb.XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=4.0,
        early_stopping_rounds=50,
        random_state=random_state,
        n_jobs=-1,
    )


def get_xgb_logistic_regressor(random_state: int = 42) -> xgb.XGBRegressor:
    return xgb.XGBRegressor(
        objective="reg:logistic",
        eval_metric="logloss",
        n_estimators=1000,
        learning_rate=0.025,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=2.0,
        reg_lambda=4.0,
        early_stopping_rounds=50,
        random_state=random_state,
        n_jobs=-1,
    )


def get_catboost_regressor(random_state: int = 42) -> CatBoostRegressor:
    return CatBoostRegressor(
        loss_function="RMSE",
        eval_metric="RMSE",
        iterations=1000,
        learning_rate=0.03,
        depth=4,
        l2_leaf_reg=5.0,
        subsample=0.8,
        random_seed=random_state,
        verbose=False,
        thread_count=-1,
    )


def get_svr_regressor(C: float = 1.5, epsilon: float = 0.1) -> SVR:
    return SVR(kernel="rbf", C=C, epsilon=epsilon, gamma="scale")


def get_svr_poly_regressor(C: float = 1.0, degree: int = 2) -> SVR:
    return SVR(kernel="poly", degree=degree, C=C, epsilon=0.1)


def get_kernel_ridge_regressor(alpha: float = 2.0, gamma: float = 0.01) -> KernelRidge:
    return KernelRidge(alpha=alpha, kernel="rbf", gamma=gamma)


def get_bayesian_ridge_regressor() -> BayesianRidge:
    return BayesianRidge(alpha_1=1e-3, alpha_2=1e-3, lambda_1=1e-3, lambda_2=1e-3)


def get_knn_regressor(n_neighbors: int = 25) -> KNeighborsRegressor:
    return KNeighborsRegressor(
        n_neighbors=n_neighbors, weights="distance", metric="minkowski", p=2
    )


def get_hist_gradient_boosting_regressor(
    random_state: int = 42,
) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.03,
        max_iter=300,
        max_leaf_nodes=15,
        min_samples_leaf=20,
        l2_regularization=2.0,
        random_state=random_state,
    )


def get_extra_trees_regressor(random_state: int = 42) -> ExtraTreesRegressor:
    return ExtraTreesRegressor(
        n_estimators=300,
        max_depth=6,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features=0.8,
        random_state=random_state,
        n_jobs=-1,
    )


def get_ridge_regressor(alpha: float = 15.0, random_state: int = 42) -> Ridge:
    return Ridge(alpha=alpha, random_state=random_state)


def train_gated_nn_fold(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    X_te: np.ndarray,
    input_dim: int,
    device: torch.device,
    epochs: int = 40,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Train PawpularityGatedNet with BCEWithLogitsLoss on [0, 1] normalized targets.
    Tracks validation RMSE on the original [1, 100] scale.
    """
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_va_s = scaler.transform(X_va)
    X_te_s = scaler.transform(X_te)

    y_tr_norm = (y_tr / 100.0).astype(np.float32)
    y_va_norm = (y_va / 100.0).astype(np.float32)

    train_dataset = TensorDataset(
        torch.tensor(X_tr_s, dtype=torch.float32),
        torch.tensor(y_tr_norm, dtype=torch.float32),
    )
    val_dataset = TensorDataset(
        torch.tensor(X_va_s, dtype=torch.float32),
        torch.tensor(y_va_norm, dtype=torch.float32),
    )
    test_tensor = torch.tensor(X_te_s, dtype=torch.float32).to(device)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=False
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = PawpularityGatedNet(input_dim=input_dim).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer, scheduler = get_optimizer_and_scheduler(
        model, lr=lr, weight_decay=weight_decay, total_steps=epochs
    )

    best_val_loss = float("inf")
    best_weights = None

    for epoch in range(epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        scheduler.step()

        model.eval()
        val_preds_list = []
        with torch.no_grad():
            for batch_x, _ in val_loader:
                batch_x = batch_x.to(device)
                probs = torch.sigmoid(model(batch_x)).cpu().numpy()
                val_preds_list.append(probs * 100.0)

        val_preds_epoch = np.concatenate(val_preds_list)
        epoch_val_rmse = float(np.sqrt(mean_squared_error(y_va, val_preds_epoch)))

        if epoch_val_rmse < best_val_loss:
            best_val_loss = epoch_val_rmse
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_weights is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_weights.items()})

    model.eval()
    with torch.no_grad():
        val_preds = []
        for batch_x, _ in val_loader:
            batch_x = batch_x.to(device)
            probs = torch.sigmoid(model(batch_x)).cpu().numpy()
            val_preds.append(probs * 100.0)
        val_preds = np.concatenate(val_preds)
        test_probs = torch.sigmoid(model(test_tensor)).cpu().numpy()
        test_preds = test_probs * 100.0

    return val_preds, test_preds


def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("./working", exist_ok=True)
    os.makedirs("./submission", exist_ok=True)

    train_path = "./input/train.parquet"
    test_path = "./input/test.parquet"

    train_raw = pd.read_parquet(train_path)
    test_raw = pd.read_parquet(test_path)

    train_df = standardize_column_names(train_raw)
    test_df = standardize_column_names(test_raw)

    target_col = "pawpularity"
    id_col = "id" if "id" in train_df.columns else "id"

    reserved_cols = {target_col, id_col, "fold"}
    binary_cols = [
        c
        for c in train_df.columns
        if c not in reserved_cols and train_df[c].nunique() <= 2
    ]

    train_feat = engineer_domain_features(train_df, binary_cols)
    test_feat = engineer_domain_features(test_df, binary_cols)

    # 5-Fold Stratified K-Fold based on binned Pawpularity score
    n_splits = 5
    num_bins = 10
    train_feat["target_bin"] = pd.qcut(
        train_feat[target_col], q=num_bins, labels=False, duplicates="drop"
    )

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    train_feat["fold"] = -1
    for fold, (train_idx, val_idx) in enumerate(
        skf.split(train_feat, train_feat["target_bin"])
    ):
        train_feat.loc[val_idx, "fold"] = fold

    train_feat.drop(columns=["target_bin"], inplace=True)

    base_feature_cols = [
        c for c in train_feat.columns if c not in [id_col, target_col, "fold", "binary_signature"]
    ]

    y_train_all = train_feat[target_col].values.astype(np.float32)

    oof_lgb_mse = np.zeros(len(train_feat), dtype=np.float32)
    oof_lgb_bce = np.zeros(len(train_feat), dtype=np.float32)
    oof_xgb_mse = np.zeros(len(train_feat), dtype=np.float32)
    oof_xgb_bce = np.zeros(len(train_feat), dtype=np.float32)
    oof_cat = np.zeros(len(train_feat), dtype=np.float32)
    oof_hgb = np.zeros(len(train_feat), dtype=np.float32)
    oof_et = np.zeros(len(train_feat), dtype=np.float32)
    oof_svr = np.zeros(len(train_feat), dtype=np.float32)
    oof_svr_poly = np.zeros(len(train_feat), dtype=np.float32)
    oof_kridge = np.zeros(len(train_feat), dtype=np.float32)
    oof_bridge = np.zeros(len(train_feat), dtype=np.float32)
    oof_knn = np.zeros(len(train_feat), dtype=np.float32)
    oof_nn = np.zeros(len(train_feat), dtype=np.float32)

    test_preds_lgb_mse = np.zeros(len(test_feat), dtype=np.float32)
    test_preds_lgb_bce = np.zeros(len(test_feat), dtype=np.float32)
    test_preds_xgb_mse = np.zeros(len(test_feat), dtype=np.float32)
    test_preds_xgb_bce = np.zeros(len(test_feat), dtype=np.float32)
    test_preds_cat = np.zeros(len(test_feat), dtype=np.float32)
    test_preds_hgb = np.zeros(len(test_feat), dtype=np.float32)
    test_preds_et = np.zeros(len(test_feat), dtype=np.float32)
    test_preds_svr = np.zeros(len(test_feat), dtype=np.float32)
    test_preds_svr_poly = np.zeros(len(test_feat), dtype=np.float32)
    test_preds_kridge = np.zeros(len(test_feat), dtype=np.float32)
    test_preds_bridge = np.zeros(len(test_feat), dtype=np.float32)
    test_preds_knn = np.zeros(len(test_feat), dtype=np.float32)
    test_preds_nn = np.zeros(len(test_feat), dtype=np.float32)

    for fold in range(n_splits):
        train_mask = train_feat["fold"] != fold
        val_mask = train_feat["fold"] == fold

        df_tr = train_feat[train_mask].copy()
        df_va = train_feat[val_mask].copy()
        df_te = test_feat.copy()

        y_tr = df_tr[target_col].values.astype(np.float32)
        y_va = df_va[target_col].values.astype(np.float32)
        y_tr_norm = (y_tr / 100.0).astype(np.float32)
        y_va_norm = (y_va / 100.0).astype(np.float32)

        # Signature frequency encoding and log-odds
        sig_counts = df_tr["binary_signature"].value_counts(normalize=True).to_dict()
        df_tr["signature_freq"] = df_tr["binary_signature"].map(sig_counts).fillna(0.0).astype(np.float32)
        df_va["signature_freq"] = df_va["binary_signature"].map(sig_counts).fillna(0.0).astype(np.float32)
        df_te["signature_freq"] = df_te["binary_signature"].map(sig_counts).fillna(0.0).astype(np.float32)

        eps = 1e-5
        df_tr["signature_log_odds"] = np.log(
            (df_tr["signature_freq"] + eps) / (1.0 - df_tr["signature_freq"] + eps)
        ).astype(np.float32)
        df_va["signature_log_odds"] = np.log(
            (df_va["signature_freq"] + eps) / (1.0 - df_va["signature_freq"] + eps)
        ).astype(np.float32)
        df_te["signature_log_odds"] = np.log(
            (df_te["signature_freq"] + eps) / (1.0 - df_te["signature_freq"] + eps)
        ).astype(np.float32)

        # Out-of-fold multi-prior smoothed Bayesian target encoding
        tr_encs, va_encs, te_encs = compute_multi_prior_target_encoding(
            df_tr["binary_signature"],
            y_tr,
            df_va["binary_signature"],
            df_te["binary_signature"],
            priors=[5.0, 15.0, 30.0],
        )
        for col_k in tr_encs:
            df_tr[col_k] = tr_encs[col_k]
            df_va[col_k] = va_encs[col_k]
            df_te[col_k] = te_encs[col_k]

        cur_feature_cols = (
            base_feature_cols
            + ["signature_freq", "signature_log_odds"]
            + list(tr_encs.keys())
        )

        # Manifold Decompositions: PCA, TruncatedSVD, Factor Analysis, KMeans distances
        scaler_manifold = StandardScaler()
        tr_scaled_m = scaler_manifold.fit_transform(df_tr[cur_feature_cols].fillna(0.0))
        va_scaled_m = scaler_manifold.transform(df_va[cur_feature_cols].fillna(0.0))
        te_scaled_m = scaler_manifold.transform(df_te[cur_feature_cols].fillna(0.0))

        pca = PCA(n_components=5, random_state=42)
        tr_pca = pca.fit_transform(tr_scaled_m)
        va_pca = pca.transform(va_scaled_m)
        te_pca = pca.transform(te_scaled_m)

        svd = TruncatedSVD(n_components=5, random_state=42)
        tr_svd = svd.fit_transform(tr_scaled_m)
        va_svd = svd.transform(va_scaled_m)
        te_svd = svd.transform(te_scaled_m)

        fa = FactorAnalysis(n_components=4, random_state=42)
        tr_fa = fa.fit_transform(tr_scaled_m)
        va_fa = fa.transform(va_scaled_m)
        te_fa = fa.transform(te_scaled_m)

        kmeans = KMeans(n_clusters=6, random_state=42, n_init=10)
        tr_km = kmeans.fit_transform(tr_scaled_m)
        va_km = kmeans.transform(va_scaled_m)
        te_km = kmeans.transform(te_scaled_m)

        X_tr = np.hstack([
            df_tr[cur_feature_cols].values.astype(np.float32),
            tr_pca.astype(np.float32),
            tr_svd.astype(np.float32),
            tr_fa.astype(np.float32),
            tr_km.astype(np.float32),
        ])
        X_va = np.hstack([
            df_va[cur_feature_cols].values.astype(np.float32),
            va_pca.astype(np.float32),
            va_svd.astype(np.float32),
            va_fa.astype(np.float32),
            va_km.astype(np.float32),
        ])
        X_te = np.hstack([
            df_te[cur_feature_cols].values.astype(np.float32),
            te_pca.astype(np.float32),
            te_svd.astype(np.float32),
            te_fa.astype(np.float32),
            te_km.astype(np.float32),
        ])
        input_dim = X_tr.shape[1]

        val_indices = train_feat.index[val_mask].values

        # 1. LightGBM (MSE Objective)
        lgb_mse = get_lgbm_regressor(random_state=42 + fold)
        lgb_mse.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(-1),
            ],
        )
        oof_lgb_mse[val_indices] = lgb_mse.predict(X_va)
        test_preds_lgb_mse += lgb_mse.predict(X_te) / n_splits

        # 2. LightGBM (BCE Objective on [0, 1])
        lgb_bce = get_lgbm_bce_regressor(random_state=42 + fold)
        lgb_bce.fit(
            X_tr,
            y_tr_norm,
            eval_set=[(X_va, y_va_norm)],
            callbacks=[
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(-1),
            ],
        )
        oof_lgb_bce[val_indices] = lgb_bce.predict(X_va) * 100.0
        test_preds_lgb_bce += (lgb_bce.predict(X_te) * 100.0) / n_splits

        # 3. XGBoost (MSE Objective)
        xgb_mse = get_xgb_regressor(random_state=42 + fold)
        xgb_mse.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        oof_xgb_mse[val_indices] = xgb_mse.predict(X_va)
        test_preds_xgb_mse += xgb_mse.predict(X_te) / n_splits

        # 4. XGBoost (Logistic BCE Objective)
        xgb_logistic = get_xgb_logistic_regressor(random_state=42 + fold)
        xgb_logistic.fit(X_tr, y_tr_norm, eval_set=[(X_va, y_va_norm)], verbose=False)
        oof_xgb_bce[val_indices] = xgb_logistic.predict(X_va) * 100.0
        test_preds_xgb_bce += (xgb_logistic.predict(X_te) * 100.0) / n_splits

        # 5. CatBoost (MSE Objective)
        cat_model = get_catboost_regressor(random_state=42 + fold)
        cat_model.fit(
            X_tr,
            y_tr,
            eval_set=(X_va, y_va),
            early_stopping_rounds=50,
            verbose=False,
        )
        oof_cat[val_indices] = cat_model.predict(X_va)
        test_preds_cat += cat_model.predict(X_te) / n_splits

        # 6. HistGradientBoostingRegressor
        hgb_model = get_hist_gradient_boosting_regressor(random_state=42 + fold)
        hgb_model.fit(X_tr, y_tr)
        oof_hgb[val_indices] = hgb_model.predict(X_va)
        test_preds_hgb += hgb_model.predict(X_te) / n_splits

        # 7. ExtraTreesRegressor
        et_model = get_extra_trees_regressor(random_state=42 + fold)
        et_model.fit(X_tr, y_tr)
        oof_et[val_indices] = et_model.predict(X_va)
        test_preds_et += et_model.predict(X_te) / n_splits

        # 8. SVR (RBF & Poly), Kernel Ridge, Bayesian Ridge, KNN on Scaled Representations
        scaler_fold = StandardScaler()
        X_tr_s = scaler_fold.fit_transform(X_tr)
        X_va_s = scaler_fold.transform(X_va)
        X_te_s = scaler_fold.transform(X_te)

        svr_model = get_svr_regressor(C=2.0, epsilon=0.1)
        svr_model.fit(X_tr_s, y_tr)
        oof_svr[val_indices] = svr_model.predict(X_va_s)
        test_preds_svr += svr_model.predict(X_te_s) / n_splits

        svr_poly_model = get_svr_poly_regressor(C=1.0, degree=2)
        svr_poly_model.fit(X_tr_s, y_tr)
        oof_svr_poly[val_indices] = svr_poly_model.predict(X_va_s)
        test_preds_svr_poly += svr_poly_model.predict(X_te_s) / n_splits

        kr_model = get_kernel_ridge_regressor(alpha=2.0, gamma=0.01)
        kr_model.fit(X_tr_s, y_tr)
        oof_kridge[val_indices] = kr_model.predict(X_va_s)
        test_preds_kridge += kr_model.predict(X_te_s) / n_splits

        bridge_model = get_bayesian_ridge_regressor()
        bridge_model.fit(X_tr_s, y_tr)
        oof_bridge[val_indices] = bridge_model.predict(X_va_s)
        test_preds_bridge += bridge_model.predict(X_te_s) / n_splits

        knn_model = get_knn_regressor(n_neighbors=30)
        knn_model.fit(X_tr_s, y_tr)
        oof_knn[val_indices] = knn_model.predict(X_va_s)
        test_preds_knn += knn_model.predict(X_te_s) / n_splits

        # 9. PawpularityGatedNet (BCE with Logits)
        val_p_nn, test_p_nn = train_gated_nn_fold(
            X_tr=X_tr,
            y_tr=y_tr,
            X_va=X_va,
            y_va=y_va,
            X_te=X_te,
            input_dim=input_dim,
            device=device,
            epochs=40,
            batch_size=128,
            lr=1e-3,
            weight_decay=1e-4,
        )
        oof_nn[val_indices] = val_p_nn
        test_preds_nn += test_p_nn / n_splits

    # Constrained L2-Regularized Ensemble Optimization (SLSQP) & Ridge Blending
    models_oof = [
        oof_lgb_mse,
        oof_lgb_bce,
        oof_xgb_mse,
        oof_xgb_bce,
        oof_cat,
        oof_hgb,
        oof_et,
        oof_svr,
        oof_svr_poly,
        oof_kridge,
        oof_bridge,
        oof_knn,
        oof_nn,
    ]
    models_test = [
        test_preds_lgb_mse,
        test_preds_lgb_bce,
        test_preds_xgb_mse,
        test_preds_xgb_bce,
        test_preds_cat,
        test_preds_hgb,
        test_preds_et,
        test_preds_svr,
        test_preds_svr_poly,
        test_preds_kridge,
        test_preds_bridge,
        test_preds_knn,
        test_preds_nn,
    ]
    n_models = len(models_oof)
    oof_matrix = np.column_stack(models_oof)
    test_matrix = np.column_stack(models_test)

    def blend_objective(weights):
        pred = oof_matrix @ weights
        rmse = np.sqrt(mean_squared_error(y_train_all, pred))
        l2_reg = 1e-4 * np.sum(weights**2)
        return rmse + l2_reg

    initial_weights = np.ones(n_models) / float(n_models)
    bounds = [(0.0, 1.0) for _ in range(n_models)]
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

    opt_result = minimize(
        blend_objective,
        initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    best_weights = opt_result.x / np.sum(opt_result.x)

    # Secondary Ridge Stacking for Meta Regularization
    ridge_meta = Ridge(alpha=10.0, fit_intercept=True)
    ridge_meta.fit(oof_matrix, y_train_all)
    oof_ridge_pred = ridge_meta.predict(oof_matrix)
    test_ridge_pred = ridge_meta.predict(test_matrix)

    # 50/50 blend of SLSQP convex combination and Ridge meta-learner
    oof_slsqp_pred = oof_matrix @ best_weights
    test_slsqp_pred = test_matrix @ best_weights

    oof_ensemble = np.clip(0.5 * oof_slsqp_pred + 0.5 * oof_ridge_pred, 1.0, 100.0)
    final_oof_rmse = float(np.sqrt(mean_squared_error(y_train_all, oof_ensemble)))

    test_ensemble = np.clip(0.5 * test_slsqp_pred + 0.5 * test_ridge_pred, 1.0, 100.0)

    # Export Submission
    submission_df = pd.DataFrame(
        {
            "id": np.arange(len(test_feat), dtype=np.int64),
            "Pawpularity": test_ensemble.astype(np.float64),
        }
    )
    submission_path = "./submission/submission.csv"
    submission_df.to_csv(submission_path, index=False)

    print(f"Final Validation Score: {final_oof_rmse}")


if __name__ == "__main__":
    main()