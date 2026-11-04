"""Satellite Image Classification Pipeline.

End-to-end pipeline:
1. Domain-informed spatial-spectral feature engineering (NDVI, NDWI, SAVI, gradients, spatial patches).
2. Stratified 5-Fold Cross-Validation.
3. Multi-model ensembling combining LightGBM, XGBoost, CatBoost, and PyTorch SpatialSpectralNet.
4. Out-of-fold validation evaluation and submission generation.
"""

import json
import os
import random
from typing import Any, Dict, List, Optional, Tuple

from catboost import CatBoostClassifier
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from xgboost import XGBClassifier


# =========================================================================
# 0. Reproducibility & Seeding
# =========================================================================
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# =========================================================================
# 1. Feature Engineering & Preprocessing
# =========================================================================
def get_feature_columns(df: pd.DataFrame, target_col: str = "class") -> List[str]:
    """Identify the 36 spectral/spatial feature columns in the dataset."""
    feature_cols = [col for col in df.columns if col != target_col and col != "id"]
    if len(feature_cols) != 36:
        feature_cols = sorted(feature_cols)
    return feature_cols


def compute_spectral_and_spatial_features(
    df: pd.DataFrame, feature_cols: List[str]
) -> pd.DataFrame:
    """Extract comprehensive remote sensing spectral indices and 2D spatial convolution features.

    Landsat MSS 4 Bands across 9 pixels in a 3x3 patch:
    - Band 1 (index 0): Green
    - Band 2 (index 1): Red
    - Band 3 (index 2): NIR1
    - Band 4 (index 3): NIR2
    Pixel index 4 is the central pixel.
    """
    X_raw = df[feature_cols].values.astype(np.float32)
    n_samples = X_raw.shape[0]
    eps = 1e-6

    # Reshape to (N, 9, 4) and (N, 3, 3, 4)
    pixels_4b = X_raw.reshape(n_samples, 9, 4)
    patch_3x3_4b = X_raw.reshape(n_samples, 3, 3, 4)

    center_b1 = pixels_4b[:, 4, 0]  # Green
    center_b2 = pixels_4b[:, 4, 1]  # Red
    center_b3 = pixels_4b[:, 4, 2]  # NIR1
    center_b4 = pixels_4b[:, 4, 3]  # NIR2

    surround_indices = [0, 1, 2, 3, 5, 6, 7, 8]
    surround_pixels = pixels_4b[:, surround_indices, :]

    features_dict: Dict[str, np.ndarray] = {}

    # 1. Raw features
    for i, col_name in enumerate(feature_cols):
        features_dict[f"raw_{col_name}"] = X_raw[:, i]

    # 2. Comprehensive Central Pixel Spectral Indices
    # Pairwise Normalized Differences across all band pairs (6 combinations)
    features_dict["center_nd_b1_b2"] = (center_b1 - center_b2) / (center_b1 + center_b2 + eps)
    features_dict["center_nd_b1_b3"] = (center_b1 - center_b3) / (center_b1 + center_b3 + eps)
    features_dict["center_nd_b1_b4"] = (center_b1 - center_b4) / (center_b1 + center_b4 + eps)
    features_dict["center_ndvi_b3"] = (center_b3 - center_b2) / (center_b3 + center_b2 + eps)
    features_dict["center_ndvi_b4"] = (center_b4 - center_b2) / (center_b4 + center_b2 + eps)
    features_dict["center_nd_b3_b4"] = (center_b4 - center_b3) / (center_b4 + center_b3 + eps)

    features_dict["center_ndwi_b3"] = features_dict["center_nd_b1_b3"]
    features_dict["center_ndwi_b4"] = features_dict["center_nd_b1_b4"]
    features_dict["center_savi_b3"] = ((center_b3 - center_b2) * 1.5) / (
        center_b3 + center_b2 + 0.5 + eps
    )
    features_dict["center_savi_b4"] = ((center_b4 - center_b2) * 1.5) / (
        center_b4 + center_b2 + 0.5 + eps
    )
    features_dict["center_rvi_b3"] = (center_b3 + eps) / (center_b2 + eps)
    features_dict["center_rvi_b4"] = (center_b4 + eps) / (center_b2 + eps)
    features_dict["center_brightness"] = (
        center_b1 + center_b2 + center_b3 + center_b4
    ) / 4.0
    features_dict["center_nir_ratio"] = (center_b3 + center_b4) / (
        center_b1 + center_b2 + eps
    )
    features_dict["center_red_green_ratio"] = (center_b2 + eps) / (center_b1 + eps)
    features_dict["center_nir_diff"] = center_b4 - center_b3
    features_dict["center_vis_diff"] = center_b2 - center_b1

    # 3. 2D Spatial Convolution Kernels and Patch Aggregations per Band
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    laplacian_4 = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float32)
    laplacian_8 = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.float32)

    band_names = ["green", "red", "nir1", "nir2"]
    for b_idx, b_name in enumerate(band_names):
        band_patch = patch_3x3_4b[:, :, :, b_idx]  # (N, 3, 3)
        band_all_9 = pixels_4b[:, :, b_idx]
        band_surround_8 = surround_pixels[:, :, b_idx]
        band_center = pixels_4b[:, 4, b_idx]

        b_mean = np.mean(band_all_9, axis=1)
        b_std = np.std(band_all_9, axis=1)
        b_min = np.min(band_all_9, axis=1)
        b_max = np.max(band_all_9, axis=1)
        b_median = np.median(band_all_9, axis=1)

        features_dict[f"patch_{b_name}_mean"] = b_mean
        features_dict[f"patch_{b_name}_std"] = b_std
        features_dict[f"patch_{b_name}_min"] = b_min
        features_dict[f"patch_{b_name}_max"] = b_max
        features_dict[f"patch_{b_name}_range"] = b_max - b_min
        features_dict[f"patch_{b_name}_median"] = b_median

        surround_mean = np.mean(band_surround_8, axis=1)
        surround_std = np.std(band_surround_8, axis=1)
        features_dict[f"surround_{b_name}_mean"] = surround_mean
        features_dict[f"surround_{b_name}_std"] = surround_std
        features_dict[f"diff_center_vs_surround_mean_{b_name}"] = (
            band_center - surround_mean
        )
        features_dict[f"ratio_center_vs_surround_mean_{b_name}"] = (
            band_center + eps
        ) / (surround_mean + eps)

        # 2D Spatial Convolutions
        features_dict[f"sobel_x_{b_name}"] = (band_patch * sobel_x).sum(axis=(1, 2))
        features_dict[f"sobel_y_{b_name}"] = (band_patch * sobel_y).sum(axis=(1, 2))
        features_dict[f"grad_mag_{b_name}"] = np.sqrt(
            features_dict[f"sobel_x_{b_name}"] ** 2
            + features_dict[f"sobel_y_{b_name}"] ** 2
            + eps
        )
        features_dict[f"laplacian4_{b_name}"] = (band_patch * laplacian_4).sum(axis=(1, 2))
        features_dict[f"laplacian8_{b_name}"] = (band_patch * laplacian_8).sum(axis=(1, 2))

        # Corner vs Center Contrast
        corners_mean = (
            band_patch[:, 0, 0]
            + band_patch[:, 0, 2]
            + band_patch[:, 2, 0]
            + band_patch[:, 2, 2]
        ) / 4.0
        features_dict[f"corner_contrast_{b_name}"] = band_center - corners_mean

    # 4. Patch-wide aggregate Spectral Indices across all 9 pixels
    all_b1 = pixels_4b[:, :, 0]
    all_b2 = pixels_4b[:, :, 1]
    all_b3 = pixels_4b[:, :, 2]
    all_b4 = pixels_4b[:, :, 3]

    all_ndvi_b3 = (all_b3 - all_b2) / (all_b3 + all_b2 + eps)
    all_ndvi_b4 = (all_b4 - all_b2) / (all_b4 + all_b2 + eps)
    all_ndwi_b3 = (all_b1 - all_b3) / (all_b1 + all_b3 + eps)
    all_ndgr = (all_b1 - all_b2) / (all_b1 + all_b2 + eps)

    features_dict["patch_ndvi_b3_mean"] = np.mean(all_ndvi_b3, axis=1)
    features_dict["patch_ndvi_b3_std"] = np.std(all_ndvi_b3, axis=1)
    features_dict["patch_ndvi_b4_mean"] = np.mean(all_ndvi_b4, axis=1)
    features_dict["patch_ndvi_b4_std"] = np.std(all_ndvi_b4, axis=1)
    features_dict["patch_ndwi_b3_mean"] = np.mean(all_ndwi_b3, axis=1)
    features_dict["patch_ndwi_b3_std"] = np.std(all_ndwi_b3, axis=1)
    features_dict["patch_ndgr_mean"] = np.mean(all_ndgr, axis=1)
    features_dict["patch_ndgr_std"] = np.std(all_ndgr, axis=1)
    features_dict["diff_center_ndvi_vs_patch_mean"] = (
        features_dict["center_ndvi_b3"] - features_dict["patch_ndvi_b3_mean"]
    )

    engineered_df = pd.DataFrame(features_dict, index=df.index)
    return engineered_df


def prepare_spatial_patches(df: pd.DataFrame) -> np.ndarray:
    """Extract and reshape the 36 raw features into 4-band 3x3 spatial patches: (N, 4, 3, 3)."""
    raw_cols = [col for col in df.columns if col.startswith("raw_")]
    if len(raw_cols) != 36:
        feat_cols = [c for c in df.columns if c not in ["target", "fold", "id"]]
        raw_cols = feat_cols[:36]

    raw_values = df[raw_cols].values.astype(np.float32)
    n_samples = raw_values.shape[0]
    patch_3x3_4b = raw_values.reshape(n_samples, 3, 3, 4)
    patch_tensor = np.transpose(patch_3x3_4b, (0, 3, 1, 2))  # (N, 4, 3, 3)
    return patch_tensor


# =========================================================================
# 2. PyTorch Model Architectures & Loss
# =========================================================================
class SqueezeExcitation2D(nn.Module):
    """Squeeze-and-Excitation channel attention for 2D spatial patches."""

    def __init__(self, channels: int, reduction: int = 2):
        super().__init__()
        reduced_ch = max(4, channels // reduction)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, reduced_ch, bias=False),
            nn.GELU(),
            nn.Linear(reduced_ch, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        w = self.fc(x).view(b, c, 1, 1)
        return x * w


class ConvPatchEncoder(nn.Module):
    """2D Convolutional encoder for 3x3 multi-spectral spatial patches."""

    def __init__(self, in_channels: int = 4, out_dim: int = 64, dropout: float = 0.15):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            SqueezeExcitation2D(32),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            SqueezeExcitation2D(64),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = self.conv2(h)
        return self.fc(h)


class TabularMLPEncoder(nn.Module):
    """Residual MLP encoder for engineered spectral and spatial tabular features."""

    def __init__(self, in_features: int, hidden_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.block1_fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.block1_bn1 = nn.BatchNorm1d(hidden_dim)
        self.block1_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.block1_bn2 = nn.BatchNorm1d(hidden_dim)
        self.block1_drop = nn.Dropout(dropout)

        self.block2_fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.block2_bn1 = nn.BatchNorm1d(hidden_dim // 2)
        self.block2_fc2 = nn.Linear(hidden_dim // 2, hidden_dim // 2)
        self.block2_bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.block2_shortcut = nn.Linear(hidden_dim, hidden_dim // 2)
        self.block2_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h0 = self.input_layer(x)
        h1 = F.gelu(self.block1_bn1(self.block1_fc1(h0)))
        h1 = self.block1_drop(self.block1_bn2(self.block1_fc2(h1)))
        h1 = F.gelu(h1 + h0)

        h2 = F.gelu(self.block2_bn1(self.block2_fc1(h1)))
        h2 = self.block2_drop(self.block2_bn2(self.block2_fc2(h2)))
        out = F.gelu(h2 + self.block2_shortcut(h1))
        return out


class SpatialSpectralNet(nn.Module):
    """Dual-Stream Spatial-Spectral Fusion Network."""

    def __init__(
        self,
        num_tabular_features: int,
        num_classes: int = 6,
        conv_dim: int = 64,
        mlp_dim: int = 128,
        head_dim: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.conv_encoder = ConvPatchEncoder(
            in_channels=4, out_dim=conv_dim, dropout=dropout
        )
        self.mlp_encoder = TabularMLPEncoder(
            in_features=num_tabular_features, hidden_dim=mlp_dim, dropout=dropout
        )

        fused_dim = conv_dim + (mlp_dim // 2)
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, head_dim),
            nn.BatchNorm1d(head_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_dim, num_classes),
        )

    def forward(
        self, patch_tensor: torch.Tensor, tabular_tensor: torch.Tensor
    ) -> torch.Tensor:
        spatial_emb = self.conv_encoder(patch_tensor)
        tabular_emb = self.mlp_encoder(tabular_tensor)
        fused = torch.cat([spatial_emb, tabular_emb], dim=1)
        return self.classifier(fused)


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross-entropy loss with uniform label smoothing."""

    def __init__(self, smoothing: float = 0.05):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_preds = F.log_softmax(logits, dim=-1)
        loss = -log_preds.gather(dim=-1, index=targets.unsqueeze(1)).squeeze(1)
        smooth_loss = -log_preds.mean(dim=-1)
        return ((1.0 - self.smoothing) * loss + self.smoothing * smooth_loss).mean()


class SatelliteSpatialSpectralDataset(Dataset):
    """PyTorch Dataset yielding (patch_tensor, tabular_tensor, target) with D4 spatial augmentation."""

    def __init__(
        self,
        patch_arrays: np.ndarray,
        tabular_features: np.ndarray,
        labels: Optional[np.ndarray] = None,
        augment: bool = False,
    ):
        self.patches = torch.tensor(patch_arrays, dtype=torch.float32)
        self.tabular = torch.tensor(tabular_features, dtype=torch.float32)
        self.labels = (
            torch.tensor(labels, dtype=torch.long) if labels is not None else None
        )
        self.augment = augment

    def __len__(self) -> int:
        return len(self.tabular)

    def __getitem__(self, idx: int):
        patch = self.patches[idx]
        tabular = self.tabular[idx]

        if self.augment:
            # D4 Dihedral group rotations and flips
            k = random.randint(0, 3)
            if k > 0:
                patch = torch.rot90(patch, k, dims=[-2, -1])
            if random.random() > 0.5:
                patch = torch.flip(patch, dims=[-1])
            if random.random() > 0.5:
                patch = torch.flip(patch, dims=[-2])

        if self.labels is not None:
            return patch, tabular, self.labels[idx]
        return patch, tabular


# =========================================================================
# 3. Model Parameter Factories & Utilities
# =========================================================================
def get_lgbm_params(num_classes: int = 6, seed: int = 42) -> Dict[str, Any]:
    return {
        "objective": "multiclass",
        "num_class": num_classes,
        "metric": "multi_logloss",
        "boosting_type": "gbdt",
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": 6,
        "min_child_samples": 15,
        "subsample": 0.85,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "n_estimators": 1200,
        "random_state": seed,
        "n_jobs": -1,
        "verbose": -1,
    }


def get_xgboost_params(num_classes: int = 6, seed: int = 42) -> Dict[str, Any]:
    return {
        "objective": "multi:softprob",
        "num_class": num_classes,
        "eval_metric": "mlogloss",
        "learning_rate": 0.03,
        "max_depth": 6,
        "min_child_weight": 2,
        "subsample": 0.85,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "n_estimators": 1200,
        "random_state": seed,
        "n_jobs": -1,
        "tree_method": "hist",
    }


def get_catboost_params(num_classes: int = 6, seed: int = 42) -> Dict[str, Any]:
    return {
        "loss_function": "MultiClass",
        "eval_metric": "MultiClass",
        "iterations": 1200,
        "learning_rate": 0.04,
        "depth": 6,
        "l2_leaf_reg": 3.0,
        "random_seed": seed,
        "verbose": False,
        "thread_count": -1,
    }


def get_extratrees_params(seed: int = 42) -> Dict[str, Any]:
    return {
        "n_estimators": 400,
        "max_depth": 22,
        "min_samples_split": 3,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "random_state": seed,
        "n_jobs": -1,
    }


def optimize_ensemble_weights(
    oof_preds_list: List[np.ndarray], y_true: np.ndarray
) -> np.ndarray:
    """Optimize ensemble weights on OOF probability predictions using SLSQP constrained optimization."""
    n_models = len(oof_preds_list)
    eps = 1e-15

    def objective(weights: np.ndarray) -> float:
        w = np.maximum(0.0, weights)
        w = w / (np.sum(w) + eps)
        blended = np.zeros_like(oof_preds_list[0])
        for weight, preds in zip(w, oof_preds_list):
            blended += weight * preds
        blended = np.clip(blended, eps, 1.0 - eps)
        blended = blended / blended.sum(axis=1, keepdims=True)
        return log_loss(y_true, blended)

    init_weights = np.ones(n_models) / n_models
    bounds = [(0.0, 1.0) for _ in range(n_models)]
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

    res = minimize(
        objective,
        init_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-7},
    )

    opt_weights = np.maximum(0.0, res.x)
    opt_weights = opt_weights / np.sum(opt_weights)
    return opt_weights


def train_pytorch_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    epochs: int = 45,
    device: torch.device = torch.device("cpu"),
    save_path: str = "./working/best_nn_model.pt",
) -> Tuple[nn.Module, float]:
    best_val_loss = float("inf")
    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        for patches, tabular, targets in train_loader:
            patches = patches.to(device)
            tabular = tabular.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(patches, tabular)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        scheduler.step()

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for patches, tabular, targets in val_loader:
                patches = patches.to(device)
                tabular = tabular.to(device)
                targets = targets.to(device)

                outputs = model(patches, tabular)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * len(targets)

                preds = outputs.argmax(dim=-1)
                correct += (preds == targets).sum().item()
                total += len(targets)

        val_loss = val_loss / max(total, 1)
        val_acc = correct / max(total, 1)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)

    model.load_state_dict(torch.load(save_path))
    return model, best_val_acc


def predict_pytorch_proba(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_tta: bool = True,
) -> np.ndarray:
    model.eval()
    all_probs = []
    transforms = [
        lambda x: x,
        lambda x: torch.rot90(x, 1, dims=[-2, -1]),
        lambda x: torch.rot90(x, 2, dims=[-2, -1]),
        lambda x: torch.rot90(x, 3, dims=[-2, -1]),
        lambda x: torch.flip(x, dims=[-1]),
        lambda x: torch.rot90(torch.flip(x, dims=[-1]), 1, dims=[-2, -1]),
        lambda x: torch.rot90(torch.flip(x, dims=[-1]), 2, dims=[-2, -1]),
        lambda x: torch.rot90(torch.flip(x, dims=[-1]), 3, dims=[-2, -1]),
    ]
    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, (list, tuple)):
                patches, tabular = batch[0], batch[1]
            else:
                patches, tabular = batch
            patches = patches.to(device)
            tabular = tabular.to(device)

            if use_tta:
                batch_probs = torch.zeros(
                    (patches.shape[0], model.classifier[-1].out_features),
                    device=device,
                )
                for t in transforms:
                    aug_patch = t(patches)
                    logits = model(aug_patch, tabular)
                    batch_probs += torch.softmax(logits, dim=-1)
                batch_probs /= len(transforms)
                probs = batch_probs.cpu().numpy()
            else:
                logits = model(patches, tabular)
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
            all_probs.append(probs)
    return np.vstack(all_probs)


# =========================================================================
# 4. Main Pipeline Execution
# =========================================================================
def main():
    set_seed(42)
    os.makedirs("./working", exist_ok=True)
    os.makedirs("./submission", exist_ok=True)

    # 1. Load Data
    train_path = "./input/train.parquet"
    test_path = "./input/test.parquet"
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    target_col = "class"
    feature_cols = get_feature_columns(train_df, target_col=target_col)

    unique_classes = sorted(train_df[target_col].unique().tolist())
    class_to_idx = {cls: idx for idx, cls in enumerate(unique_classes)}
    idx_to_class = {idx: cls for idx, cls in enumerate(unique_classes)}
    num_classes = len(unique_classes)

    y_train = train_df[target_col].map(class_to_idx).values.astype(np.int64)

    # 2. Extract Features
    X_train_fe = compute_spectral_and_spatial_features(train_df, feature_cols)
    X_test_fe = compute_spectral_and_spatial_features(test_df, feature_cols)
    tabular_feature_names = list(X_test_fe.columns)
    n_features = len(tabular_feature_names)

    # 3. Stratified 5-Fold Split
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_assignments = np.zeros(len(train_df), dtype=np.int32)
    for fold, (_, val_idx) in enumerate(skf.split(X_train_fe, y_train)):
        fold_assignments[val_idx] = fold

    # 4. Extract Spatial Patches for Neural Network
    patches_train_raw = prepare_spatial_patches(X_train_fe)
    patches_test_raw = prepare_spatial_patches(X_test_fe)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_train = len(train_df)
    n_test = len(test_df)

    oof_lgb = np.zeros((n_train, num_classes), dtype=np.float32)
    oof_xgb = np.zeros((n_train, num_classes), dtype=np.float32)
    oof_cat = np.zeros((n_train, num_classes), dtype=np.float32)
    oof_et = np.zeros((n_train, num_classes), dtype=np.float32)
    oof_nn = np.zeros((n_train, num_classes), dtype=np.float32)

    test_lgb = np.zeros((n_test, num_classes), dtype=np.float32)
    test_xgb = np.zeros((n_test, num_classes), dtype=np.float32)
    test_cat = np.zeros((n_test, num_classes), dtype=np.float32)
    test_et = np.zeros((n_test, num_classes), dtype=np.float32)
    test_nn = np.zeros((n_test, num_classes), dtype=np.float32)

    # 5. Training Loop across Folds
    for fold in range(n_splits):
        train_idx = np.where(fold_assignments != fold)[0]
        val_idx = np.where(fold_assignments == fold)[0]

        X_tr = X_train_fe.iloc[train_idx][tabular_feature_names].values.astype(
            np.float32
        )
        y_tr = y_train[train_idx]
        X_va = X_train_fe.iloc[val_idx][tabular_feature_names].values.astype(np.float32)
        y_va = y_train[val_idx]
        X_te = X_test_fe[tabular_feature_names].values.astype(np.float32)

        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)
        X_va_scaled = scaler.transform(X_va)
        X_te_scaled = scaler.transform(X_te)

        # LightGBM
        lgb_params = get_lgbm_params(num_classes=num_classes, seed=42 + fold)
        lgb_train_data = lgb.Dataset(X_tr, label=y_tr)
        lgb_val_data = lgb.Dataset(X_va, label=y_va, reference=lgb_train_data)
        lgb_model = lgb.train(
            lgb_params,
            lgb_train_data,
            num_boost_round=1200,
            valid_sets=[lgb_val_data],
            callbacks=[
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(0),
            ],
        )
        oof_lgb[val_idx] = lgb_model.predict(X_va)
        test_lgb += lgb_model.predict(X_te) / n_splits

        # XGBoost
        xgb_params = get_xgboost_params(num_classes=num_classes, seed=42 + fold)
        xgb_model = XGBClassifier(**xgb_params, early_stopping_rounds=50)
        xgb_model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        oof_xgb[val_idx] = xgb_model.predict_proba(X_va)
        test_xgb += xgb_model.predict_proba(X_te) / n_splits

        # CatBoost
        cb_params = get_catboost_params(num_classes=num_classes, seed=42 + fold)
        cb_model = CatBoostClassifier(**cb_params, early_stopping_rounds=50)
        cb_model.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=False)
        oof_cat[val_idx] = cb_model.predict_proba(X_va)
        test_cat += cb_model.predict_proba(X_te) / n_splits

        # ExtraTrees
        et_params = get_extratrees_params(seed=42 + fold)
        et_model = ExtraTreesClassifier(**et_params)
        et_model.fit(X_tr, y_tr)
        oof_et[val_idx] = et_model.predict_proba(X_va)
        test_et += et_model.predict_proba(X_te) / n_splits

        # SpatialSpectralNet (PyTorch Dual-Stream)
        patches_tr = patches_train_raw[train_idx]
        patches_va = patches_train_raw[val_idx]

        train_ds = SatelliteSpatialSpectralDataset(
            patches_tr, X_tr_scaled, y_tr, augment=True
        )
        val_ds = SatelliteSpatialSpectralDataset(patches_va, X_va_scaled, y_va)
        test_ds = SatelliteSpatialSpectralDataset(patches_test_raw, X_te_scaled)

        train_loader = DataLoader(
            train_ds, batch_size=64, shuffle=True, drop_last=False
        )
        val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)

        nn_model = SpatialSpectralNet(
            num_tabular_features=n_features,
            num_classes=num_classes,
            conv_dim=64,
            mlp_dim=128,
            head_dim=64,
            dropout=0.2,
        ).to(device)

        criterion = LabelSmoothingCrossEntropy(smoothing=0.05)
        optimizer = AdamW(nn_model.parameters(), lr=1.5e-3, weight_decay=1e-3)
        scheduler = CosineAnnealingLR(optimizer, T_max=45, eta_min=1e-6)

        nn_save_path = f"./working/spatial_spectral_net_fold{fold}.pt"
        best_nn, _ = train_pytorch_model(
            nn_model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            scheduler,
            epochs=45,
            device=device,
            save_path=nn_save_path,
        )

        oof_nn[val_idx] = predict_pytorch_proba(
            best_nn, val_loader, device, use_tta=True
        )
        test_nn += (
            predict_pytorch_proba(best_nn, test_loader, device, use_tta=True)
            / n_splits
        )

    # 6. Optimized Ensemble Blending using SLSQP
    oof_models = [oof_lgb, oof_xgb, oof_cat, oof_et, oof_nn]
    test_models = [test_lgb, test_xgb, test_cat, test_et, test_nn]

    opt_weights = optimize_ensemble_weights(oof_models, y_train)

    oof_ensemble = np.zeros_like(oof_lgb)
    test_ensemble = np.zeros_like(test_lgb)
    for w, oof_p, test_p in zip(opt_weights, oof_models, test_models):
        oof_ensemble += w * oof_p
        test_ensemble += w * test_p

    oof_preds = oof_ensemble.argmax(axis=1)
    final_score = accuracy_score(y_train, oof_preds)

    # 7. Generate Submission
    test_pred_indices = test_ensemble.argmax(axis=1)
    test_pred_classes = [idx_to_class[int(idx)] for idx in test_pred_indices]

    submission_df = pd.DataFrame(
        {
            "id": np.arange(len(test_pred_classes), dtype=np.int64),
            "class": test_pred_classes,
        }
    )
    submission_path = "./submission/submission.csv"
    submission_df.to_csv(submission_path, index=False)

    print(f"Final Validation Score: {final_score}")


if __name__ == "__main__":
    main()