import copy
import json
import math
import os
import warnings
from typing import Any, Dict, List, Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import kurtosis, skew
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import (
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)
from sklearn.random_projection import GaussianRandomProjection
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb

try:
    import catboost as cb

    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

warnings.filterwarnings("ignore")


# =========================================================================
# 1. Feature Engineering & Preprocessing
# =========================================================================
def extract_row_aggregates(df: pd.DataFrame, prefix: str = "") -> pd.DataFrame:
    mat = df.values.astype(np.float32)
    feats = pd.DataFrame(index=df.index)

    feats[f"{prefix}nonzero_count"] = np.count_nonzero(mat, axis=1)
    feats[f"{prefix}nonzero_ratio"] = feats[f"{prefix}nonzero_count"] / mat.shape[1]
    feats[f"{prefix}mean"] = np.mean(mat, axis=1)
    feats[f"{prefix}std"] = np.std(mat, axis=1)
    feats[f"{prefix}sum"] = np.sum(mat, axis=1)
    feats[f"{prefix}max"] = np.max(mat, axis=1)
    feats[f"{prefix}min"] = np.min(mat, axis=1)
    feats[f"{prefix}median"] = np.median(mat, axis=1)
    feats[f"{prefix}q25"] = np.percentile(mat, 25, axis=1)
    feats[f"{prefix}q75"] = np.percentile(mat, 75, axis=1)
    feats[f"{prefix}iqr"] = feats[f"{prefix}q75"] - feats[f"{prefix}q25"]
    feats[f"{prefix}l1_norm"] = np.sum(np.abs(mat), axis=1)
    feats[f"{prefix}l2_norm"] = np.sqrt(np.sum(np.square(mat), axis=1) + 1e-8)
    feats[f"{prefix}skew"] = skew(mat, axis=1, nan_policy="omit")
    feats[f"{prefix}kurtosis"] = kurtosis(mat, axis=1, nan_policy="omit")

    eps = 1e-7
    p = np.abs(mat) / (np.sum(np.abs(mat), axis=1, keepdims=True) + eps)
    feats[f"{prefix}entropy"] = -np.sum(p * np.log(p + eps), axis=1)

    feats.fillna(0.0, inplace=True)
    return feats


def compute_fold_manifold_features(
    X_tr: np.ndarray,
    X_va: np.ndarray,
    X_test: np.ndarray,
    n_svd: int = 15,
    n_pca: int = 10,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    scaler = StandardScaler()
    X_tr_std = scaler.fit_transform(X_tr)
    X_va_std = scaler.transform(X_va)
    X_test_std = scaler.transform(X_test)

    svd = TruncatedSVD(n_components=n_svd, random_state=random_state)
    svd_tr = svd.fit_transform(X_tr)
    svd_va = svd.transform(X_va)
    svd_test = svd.transform(X_test)

    pca = PCA(n_components=n_pca, random_state=random_state)
    pca_tr = pca.fit_transform(X_tr_std)
    pca_va = pca.transform(X_va_std)
    pca_test = pca.transform(X_test_std)

    nn = NearestNeighbors(n_neighbors=26, metric="euclidean", n_jobs=-1)
    nn.fit(X_tr_std)

    tr_dists, _ = nn.kneighbors(X_tr_std, n_neighbors=26)
    tr_knn_5 = np.mean(tr_dists[:, 1:6], axis=1, keepdims=True)
    tr_knn_15 = np.mean(tr_dists[:, 1:16], axis=1, keepdims=True)
    tr_knn_25 = np.mean(tr_dists[:, 1:26], axis=1, keepdims=True)
    tr_knn_min = tr_dists[:, 1:2]
    tr_knn_std = np.std(tr_dists[:, 1:26], axis=1, keepdims=True)
    tr_knn_feats = np.hstack([tr_knn_5, tr_knn_15, tr_knn_25, tr_knn_min, tr_knn_std])

    va_dists, _ = nn.kneighbors(X_va_std, n_neighbors=25)
    va_knn_5 = np.mean(va_dists[:, :5], axis=1, keepdims=True)
    va_knn_15 = np.mean(va_dists[:, :15], axis=1, keepdims=True)
    va_knn_25 = np.mean(va_dists[:, :25], axis=1, keepdims=True)
    va_knn_min = va_dists[:, 0:1]
    va_knn_std = np.std(va_dists[:, :25], axis=1, keepdims=True)
    va_knn_feats = np.hstack([va_knn_5, va_knn_15, va_knn_25, va_knn_min, va_knn_std])

    test_dists, _ = nn.kneighbors(X_test_std, n_neighbors=25)
    test_knn_5 = np.mean(test_dists[:, :5], axis=1, keepdims=True)
    test_knn_15 = np.mean(test_dists[:, :15], axis=1, keepdims=True)
    test_knn_25 = np.mean(test_dists[:, :25], axis=1, keepdims=True)
    test_knn_min = test_dists[:, 0:1]
    test_knn_std = np.std(test_dists[:, :25], axis=1, keepdims=True)
    test_knn_feats = np.hstack([test_knn_5, test_knn_15, test_knn_25, test_knn_min, test_knn_std])

    tr_all = np.hstack([X_tr, svd_tr, pca_tr, tr_knn_feats]).astype(np.float32)
    va_all = np.hstack([X_va, svd_va, pca_va, va_knn_feats]).astype(np.float32)
    test_all = np.hstack([X_test, svd_test, pca_test, test_knn_feats]).astype(np.float32)

    return tr_all, va_all, test_all


def process_and_engineer_features(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    target_col = "Activity"
    feature_cols = [col for col in train_df.columns if col != target_col]

    y_train = train_df[target_col].values.astype(np.int32)
    X_train_raw = train_df[feature_cols].copy()
    X_test_raw = test_df[feature_cols].copy()

    # Identify and filter invariant/zero-variance features in train
    std_train = X_train_raw.std(axis=0)
    non_constant_cols = std_train[std_train > 1e-8].index.tolist()

    X_train_clean = X_train_raw[non_constant_cols].copy()
    X_test_clean = X_test_raw[non_constant_cols].copy()

    # Categorize descriptors: binary structural fingerprints vs continuous properties
    binary_cols = []
    continuous_cols = []
    for col in non_constant_cols:
        unique_vals = np.unique(X_train_clean[col].dropna().values)
        if len(unique_vals) <= 2 and np.all(np.isin(unique_vals, [0, 1])):
            binary_cols.append(col)
        else:
            continuous_cols.append(col)

    # Row-level molecular aggregations
    train_global_agg = extract_row_aggregates(X_train_clean, prefix="mol_all_")
    test_global_agg = extract_row_aggregates(X_test_clean, prefix="mol_all_")

    if len(binary_cols) > 0:
        train_bin_agg = extract_row_aggregates(
            X_train_clean[binary_cols], prefix="mol_bin_"
        )
        test_bin_agg = extract_row_aggregates(
            X_test_clean[binary_cols], prefix="mol_bin_"
        )
    else:
        train_bin_agg = pd.DataFrame(index=train_df.index)
        test_bin_agg = pd.DataFrame(index=test_df.index)

    if len(continuous_cols) > 0:
        train_cont_agg = extract_row_aggregates(
            X_train_clean[continuous_cols], prefix="mol_cont_"
        )
        test_cont_agg = extract_row_aggregates(
            X_test_clean[continuous_cols], prefix="mol_cont_"
        )
    else:
        train_cont_agg = pd.DataFrame(index=train_df.index)
        test_cont_agg = pd.DataFrame(index=test_df.index)

    # Concatenate deterministic row-level feature representations
    train_engineered = pd.concat(
        [
            X_train_clean,
            train_global_agg,
            train_bin_agg,
            train_cont_agg,
        ],
        axis=1,
    )

    test_engineered = pd.concat(
        [
            X_test_clean,
            test_global_agg,
            test_bin_agg,
            test_cont_agg,
        ],
        axis=1,
    )

    return train_engineered, test_engineered, y_train


# =========================================================================
# 2. Model Architectures & Parameter Configurations
# =========================================================================
class SqueezeExcitation1D(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        reduced_dim = max(8, channels // reduction)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_dim, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.fc(x)
        return x * weights


class ResNetBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout_rate: float = 0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.se = SqueezeExcitation1D(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.se(self.block(x))


class TabularResNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_blocks: int = 3,
        dropout_rate: float = 0.25,
        input_dropout: float = 0.1,
    ):
        super().__init__()
        self.input_drop = nn.Dropout(p=input_dropout)
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )

        self.blocks = nn.ModuleList(
            [
                ResNetBlock(hidden_dim=hidden_dim, dropout_rate=dropout_rate)
                for _ in range(num_blocks)
            ]
        )

        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_drop(x)
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h)
        logits = self.head(h).squeeze(-1)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits)


class SmoothBCEWithLogitsLoss(nn.Module):
    def __init__(self, smoothing: float = 0.005):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets_smoothed = targets * (1.0 - 2.0 * self.smoothing) + self.smoothing
        return F.binary_cross_entropy_with_logits(logits, targets_smoothed)


def build_optimizer_and_scheduler(
    model: nn.Module,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    epochs: int = 35,
    min_lr: float = 1e-6,
) -> Tuple[torch.optim.Optimizer, Any]:
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith(".bias") or "norm" in name.lower():
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    optimizer = AdamW(optim_groups, lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=min_lr)
    return optimizer, scheduler


def mixup_data(
    x: torch.Tensor, y: torch.Tensor, alpha: float = 0.2
) -> Tuple[torch.Tensor, torch.Tensor]:
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    mixed_y = lam * y + (1 - lam) * y[index]
    return mixed_x, mixed_y


def calibrate_probabilities(
    val_probs: np.ndarray, y_val: np.ndarray, test_probs: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, float]:
    eps = 1e-7
    val_probs_clipped = np.clip(val_probs, eps, 1.0 - eps)
    test_probs_clipped = np.clip(test_probs, eps, 1.0 - eps)

    val_logits = np.log(val_probs_clipped / (1.0 - val_probs_clipped))
    test_logits = np.log(test_probs_clipped / (1.0 - test_probs_clipped))

    def nll_objective(temperature: float) -> float:
        scaled_logits = val_logits / max(temperature, 1e-4)
        scaled_probs = 1.0 / (1.0 + np.exp(-np.clip(scaled_logits, -35.0, 35.0)))
        return log_loss(y_val, scaled_probs)

    res = minimize_scalar(nll_objective, bounds=(0.3, 3.5), method="bounded")
    best_temp = float(res.x)

    cal_val_probs = 1.0 / (
        1.0 + np.exp(-np.clip(val_logits / best_temp, -35.0, 35.0))
    )
    cal_test_probs = 1.0 / (
        1.0 + np.exp(-np.clip(test_logits / best_temp, -35.0, 35.0))
    )

    return cal_val_probs, cal_test_probs, best_temp


def get_extratrees_model(seed: int = 42) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=350,
        max_features=0.25,
        min_samples_split=4,
        min_samples_leaf=2,
        max_depth=25,
        bootstrap=False,
        random_state=seed,
        n_jobs=-1,
    )


def get_lgbm_params(seed: int = 42) -> Dict[str, Any]:
    return {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": 6,
        "feature_fraction": 0.3,
        "bagging_fraction": 0.75,
        "bagging_freq": 1,
        "min_child_samples": 20,
        "lambda_l1": 0.2,
        "lambda_l2": 3.0,
        "verbose": -1,
        "random_state": seed,
        "n_jobs": -1,
    }


def get_xgb_params(seed: int = 42) -> Dict[str, Any]:
    return {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "learning_rate": 0.03,
        "max_depth": 5,
        "subsample": 0.75,
        "colsample_bytree": 0.3,
        "colsample_bylevel": 0.7,
        "reg_alpha": 0.2,
        "reg_lambda": 4.0,
        "min_child_weight": 3,
        "random_state": seed,
        "n_jobs": -1,
        "tree_method": "hist",
    }


def get_catboost_params(seed: int = 42) -> Dict[str, Any]:
    return {
        "loss_function": "Logloss",
        "eval_metric": "Logloss",
        "learning_rate": 0.03,
        "depth": 6,
        "rsm": 0.3,
        "l2_leaf_reg": 5.0,
        "random_seed": seed,
        "verbose": False,
        "thread_count": -1,
    }


# =========================================================================
# 3. Main Training, Ensembling, and Submission Pipeline
# =========================================================================
def main():
    input_dir = "./input"
    submission_dir = "./submission"
    os.makedirs(submission_dir, exist_ok=True)

    train_path = os.path.join(input_dir, "train.parquet")
    test_path = os.path.join(input_dir, "test.parquet")

    train_raw = pd.read_parquet(train_path)
    test_raw = pd.read_parquet(test_path)

    # Feature Engineering
    train_feats, test_feats, y_all = process_and_engineer_features(train_raw, test_raw)
    feature_cols = train_feats.columns.tolist()

    X_test_all = test_feats.values.astype(np.float32)
    n_train = len(train_feats)
    n_test = len(test_feats)
    n_splits = 5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set up 5-Fold Stratified Validation
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    folds = list(skf.split(train_feats, y_all))

    oof_preds = {
        "resnet": np.zeros(n_train, dtype=np.float64),
        "extratrees": np.zeros(n_train, dtype=np.float64),
        "lgbm": np.zeros(n_train, dtype=np.float64),
        "xgb": np.zeros(n_train, dtype=np.float64),
    }
    test_preds = {
        "resnet": np.zeros(n_test, dtype=np.float64),
        "extratrees": np.zeros(n_test, dtype=np.float64),
        "lgbm": np.zeros(n_test, dtype=np.float64),
        "xgb": np.zeros(n_test, dtype=np.float64),
    }

    if CATBOOST_AVAILABLE:
        oof_preds["catboost"] = np.zeros(n_train, dtype=np.float64)
        test_preds["catboost"] = np.zeros(n_test, dtype=np.float64)

    # Train models over CV folds
    for fold, (train_idx, val_idx) in enumerate(folds):
        X_tr_raw = train_feats.iloc[train_idx].values.astype(np.float32)
        y_tr = y_all[train_idx]
        X_va_raw = train_feats.iloc[val_idx].values.astype(np.float32)
        y_va = y_all[val_idx]

        # Extract per-fold unsupervised manifold representations without data leakage
        X_tr, X_va, X_test_fold = compute_fold_manifold_features(
            X_tr_raw, X_va_raw, X_test_all, n_svd=15, n_pca=10, random_state=42 + fold
        )

        # --- A. Tabular ResNet with Mixup Regularization ---
        scaler = StandardScaler()
        X_tr_norm = scaler.fit_transform(X_tr)
        X_va_norm = scaler.transform(X_va)
        X_test_norm = scaler.transform(X_test_fold)

        tr_dataset = TensorDataset(
            torch.tensor(X_tr_norm, dtype=torch.float32),
            torch.tensor(y_tr, dtype=torch.float32),
        )
        va_dataset = TensorDataset(
            torch.tensor(X_va_norm, dtype=torch.float32),
            torch.tensor(y_va, dtype=torch.float32),
        )

        tr_loader = DataLoader(
            tr_dataset, batch_size=128, shuffle=True, drop_last=False
        )
        va_loader = DataLoader(
            va_dataset, batch_size=256, shuffle=False, drop_last=False
        )

        resnet_model = TabularResNet(
            input_dim=X_tr.shape[1],
            hidden_dim=256,
            num_blocks=3,
            dropout_rate=0.25,
            input_dropout=0.1,
        ).to(device)

        criterion = SmoothBCEWithLogitsLoss(smoothing=0.005)
        epochs = 35
        optimizer, scheduler = build_optimizer_and_scheduler(
            resnet_model, lr=1e-3, weight_decay=1e-4, epochs=epochs
        )

        best_val_loss = float("inf")
        best_model_weights = None

        for epoch in range(1, epochs + 1):
            resnet_model.train()
            train_loss = 0.0
            for bx, by in tr_loader:
                bx, by = bx.to(device), by.to(device)
                if np.random.rand() < 0.6:
                    bx, by = mixup_data(bx, by, alpha=0.2)
                optimizer.zero_grad()
                logits = resnet_model(bx)
                loss = criterion(logits, by)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(resnet_model.parameters(), max_norm=2.0)
                optimizer.step()
                train_loss += loss.item() * len(by)
            train_loss /= len(tr_dataset)
            scheduler.step()

            resnet_model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for bx, by in va_loader:
                    bx, by = bx.to(device), by.to(device)
                    logits = resnet_model(bx)
                    loss = nn.functional.binary_cross_entropy_with_logits(logits, by)
                    val_loss += loss.item() * len(by)
            val_loss /= len(va_dataset)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_weights = copy.deepcopy(resnet_model.state_dict())

        resnet_model.load_state_dict(best_model_weights)
        resnet_model.eval()

        with torch.no_grad():
            va_tensor = torch.tensor(X_va_norm, dtype=torch.float32).to(device)
            raw_val_probs_resnet = resnet_model.predict_proba(va_tensor).cpu().numpy()

            test_tensor = torch.tensor(X_test_norm, dtype=torch.float32).to(device)
            raw_test_probs_resnet = resnet_model.predict_proba(test_tensor).cpu().numpy()

        cal_val_resnet, cal_test_resnet, _ = calibrate_probabilities(
            raw_val_probs_resnet, y_va, raw_test_probs_resnet
        )
        oof_preds["resnet"][val_idx] = cal_val_resnet
        test_preds["resnet"] += cal_test_resnet / n_splits

        # --- B. ExtraTrees Classifier ---
        et_model = get_extratrees_model(seed=42 + fold)
        et_model.fit(X_tr, y_tr)
        raw_val_probs_et = et_model.predict_proba(X_va)[:, 1]
        raw_test_probs_et = et_model.predict_proba(X_test_fold)[:, 1]

        cal_val_et, cal_test_et, _ = calibrate_probabilities(
            raw_val_probs_et, y_va, raw_test_probs_et
        )
        oof_preds["extratrees"][val_idx] = cal_val_et
        test_preds["extratrees"] += cal_test_et / n_splits

        # --- C. LightGBM ---
        lgb_params = get_lgbm_params(seed=42 + fold)
        lgb_train = lgb.Dataset(X_tr, label=y_tr)
        lgb_val = lgb.Dataset(X_va, label=y_va, reference=lgb_train)

        lgb_model = lgb.train(
            lgb_params,
            lgb_train,
            num_boost_round=1200,
            valid_sets=[lgb_val],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
        )

        raw_val_probs_lgb = lgb_model.predict(X_va, num_iteration=lgb_model.best_iteration)
        raw_test_probs_lgb = lgb_model.predict(
            X_test_fold, num_iteration=lgb_model.best_iteration
        )

        cal_val_lgb, cal_test_lgb, _ = calibrate_probabilities(
            raw_val_probs_lgb, y_va, raw_test_probs_lgb
        )
        oof_preds["lgbm"][val_idx] = cal_val_lgb
        test_preds["lgbm"] += cal_test_lgb / n_splits

        # --- D. XGBoost ---
        xgb_params = get_xgb_params(seed=42 + fold)
        xgb_model = xgb.XGBClassifier(
            **xgb_params,
            n_estimators=1200,
            early_stopping_rounds=50,
        )
        xgb_model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            verbose=False,
        )

        raw_val_probs_xgb = xgb_model.predict_proba(X_va)[:, 1]
        raw_test_probs_xgb = xgb_model.predict_proba(X_test_fold)[:, 1]

        cal_val_xgb, cal_test_xgb, _ = calibrate_probabilities(
            raw_val_probs_xgb, y_va, raw_test_probs_xgb
        )
        oof_preds["xgb"][val_idx] = cal_val_xgb
        test_preds["xgb"] += cal_test_xgb / n_splits

        # --- E. CatBoost (if available) ---
        if CATBOOST_AVAILABLE:
            cb_params = get_catboost_params(seed=42 + fold)
            cb_model = cb.CatBoostClassifier(
                **cb_params,
                iterations=1200,
                early_stopping_rounds=50,
            )
            cb_model.fit(
                X_tr,
                y_tr,
                eval_set=(X_va, y_va),
                verbose=False,
            )

            raw_val_probs_cb = cb_model.predict_proba(X_va)[:, 1]
            raw_test_probs_cb = cb_model.predict_proba(X_test_fold)[:, 1]

            cal_val_cb, cal_test_cb, _ = calibrate_probabilities(
                raw_val_probs_cb, y_va, raw_test_probs_cb
            )
            oof_preds["catboost"][val_idx] = cal_val_cb
            test_preds["catboost"] += cal_test_cb / n_splits

    # Constrained Ensemble Optimization (SLSQP)
    model_names = list(oof_preds.keys())

    def loss_func(weights):
        weights = weights / np.sum(weights)
        blended_oof = np.zeros(n_train, dtype=np.float64)
        for i, name in enumerate(model_names):
            blended_oof += weights[i] * oof_preds[name]
        blended_oof = np.clip(blended_oof, 1e-15, 1 - 1e-15)
        return log_loss(y_all, blended_oof)

    init_weights = np.ones(len(model_names)) / len(model_names)
    bounds = [(0.0, 1.0) for _ in range(len(model_names))]
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

    res = minimize(
        loss_func,
        init_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    opt_weights = res.x / np.sum(res.x)

    # Compute Final Out-of-Fold Score
    final_oof_probs = np.zeros(n_train, dtype=np.float64)
    for i, name in enumerate(model_names):
        final_oof_probs += opt_weights[i] * oof_preds[name]

    final_oof_probs = np.clip(final_oof_probs, 1e-15, 1 - 1e-15)
    final_score = log_loss(y_all, final_oof_probs)

    # Generate Submission Predictions
    final_test_probs = np.zeros(n_test, dtype=np.float64)
    for i, name in enumerate(model_names):
        final_test_probs += opt_weights[i] * test_preds[name]

    final_test_probs = np.clip(final_test_probs, 1e-15, 1 - 1e-15)

    sub_df = pd.DataFrame(
        {
            "id": np.arange(n_test, dtype=np.int32),
            "Activity": final_test_probs.astype(np.float64),
        }
    )

    sub_path = os.path.join(submission_dir, "submission.csv")
    sub_df.to_csv(sub_path, index=False)

    print(f"Final Validation Score: {final_score}")


if __name__ == "__main__":
    main()