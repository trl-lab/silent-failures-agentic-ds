import os
import pickle
import numpy as np
import pandas as pd
import scipy.optimize as opt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    IsolationForest,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.preprocessing import RobustScaler

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier


# ==============================================================================
# 1. FEATURE ENGINEERING CLASS
# ==============================================================================
class ClimateFeatureEngineer:

    def __init__(self, n_pca_components=5, knn_k=5):
        self.n_pca_components = n_pca_components
        self.knn_k = knn_k
        self.scaler = RobustScaler()
        self.pca = PCA(n_components=n_pca_components, random_state=42)
        self.nn_model = NearestNeighbors(n_neighbors=knn_k + 1, metric="euclidean")

    def _get_base_param_cols(self, df):
        ignore_cols = {
            "id",
            "outcome",
            "target",
            "Study",
            "Run",
            "study",
            "run",
            "batch",
            "Batch",
        }
        return [col for col in df.columns if col not in ignore_cols]

    def fit_transform(self, X_df):
        param_cols = self._get_base_param_cols(X_df)
        X_raw = X_df[param_cols].values.astype(np.float64)

        features = [X_raw]

        # 1. Hypercube Boundary and Centroid Geometry Features
        dist_to_center_l2 = np.linalg.norm(X_raw - 0.5, axis=1, keepdims=True)
        dist_to_center_l1 = np.sum(np.abs(X_raw - 0.5), axis=1, keepdims=True)
        dist_to_center_linf = np.max(np.abs(X_raw - 0.5), axis=1, keepdims=True)

        dist_to_boundary_min = np.min(
            np.minimum(X_raw, 1.0 - X_raw), axis=1, keepdims=True
        )
        dist_to_boundary_mean = np.mean(
            np.minimum(X_raw, 1.0 - X_raw), axis=1, keepdims=True
        )

        features.extend(
            [
                dist_to_center_l2,
                dist_to_center_l1,
                dist_to_center_linf,
                dist_to_boundary_min,
                dist_to_boundary_mean,
            ]
        )

        # 2. Non-linear Physical Dynamics Transformations
        sin_params = np.sin(np.pi * X_raw)
        cos_params = np.cos(np.pi * X_raw)
        sq_params = np.square(X_raw - 0.5)
        features.extend([sin_params, cos_params, sq_params])

        # 3. Parameter Statistics across physical components
        row_mean = np.mean(X_raw, axis=1, keepdims=True)
        row_std = np.std(X_raw, axis=1, keepdims=True)
        row_skew = np.mean((X_raw - row_mean) ** 3, axis=1, keepdims=True) / (
            row_std**3 + 1e-6
        )
        features.extend([row_mean, row_std, row_skew])

        # 4. Pairwise Non-linear Parameter Interactions
        variances = np.var(X_raw, axis=0)
        self.top_idx_ = np.argsort(variances)[-6:]
        interaction_list = []
        for i in range(len(self.top_idx_)):
            for j in range(i + 1, len(self.top_idx_)):
                col1, col2 = self.top_idx_[i], self.top_idx_[j]
                prod = (X_raw[:, col1] * X_raw[:, col2])[:, None]
                ratio = ((X_raw[:, col1] + 1e-4) / (X_raw[:, col2] + 1e-4))[:, None]
                interaction_list.extend([prod, ratio])
        features.extend(interaction_list)

        X_all = np.hstack(features)
        X_scaled = self.scaler.fit_transform(X_all)

        # 5. Latent Dimensionality Features (PCA)
        pca_feats = self.pca.fit_transform(X_scaled)

        # 6. KNN Local Density Metrics
        self.nn_model.fit(X_scaled)
        distances, _ = self.nn_model.kneighbors(X_scaled)
        knn_mean_dist = np.mean(distances[:, 1:], axis=1, keepdims=True)
        knn_max_dist = np.max(distances[:, 1:], axis=1, keepdims=True)

        X_final = np.hstack([X_scaled, pca_feats, knn_mean_dist, knn_max_dist])
        return np.nan_to_num(X_final, nan=0.0, posinf=0.0, neginf=0.0)

    def transform(self, X_df):
        param_cols = self._get_base_param_cols(X_df)
        X_raw = X_df[param_cols].values.astype(np.float64)

        features = [X_raw]

        dist_to_center_l2 = np.linalg.norm(X_raw - 0.5, axis=1, keepdims=True)
        dist_to_center_l1 = np.sum(np.abs(X_raw - 0.5), axis=1, keepdims=True)
        dist_to_center_linf = np.max(np.abs(X_raw - 0.5), axis=1, keepdims=True)

        dist_to_boundary_min = np.min(
            np.minimum(X_raw, 1.0 - X_raw), axis=1, keepdims=True
        )
        dist_to_boundary_mean = np.mean(
            np.minimum(X_raw, 1.0 - X_raw), axis=1, keepdims=True
        )

        features.extend(
            [
                dist_to_center_l2,
                dist_to_center_l1,
                dist_to_center_linf,
                dist_to_boundary_min,
                dist_to_boundary_mean,
            ]
        )

        sin_params = np.sin(np.pi * X_raw)
        cos_params = np.cos(np.pi * X_raw)
        sq_params = np.square(X_raw - 0.5)
        features.extend([sin_params, cos_params, sq_params])

        row_mean = np.mean(X_raw, axis=1, keepdims=True)
        row_std = np.std(X_raw, axis=1, keepdims=True)
        row_skew = np.mean((X_raw - row_mean) ** 3, axis=1, keepdims=True) / (
            row_std**3 + 1e-6
        )
        features.extend([row_mean, row_std, row_skew])

        interaction_list = []
        for i in range(len(self.top_idx_)):
            for j in range(i + 1, len(self.top_idx_)):
                col1, col2 = self.top_idx_[i], self.top_idx_[j]
                prod = (X_raw[:, col1] * X_raw[:, col2])[:, None]
                ratio = ((X_raw[:, col1] + 1e-4) / (X_raw[:, col2] + 1e-4))[:, None]
                interaction_list.extend([prod, ratio])
        features.extend(interaction_list)

        X_all = np.hstack(features)
        X_scaled = self.scaler.transform(X_all)

        pca_feats = self.pca.transform(X_scaled)

        distances, _ = self.nn_model.kneighbors(X_scaled, n_neighbors=self.knn_k)
        knn_mean_dist = np.mean(distances, axis=1, keepdims=True)
        knn_max_dist = np.max(distances, axis=1, keepdims=True)

        X_final = np.hstack([X_scaled, pca_feats, knn_mean_dist, knn_max_dist])
        return np.nan_to_num(X_final, nan=0.0, posinf=0.0, neginf=0.0)


# ==============================================================================
# 2. NEURAL NETWORK & LOSS DEFINITIONS
# ==============================================================================
class FocalBCEWithLogitsLoss(nn.Module):

    def __init__(
        self, alpha: float = 0.75, gamma: float = 2.0, reduction: str = "mean"
    ):
        super(FocalBCEWithLogitsLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        probs = torch.sigmoid(inputs)
        p_t = targets * probs + (1 - targets) * (1 - probs)
        alpha_factor = targets * self.alpha + (1 - targets) * (1.0 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        focal_loss = alpha_factor * modulating_factor * bce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class FeatureGatingBlock(nn.Module):

    def __init__(self, dim: int, reduction: int = 4):
        super(FeatureGatingBlock, self).__init__()
        reduced_dim = max(4, dim // reduction)
        self.gate = nn.Sequential(
            nn.Linear(dim, reduced_dim),
            nn.ReLU(),
            nn.Linear(reduced_dim, dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gate(x)


class ResBlock(nn.Module):

    def __init__(self, dim: int, dropout_rate: float = 0.2):
        super(ResBlock, self).__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)
        self.ln2 = nn.LayerNorm(dim)
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.ln1(x)
        out = self.act(self.fc1(out))
        out = self.dropout(out)
        out = self.ln2(out)
        out = self.fc2(out)
        return residual + out


class ClimateCrashResNet(nn.Module):

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_blocks: int = 2,
        dropout_rate: float = 0.25,
    ):
        super(ClimateCrashResNet, self).__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )
        self.gating = FeatureGatingBlock(hidden_dim)
        self.blocks = nn.ModuleList(
            [ResBlock(hidden_dim, dropout_rate=dropout_rate) for _ in range(num_blocks)]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate / 2.0),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        h = self.gating(h)
        for block in self.blocks:
            h = block(h)
        return self.head(h).squeeze(-1)


def create_optimizer_and_scheduler(
    model: nn.Module,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-3,
    total_epochs: int = 100,
):
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=1e-5)
    return optimizer, scheduler


def get_model_factory(random_state: int = 42, pos_weight: float = 1.0):
    return {
        "lightgbm": lambda: lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=3,
            num_leaves=7,
            min_child_samples=10,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.5,
            reg_lambda=1.0,
            scale_pos_weight=pos_weight,
            random_state=random_state,
            verbosity=-1,
        ),
        "xgboost": lambda: xgb.XGBClassifier(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=3,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.5,
            reg_lambda=1.0,
            scale_pos_weight=pos_weight,
            random_state=random_state,
            eval_metric="auc",
            tree_method="hist",
        ),
        "catboost": lambda: CatBoostClassifier(
            iterations=300,
            learning_rate=0.03,
            depth=4,
            l2_leaf_reg=3.0,
            scale_pos_weight=pos_weight,
            random_seed=random_state,
            verbose=False,
        ),
        "extratrees": lambda: ExtraTreesClassifier(
            n_estimators=250,
            max_depth=6,
            min_samples_split=4,
            min_samples_leaf=2,
            class_weight="balanced" if pos_weight != 1.0 else None,
            random_state=random_state,
            n_jobs=-1,
        ),
        "hist_gbdt": lambda: HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.03,
            max_depth=4,
            min_samples_leaf=10,
            l2_regularization=1.0,
            class_weight="balanced" if pos_weight != 1.0 else None,
            random_state=random_state,
        ),
        "logistic_reg": lambda: LogisticRegression(
            C=0.5,
            penalty="l2",
            solver="lbfgs",
            max_iter=1000,
            class_weight="balanced" if pos_weight != 1.0 else None,
            random_state=random_state,
        ),
    }


# ==============================================================================
# 3. MAIN TRAINING AND EVALUATION PIPELINE
# ==============================================================================
def main():
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    os.makedirs("./working", exist_ok=True)
    os.makedirs("./submission", exist_ok=True)

    # 1. Load Datasets
    df_train_raw = pd.read_parquet("./input/train.parquet")
    df_test_raw = pd.read_parquet("./input/test.parquet")

    target_col = "outcome"
    drop_batch_cols = [
        col
        for col in ["Study", "Run", "study", "run", "batch", "Batch"]
        if col in df_train_raw.columns
    ]
    if drop_batch_cols:
        df_train_raw = df_train_raw.drop(columns=drop_batch_cols)
        df_test_raw = df_test_raw.drop(
            columns=[c for c in drop_batch_cols if c in df_test_raw.columns]
        )

    y = df_train_raw[target_col].values.astype(int)
    X_raw = df_train_raw.drop(columns=[target_col])

    if "id" in df_test_raw.columns:
        test_ids = df_test_raw["id"].values
        X_test_raw = df_test_raw.drop(columns=["id"])
    else:
        test_ids = np.arange(len(df_test_raw))
        X_test_raw = df_test_raw.copy()

    # 5-Fold Stratified Cross-Validation setup
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    model_names = [
        "climate_resnet",
        "lightgbm",
        "xgboost",
        "catboost",
        "extratrees",
        "hist_gbdt",
        "logistic_reg",
    ]

    oof_predictions_dict = {m: np.zeros(len(y)) for m in model_names}
    test_predictions_dict = {m: np.zeros(len(X_test_raw)) for m in model_names}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_raw, y)):
        X_tr_raw = X_raw.iloc[train_idx]
        X_va_raw = X_raw.iloc[val_idx]
        y_train = y[train_idx]
        y_val = y[val_idx]

        # Feature transformation strictly isolated per fold
        fe = ClimateFeatureEngineer(n_pca_components=5, knn_k=5)
        X_train = fe.fit_transform(X_tr_raw)
        X_val = fe.transform(X_va_raw)
        X_test = fe.transform(X_test_raw)

        input_dim = X_train.shape[1]
        num_pos = np.sum(y_train == 1)
        num_neg = np.sum(y_train == 0)
        pos_weight = float(num_neg) / float(num_pos + 1e-6)

        # 2. Train PyTorch ClimateCrashResNet with Focal Loss
        train_ds = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
        )
        val_ds = TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.float32),
        )
        test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

        train_loader = DataLoader(
            train_ds, batch_size=32, shuffle=True, drop_last=False, num_workers=0
        )
        val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)

        nn_model = ClimateCrashResNet(
            input_dim=input_dim, hidden_dim=64, num_blocks=2, dropout_rate=0.25
        ).to(device)
        loss_fn = FocalBCEWithLogitsLoss(alpha=0.75, gamma=2.0)
        optimizer, scheduler = create_optimizer_and_scheduler(
            nn_model, learning_rate=1e-3, weight_decay=1e-3, total_epochs=80
        )

        best_val_auc = -1.0
        best_val_preds_nn = None
        best_test_preds_nn = None

        for epoch in range(1, 81):
            nn_model.train()
            for bx, by in train_loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                logits = nn_model(bx)
                loss = loss_fn(logits, by)
                loss.backward()
                optimizer.step()
            scheduler.step()

            nn_model.eval()
            val_preds_list = []
            with torch.no_grad():
                for bx, _ in val_loader:
                    bx = bx.to(device)
                    logits = nn_model(bx)
                    probs = torch.sigmoid(logits).cpu().numpy()
                    val_preds_list.append(probs)

            val_preds_nn = np.concatenate(val_preds_list)
            val_auc = roc_auc_score(y_val, val_preds_nn)

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_val_preds_nn = val_preds_nn
                with torch.no_grad():
                    test_logits = nn_model(test_tensor)
                    best_test_preds_nn = torch.sigmoid(test_logits).cpu().numpy()

        oof_predictions_dict["climate_resnet"][val_idx] = best_val_preds_nn
        test_predictions_dict["climate_resnet"] += best_test_preds_nn / n_splits

        # 3. Train GBDT and Linear Models
        model_factories = get_model_factory(random_state=42, pos_weight=pos_weight)

        for name, factory in model_factories.items():
            model = factory()
            if name == "xgboost":
                model.fit(
                    X_train,
                    y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=False,
                )
            elif name == "catboost":
                model.fit(
                    X_train,
                    y_train,
                    eval_set=(X_val, y_val),
                    early_stopping_rounds=50,
                    verbose=False,
                )
            elif name == "lightgbm":
                model.fit(
                    X_train,
                    y_train,
                    eval_set=[(X_val, y_val)],
                    callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
                )
            else:
                model.fit(X_train, y_train)

            val_probs = model.predict_proba(X_val)[:, 1]
            test_probs = model.predict_proba(X_test)[:, 1]

            oof_predictions_dict[name][val_idx] = val_probs
            test_predictions_dict[name] += test_probs / n_splits

    # 4. SLSQP Ensemble Blending Optimization on OOF predictions
    oof_matrix = np.column_stack([oof_predictions_dict[m] for m in model_names])
    test_matrix = np.column_stack([test_predictions_dict[m] for m in model_names])

    num_models = len(model_names)

    def objective(weights):
        weights = np.array(weights)
        weights = weights / (np.sum(weights) + 1e-8)
        blend_oof = oof_matrix @ weights
        return -roc_auc_score(y, blend_oof)

    init_weights = np.ones(num_models) / num_models
    bounds = [(0, 1) for _ in range(num_models)]
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

    res = opt.minimize(
        objective,
        init_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    opt_weights = res.x / np.sum(res.x)

    final_val_preds = oof_matrix @ opt_weights
    final_val_score = roc_auc_score(y, final_val_preds)
    final_test_preds = test_matrix @ opt_weights

    # 5. Output Submission File
    submission_df = pd.DataFrame({"id": test_ids, "outcome": final_test_preds})
    submission_df.to_csv("./submission/submission.csv", index=False)

    print(f"Final Validation Score: {final_val_score:.5f}")


if __name__ == "__main__":
    main()