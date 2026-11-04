import copy
import os
import re
import warnings
from typing import Any, Dict, Tuple

import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb

warnings.filterwarnings("ignore")


def seed_everything(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# MODEL DEFINITIONS (PyTorch Tabular ResNet & Tree Parameter Factories)
# =============================================================================


class ResidualBlock(nn.Module):
    """
    Residual Block tailored for dense tabular features with LayerNorm
    and Mish non-linear activation to ensure smooth gradient flow.
    """

    def __init__(self, hidden_dim: int, dropout_rate: float = 0.2):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.act1 = nn.Mish()
        self.drop1 = nn.Dropout(dropout_rate)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.act2 = nn.Mish()
        self.drop2 = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.fc1(x)
        out = self.norm1(out)
        out = self.act1(out)
        out = self.drop1(out)

        out = self.fc2(out)
        out = self.norm2(out)
        out = self.act2(out)
        out = self.drop2(out)

        return residual + out


class TabularResNet(nn.Module):
    """
    Deep Tabular ResNet model incorporating projection, multi-stage residual
    blocks, and dropout regularization to model log-transformed transaction amounts.
    """

    def __init__(
        self,
        in_features: int,
        hidden_dim: int = 256,
        num_blocks: int = 3,
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Mish(),
            nn.Dropout(dropout_rate),
        )

        self.blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim, dropout_rate) for _ in range(num_blocks)]
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.Mish(),
            nn.Dropout(dropout_rate / 2.0),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(x)
        for block in self.blocks:
            x = block(x)
        out = self.head(x)
        return out.squeeze(-1)


class LogRMSELoss(nn.Module):
    """
    Differentiable Root Mean Squared Error loss on log-transformed target space
    with epsilon stabilizing numerical gradients.
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        mse = torch.mean((y_pred - y_true) ** 2)
        return torch.sqrt(mse + self.eps)


def get_torch_model_and_optimizer(
    in_features: int,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    hidden_dim: int = 256,
    num_blocks: int = 3,
    dropout_rate: float = 0.2,
) -> Tuple[nn.Module, nn.Module, torch.optim.Optimizer]:
    """
    Factory function for initializing PyTorch Tabular ResNet, loss criterion, and AdamW optimizer.
    """
    model = TabularResNet(
        in_features=in_features,
        hidden_dim=hidden_dim,
        num_blocks=num_blocks,
        dropout_rate=dropout_rate,
    )
    criterion = LogRMSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    return model, criterion, optimizer


def get_lgb_model_params() -> Dict[str, Any]:
    """
    Returns regularized hyperparameter configurations for LightGBM regressor.
    """
    return {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "n_estimators": 1500,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": 6,
        "min_child_samples": 20,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.7,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }


def get_xgb_model_params() -> Dict[str, Any]:
    """
    Returns regularized hyperparameter configurations for XGBoost regressor.
    """
    return {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "n_estimators": 1500,
        "learning_rate": 0.03,
        "max_depth": 6,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": -1,
    }


def get_catboost_model_params() -> Dict[str, Any]:
    """
    Returns regularized hyperparameter configurations for CatBoost regressor.
    """
    return {
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "iterations": 1500,
        "learning_rate": 0.03,
        "depth": 6,
        "l2_leaf_reg": 3.0,
        "random_seed": 42,
        "verbose": 0,
    }


def train_pytorch_model(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    X_te: np.ndarray,
    in_features: int,
    epochs: int = 100,
    patience: int = 15,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Train PyTorch Tabular ResNet model on log target with early stopping.
    """
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_va_s = scaler.transform(X_va)
    X_te_s = scaler.transform(X_te)

    tr_ds = TensorDataset(
        torch.tensor(X_tr_s, dtype=torch.float32),
        torch.tensor(y_tr, dtype=torch.float32),
    )
    va_ds = TensorDataset(
        torch.tensor(X_va_s, dtype=torch.float32),
        torch.tensor(y_va, dtype=torch.float32),
    )

    tr_loader = DataLoader(tr_ds, batch_size=64, shuffle=True)
    va_loader = DataLoader(va_ds, batch_size=128, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, criterion, optimizer = get_torch_model_and_optimizer(in_features=in_features)
    model.to(device)

    best_loss = float("inf")
    best_weights = None
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

        model.eval()
        val_loss = 0.0
        val_count = 0
        with torch.no_grad():
            for bx, by in va_loader:
                bx, by = bx.to(device), by.to(device)
                out = model(bx)
                val_loss += criterion(out, by).item() * len(by)
                val_count += len(by)
        val_loss /= val_count

        if val_loss < best_loss:
            best_loss = val_loss
            best_weights = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_weights is not None:
        model.load_state_dict(best_weights)

    model.eval()
    with torch.no_grad():
        va_tensor = torch.tensor(X_va_s, dtype=torch.float32).to(device)
        te_tensor = torch.tensor(X_te_s, dtype=torch.float32).to(device)
        oof_pred = model(va_tensor).cpu().numpy()
        test_pred = model(te_tensor).cpu().numpy()

    return oof_pred, test_pred


# =============================================================================
# MAIN PIPELINE EXECUTION
# =============================================================================


def main():
    seed_everything(42)

    # 1. Load Data
    train_path = "./input/train.parquet"
    test_path = "./input/test.parquet"

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    target_col = "Amount in USD"
    assert (
        target_col in train_df.columns
    ), f"Target column '{target_col}' missing from train data."

    # Extract target and log1p transform
    y_train_raw = np.maximum(0, train_df[target_col].values.astype(float))
    y_train_log = np.log1p(y_train_raw)

    id_col = (
        "id" if "id" in test_df.columns else ("ID" if "ID" in test_df.columns else None)
    )
    if id_col:
        test_ids = test_df[id_col].values
    else:
        test_ids = np.arange(len(test_df))

    train_features = train_df.drop(columns=[target_col], errors="ignore")
    test_features = test_df.copy()

    if id_col and id_col in train_features.columns:
        train_features = train_features.drop(columns=[id_col])
    if id_col and id_col in test_features.columns:
        test_features = test_features.drop(columns=[id_col])

    n_train = len(train_df)
    combined = pd.concat([train_features, test_features], axis=0, ignore_index=True)

    # 2. Data Preprocessing & Feature Engineering
    date_cols = []
    for col in combined.columns:
        if (
            pd.api.types.is_datetime64_any_dtype(combined[col])
            or "date" in col.lower()
            or "year" in col.lower()
        ):
            try:
                combined[col] = pd.to_datetime(combined[col], errors="coerce")
                date_cols.append(col)
            except Exception:
                pass

    for col in date_cols:
        combined[f"{col}_year"] = combined[col].dt.year.fillna(-1)
        combined[f"{col}_month"] = combined[col].dt.month.fillna(-1)
        combined[f"{col}_quarter"] = combined[col].dt.quarter.fillna(-1)
        combined[f"{col}_dayofweek"] = combined[col].dt.dayofweek.fillna(-1)
        combined[f"{col}_dayofyear"] = combined[col].dt.dayofyear.fillna(-1)
        combined[f"{col}_is_month_end"] = combined[col].dt.is_month_end.astype(int)
        ref_date = pd.Timestamp("2000-01-01")
        combined[f"{col}_days_since_2000"] = (combined[col] - ref_date).dt.days.fillna(
            -1
        )
        combined.drop(columns=[col], inplace=True)

    cat_cols = []
    num_cols = []
    for col in combined.columns:
        if combined[col].dtype == "object" or combined[col].dtype.name == "category":
            cat_cols.append(col)
        else:
            num_cols.append(col)

    for col in cat_cols:
        combined[col] = combined[col].astype(str).fillna("Missing")
        freq = combined.iloc[:n_train][col].value_counts().to_dict()
        combined[f"{col}_freq"] = combined[col].map(freq).fillna(0)

        le = LabelEncoder()
        combined[col] = le.fit_transform(combined[col])

    for col in num_cols:
        med = combined.iloc[:n_train][col].median()
        combined[col] = combined[col].fillna(med if not np.isnan(med) else 0)

    X_train_df = combined.iloc[:n_train].copy()
    X_test_df = combined.iloc[n_train:].copy()

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # Save processed features to ./working
    os.makedirs("./working", exist_ok=True)
    X_train_df["target_log"] = y_train_log
    X_train_df["target_raw"] = y_train_raw
    X_train_df.to_parquet("./working/train_processed.parquet", index=False)
    X_test_df.to_parquet("./working/test_processed.parquet", index=False)

    feature_cols = [
        c for c in X_train_df.columns if c not in ["target_log", "target_raw"]
    ]
    X_train = X_train_df[feature_cols].values.astype(np.float32)
    X_test = X_test_df[feature_cols].values.astype(np.float32)

    n_samples = len(X_train)
    n_test = len(X_test)
    n_features = len(feature_cols)

    # 3. Model Training & Cross-Validation
    oof_lgb = np.zeros(n_samples)
    oof_xgb = np.zeros(n_samples)
    oof_cb = np.zeros(n_samples)
    oof_nn = np.zeros(n_samples)

    test_lgb = np.zeros(n_test)
    test_xgb = np.zeros(n_test)
    test_cb = np.zeros(n_test)
    test_nn = np.zeros(n_test)

    n_splits = 5
    lgb_params = get_lgb_model_params()
    xgb_params = get_xgb_model_params()
    cb_params = get_catboost_model_params()

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_df)):
        X_tr_df, X_va_df = X_train_df.iloc[train_idx].copy(), X_train_df.iloc[val_idx].copy()
        X_te_df = X_test_df.copy()
        y_tr, y_va = y_train_log[train_idx], y_train_log[val_idx]

        global_mean = y_tr.mean()
        for col in cat_cols:
            means = pd.Series(y_tr, index=X_tr_df.index).groupby(X_tr_df[col]).mean()
            stds = pd.Series(y_tr, index=X_tr_df.index).groupby(X_tr_df[col]).std().fillna(0)

            X_tr_df[f"{col}_te_mean"] = X_tr_df[col].map(means).fillna(global_mean)
            X_tr_df[f"{col}_te_std"] = X_tr_df[col].map(stds).fillna(0)
            X_va_df[f"{col}_te_mean"] = X_va_df[col].map(means).fillna(global_mean)
            X_va_df[f"{col}_te_std"] = X_va_df[col].map(stds).fillna(0)
            X_te_df[f"{col}_te_mean"] = X_te_df[col].map(means).fillna(global_mean)
            X_te_df[f"{col}_te_std"] = X_te_df[col].map(stds).fillna(0)

        fold_features = [c for c in X_tr_df.columns if c not in ["target_log", "target_raw"]]
        X_tr = X_tr_df[fold_features].values.astype(np.float32)
        X_va = X_va_df[fold_features].values.astype(np.float32)
        X_test = X_te_df[fold_features].values.astype(np.float32)
        n_features = len(fold_features)

        # --- A. LightGBM ---
        model_lgb = lgb.LGBMRegressor(**lgb_params)
        model_lgb.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
        )
        oof_lgb[val_idx] = model_lgb.predict(X_va)
        test_lgb += model_lgb.predict(X_test) / n_splits

        # --- B. XGBoost ---
        xgb_fold_params = copy.deepcopy(xgb_params)
        xgb_fold_params["early_stopping_rounds"] = 50
        model_xgb = xgb.XGBRegressor(**xgb_fold_params)
        model_xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        oof_xgb[val_idx] = model_xgb.predict(X_va)
        test_xgb += model_xgb.predict(X_test) / n_splits

        # --- C. CatBoost ---
        cb_fold_params = copy.deepcopy(cb_params)
        cb_fold_params["early_stopping_rounds"] = 50
        model_cb = cb.CatBoostRegressor(**cb_fold_params)
        model_cb.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=False)
        oof_cb[val_idx] = model_cb.predict(X_va)
        test_cb += model_cb.predict(X_test) / n_splits

        # --- D. PyTorch Tabular ResNet ---
        nn_oof, nn_te = train_pytorch_model(
            X_tr,
            y_tr,
            X_va,
            y_va,
            X_test,
            in_features=n_features,
            epochs=100,
            patience=15,
        )
        oof_nn[val_idx] = nn_oof
        test_nn += nn_te / n_splits

    # 4. Ensembling & SLSQP Weight Optimization on Raw Scale RMSE
    def loss_func(weights):
        w = weights / np.sum(weights)
        pred_log = w[0] * oof_lgb + w[1] * oof_xgb + w[2] * oof_cb + w[3] * oof_nn
        pred_usd = np.expm1(np.maximum(0, pred_log))
        return np.sqrt(mean_squared_error(y_train_raw, pred_usd))

    init_weights = [0.30, 0.30, 0.25, 0.15]
    bounds = [(0, 1)] * 4
    res = minimize(loss_func, init_weights, method="SLSQP", bounds=bounds)
    opt_weights = res.x / np.sum(res.x)

    oof_final_log = (
        opt_weights[0] * oof_lgb
        + opt_weights[1] * oof_xgb
        + opt_weights[2] * oof_cb
        + opt_weights[3] * oof_nn
    )
    test_final_log = (
        opt_weights[0] * test_lgb
        + opt_weights[1] * test_xgb
        + opt_weights[2] * test_cb
        + opt_weights[3] * test_nn
    )

    oof_final_usd = np.expm1(np.maximum(0, oof_final_log))
    test_final_usd = np.expm1(np.maximum(0, test_final_log))

    test_final_usd = np.nan_to_num(test_final_usd, nan=0.0, posinf=0.0, neginf=0.0)
    test_final_usd = np.maximum(0.0, test_final_usd)

    cv_rmse = float(np.sqrt(mean_squared_error(y_train_raw, oof_final_usd)))

    # 5. Output Submission File
    os.makedirs("./submission", exist_ok=True)
    sub_df = pd.DataFrame({"id": test_ids, "Amount in USD": test_final_usd})
    sub_df.to_csv("./submission/submission.csv", index=False)

    print(f"Final Validation Score: {cv_rmse}")


if __name__ == "__main__":
    main()