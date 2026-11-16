import os
import random
from typing import Any, Dict, Tuple
import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb

# =========================================================================
# 0. Global Setup and Seed Initialization
# =========================================================================


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)

INPUT_DIR = "./input"
WORKING_DIR = "./working"
SUBMISSION_DIR = "./submission"
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================================================================
# 1. Data Loading and Feature Engineering
# =========================================================================

train_df = pd.read_parquet(os.path.join(INPUT_DIR, "train.parquet"))
test_df = pd.read_parquet(os.path.join(INPUT_DIR, "test.parquet"))

target_col = "usr"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes domain-specific system performance metrics, ratios,
    non-linear transformations, and metric group summaries.
    """
    df_feat = df.copy()
    eps = 1e-6

    # --- Domain Ratios: System Calls & Process Management ---
    if "scall" in df_feat.columns:
        if "sread" in df_feat.columns:
            df_feat["sread_scall_ratio"] = df_feat["sread"] / (df_feat["scall"] + eps)
        if "swrite" in df_feat.columns:
            df_feat["swrite_scall_ratio"] = df_feat["swrite"] / (df_feat["scall"] + eps)
        if "sread" in df_feat.columns and "swrite" in df_feat.columns:
            df_feat["rw_scall_ratio"] = (df_feat["sread"] + df_feat["swrite"]) / (
                df_feat["scall"] + eps
            )
            df_feat["other_syscalls"] = np.maximum(
                0, df_feat["scall"] - (df_feat["sread"] + df_feat["swrite"])
            )
        if "fork" in df_feat.columns:
            df_feat["fork_scall_ratio"] = df_feat["fork"] / (df_feat["scall"] + eps)
        if "exec" in df_feat.columns:
            df_feat["exec_scall_ratio"] = df_feat["exec"] / (df_feat["scall"] + eps)

    if "fork" in df_feat.columns and "exec" in df_feat.columns:
        df_feat["fork_exec_ratio"] = (df_feat["fork"] + eps) / (df_feat["exec"] + eps)
        df_feat["total_proc_ops"] = df_feat["fork"] + df_feat["exec"]

    # --- Domain Ratios: I/O Throughput & Transfer Densities ---
    if "rchar" in df_feat.columns and "sread" in df_feat.columns:
        df_feat["bytes_per_read"] = df_feat["rchar"] / (df_feat["sread"] + eps)
    if "wchar" in df_feat.columns and "swrite" in df_feat.columns:
        df_feat["bytes_per_write"] = df_feat["wchar"] / (df_feat["swrite"] + eps)
    if "rchar" in df_feat.columns and "wchar" in df_feat.columns:
        df_feat["total_chars"] = df_feat["rchar"] + df_feat["wchar"]
        df_feat["rw_char_ratio"] = (df_feat["rchar"] + eps) / (df_feat["wchar"] + eps)
    if "lread" in df_feat.columns and "lwrite" in df_feat.columns:
        df_feat["total_ltransfers"] = df_feat["lread"] + df_feat["lwrite"]
        df_feat["lread_lwrite_ratio"] = (df_feat["lread"] + eps) / (
            df_feat["lwrite"] + eps
        )

    # --- Domain Ratios: Paging Activity & Virtual Memory ---
    if "ppgin" in df_feat.columns and "pgin" in df_feat.columns:
        df_feat["ppgin_per_pgin"] = df_feat["ppgin"] / (df_feat["pgin"] + eps)
    if "ppgout" in df_feat.columns and "pgout" in df_feat.columns:
        df_feat["ppgout_per_pgout"] = df_feat["ppgout"] / (df_feat["pgout"] + eps)
    if "pgin" in df_feat.columns and "pgout" in df_feat.columns:
        df_feat["total_pg"] = df_feat["pgin"] + df_feat["pgout"]
        df_feat["pgin_pgout_ratio"] = (df_feat["pgin"] + eps) / (df_feat["pgout"] + eps)
    if "ppgin" in df_feat.columns and "ppgout" in df_feat.columns:
        df_feat["total_ppg"] = df_feat["ppgin"] + df_feat["ppgout"]
    if "pgfree" in df_feat.columns and "pgscan" in df_feat.columns:
        df_feat["scan_efficiency"] = df_feat["pgfree"] / (df_feat["pgscan"] + eps)

    # --- Domain Ratios: Faults & Address Translation ---
    if "pflt" in df_feat.columns and "vflt" in df_feat.columns:
        df_feat["total_faults"] = df_feat["pflt"] + df_feat["vflt"]
        df_feat["pflt_vflt_ratio"] = (df_feat["pflt"] + eps) / (df_feat["vflt"] + eps)
    if "atch" in df_feat.columns and "total_faults" in df_feat.columns:
        df_feat["atch_fault_ratio"] = df_feat["atch"] / (df_feat["total_faults"] + eps)

    # --- Domain Ratios: Queue Saturation & Memory Pressure ---
    if "runqsz" in df_feat.columns:
        if "freemem" in df_feat.columns:
            df_feat["runq_freemem_ratio"] = df_feat["runqsz"] / (
                df_feat["freemem"] + eps
            )
        if "scall" in df_feat.columns:
            df_feat["scall_per_runq"] = df_feat["scall"] / (df_feat["runqsz"] + eps)
        if "total_faults" in df_feat.columns:
            df_feat["faults_per_runq"] = df_feat["total_faults"] / (
                df_feat["runqsz"] + eps
            )
        if "total_proc_ops" in df_feat.columns:
            df_feat["proc_ops_per_runq"] = df_feat["total_proc_ops"] / (
                df_feat["runqsz"] + eps
            )

    if "freemem" in df_feat.columns and "freeswap" in df_feat.columns:
        df_feat["total_free_storage"] = df_feat["freemem"] + df_feat["freeswap"]
        df_feat["freemem_ratio"] = df_feat["freemem"] / (
            df_feat["total_free_storage"] + eps
        )
        df_feat["mem_swap_ratio"] = df_feat["freemem"] / (df_feat["freeswap"] + eps)

    # --- Interaction Features for Resource Contention ---
    if "runqsz" in df_feat.columns:
        if "scall" in df_feat.columns:
            df_feat["scall_x_runqsz"] = df_feat["scall"] * df_feat["runqsz"]
        if "rchar" in df_feat.columns:
            df_feat["rchar_x_runqsz"] = df_feat["rchar"] * df_feat["runqsz"]
        if "freemem" in df_feat.columns:
            df_feat["freemem_x_runqsz"] = df_feat["freemem"] * df_feat["runqsz"]

    # --- Log1p Non-linear Transforms on Heavy-Tailed Counts ---
    count_cols = [
        "lread",
        "lwrite",
        "scall",
        "sread",
        "swrite",
        "fork",
        "exec",
        "rchar",
        "wchar",
        "pgout",
        "ppgout",
        "pgfree",
        "pgscan",
        "atch",
        "pgin",
        "ppgin",
        "pflt",
        "vflt",
        "runqsz",
        "freemem",
        "freeswap",
    ]
    for col in count_cols:
        if col in df.columns:
            df_feat[f"log1p_{col}"] = np.log1p(np.maximum(0, df[col].values))

    # --- Group Summary Features ---
    syscall_group = [
        c for c in ["scall", "sread", "swrite", "fork", "exec"] if c in df_feat.columns
    ]
    if syscall_group:
        df_feat["syscall_sum"] = df_feat[syscall_group].sum(axis=1)
        df_feat["syscall_mean"] = df_feat[syscall_group].mean(axis=1)
        df_feat["syscall_std"] = df_feat[syscall_group].std(axis=1)

    paging_group = [
        c
        for c in ["pgout", "ppgout", "pgfree", "pgscan", "atch", "pgin", "ppgin"]
        if c in df_feat.columns
    ]
    if paging_group:
        df_feat["paging_sum"] = df_feat[paging_group].sum(axis=1)
        df_feat["paging_mean"] = df_feat[paging_group].mean(axis=1)
        df_feat["paging_std"] = df_feat[paging_group].std(axis=1)

    return df_feat


X_train_raw = train_df.drop(columns=[target_col])
y_train = train_df[target_col].values

train_features = engineer_features(X_train_raw)
train_features[target_col] = y_train
test_features = engineer_features(test_df)

# Create 5-Fold Cross-Validation Splits
kf = KFold(n_splits=5, shuffle=True, random_state=42)
train_features["fold"] = -1
for fold, (train_idx, val_idx) in enumerate(kf.split(train_features)):
    train_features.loc[val_idx, "fold"] = fold

# Clean Inf / NaN Values Robustly
train_features = train_features.replace([np.inf, -np.inf], np.nan).fillna(0.0)
test_features = test_features.replace([np.inf, -np.inf], np.nan).fillna(0.0)

feature_list = [c for c in train_features.columns if c not in [target_col, "fold"]]

# =========================================================================
# 2. Model Architectures & Configurations
# =========================================================================


class ResidualBlock(nn.Module):
    """
    Residual dense block with LayerNorm, GELU, and Dropout regularization.
    """

    def __init__(self, hidden_dim: int, dropout_rate: float = 0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout_rate),
        )
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.block(x))


class TabularResNet(nn.Module):
    """
    Deep Tabular Residual Network for continuous regression.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_blocks: int = 2,
        dropout_rate: float = 0.15,
        out_dim: int = 1,
    ):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )
        self.res_blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim, dropout_rate) for _ in range(num_blocks)]
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_layer(x)
        for block in self.res_blocks:
            h = block(h)
        out = self.head(h)
        return out.squeeze(-1)


def build_neural_model(
    input_dim: int,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    epochs: int = 80,
) -> Tuple[nn.Module, nn.Module, optim.Optimizer, Any]:
    model = TabularResNet(
        input_dim=input_dim, hidden_dim=128, num_blocks=2, dropout_rate=0.1
    )
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay, eps=1e-8
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    return model, criterion, optimizer, scheduler


def get_lgbm_params(seed: int = 42) -> Dict[str, Any]:
    return {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "n_estimators": 2000,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 20,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": seed,
        "n_jobs": -1,
        "verbose": -1,
    }


def get_xgb_params(seed: int = 42) -> Dict[str, Any]:
    return {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "n_estimators": 2000,
        "learning_rate": 0.03,
        "max_depth": 6,
        "min_child_weight": 3,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": seed,
        "n_jobs": -1,
    }


def get_catboost_params(seed: int = 42) -> Dict[str, Any]:
    return {
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "iterations": 2000,
        "learning_rate": 0.03,
        "depth": 6,
        "l2_leaf_reg": 3.0,
        "subsample": 0.8,
        "random_seed": seed,
        "verbose": False,
        "thread_count": -1,
    }


# =========================================================================
# 3. Model Training & Cross-Validation Loop
# =========================================================================

X = train_features[feature_list].values
y = train_features[target_col].values
folds = train_features["fold"].values
X_test = test_features[feature_list].values

n_train = len(train_features)
n_test = len(test_features)
n_folds = 5

oof_lgb = np.zeros(n_train)
oof_xgb = np.zeros(n_train)
oof_cat = np.zeros(n_train)
oof_nn = np.zeros(n_train)

test_preds_lgb = np.zeros(n_test)
test_preds_xgb = np.zeros(n_test)
test_preds_cat = np.zeros(n_test)
test_preds_nn = np.zeros(n_test)

for fold in range(n_folds):
    train_mask = folds != fold
    val_mask = folds == fold

    X_train_f, y_train_f = X[train_mask], y[train_mask]
    X_val_f, y_val_f = X[val_mask], y[val_mask]

    # --- 1. LightGBM ---
    lgb_params = get_lgbm_params(seed=42 + fold)
    model_lgb = lgb.LGBMRegressor(**lgb_params)
    model_lgb.fit(
        X_train_f,
        y_train_f,
        eval_set=[(X_val_f, y_val_f)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )
    oof_lgb[val_mask] = model_lgb.predict(X_val_f)
    test_preds_lgb += model_lgb.predict(X_test) / n_folds

    # --- 2. XGBoost ---
    xgb_params = get_xgb_params(seed=42 + fold)
    model_xgb = xgb.XGBRegressor(**xgb_params, early_stopping_rounds=50)
    model_xgb.fit(X_train_f, y_train_f, eval_set=[(X_val_f, y_val_f)], verbose=False)
    oof_xgb[val_mask] = model_xgb.predict(X_val_f)
    test_preds_xgb += model_xgb.predict(X_test) / n_folds

    # --- 3. CatBoost ---
    cat_params = get_catboost_params(seed=42 + fold)
    model_cat = cb.CatBoostRegressor(**cat_params, early_stopping_rounds=50)
    model_cat.fit(X_train_f, y_train_f, eval_set=(X_val_f, y_val_f), verbose=False)
    oof_cat[val_mask] = model_cat.predict(X_val_f)
    test_preds_cat += model_cat.predict(X_test) / n_folds

    # --- 4. PyTorch Tabular ResNet ---
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_train_f)
    X_va_scaled = scaler.transform(X_val_f)
    X_te_scaled = scaler.transform(X_test)

    train_tensor_x = torch.tensor(X_tr_scaled, dtype=torch.float32)
    train_tensor_y = torch.tensor(y_train_f, dtype=torch.float32)
    val_tensor_x = torch.tensor(X_va_scaled, dtype=torch.float32).to(device)

    train_dataset = TensorDataset(train_tensor_x, train_tensor_y)
    train_loader = DataLoader(
        train_dataset, batch_size=64, shuffle=True, drop_last=False
    )

    nn_epochs = 80
    nn_model, criterion, optimizer, scheduler = build_neural_model(
        input_dim=X.shape[1], lr=1e-3, weight_decay=1e-4, epochs=nn_epochs
    )
    nn_model.to(device)

    best_val_rmse = float("inf")
    best_nn_val_preds = None
    patience = 15
    patience_counter = 0

    for epoch in range(nn_epochs):
        nn_model.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = nn_model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
        scheduler.step()

        nn_model.eval()
        with torch.no_grad():
            val_preds_tensor = nn_model(val_tensor_x)
            val_loss = criterion(
                val_preds_tensor, torch.tensor(y_val_f, dtype=torch.float32).to(device)
            ).item()
            val_rmse = np.sqrt(val_loss)

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_nn_val_preds = val_preds_tensor.cpu().numpy()
            torch.save(
                nn_model.state_dict(),
                os.path.join(WORKING_DIR, f"nn_model_fold_{fold}.pt"),
            )
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    oof_nn[val_mask] = best_nn_val_preds

    # Load best NN model for test inference
    best_state = torch.load(
        os.path.join(WORKING_DIR, f"nn_model_fold_{fold}.pt"), map_location=device
    )
    nn_model.load_state_dict(best_state)
    nn_model.eval()
    with torch.no_grad():
        test_tensor_x = torch.tensor(X_te_scaled, dtype=torch.float32).to(device)
        nn_test_pred = nn_model(test_tensor_x).cpu().numpy()
        test_preds_nn += nn_test_pred / n_folds

    fold_rmse_lgb = np.sqrt(mean_squared_error(y_val_f, oof_lgb[val_mask]))
    fold_rmse_xgb = np.sqrt(mean_squared_error(y_val_f, oof_xgb[val_mask]))
    fold_rmse_cat = np.sqrt(mean_squared_error(y_val_f, oof_cat[val_mask]))
    fold_rmse_nn = np.sqrt(mean_squared_error(y_val_f, oof_nn[val_mask]))
    print(
        f"Fold {fold+1} | LGB: {fold_rmse_lgb:.4f} | XGB: {fold_rmse_xgb:.4f} | CAT: {fold_rmse_cat:.4f} | NN: {fold_rmse_nn:.4f}"
    )

# =========================================================================
# 4. Out-of-Fold Optimization & Submission Generation
# =========================================================================

OOF_matrix = np.column_stack([oof_lgb, oof_xgb, oof_cat, oof_nn])
TEST_matrix = np.column_stack(
    [test_preds_lgb, test_preds_xgb, test_preds_cat, test_preds_nn]
)


def loss_func(weights: np.ndarray) -> float:
    weights = weights / np.sum(weights)
    pred = np.dot(OOF_matrix, weights)
    return np.sqrt(mean_squared_error(y, pred))


init_weights = np.ones(4) / 4.0
bounds = [(0.0, 1.0) for _ in range(4)]
constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
opt_res = minimize(
    loss_func,
    init_weights,
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
    options={"maxiter": 500},
)

optimal_weights = opt_res.x if opt_res.success else np.array([0.35, 0.35, 0.20, 0.10])
optimal_weights = optimal_weights / np.sum(optimal_weights)

oof_ensemble = np.dot(OOF_matrix, optimal_weights)
oof_ensemble = np.clip(oof_ensemble, 0.0, 100.0)
final_val_rmse = float(np.sqrt(mean_squared_error(y, oof_ensemble)))

final_test_preds = np.dot(TEST_matrix, optimal_weights)
final_test_preds = np.clip(final_test_preds, 0.0, 100.0)

submission_df = pd.DataFrame(
    {"id": np.arange(len(final_test_preds)), "usr": final_test_preds}
)
submission_path = os.path.join(SUBMISSION_DIR, "submission.csv")
submission_df.to_csv(submission_path, index=False)

print(f"Final Validation Score: {final_val_rmse}")
