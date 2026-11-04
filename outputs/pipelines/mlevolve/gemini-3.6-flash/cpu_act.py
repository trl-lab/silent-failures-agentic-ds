import os
import random
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler, StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import lightgbm as lgb
import xgboost as xgb
import catboost as cb


# -----------------------------------------------------------------------------
# 1. Reproducibility & Environment Setup
# -----------------------------------------------------------------------------
def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------------------------------------------------------
# 2. Data Processing & Feature Engineering
# -----------------------------------------------------------------------------
def load_raw_data():
    """Load train and test parquet files from input directory."""
    train_path = "./input/train.parquet"
    test_path = "./input/test.parquet"

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)
    return train_df, test_df


def engineer_telemetry_features(df):
    """Construct domain-specific CPU activity telemetry features."""
    data = df.copy()
    eps = 1e-5

    # 1. System Call Ratios & Granularity
    if "scall" in data.columns:
        if "sread" in data.columns:
            data["sread_ratio"] = data["sread"] / (data["scall"] + eps)
        if "swrite" in data.columns:
            data["swrite_ratio"] = data["swrite"] / (data["scall"] + eps)
        if "sread" in data.columns and "swrite" in data.columns:
            data["other_scall"] = data["scall"] - data["sread"] - data["swrite"]
            data["sread_swrite_ratio"] = data["sread"] / (data["swrite"] + eps)
            data["sread_swrite_sum"] = data["sread"] + data["swrite"]

    # 2. Character Data Transfer & Subsystem Ratios
    if "rchar" in data.columns and "sread" in data.columns:
        data["rchar_per_sread"] = data["rchar"] / (data["sread"] + eps)
    if "wchar" in data.columns and "swrite" in data.columns:
        data["wchar_per_swrite"] = data["wchar"] / (data["swrite"] + eps)
    if "rchar" in data.columns and "lread" in data.columns:
        data["rchar_per_lread"] = data["rchar"] / (data["lread"] + eps)
    if "wchar" in data.columns and "lwrite" in data.columns:
        data["wchar_per_lwrite"] = data["wchar"] / (data["lwrite"] + eps)
    if "rchar" in data.columns and "wchar" in data.columns:
        data["char_total"] = data["rchar"] + data["wchar"]
        data["char_ratio"] = data["rchar"] / (data["wchar"] + eps)
    if "lread" in data.columns and "lwrite" in data.columns:
        data["lread_lwrite_ratio"] = data["lread"] / (data["lwrite"] + eps)
        data["lread_lwrite_sum"] = data["lread"] + data["lwrite"]

    # 3. Process Execution & Scheduling Dynamics
    if "fork" in data.columns and "scall" in data.columns:
        data["fork_per_scall"] = data["fork"] / (data["scall"] + eps)
    if "exec" in data.columns and "scall" in data.columns:
        data["exec_per_scall"] = data["exec"] / (data["scall"] + eps)
    if "fork" in data.columns and "exec" in data.columns:
        data["fork_exec_ratio"] = data["fork"] / (data["exec"] + eps)
    if "runqsz" in data.columns and "freemem" in data.columns:
        data["runqsz_per_freemem"] = data["runqsz"] / (data["freemem"] + eps)

    # 4. Paging & Memory Pressure Dynamics
    if "ppgout" in data.columns and "pgout" in data.columns:
        data["ppgout_per_pgout"] = data["ppgout"] / (data["pgout"] + eps)
    if "ppgin" in data.columns and "pgin" in data.columns:
        data["ppgin_per_pgin"] = data["ppgin"] / (data["pgin"] + eps)
    if "pgin" in data.columns and "pgout" in data.columns:
        data["pgin_pgout_ratio"] = data["pgin"] / (data["pgout"] + eps)
        data["pgin_pgout_sum"] = data["pgin"] + data["pgout"]
    if "ppgin" in data.columns and "ppgout" in data.columns:
        data["ppgin_ppgout_ratio"] = data["ppgin"] / (data["ppgout"] + eps)
        data["ppgin_ppgout_sum"] = data["ppgin"] + data["ppgout"]
    if "pgfree" in data.columns and "pgscan" in data.columns:
        data["pgscan_efficiency"] = data["pgfree"] / (data["pgscan"] + eps)
    if "pflt" in data.columns and "vflt" in data.columns:
        data["fault_ratio"] = data["pflt"] / (data["vflt"] + eps)
        data["total_faults"] = data["pflt"] + data["vflt"]
        if "runqsz" in data.columns:
            data["faults_per_runqsz"] = data["total_faults"] / (data["runqsz"] + eps)

    paging_cols = [
        col for col in ["ppgout", "ppgin", "pgfree", "pgscan"] if col in data
    ]
    if len(paging_cols) > 0:
        data["total_paging_activity"] = data[paging_cols].sum(axis=1)

    if "freemem" in data.columns and "freeswap" in data.columns:
        data["freemem_freeswap_ratio"] = data["freemem"] / (data["freeswap"] + eps)
        data["total_free_mem_swap"] = data["freemem"] + data["freeswap"]

    # 4b. Subsystem Statistical Family Summaries
    io_family = [col for col in ["rchar", "wchar", "sread", "swrite", "lread", "lwrite"] if col in data.columns]
    if len(io_family) > 0:
        data["io_family_mean"] = data[io_family].mean(axis=1)
        data["io_family_std"] = data[io_family].std(axis=1)
        data["io_family_max"] = data[io_family].max(axis=1)
        data["io_family_min"] = data[io_family].min(axis=1)

    mem_family = [col for col in ["freemem", "freeswap", "pgin", "pgout", "ppgin", "ppgout", "pgfree", "pgscan", "vflt", "pflt"] if col in data.columns]
    if len(mem_family) > 0:
        data["mem_family_mean"] = data[mem_family].mean(axis=1)
        data["mem_family_std"] = data[mem_family].std(axis=1)
        data["mem_family_max"] = data[mem_family].max(axis=1)
        data["mem_family_min"] = data[mem_family].min(axis=1)

    # 5. Non-User CPU Time Composition & Advanced Interactions
    if "scall" in data.columns and "runqsz" in data.columns:
        data["scall_per_runqsz"] = data["scall"] / (data["runqsz"] + eps)

    pg_out_col = "pgout" if "pgout" in data.columns else ("ppgout" if "ppgout" in data.columns else None)
    pg_in_col = "pgin" if "pgin" in data.columns else ("ppgin" if "ppgin" in data.columns else None)
    if "freemem" in data.columns:
        pg_sum = 0
        if pg_out_col:
            pg_sum = pg_sum + data[pg_out_col]
        if pg_in_col:
            pg_sum = pg_sum + data[pg_in_col]
        data["paging_pressure"] = pg_sum / (data["freemem"] + eps)

    if (
        "rchar" in data.columns
        and "wchar" in data.columns
        and "lread" in data.columns
        and "lwrite" in data.columns
    ):
        data["io_throughput_rate"] = (data["rchar"] + data["wchar"]) / (
            data["lread"] + data["lwrite"] + eps
        )

    non_usr_components = [col for col in ["sys", "wio", "idle"] if col in data.columns]
    if len(non_usr_components) > 0:
        data["non_usr_time"] = data[non_usr_components].sum(axis=1)
        data["cpu_budget_remainder"] = 100.0 - data["non_usr_time"]
    if "sys" in data.columns and "idle" in data.columns:
        data["sys_to_idle_ratio"] = data["sys"] / (data["idle"] + eps)
    if "sys" in data.columns and "scall" in data.columns:
        data["sys_call_intensity"] = data["scall"] / (data["sys"] + eps)

    # 6. Log-Transform High-Skew Volumetric Features
    skewed_cols = [
        "scall",
        "rchar",
        "wchar",
        "vflt",
        "pflt",
        "pgscan",
        "freemem",
        "freeswap",
        "lread",
        "lwrite",
    ]
    for c in skewed_cols:
        if c in data.columns:
            data[f"log_{c}"] = np.log1p(np.maximum(0, data[c]))

    return data


def process_data_and_create_splits():
    """Execute complete data preparation and feature engineering pipeline."""
    os.makedirs("./working", exist_ok=True)

    train_df, test_df = load_raw_data()
    target_col = "usr"

    train_targets = train_df[target_col].copy()
    train_features_raw = train_df.drop(columns=[target_col])
    test_features_raw = test_df.copy()

    train_engineered = engineer_telemetry_features(train_features_raw)
    test_engineered = engineer_telemetry_features(test_features_raw)

    feature_cols = list(train_engineered.columns)

    # Missing / infinite handling using train statistics
    for col in feature_cols:
        train_engineered[col] = train_engineered[col].replace([np.inf, -np.inf], np.nan)
        test_engineered[col] = test_engineered[col].replace([np.inf, -np.inf], np.nan)

        col_median = train_engineered[col].median()
        train_engineered[col] = train_engineered[col].fillna(col_median)
        test_engineered[col] = test_engineered[col].fillna(col_median)

    train_engineered[target_col] = train_targets

    # 10-Fold Stratified K-Fold Split on Binned Target
    target_binned = pd.qcut(
        train_engineered[target_col], q=10, labels=False, duplicates="drop"
    )
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    train_engineered["fold"] = -1
    for fold, (train_idx, val_idx) in enumerate(
        skf.split(train_engineered, target_binned)
    ):
        train_engineered.loc[val_idx, "fold"] = fold

    return train_engineered, test_engineered


# -----------------------------------------------------------------------------
# 3. Neural Architecture Definition
# -----------------------------------------------------------------------------
class TelemetryBlock(nn.Module):
    """Residual block with LayerNorm, GELU activation, and Dropout."""

    def __init__(self, dim: int, dropout: float = 0.15):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.act = nn.GELU()
        self.norm2 = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.norm1(x)
        out = self.act(self.fc1(out))
        out = self.norm2(out)
        out = self.drop(out)
        out = self.fc2(out)
        return residual + out


class TelemetryResNet(nn.Module):
    """Deep Residual Neural Network designed for continuous OS telemetry regression."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_blocks: int = 3,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.in_proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [TelemetryBlock(hidden_dim, dropout) for _ in range(num_blocks)]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_proj(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x).squeeze(-1)


# -----------------------------------------------------------------------------
# 4. SLSQP Ensemble Weight Optimization
# -----------------------------------------------------------------------------
def find_optimal_blend_weights(oof_preds_list: list, y_true: np.ndarray) -> np.ndarray:
    """Find non-negative ensemble weights summing to 1 that minimize OOF RMSE."""
    matrix = np.column_stack(oof_preds_list)

    def loss_func(weights):
        w = weights / np.sum(weights)
        pred = matrix @ w
        return np.sqrt(mean_squared_error(y_true, pred))

    num_models = len(oof_preds_list)
    init_weights = np.ones(num_models) / num_models
    bounds = [(0, 1)] * num_models
    constraints = {"type": "eq", "fun": lambda w: 1.0 - np.sum(w)}

    res = minimize(
        loss_func,
        init_weights,
        bounds=bounds,
        constraints=constraints,
        method="SLSQP",
        options={"maxiter": 500},
    )
    best_weights = res.x / np.sum(res.x)
    return best_weights


# -----------------------------------------------------------------------------
# 5. Main Execution Flow
# -----------------------------------------------------------------------------
def main():
    os.makedirs("./working", exist_ok=True)
    os.makedirs("./submission", exist_ok=True)

    train_df, test_df = process_data_and_create_splits()

    target_col = "usr"
    fold_col = "fold"
    feature_cols = [c for c in train_df.columns if c not in [target_col, fold_col]]

    X_test = test_df[feature_cols].values.astype(np.float32)
    y_all = train_df[target_col].values.astype(np.float32)
    n_splits = train_df[fold_col].nunique()

    oof_lgb = np.zeros(len(train_df), dtype=np.float32)
    oof_xgb = np.zeros(len(train_df), dtype=np.float32)
    oof_cb = np.zeros(len(train_df), dtype=np.float32)
    oof_nn = np.zeros(len(train_df), dtype=np.float32)

    test_preds_lgb = np.zeros(len(test_df), dtype=np.float32)
    test_preds_xgb = np.zeros(len(test_df), dtype=np.float32)
    test_preds_cb = np.zeros(len(test_df), dtype=np.float32)
    test_preds_nn = np.zeros(len(test_df), dtype=np.float32)

    for fold in range(n_splits):
        train_mask = train_df[fold_col] != fold
        val_mask = train_df[fold_col] == fold

        X_tr = train_df.loc[train_mask, feature_cols].values.astype(np.float32)
        y_tr = train_df.loc[train_mask, target_col].values.astype(np.float32)
        X_va = train_df.loc[val_mask, feature_cols].values.astype(np.float32)
        y_va = train_df.loc[val_mask, target_col].values.astype(np.float32)
        X_te = test_df[feature_cols].values.astype(np.float32)

        scaler_fold = RobustScaler()
        X_tr_sc = scaler_fold.fit_transform(X_tr)
        X_va_sc = scaler_fold.transform(X_va)
        X_te_sc = scaler_fold.transform(X_te)

        n_pca = min(5, X_tr_sc.shape[1])
        pca_fold = PCA(n_components=n_pca, random_state=42)
        X_tr_pca = pca_fold.fit_transform(X_tr_sc)
        X_va_pca = pca_fold.transform(X_va_sc)
        X_te_pca = pca_fold.transform(X_te_sc)

        X_tr = np.hstack([X_tr, X_tr_pca])
        X_va = np.hstack([X_va, X_va_pca])
        X_test_fold = np.hstack([X_te, X_te_pca])

        val_idx = train_df.index[val_mask].values

        # --- Model 1: LightGBM Regressor ---
        lgb_model = lgb.LGBMRegressor(
            n_estimators=1200,
            learning_rate=0.025,
            num_leaves=31,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42 + fold,
            n_jobs=-1,
            verbosity=-1,
        )
        lgb_model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )

        val_pred_lgb = lgb_model.predict(X_va)
        oof_lgb[val_idx] = val_pred_lgb
        test_preds_lgb += lgb_model.predict(X_test_fold) / n_splits

        # --- Model 2: XGBoost Regressor ---
        xgb_model = xgb.XGBRegressor(
            n_estimators=1200,
            learning_rate=0.025,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            early_stopping_rounds=50,
            random_state=42 + fold,
            n_jobs=-1,
        )
        xgb_model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

        val_pred_xgb = xgb_model.predict(X_va)
        oof_xgb[val_idx] = val_pred_xgb
        test_preds_xgb += xgb_model.predict(X_test_fold) / n_splits

        # --- Model 3: CatBoost Regressor ---
        cb_model = cb.CatBoostRegressor(
            iterations=1200,
            learning_rate=0.025,
            depth=6,
            l2_leaf_reg=3.0,
            subsample=0.8,
            random_seed=42 + fold,
            verbose=0,
            early_stopping_rounds=50,
        )
        cb_model.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=False)

        val_pred_cb = cb_model.predict(X_va)
        oof_cb[val_idx] = val_pred_cb
        test_preds_cb += cb_model.predict(X_test_fold) / n_splits

        # --- Model 4: TelemetryResNet PyTorch Neural Network ---
        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr)
        X_va_sc = scaler.transform(X_va)
        X_te_sc = scaler.transform(X_test_fold)

        train_ds = TensorDataset(
            torch.tensor(X_tr_sc, dtype=torch.float32),
            torch.tensor(y_tr, dtype=torch.float32),
        )
        val_ds = TensorDataset(
            torch.tensor(X_va_sc, dtype=torch.float32),
            torch.tensor(y_va, dtype=torch.float32),
        )

        train_loader = DataLoader(
            train_ds, batch_size=128, shuffle=True, num_workers=0, pin_memory=True
        )
        val_loader = DataLoader(
            val_ds, batch_size=256, shuffle=False, num_workers=0, pin_memory=True
        )

        nn_model = TelemetryResNet(
            input_dim=X_tr.shape[1], hidden_dim=128, num_blocks=3, dropout=0.15
        ).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(nn_model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

        best_val_loss = float("inf")
        best_state = None
        patience = 20
        patience_ctr = 0

        for epoch in range(100):
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
            val_loss = 0.0
            val_count = 0
            with torch.no_grad():
                for bx, by in val_loader:
                    bx, by = bx.to(device), by.to(device)
                    out = nn_model(bx)
                    val_loss += criterion(out, by).item() * len(by)
                    val_count += len(by)

            val_rmse = np.sqrt(val_loss / val_count)
            if val_rmse < best_val_loss:
                best_val_loss = val_rmse
                best_state = {
                    k: v.cpu().clone() for k, v in nn_model.state_dict().items()
                }
                patience_ctr = 0
            else:
                patience_ctr += 1
                if patience_ctr >= patience:
                    break

        nn_model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        nn_model.eval()
        with torch.no_grad():
            val_preds_nn = (
                nn_model(torch.tensor(X_va_sc, dtype=torch.float32).to(device))
                .cpu()
                .numpy()
            )
            te_preds_nn = (
                nn_model(torch.tensor(X_te_sc, dtype=torch.float32).to(device))
                .cpu()
                .numpy()
            )

        oof_nn[val_idx] = val_preds_nn
        test_preds_nn += te_preds_nn / n_splits

    # --- Ensemble Blending & Clipping ---
    blend_weights = find_optimal_blend_weights(
        [oof_lgb, oof_xgb, oof_cb, oof_nn], y_all
    )
    oof_blend = (
        blend_weights[0] * oof_lgb
        + blend_weights[1] * oof_xgb
        + blend_weights[2] * oof_cb
        + blend_weights[3] * oof_nn
    )
    oof_blend = np.clip(oof_blend, 0.0, 100.0)

    test_blend = (
        blend_weights[0] * test_preds_lgb
        + blend_weights[1] * test_preds_xgb
        + blend_weights[2] * test_preds_cb
        + blend_weights[3] * test_preds_nn
    )
    test_blend = np.clip(test_blend, 0.0, 100.0)

    final_score = np.sqrt(mean_squared_error(y_all, oof_blend))

    # --- Save Output Predictions ---
    submission_path = "./submission/submission.csv"
    submission_df = pd.DataFrame({"id": np.arange(len(test_blend)), "usr": test_blend})
    submission_df.to_csv(submission_path, index=False)

    print(f"Final Validation Score: {final_score:.5f}")


if __name__ == "__main__":
    main()