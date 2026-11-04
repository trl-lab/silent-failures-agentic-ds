import copy
import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import rankdata
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

import lightgbm as lgb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb

try:
    from catboost import CatBoostClassifier

    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False


# ==========================================
# 1. Feature Engineering
# ==========================================
def build_rfm_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineers concise domain-specific behavioral features from raw RFM blood donor data."""
    data = df.copy()

    v1 = data["V1"].astype(float)
    v2 = data["V2"].astype(float)
    v4 = data["V4"].astype(float)

    # High-signal domain RFM features
    data["recency_ratio"] = v1 / (v4 + 1.0)
    data["frequency_rate"] = v2 / (v4 + 1.0)

    active_interval = (v4 - v1) / np.maximum(1.0, v2 - 1.0)
    data["active_interval"] = active_interval
    data["overdue_score"] = v1 - active_interval

    data["log_v1"] = np.log1p(v1)
    data["log_v2"] = np.log1p(v2)
    data["log_v4"] = np.log1p(v4)

    return data


# ==========================================
# 2. PyTorch Tabular Deep MLP Architecture
# ==========================================
class TabularMLP(nn.Module):
    """Deep Neural Network with LayerNorm, Mish activations, and Dropout for tabular data."""

    def __init__(
        self, input_dim: int, hidden_dims=[64, 32], dropout_rate: float = 0.25
    ):
        super().__init__()
        layers = []
        curr_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(curr_dim, h_dim))
            layers.append(nn.LayerNorm(h_dim))
            layers.append(nn.Mish())
            layers.append(nn.Dropout(dropout_rate))
            curr_dim = h_dim
        layers.append(nn.Linear(curr_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


# ==========================================
# 3. Probability Rank Normalization
# ==========================================
def to_rank_percentile(arr: np.ndarray) -> np.ndarray:
    """Converts continuous predictions into uniform rank percentiles strictly in [0, 1]."""
    if len(arr) == 0:
        return arr
    return (rankdata(arr) - 0.5) / len(arr)


# ==========================================
# 4. Main Training and Evaluation Pipeline
# ==========================================
def main():
    # Set global random seeds for exact reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    # Load Data
    train_raw = pd.read_parquet("./input/train.parquet")
    test_raw = pd.read_parquet("./input/test.parquet")

    target_col = "Class"
    y_raw = train_raw[target_col]

    # Handle standard binary target formats cleanly
    unique_vals = y_raw.unique()
    if set(unique_vals).issubset({1, 2}) or set(unique_vals).issubset({"1", "2"}):
        y = (y_raw.astype(str) == "2").astype(int).values
    else:
        y = y_raw.astype(int).values

    if "id" in test_raw.columns:
        test_ids = test_raw["id"].values
        test_features_raw = test_raw.drop(columns=["id"])
    else:
        test_ids = np.arange(len(test_raw))
        test_features_raw = test_raw.copy()

    if "id" in train_raw.columns:
        train_features_raw = train_raw.drop(columns=["id", target_col])
    else:
        train_features_raw = train_raw.drop(columns=[target_col])

    # Build Engineered Features
    X_train_df = build_rfm_features(train_features_raw)
    X_test_df = build_rfm_features(test_features_raw)

    X = X_train_df.values
    X_test = X_test_df.values

    # Stratified 5-Fold Cross Validation Setup
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    num_models = 5
    oof_ranks = np.zeros((len(X), num_models))
    test_ranks = np.zeros((len(X_test), num_models))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_va, y_va = X[val_idx], y[val_idx]

        # Strict Isolation: Fit scaler strictly on train split
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)
        X_va_scaled = scaler.transform(X_va)
        X_te_scaled = scaler.transform(X_test)

        # --- Model 1: LightGBM Classifier ---
        lgb_model = lgb.LGBMClassifier(
            n_estimators=400,
            learning_rate=0.025,
            num_leaves=7,
            max_depth=3,
            min_child_samples=12,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42 + fold,
            verbosity=-1,
        )
        lgb_model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(-1),
            ],
        )
        pred_va_lgb = lgb_model.predict_proba(X_va)[:, 1]
        pred_te_lgb = lgb_model.predict_proba(X_test)[:, 1]

        # --- Model 2: XGBoost Classifier ---
        xgb_model = xgb.XGBClassifier(
            n_estimators=400,
            learning_rate=0.025,
            max_depth=3,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42 + fold,
            eval_metric="auc",
            early_stopping_rounds=50,
        )
        xgb_model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            verbose=False,
        )
        pred_va_xgb = xgb_model.predict_proba(X_va)[:, 1]
        pred_te_xgb = xgb_model.predict_proba(X_test)[:, 1]

        # --- Model 3: CatBoost Classifier ---
        if HAS_CATBOOST:
            cb_model = CatBoostClassifier(
                iterations=400,
                learning_rate=0.025,
                depth=4,
                l2_leaf_reg=3.0,
                random_seed=42 + fold,
                verbose=False,
                early_stopping_rounds=50,
            )
            cb_model.fit(X_tr, y_tr, eval_set=(X_va, y_va))
            pred_va_cb = cb_model.predict_proba(X_va)[:, 1]
            pred_te_cb = cb_model.predict_proba(X_test)[:, 1]
        else:
            pred_va_cb = pred_va_xgb
            pred_te_cb = pred_te_xgb

        # --- Model 4: ExtraTrees Classifier ---
        et_model = ExtraTreesClassifier(
            n_estimators=200,
            max_depth=5,
            min_samples_leaf=4,
            random_state=42 + fold,
        )
        et_model.fit(X_tr, y_tr)
        pred_va_et = et_model.predict_proba(X_va)[:, 1]
        pred_te_et = et_model.predict_proba(X_test)[:, 1]

        # --- Model 5: PyTorch Deep Tabular MLP ---
        torch_dataset = TensorDataset(
            torch.tensor(X_tr_scaled, dtype=torch.float32),
            torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1),
        )
        loader = DataLoader(torch_dataset, batch_size=32, shuffle=True)

        va_tensor = torch.tensor(X_va_scaled, dtype=torch.float32)
        va_target_tensor = torch.tensor(y_va, dtype=torch.float32).unsqueeze(1)
        te_tensor = torch.tensor(X_te_scaled, dtype=torch.float32)

        mlp_model = TabularMLP(
            input_dim=X.shape[1], hidden_dims=[64, 32], dropout_rate=0.25
        )
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(mlp_model.parameters(), lr=1e-3, weight_decay=1e-2)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150)

        best_val_loss = float("inf")
        best_state = None
        patience = 20
        patience_counter = 0

        for epoch in range(150):
            mlp_model.train()
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                out = mlp_model(batch_x)
                loss = criterion(out, batch_y)
                loss.backward()
                optimizer.step()
            scheduler.step()

            # Early stopping check on validation set
            mlp_model.eval()
            with torch.no_grad():
                val_out = mlp_model(va_tensor)
                val_loss = criterion(val_out, va_target_tensor).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = copy.deepcopy(mlp_model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        if best_state is not None:
            mlp_model.load_state_dict(best_state)

        mlp_model.eval()
        with torch.no_grad():
            pred_va_mlp = torch.sigmoid(mlp_model(va_tensor)).numpy().squeeze()
            pred_te_mlp = torch.sigmoid(mlp_model(te_tensor)).numpy().squeeze()

        # Calibration: Convert raw probabilities into uniform rank percentiles
        rank_va_lgb = to_rank_percentile(pred_va_lgb)
        rank_va_xgb = to_rank_percentile(pred_va_xgb)
        rank_va_cb = to_rank_percentile(pred_va_cb)
        rank_va_et = to_rank_percentile(pred_va_et)
        rank_va_mlp = to_rank_percentile(pred_va_mlp)

        rank_te_lgb = to_rank_percentile(pred_te_lgb)
        rank_te_xgb = to_rank_percentile(pred_te_xgb)
        rank_te_cb = to_rank_percentile(pred_te_cb)
        rank_te_et = to_rank_percentile(pred_te_et)
        rank_te_mlp = to_rank_percentile(pred_te_mlp)

        # Store OOF rank predictions
        oof_ranks[val_idx, 0] = rank_va_lgb
        oof_ranks[val_idx, 1] = rank_va_xgb
        oof_ranks[val_idx, 2] = rank_va_cb
        oof_ranks[val_idx, 3] = rank_va_et
        oof_ranks[val_idx, 4] = rank_va_mlp

        # Accumulate test rank predictions
        test_ranks[:, 0] += rank_te_lgb / n_splits
        test_ranks[:, 1] += rank_te_xgb / n_splits
        test_ranks[:, 2] += rank_te_cb / n_splits
        test_ranks[:, 3] += rank_te_et / n_splits
        test_ranks[:, 4] += rank_te_mlp / n_splits

    # SLSQP Optimization to directly maximize ROC AUC score
    def loss_func(weights):
        w = weights / np.sum(weights)
        blend = np.dot(oof_ranks, w)
        return -roc_auc_score(y, blend)

    init_weights = np.ones(num_models) / num_models
    bounds = [(0, 1)] * num_models
    constraints = ({'type': 'eq', 'fun': lambda w: 1.0 - np.sum(w)})

    res = minimize(loss_func, init_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    best_weights = res.x / np.sum(res.x)

    oof_preds = np.dot(oof_ranks, best_weights)
    test_preds = np.dot(test_ranks, best_weights)

    # Compute overall Out-of-Fold Score
    oof_auc = roc_auc_score(y, oof_preds)

    # Generate Submission File
    os.makedirs("./submission", exist_ok=True)
    submission = pd.DataFrame({"id": test_ids, "Class": test_preds})
    submission.to_csv("./submission/submission.csv", index=False)

    print(f"Final Validation Score: {oof_auc}")


if __name__ == "__main__":
    main()
