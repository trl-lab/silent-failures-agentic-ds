import json
import os
from typing import Any, Dict, Optional, Tuple

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)


# =========================================================================
# 1. Model Definitions
# =========================================================================
def get_lgbm_model(
    custom_params: Optional[Dict[str, Any]] = None, random_state: int = 42
) -> lgb.LGBMClassifier:
    """Returns a configured LightGBM Classifier for binary price direction prediction."""
    default_params = {
        "n_estimators": 1500,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 20,
        "subsample": 0.85,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "objective": "binary",
        "random_state": random_state,
        "n_jobs": -1,
        "importance_type": "gain",
        "verbose": -1,
    }
    if custom_params:
        default_params.update(custom_params)
    return lgb.LGBMClassifier(**default_params)


def get_xgb_model(
    custom_params: Optional[Dict[str, Any]] = None, random_state: int = 42
) -> xgb.XGBClassifier:
    """Returns a configured XGBoost Classifier for binary price direction prediction."""
    default_params = {
        "n_estimators": 1500,
        "learning_rate": 0.03,
        "max_depth": 6,
        "min_child_weight": 2,
        "subsample": 0.85,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "random_state": random_state,
        "n_jobs": -1,
        "tree_method": "hist",
    }
    if custom_params:
        default_params.update(custom_params)
    return xgb.XGBClassifier(**default_params)


class TabularResBlock(nn.Module):
    """Residual Block for tabular continuous feature representations."""

    def __init__(self, hidden_dim: int, dropout_rate: float = 0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class TabularResNet(nn.Module):
    """Deep Tabular Network with residual connections, layer normalization, and dropout regularization."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_blocks: int = 3,
        dropout_rate: float = 0.2,
        output_dim: int = 1,
    ):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate / 2.0),
        )
        self.res_blocks = nn.ModuleList(
            [TabularResBlock(hidden_dim, dropout_rate) for _ in range(num_blocks)]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate / 2.0),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_layer(x)
        for block in self.res_blocks:
            h = block(h)
        logits = self.head(h)
        return logits.squeeze(-1)


def build_neural_model_and_optimizer(
    input_dim: int,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    epochs: int = 40,
    hidden_dim: int = 128,
    num_blocks: int = 3,
    dropout_rate: float = 0.2,
    device: Optional[torch.device] = None,
) -> Tuple[nn.Module, nn.Module, torch.optim.Optimizer, Any]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TabularResNet(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_blocks=num_blocks,
        dropout_rate=dropout_rate,
        output_dim=1,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    return model, criterion, optimizer, scheduler


# =========================================================================
# 2. Data Loading & Leak-Free Feature Engineering
# =========================================================================
train_path = "./input/train.parquet"
test_path = "./input/test.parquet"

df_train = pd.read_parquet(train_path)
df_test = pd.read_parquet(test_path)

train_cols_map = {c: c.lower() for c in df_train.columns}
test_cols_map = {c: c.lower() for c in df_test.columns}
df_train = df_train.rename(columns=train_cols_map)
df_test = df_test.rename(columns=test_cols_map)

target_col = "class"
if target_col not in df_train.columns:
    possible_targets = [c for c in df_train.columns if "class" in c or "target" in c]
    target_col = possible_targets[0]

unique_targets = df_train[target_col].dropna().unique()
if set(unique_targets).issubset({"UP", "DOWN", "up", "down"}):
    target_mapping = {"DOWN": 0, "down": 0, "UP": 1, "up": 1}
    reverse_mapping = {0: "DOWN", 1: "UP"}
elif set(unique_targets).issubset({0, 1, "0", "1"}):
    target_mapping = {0: 0, "0": 0, 1: 1, "1": 1}
    reverse_mapping = {0: 0, 1: 1}
else:
    sorted_vals = sorted(list(unique_targets))
    target_mapping = {val: idx for idx, val in enumerate(sorted_vals)}
    reverse_mapping = {idx: val for idx, val in enumerate(sorted_vals)}

y_raw = df_train[target_col].map(target_mapping).astype(int).values
X_raw = df_train.drop(columns=[target_col]).copy()
X_test_raw = df_test.copy()

if "id" in X_test_raw.columns:
    test_ids = X_test_raw["id"].values
else:
    test_ids = np.arange(len(X_test_raw))

# Stratified Validation Split (80/20)
train_idx, val_idx = train_test_split(
    np.arange(len(df_train)), test_size=0.20, random_state=42, stratify=y_raw
)

df_tr = X_raw.iloc[train_idx].copy().reset_index(drop=True)
y_tr = y_raw[train_idx]

df_val = X_raw.iloc[val_idx].copy().reset_index(drop=True)
y_val = y_raw[val_idx]

df_te = X_test_raw.copy().reset_index(drop=True)


def extract_base_and_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    feat = pd.DataFrame(index=df.index)

    nsw_price = pd.to_numeric(df.get("nswprice", 0), errors="coerce").fillna(0.0)
    nsw_demand = pd.to_numeric(df.get("nswdemand", 0), errors="coerce").fillna(0.0)
    vic_price = pd.to_numeric(df.get("vicprice", 0), errors="coerce").fillna(0.0)
    vic_demand = pd.to_numeric(df.get("vicdemand", 0), errors="coerce").fillna(0.0)
    transfer = pd.to_numeric(df.get("transfer", 0), errors="coerce").fillna(0.0)
    day = pd.to_numeric(df.get("day", 1), errors="coerce").fillna(1.0)
    period = pd.to_numeric(df.get("period", 0), errors="coerce").fillna(0.0)
    date_val = pd.to_numeric(df.get("date", 0), errors="coerce").fillna(0.0)

    feat["date"] = date_val
    feat["day"] = day
    feat["period"] = period
    feat["nswprice"] = nsw_price
    feat["nswdemand"] = nsw_demand
    feat["vicprice"] = vic_price
    feat["vicdemand"] = vic_demand
    feat["transfer"] = transfer

    # 1. Cyclical time encodings
    period_norm = period if period.max() <= 1.0 else (period - 1.0) / 48.0
    feat["sin_period"] = np.sin(2.0 * np.pi * period_norm)
    feat["cos_period"] = np.cos(2.0 * np.pi * period_norm)

    day_norm = (day - 1.0) / 7.0
    feat["sin_day"] = np.sin(2.0 * np.pi * day_norm)
    feat["cos_day"] = np.cos(2.0 * np.pi * day_norm)
    feat["is_weekend"] = ((day == 1) | (day == 7)).astype(float)

    # 2. Market Spreads and Relative Pricing Dynamics
    eps = 1e-5
    feat["price_spread_nsw_vic"] = nsw_price - vic_price
    feat["abs_price_spread"] = np.abs(nsw_price - vic_price)
    feat["demand_spread_nsw_vic"] = nsw_demand - vic_demand
    feat["total_demand"] = nsw_demand + vic_demand
    feat["demand_ratio_nsw"] = nsw_demand / (feat["total_demand"] + eps)

    # 3. Market Pressure and Reserve Stress Indices
    feat["nsw_market_pressure"] = nsw_price * nsw_demand
    feat["vic_market_pressure"] = vic_price * vic_demand
    feat["nsw_price_elasticity"] = nsw_price / (nsw_demand + eps)
    feat["vic_price_elasticity"] = vic_price / (vic_demand + eps)

    # 4. Inter-state Transmission & Energy Flow Dynamics
    feat["transfer_to_nsw_demand"] = transfer / (nsw_demand + eps)
    feat["transfer_to_vic_demand"] = transfer / (vic_demand + eps)
    feat["transfer_price_interaction"] = transfer * (nsw_price - vic_price)
    feat["transfer_direction_flow"] = transfer - 0.5

    # 5. Non-linear polynomial terms
    feat["nsw_price_sq"] = nsw_price**2
    feat["nsw_demand_sq"] = nsw_demand**2
    feat["vic_price_sq"] = vic_price**2
    feat["vic_demand_sq"] = vic_demand**2

    return feat


X_tr = extract_base_and_interaction_features(df_tr)
X_val = extract_base_and_interaction_features(df_val)
X_te = extract_base_and_interaction_features(df_te)

# 6. Group baseline stats computed strictly on train split
group_cols = ["day", "period"]
stats_cols = ["nswprice", "nswdemand", "vicprice", "vicdemand", "transfer"]

grouped = X_tr.groupby(group_cols)[stats_cols].agg(["mean", "std"]).reset_index()
grouped.columns = [
    f"{col}_{stat}_by_day_period" if stat != "" else col
    for col, stat in grouped.columns
]

X_tr = pd.merge(X_tr, grouped, on=group_cols, how="left")
X_val = pd.merge(X_val, grouped, on=group_cols, how="left")
X_te = pd.merge(X_te, grouped, on=group_cols, how="left")

for col in stats_cols:
    mean_col = f"{col}_mean_by_day_period"
    std_col = f"{col}_std_by_day_period"

    global_mean = float(X_tr[col].mean())
    global_std = float(X_tr[col].std()) if float(X_tr[col].std()) > 0.0 else 1.0

    X_tr[mean_col] = X_tr[mean_col].fillna(global_mean)
    X_tr[std_col] = X_tr[std_col].fillna(global_std).replace(0.0, global_std)

    X_val[mean_col] = X_val[mean_col].fillna(global_mean)
    X_val[std_col] = X_val[std_col].fillna(global_std).replace(0.0, global_std)

    X_te[mean_col] = X_te[mean_col].fillna(global_mean)
    X_te[std_col] = X_te[std_col].fillna(global_std).replace(0.0, global_std)

    X_tr[f"{col}_zscore_period"] = (X_tr[col] - X_tr[mean_col]) / (X_tr[std_col] + 1e-5)
    X_val[f"{col}_zscore_period"] = (X_val[col] - X_val[mean_col]) / (
        X_val[std_col] + 1e-5
    )
    X_te[f"{col}_zscore_period"] = (X_te[col] - X_te[mean_col]) / (X_te[std_col] + 1e-5)

X_tr = X_tr.replace([np.inf, -np.inf], np.nan).fillna(0.0)
X_val = X_val.replace([np.inf, -np.inf], np.nan).fillna(0.0)
X_te = X_te.replace([np.inf, -np.inf], np.nan).fillna(0.0)

feature_names = list(X_tr.columns)
X_val = X_val[feature_names]
X_te = X_te[feature_names]


# =========================================================================
# 3. Model Training: LightGBM & XGBoost
# =========================================================================
# LightGBM
lgb_model = get_lgbm_model(random_state=42)
lgb_model.fit(
    X_tr,
    y_tr,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
)
joblib.dump(lgb_model, "./working/best_lgb_model.joblib")
val_probs_lgb = lgb_model.predict_proba(X_val)[:, 1]
test_probs_lgb = lgb_model.predict_proba(X_te)[:, 1]

# XGBoost
xgb_model = get_xgb_model(custom_params={"early_stopping_rounds": 50}, random_state=42)
xgb_model.fit(
    X_tr,
    y_tr,
    eval_set=[(X_val, y_val)],
    verbose=False,
)
joblib.dump(xgb_model, "./working/best_xgb_model.joblib")
val_probs_xgb = xgb_model.predict_proba(X_val)[:, 1]
test_probs_xgb = xgb_model.predict_proba(X_te)[:, 1]


# =========================================================================
# 4. Model Training: TabularResNet (PyTorch)
# =========================================================================
scaler = StandardScaler()
X_tr_scaled = scaler.fit_transform(X_tr.values)
X_val_scaled = scaler.transform(X_val.values)
X_te_scaled = scaler.transform(X_te.values)
joblib.dump(scaler, "./working/scaler.joblib")

train_tensor = TensorDataset(
    torch.tensor(X_tr_scaled, dtype=torch.float32),
    torch.tensor(y_tr, dtype=torch.float32),
)
val_tensor = TensorDataset(
    torch.tensor(X_val_scaled, dtype=torch.float32),
    torch.tensor(y_val, dtype=torch.float32),
)
test_tensor = TensorDataset(torch.tensor(X_te_scaled, dtype=torch.float32))

train_loader = DataLoader(
    train_tensor, batch_size=256, shuffle=True, drop_last=False, num_workers=2
)
val_loader = DataLoader(val_tensor, batch_size=512, shuffle=False, num_workers=2)
test_loader = DataLoader(test_tensor, batch_size=512, shuffle=False, num_workers=2)

epochs = 40
nn_model, nn_loss, nn_opt, nn_sched = build_neural_model_and_optimizer(
    input_dim=X_tr_scaled.shape[1],
    lr=1e-3,
    weight_decay=1e-4,
    epochs=epochs,
    device=device,
)

best_val_loss = float("inf")
best_model_path = "./working/best_tabular_resnet.pth"

for epoch in range(1, epochs + 1):
    nn_model.train()
    for bx, by in train_loader:
        bx, by = bx.to(device), by.to(device)
        nn_opt.zero_grad()
        logits = nn_model(bx)
        loss = nn_loss(logits, by)
        loss.backward()
        nn_opt.step()
    nn_sched.step()

    nn_model.eval()
    total_val_loss = 0.0
    with torch.no_grad():
        for bx, by in val_loader:
            bx, by = bx.to(device), by.to(device)
            logits = nn_model(bx)
            loss = nn_loss(logits, by)
            total_val_loss += loss.item() * len(by)

    epoch_val_loss = total_val_loss / len(y_val)
    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        torch.save(nn_model.state_dict(), best_model_path)

    if epoch % 10 == 0 or epoch == epochs:
        print(f"Epoch {epoch}/{epochs} | Val Loss: {epoch_val_loss:.4f}")

nn_model.load_state_dict(torch.load(best_model_path, map_location=device))
nn_model.eval()

val_probs_nn_list = []
with torch.no_grad():
    for bx, _ in val_loader:
        bx = bx.to(device)
        probs = torch.sigmoid(nn_model(bx)).cpu().numpy()
        val_probs_nn_list.extend(probs)
val_probs_nn = np.array(val_probs_nn_list)

test_probs_nn_list = []
with torch.no_grad():
    for (bx,) in test_loader:
        bx = bx.to(device)
        probs = torch.sigmoid(nn_model(bx)).cpu().numpy()
        test_probs_nn_list.extend(probs)
test_probs_nn = np.array(test_probs_nn_list)


# =========================================================================
# 5. Validation Ensembling & Post-Processing
# =========================================================================
best_acc = 0.0
best_weights = (0.45, 0.45, 0.10)
best_thresh = 0.5

w_candidates = [
    (0.50, 0.50, 0.00),
    (0.45, 0.45, 0.10),
    (0.40, 0.40, 0.20),
    (0.60, 0.40, 0.00),
    (0.40, 0.60, 0.00),
    (0.50, 0.35, 0.15),
    (0.35, 0.50, 0.15),
]

for w_lgb, w_xgb, w_nn in w_candidates:
    blend_val = w_lgb * val_probs_lgb + w_xgb * val_probs_xgb + w_nn * val_probs_nn
    for thresh in np.linspace(0.45, 0.55, 11):
        cand_preds = (blend_val >= thresh).astype(int)
        cand_acc = accuracy_score(y_val, cand_preds)
        if cand_acc > best_acc:
            best_acc = cand_acc
            best_weights = (w_lgb, w_xgb, w_nn)
            best_thresh = thresh

w_lgb, w_xgb, w_nn = best_weights
val_ensemble_probs = w_lgb * val_probs_lgb + w_xgb * val_probs_xgb + w_nn * val_probs_nn
final_val_preds = (val_ensemble_probs >= best_thresh).astype(int)
final_val_score = accuracy_score(y_val, final_val_preds)


# =========================================================================
# 6. Submission Generation
# =========================================================================
test_ensemble_probs = (
    w_lgb * test_probs_lgb + w_xgb * test_probs_xgb + w_nn * test_probs_nn
)
final_test_preds_binary = (test_ensemble_probs >= best_thresh).astype(int)
test_pred_labels = [reverse_mapping.get(int(p), p) for p in final_test_preds_binary]

submission_df = pd.DataFrame({"id": test_ids, "class": test_pred_labels})
submission_df.to_csv("./submission/submission.csv", index=False)

print(f"Final Validation Score: {final_val_score}")
