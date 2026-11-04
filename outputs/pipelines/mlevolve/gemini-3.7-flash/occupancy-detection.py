import math
import os
from typing import Any, Dict, List, Optional, Tuple, Union
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb

warnings.filterwarnings("ignore")

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create required output directories
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)


# =========================================================================
# 1. Feature Engineering
# =========================================================================
def engineer_features(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    """
    Engineer comprehensive physical, temporal, and dynamic time-series features
    for room occupancy detection.
    """
    data = df.copy()

    # Create integer id if not present to preserve original ordering
    if "id" not in data.columns:
        data["id"] = np.arange(len(data))

    core_sensors = ["CO2", "Light", "Temperature", "Humidity", "HumidityRatio"]
    for col in core_sensors:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")
    data[core_sensors] = data[core_sensors].ffill().bfill().fillna(0.0)

    if "Occupancy" in data.columns:
        data["Occupancy"] = pd.to_numeric(data["Occupancy"], errors="coerce").fillna(0).astype(int)

    # Ensure sorted by date/id for accurate sequential and rolling feature computation
    if "date" in data.columns:
        data["date"] = pd.to_datetime(data["date"], errors="coerce")
        data = data.sort_values(by=["date", "id"]).reset_index(drop=True)
    else:
        data = data.sort_values(by="id").reset_index(drop=True)

    # --- Temporal & Cyclical Features ---
    if "date" in data.columns:
        data["hour"] = data["date"].dt.hour
        data["minute"] = data["date"].dt.minute
        data["dayofweek"] = data["date"].dt.dayofweek
        data["is_weekend"] = (data["dayofweek"] >= 5).astype(int)

        # Work hours indicator (e.g. 7:30 to 18:30 on weekdays)
        minute_of_day = data["hour"] * 60 + data["minute"]
        data["minute_of_day"] = minute_of_day
        data["is_work_hours"] = (
            (data["is_weekend"] == 0)
            & (minute_of_day >= 7 * 60 + 30)
            & (minute_of_day <= 18 * 60 + 30)
        ).astype(int)

        # Cyclical transformations
        data["sin_hour"] = np.sin(2 * np.pi * data["hour"] / 24.0)
        data["cos_hour"] = np.cos(2 * np.pi * data["hour"] / 24.0)
        data["sin_min_of_day"] = np.sin(2 * np.pi * minute_of_day / 1440.0)
        data["cos_min_of_day"] = np.cos(2 * np.pi * minute_of_day / 1440.0)
        data["sin_dayofweek"] = np.sin(2 * np.pi * data["dayofweek"] / 7.0)
        data["cos_dayofweek"] = np.cos(2 * np.pi * data["dayofweek"] / 7.0)

    # --- Sensor Physical & Thermodynamic Features ---
    # Log transforms for skewed positive features
    data["Light_log"] = np.log1p(np.maximum(data["Light"], 0))
    data["CO2_log"] = np.log1p(np.maximum(data["CO2"], 0))

    # Binary threshold indicator for light
    data["Light_is_zero"] = (data["Light"] <= 0).astype(int)
    data["Light_gt_100"] = (data["Light"] > 100).astype(int)
    data["Light_gt_400"] = (data["Light"] > 400).astype(int)

    # CO2 threshold indicators (ambient outdoor CO2 is ~400 ppm)
    data["CO2_elevated"] = (data["CO2"] > 500).astype(int)
    data["CO2_high"] = (data["CO2"] > 800).astype(int)
    data["CO2_excess"] = np.maximum(data["CO2"] - 400.0, 0.0)

    # Physical sensor interactions
    data["Temp_x_Humidity"] = data["Temperature"] * data["Humidity"]
    data["CO2_x_Light"] = data["CO2"] * data["Light"]
    data["CO2_per_Light"] = data["CO2"] / (data["Light"] + 1.0)
    data["CO2_per_Temp"] = data["CO2"] / (data["Temperature"] + 1e-5)
    data["Humidity_per_Temp"] = data["Humidity"] / (data["Temperature"] + 1e-5)

    # Magnus formula for Dew Point approximation (°C)
    a, b = 17.27, 237.7
    temp_c = data["Temperature"]
    rh_norm = np.clip(data["Humidity"] / 100.0, 1e-4, 1.0)
    alpha = ((a * temp_c) / (b + temp_c)) + np.log(rh_norm)
    data["Dew_Point"] = (b * alpha) / (a - alpha)
    data["Temp_DewPoint_Spread"] = data["Temperature"] - data["Dew_Point"]

    # Specific Enthalpy approximation (kJ/kg)
    data["Enthalpy"] = 1.006 * data["Temperature"] + data["HumidityRatio"] * (
        2501.0 + 1.86 * data["Temperature"]
    )

    # --- Dynamic Sequence & Rolling Features ---
    core_sensors = ["CO2", "Light", "Temperature", "Humidity", "HumidityRatio"]

    for col in core_sensors:
        # Backward Lags (History)
        data[f"{col}_lag1"] = data[col].shift(1)
        data[f"{col}_lag2"] = data[col].shift(2)
        data[f"{col}_lag5"] = data[col].shift(5)
        data[f"{col}_lag15"] = data[col].shift(15)

        # Forward Leads (Future sequence context)
        data[f"{col}_lead1"] = data[col].shift(-1)
        data[f"{col}_lead2"] = data[col].shift(-2)
        data[f"{col}_lead5"] = data[col].shift(-5)

        # Rates of change (First Derivative)
        data[f"{col}_diff1"] = data[col] - data[f"{col}_lag1"]
        data[f"{col}_diff5"] = data[col] - data[f"{col}_lag5"]
        data[f"{col}_diff15"] = data[col] - data[f"{col}_lag15"]

        # Acceleration (Second Derivative)
        data[f"{col}_accel"] = data[f"{col}_diff1"] - data[f"{col}_diff1"].shift(1)

        # Multi-scale Rolling Statistics
        for window in [5, 15, 30, 60]:
            roll = data[col].rolling(window=window, min_periods=1)
            data[f"{col}_roll_mean_{window}"] = roll.mean()
            data[f"{col}_roll_std_{window}"] = roll.std().fillna(0)
            data[f"{col}_roll_min_{window}"] = roll.min()
            data[f"{col}_roll_max_{window}"] = roll.max()
            data[f"{col}_roll_range_{window}"] = (
                data[f"{col}_roll_max_{window}"] - data[f"{col}_roll_min_{window}"]
            )
            data[f"{col}_diff_from_roll_mean_{window}"] = (
                data[col] - data[f"{col}_roll_mean_{window}"]
            )

        # Exponential Moving Averages
        data[f"{col}_ema_5"] = data[col].ewm(span=5, adjust=False).mean()
        data[f"{col}_ema_15"] = data[col].ewm(span=15, adjust=False).mean()
        data[f"{col}_ema_60"] = data[col].ewm(span=60, adjust=False).mean()
        data[f"{col}_diff_from_ema_15"] = data[col] - data[f"{col}_ema_15"]

    # Backfill/Forward-fill edge missing values produced by shifts
    data = data.bfill().ffill()

    # Restore original order by id
    data = data.sort_values(by="id").reset_index(drop=True)
    return data


# =========================================================================
# 2. Model Definitions
# =========================================================================
def get_lgbm_params() -> Dict[str, Any]:
    return {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "n_estimators": 2500,
        "learning_rate": 0.02,
        "num_leaves": 31,
        "max_depth": 6,
        "min_child_samples": 20,
        "subsample": 0.85,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }


def get_xgb_params() -> Dict[str, Any]:
    return {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "booster": "gbtree",
        "n_estimators": 2500,
        "learning_rate": 0.02,
        "max_depth": 5,
        "min_child_weight": 3,
        "subsample": 0.85,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": -1,
        "tree_method": "hist",
    }


class ResidualBlock(nn.Module):
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
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.block(x))


class TabularResNetClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_blocks: int = 3,
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )

        self.res_blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim, dropout_rate) for _ in range(num_blocks)]
        )

        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate / 2.0),
            nn.Linear(hidden_dim // 2, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_layer(x)
        for block in self.res_blocks:
            h = block(h)
        logits = self.head(h)
        return logits.squeeze(-1)


def get_neural_criterion(pos_weight: Optional[torch.Tensor] = None) -> nn.Module:
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)


def get_neural_optimizer(
    model: nn.Module,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
) -> optim.Optimizer:
    return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def get_neural_scheduler(
    optimizer: optim.Optimizer,
    max_epochs: int,
    steps_per_epoch: int = 1,
) -> optim.lr_scheduler.CosineAnnealingLR:
    return optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_epochs * steps_per_epoch, eta_min=1e-6
    )


# =========================================================================
# 3. Data Loading & Feature Generation
# =========================================================================
print("Loading raw datasets...")
raw_train = pd.read_parquet("./input/train.parquet")
raw_test = pd.read_parquet("./input/test.parquet")
print(f"Raw Train: {raw_train.shape}, Raw Test: {raw_test.shape}")

print("Engineering features...")
train_df = engineer_features(raw_train, is_train=True)
test_df = engineer_features(raw_test, is_train=False)

# Identify features and setup Stratified K-Fold
drop_cols = ["id", "date", "Occupancy"]
feature_cols = [c for c in train_df.columns if c not in drop_cols]
n_features = len(feature_cols)
n_splits = 5

skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
train_df["fold"] = -1
for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df["Occupancy"])):
    train_df.loc[val_idx, "fold"] = fold

X = train_df[feature_cols].values.astype(np.float32)
y = train_df["Occupancy"].values.astype(np.float32)
folds = train_df["fold"].values
X_test = test_df[feature_cols].values.astype(np.float32)

# Sanitize potential numerical anomalies
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

print(f"Features: {n_features}, Train rows: {len(train_df)}, Test rows: {len(test_df)}")

# =========================================================================
# 4. Model Training & Cross-Validation
# =========================================================================
oof_lgb = np.zeros(len(train_df), dtype=np.float64)
oof_xgb = np.zeros(len(train_df), dtype=np.float64)
oof_nn = np.zeros(len(train_df), dtype=np.float64)

test_preds_lgb = np.zeros(len(test_df), dtype=np.float64)
test_preds_xgb = np.zeros(len(test_df), dtype=np.float64)
test_preds_nn = np.zeros(len(test_df), dtype=np.float64)

for fold in range(n_splits):
    train_idx = np.where(folds != fold)[0]
    val_idx = np.where(folds == fold)[0]

    X_train_f, y_train_f = X[train_idx], y[train_idx]
    X_val_f, y_val_f = X[val_idx], y[val_idx]

    # --- A. LightGBM ---
    lgb_params = get_lgbm_params()
    lgb_clf = lgb.LGBMClassifier(**lgb_params)
    lgb_clf.fit(
        X_train_f,
        y_train_f,
        eval_set=[(X_val_f, y_val_f)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )
    val_pred_lgb = lgb_clf.predict_proba(X_val_f)[:, 1]
    oof_lgb[val_idx] = val_pred_lgb
    test_preds_lgb += lgb_clf.predict_proba(X_test)[:, 1] / n_splits

    # --- B. XGBoost ---
    xgb_params = get_xgb_params()
    xgb_clf = xgb.XGBClassifier(early_stopping_rounds=50, **xgb_params)
    xgb_clf.fit(
        X_train_f,
        y_train_f,
        eval_set=[(X_val_f, y_val_f)],
        verbose=False,
    )
    val_pred_xgb = xgb_clf.predict_proba(X_val_f)[:, 1]
    oof_xgb[val_idx] = val_pred_xgb
    test_preds_xgb += xgb_clf.predict_proba(X_test)[:, 1] / n_splits

    # --- C. Tabular Residual Neural Network ---
    scaler = StandardScaler()
    X_train_f_scaled = scaler.fit_transform(X_train_f)
    X_val_f_scaled = scaler.transform(X_val_f)
    X_test_scaled = scaler.transform(X_test)

    train_dataset = TensorDataset(
        torch.tensor(X_train_f_scaled, dtype=torch.float32),
        torch.tensor(y_train_f, dtype=torch.float32),
    )
    train_loader = DataLoader(
        train_dataset, batch_size=256, shuffle=True, drop_last=False
    )

    nn_model = TabularResNetClassifier(
        input_dim=n_features, hidden_dim=128, num_blocks=3, dropout_rate=0.2
    ).to(device)
    criterion = get_neural_criterion()
    optimizer = get_neural_optimizer(nn_model, lr=1e-3, weight_decay=1e-4)
    epochs = 25
    scheduler = get_neural_scheduler(
        optimizer, max_epochs=epochs, steps_per_epoch=len(train_loader)
    )

    best_val_auc = 0.0
    best_nn_preds = None
    test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
    best_test_nn_preds = None

    for epoch in range(epochs):
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
        with torch.no_grad():
            val_tensor = torch.tensor(X_val_f_scaled, dtype=torch.float32).to(device)
            val_logits = nn_model(val_tensor)
            val_probs = torch.sigmoid(val_logits).cpu().numpy()
            fold_auc = roc_auc_score(y_val_f, val_probs)

            if fold_auc > best_val_auc:
                best_val_auc = fold_auc
                best_nn_preds = val_probs
                best_test_nn_preds = torch.sigmoid(nn_model(test_tensor)).cpu().numpy()

    oof_nn[val_idx] = best_nn_preds
    test_preds_nn += best_test_nn_preds / n_splits

    fold_lgb_auc = roc_auc_score(y_val_f, val_pred_lgb)
    fold_xgb_auc = roc_auc_score(y_val_f, val_pred_xgb)
    print(
        f"Fold {fold + 1}/{n_splits} | LGB AUC: {fold_lgb_auc:.5f} | XGB AUC: {fold_xgb_auc:.5f} | ResNet AUC: {best_val_auc:.5f}"
    )

# =========================================================================
# 5. Ensemble Optimization & Validation Scoring
# =========================================================================
lgb_total_auc = roc_auc_score(y, oof_lgb)
xgb_total_auc = roc_auc_score(y, oof_xgb)
nn_total_auc = roc_auc_score(y, oof_nn)

print(
    f"\nIndividual OOF Scores -> LGB: {lgb_total_auc:.5f} | XGB: {xgb_total_auc:.5f} | ResNet: {nn_total_auc:.5f}"
)


def loss_func(weights):
    w1, w2, w3 = weights
    w_sum = w1 + w2 + w3
    if w_sum <= 0:
        return 1.0
    pred = (w1 * oof_lgb + w2 * oof_xgb + w3 * oof_nn) / w_sum
    return -roc_auc_score(y, pred)


initial_weights = [0.45, 0.45, 0.10]
bounds = [(0, 1), (0, 1), (0, 1)]
opt_res = minimize(loss_func, initial_weights, bounds=bounds, method="Nelder-Mead")
best_w = opt_res.x / np.sum(opt_res.x)
print(
    f"Ensemble Weights: LGB={best_w[0]:.4f}, XGB={best_w[1]:.4f}, ResNet={best_w[2]:.4f}"
)

oof_ensemble = best_w[0] * oof_lgb + best_w[1] * oof_xgb + best_w[2] * oof_nn
final_val_score = roc_auc_score(y, oof_ensemble)

# =========================================================================
# 6. Generate Submission
# =========================================================================
test_ensemble = (
    best_w[0] * test_preds_lgb + best_w[1] * test_preds_xgb + best_w[2] * test_preds_nn
)

submission_df = pd.DataFrame(
    {"id": test_df["id"].astype(int), "Occupancy": test_ensemble}
)

submission_path = "./submission/submission.csv"
submission_df.to_csv(submission_path, index=False)
print(f"Submission successfully saved to {submission_path}.")

# Final Validation Score Output
print(f"Final Validation Score: {final_val_score}")
