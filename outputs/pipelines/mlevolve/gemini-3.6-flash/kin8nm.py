import os
import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Ensure output directories exist
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# 1. Load Raw Data
train_path = "./input/train.parquet"
test_path = "./input/test.parquet"

train_raw = pd.read_parquet(train_path)
test_raw = pd.read_parquet(test_path)

target_col = "y"
id_col = "id" if "id" in test_raw.columns else None

if id_col is not None:
    raw_feature_cols = [c for c in train_raw.columns if c not in [target_col, id_col]]
else:
    raw_feature_cols = [c for c in train_raw.columns if c != target_col]


# 2. Kinematic Feature Engineering
def create_kinematic_features(df, feat_cols):
    """Computes domain-specific forward kinematic transformations for an 8-link arm."""
    X_raw = df[feat_cols].values.astype(np.float64)
    n_samples, n_joints = X_raw.shape

    features = {}

    # Raw features
    for i, col in enumerate(feat_cols):
        features[f"{col}"] = X_raw[:, i]

    # Individual trigonometric transformations
    sin_x = np.sin(X_raw)
    cos_x = np.cos(X_raw)
    for i, col in enumerate(feat_cols):
        features[f"sin_{col}"] = sin_x[:, i]
        features[f"cos_{col}"] = cos_x[:, i]

    # Cumulative angle sums (kinematic chain propagation)
    cum_angles = np.cumsum(X_raw, axis=1)
    for k in range(n_joints):
        features[f"cum_angle_{k+1}"] = cum_angles[:, k]
        features[f"sin_cum_{k+1}"] = np.sin(cum_angles[:, k])
        features[f"cos_cum_{k+1}"] = np.cos(cum_angles[:, k])

    # Kinematic joint positions in 2D Cartesian plane assuming unit link lengths
    cos_cum_all = np.cos(cum_angles)
    sin_cum_all = np.sin(cum_angles)

    PX = np.cumsum(cos_cum_all, axis=1)
    PY = np.cumsum(sin_cum_all, axis=1)

    for k in range(n_joints):
        features[f"PX_{k+1}"] = PX[:, k]
        features[f"PY_{k+1}"] = PY[:, k]
        features[f"radial_dist_{k+1}"] = np.sqrt(PX[:, k] ** 2 + PY[:, k] ** 2)
        features[f"polar_angle_{k+1}"] = np.arctan2(PY[:, k], PX[:, k])

    # Relative adjacent joint angular differences
    for i in range(1, n_joints):
        diff = X_raw[:, i] - X_raw[:, i - 1]
        features[f"angle_diff_{i+1}_{i}"] = diff
        features[f"sin_diff_{i+1}_{i}"] = np.sin(diff)
        features[f"cos_diff_{i+1}_{i}"] = np.cos(diff)

    # Cross-product interaction terms between adjacent trigonometric components
    for i in range(n_joints - 1):
        features[f"sin_prod_{i+1}_{i+2}"] = sin_x[:, i] * sin_x[:, i + 1]
        features[f"cos_prod_{i+1}_{i+2}"] = cos_x[:, i] * cos_x[:, i + 1]
        features[f"sincos_cross_{i+1}_{i+2}"] = sin_x[:, i] * cos_x[:, i + 1]

    # Statistical summary metrics per state
    features["sum_abs_angles"] = np.sum(np.abs(X_raw), axis=1)
    features["std_angles"] = np.std(X_raw, axis=1)
    features["max_angle"] = np.max(X_raw, axis=1)
    features["min_angle"] = np.min(X_raw, axis=1)
    features["l2_norm_angles"] = np.linalg.norm(X_raw, axis=1)

    return pd.DataFrame(features, index=df.index)


train_features = create_kinematic_features(train_raw, raw_feature_cols)
test_features = create_kinematic_features(test_raw, raw_feature_cols)

feature_cols = list(train_features.columns)
n_features = len(feature_cols)

# Set targets and test IDs
y_all = train_raw[target_col].values
if id_col is not None and id_col in test_raw.columns:
    test_ids = test_raw[id_col].values
else:
    test_ids = np.arange(len(test_raw))

# Generate 5-fold cross validation split
kf = KFold(n_splits=5, shuffle=True, random_state=42)
folds_all = np.zeros(len(train_features), dtype=int)
for fold, (_, val_idx) in enumerate(kf.split(train_features)):
    folds_all[val_idx] = fold

X_all = train_features.values
X_test = test_features.values
n_splits = 5


# 3. PyTorch Deep Kinematic ResNet Architecture
class KinematicBlock(nn.Module):

    def __init__(self, hidden_dim: int, dropout_rate: float = 0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class KinematicResNet(nn.Module):

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_blocks: int = 3,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU()
        )
        self.blocks = nn.ModuleList(
            [KinematicBlock(hidden_dim, dropout_rate) for _ in range(num_blocks)]
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x)
        for block in self.blocks:
            h = block(h)
        out = self.head(h)
        return out.squeeze(-1)


# 4. Hyperparameter Configurations for GBDT Models
lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "n_estimators": 3000,
    "learning_rate": 0.03,
    "num_leaves": 63,
    "max_depth": 8,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "max_bin": 512,
    "random_state": 42,
    "verbose": -1,
    "n_jobs": -1,
}

xgb_params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "n_estimators": 3000,
    "learning_rate": 0.03,
    "max_depth": 7,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": -1,
    "tree_method": "hist",
}

cat_params = {
    "loss_function": "RMSE",
    "eval_metric": "RMSE",
    "iterations": 3000,
    "learning_rate": 0.03,
    "depth": 7,
    "l2_leaf_reg": 3.0,
    "random_seed": 42,
    "verbose": False,
    "thread_count": -1,
}

# Prediction arrays for ensemble [LightGBM, XGBoost, CatBoost, KinematicResNet]
oof_preds = np.zeros((len(train_features), 4))
test_preds = np.zeros((len(test_features), 4))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 5. Cross-Validation Training Loop
for fold in range(n_splits):
    train_idx = np.where(folds_all != fold)[0]
    val_idx = np.where(folds_all == fold)[0]

    X_tr, y_tr = X_all[train_idx], y_all[train_idx]
    X_va, y_va = X_all[val_idx], y_all[val_idx]

    # --- Model 1: LightGBM ---
    model_lgb = lgb.LGBMRegressor(**lgb_params)
    model_lgb.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[
            lgb.early_stopping(50, verbose=False),
            lgb.log_evaluation(-1),
        ],
    )
    oof_preds[val_idx, 0] = model_lgb.predict(X_va)
    test_preds[:, 0] += model_lgb.predict(X_test) / n_splits

    # --- Model 2: XGBoost ---
    model_xgb = xgb.XGBRegressor(**xgb_params, early_stopping_rounds=50)
    model_xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    oof_preds[val_idx, 1] = model_xgb.predict(X_va)
    test_preds[:, 1] += model_xgb.predict(X_test) / n_splits

    # --- Model 3: CatBoost ---
    model_cat = cb.CatBoostRegressor(
        **cat_params, early_stopping_rounds=50
    )
    model_cat.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=False)
    oof_preds[val_idx, 2] = model_cat.predict(X_va)
    test_preds[:, 2] += model_cat.predict(X_test) / n_splits

    # --- Model 4: PyTorch KinematicResNet ---
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_va_sc = scaler.transform(X_va)
    X_te_sc = scaler.transform(X_test)

    train_ds = TensorDataset(
        torch.tensor(X_tr_sc, dtype=torch.float32),
        torch.tensor(y_tr, dtype=torch.float32),
    )
    val_ds = TensorDataset(
        torch.tensor(X_va_sc, dtype=torch.float32),
        torch.tensor(y_va, dtype=torch.float32),
    )

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)

    model_nn = KinematicResNet(
        input_dim=n_features, hidden_dim=256, num_blocks=3, dropout_rate=0.1
    ).to(device)
    optimizer = optim.AdamW(model_nn.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_nn_weights = None
    patience, max_patience = 0, 15

    for epoch in range(100):
        model_nn.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model_nn(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
        scheduler.step()

        model_nn.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                out = model_nn(bx)
                val_loss += criterion(out, by).item() * len(by)
        val_loss /= len(val_ds)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_nn_weights = {
                k: v.cpu().clone() for k, v in model_nn.state_dict().items()
            }
            patience = 0
        else:
            patience += 1
            if patience >= max_patience:
                break

    model_nn.load_state_dict(best_nn_weights)
    model_nn.eval()

    with torch.no_grad():
        va_tensor = torch.tensor(X_va_sc, dtype=torch.float32).to(device)
        te_tensor = torch.tensor(X_te_sc, dtype=torch.float32).to(device)
        oof_preds[val_idx, 3] = model_nn(va_tensor).cpu().numpy()
        test_preds[:, 3] += model_nn(te_tensor).cpu().numpy() / n_splits

# 6. Meta-Learner Stacking (Ridge Regression)
meta_model = Ridge(alpha=10.0, random_state=42)
meta_model.fit(oof_preds, y_all)

final_oof_preds = meta_model.predict(oof_preds)
final_test_preds = meta_model.predict(test_preds)

final_rmse = np.sqrt(mean_squared_error(y_all, final_oof_preds))

# 7. Write Submission File
sub_df = pd.DataFrame({"id": test_ids, "y": final_test_preds})
sub_df.to_csv("./submission/submission.csv", index=False)

print(f"Final Validation Score: {final_rmse}")
