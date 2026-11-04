import os
import random
import numpy as np
import pandas as pd
import scipy.optimize as opt
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import lightgbm as lgb
import xgboost as xgb

# ---------------------------------------------------------
# 0. Global Setup & Seed Configuration
# ---------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)


# ---------------------------------------------------------
# 1. Feature Engineering: Physics-Informed Kinematics
# ---------------------------------------------------------
def engineer_kinematic_features(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """
    Computes forward kinematics representations for an 8-link robot arm:
    - Single-joint harmonics (sin, cos, sin2x, cos2x)
    - Cumulative kinematic chain angles and harmonics
    - Partial and total cumulative Cartesian coordinate proxies (X_k, Y_k) and reach radii R_k
    - Relative adjacent joint differential flex
    - Global kinematic dispersion metrics
    """
    df_feat = pd.DataFrame(index=df.index)
    angles = df[feature_cols].values
    num_joints = angles.shape[1]

    # Raw joint angles
    for i, col in enumerate(feature_cols):
        df_feat[col] = angles[:, i]

    # Single-joint trigonometric harmonic embeddings
    for i, col in enumerate(feature_cols):
        th = angles[:, i]
        df_feat[f"{col}_sin"] = np.sin(th)
        df_feat[f"{col}_cos"] = np.cos(th)
        df_feat[f"{col}_sin2"] = np.sin(2.0 * th)
        df_feat[f"{col}_cos2"] = np.cos(2.0 * th)

    # Cumulative kinematic chain angles
    cum_angles = np.cumsum(angles, axis=1)
    for i in range(num_joints):
        df_feat[f"cum_angle_{i+1}"] = cum_angles[:, i]
        df_feat[f"cum_sin_{i+1}"] = np.sin(cum_angles[:, i])
        df_feat[f"cum_cos_{i+1}"] = np.cos(cum_angles[:, i])

    # Cumulative Cartesian Reach Proxies
    cum_cos_all = np.cos(cum_angles)
    cum_sin_all = np.sin(cum_angles)
    X_cum = np.cumsum(cum_cos_all, axis=1)
    Y_cum = np.cumsum(cum_sin_all, axis=1)

    for i in range(num_joints):
        df_feat[f"reach_X_{i+1}"] = X_cum[:, i]
        df_feat[f"reach_Y_{i+1}"] = Y_cum[:, i]
        df_feat[f"reach_R_{i+1}"] = np.sqrt(X_cum[:, i] ** 2 + Y_cum[:, i] ** 2)
        df_feat[f"reach_angle_{i+1}"] = np.arctan2(Y_cum[:, i], X_cum[:, i])

    # Adjacent joint differential angles
    for i in range(1, num_joints):
        diff = angles[:, i] - angles[:, i - 1]
        df_feat[f"diff_angle_{i}_{i-1}"] = diff
        df_feat[f"diff_sin_{i}_{i-1}"] = np.sin(diff)
        df_feat[f"diff_cos_{i}_{i-1}"] = np.cos(diff)

    # Global Kinematic Aggregates
    df_feat["angle_mean"] = np.mean(angles, axis=1)
    df_feat["angle_std"] = np.std(angles, axis=1)
    df_feat["angle_abs_sum"] = np.sum(np.abs(angles), axis=1)
    df_feat["angle_sq_sum"] = np.sum(angles**2, axis=1)
    df_feat["total_reach_manhattan"] = np.sum(np.abs(cum_cos_all), axis=1) + np.sum(
        np.abs(cum_sin_all), axis=1
    )
    df_feat["end_effector_R"] = df_feat[f"reach_R_{num_joints}"]

    return df_feat


# Load train and test data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "y"
raw_feature_cols = sorted(
    [c for c in train_df.columns if c != target_col and c != "id"]
)

X_train_eng = engineer_kinematic_features(train_df, raw_feature_cols)
X_test_eng = engineer_kinematic_features(test_df, raw_feature_cols)

feature_cols = sorted(list(X_train_eng.columns))
num_features = len(feature_cols)

train_processed = X_train_eng.copy()
train_processed[target_col] = train_df[target_col].values

test_processed = X_test_eng.copy()
if "id" in test_df.columns:
    test_ids = test_df["id"].values
else:
    test_ids = np.arange(len(test_df))
test_processed["id"] = test_ids

# Leak-free cross-validation partitioning
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
train_processed["fold"] = -1
for fold_idx, (train_idx, val_idx) in enumerate(kf.split(train_processed)):
    train_processed.loc[val_idx, "fold"] = fold_idx


# ---------------------------------------------------------
# 2. Model Architectures & Builders
# ---------------------------------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout_rate: float = 0.05, expansion: int = 2):
        super().__init__()
        mid_dim = hidden_dim * expansion
        self.block = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, mid_dim),
            nn.Mish(),
            nn.Dropout(dropout_rate),
            nn.Linear(mid_dim, hidden_dim),
            nn.Mish(),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class KinematicResNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = [256, 256, 128],
        dropout_rate: float = 0.05,
    ):
        super().__init__()
        self.input_proj = nn.Sequential(nn.Linear(input_dim, hidden_dims[0]), nn.Mish())

        stages = []
        for i in range(len(hidden_dims)):
            curr_dim = hidden_dims[i]
            stages.append(ResidualBlock(hidden_dim=curr_dim, dropout_rate=dropout_rate))
            stages.append(ResidualBlock(hidden_dim=curr_dim, dropout_rate=dropout_rate))
            if i < len(hidden_dims) - 1:
                next_dim = hidden_dims[i + 1]
                stages.append(
                    nn.Sequential(
                        nn.LayerNorm(curr_dim),
                        nn.Linear(curr_dim, next_dim),
                        nn.Mish(),
                    )
                )

        self.backbone = nn.Sequential(*stages)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dims[-1]),
            nn.Linear(hidden_dims[-1], hidden_dims[-1] // 2),
            nn.Mish(),
            nn.Linear(hidden_dims[-1] // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        h = self.backbone(h)
        return self.head(h).squeeze(-1)


class KinematicDenseNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = [192, 192, 192],
        dropout_rate: float = 0.05,
    ):
        super().__init__()
        self.in_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.Mish(),
        )
        self.dense_blocks = nn.ModuleList()
        curr_in = hidden_dims[0]
        for h_dim in hidden_dims:
            layer = nn.Sequential(
                nn.Linear(curr_in, h_dim),
                nn.LayerNorm(h_dim),
                nn.Mish(),
                nn.Dropout(dropout_rate),
            )
            self.dense_blocks.append(layer)
            curr_in += h_dim

        self.out_head = nn.Sequential(
            nn.LayerNorm(curr_in),
            nn.Linear(curr_in, 64),
            nn.Mish(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_layer(x)
        features = [h]
        for block in self.dense_blocks:
            in_feat = torch.cat(features, dim=-1)
            out_feat = block(in_feat)
            features.append(out_feat)

        all_feats = torch.cat(features, dim=-1)
        return self.out_head(all_feats).squeeze(-1)


def build_neural_model(model_name: str, input_dim: int) -> nn.Module:
    if model_name == "kinematic_resnet":
        return KinematicResNet(input_dim=input_dim)
    elif model_name == "kinematic_densenet":
        return KinematicDenseNet(input_dim=input_dim)
    else:
        raise ValueError(f"Unknown model name: {model_name}")


def get_lgbm_regressor_params(random_state: int = 42) -> dict:
    return {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "n_estimators": 2500,
        "learning_rate": 0.03,
        "num_leaves": 63,
        "max_depth": 8,
        "min_child_samples": 20,
        "subsample": 0.85,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": random_state,
        "n_jobs": -1,
        "verbose": -1,
    }


def get_xgb_regressor_params(random_state: int = 42) -> dict:
    return {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "n_estimators": 2500,
        "learning_rate": 0.03,
        "max_depth": 7,
        "subsample": 0.85,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": random_state,
        "n_jobs": -1,
        "tree_method": "hist",
    }


def train_neural_net(
    model_name: str,
    X_train_t: torch.Tensor,
    y_train_t: torch.Tensor,
    X_val_t: torch.Tensor,
    y_val_t: torch.Tensor,
    epochs: int = 65,
    batch_size: int = 64,
    lr: float = 1.2e-3,
    weight_decay: float = 1e-4,
) -> nn.Module:
    model = build_neural_model(model_name, input_dim=num_features).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )

    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=False
    )

    best_val_loss = float("inf")
    best_weights = None
    patience = 14
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_x, val_y = X_val_t.to(device), y_val_t.to(device)
            val_pred = model(val_x)
            val_loss = criterion(val_pred, val_y).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    model.load_state_dict(best_weights)
    model.eval()
    return model


# ---------------------------------------------------------
# 3. 5-Fold Cross-Validation Training & Inference
# ---------------------------------------------------------
test_features_raw = test_processed[feature_cols].values

oof_preds = {
    "resnet": np.zeros(len(train_processed)),
    "densenet": np.zeros(len(train_processed)),
    "lgbm": np.zeros(len(train_processed)),
    "xgb": np.zeros(len(train_processed)),
}

test_preds = {
    "resnet": np.zeros(len(test_processed)),
    "densenet": np.zeros(len(test_processed)),
    "lgbm": np.zeros(len(test_processed)),
    "xgb": np.zeros(len(test_processed)),
}

for fold in range(n_splits):
    train_mask = (train_processed["fold"] != fold).values
    val_mask = (train_processed["fold"] == fold).values

    X_tr_raw = train_processed.loc[train_mask, feature_cols].values
    y_tr = train_processed.loc[train_mask, target_col].values
    X_va_raw = train_processed.loc[val_mask, feature_cols].values
    y_va = train_processed.loc[val_mask, target_col].values

    # Leak-free standard scaling fitted strictly on training fold
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr_raw)
    X_va = scaler.transform(X_va_raw)
    X_te = scaler.transform(test_features_raw)

    X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32)
    X_va_t = torch.tensor(X_va, dtype=torch.float32)
    y_va_t = torch.tensor(y_va, dtype=torch.float32)
    X_te_t = torch.tensor(X_te, dtype=torch.float32).to(device)

    # 1. KinematicResNet
    resnet_model = train_neural_net(
        "kinematic_resnet",
        X_tr_t,
        y_tr_t,
        X_va_t,
        y_va_t,
        epochs=65,
        batch_size=64,
        lr=1.2e-3,
    )
    with torch.no_grad():
        val_pred_res = resnet_model(X_va_t.to(device)).cpu().numpy().flatten()
        test_pred_res = resnet_model(X_te_t).cpu().numpy().flatten()
    oof_preds["resnet"][val_mask] = val_pred_res
    test_preds["resnet"] += test_pred_res / n_splits

    # 2. KinematicDenseNet
    densenet_model = train_neural_net(
        "kinematic_densenet",
        X_tr_t,
        y_tr_t,
        X_va_t,
        y_va_t,
        epochs=65,
        batch_size=64,
        lr=1.2e-3,
    )
    with torch.no_grad():
        val_pred_dense = densenet_model(X_va_t.to(device)).cpu().numpy().flatten()
        test_pred_dense = densenet_model(X_te_t).cpu().numpy().flatten()
    oof_preds["densenet"][val_mask] = val_pred_dense
    test_preds["densenet"] += test_pred_dense / n_splits

    # 3. LightGBM
    lgb_params = get_lgbm_regressor_params(random_state=SEED + fold)
    lgb_model = lgb.LGBMRegressor(**lgb_params)
    lgb_model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(0),
        ],
    )
    oof_preds["lgbm"][val_mask] = lgb_model.predict(X_va)
    test_preds["lgbm"] += lgb_model.predict(X_te) / n_splits

    # 4. XGBoost
    xgb_params = get_xgb_regressor_params(random_state=SEED + fold)
    xgb_model = xgb.XGBRegressor(**xgb_params, early_stopping_rounds=50)
    xgb_model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    oof_preds["xgb"][val_mask] = xgb_model.predict(X_va)
    test_preds["xgb"] += xgb_model.predict(X_te) / n_splits

    print(
        f"Fold {fold+1}/{n_splits} Complete | "
        f"ResNet RMSE: {root_mean_squared_error(y_va, val_pred_res):.5f} | "
        f"DenseNet RMSE: {root_mean_squared_error(y_va, val_pred_dense):.5f} | "
        f"LGBM RMSE: {root_mean_squared_error(y_va, oof_preds['lgbm'][val_mask]):.5f} | "
        f"XGB RMSE: {root_mean_squared_error(y_va, oof_preds['xgb'][val_mask]):.5f}"
    )

# ---------------------------------------------------------
# 4. Optimization & Ensembling
# ---------------------------------------------------------
y_true = train_processed[target_col].values
model_keys = ["resnet", "densenet", "lgbm", "xgb"]
oof_matrix = np.column_stack([oof_preds[k] for k in model_keys])
test_matrix = np.column_stack([test_preds[k] for k in model_keys])


def ensemble_loss(weights):
    w = np.array(weights)
    pred = np.dot(oof_matrix, w)
    return root_mean_squared_error(y_true, pred)


initial_weights = np.ones(len(model_keys)) / len(model_keys)
bounds = [(0.0, 1.0) for _ in model_keys]
constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

opt_res = opt.minimize(
    ensemble_loss,
    initial_weights,
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
)
optimal_weights = opt_res.x if opt_res.success else np.array([0.45, 0.45, 0.05, 0.05])
optimal_weights = optimal_weights / np.sum(optimal_weights)

final_oof_pred = np.dot(oof_matrix, optimal_weights)
final_val_score = float(root_mean_squared_error(y_true, final_oof_pred))

# ---------------------------------------------------------
# 5. Test Inference & Submission Generation
# ---------------------------------------------------------
final_test_pred = np.dot(test_matrix, optimal_weights)
submission_df = pd.DataFrame({"id": test_ids, "y": final_test_pred})
submission_df.to_csv("./submission/submission.csv", index=False)

print(f"Final Validation Score: {final_val_score}")
