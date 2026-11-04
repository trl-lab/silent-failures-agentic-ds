import json
import os
import re
from typing import Any, Dict, Optional, Tuple
from catboost import CatBoostClassifier
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb

# =========================================================================
# 0. Global Setup and Reproducibility
# =========================================================================
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# =========================================================================
# 1. Feature Engineering on Circular Ultrasound Sensors
# =========================================================================
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

sensor_cols = [col for col in train_df.columns if col.upper().startswith("US")]
if len(sensor_cols) != 24:
    non_feature_cols = {"class", "target", "id", "ID", "label"}
    sensor_cols = [c for c in train_df.columns if c not in non_feature_cols]


def extract_sensor_idx(col_name: str) -> int:
    match = re.search(r"\d+", col_name)
    return int(match.group()) if match else 0


sensor_cols = sorted(sensor_cols, key=extract_sensor_idx)

# Physical sensor angles in degrees for US1 through US24
sensor_angles_deg = np.array(
    [
        180.0,
        -165.0,
        -150.0,
        -135.0,
        -120.0,
        -105.0,
        -90.0,
        -75.0,
        -60.0,
        -45.0,
        -30.0,
        -15.0,
        0.0,
        15.0,
        30.0,
        45.0,
        60.0,
        75.0,
        90.0,
        105.0,
        120.0,
        135.0,
        150.0,
        165.0,
    ]
)
sensor_angles_rad = np.radians(sensor_angles_deg)
cos_angles = np.cos(sensor_angles_rad)
sin_angles = np.sin(sensor_angles_rad)


def engineer_robot_features(df: pd.DataFrame, sensor_columns: list) -> pd.DataFrame:
    """Extract geometrically grounded spatial and dynamic clearance features."""
    feats = pd.DataFrame(index=df.index)
    X_sensors = df[sensor_columns].to_numpy(dtype=np.float64)
    eps = 1e-4

    # 1. Base sensor readings
    for i, col in enumerate(sensor_columns):
        feats[col] = X_sensors[:, i]

    # 2. Cartesian 2D Obstacle Projection & Repulsion Physics
    X_cart = X_sensors * cos_angles[np.newaxis, :]
    Y_cart = X_sensors * sin_angles[np.newaxis, :]

    feats["obs_mean_x"] = np.mean(X_cart, axis=1)
    feats["obs_mean_y"] = np.mean(Y_cart, axis=1)
    feats["obs_std_x"] = np.std(X_cart, axis=1)
    feats["obs_std_y"] = np.std(Y_cart, axis=1)
    feats["obs_cov_xy"] = (
        np.mean(X_cart * Y_cart, axis=1) - feats["obs_mean_x"] * feats["obs_mean_y"]
    )

    weights = 1.0 / (X_sensors + eps)
    sum_w = np.sum(weights, axis=1, keepdims=True)
    repulse_x = (
        np.sum(weights * cos_angles[np.newaxis, :], axis=1, keepdims=True) / sum_w
    )
    repulse_y = (
        np.sum(weights * sin_angles[np.newaxis, :], axis=1, keepdims=True) / sum_w
    )

    feats["repulse_x"] = repulse_x.squeeze(axis=1)
    feats["repulse_y"] = repulse_y.squeeze(axis=1)
    feats["repulse_mag"] = np.sqrt(feats["repulse_x"] ** 2 + feats["repulse_y"] ** 2)
    feats["repulse_angle"] = np.arctan2(feats["repulse_y"], feats["repulse_x"])

    # 3. Eight Navigation Sectors
    sector_indices = {
        "front": [0, 1, 23],
        "front_left": [2, 3, 4],
        "left": [5, 6, 7],
        "rear_left": [8, 9, 10],
        "rear": [11, 12, 13],
        "rear_right": [14, 15, 16],
        "right": [17, 18, 19],
        "front_right": [20, 21, 22],
    }

    sector_stats = {}
    for name, idxs in sector_indices.items():
        sub_mat = X_sensors[:, idxs]
        sec_min = np.min(sub_mat, axis=1)
        sec_mean = np.mean(sub_mat, axis=1)
        sec_std = np.std(sub_mat, axis=1)
        feats[f"sec_{name}_min"] = sec_min
        feats[f"sec_{name}_mean"] = sec_mean
        feats[f"sec_{name}_std"] = sec_std
        sector_stats[name] = (sec_min, sec_mean)

    # 4. Clearance Differential & Asymmetry Ratios
    feats["diff_right_left_min"] = sector_stats["right"][0] - sector_stats["left"][0]
    feats["diff_right_left_mean"] = sector_stats["right"][1] - sector_stats["left"][1]
    feats["diff_front_right_left_min"] = (
        sector_stats["front_right"][0] - sector_stats["front_left"][0]
    )
    feats["diff_front_rear_min"] = sector_stats["front"][0] - sector_stats["rear"][0]
    feats["diff_front_right_min"] = sector_stats["front"][0] - sector_stats["right"][0]

    feats["ratio_right_left"] = sector_stats["right"][0] / (
        sector_stats["left"][0] + eps
    )
    feats["ratio_front_right"] = sector_stats["front"][0] / (
        sector_stats["right"][0] + eps
    )
    feats["ratio_front_left"] = sector_stats["front"][0] / (
        sector_stats["left"][0] + eps
    )
    feats["ratio_front_rear"] = sector_stats["front"][0] / (
        sector_stats["rear"][0] + eps
    )

    # 5. Bilateral Symmetric Pair Differences
    bilateral_diffs = []
    for k in range(11):
        left_idx = 1 + k
        right_idx = 23 - k
        diff = X_sensors[:, right_idx] - X_sensors[:, left_idx]
        feats[f"pair_diff_{k+1}"] = diff
        bilateral_diffs.append(diff)

    bilateral_stack = np.column_stack(bilateral_diffs)
    feats["total_signed_lateral_imbalance"] = np.sum(bilateral_stack, axis=1)
    feats["total_abs_lateral_imbalance"] = np.sum(np.abs(bilateral_stack), axis=1)

    # 6. Circular Derivatives
    X_sensors_rolled_forward = np.roll(X_sensors, shift=-1, axis=1)
    X_sensors_rolled_backward = np.roll(X_sensors, shift=1, axis=1)

    circ_grad = X_sensors_rolled_forward - X_sensors
    circ_curvature = (
        X_sensors_rolled_forward - 2.0 * X_sensors + X_sensors_rolled_backward
    )

    feats["circ_grad_abs_sum"] = np.sum(np.abs(circ_grad), axis=1)
    feats["circ_grad_max"] = np.max(circ_grad, axis=1)
    feats["circ_grad_min"] = np.min(circ_grad, axis=1)
    feats["circ_curv_abs_sum"] = np.sum(np.abs(circ_curvature), axis=1)

    # 7. Diametric Corridor Cross-Span Features (US[i] + US[i+12])
    diametric_spans = []
    for k in range(12):
        d_span = X_sensors[:, k] + X_sensors[:, k + 12]
        d_diff = X_sensors[:, k] - X_sensors[:, k + 12]
        feats[f"diametric_span_{k+1}"] = d_span
        feats[f"diametric_diff_{k+1}"] = d_diff
        diametric_spans.append(d_span)

    diametric_mat = np.column_stack(diametric_spans)
    feats["diametric_span_min"] = np.min(diametric_mat, axis=1)
    feats["diametric_span_max"] = np.max(diametric_mat, axis=1)
    feats["diametric_span_mean"] = np.mean(diametric_mat, axis=1)
    feats["diametric_span_std"] = np.std(diametric_mat, axis=1)

    # 8. Multi-scale Directional Sector Statistics (30/60/90 degree cones)
    cones = {
        "cone_front_30": [11, 12, 13],
        "cone_front_60": [10, 11, 12, 13, 14],
        "cone_front_90": [9, 10, 11, 12, 13, 14, 15],
        "cone_right_30": [17, 18, 19],
        "cone_right_60": [16, 17, 18, 19, 20],
        "cone_right_90": [15, 16, 17, 18, 19, 20, 21],
        "cone_left_30": [5, 6, 7],
        "cone_left_60": [4, 5, 6, 7, 8],
        "cone_left_90": [3, 4, 5, 6, 7, 8, 9],
        "cone_rear_30": [23, 0, 1],
        "cone_rear_60": [22, 23, 0, 1, 2],
        "cone_rear_90": [21, 22, 23, 0, 1, 2, 3],
    }
    for cname, cindices in cones.items():
        sub_c = X_sensors[:, cindices]
        feats[f"{cname}_min"] = np.min(sub_c, axis=1)
        feats[f"{cname}_mean"] = np.mean(sub_c, axis=1)
        feats[f"{cname}_std"] = np.std(sub_c, axis=1)

    # 9. Flank Wall Slope Derivatives & Alignment
    r_flank = X_sensors[:, 12:20]
    feats["right_flank_slope"] = (X_sensors[:, 18] - X_sensors[:, 12]) / 6.0
    feats["right_flank_diff_mean"] = np.mean(np.diff(r_flank, axis=1), axis=1)
    feats["right_flank_diff_std"] = np.std(np.diff(r_flank, axis=1), axis=1)

    l_flank = X_sensors[:, 6:14]
    feats["left_flank_slope"] = (X_sensors[:, 12] - X_sensors[:, 6]) / 6.0
    feats["left_flank_diff_mean"] = np.mean(np.diff(l_flank, axis=1), axis=1)
    feats["left_flank_diff_std"] = np.std(np.diff(l_flank, axis=1), axis=1)

    # 10. Non-linear Proximity Fields (Log and Inverse-Square)
    inv_prox = 1.0 / (X_sensors + eps)
    log_prox = np.log1p(inv_prox)
    inv_sq_prox = 1.0 / ((X_sensors + eps) ** 2)

    feats["prox_log_sum"] = np.sum(log_prox, axis=1)
    feats["prox_log_max"] = np.max(log_prox, axis=1)
    feats["prox_inv_sq_sum"] = np.sum(inv_sq_prox, axis=1)
    feats["prox_inv_sq_max"] = np.max(inv_sq_prox, axis=1)
    feats["prox_front_log"] = np.mean(log_prox[:, [11, 12, 13]], axis=1)
    feats["prox_right_log"] = np.mean(log_prox[:, [17, 18, 19]], axis=1)
    feats["prox_left_log"] = np.mean(log_prox[:, [5, 6, 7]], axis=1)

    # 11. Global Distribution Properties
    feats["global_min_dist"] = np.min(X_sensors, axis=1)
    feats["global_max_dist"] = np.max(X_sensors, axis=1)
    feats["global_mean_dist"] = np.mean(X_sensors, axis=1)
    feats["global_std_dist"] = np.std(X_sensors, axis=1)
    feats["global_dist_range"] = feats["global_max_dist"] - feats["global_min_dist"]

    min_indices = np.argmin(X_sensors, axis=1)
    closest_angles_rad = sensor_angles_rad[min_indices]
    feats["closest_angle_cos"] = np.cos(closest_angles_rad)
    feats["closest_angle_sin"] = np.sin(closest_angles_rad)
    feats["closest_sensor_idx"] = min_indices

    return feats


X_train_engineered = engineer_robot_features(train_df, sensor_cols)
X_test_engineered = engineer_robot_features(test_df, sensor_cols)

target_col = "class" if "class" in train_df.columns else "target"
y_raw = train_df[target_col].copy()

unique_classes = sorted(list(y_raw.unique()))
class_to_idx = {cls_val: int(i) for i, cls_val in enumerate(unique_classes)}
idx_to_class = {int(i): cls_val for i, cls_val in enumerate(unique_classes)}
y_encoded = y_raw.map(class_to_idx).to_numpy(dtype=np.int64)

# Create 5-Fold Stratified CV
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
fold_assignments = np.zeros(len(train_df), dtype=np.int32)
for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_engineered, y_encoded)):
    fold_assignments[val_idx] = fold

feature_cols = list(X_train_engineered.columns)
raw_sensor_cols = [c for c in sensor_cols if c in feature_cols]
raw_sensor_cols = sorted(raw_sensor_cols, key=extract_sensor_idx)
engineered_cols = [col for col in feature_cols if col not in raw_sensor_cols]

sensor_indices = [feature_cols.index(c) for c in raw_sensor_cols]
eng_indices = [feature_cols.index(c) for c in engineered_cols]

X_all = X_train_engineered.to_numpy(dtype=np.float32)
X_test_mat = X_test_engineered.to_numpy(dtype=np.float32)
num_classes = len(unique_classes)
num_folds = 5
n_test = len(test_df)


# =========================================================================
# 2. Neural Architecture Definitions
# =========================================================================
class ResidualBlock(nn.Module):

    def __init__(self, hidden_dim: int, dropout_rate: float = 0.15):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.act1 = nn.GELU()
        self.drop1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.act2 = nn.GELU()
        self.drop2 = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.drop1(self.act1(self.norm1(self.fc1(x))))
        out = self.drop2(self.act2(self.norm2(self.fc2(out))))
        return out + residual


class RobotNavResNet(nn.Module):

    def __init__(
        self,
        in_features: int,
        num_classes: int = 4,
        hidden_dim: int = 256,
        num_blocks: int = 3,
        dropout_rate: float = 0.15,
    ):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )
        self.blocks = nn.ModuleList(
            [
                ResidualBlock(hidden_dim, dropout_rate=dropout_rate)
                for _ in range(num_blocks)
            ]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h)
        return self.head(h)


class CircularConv1d(nn.Module):

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1
    ):
        super().__init__()
        self.pad = kernel_size // 2
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=0
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pad > 0:
            x = torch.cat([x[..., -self.pad:], x, x[..., :self.pad]], dim=-1)
        return self.conv(x)


class SqueezeExcitation1d(nn.Module):

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, max(channels // reduction, 8)),
            nn.GELU(),
            nn.Linear(max(channels // reduction, 8), channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _ = x.shape
        w = self.fc(x).view(b, c, 1)
        return x * w


class CircularAttentionNet(nn.Module):

    def __init__(
        self,
        num_raw_sensors: int = 24,
        num_engineered_feats: int = 100,
        num_classes: int = 4,
        conv_channels: int = 64,
        dense_hidden: int = 128,
        num_heads: int = 4,
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        self.conv1 = CircularConv1d(1, conv_channels // 2, kernel_size=3)
        self.bn1 = nn.BatchNorm1d(conv_channels // 2)
        self.conv2 = CircularConv1d(conv_channels // 2, conv_channels, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(conv_channels)
        self.se1 = SqueezeExcitation1d(conv_channels)
        self.conv3 = CircularConv1d(conv_channels, conv_channels, kernel_size=3)
        self.bn3 = nn.BatchNorm1d(conv_channels)
        self.se2 = SqueezeExcitation1d(conv_channels)

        self.attn = nn.MultiheadAttention(
            embed_dim=conv_channels,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout_rate,
        )
        self.attn_norm = nn.LayerNorm(conv_channels)
        self.pool = nn.AdaptiveAvgPool1d(1)

        self.dense_stream = nn.Sequential(
            nn.Linear(num_engineered_feats, dense_hidden),
            nn.LayerNorm(dense_hidden),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dense_hidden, dense_hidden),
            nn.LayerNorm(dense_hidden),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )

        fusion_dim = conv_channels + dense_hidden
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, dense_hidden),
            nn.LayerNorm(dense_hidden),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dense_hidden, num_classes),
        )

    def forward(
        self, x_raw_sensors: torch.Tensor, x_engineered: torch.Tensor
    ) -> torch.Tensor:
        x_3d = x_raw_sensors.unsqueeze(1)
        c = F.gelu(self.bn1(self.conv1(x_3d)))
        c = self.se1(F.gelu(self.bn2(self.conv2(c))))
        c = self.se2(F.gelu(self.bn3(self.conv3(c))))

        seq = c.permute(0, 2, 1)
        attn_out, _ = self.attn(seq, seq, seq)
        seq = self.attn_norm(seq + attn_out)
        c_attn = seq.permute(0, 2, 1)

        conv_feats = self.pool(c_attn).squeeze(-1)
        dense_feats = self.dense_stream(x_engineered)
        fused = torch.cat([conv_feats, dense_feats], dim=1)
        return self.classifier(fused)


def get_optimizer_and_scheduler(
    model: nn.Module, lr: float = 1e-3, weight_decay: float = 1e-4, epochs: int = 70
):
    decay_params, no_decay_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name.endswith(".bias") or "norm" in name.lower():
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    optimizer = AdamW(
        [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=lr,
        eps=1e-8,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    return optimizer, scheduler


# =========================================================================
# 3. Model Training & Cross-Validation
# =========================================================================
oof_lgb_probs = np.zeros((len(train_df), num_classes), dtype=np.float64)
test_lgb_probs = np.zeros((n_test, num_classes), dtype=np.float64)

oof_xgb_probs = np.zeros((len(train_df), num_classes), dtype=np.float64)
test_xgb_probs = np.zeros((n_test, num_classes), dtype=np.float64)

oof_cat_probs = np.zeros((len(train_df), num_classes), dtype=np.float64)
test_cat_probs = np.zeros((n_test, num_classes), dtype=np.float64)

oof_resnet_probs = np.zeros((len(train_df), num_classes), dtype=np.float64)
test_resnet_probs = np.zeros((n_test, num_classes), dtype=np.float64)

oof_circ_probs = np.zeros((len(train_df), num_classes), dtype=np.float64)
test_circ_probs = np.zeros((n_test, num_classes), dtype=np.float64)

for fold in range(num_folds):
    train_mask = fold_assignments != fold
    val_mask = fold_assignments == fold

    X_train_raw, y_train = X_all[train_mask], y_encoded[train_mask]
    X_val_raw, y_val = X_all[val_mask], y_encoded[val_mask]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw).astype(np.float32)
    X_val_scaled = scaler.transform(X_val_raw).astype(np.float32)
    X_test_scaled = scaler.transform(X_test_mat).astype(np.float32)

    # 1. LightGBM
    lgb_model = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=num_classes,
        n_estimators=1500,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=-1,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        min_child_samples=15,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=RANDOM_SEED + fold,
        n_jobs=-1,
        verbose=-1,
    )
    lgb_model.fit(
        X_train_raw,
        y_train,
        eval_set=[(X_val_raw, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )
    oof_lgb_probs[val_mask] = lgb_model.predict_proba(X_val_raw)
    test_lgb_probs += lgb_model.predict_proba(X_test_mat) / num_folds

    # 2. XGBoost
    xgb_model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=num_classes,
        n_estimators=1500,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=2,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=RANDOM_SEED + fold,
        n_jobs=-1,
        eval_metric="mlogloss",
        early_stopping_rounds=50,
    )
    xgb_model.fit(X_train_raw, y_train, eval_set=[(X_val_raw, y_val)], verbose=False)
    oof_xgb_probs[val_mask] = xgb_model.predict_proba(X_val_raw)
    test_xgb_probs += xgb_model.predict_proba(X_test_mat) / num_folds

    # 3. CatBoost
    cat_model = CatBoostClassifier(
        iterations=1500,
        learning_rate=0.03,
        depth=6,
        loss_function="MultiClass",
        eval_metric="MultiClass",
        random_seed=RANDOM_SEED + fold,
        verbose=False,
        early_stopping_rounds=50,
        thread_count=-1,
    )
    cat_model.fit(X_train_raw, y_train, eval_set=(X_val_raw, y_val), verbose=False)
    oof_cat_probs[val_mask] = cat_model.predict_proba(X_val_raw)
    test_cat_probs += cat_model.predict_proba(X_test_mat) / num_folds

    # 4. Tabular ResNet
    train_dataset_res = TensorDataset(
        torch.tensor(X_train_scaled), torch.tensor(y_train)
    )
    val_dataset_res = TensorDataset(torch.tensor(X_val_scaled), torch.tensor(y_val))
    test_tensor_res = torch.tensor(X_test_scaled).to(device)

    train_loader_res = DataLoader(
        train_dataset_res, batch_size=64, shuffle=True, drop_last=False
    )
    val_loader_res = DataLoader(
        val_dataset_res, batch_size=128, shuffle=False, drop_last=False
    )

    resnet_model = RobotNavResNet(
        in_features=X_all.shape[1],
        num_classes=num_classes,
        hidden_dim=256,
        num_blocks=3,
        dropout_rate=0.15,
    ).to(device)
    criterion_res = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer_res, scheduler_res = get_optimizer_and_scheduler(
        resnet_model, lr=1e-3, weight_decay=1e-4, epochs=70
    )

    best_res_loss = float("inf")
    best_res_state = None

    for epoch in range(70):
        resnet_model.train()
        for x_b, y_b in train_loader_res:
            x_b, y_b = x_b.to(device), y_b.to(device)
            optimizer_res.zero_grad()
            out = resnet_model(x_b)
            loss = criterion_res(out, y_b)
            loss.backward()
            optimizer_res.step()
        scheduler_res.step()

        resnet_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_b, y_b in val_loader_res:
                x_b, y_b = x_b.to(device), y_b.to(device)
                val_loss += criterion_res(resnet_model(x_b), y_b).item() * len(y_b)
        val_loss /= len(val_dataset_res)
        if val_loss < best_res_loss:
            best_res_loss = val_loss
            best_res_state = {
                k: v.cpu().clone() for k, v in resnet_model.state_dict().items()
            }

    resnet_model.load_state_dict(best_res_state)
    resnet_model.eval()
    with torch.no_grad():
        val_res_t = torch.tensor(X_val_scaled).to(device)
        oof_resnet_probs[val_mask] = (
            torch.softmax(resnet_model(val_res_t), dim=1).cpu().numpy()
        )
        test_resnet_probs += (
            torch.softmax(resnet_model(test_tensor_res), dim=1).cpu().numpy()
            / num_folds
        )

    # 4. Circular Sensor 1D-CNN Net
    X_train_sensors = torch.tensor(X_train_scaled[:, sensor_indices])
    X_train_eng = torch.tensor(X_train_scaled[:, eng_indices])
    X_val_sensors = torch.tensor(X_val_scaled[:, sensor_indices])
    X_val_eng = torch.tensor(X_val_scaled[:, eng_indices])

    train_dataset_circ = TensorDataset(
        X_train_sensors, X_train_eng, torch.tensor(y_train)
    )
    val_dataset_circ = TensorDataset(X_val_sensors, X_val_eng, torch.tensor(y_val))

    train_loader_circ = DataLoader(
        train_dataset_circ, batch_size=64, shuffle=True, drop_last=False
    )
    val_loader_circ = DataLoader(
        val_dataset_circ, batch_size=128, shuffle=False, drop_last=False
    )

    circ_model = CircularAttentionNet(
        num_raw_sensors=len(raw_sensor_cols),
        num_engineered_feats=len(eng_indices),
        num_classes=num_classes,
        conv_channels=64,
        dense_hidden=128,
        num_heads=4,
        dropout_rate=0.2,
    ).to(device)
    criterion_circ = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer_circ, scheduler_circ = get_optimizer_and_scheduler(
        circ_model, lr=1e-3, weight_decay=1e-4, epochs=70
    )

    best_circ_loss = float("inf")
    best_circ_state = None

    for epoch in range(70):
        circ_model.train()
        for x_s, x_e, y_b in train_loader_circ:
            x_s, x_e, y_b = x_s.to(device), x_e.to(device), y_b.to(device)
            optimizer_circ.zero_grad()
            out = circ_model(x_s, x_e)
            loss = criterion_circ(out, y_b)
            loss.backward()
            optimizer_circ.step()
        scheduler_circ.step()

        circ_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_s, x_e, y_b in val_loader_circ:
                x_s, x_e, y_b = x_s.to(device), x_e.to(device), y_b.to(device)
                val_loss += criterion_circ(circ_model(x_s, x_e), y_b).item() * len(y_b)
        val_loss /= len(val_dataset_circ)
        if val_loss < best_circ_loss:
            best_circ_loss = val_loss
            best_circ_state = {
                k: v.cpu().clone() for k, v in circ_model.state_dict().items()
            }

    circ_model.load_state_dict(best_circ_state)
    circ_model.eval()
    with torch.no_grad():
        val_s_t = torch.tensor(X_val_scaled[:, sensor_indices]).to(device)
        val_e_t = torch.tensor(X_val_scaled[:, eng_indices]).to(device)
        oof_circ_probs[val_mask] = (
            torch.softmax(circ_model(val_s_t, val_e_t), dim=1).cpu().numpy()
        )

        test_s_t = torch.tensor(X_test_scaled[:, sensor_indices]).to(device)
        test_e_t = torch.tensor(X_test_scaled[:, eng_indices]).to(device)
        test_circ_probs += (
            torch.softmax(circ_model(test_s_t, test_e_t), dim=1).cpu().numpy()
            / num_folds
        )

# =========================================================================
# 4. Blending, Evaluation, and Submission
# =========================================================================
def ensemble_loss(weights: np.ndarray) -> float:
    w = np.array(weights)
    w = w / (np.sum(w) + 1e-12)
    pred_blend = (
        w[0] * oof_lgb_probs
        + w[1] * oof_xgb_probs
        + w[2] * oof_cat_probs
        + w[3] * oof_resnet_probs
        + w[4] * oof_circ_probs
    )
    pred_blend = np.clip(pred_blend, 1e-15, 1.0 - 1e-15)
    pred_blend = pred_blend / np.sum(pred_blend, axis=1, keepdims=True)
    return log_loss(y_encoded, pred_blend)

init_weights = [0.25, 0.25, 0.20, 0.15, 0.15]
bounds = [(0.0, 1.0) for _ in range(5)]
constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

opt_res = minimize(
    ensemble_loss,
    init_weights,
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
    options={"maxiter": 200, "ftol": 1e-7},
)
opt_weights = opt_res.x / np.sum(opt_res.x)
print(f"Optimized Ensemble Weights (LGB, XGB, CAT, RES, CIRC): {opt_weights.round(4)}")

oof_ensemble_probs = (
    opt_weights[0] * oof_lgb_probs
    + opt_weights[1] * oof_xgb_probs
    + opt_weights[2] * oof_cat_probs
    + opt_weights[3] * oof_resnet_probs
    + opt_weights[4] * oof_circ_probs
)
oof_preds = np.argmax(oof_ensemble_probs, axis=1)
final_accuracy = accuracy_score(y_encoded, oof_preds)

test_ensemble_probs = (
    opt_weights[0] * test_lgb_probs
    + opt_weights[1] * test_xgb_probs
    + opt_weights[2] * test_cat_probs
    + opt_weights[3] * test_resnet_probs
    + opt_weights[4] * test_circ_probs
)
test_pred_indices = np.argmax(test_ensemble_probs, axis=1)
test_pred_classes = [idx_to_class[idx] for idx in test_pred_indices]

submission_df = pd.DataFrame(
    {"id": np.arange(len(test_pred_classes), dtype=int), "class": test_pred_classes}
)
submission_path = "./submission/submission.csv"
submission_df.to_csv(submission_path, index=False)

print(f"Final Validation Score: {final_accuracy}")
