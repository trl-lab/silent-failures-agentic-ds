import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import lightgbm as lgb
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

# Ensure output and working directories exist
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# ==============================================================================
# 1. DATA PREPROCESSING & FEATURE ENGINEERING
# ==============================================================================
print("Loading raw datasets from ./input...")
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "class"
feature_cols = [c for c in train_df.columns if c != target_col]

if "id" in test_df.columns:
    test_ids = test_df["id"].values
    test_features_cols = [c for c in test_df.columns if c not in ["id", target_col]]
else:
    test_ids = np.arange(len(test_df))
    test_features_cols = [c for c in test_df.columns if c != target_col]


def clean_board_values(df, cols):
    data = df[cols].copy()
    mapping = {"b": 0, "x": 1, "o": 2, "0": 0, "1": 1, "2": 2, 0: 0, 1: 1, 2: 2}
    for c in cols:
        data[c] = data[c].map(lambda v: mapping.get(v, v)).astype(np.int8)
    return data.values


X_raw_all = clean_board_values(train_df, feature_cols)
y_all = train_df[target_col].values.astype(np.int64)
X_raw_test = clean_board_values(test_df, test_features_cols)


def get_winning_lines():
    lines = []
    # Horizontal lines (6 rows x 4 start cols)
    for r in range(6):
        for c in range(4):
            lines.append([r + (c + i) * 6 for i in range(4)])
    # Vertical lines (7 cols x 3 start rows)
    for c in range(7):
        for r in range(3):
            lines.append([(r + i) + c * 6 for i in range(4)])
    # Positive diagonal (bottom-left to top-right)
    for r in range(3):
        for c in range(4):
            lines.append([(r + i) + (c + i) * 6 for i in range(4)])
    # Negative diagonal (top-left to bottom-right)
    for r in range(3, 6):
        for c in range(4):
            lines.append([(r - i) + (c + i) * 6 for i in range(4)])
    return np.array(lines)


WINNING_LINES = get_winning_lines()  # Shape: (69, 4)


def extract_connect4_features(board_array):
    N = len(board_array)

    # Reshape to (N, 6, 7) - (rows, cols)
    grid = board_array.reshape(N, 7, 6).transpose(0, 2, 1)

    # Piece counts
    p1_count = (board_array == 1).sum(axis=1, keepdims=True)
    p2_count = (board_array == 2).sum(axis=1, keepdims=True)
    empty_count = (board_array == 0).sum(axis=1, keepdims=True)
    piece_diff = p1_count - p2_count

    # Column heights
    col_occupied = grid != 0
    col_heights = col_occupied.sum(axis=1)

    col_diffs = np.diff(col_heights, axis=1)
    max_height = col_heights.max(axis=1, keepdims=True)
    min_height = col_heights.min(axis=1, keepdims=True)
    height_std = col_heights.std(axis=1, keepdims=True)

    # Center dominance
    center_p1 = (grid[:, :, 3] == 1).sum(axis=1, keepdims=True) + 0.5 * (
        (grid[:, :, 2] == 1) | (grid[:, :, 4] == 1)
    ).sum(axis=1, keepdims=True)
    center_p2 = (grid[:, :, 3] == 2).sum(axis=1, keepdims=True) + 0.5 * (
        (grid[:, :, 2] == 2) | (grid[:, :, 4] == 2)
    ).sum(axis=1, keepdims=True)

    # Row Parity
    odd_rows_p1 = (grid[:, 0::2, :] == 1).sum(axis=(1, 2)).reshape(-1, 1)
    even_rows_p1 = (grid[:, 1::2, :] == 1).sum(axis=(1, 2)).reshape(-1, 1)
    odd_rows_p2 = (grid[:, 0::2, :] == 2).sum(axis=(1, 2)).reshape(-1, 1)
    even_rows_p2 = (grid[:, 1::2, :] == 2).sum(axis=(1, 2)).reshape(-1, 1)

    # Tactical 69 Line Analysis
    line_cells = board_array[:, WINNING_LINES]
    p1_line_counts = (line_cells == 1).sum(axis=2)
    p2_line_counts = (line_cells == 2).sum(axis=2)
    empty_line_counts = (line_cells == 0).sum(axis=2)

    p1_3inrow = ((p1_line_counts == 3) & (empty_line_counts == 1)).sum(
        axis=1, keepdims=True
    )
    p2_3inrow = ((p2_line_counts == 3) & (empty_line_counts == 1)).sum(
        axis=1, keepdims=True
    )
    p1_2inrow = ((p1_line_counts == 2) & (empty_line_counts == 2)).sum(
        axis=1, keepdims=True
    )
    p2_2inrow = ((p2_line_counts == 2) & (empty_line_counts == 2)).sum(
        axis=1, keepdims=True
    )
    blocked_lines = ((p1_line_counts > 0) & (p2_line_counts > 0)).sum(
        axis=1, keepdims=True
    )
    open_p1_lines = ((p1_line_counts > 0) & (p2_line_counts == 0)).sum(
        axis=1, keepdims=True
    )
    open_p2_lines = ((p2_line_counts > 0) & (p1_line_counts == 0)).sum(
        axis=1, keepdims=True
    )

    engineered = np.hstack(
        [
            board_array,
            p1_count,
            p2_count,
            empty_count,
            piece_diff,
            col_heights,
            col_diffs,
            max_height,
            min_height,
            height_std,
            center_p1,
            center_p2,
            odd_rows_p1,
            even_rows_p1,
            odd_rows_p2,
            even_rows_p2,
            p1_3inrow,
            p2_3inrow,
            p1_2inrow,
            p2_2inrow,
            blocked_lines,
            open_p1_lines,
            open_p2_lines,
        ]
    )

    return engineered.astype(np.float32)


def extract_spatial_grid(board_array):
    N = len(board_array)
    grid = board_array.reshape(N, 7, 6).transpose(0, 2, 1)
    spatial = np.zeros((N, 3, 6, 7), dtype=np.float32)
    spatial[:, 0, :, :] = (grid == 0).astype(np.float32)
    spatial[:, 1, :, :] = (grid == 1).astype(np.float32)
    spatial[:, 2, :, :] = (grid == 2).astype(np.float32)
    return spatial


print("Splitting data and extracting spatial and tabular features...")
X_train_raw, X_val_raw, y_train, y_val = train_test_split(
    X_raw_all, y_all, test_size=0.20, random_state=42, stratify=y_all
)

X_train_feat = extract_connect4_features(X_train_raw)
X_val_feat = extract_connect4_features(X_val_raw)
X_test_feat = extract_connect4_features(X_raw_test)

S_train = extract_spatial_grid(X_train_raw)
S_val = extract_spatial_grid(X_val_raw)
S_test = extract_spatial_grid(X_raw_test)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_feat)
X_val_scaled = scaler.transform(X_val_feat)
X_test_scaled = scaler.transform(X_test_feat)

tabular_dim = X_train_scaled.shape[1]
num_classes = len(np.unique(y_train))


# ==============================================================================
# 2. MODEL ARCHITECTURE DEFINITIONS
# ==============================================================================
class SpatialSymmetryConvBlock(nn.Module):
    def __init__(self, in_channels=3, out_channels=64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward_single(self, x):
        res = self.residual(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)) + res)
        return out

    def forward(self, x):
        feat_orig = self.forward_single(x)
        x_flipped = torch.flip(x, dims=[3])
        feat_flipped = self.forward_single(x_flipped)
        feat_flipped_back = torch.flip(feat_flipped, dims=[3])
        return 0.5 * (feat_orig + feat_flipped_back)


class Connect4SymmetryNet(nn.Module):
    def __init__(self, tabular_dim=65, num_classes=3, dropout_rate=0.25):
        super().__init__()
        self.spatial_conv1 = SpatialSymmetryConvBlock(in_channels=3, out_channels=48)
        self.spatial_conv2 = SpatialSymmetryConvBlock(in_channels=48, out_channels=96)
        self.spatial_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(96 * 6 * 7, 128),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
        )

        self.tab_fc1 = nn.Linear(tabular_dim, 256)
        self.tab_bn1 = nn.BatchNorm1d(256)
        self.tab_fc2 = nn.Linear(256, 128)
        self.tab_bn2 = nn.BatchNorm1d(128)
        self.tab_dropout = nn.Dropout(dropout_rate)

        fusion_dim = 128 + 128
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Dropout(dropout_rate / 2.0),
            nn.Linear(64, num_classes),
        )

    def forward(self, x_tab, s_grid):
        spatial_out = self.spatial_conv1(s_grid)
        spatial_out = self.spatial_conv2(spatial_out)
        spatial_feat = self.spatial_fc(spatial_out)

        tab_h = F.silu(self.tab_bn1(self.tab_fc1(x_tab)))
        tab_h = self.tab_dropout(tab_h)
        tab_feat = F.silu(self.tab_bn2(self.tab_fc2(tab_h)))

        fused = torch.cat([spatial_feat, tab_feat], dim=1)
        return self.fusion_head(fused)


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.05):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits, target):
        log_preds = F.log_softmax(logits, dim=-1)
        loss_ce = -log_preds.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
        smooth_loss = -log_preds.mean(dim=-1)
        return ((1.0 - self.smoothing) * loss_ce + self.smoothing * smooth_loss).mean()


# ==============================================================================
# 3. MODEL TRAINING & EVALUATION
# ==============================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset = TensorDataset(
    torch.tensor(X_train_scaled, dtype=torch.float32),
    torch.tensor(S_train, dtype=torch.float32),
    torch.tensor(y_train, dtype=torch.long),
)
val_dataset = TensorDataset(
    torch.tensor(X_val_scaled, dtype=torch.float32),
    torch.tensor(S_val, dtype=torch.float32),
    torch.tensor(y_val, dtype=torch.long),
)
test_dataset = TensorDataset(
    torch.tensor(X_test_scaled, dtype=torch.float32),
    torch.tensor(S_test, dtype=torch.float32),
)

train_loader = DataLoader(
    train_dataset, batch_size=256, shuffle=True, num_workers=2, pin_memory=True
)
val_loader = DataLoader(
    val_dataset, batch_size=512, shuffle=False, num_workers=2, pin_memory=True
)
test_loader = DataLoader(
    test_dataset, batch_size=512, shuffle=False, num_workers=2, pin_memory=True
)

nn_model = Connect4SymmetryNet(tabular_dim=tabular_dim, num_classes=num_classes).to(
    device
)
criterion = LabelSmoothingCrossEntropy(smoothing=0.05)
optimizer = torch.optim.AdamW(nn_model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=35, eta_min=1e-5
)

best_val_acc = 0.0
best_nn_path = "./working/best_nn_model.pt"
num_epochs = 35

print(f"Training Connect4SymmetryNet on device: {device}...")
for epoch in range(1, num_epochs + 1):
    nn_model.train()
    running_loss = 0.0
    for bx_tab, bs_grid, by in train_loader:
        bx_tab, bs_grid, by = bx_tab.to(device), bs_grid.to(device), by.to(device)

        optimizer.zero_grad()
        logits = nn_model(bx_tab, bs_grid)
        loss = criterion(logits, by)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * len(by)

    scheduler.step()
    epoch_loss = running_loss / len(train_dataset)

    nn_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for bx_tab, bs_grid, by in val_loader:
            bx_tab, bs_grid, by = bx_tab.to(device), bs_grid.to(device), by.to(device)
            logits = nn_model(bx_tab, bs_grid)
            preds = logits.argmax(dim=1)
            correct += (preds == by).sum().item()
            total += len(by)

    val_acc = correct / total
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(nn_model.state_dict(), best_nn_path)

    print(
        f"Epoch [{epoch:02d}/{num_epochs:02d}] - Loss: {epoch_loss:.4f} - Val Acc: {val_acc:.4f}"
    )

nn_model.load_state_dict(torch.load(best_nn_path))
nn_model.eval()


def get_nn_probs(loader):
    all_probs = []
    with torch.no_grad():
        for batch in loader:
            bx_tab = batch[0].to(device)
            bs_grid = batch[1].to(device)
            logits = nn_model(bx_tab, bs_grid)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
    return np.vstack(all_probs)


val_probs_nn = get_nn_probs(val_loader)
test_probs_nn = get_nn_probs(test_loader)

print("Training tree-based models (LightGBM, XGBoost, ExtraTrees)...")
lgb_model = LGBMClassifier(
    n_estimators=600,
    learning_rate=0.03,
    num_leaves=63,
    max_depth=8,
    subsample=0.85,
    colsample_bytree=0.80,
    min_child_samples=20,
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)
lgb_model.fit(
    X_train_scaled,
    y_train,
    eval_set=[(X_val_scaled, y_val)],
    callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
)
val_probs_lgb = lgb_model.predict_proba(X_val_scaled)
test_probs_lgb = lgb_model.predict_proba(X_test_scaled)

xgb_model = XGBClassifier(
    n_estimators=600,
    learning_rate=0.03,
    max_depth=7,
    subsample=0.85,
    colsample_bytree=0.80,
    tree_method="hist",
    random_state=42,
    n_jobs=-1,
    eval_metric="mlogloss",
    early_stopping_rounds=50,
)
xgb_model.fit(X_train_scaled, y_train, eval_set=[(X_val_scaled, y_val)], verbose=False)
val_probs_xgb = xgb_model.predict_proba(X_val_scaled)
test_probs_xgb = xgb_model.predict_proba(X_test_scaled)

et_model = ExtraTreesClassifier(
    n_estimators=400,
    max_depth=18,
    min_samples_split=4,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
)
et_model.fit(X_train_scaled, y_train)
val_probs_et = et_model.predict_proba(X_val_scaled)
test_probs_et = et_model.predict_proba(X_test_scaled)

print("Optimizing ensemble blending weights...")


def ensemble_obj(weights):
    w = np.maximum(weights, 0)
    w_sum = w.sum()
    if w_sum > 0:
        w = w / w_sum
    else:
        w = np.ones_like(w) / len(w)

    fused_val = (
        w[0] * val_probs_nn
        + w[1] * val_probs_lgb
        + w[2] * val_probs_xgb
        + w[3] * val_probs_et
    )
    preds = np.argmax(fused_val, axis=1)
    return -accuracy_score(y_val, preds)


init_weights = [0.25, 0.25, 0.25, 0.25]
res = minimize(
    ensemble_obj,
    init_weights,
    method="Nelder-Mead",
    options={"maxiter": 500, "disp": False},
)
opt_w = np.maximum(res.x, 0)
opt_w = opt_w / opt_w.sum()

val_probs_blend = (
    opt_w[0] * val_probs_nn
    + opt_w[1] * val_probs_lgb
    + opt_w[2] * val_probs_xgb
    + opt_w[3] * val_probs_et
)
val_preds = np.argmax(val_probs_blend, axis=1)
final_val_accuracy = accuracy_score(y_val, val_preds)

test_probs_blend = (
    opt_w[0] * test_probs_nn
    + opt_w[1] * test_probs_lgb
    + opt_w[2] * test_probs_xgb
    + opt_w[3] * test_probs_et
)
test_preds = np.argmax(test_probs_blend, axis=1)

sub_df = pd.DataFrame({"id": test_ids, "class": test_preds})
sub_df.to_csv("./submission/submission.csv", index=False)

print(f"Final Validation Score: {final_val_accuracy:.6f}")
