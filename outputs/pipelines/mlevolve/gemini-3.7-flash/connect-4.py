import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------
# 0. Reproducibility Configuration
# ---------------------------------------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)

# ---------------------------------------------------------
# 1. Load Data
# ---------------------------------------------------------
print("Loading datasets from ./input...")
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "class"
if target_col not in train_df.columns:
    possible_targets = [c for c in train_df.columns if c not in test_df.columns]
    target_col = possible_targets[0] if possible_targets else "class"

print(
    f"Train shape: {train_df.shape}, Test shape: {test_df.shape}, Target column: {target_col}"
)

# ---------------------------------------------------------
# 2. Board Layout and Winning Lines Definition (6 rows x 7 cols)
# ---------------------------------------------------------
ROWS = 6
COLS = 7
NUM_SQUARES = 42

non_feature_cols = {target_col, "id", "Id", "ID"}
board_cols = [c for c in train_df.columns if c not in non_feature_cols]

if len(board_cols) == NUM_SQUARES:

    def sort_key(col_name):
        col_name_str = str(col_name)
        if (
            len(col_name_str) >= 2
            and col_name_str[0].isalpha()
            and col_name_str[1:].isdigit()
        ):
            col_idx = ord(col_name_str[0].lower()) - ord("a")
            row_idx = int(col_name_str[1:]) - 1
            return col_idx * ROWS + row_idx
        return board_cols.index(col_name)

    try:
        board_cols = sorted(board_cols, key=sort_key)
    except Exception:
        pass

# Generate all 69 winning lines grouped by 4 discrete orientations
h_lines = []
for r in range(ROWS):
    for c in range(COLS - 3):
        h_lines.append([(r, c + i) for i in range(4)])

v_lines = []
for r in range(ROWS - 3):
    for c in range(COLS):
        v_lines.append([(r + i, c) for i in range(4)])

diag1_lines = []
for r in range(ROWS - 3):
    for c in range(COLS - 3):
        diag1_lines.append([(r + i, c + i) for i in range(4)])

diag2_lines = []
for r in range(3, ROWS):
    for c in range(COLS - 3):
        diag2_lines.append([(r - i, c + i) for i in range(4)])

winning_lines = h_lines + v_lines + diag1_lines + diag2_lines
directional_line_groups = {
    "h": h_lines,
    "v": v_lines,
    "d1": diag1_lines,
    "d2": diag2_lines,
}
assert len(winning_lines) == 69, f"Expected 69 lines, found {len(winning_lines)}"


# ---------------------------------------------------------
# 3. Standardize Board Encoding to Numeric Matrices
# ---------------------------------------------------------
def parse_board_values(df: pd.DataFrame, feature_cols: list) -> np.ndarray:
    board_mat = np.zeros((len(df), ROWS, COLS), dtype=np.int8)
    raw_vals = df[feature_cols].values

    sample_val = raw_vals[0, 0]
    val_map = {}
    if isinstance(sample_val, str):
        for val in np.unique(raw_vals):
            v_str = str(val).strip().lower()
            if v_str in ("x", "1", "player1", "p1"):
                val_map[val] = 1
            elif v_str in ("o", "2", "player2", "p2", "-1"):
                val_map[val] = 2
            else:
                val_map[val] = 0
    else:
        for val in np.unique(raw_vals):
            if val in (1, 1.0):
                val_map[val] = 1
            elif val in (2, 2.0, -1, -1.0):
                val_map[val] = 2
            else:
                val_map[val] = 0

    mapped_raw = np.vectorize(lambda x: val_map.get(x, 0))(raw_vals)

    for idx, col_name in enumerate(feature_cols):
        col_name_str = str(col_name)
        if (
            len(col_name_str) >= 2
            and col_name_str[0].isalpha()
            and col_name_str[1:].isdigit()
        ):
            c = ord(col_name_str[0].lower()) - ord("a")
            r = int(col_name_str[1:]) - 1
            if 0 <= r < ROWS and 0 <= c < COLS:
                board_mat[:, r, c] = mapped_raw[:, idx]
            else:
                r_idx = idx % ROWS
                c_idx = idx // ROWS
                board_mat[:, r_idx, c_idx] = mapped_raw[:, idx]
        else:
            r_idx = idx % ROWS
            c_idx = idx // ROWS
            if c_idx < COLS and r_idx < ROWS:
                board_mat[:, r_idx, c_idx] = mapped_raw[:, idx]

    return board_mat


print("Converting board positions to 2D geometric matrices...")
train_boards = parse_board_values(train_df, board_cols)
test_boards = parse_board_values(test_df, board_cols)


# ---------------------------------------------------------
# 4. Domain Feature Engineering
# ---------------------------------------------------------
def extract_connect4_features(boards: np.ndarray) -> np.ndarray:
    n_samples = boards.shape[0]
    feats = []

    p1_grid = (boards == 1).astype(np.float32)
    p2_grid = (boards == 2).astype(np.float32)

    feats.append(p1_grid.reshape(n_samples, -1))
    feats.append(p2_grid.reshape(n_samples, -1))

    col_heights = np.sum(boards != 0, axis=1).astype(np.float32)
    feats.append(col_heights)
    feats.append(np.max(col_heights, axis=1, keepdims=True))
    feats.append(np.min(col_heights, axis=1, keepdims=True))
    feats.append(np.std(col_heights, axis=1, keepdims=True))
    feats.append(np.sum(col_heights, axis=1, keepdims=True))

    top_pieces_p1 = np.zeros((n_samples, COLS), dtype=np.float32)
    top_pieces_p2 = np.zeros((n_samples, COLS), dtype=np.float32)
    bottom_pieces_p1 = (boards[:, 0, :] == 1).astype(np.float32)
    bottom_pieces_p2 = (boards[:, 0, :] == 2).astype(np.float32)

    for c in range(COLS):
        h = col_heights[:, c].astype(int)
        valid_mask = h > 0
        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) > 0:
            top_rows = h[valid_indices] - 1
            top_vals = boards[valid_indices, top_rows, c]
            top_pieces_p1[valid_indices, c] = (top_vals == 1).astype(np.float32)
            top_pieces_p2[valid_indices, c] = (top_vals == 2).astype(np.float32)

    feats.append(top_pieces_p1)
    feats.append(top_pieces_p2)
    feats.append(bottom_pieces_p1)
    feats.append(bottom_pieces_p2)

    p1_line_counts = np.zeros((n_samples, len(winning_lines)), dtype=np.float32)
    p2_line_counts = np.zeros((n_samples, len(winning_lines)), dtype=np.float32)

    for line_idx, line in enumerate(winning_lines):
        r_coords = [sq[0] for sq in line]
        c_coords = [sq[1] for sq in line]

        p1_line_counts[:, line_idx] = np.sum(p1_grid[:, r_coords, c_coords], axis=1)
        p2_line_counts[:, line_idx] = np.sum(p2_grid[:, r_coords, c_coords], axis=1)

    p1_active_lines = (p1_line_counts > 0) & (p2_line_counts == 0)
    p2_active_lines = (p2_line_counts > 0) & (p1_line_counts == 0)

    p1_threats_3 = np.sum(
        p1_active_lines & (p1_line_counts == 3), axis=1, keepdims=True
    ).astype(np.float32)
    p2_threats_3 = np.sum(
        p2_active_lines & (p2_line_counts == 3), axis=1, keepdims=True
    ).astype(np.float32)
    p1_threats_2 = np.sum(
        p1_active_lines & (p1_line_counts == 2), axis=1, keepdims=True
    ).astype(np.float32)
    p2_threats_2 = np.sum(
        p2_active_lines & (p2_line_counts == 2), axis=1, keepdims=True
    ).astype(np.float32)
    p1_open_1 = np.sum(
        p1_active_lines & (p1_line_counts == 1), axis=1, keepdims=True
    ).astype(np.float32)
    p2_open_1 = np.sum(
        p2_active_lines & (p2_line_counts == 1), axis=1, keepdims=True
    ).astype(np.float32)
    p1_total_open = np.sum(p1_active_lines, axis=1, keepdims=True).astype(np.float32)
    p2_total_open = np.sum(p2_active_lines, axis=1, keepdims=True).astype(np.float32)

    feats.extend(
        [
            p1_threats_3,
            p2_threats_3,
            p1_threats_2,
            p2_threats_2,
            p1_open_1,
            p2_open_1,
            p1_total_open,
            p2_total_open,
            p1_threats_3 - p2_threats_3,
            p1_threats_2 - p2_threats_2,
            p1_total_open - p2_total_open,
        ]
    )

    center_weights = np.array(
        [1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0], dtype=np.float32
    ).reshape(1, 1, COLS)
    p1_center_score = np.sum(p1_grid * center_weights, axis=(1, 2), keepdims=True)
    p2_center_score = np.sum(p2_grid * center_weights, axis=(1, 2), keepdims=True)

    feats.append(p1_center_score)
    feats.append(p2_center_score)
    feats.append(p1_center_score - p2_center_score)

    p1_center_col = np.sum(p1_grid[:, :, 3], axis=1, keepdims=True)
    p2_center_col = np.sum(p2_grid[:, :, 3], axis=1, keepdims=True)
    feats.append(p1_center_col)
    feats.append(p2_center_col)
    feats.append(p1_center_col - p2_center_col)

    odd_rows_mask = np.array([0, 1, 0, 1, 0, 1], dtype=np.float32).reshape(1, ROWS, 1)
    even_rows_mask = np.array([1, 0, 1, 0, 1, 0], dtype=np.float32).reshape(1, ROWS, 1)

    p1_odd_row_pieces = np.sum(p1_grid * odd_rows_mask, axis=(1, 2), keepdims=True)
    p1_even_row_pieces = np.sum(p1_grid * even_rows_mask, axis=(1, 2), keepdims=True)
    p2_odd_row_pieces = np.sum(p2_grid * odd_rows_mask, axis=(1, 2), keepdims=True)
    p2_even_row_pieces = np.sum(p2_grid * even_rows_mask, axis=(1, 2), keepdims=True)

    feats.extend(
        [
            p1_odd_row_pieces,
            p1_even_row_pieces,
            p2_odd_row_pieces,
            p2_even_row_pieces,
            p1_odd_row_pieces - p1_even_row_pieces,
            p2_odd_row_pieces - p2_even_row_pieces,
        ]
    )

    left_p1 = p1_grid[:, :, :3]
    right_p1_flipped = p1_grid[:, :, 6:3:-1]
    left_p2 = p2_grid[:, :, :3]
    right_p2_flipped = p2_grid[:, :, 6:3:-1]

    p1_asymmetry = np.sum(
        np.abs(left_p1 - right_p1_flipped), axis=(1, 2), keepdims=True
    )
    p2_asymmetry = np.sum(
        np.abs(left_p2 - right_p2_flipped), axis=(1, 2), keepdims=True
    )
    feats.append(p1_asymmetry)
    feats.append(p2_asymmetry)

    flat_feats = [f.reshape(n_samples, -1) for f in feats]
    return np.hstack(flat_feats).astype(np.float32)


print("Extracting domain-specific features for Train and Test sets...")
X_train_raw = extract_connect4_features(train_boards)
X_test_raw = extract_connect4_features(test_boards)


def create_spatial_tensors(boards: np.ndarray) -> np.ndarray:
    n = boards.shape[0]
    tensors = np.zeros((n, 4, ROWS, COLS), dtype=np.float32)
    tensors[:, 0] = (boards == 1).astype(np.float32)
    tensors[:, 1] = (boards == 2).astype(np.float32)
    tensors[:, 2] = (boards == 0).astype(np.float32)

    col_h = np.sum(boards != 0, axis=1)
    for i in range(n):
        for c in range(COLS):
            h = col_h[i, c]
            if 0 < h <= ROWS:
                tensors[i, 3, h - 1, c] = 1.0
    return tensors


spatial_train = create_spatial_tensors(train_boards)
spatial_test = create_spatial_tensors(test_boards)

# ---------------------------------------------------------
# 5. Target Encoding & Leak-Free Train/Validation Split
# ---------------------------------------------------------
target_encoder = LabelEncoder()
y_train = target_encoder.fit_transform(train_df[target_col].values)
target_classes = target_encoder.classes_

train_idx, val_idx = train_test_split(
    np.arange(len(y_train)), test_size=0.20, random_state=42, stratify=y_train
)

# Stateful scaling fitted strictly on training partition
scaler = StandardScaler()
scaler.fit(X_train_raw[train_idx])

X_tr = scaler.transform(X_train_raw[train_idx])
X_val = scaler.transform(X_train_raw[val_idx])
X_test = scaler.transform(X_test_raw)

y_tr = y_train[train_idx]
y_val = y_train[val_idx]

spatial_tr = spatial_train[train_idx]
spatial_val = spatial_train[val_idx]


# ---------------------------------------------------------
# 6. Neural Network Architecture Definition
# ---------------------------------------------------------
class DirectionalConvBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        branch_channels = out_channels // 4
        self.conv_h = nn.Conv2d(
            in_channels, branch_channels, kernel_size=(1, 4), padding=(0, 2)
        )
        self.conv_v = nn.Conv2d(
            in_channels, branch_channels, kernel_size=(4, 1), padding=(2, 0)
        )
        self.conv_3x3 = nn.Conv2d(
            in_channels, branch_channels, kernel_size=3, padding=1
        )
        self.conv_1x1 = nn.Conv2d(in_channels, branch_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.Mish()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_feat = self.conv_h(x)[:, :, :6, :7]
        v_feat = self.conv_v(x)[:, :, :6, :7]
        c_feat = self.conv_3x3(x)
        p_feat = self.conv_1x1(x)
        out = torch.cat([h_feat, v_feat, c_feat, p_feat], dim=1)
        return self.act(self.bn(out))


class SpatialResNetBlock(nn.Module):

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.act = nn.Mish()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + residual)


class SpatialStream(nn.Module):

    def __init__(self, in_channels: int = 4, hidden_dim: int = 128, out_dim: int = 128):
        super().__init__()
        self.init_conv = DirectionalConvBlock(in_channels, hidden_dim)
        self.res_block1 = SpatialResNetBlock(hidden_dim)
        self.res_block2 = SpatialResNetBlock(hidden_dim)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, out_dim), nn.BatchNorm1d(out_dim), nn.Mish()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.init_conv(x)
        feat = self.res_block1(feat)
        feat = self.res_block2(feat)
        pooled = self.global_pool(feat).view(x.size(0), -1)
        return self.fc(pooled)


class TabularGatedBlock(nn.Module):

    def __init__(self, dim: int, dropout_rate: float = 0.15):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * 2)
        self.bn1 = nn.BatchNorm1d(dim * 2)
        self.fc2 = nn.Linear(dim * 2, dim)
        self.bn2 = nn.BatchNorm1d(dim)
        self.gate = nn.Sequential(nn.Linear(dim, dim), nn.Sigmoid())
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.drop(self.act(self.bn1(self.fc1(x))))
        h = self.bn2(self.fc2(h))
        g = self.gate(x)
        return self.act(residual + h * g)


class TabularStream(nn.Module):

    def __init__(self, in_features: int, hidden_dim: int = 256, out_dim: int = 128):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.10),
        )
        self.block1 = TabularGatedBlock(hidden_dim, dropout_rate=0.15)
        self.block2 = TabularGatedBlock(hidden_dim, dropout_rate=0.15)
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, out_dim), nn.BatchNorm1d(out_dim), nn.GELU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_layer(x)
        h = self.block1(h)
        h = self.block2(h)
        return self.proj(h)


class Connect4DualResNet(nn.Module):

    def __init__(
        self,
        tabular_in_features: int,
        spatial_in_channels: int = 4,
        spatial_dim: int = 128,
        tabular_dim: int = 128,
        num_classes: int = 3,
        dropout_rate: float = 0.20,
    ):
        super().__init__()
        self.spatial_stream = SpatialStream(
            in_channels=spatial_in_channels, hidden_dim=128, out_dim=spatial_dim
        )
        self.tabular_stream = TabularStream(
            in_features=tabular_in_features, hidden_dim=256, out_dim=tabular_dim
        )

        fusion_dim = spatial_dim + tabular_dim
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.Mish(),
            nn.Dropout(dropout_rate),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.BatchNorm1d(fusion_dim // 2),
            nn.Mish(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(fusion_dim // 2, num_classes),
        )

    def forward(self, spatial_x: torch.Tensor, tabular_x: torch.Tensor) -> torch.Tensor:
        s_emb = self.spatial_stream(spatial_x)
        t_emb = self.tabular_stream(tabular_x)
        fused = torch.cat([s_emb, t_emb], dim=1)
        return self.fusion_head(fused)


# ---------------------------------------------------------
# 7. Dataset & DataLoader Preparation
# ---------------------------------------------------------
class Connect4Dataset(Dataset):

    def __init__(self, spatial_data, tabular_data, labels=None):
        self.spatial = torch.tensor(spatial_data, dtype=torch.float32)
        self.tabular = torch.tensor(tabular_data, dtype=torch.float32)
        self.labels = (
            torch.tensor(labels, dtype=torch.long) if labels is not None else None
        )

    def __len__(self):
        return len(self.spatial)

    def __getitem__(self, idx):
        if self.labels is not None:
            return self.spatial[idx], self.tabular[idx], self.labels[idx]
        return self.spatial[idx], self.tabular[idx]


train_dataset = Connect4Dataset(spatial_tr, X_tr, y_tr)
val_dataset = Connect4Dataset(spatial_val, X_val, y_val)
test_dataset = Connect4Dataset(spatial_test, X_test)

BATCH_SIZE = 256
train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False
)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False
)

# ---------------------------------------------------------
# 8. Model Training and Validation Routine
# ---------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Executing training on device: {device}")

model = Connect4DualResNet(
    tabular_in_features=X_tr.shape[1],
    spatial_in_channels=spatial_tr.shape[1],
    spatial_dim=128,
    tabular_dim=128,
    num_classes=len(target_classes),
    dropout_rate=0.20,
).to(device)

criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
optimizer = optim.AdamW(
    model.parameters(), lr=1e-3, weight_decay=1e-4, betas=(0.9, 0.999), eps=1e-8
)

EPOCHS = 35
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=EPOCHS, eta_min=1e-5
)

os.makedirs("./working", exist_ok=True)
best_model_path = "./working/best_connect4_dual_resnet.pt"
best_val_acc = 0.0
patience = 12
patience_counter = 0

print("Starting Connect4DualResNet model training...")
for epoch in range(1, EPOCHS + 1):
    model.train()
    total_train_loss = 0.0
    total_train_samples = 0

    for spatial_b, tabular_b, labels_b in train_loader:
        spatial_b = spatial_b.to(device)
        tabular_b = tabular_b.to(device)
        labels_b = labels_b.to(device)

        optimizer.zero_grad()
        logits = model(spatial_b, tabular_b)
        loss = criterion(logits, labels_b)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_train_loss += loss.item() * len(labels_b)
        total_train_samples += len(labels_b)

    avg_train_loss = total_train_loss / total_train_samples

    model.eval()
    total_val_loss = 0.0
    total_val_samples = 0
    val_preds_list = []
    val_targets_list = []

    with torch.no_grad():
        for spatial_b, tabular_b, labels_b in val_loader:
            spatial_b = spatial_b.to(device)
            tabular_b = tabular_b.to(device)
            labels_b = labels_b.to(device)

            logits = model(spatial_b, tabular_b)
            loss = criterion(logits, labels_b)

            preds = torch.argmax(logits, dim=1)
            total_val_loss += loss.item() * len(labels_b)
            total_val_samples += len(labels_b)

            val_preds_list.append(preds.cpu().numpy())
            val_targets_list.append(labels_b.cpu().numpy())

    avg_val_loss = total_val_loss / total_val_samples
    all_val_preds = np.concatenate(val_preds_list)
    all_val_targets = np.concatenate(val_targets_list)
    epoch_val_acc = accuracy_score(all_val_targets, all_val_preds)

    scheduler.step()
    current_lr = scheduler.get_last_lr()[0]

    print(
        f"Epoch {epoch:02d}/{EPOCHS:02d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {epoch_val_acc:.4f} | LR: {current_lr:.6f}"
    )

    if epoch_val_acc > best_val_acc:
        best_val_acc = epoch_val_acc
        patience_counter = 0
        torch.save(model.state_dict(), best_model_path)
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch}.")
            break

# ---------------------------------------------------------
# 9. Evaluation on Hold-Out Validation Set
# ---------------------------------------------------------
model.load_state_dict(torch.load(best_model_path, map_location=device))
model.eval()

eval_preds_list = []
with torch.no_grad():
    for spatial_b, tabular_b, _ in val_loader:
        spatial_b = spatial_b.to(device)
        tabular_b = tabular_b.to(device)
        logits = model(spatial_b, tabular_b)
        eval_preds_list.append(torch.argmax(logits, dim=1).cpu().numpy())

final_val_preds = np.concatenate(eval_preds_list)
final_score = accuracy_score(y_val, final_val_preds)

# ---------------------------------------------------------
# 10. Test Prediction and Submission Generation
# ---------------------------------------------------------
test_preds_list = []
with torch.no_grad():
    for spatial_b, tabular_b in test_loader:
        spatial_b = spatial_b.to(device)
        tabular_b = tabular_b.to(device)
        logits = model(spatial_b, tabular_b)
        test_preds_list.append(torch.argmax(logits, dim=1).cpu().numpy())

test_preds_encoded = np.concatenate(test_preds_list)
test_preds_decoded = target_encoder.inverse_transform(test_preds_encoded)

os.makedirs("./submission", exist_ok=True)
submission_df = pd.DataFrame(
    {"id": np.arange(len(test_df), dtype=int), "class": test_preds_decoded}
)

submission_path = "./submission/submission.csv"
submission_df.to_csv(submission_path, index=False)
print(f"Submission successfully saved to {submission_path}.")

assert len(submission_df) == len(test_df), "Row count mismatch in submission file."
assert list(submission_df.columns) == ["id", "class"], "Invalid submission columns."
assert not submission_df.isnull().any().any(), "Submission contains null values."

print(f"Final Validation Score: {final_score}")