import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


seed_everything(42)

# 1. Load Data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

feature_cols = [col for col in train_df.columns if col != "y"]
target_col = "y"


def engineer_kinematics_features(df, cols):
    X_raw = df[cols].values
    n_samples, n_joints = X_raw.shape
    features = [X_raw]

    # All contiguous subsegment angular sums: sum_{k=i}^j theta_k
    subsegment_sums = []
    for i in range(n_joints):
        current_sum = np.zeros(n_samples)
        for j in range(i, n_joints):
            current_sum = current_sum + X_raw[:, j]
            subsegment_sums.append(current_sum.copy())

    subsegment_arr = np.column_stack(subsegment_sums)
    features.append(subsegment_arr)

    # Base harmonics for all subsegments
    features.append(np.sin(subsegment_arr))
    features.append(np.cos(subsegment_arr))

    # Double-frequency harmonics for subsegments
    features.append(np.sin(2.0 * subsegment_arr))
    features.append(np.cos(2.0 * subsegment_arr))

    # Pairwise joint differences and their trig functions
    diffs = []
    for i in range(n_joints):
        for j in range(i + 1, n_joints):
            diffs.append(X_raw[:, i] - X_raw[:, j])
    if diffs:
        diffs_arr = np.column_stack(diffs)
        features.append(diffs_arr)
        features.append(np.sin(diffs_arr))
        features.append(np.cos(diffs_arr))

    return np.hstack(features)


X_train_full = engineer_kinematics_features(train_df, feature_cols)
y_train_full = train_df[target_col].values
X_test_full = engineer_kinematics_features(test_df, feature_cols)


# 2. PyTorch Model Architecture
class KinematicsResBlock(nn.Module):
    def __init__(self, hidden_dim, dropout=0.03):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(x + self.block(x))


class KinematicsNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=384, num_blocks=4, dropout=0.03):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU()
        )
        self.blocks = nn.ModuleList(
            [KinematicsResBlock(hidden_dim, dropout) for _ in range(num_blocks)]
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        x = self.input_layer(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x).squeeze(-1)


# 3. 5-Fold Cross Validation Setup with Multi-Seed Bagging
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

seeds_per_fold = [42, 101, 2024]
oof_preds = np.zeros(len(train_df))
test_preds = np.zeros(len(test_df))

epochs = 180
batch_size = 64
lr = 1e-3
weight_decay = 1e-4

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_full, y_train_full)):
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train_full[train_idx])
    y_tr = y_train_full[train_idx]
    X_va = scaler.transform(X_train_full[val_idx])
    y_va = y_train_full[val_idx]
    X_te = scaler.transform(X_test_full)

    val_dataset = TensorDataset(
        torch.tensor(X_va, dtype=torch.float32), torch.tensor(y_va, dtype=torch.float32)
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size * 2, shuffle=False)

    fold_val_preds = np.zeros(len(val_idx))
    fold_test_preds = np.zeros(len(test_df))

    for seed in seeds_per_fold:
        seed_everything(seed + fold * 100)

        train_dataset = TensorDataset(
            torch.tensor(X_tr, dtype=torch.float32),
            torch.tensor(y_tr, dtype=torch.float32),
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        model = KinematicsNet(
            input_dim=X_tr.shape[1], hidden_dim=384, num_blocks=4, dropout=0.03
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-6
        )
        criterion = nn.MSELoss()

        best_val_rmse = float("inf")
        best_weights = None

        for epoch in range(epochs):
            model.train()
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                preds = model(batch_x)
                loss = criterion(preds, batch_y)
                loss.backward()
                optimizer.step()
            scheduler.step()

            # Periodic validation
            model.eval()
            val_preds_list = []
            with torch.no_grad():
                for batch_x, _ in val_loader:
                    batch_x = batch_x.to(device)
                    preds = model(batch_x)
                    val_preds_list.extend(preds.cpu().numpy())

            val_rmse = root_mean_squared_error(y_va, val_preds_list)
            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                best_weights = {k: v.cpu() for k, v in model.state_dict().items()}

        model.load_state_dict(best_weights)
        model.eval()

        with torch.no_grad():
            val_tensor = torch.tensor(X_va, dtype=torch.float32).to(device)
            fold_val_preds += model(val_tensor).cpu().numpy() / len(seeds_per_fold)

            test_tensor = torch.tensor(X_te, dtype=torch.float32).to(device)
            fold_test_preds += model(test_tensor).cpu().numpy() / len(seeds_per_fold)

    oof_preds[val_idx] = fold_val_preds
    test_preds += fold_test_preds / n_splits

    fold_rmse = root_mean_squared_error(y_va, oof_preds[val_idx])
    print(f"Fold {fold + 1} RMSE: {fold_rmse:.5f}")

overall_rmse = root_mean_squared_error(y_train_full, oof_preds)
print(f"Overall Out-of-Fold RMSE: {overall_rmse:.5f}")

# 4. Save Submission
os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame({"id": np.arange(len(test_preds)), "y": test_preds})
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved successfully to ./working/submission.csv")
