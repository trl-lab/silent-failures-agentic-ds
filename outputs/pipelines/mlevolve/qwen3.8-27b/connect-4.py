import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

SEED = 42
os.makedirs("./submission", exist_ok=True)
os.makedirs("./working", exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Connect4BoardDataset(Dataset):
    """
    Converts flat board feature rows into a 2-channel 6x7 tensor:
      channel 0: Player 1 occupancy
      channel 1: Player 2 occupancy
    """

    def __init__(self, features, labels=None):
        self.features = np.asarray(features, dtype=np.float32)
        self.labels = None if labels is None else np.asarray(labels, dtype=np.int64)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = self.features[idx]
        board = torch.from_numpy(x.reshape(6, 7))
        x = torch.stack(
            (
                (board == 1).float(),
                (board == 2).float(),
            ),
            dim=0,
        )
        if self.labels is None:
            return x
        return x, int(self.labels[idx])


class ResidualBoardBlock(nn.Module):
    def __init__(self, channels, dropout=0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(4, channels),
            nn.SiLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(4, channels),
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(x + self.net(x))


class SymmetricConnect4CNN(nn.Module):
    """
    Compact symmetry-aware CNN for 6x7 Connect-4 boards.
    """

    def __init__(
        self, num_classes=3, base_channels=32, dropout=0.15, head_dropout=0.20
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(2, base_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(4, base_channels),
            nn.SiLU(inplace=True),
        )

        mid_channels = base_channels * 2
        self.mid = nn.Sequential(
            nn.Conv2d(
                base_channels, mid_channels, kernel_size=3, padding=1, bias=False
            ),
            nn.GroupNorm(4, mid_channels),
            nn.SiLU(inplace=True),
        )

        self.blocks = nn.Sequential(
            ResidualBoardBlock(mid_channels, dropout=dropout),
            ResidualBoardBlock(mid_channels, dropout=dropout),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(head_dropout),
            nn.Linear(mid_channels, 64),
            nn.SiLU(inplace=True),
            nn.Dropout(head_dropout * 0.5),
            nn.Linear(64, num_classes),
        )

    def _forward_once(self, x):
        h = self.stem(x)
        h = self.mid(h)
        h = self.blocks(h)
        h = self.pool(h)
        return self.head(h)

    def forward(self, x):
        logits = self._forward_once(x)
        flipped_logits = self._forward_once(torch.flip(x, dims=[-1]))
        return 0.5 * (logits + flipped_logits)


def build_model(num_classes=3):
    return SymmetricConnect4CNN(
        num_classes=num_classes,
        base_channels=32,
        dropout=0.15,
        head_dropout=0.20,
    ).to(DEVICE)


def build_criterion(y_train):
    """
    Class-weighted cross-entropy with capped inverse-frequency weights.
    """
    weights = torch.ones(3, dtype=torch.float32)

    if y_train is not None and len(y_train) > 0:
        counts = np.bincount(y_train, minlength=3).astype(np.float32)
        counts = np.maximum(counts, 1.0)
        inv = 1.0 / counts
        inv = inv / inv.sum()

        max_weight = 2.0
        weights = torch.from_numpy(inv).to(DEVICE)
        weights = torch.clamp(weights, max=max_weight)
        weights = weights / weights.sum()

    return nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)


def build_optimizer(model):
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "weight" in name and param.ndim >= 2:
            decay_params.append(param)
        else:
            no_decay_params.append(param)

    return torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": 1e-2},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=3e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
    )


def build_scheduler(optimizer, total_epochs):
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, total_epochs),
        eta_min=1e-5,
    )


def extract_features(df, feature_columns):
    x = df[feature_columns].to_numpy(dtype=np.float32)
    return np.clip(x, 0.0, 2.0)


def predict_probs(model, loader):
    """
    Identical inference path for validation and test:
    model.eval() -> forward pass -> softmax.
    """
    model.eval()
    all_probs = []
    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, (tuple, list)):
                x = batch[0].to(DEVICE, non_blocking=True)
            else:
                x = batch.to(DEVICE, non_blocking=True)

            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            all_probs.append(probs.cpu().numpy())

    return np.vstack(all_probs)


def main():
    set_seed()

    train_df = pd.read_parquet("./input/train.parquet")
    test_df = pd.read_parquet("./input/test.parquet")

    target_col = "class"
    feature_columns = [c for c in train_df.columns if c not in {target_col, "id"}]

    if "id" in test_df.columns:
        test_ids = test_df["id"].to_numpy()
    else:
        test_ids = np.arange(len(test_df), dtype=int)

    X = extract_features(train_df, feature_columns)
    y = train_df[target_col].to_numpy(dtype=np.int64)

    try:
        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=0.15,
            stratify=y,
            random_state=SEED,
        )
    except ValueError:
        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=0.15,
            stratify=None,
            random_state=SEED,
        )

    X_test = extract_features(test_df, feature_columns)

    train_dataset = Connect4BoardDataset(X_train, y_train)
    val_dataset = Connect4BoardDataset(X_val, y_val)
    test_dataset = Connect4BoardDataset(X_test)

    batch_size = 256
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    model = build_model(num_classes=3)
    criterion = build_criterion(y_train)
    optimizer = build_optimizer(model)

    max_epochs = 60
    patience = 8
    best_model_path = "./working/best_model.pt"

    scheduler = build_scheduler(optimizer, total_epochs=max_epochs)

    best_val_acc = -1.0
    best_state = None
    best_epoch = 0
    bad_epochs = 0

    for epoch in range(1, max_epochs + 1):
        model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        for x, y_batch in train_loader:
            x = x.to(DEVICE, non_blocking=True)
            y_batch = y_batch.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            logits = model(x)
            loss = criterion(logits, y_batch)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.detach().item() * x.size(0)
            correct += (logits.argmax(dim=1) == y_batch).sum().item()
            total += x.size(0)

        scheduler.step()

        val_probs = predict_probs(model, val_loader)
        val_pred = val_probs.argmax(axis=1)
        val_acc = float(accuracy_score(y_val, val_pred))

        train_acc = correct / max(total, 1)
        avg_loss = total_loss / max(total, 1)

        print(
            f"Epoch {epoch:02d} | "
            f"loss {avg_loss:.6f} | "
            f"train_acc {train_acc:.6f} | "
            f"val_acc {val_acc:.6f}"
        )

        if val_acc > best_val_acc + 1e-7:
            best_val_acc = val_acc
            best_epoch = epoch
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            torch.save(best_state, best_model_path)
            bad_epochs = 0
        else:
            bad_epochs += 1

        if bad_epochs >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    elif os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))

    final_val_probs = predict_probs(model, val_loader)
    final_val_pred = final_val_probs.argmax(axis=1)
    score = float(accuracy_score(y_val, final_val_pred))

    test_probs = predict_probs(model, test_loader)
    test_pred = test_probs.argmax(axis=1)

    submission = pd.DataFrame(
        {
            "id": np.asarray(test_ids, dtype=int),
            "class": np.asarray(test_pred, dtype=int),
        }
    )

    submission.to_csv("./submission/submission.csv", index=False)

    print(f"Final Validation Score: {score}")


if __name__ == "__main__":
    main()
