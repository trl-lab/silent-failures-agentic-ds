import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "car" if "car" in train_df.columns else "class"
feature_cols = [c for c in train_df.columns if c != target_col]

# Encode target
target_encoder = LabelEncoder()
y_train_full = target_encoder.fit_transform(train_df[target_col].astype(str))
num_classes = len(target_encoder.classes_)

# Encode categorical features
cat_encoders = {}
X_train_encoded = np.zeros((len(train_df), len(feature_cols)), dtype=np.int64)
X_test_encoded = np.zeros((len(test_df), len(feature_cols)), dtype=np.int64)
embedding_sizes = []

for idx, col in enumerate(feature_cols):
    le = LabelEncoder()
    combined_vals = pd.concat([train_df[col], test_df[col]]).astype(str)
    le.fit(combined_vals)
    cat_encoders[col] = le
    X_train_encoded[:, idx] = le.transform(train_df[col].astype(str))
    X_test_encoded[:, idx] = le.transform(test_df[col].astype(str))
    num_unique = len(le.classes_)
    emb_dim = min(50, max(4, (num_unique + 1) // 2))
    embedding_sizes.append((num_unique, emb_dim))


class CarDataset(Dataset):
    def __init__(self, x_data, y_data=None):
        self.x = torch.tensor(x_data, dtype=torch.long)
        self.y = torch.tensor(y_data, dtype=torch.long) if y_data is not None else None

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.x[idx], self.y[idx]
        return self.x[idx]


class CarMLP(nn.Module):
    def __init__(self, emb_sizes, num_classes):
        super(CarMLP, self).__init__()
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(num_categories, emb_dim)
                for num_categories, emb_dim in emb_sizes
            ]
        )
        total_emb_dim = sum(emb_dim for _, emb_dim in emb_sizes)
        self.network = nn.Sequential(
            nn.Linear(total_emb_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        embedded = [emb(x[:, i]) for i, emb in enumerate(self.embeddings)]
        x_concat = torch.cat(embedded, dim=1)
        return self.network(x_concat)


# 5-fold Stratified Cross-Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_predictions = np.zeros((len(train_df), num_classes))
test_dataset = CarDataset(X_test_encoded)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
test_preds_folds = np.zeros((len(test_df), num_classes))

epochs = 120
batch_size = 32

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_encoded, y_train_full)):
    train_sub_ds = CarDataset(X_train_encoded[train_idx], y_train_full[train_idx])
    val_sub_ds = CarDataset(X_train_encoded[val_idx], y_train_full[val_idx])

    train_loader = DataLoader(train_sub_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_sub_ds, batch_size=64, shuffle=False)

    model = CarMLP(embedding_sizes, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    best_weights = None

    for epoch in range(epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item() * len(batch_y)
        val_loss /= len(val_idx)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Load best model for fold evaluation
    model.load_state_dict(best_weights)
    model.eval()
    val_preds_list = []
    with torch.no_grad():
        for batch_x, _ in val_loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            val_preds_list.append(torch.softmax(outputs, dim=1).cpu().numpy())
    oof_predictions[val_idx] = np.concatenate(val_preds_list, axis=0)

    # Test predictions for this fold
    fold_test_preds = []
    with torch.no_grad():
        for batch_x in test_loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            fold_test_preds.append(torch.softmax(outputs, dim=1).cpu().numpy())
    test_preds_folds += np.concatenate(fold_test_preds, axis=0) / 5.0

# Calculate OOF Accuracy
oof_pred_labels = np.argmax(oof_predictions, axis=1)
cv_accuracy = accuracy_score(y_train_full, oof_pred_labels)
print(f"Validation Metric (5-Fold CV Accuracy): {cv_accuracy:.5f}")

# Generate Submission
final_test_labels = target_encoder.inverse_transform(
    np.argmax(test_preds_folds, axis=1)
)
os.makedirs("./working", exist_ok=True)
submission_df = pd.DataFrame({"id": np.arange(len(test_df)), "car": final_test_labels})
submission_df.to_csv("./working/submission.csv", index=False)
print("Submission saved successfully to ./working/submission.csv")
