import os
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# Set random seeds for reproducibility
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


seed_everything(42)

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Encode target variable (Class 2 -> 1, Class 1 -> 0)
le = LabelEncoder()
y = le.fit_transform(train_df["Class"])


# Feature Engineering function
def extract_features(df):
    data = df.copy()
    v1 = data["V1"].astype(float)
    v2 = data["V2"].astype(float)
    v3 = data["V3"].astype(float)
    v4 = data["V4"].astype(float)

    feats = pd.DataFrame(index=df.index)
    feats["V1"] = v1
    feats["V2"] = v2
    feats["V3"] = v3
    feats["V4"] = v4

    # Interaction & domain features
    feats["active_span"] = v4 - v1
    feats["donation_rate"] = v2 / (v4 + 1.0)
    feats["recency_ratio"] = v1 / (v4 + 1.0)
    feats["avg_interval"] = (v4 - v1) / np.maximum(v2 - 1.0, 1.0)
    feats["active_freq_rate"] = v2 / (feats["active_span"] + 1.0)

    # Non-linear transformations
    feats["log_V1"] = np.log1p(v1)
    feats["log_V2"] = np.log1p(v2)
    feats["log_V4"] = np.log1p(v4)

    return feats


X_train_df = extract_features(train_df)
X_test_df = extract_features(test_df)

feature_names = X_train_df.columns.tolist()


# Define Tabular MLP Model with Residual Connection
class TabularMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, dropout_rate=0.2):
        super(TabularMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.act1 = nn.SiLU()
        self.drop1 = nn.Dropout(dropout_rate)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.act2 = nn.SiLU()
        self.drop2 = nn.Dropout(dropout_rate)

        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h = self.drop1(self.act1(self.bn1(self.fc1(x))))
        res = h
        h = self.drop2(self.act2(self.bn2(self.fc2(h))))
        h = h + res
        return self.out(h)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 5-fold Stratified Cross-Validation
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_preds = np.zeros(len(train_df))
test_preds = np.zeros(len(test_df))

num_epochs = 120
batch_size = 32

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_df, y)):
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train_df.iloc[train_idx])
    X_val = scaler.transform(X_train_df.iloc[val_idx])
    X_te = scaler.transform(X_test_df)

    y_tr = y[train_idx]
    y_val = y[val_idx]

    train_dataset = TensorDataset(
        torch.tensor(X_tr, dtype=torch.float32),
        torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1),
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32).unsqueeze(1),
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = TabularMLP(input_dim=X_tr.shape[1], hidden_dim=64, dropout_rate=0.25).to(
        device
    )
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-4
    )

    best_val_auc = 0.0
    best_val_preds = None

    for epoch in range(num_epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_x_t = torch.tensor(X_val, dtype=torch.float32).to(device)
            val_logits = model(val_x_t).cpu().squeeze(1)
            val_probs = torch.sigmoid(val_logits).numpy()
            current_auc = roc_auc_score(y_val, val_probs)

            if current_auc > best_val_auc:
                best_val_auc = current_auc
                best_val_preds = val_probs

                # Evaluate on test set
                te_x_t = torch.tensor(X_te, dtype=torch.float32).to(device)
                best_fold_test_preds = (
                    torch.sigmoid(model(te_x_t)).cpu().squeeze(1).numpy()
                )

    oof_preds[val_idx] = best_val_preds
    test_preds += best_fold_test_preds / n_splits

cv_roc_auc = roc_auc_score(y, oof_preds)
print(f"5-Fold Cross-Validation ROC-AUC: {cv_roc_auc:.5f}")

# Save submission file
os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame({"id": np.arange(len(test_df)), "Class": test_preds})
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved successfully to ./working/submission.csv")
