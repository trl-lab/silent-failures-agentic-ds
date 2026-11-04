import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# Ensure reproducibility and required output directories
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# ---------------------------------------------------------
# Step 1: Data Processing & Feature Engineering
# ---------------------------------------------------------
train_path = "./input/train.parquet"
test_path = "./input/test.parquet"

train_raw = pd.read_parquet(train_path)
test_raw = pd.read_parquet(test_path)

target_col = "credit_amount"
if target_col not in train_raw.columns:
    raise ValueError(f"Target column '{target_col}' not found in train dataset.")

if "id" in test_raw.columns:
    test_ids = test_raw["id"].values
else:
    test_ids = np.arange(len(test_raw))

y_train_full = train_raw[target_col].values.astype(np.float32)
X_train_full = train_raw.drop(columns=[target_col])
X_test_full = test_raw.copy()

if "id" in X_train_full.columns:
    X_train_full = X_train_full.drop(columns=["id"])
if "id" in X_test_full.columns:
    X_test_full = X_test_full.drop(columns=["id"])

# 80/20 Train/Validation Split
train_idx, val_idx = train_test_split(
    np.arange(len(train_raw)), test_size=0.2, random_state=42
)

df_train = X_train_full.iloc[train_idx].copy()
y_train = y_train_full[train_idx]

df_val = X_train_full.iloc[val_idx].copy()
y_val = y_train_full[val_idx]

df_test = X_test_full.copy()

# Categorize column types dynamically
datetime_cols = []
num_cols = []
cat_cols = []

for col in df_train.columns:
    if pd.api.types.is_datetime64_any_dtype(df_train[col]):
        datetime_cols.append(col)
    elif pd.api.types.is_numeric_dtype(df_train[col]):
        num_cols.append(col)
    else:
        cat_cols.append(col)

# Datetime feature extraction
for col in datetime_cols:
    for df in [df_train, df_val, df_test]:
        df[f"{col}_year"] = df[col].dt.year
        df[f"{col}_month"] = df[col].dt.month
        df[f"{col}_day"] = df[col].dt.day
        df[f"{col}_dayofweek"] = df[col].dt.dayofweek
        df[f"{col}_quarter"] = df[col].dt.quarter
        df[f"{col}_month_sin"] = np.sin(2 * np.pi * df[f"{col}_month"] / 12)
        df[f"{col}_month_cos"] = np.cos(2 * np.pi * df[f"{col}_month"] / 12)
        df.drop(columns=[col], inplace=True)

# Missing Value Handling & Missingness Indicators
missing_cols = [col for col in df_train.columns if df_train[col].isnull().sum() > 0]
for col in missing_cols:
    for df in [df_train, df_val, df_test]:
        df[f"{col}_isna"] = df[col].isnull().astype(int)

for col in cat_cols:
    mode_val = df_train[col].mode()[0] if not df_train[col].mode().empty else "Missing"
    df_train[col] = df_train[col].fillna(mode_val).astype(str)
    df_val[col] = df_val[col].fillna(mode_val).astype(str)
    df_test[col] = df_test[col].fillna(mode_val).astype(str)

for col in num_cols:
    median_val = df_train[col].median()
    if pd.isna(median_val):
        median_val = 0.0
    df_train[col] = df_train[col].fillna(median_val)
    df_val[col] = df_val[col].fillna(median_val)
    df_test[col] = df_test[col].fillna(median_val)

# Out-of-Fold Target Encoding & Frequency Encoding
kf = KFold(n_splits=5, shuffle=True, random_state=42)
global_mean = float(y_train.mean())

for col in cat_cols:
    freq_map = df_train[col].value_counts(normalize=True).to_dict()
    df_train[f"{col}_freq"] = df_train[col].map(freq_map).fillna(0.0)
    df_val[f"{col}_freq"] = df_val[col].map(freq_map).fillna(0.0)
    df_test[f"{col}_freq"] = df_test[col].map(freq_map).fillna(0.0)

    df_train[f"{col}_te"] = np.nan
    for tr_idx, te_idx in kf.split(df_train):
        col_tr = df_train[col].iloc[tr_idx]
        y_tr = y_train[tr_idx]
        te_map = pd.Series(y_tr).groupby(col_tr.values).mean().to_dict()
        df_train.iloc[te_idx, df_train.columns.get_loc(f"{col}_te")] = (
            df_train[col].iloc[te_idx].map(te_map)
        )

    df_train[f"{col}_te"] = df_train[f"{col}_te"].fillna(global_mean)

    full_te_map = pd.Series(y_train).groupby(df_train[col].values).mean().to_dict()
    df_val[f"{col}_te"] = df_val[col].map(full_te_map).fillna(global_mean)
    df_test[f"{col}_te"] = df_test[col].map(full_te_map).fillna(global_mean)

# Group Aggregations
top_cats = cat_cols[:3]
top_nums = num_cols[:5]

for c_col in top_cats:
    for n_col in top_nums:
        agg_df = df_train.groupby(c_col)[n_col].agg(["mean", "std"]).reset_index()
        agg_df.rename(
            columns={"mean": f"{c_col}_{n_col}_mean", "std": f"{c_col}_{n_col}_std"},
            inplace=True,
        )

        df_train = df_train.merge(agg_df, on=c_col, how="left")
        df_val = df_val.merge(agg_df, on=c_col, how="left")
        df_test = df_test.merge(agg_df, on=c_col, how="left")

        df_train[f"{c_col}_{n_col}_std"] = df_train[f"{c_col}_{n_col}_std"].fillna(0.0)
        df_val[f"{c_col}_{n_col}_std"] = df_val[f"{c_col}_{n_col}_std"].fillna(0.0)
        df_test[f"{c_col}_{n_col}_std"] = df_test[f"{c_col}_{n_col}_std"].fillna(0.0)

# Skewness Log Transformation
numeric_feature_cols = [
    col
    for col in df_train.columns
    if col not in cat_cols and pd.api.types.is_numeric_dtype(df_train[col])
]

for col in numeric_feature_cols:
    skewness = df_train[col].skew()
    if skewness > 0.75 and (df_train[col] >= 0).all():
        df_train[col] = np.log1p(df_train[col])
        df_val[col] = np.log1p(df_val[col])
        df_test[col] = np.log1p(df_test[col])

# One-Hot Encoding for Raw Categorical Columns
if cat_cols:
    all_cat_data = pd.concat(
        [df_train[cat_cols], df_val[cat_cols], df_test[cat_cols]], axis=0
    )
    all_cat_encoded = pd.get_dummies(all_cat_data, drop_first=True, dtype=float)

    df_train = df_train.drop(columns=cat_cols)
    df_val = df_val.drop(columns=cat_cols)
    df_test = df_test.drop(columns=cat_cols)

    n_tr = len(df_train)
    n_va = len(df_val)

    encoded_tr = all_cat_encoded.iloc[:n_tr].reset_index(drop=True)
    encoded_va = all_cat_encoded.iloc[n_tr : n_tr + n_va].reset_index(drop=True)
    encoded_te = all_cat_encoded.iloc[n_tr + n_va :].reset_index(drop=True)

    df_train = pd.concat([df_train.reset_index(drop=True), encoded_tr], axis=1)
    df_val = pd.concat([df_val.reset_index(drop=True), encoded_va], axis=1)
    df_test = pd.concat([df_test.reset_index(drop=True), encoded_te], axis=1)

feature_names = list(df_train.columns)
df_val = df_val.reindex(columns=feature_names, fill_value=0.0)
df_test = df_test.reindex(columns=feature_names, fill_value=0.0)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(df_train).astype(np.float32)
X_val = scaler.transform(df_val).astype(np.float32)
X_test = scaler.transform(df_test).astype(np.float32)

input_dim = X_train.shape[1]


# ---------------------------------------------------------
# Step 2: Model Architecture
# ---------------------------------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout_rate: float = 0.15):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.fc1(x)
        out = self.ln1(out)
        out = self.act(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = self.ln2(out)
        out = self.act(out + residual)
        return out


class TabularResNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_blocks: int = 3,
        dropout_rate: float = 0.15,
    ):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
        )
        self.blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim, dropout_rate) for _ in range(num_blocks)]
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x).squeeze(-1)


# ---------------------------------------------------------
# Step 3: Model Training & Evaluation
# ---------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

hidden_dim = 256
num_blocks = 3
dropout_rate = 0.15
learning_rate = 1e-3
weight_decay = 1e-4

model = TabularResNet(
    input_dim=input_dim,
    hidden_dim=hidden_dim,
    num_blocks=num_blocks,
    dropout_rate=dropout_rate,
).to(device)

criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

batch_size = 64
train_dataset = TensorDataset(
    torch.tensor(X_train, dtype=torch.float32),
    torch.tensor(y_train, dtype=torch.float32),
)
val_dataset = TensorDataset(
    torch.tensor(X_val, dtype=torch.float32),
    torch.tensor(y_val, dtype=torch.float32),
)
test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32))

train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, drop_last=False
)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

epochs = 150
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=epochs, eta_min=1e-6
)

best_val_rmse = float("inf")
best_model_path = "./working/best_tabular_resnet.pth"
patience = 20
patience_counter = 0

for epoch in range(1, epochs + 1):
    model.train()
    train_loss_sum = 0.0
    train_count = 0

    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        train_loss_sum += loss.item() * len(batch_y)
        train_count += len(batch_y)

    scheduler.step()
    train_rmse = np.sqrt(train_loss_sum / train_count)

    model.eval()
    val_preds = []
    with torch.no_grad():
        for batch_x, _ in val_loader:
            batch_x = batch_x.to(device)
            preds = model(batch_x)
            val_preds.extend(preds.cpu().numpy().tolist())

    val_preds = np.array(val_preds, dtype=np.float32)
    val_preds = np.clip(val_preds, a_min=0.0, a_max=None)
    val_rmse = np.sqrt(np.mean((y_val - val_preds) ** 2))

    print(
        f"Epoch {epoch:03d} | Train RMSE: {train_rmse:.4f} | Val RMSE: {val_rmse:.4f}"
    )

    if val_rmse < best_val_rmse:
        best_val_rmse = val_rmse
        torch.save(model.state_dict(), best_model_path)
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch}")
            break

# Load best model for final evaluation and inference
model.load_state_dict(torch.load(best_model_path))
model.eval()

# Final Validation Score
val_preds = []
with torch.no_grad():
    for batch_x, _ in val_loader:
        batch_x = batch_x.to(device)
        preds = model(batch_x)
        val_preds.extend(preds.cpu().numpy().tolist())

val_preds = np.array(val_preds, dtype=np.float32)
val_preds = np.clip(val_preds, a_min=0.0, a_max=None)
final_val_score = float(np.sqrt(np.mean((y_val - val_preds) ** 2)))

# Generate Test Predictions
test_preds = []
with torch.no_grad():
    for (batch_x,) in test_loader:
        batch_x = batch_x.to(device)
        preds = model(batch_x)
        test_preds.extend(preds.cpu().numpy().tolist())

test_preds = np.array(test_preds, dtype=np.float32)
test_preds = np.clip(test_preds, a_min=0.0, a_max=None)

# Save Official Submission File
sub = pd.DataFrame({"id": test_ids, "credit_amount": test_preds})
sub["id"] = sub["id"].astype(int)
sub.to_csv("./submission/submission.csv", index=False)

print(f"Final Validation Score: {final_val_score:.6f}")