import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# 1. LOAD DATA
# ============================================================
print("Loading data...")
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")

target_col = "Population for whom broadband available, 2019 (%)"
print(f"Target: {target_col}")
print(
    f"Target stats: mean={train_df[target_col].mean():.4f}, std={train_df[target_col].std():.4f}"
)
print(
    f"Target range: [{train_df[target_col].min():.4f}, {train_df[target_col].max():.4f}]"
)

# ============================================================
# 2. SPLIT DATA (FIRST, before any processing)
# ============================================================
y_full = train_df[target_col].values
X_full = train_df.drop(columns=[target_col])

id_col = None
for col in ["id", "Id", "ID", "index"]:
    if col in X_full.columns:
        id_col = col
        break

X_train, X_val, y_train, y_val = train_test_split(
    X_full, y_full, test_size=0.2, random_state=42
)

print(f"Train split: {X_train.shape}, Val split: {X_val.shape}")
print(f"Train target mean: {y_train.mean():.4f}, Val target mean: {y_val.mean():.4f}")

# ============================================================
# 3. FEATURE ENGINEERING (fit on train only)
# ============================================================
print("\n--- Feature Engineering ---")

numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = X_train.select_dtypes(
    include=["object", "category"]
).columns.tolist()

if id_col and id_col in numeric_cols:
    numeric_cols.remove(id_col)
if id_col and id_col in categorical_cols:
    categorical_cols.remove(id_col)

print(f"Numeric features: {len(numeric_cols)}")
print(f"Categorical features: {len(categorical_cols)}")

# --- Handle missing values ---
numeric_medians = {}
for col in numeric_cols:
    if col in X_train.columns:
        numeric_medians[col] = (
            X_train[col].median() if X_train[col].notna().any() else 0.0
        )

categorical_modes = {}
for col in categorical_cols:
    if col in X_train.columns:
        mode_val = X_train[col].mode()
        categorical_modes[col] = mode_val[0] if len(mode_val) > 0 else "UNKNOWN"


def impute_data(df, numeric_medians, categorical_modes, numeric_cols, categorical_cols):
    df = df.copy()
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].fillna(numeric_medians.get(col, 0.0))
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].fillna(categorical_modes.get(col, "UNKNOWN"))
    return df


X_train = impute_data(
    X_train, numeric_medians, categorical_modes, numeric_cols, categorical_cols
)
X_val = impute_data(
    X_val, numeric_medians, categorical_modes, numeric_cols, categorical_cols
)

test_df_clean = test_df.copy()
if target_col in test_df_clean.columns:
    test_df_clean = test_df_clean.drop(columns=[target_col])
test_df_clean = impute_data(
    test_df_clean, numeric_medians, categorical_modes, numeric_cols, categorical_cols
)

# --- Encode categoricals ---
for col in categorical_cols:
    if col in X_train.columns:
        unique_vals = X_train[col].unique()
        mapping = {val: idx for idx, val in enumerate(sorted(unique_vals, key=str))}
        X_train[col] = X_train[col].map(mapping).fillna(-1).astype(int)
        X_val[col] = X_val[col].map(mapping).fillna(-1).astype(int)
        if col in test_df_clean.columns:
            test_df_clean[col] = test_df_clean[col].map(mapping).fillna(-1).astype(int)

# --- Create interaction features ---
print("Creating interaction features...")

variances = X_train[numeric_cols].var().sort_values(ascending=False)
top_numeric = variances.index[: min(8, len(variances))].tolist()

for i in range(len(top_numeric)):
    for j in range(i + 1, len(top_numeric)):
        feat_name = f"inter_{top_numeric[i]}_x_{top_numeric[j]}"
        X_train[feat_name] = X_train[top_numeric[i]] * X_train[top_numeric[j]]
        X_val[feat_name] = X_val[top_numeric[i]] * X_val[top_numeric[j]]
        test_df_clean[feat_name] = (
            test_df_clean[top_numeric[i]] * test_df_clean[top_numeric[j]]
        )

for i in range(len(top_numeric)):
    for j in range(len(top_numeric)):
        if i != j:
            feat_name = f"ratio_{top_numeric[i]}_div_{top_numeric[j]}"
            denom = X_train[top_numeric[j]].replace(0, np.nan)
            X_train[feat_name] = (X_train[top_numeric[i]] / denom).fillna(0)
            denom_val = X_val[top_numeric[j]].replace(0, np.nan)
            X_val[feat_name] = (X_val[top_numeric[i]] / denom_val).fillna(0)
            denom_test = test_df_clean[top_numeric[j]].replace(0, np.nan)
            test_df_clean[feat_name] = (
                test_df_clean[top_numeric[i]] / denom_test
            ).fillna(0)

for col in top_numeric[:5]:
    feat_name = f"{col}_squared"
    X_train[feat_name] = X_train[col] ** 2
    X_val[feat_name] = X_val[col] ** 2
    test_df_clean[feat_name] = test_df_clean[col] ** 2

for col in numeric_cols[:10]:
    if X_train[col].min() > 0:
        feat_name = f"log_{col}"
        X_train[feat_name] = np.log1p(X_train[col])
        X_val[feat_name] = np.log1p(X_val[col])
        test_df_clean[feat_name] = np.log1p(test_df_clean[col])

# --- Group statistics ---
group_cols = []
for col in categorical_cols:
    if (
        col in X_train.columns
        and X_train[col].nunique() < 50
        and X_train[col].nunique() > 1
    ):
        group_cols.append(col)

if group_cols:
    print(f"Group columns detected: {group_cols}")
    for gcol in group_cols[:3]:
        group_mean = (
            pd.DataFrame({"group": X_train[gcol], "target": y_train})
            .groupby("group")["target"]
            .mean()
        )
        X_train[f"{gcol}_target_mean"] = (
            X_train[gcol].map(group_mean).fillna(y_train.mean())
        )
        X_val[f"{gcol}_target_mean"] = (
            X_val[gcol].map(group_mean).fillna(y_train.mean())
        )
        test_df_clean[f"{gcol}_target_mean"] = (
            test_df_clean[gcol].map(group_mean).fillna(y_train.mean())
        )

        group_std = (
            pd.DataFrame({"group": X_train[gcol], "target": y_train})
            .groupby("group")["target"]
            .std()
        )
        X_train[f"{gcol}_target_std"] = X_train[gcol].map(group_std).fillna(0)
        X_val[f"{gcol}_target_std"] = X_val[gcol].map(group_std).fillna(0)
        test_df_clean[f"{gcol}_target_std"] = (
            test_df_clean[gcol].map(group_std).fillna(0)
        )

# --- Scale numeric features ---
print("Scaling numeric features...")
final_numeric_cols = [
    col
    for col in X_train.columns
    if X_train[col].dtype in [np.float64, np.float32, np.int64, np.int32]
]

scaler = StandardScaler()
X_train[final_numeric_cols] = scaler.fit_transform(X_train[final_numeric_cols])
X_val[final_numeric_cols] = scaler.transform(X_val[final_numeric_cols])
test_df_clean[final_numeric_cols] = scaler.transform(test_df_clean[final_numeric_cols])

print(
    f"Final feature shapes: Train={X_train.shape}, Val={X_val.shape}, Test={test_df_clean.shape}"
)

assert X_train.isnull().sum().sum() == 0, "NaNs remain in train!"
assert X_val.isnull().sum().sum() == 0, "NaNs remain in val!"
assert test_df_clean.isnull().sum().sum() == 0, "NaNs remain in test!"

# ============================================================
# 4. MODEL DESIGN (TabularMLP with residual connections)
# ============================================================


class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout_rate=0.3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(out_features, out_features),
            nn.BatchNorm1d(out_features),
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.shortcut = (
            nn.Identity()
            if in_features == out_features
            else nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.BatchNorm1d(out_features),
            )
        )

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.block(x)
        out = self.relu(out + residual)
        return self.dropout(out)


class TabularMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=(256, 128, 64), dropout_rate=0.3):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        self.blocks = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.blocks.append(
                ResidualBlock(hidden_dims[i], hidden_dims[i + 1], dropout_rate)
            )
        self.head = nn.Sequential(
            nn.Linear(hidden_dims[-1], 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(32, 1),
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x):
        x = self.input_layer(x)
        for block in self.blocks:
            x = block(x)
        x = self.head(x)
        return x.squeeze(-1)


criterion = nn.MSELoss()
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 32
EPOCHS = 300

# ============================================================
# 5. TRAINING & EVALUATION
# ============================================================
print("\n=== Training & Evaluation ===")

X_train_np = X_train.values.astype(np.float32)
X_val_np = X_val.values.astype(np.float32)
y_train_np = y_train.astype(np.float32)
y_val_np = y_val.astype(np.float32)
test_np = test_df_clean.values.astype(np.float32)

X_train_np = np.nan_to_num(X_train_np, nan=0.0, posinf=0.0, neginf=0.0)
X_val_np = np.nan_to_num(X_val_np, nan=0.0, posinf=0.0, neginf=0.0)
test_np = np.nan_to_num(test_np, nan=0.0, posinf=0.0, neginf=0.0)

X_train_t = torch.tensor(X_train_np, dtype=torch.float32)
X_val_t = torch.tensor(X_val_np, dtype=torch.float32)
y_train_t = torch.tensor(y_train_np, dtype=torch.float32)
y_val_t = torch.tensor(y_val_np, dtype=torch.float32)
test_t = torch.tensor(test_np, dtype=torch.float32)

train_dataset = TensorDataset(X_train_t, y_train_t)
val_dataset = TensorDataset(X_val_t, y_val_t)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
)

input_dim = X_train_np.shape[1]
print(f"Input dimension: {input_dim}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = TabularMLP(input_dim=input_dim).to(device)

optimizer = torch.optim.AdamW(
    model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=50, T_mult=2, eta_min=1e-5
)

best_val_rmse = float("inf")
best_model_state = None
patience = 30
patience_counter = 0

print(f"Starting training for max {EPOCHS} epochs (early stopping patience={patience})")

for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss = 0.0
    n_batches = 0
    for batch_X, batch_y in train_loader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)
        optimizer.zero_grad()
        preds = model(batch_X)
        loss = criterion(preds, batch_y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        n_batches += 1
    train_loss /= max(n_batches, 1)

    model.eval()
    val_preds_list = []
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X = batch_X.to(device)
            preds = model(batch_X)
            val_preds_list.append(preds.cpu().numpy())

    val_preds = np.concatenate(val_preds_list)
    val_rmse = np.sqrt(mean_squared_error(y_val_np, val_preds))

    scheduler.step()

    if epoch % 10 == 0 or epoch == 1:
        print(
            f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | Val RMSE: {val_rmse:.4f}"
        )

    if val_rmse < best_val_rmse:
        best_val_rmse = val_rmse
        best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(
                f"Early stopping at epoch {epoch}. Best Val RMSE: {best_val_rmse:.4f}"
            )
            break

# Load best model
model.load_state_dict(best_model_state)
model.eval()

print(f"\nBest Validation RMSE: {best_val_rmse:.6f}")

# ============================================================
# 6. FINAL VALIDATION & TEST INFERENCE
# ============================================================
with torch.no_grad():
    val_pred_final = model(X_val_t.to(device)).cpu().numpy()
    test_pred = model(test_t.to(device)).cpu().numpy()

val_pred_final = np.clip(val_pred_final, 0, 100)
test_pred = np.clip(test_pred, 0, 100)

score = np.sqrt(mean_squared_error(y_val_np, val_pred_final))
print(f"Validation RMSE (clipped): {score:.6f}")

# ============================================================
# 7. SAVE SUBMISSION
# ============================================================
os.makedirs("./submission", exist_ok=True)

submission = pd.DataFrame(
    {
        "id": range(len(test_pred)),
        "Population for whom broadband available, 2019 (%)": test_pred,
    }
)
submission.to_csv("./submission/submission.csv", index=False)
print(f"\nSubmission saved: {submission.shape}")
print(submission.head())

print(f"Final Validation Score: {score}")
