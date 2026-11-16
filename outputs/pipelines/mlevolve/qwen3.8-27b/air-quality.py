import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
import os
import warnings

warnings.filterwarnings("ignore")
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# ============================================================
# 1. LOAD DATA
# ============================================================
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")


# ============================================================
# 2. CREATE TEMPORAL FEATURES
# ============================================================
def create_temporal_features(df):
    df = df.copy()
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
    if "Time" in df.columns:
        if df["Time"].dtype == object:
            df["Time"] = pd.to_datetime(df["Time"], format="%H:%M:%S", errors="coerce")
        elif df["Time"].dtype == int:
            df["Time_hour"] = df["Time"] // 10000
            df["Time_min"] = (df["Time"] // 100) % 100
        else:
            df["Time_hour"] = df["Time"].dt.hour
            df["Time_min"] = df["Time"].dt.minute
    if "Date" in df.columns and "Time" in df.columns:
        if "Time_hour" in df.columns:
            df["datetime"] = (
                df["Date"].dt.normalize()
                + pd.to_timedelta(df["Time_hour"], unit="h")
                + pd.to_timedelta(df["Time_min"], unit="min")
            )
        else:
            df["datetime"] = pd.to_datetime(
                df["Date"].astype(str) + " " + df["Time"].astype(str)
            )
    elif "Date" in df.columns:
        df["datetime"] = pd.to_datetime(df["Date"])
    return df


train = create_temporal_features(train)
test = create_temporal_features(test)

# ============================================================
# 3. SORT BY TIME AND CONCATENATE FOR LAG COMPUTATION
# ============================================================
train["dataset"] = "train"
test["dataset"] = "test"
combined = pd.concat([train, test], axis=0, ignore_index=True)
combined = combined.sort_values("datetime").reset_index(drop=True)

print(f"Combined shape: {combined.shape}")
print(f"Time range: {combined['datetime'].min()} to {combined['datetime'].max()}")

# ============================================================
# 4. FEATURE ENGINEERING
# ============================================================
# --- 4a. Cyclical Temporal Features ---
combined["hour"] = combined["datetime"].dt.hour
combined["day_of_week"] = combined["datetime"].dt.dayofweek
combined["month"] = combined["datetime"].dt.month
combined["day_of_year"] = combined["datetime"].dt.dayofyear
combined["week_of_year"] = combined["datetime"].dt.isocalendar().week.astype(int)
combined["is_weekend"] = (combined["day_of_week"] >= 5).astype(int)

combined["hour_sin"] = np.sin(2 * np.pi * combined["hour"] / 24)
combined["hour_cos"] = np.cos(2 * np.pi * combined["hour"] / 24)
combined["dow_sin"] = np.sin(2 * np.pi * combined["day_of_week"] / 7)
combined["dow_cos"] = np.cos(2 * np.pi * combined["day_of_week"] / 7)
combined["month_sin"] = np.sin(2 * np.pi * combined["month"] / 12)
combined["month_cos"] = np.cos(2 * np.pi * combined["month"] / 12)
combined["doy_sin"] = np.sin(2 * np.pi * combined["day_of_year"] / 365)
combined["doy_cos"] = np.cos(2 * np.pi * combined["day_of_year"] / 365)

# --- 4b. Identify numeric feature columns ---
sensor_cols = [c for c in combined.columns if c.startswith("PT08")]
gt_cols = [c for c in combined.columns if "(GT)" in c and c != "CO(GT)"]
env_cols = ["T", "RH", "AH"]

numeric_features = sensor_cols + gt_cols + env_cols + ["CO(GT)"]
numeric_features = [
    c
    for c in numeric_features
    if c in combined.columns and pd.api.types.is_numeric_dtype(combined[c])
]
print(f"Numeric features for lags/rolling: {numeric_features}")

# --- 4c. Lag Features ---
lag_windows = [1, 2, 3, 6, 12, 24]
for col in numeric_features:
    for lag in lag_windows:
        combined[f"{col}_lag{lag}"] = combined[col].shift(lag)

# --- 4d. Rolling Features ---
roll_windows = [3, 6, 12, 24]
for col in numeric_features:
    for w in roll_windows:
        rolling = combined[col].rolling(window=w, min_periods=1)
        combined[f"{col}_rollmean{w}"] = rolling.mean()
        combined[f"{col}_rollstd{w}"] = rolling.std()

# --- 4e. First Differences ---
for col in numeric_features:
    combined[f"{col}_diff1"] = combined[col].diff(1)
    combined[f"{col}_diff2"] = combined[col].diff(2)

# --- 4f. Interaction Features: Sensor/GT ratios ---
sensor_gt_pairs = [
    ("PT08.S1(CO)", "CO(GT)"),
    ("PT08.S2(NMHC)", "NMHC(GT)"),
    ("PT08.S3(NOx)", "NOx(GT)"),
    ("PT08.S4(NO2)", "NO2(GT)"),
]
for sensor, gt in sensor_gt_pairs:
    if sensor in combined.columns and gt in combined.columns:
        combined[f"{sensor}_ratio_{gt}"] = combined[sensor] / (combined[gt] + 1e-8)
        combined[f"{sensor}_diff_{gt}"] = combined[sensor] - combined[gt]

# Cross-sensor interactions
for i in range(len(sensor_cols)):
    for j in range(i + 1, len(sensor_cols)):
        s1, s2 = sensor_cols[i], sensor_cols[j]
        if s1 in combined.columns and s2 in combined.columns:
            combined[f"{s1}_x_{s2}"] = combined[s1] * combined[s2]
            combined[f"{s1}_div_{s2}"] = combined[s1] / (combined[s2] + 1e-8)

# --- 4g. Environment-Sensor interactions ---
for sensor in sensor_cols:
    if sensor in combined.columns:
        combined[f"{sensor}_x_T"] = combined[sensor] * combined["T"]
        combined[f"{sensor}_x_RH"] = combined[sensor] * combined["RH"]

# --- 4h. Polynomial features ---
for col in ["T", "RH", "AH"]:
    if col in combined.columns:
        combined[f"{col}_sq"] = combined[col] ** 2

# --- 4i. CO-specific features ---
if "CO(GT)" in combined.columns:
    combined["CO_log"] = np.log1p(combined["CO(GT)"].clip(lower=0))
    combined["CO_sq"] = combined["CO(GT)"] ** 2

# ============================================================
# 5. SPLIT BACK INTO TRAIN / VALIDATION / TEST
# ============================================================
train_mask = combined["dataset"] == "train"
test_mask = combined["dataset"] == "test"

train_indices = combined[train_mask].index
test_indices = combined[test_mask].index

n_train = len(train_indices)
val_size = max(1, int(n_train * 0.1))
val_indices = train_indices[-val_size:]
train_only_indices = train_indices[:-val_size]

print(
    f"Train: {len(train_only_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}"
)

# ============================================================
# 6. DEFINE FEATURE COLUMNS
# ============================================================
exclude_cols = {
    "Date",
    "Time",
    "datetime",
    "dataset",
    "id",
    "CO(GT)",
    "Time_hour",
    "Time_min",
}
feature_cols = [c for c in combined.columns if c not in exclude_cols]

for c in feature_cols:
    if not pd.api.types.is_numeric_dtype(combined[c]):
        combined[c] = pd.to_numeric(combined[c], errors="coerce")
        if combined[c].isna().all():
            feature_cols.remove(c)

print(f"Total feature columns: {len(feature_cols)}")

# ============================================================
# 7. HANDLE NaN FROM LAGS/ROLLING
# ============================================================
train_data = combined.loc[train_only_indices, feature_cols].copy()
val_data = combined.loc[val_indices, feature_cols].copy()
test_data = combined.loc[test_indices, feature_cols].copy()

imputer = SimpleImputer(strategy="median")
imputer.fit(train_data)

train_imp = pd.DataFrame(
    imputer.transform(train_data), columns=feature_cols, index=train_only_indices
)
val_imp = pd.DataFrame(
    imputer.transform(val_data), columns=feature_cols, index=val_indices
)
test_imp = pd.DataFrame(
    imputer.transform(test_data), columns=feature_cols, index=test_indices
)

# ============================================================
# 8. SCALING (fit on train only)
# ============================================================
scaler = StandardScaler()
scaler.fit(train_imp)

train_scaled = pd.DataFrame(
    scaler.transform(train_imp), columns=feature_cols, index=train_only_indices
)
val_scaled = pd.DataFrame(
    scaler.transform(val_imp), columns=feature_cols, index=val_indices
)
test_scaled = pd.DataFrame(
    scaler.transform(test_imp), columns=feature_cols, index=test_indices
)

# ============================================================
# 9. PREPARE TENSORS
# ============================================================
X_train = train_scaled.values.astype(np.float32)
y_train = combined.loc[train_only_indices, "CO(GT)"].values.astype(np.float32)
X_val = val_scaled.values.astype(np.float32)
y_val = combined.loc[val_indices, "CO(GT)"].values.astype(np.float32)
X_test = test_scaled.values.astype(np.float32)

X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)
X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
y_train = np.nan_to_num(y_train, nan=0.0, posinf=0.0, neginf=0.0)
y_val = np.nan_to_num(y_val, nan=0.0, posinf=0.0, neginf=0.0)

# Train in log-space for better gradient behavior; inverse-transform at inference
y_train_log = np.log1p(np.clip(y_train, 0, None)).astype(np.float32)
y_val_log = np.log1p(np.clip(y_val, 0, None)).astype(np.float32)

train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train_log))
val_dataset = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val_log))
test_dataset = TensorDataset(torch.from_numpy(X_test))

train_loader = DataLoader(
    train_dataset, batch_size=64, shuffle=True, drop_last=False, num_workers=2
)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=2)


# ============================================================
# 10. MODEL DEFINITION: Residual MLP
# ============================================================
class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout_rate=0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.fc1(x)))
        out = self.dropout(out)
        out = self.bn2(self.fc2(out))
        out = self.dropout(out)
        return self.relu(out + residual)


class ResMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, n_blocks=4, dropout_rate=0.1):
        super().__init__()
        self.input_bn = nn.BatchNorm1d(input_dim)
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.input_relu = nn.ReLU(inplace=True)
        self.blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim, dropout_rate) for _ in range(n_blocks)]
        )
        self.output_bn = nn.BatchNorm1d(hidden_dim)
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        x = self.input_relu(self.input_proj(self.input_bn(x)))
        for block in self.blocks:
            x = block(x)
        x = self.output_bn(x)
        return self.output_head(x).squeeze(-1)


# ============================================================
# 11. LOSS, OPTIMIZER, SCHEDULER
# ============================================================
criterion = nn.MSELoss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = ResMLP(
    input_dim=len(feature_cols), hidden_dim=128, n_blocks=4, dropout_rate=0.1
).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=300, eta_min=1e-5
)


# ============================================================
# 12. TRAINING LOOP WITH EARLY STOPPING
# ============================================================
def predict_log(model, X, device):
    model.eval()
    with torch.no_grad():
        X_t = torch.from_numpy(X).to(device)
        preds = model(X_t).cpu().numpy()
    return preds


def inverse_transform(preds_log):
    preds = np.expm1(preds_log)
    return np.clip(preds, 0, None)


best_val_rmse = float("inf")
best_model_state = None
patience = 40
patience_counter = 0
max_epochs = 300

for epoch in range(max_epochs):
    model.train()
    total_loss = 0
    n_batches = 0

    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1

    scheduler.step()

    val_preds_log = predict_log(model, X_val, device)
    val_preds = inverse_transform(val_preds_log)
    val_rmse = np.sqrt(mean_squared_error(y_val, val_preds))

    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(
            f"Epoch {epoch+1:3d} | Loss: {total_loss/n_batches:.5f} | Val RMSE: {val_rmse:.4f}"
        )

    if val_rmse < best_val_rmse:
        best_val_rmse = val_rmse
        best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1} (best RMSE: {best_val_rmse:.4f})")
            break

# ============================================================
# 13. RESTORE BEST MODEL AND GENERATE FINAL PREDICTIONS
# ============================================================
model.load_state_dict(best_model_state)
model.to(device)
model.eval()

val_preds_log = predict_log(model, X_val, device)
val_preds = inverse_transform(val_preds_log)
final_val_rmse = np.sqrt(mean_squared_error(y_val, val_preds))

test_preds_log = predict_log(model, X_test, device)
test_preds = inverse_transform(test_preds_log)

print(f"Validation RMSE: {final_val_rmse:.4f}")
print(
    f"Test pred stats: mean={test_preds.mean():.4f}, std={test_preds.std():.4f}, min={test_preds.min():.4f}, max={test_preds.max():.4f}"
)

# ============================================================
# 14. SAVE SUBMISSION
# ============================================================
submission = pd.DataFrame({"id": range(len(test_preds)), "CO(GT)": test_preds})
submission.to_csv("./submission/submission.csv", index=False)
print(f"Submission saved: {submission.shape}")

# ============================================================
# 15. FINAL SCORE
# ============================================================
score = final_val_rmse
print(f"Final Validation Score: {score}")