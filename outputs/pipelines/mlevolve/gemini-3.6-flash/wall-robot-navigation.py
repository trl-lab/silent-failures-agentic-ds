import copy
import os
import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# Ensure required directories exist
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# Set random seeds for exact reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# ---------------------------------------------------------
# 1. Load Datasets & Define Sensor Angles
# ---------------------------------------------------------
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = (
    "class"
    if "class" in train_df.columns
    else [c for c in train_df.columns if c.lower() in ["class", "target"]][0]
)
sensor_cols = [c for c in train_df.columns if c != target_col]

angles_deg = np.array(
    [
        180,
        -165,
        -150,
        -135,
        -120,
        -105,
        -90,
        -75,
        -60,
        -45,
        -30,
        -15,
        0,
        15,
        30,
        45,
        60,
        75,
        90,
        105,
        120,
        135,
        150,
        165,
    ]
)
angles_rad = np.radians(angles_deg)

sectors = {
    "front": [22, 23, 0, 1, 2],
    "front_right": [2, 3, 4, 5],
    "right": [5, 6, 7, 8],
    "rear_right": [8, 9, 10, 11],
    "back": [11, 12, 13, 14],
    "rear_left": [14, 15, 16, 17],
    "left": [17, 18, 19, 20],
    "front_left": [20, 21, 22, 23],
}


# ---------------------------------------------------------
# 2. Feature Engineering Pipeline
# ---------------------------------------------------------
def create_engineered_features(df_input):
    df = df_input.copy()
    sensor_vals = df[sensor_cols].values

    # Global Spatial Aggregations
    df["sensor_min"] = np.min(sensor_vals, axis=1)
    df["sensor_max"] = np.max(sensor_vals, axis=1)
    df["sensor_mean"] = np.mean(sensor_vals, axis=1)
    df["sensor_std"] = np.std(sensor_vals, axis=1)
    df["sensor_range"] = df["sensor_max"] - df["sensor_min"]

    # Directional Sensor Orientation
    min_idx = np.argmin(sensor_vals, axis=1)
    max_idx = np.argmax(sensor_vals, axis=1)
    df["closest_sensor_deg"] = angles_deg[min_idx]
    df["farthest_sensor_deg"] = angles_deg[max_idx]

    # Sector Clearances
    for sector_name, idx_list in sectors.items():
        sec_vals = sensor_vals[:, idx_list]
        df[f"sec_{sector_name}_min"] = np.min(sec_vals, axis=1)
        df[f"sec_{sector_name}_mean"] = np.mean(sec_vals, axis=1)
        df[f"sec_{sector_name}_std"] = np.std(sec_vals, axis=1)

    # Spatial Asymmetry Ratios
    eps = 1e-5
    df["ratio_front_back"] = (df["sec_front_mean"] - df["sec_back_mean"]) / (
        df["sec_front_mean"] + df["sec_back_mean"] + eps
    )
    df["ratio_right_left"] = (df["sec_right_mean"] - df["sec_left_mean"]) / (
        df["sec_right_mean"] + df["sec_left_mean"] + eps
    )
    df["ratio_front_right_left"] = (
        df["sec_front_right_mean"] - df["sec_front_left_mean"]
    ) / (df["sec_front_right_mean"] + df["sec_front_left_mean"] + eps)

    # Polar Center of Mass
    weights = 1.0 / (sensor_vals + 0.05)
    total_w = np.sum(weights, axis=1, keepdims=True)
    cos_vals = np.cos(angles_rad)
    sin_vals = np.sin(angles_rad)

    x_cm = np.sum(weights * cos_vals, axis=1, keepdims=True) / total_w
    y_cm = np.sum(weights * sin_vals, axis=1, keepdims=True) / total_w

    df["obstacle_cm_x"] = x_cm.flatten()
    df["obstacle_cm_y"] = y_cm.flatten()
    df["obstacle_cm_mag"] = np.sqrt(x_cm.flatten() ** 2 + y_cm.flatten() ** 2)
    df["obstacle_cm_angle"] = np.arctan2(y_cm.flatten(), x_cm.flatten())

    # Circular Spatial Gradients
    for i in range(24):
        next_i = (i + 1) % 24
        df[f"spatial_diff_{i+1}"] = sensor_vals[:, next_i] - sensor_vals[:, i]

    # Non-linear Proximity Transformations
    for col in sensor_cols:
        df[f"{col}_log"] = np.log1p(df[col])
        df[f"{col}_inv"] = 1.0 / (df[col] + 0.01)

    # Temporal Lag and Delta Features
    lag_cols = [
        "sensor_min",
        "sec_front_min",
        "sec_right_min",
        "sec_left_min",
        "obstacle_cm_angle",
    ]
    for col in lag_cols:
        for lag in [1, 2]:
            df[f"{col}_lag{lag}"] = df[col].shift(lag).bfill()
            df[f"{col}_delta{lag}"] = df[col] - df[f"{col}_lag{lag}"]

    for col in ["sensor_min", "sec_front_min"]:
        df[f"{col}_roll3_mean"] = df[col].rolling(window=3, min_periods=1).mean()
        df[f"{col}_roll3_std"] = (
            df[col].rolling(window=3, min_periods=1).std().fillna(0)
        )

    return df


train_feat = create_engineered_features(train_df)
test_feat = create_engineered_features(test_df)

# Target Mapping
unique_classes = sorted(train_feat[target_col].unique())
label_map = {cls_val: idx for idx, cls_val in enumerate(unique_classes)}
inv_label_map = {idx: cls_val for idx, cls_val in enumerate(unique_classes)}

if train_feat[target_col].dtype == "object" or train_feat[target_col].min() != 0:
    train_feat[target_col] = train_feat[target_col].map(label_map)

# Stratified Hold-out Validation Split (80/20)
train_split, val_split = train_test_split(
    train_feat, test_size=0.20, random_state=42, stratify=train_feat[target_col]
)

feature_cols = [c for c in train_feat.columns if c != target_col]
num_classes = len(unique_classes)
input_dim = len(feature_cols)

X_train_raw = train_split[feature_cols].values
y_train = train_split[target_col].values.astype(np.int64)

X_val_raw = val_split[feature_cols].values
y_val = val_split[target_col].values.astype(np.int64)

X_test_raw = test_feat[feature_cols].values

# ---------------------------------------------------------
# 3. Data Preprocessing (Imputation & Scaling)
# ---------------------------------------------------------
imputer = SimpleImputer(strategy="median")
X_train_imp = imputer.fit_transform(X_train_raw)
X_val_imp = imputer.transform(X_val_raw)
X_test_imp = imputer.transform(X_test_raw)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imp)
X_val_scaled = scaler.transform(X_val_imp)
X_test_scaled = scaler.transform(X_test_imp)

# ---------------------------------------------------------
# 4. LightGBM Gradient Boosting Model Training
# ---------------------------------------------------------
dtrain = lgb.Dataset(X_train_imp, label=y_train)
dval = lgb.Dataset(X_val_imp, label=y_val, reference=dtrain)

lgb_params = {
    "objective": "multiclass",
    "num_class": num_classes,
    "metric": "multi_logloss",
    "boosting_type": "gbdt",
    "learning_rate": 0.03,
    "num_leaves": 31,
    "max_depth": 6,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "min_child_samples": 20,
    "lambda_l1": 0.1,
    "lambda_l2": 1.0,
    "random_state": 42,
    "n_jobs": -1,
}

lgb_model = lgb.train(
    lgb_params,
    dtrain,
    num_boost_round=1000,
    valid_sets=[dval],
    callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
)

val_preds_lgb = lgb_model.predict(X_val_imp)
test_preds_lgb = lgb_model.predict(X_test_imp)


# ---------------------------------------------------------
# 5. PyTorch Deep Residual Tabular Neural Network
# ---------------------------------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim, dropout_rate=0.2):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.bn1 = nn.LayerNorm(hidden_dim)
        self.act1 = nn.SiLU()
        self.drop1 = nn.Dropout(dropout_rate)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.LayerNorm(hidden_dim)
        self.act2 = nn.SiLU()
        self.drop2 = nn.Dropout(dropout_rate)

    def forward(self, x):
        residual = x
        out = self.drop1(self.act1(self.bn1(self.fc1(x))))
        out = self.bn2(self.fc2(out))
        out = self.act2(out + residual)
        return out


class TabularSensorResNet(nn.Module):
    def __init__(
        self,
        input_dim,
        num_classes=4,
        hidden_dim=256,
        num_blocks=2,
        dropout_rate=0.2,
    ):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
        )

        self.res_blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim, dropout_rate) for _ in range(num_blocks)]
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout_rate / 2.0),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x):
        x = self.input_layer(x)
        for block in self.res_blocks:
            x = block(x)
        logits = self.head(x)
        return logits


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TabularSensorResNet(
    input_dim=input_dim,
    num_classes=num_classes,
    hidden_dim=256,
    num_blocks=2,
    dropout_rate=0.20,
).to(device)

criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=80, eta_min=1e-5)

train_tensor_x = torch.tensor(X_train_scaled, dtype=torch.float32)
train_tensor_y = torch.tensor(y_train, dtype=torch.long)
val_tensor_x = torch.tensor(X_val_scaled, dtype=torch.float32)
val_tensor_y = torch.tensor(y_val, dtype=torch.long)

train_dataset = TensorDataset(train_tensor_x, train_tensor_y)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

epochs = 80
best_loss = float("inf")
best_model_weights = copy.deepcopy(model.state_dict())

for epoch in range(1, epochs + 1):
    model.train()
    train_loss = 0.0
    for bx, by in train_loader:
        bx, by = bx.to(device), by.to(device)
        optimizer.zero_grad()
        logits = model(bx)
        loss = criterion(logits, by)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * len(by)
    scheduler.step()

    model.eval()
    with torch.no_grad():
        val_x_dev = val_tensor_x.to(device)
        val_y_dev = val_tensor_y.to(device)
        val_logits = model(val_x_dev)
        val_loss = criterion(val_logits, val_y_dev).item()

        if val_loss < best_loss:
            best_loss = val_loss
            best_model_weights = copy.deepcopy(model.state_dict())

model.load_state_dict(best_model_weights)
model.eval()

with torch.no_grad():
    val_logits = model(val_tensor_x.to(device))
    val_preds_nn = torch.softmax(val_logits, dim=1).cpu().numpy()

    test_tensor_x = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
    test_logits = model(test_tensor_x)
    test_preds_nn = torch.softmax(test_logits, dim=1).cpu().numpy()

# ---------------------------------------------------------
# 6. Ensemble Weight Optimization & Test Submission
# ---------------------------------------------------------
best_alpha = 0.5
best_ensemble_acc = 0.0

for alpha in np.linspace(0.0, 1.0, 101):
    blend_val = alpha * val_preds_lgb + (1.0 - alpha) * val_preds_nn
    acc = accuracy_score(y_val, np.argmax(blend_val, axis=1))
    if acc > best_ensemble_acc:
        best_ensemble_acc = acc
        best_alpha = alpha

val_ensemble_preds = best_alpha * val_preds_lgb + (1.0 - best_alpha) * val_preds_nn
final_val_score = accuracy_score(y_val, np.argmax(val_ensemble_preds, axis=1))

test_ensemble_preds = best_alpha * test_preds_lgb + (1.0 - best_alpha) * test_preds_nn
test_pred_classes = np.argmax(test_ensemble_preds, axis=1)
test_pred_mapped = [inv_label_map[c] for c in test_pred_classes]

submission_df = pd.DataFrame(
    {"id": np.arange(len(test_pred_mapped)), "class": test_pred_mapped}
)

submission_path = "./submission/submission.csv"
submission_df.to_csv(submission_path, index=False)

print(f"Final Validation Score: {final_val_score}")
