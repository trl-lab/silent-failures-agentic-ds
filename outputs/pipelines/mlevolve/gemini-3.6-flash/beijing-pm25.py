import os
import pickle
import warnings
import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb

warnings.filterwarnings("ignore")

# ==========================================
# 1. Setup Directories and Load Raw Data
# ==========================================
input_dir = "./input"
working_dir = "./working"
submission_dir = "./submission"

os.makedirs(working_dir, exist_ok=True)
os.makedirs(submission_dir, exist_ok=True)

train_path = os.path.join(input_dir, "train.parquet")
test_path = os.path.join(input_dir, "test.parquet")

print("Loading raw datasets...")
train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

# Drop missing target values in training
train_df = train_df.dropna(subset=["pm2.5"]).reset_index(drop=True)

# Preserve original ordering
train_df["orig_id"] = np.arange(len(train_df))
test_df["orig_id"] = np.arange(len(test_df))

train_df["is_train"] = 1
test_df["is_train"] = 0
test_df["pm2.5"] = np.nan

full_df = pd.concat([train_df, test_df], ignore_index=True)

# ==========================================
# 2. Datetime Construction & Temporal Features
# ==========================================
full_df["datetime"] = pd.to_datetime(full_df[["year", "month", "day", "hour"]])
full_df = full_df.sort_values("datetime").reset_index(drop=True)

full_df["dayofweek"] = full_df["datetime"].dt.dayofweek
full_df["dayofyear"] = full_df["datetime"].dt.dayofyear
full_df["is_weekend"] = full_df["dayofweek"].isin([5, 6]).astype(int)

# Cyclical Encodings
full_df["hour_sin"] = np.sin(2 * np.pi * full_df["hour"] / 24.0)
full_df["hour_cos"] = np.cos(2 * np.pi * full_df["hour"] / 24.0)
full_df["month_sin"] = np.sin(2 * np.pi * full_df["month"] / 12.0)
full_df["month_cos"] = np.cos(2 * np.pi * full_df["month"] / 12.0)
full_df["dayofweek_sin"] = np.sin(2 * np.pi * full_df["dayofweek"] / 7.0)
full_df["dayofweek_cos"] = np.cos(2 * np.pi * full_df["dayofweek"] / 7.0)


# ==========================================
# 3. Domain Meteorological Feature Engineering
# ==========================================
def compute_rh(temp, dewp):
    a, b = 17.625, 243.04
    alpha_t = (a * temp) / (b + temp)
    alpha_d = (a * dewp) / (b + dewp)
    rh = 100.0 * np.exp(alpha_d - alpha_t)
    return np.clip(rh, 0.0, 100.0)


num_cols_raw = ["DEWP", "TEMP", "PRES", "Iws", "Is", "Ir"]
for col in num_cols_raw:
    full_df[col] = (
        full_df[col]
        .ffill()
        .bfill()
        .fillna(full_df[col].median() if len(full_df[col].dropna()) > 0 else 0.0)
    )

full_df["relative_humidity"] = compute_rh(
    full_df["TEMP"].values, full_df["DEWP"].values
)
full_df["dew_point_depression"] = full_df["TEMP"] - full_df["DEWP"]

# Wind Vector Decomposition
cbwd_angles = {"NW": 315.0, "NE": 45.0, "SE": 135.0, "SW": 225.0, "cv": 0.0}
full_df["cbwd_clean"] = full_df["cbwd"].fillna("cv").astype(str)
angles_deg = full_df["cbwd_clean"].map(cbwd_angles).fillna(0.0)
angles_rad = np.radians(angles_deg)

full_df["wind_u"] = -full_df["Iws"] * np.sin(angles_rad)
full_df["wind_v"] = -full_df["Iws"] * np.cos(angles_rad)

# Atmospheric Physics & Air Stagnation Indicators
full_df["air_stagnation_index"] = full_df["relative_humidity"] / (full_df["Iws"] + 1.0)
full_df["wind_power_density"] = full_df["Iws"] ** 3
full_df["pres_temp_ratio"] = full_df["PRES"] / (full_df["TEMP"] + 273.15)

# Domain Atmospheric Physics & Thermodynamic Enhancements
full_df["sat_vapor_pres"] = 0.61078 * np.exp((17.27 * full_df["TEMP"]) / (full_df["TEMP"] + 237.3))
actual_vapor_pres = full_df["sat_vapor_pres"] * (full_df["relative_humidity"] / 100.0)
full_df["vpd"] = full_df["sat_vapor_pres"] - actual_vapor_pres

# Thermal stability proxies
full_df["potential_temp"] = (full_df["TEMP"] + 273.15) * ((1000.0 / (full_df["PRES"] + 1e-5)) ** 0.286)
full_df["thermal_stability_proxy"] = full_df["potential_temp"].diff(1).fillna(0.0)

# Consecutive low-wind stagnant hour counts
is_stagnant = (full_df["Iws"] <= 1.5).astype(int)
full_df["stagnant_hours"] = is_stagnant.groupby((is_stagnant == 0).cumsum()).cumsum()

# Non-linear wind power vectors
full_df["wind_u_power"] = full_df["wind_u"] * (full_df["Iws"] ** 2)
full_df["wind_v_power"] = full_df["wind_v"] * (full_df["Iws"] ** 2)

# Cross-variable interactions
full_df["temp_x_humidity"] = full_df["TEMP"] * full_df["relative_humidity"]
full_df["pres_x_stagnation"] = full_df["PRES"] * full_df["air_stagnation_index"]

full_df = pd.get_dummies(
    full_df, columns=["cbwd_clean"], prefix="cbwd", drop_first=False
)

# ==========================================
# 4. Lags, Tendencies & Rolling Statistics
# ==========================================
base_meteo_cols = [
    "DEWP",
    "TEMP",
    "PRES",
    "Iws",
    "Is",
    "Ir",
    "relative_humidity",
    "dew_point_depression",
    "wind_u",
    "wind_v",
    "air_stagnation_index",
]

for col in ["PRES", "TEMP", "DEWP", "relative_humidity", "Iws", "air_stagnation_index"]:
    full_df[f"{col}_diff_1"] = full_df[col].diff(1).fillna(0.0)
    full_df[f"{col}_diff_3"] = full_df[col].diff(3).fillna(0.0)
    full_df[f"{col}_diff_6"] = full_df[col].diff(6).fillna(0.0)
    full_df[f"{col}_diff_12"] = full_df[col].diff(12).fillna(0.0)
    full_df[f"{col}_diff_24"] = full_df[col].diff(24).fillna(0.0)
    full_df[f"{col}_diff_48"] = full_df[col].diff(48).fillna(0.0)
    full_df[f"{col}_diff_72"] = full_df[col].diff(72).fillna(0.0)

# EWM Features
ewm_spans = [3, 6, 12, 24, 72, 168]
for col in ["PRES", "TEMP", "DEWP", "relative_humidity", "Iws", "air_stagnation_index"]:
    for span in ewm_spans:
        full_df[f"{col}_ewm_{span}"] = full_df[col].ewm(span=span, adjust=False).mean()

lag_steps = [1, 2, 3, 6, 12, 24, 48, 72]
rolling_windows = [3, 6, 12, 24, 48, 72, 168]

for col in base_meteo_cols:
    for lag in lag_steps:
        full_df[f"{col}_lag_{lag}"] = full_df[col].shift(lag).bfill()

    for window in rolling_windows:
        r = full_df[col].rolling(window, min_periods=1)
        full_df[f"{col}_roll_mean_{window}"] = r.mean()
        full_df[f"{col}_roll_std_{window}"] = r.std().fillna(0.0)
        full_df[f"{col}_roll_min_{window}"] = r.min()
        full_df[f"{col}_roll_max_{window}"] = r.max()
        if window in [48, 72, 168]:
            full_df[f"{col}_roll_q25_{window}"] = r.quantile(0.25)
            full_df[f"{col}_roll_q75_{window}"] = r.quantile(0.75)

# ==========================================
# 5. Dataset Splitting & Out-Of-Time Holdout
# ==========================================
train_processed = (
    full_df[full_df["is_train"] == 1].sort_values("datetime").reset_index(drop=True)
)
test_processed = (
    full_df[full_df["is_train"] == 0].sort_values("orig_id").reset_index(drop=True)
)

ignore_cols = ["No", "id", "datetime", "pm2.5", "is_train", "cbwd", "orig_id"]
feature_cols = [c for c in train_processed.columns if c not in ignore_cols]

X_train_full = train_processed[feature_cols].copy()
y_train_full = train_processed["pm2.5"].values.astype(np.float32)
X_test = test_processed[feature_cols].copy()

for col in feature_cols:
    X_train_full[col] = X_train_full[col].astype(np.float32)
    X_test[col] = X_test[col].astype(np.float32)

X_train_full = X_train_full.fillna(0)
X_test = X_test.fillna(0)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

print(f"Dataset Shapes - Train Full: {X_train_full.shape}, Test: {X_test.shape}")


# ==========================================
# 6. Model Architectures & Training
# ==========================================
# --- LightGBM ---
def train_lgb(X_train, y_train, X_validation, y_validation, X_testing):
    print("Training LightGBM Regressor...")
    model = lgb.LGBMRegressor(
        n_estimators=3000,
        learning_rate=0.03,
        num_leaves=63,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_validation, y_validation)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    val_preds = model.predict(X_validation)
    test_preds = model.predict(X_testing)
    return val_preds, test_preds


# --- XGBoost ---
def train_xgb(X_train, y_train, X_validation, y_validation, X_testing):
    print("Training XGBoost Regressor...")
    model = xgb.XGBRegressor(
        n_estimators=3000,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        tree_method="hist",
        random_state=42,
        early_stopping_rounds=50,
        eval_metric="rmse",
        n_jobs=-1,
    )
    model.fit(X_train, y_train, eval_set=[(X_validation, y_validation)], verbose=False)
    val_preds = model.predict(X_validation)
    test_preds = model.predict(X_testing)
    return val_preds, test_preds


# --- CatBoost ---
def train_cb(X_train, y_train, X_validation, y_validation, X_testing):
    print("Training CatBoost Regressor...")
    model = cb.CatBoostRegressor(
        iterations=3000,
        learning_rate=0.03,
        depth=7,
        l2_leaf_reg=3.0,
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=42,
        early_stopping_rounds=50,
        verbose=0,
    )
    model.fit(X_train, y_train, eval_set=(X_validation, y_validation), verbose=0)
    val_preds = model.predict(X_validation)
    test_preds = model.predict(X_testing)
    return val_preds, test_preds


# --- PyTorch Tabular ResNet ---
class TabularResBlock(nn.Module):

    def __init__(self, dim, dropout=0.15):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm1d(dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )

    def forward(self, x):
        return x + self.block(x)


class TabularResNet(nn.Module):

    def __init__(self, input_dim, hidden_dim=256, num_blocks=3, dropout=0.15):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [TabularResBlock(hidden_dim, dropout) for _ in range(num_blocks)]
        )
        self.head = nn.Sequential(
            nn.BatchNorm1d(hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        x = self.input_layer(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x).squeeze(-1)


def train_resnet(X_train, y_train, X_validation, y_validation, X_testing):
    print("Training PyTorch Tabular ResNet...")
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_validation)
    X_test_s = scaler.transform(X_testing)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = TensorDataset(
        torch.tensor(X_tr_s, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val_s, dtype=torch.float32),
        torch.tensor(y_validation, dtype=torch.float32),
    )

    train_loader = DataLoader(
        train_dataset, batch_size=256, shuffle=True, drop_last=False
    )
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False, drop_last=False)

    model = TabularResNet(input_dim=X_tr_s.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=40, eta_min=1e-5
    )

    best_val_loss = float("inf")
    best_weights_path = os.path.join(working_dir, "resnet_best.pth")

    for epoch in range(40):
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()

        scheduler.step()

        model.eval()
        val_loss = 0.0
        val_count = 0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                out = model(bx)
                val_loss += criterion(out, by).item() * len(by)
                val_count += len(by)

        val_rmse = np.sqrt(val_loss / val_count)
        if val_rmse < best_val_loss:
            best_val_loss = val_rmse
            torch.save(model.state_dict(), best_weights_path)

    model.load_state_dict(torch.load(best_weights_path))
    model.eval()

    with torch.no_grad():
        val_tensor = torch.tensor(X_val_s, dtype=torch.float32).to(device)
        val_preds = model(val_tensor).cpu().numpy()

        test_tensor = torch.tensor(X_test_s, dtype=torch.float32).to(device)
        test_preds = model(test_tensor).cpu().numpy()

    return val_preds, test_preds


# Execute 5-Fold Cross-Validation Model Fits
oof_lgb = np.zeros(len(X_train_full))
oof_xgb = np.zeros(len(X_train_full))
oof_cb = np.zeros(len(X_train_full))
oof_nn = np.zeros(len(X_train_full))

test_lgb = np.zeros(len(X_test))
test_xgb = np.zeros(len(X_test))
test_cb = np.zeros(len(X_test))
test_nn = np.zeros(len(X_test))

print("Running 5-fold cross-validation...")
for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_full, y_train_full)):
    print(f"--- Fold {fold + 1} / 5 ---")
    X_tr, y_tr = X_train_full.iloc[train_idx], y_train_full[train_idx]
    X_val, y_val = X_train_full.iloc[val_idx], y_train_full[val_idx]

    val_p_lgb, test_p_lgb = train_lgb(X_tr, y_tr, X_val, y_val, X_test)
    oof_lgb[val_idx] = val_p_lgb
    test_lgb += test_p_lgb / 5.0

    val_p_xgb, test_p_xgb = train_xgb(X_tr, y_tr, X_val, y_val, X_test)
    oof_xgb[val_idx] = val_p_xgb
    test_xgb += test_p_xgb / 5.0

    val_p_cb, test_p_cb = train_cb(X_tr, y_tr, X_val, y_val, X_test)
    oof_cb[val_idx] = val_p_cb
    test_cb += test_p_cb / 5.0

    val_p_nn, test_p_nn = train_resnet(X_tr, y_tr, X_val, y_val, X_test)
    oof_nn[val_idx] = val_p_nn
    test_nn += test_p_nn / 5.0

# ==========================================
# 7. Ensemble Weight Optimization
# ==========================================
val_matrix = np.column_stack([oof_lgb, oof_xgb, oof_cb, oof_nn])
test_matrix = np.column_stack([test_lgb, test_xgb, test_cb, test_nn])


def ensemble_loss(weights):
    w = weights / np.sum(weights)
    blend = np.clip(np.dot(val_matrix, w), 0.0, None)
    return np.sqrt(mean_squared_error(y_train_full, blend))


init_weights = [0.25, 0.25, 0.25, 0.25]
bounds = [(0, 1)] * 4
res = minimize(ensemble_loss, init_weights, bounds=bounds, method="SLSQP")
best_w = res.x / np.sum(res.x)

final_val_preds = np.clip(np.dot(val_matrix, best_w), 0.0, None)
final_test_preds = np.clip(np.dot(test_matrix, best_w), 0.0, None)

final_val_rmse = np.sqrt(mean_squared_error(y_train_full, final_val_preds))

# ==========================================
# 8. Submission Verification & Output
# ==========================================
submission_df = pd.DataFrame(
    {
        "id": np.arange(len(final_test_preds)),
        "pm2.5": final_test_preds,
    }
)

submission_path = os.path.join(submission_dir, "submission.csv")
submission_df.to_csv(submission_path, index=False)

assert os.path.exists(submission_path), f"Missing submission file at {submission_path}"
assert len(submission_df) == len(
    X_test
), f"Row mismatch: {len(submission_df)} vs {len(X_test)}"
assert (
    submission_df["pm2.5"].isnull().sum() == 0
), "NaN values present in submission predictions"

print(f"Final Validation Score: {final_val_rmse}")
