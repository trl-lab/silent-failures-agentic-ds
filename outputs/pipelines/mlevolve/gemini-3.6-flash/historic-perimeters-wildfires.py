import os
import math
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import KFold
from sklearn.cluster import MiniBatchKMeans
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Ensure output and working directories exist
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# ==========================================
# 1. DATA LOADING AND FEATURE ENGINEERING
# ==========================================
print("Loading train and test datasets...")
train_raw = pd.read_parquet("./input/train.parquet")
test_raw = pd.read_parquet("./input/test.parquet")

target_col = "gisacres"
id_col = "id" if "id" in train_raw.columns else None

# Preserve original target values
y_raw = train_raw[target_col].values.copy()
y_log = np.log1p(np.maximum(0, y_raw))

drop_cols = [
    col for col in [target_col, id_col] if col is not None and col in train_raw.columns
]

# Holdout Validation Split (80% Train, 20% Validation)
kf_split = KFold(n_splits=5, shuffle=True, random_state=42)
train_idx, val_idx = next(kf_split.split(train_raw))

df_tr = train_raw.iloc[train_idx].copy().reset_index(drop=True)
df_val = train_raw.iloc[val_idx].copy().reset_index(drop=True)
df_te = test_raw.copy().reset_index(drop=True)

y_tr_raw = y_raw[train_idx]
y_val_raw = y_raw[val_idx]
y_tr_log = y_log[train_idx]
y_val_log = y_log[val_idx]

print(f"Data Split - Train: {df_tr.shape}, Val: {df_val.shape}, Test: {df_te.shape}")


def engineer_features(df_train, df_val, df_test):
    tr = df_train.copy()
    va = df_val.copy()
    te = df_test.copy()

    # Process Datetime / Temporal Columns
    def parse_dt_safe(s):
        dt = pd.to_datetime(s, errors="coerce", format="mixed")
        if hasattr(dt, "dt"):
            dt = dt.where((dt.dt.year >= 1800) & (dt.dt.year <= 2100))
        return dt

    date_cols = [
        c
        for c in tr.columns
        if "date" in c.lower()
        or "time" in c.lower()
        or "alarm" in c.lower()
        or "disc" in c.lower()
        or "cont" in c.lower()
    ]
    for col in tr.columns:
        if col not in date_cols and (tr[col].dtype == "object" or "datetime" in str(tr[col].dtype).lower()):
            try:
                parsed = parse_dt_safe(tr[col])
                if parsed.notna().sum() > 0.5 * len(tr):
                    date_cols.append(col)
            except Exception:
                pass

    # Compute pairwise temporal duration / interval features before dropping original datetime columns
    for df in [tr, va, te]:
        dt_dict = {}
        for dcol in date_cols:
            if dcol in df.columns:
                dt_dict[dcol] = parse_dt_safe(df[dcol])

        present_dcols = list(dt_dict.keys())
        for i in range(len(present_dcols)):
            for j in range(i + 1, len(present_dcols)):
                c1, c2 = present_dcols[i], present_dcols[j]
                diff_hours = (dt_dict[c2] - dt_dict[c1]).dt.total_seconds() / 3600.0
                df[f"duration_{c1}_to_{c2}_hours"] = diff_hours
                df[f"duration_{c1}_to_{c2}_days"] = diff_hours / 24.0
                df[f"duration_{c1}_to_{c2}_hours_abs"] = diff_hours.abs()
                df[f"duration_{c1}_to_{c2}_log1p"] = np.log1p(np.maximum(0, diff_hours.abs()))

    for dcol in date_cols:
        for df in [tr, va, te]:
            if dcol in df.columns:
                dt_series = parse_dt_safe(df[dcol])
                df[f"{dcol}_year"] = dt_series.dt.year.fillna(-1).astype(int)
                df[f"{dcol}_month"] = dt_series.dt.month.fillna(-1).astype(int)
                df[f"{dcol}_day"] = dt_series.dt.day.fillna(-1).astype(int)
                df[f"{dcol}_dayofweek"] = dt_series.dt.dayofweek.fillna(-1).astype(int)
                df[f"{dcol}_dayofyear"] = dt_series.dt.dayofyear.fillna(-1).astype(int)

                df[f"{dcol}_month_sin"] = np.sin(2 * np.pi * df[f"{dcol}_month"] / 12.0)
                df[f"{dcol}_month_cos"] = np.cos(2 * np.pi * df[f"{dcol}_month"] / 12.0)
                df[f"{dcol}_doy_sin"] = np.sin(
                    2 * np.pi * df[f"{dcol}_dayofyear"] / 365.25
                )
                df[f"{dcol}_doy_cos"] = np.cos(
                    2 * np.pi * df[f"{dcol}_dayofyear"] / 365.25
                )

                df.drop(columns=[dcol], inplace=True, errors="ignore")

    # Spatial Coordinate Feature Engineering
    lat_cols = [c for c in tr.columns if "lat" in c.lower() or c.lower() == "y"]
    lon_cols = [
        c
        for c in tr.columns
        if "lon" in c.lower() or "long" in c.lower() or c.lower() == "x"
    ]

    if lat_cols and lon_cols:
        lat_col = lat_cols[0]
        lon_col = lon_cols[0]

        lat_min, lat_max = tr[lat_col].min(), tr[lat_col].max()
        lon_min, lon_max = tr[lon_col].min(), tr[lon_col].max()
        lat_center, lon_center = (lat_min + lat_max) / 2.0, (lon_min + lon_max) / 2.0

        for df in [tr, va, te]:
            df["spatial_dist_origin"] = np.sqrt(df[lat_col] ** 2 + df[lon_col] ** 2)
            df["spatial_lat_lon_prod"] = df[lat_col] * df[lon_col]
            df["spatial_lat_lon_ratio"] = df[lat_col] / (df[lon_col].abs() + 1e-5)
            df["spatial_dist_center"] = np.sqrt((df[lat_col] - lat_center) ** 2 + (df[lon_col] - lon_center) ** 2)
            df["spatial_lat_bbox_norm"] = (df[lat_col] - lat_min) / (lat_max - lat_min + 1e-6)
            df["spatial_lon_bbox_norm"] = (df[lon_col] - lon_min) / (lon_max - lon_min + 1e-6)

            for angle in [15, 30, 45, 60]:
                rad = np.radians(angle)
                df[f"coord_rot_{angle}"] = df[lat_col] * np.cos(rad) - df[
                    lon_col
                ] * np.sin(rad)

        coords_tr = tr[[lat_col, lon_col]].fillna(0).values
        coords_va = va[[lat_col, lon_col]].fillna(0).values
        coords_te = te[[lat_col, lon_col]].fillna(0).values

        for n_clusters in [10, 25]:
            kmeans = MiniBatchKMeans(
                n_clusters=n_clusters, random_state=42, batch_size=1024, n_init=3
            )
            tr[f"spatial_cluster_{n_clusters}"] = kmeans.fit_predict(coords_tr)
            va[f"spatial_cluster_{n_clusters}"] = kmeans.predict(coords_va)
            te[f"spatial_cluster_{n_clusters}"] = kmeans.predict(coords_te)

            tr[f"spatial_cluster_dist_{n_clusters}"] = kmeans.transform(coords_tr).min(
                axis=1
            )
            va[f"spatial_cluster_dist_{n_clusters}"] = kmeans.transform(coords_va).min(
                axis=1
            )
            te[f"spatial_cluster_dist_{n_clusters}"] = kmeans.transform(coords_te).min(
                axis=1
            )

    # Numerical Columns Transformations
    num_cols = tr.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c not in drop_cols]

    for c in num_cols:
        nan_cnt = tr[c].isna().sum()
        if nan_cnt > 0.01 * len(tr):
            tr[f"{c}_isna"] = tr[c].isna().astype(int)
            va[f"{c}_isna"] = va[c].isna().astype(int)
            te[f"{c}_isna"] = te[c].isna().astype(int)

        if tr[c].min() >= 0 and tr[c].skew() > 3.0:
            tr[f"{c}_log1p"] = np.log1p(tr[c])
            va[f"{c}_log1p"] = np.log1p(va[c])
            te[f"{c}_log1p"] = np.log1p(te[c])

    # Categorical Columns Encoding
    cat_cols = tr.select_dtypes(include=["object", "category"]).columns.tolist()
    cat_cols = [c for c in cat_cols if c not in drop_cols]

    for c in cat_cols:
        freq = tr[c].value_counts(normalize=True).to_dict()
        tr[f"{c}_freq"] = tr[c].map(freq).fillna(0)
        va[f"{c}_freq"] = va[c].map(freq).fillna(0)
        te[f"{c}_freq"] = te[c].map(freq).fillna(0)

        categories = set(tr[c].dropna().unique())
        cat_map = {val: idx + 1 for idx, val in enumerate(categories)}
        tr[c] = tr[c].map(cat_map).fillna(0).astype(int)
        va[c] = va[c].map(cat_map).fillna(0).astype(int)
        te[c] = te[c].map(cat_map).fillna(0).astype(int)

    return tr, va, te


print("Engineering spatial, temporal, and continuous features...")
df_tr_feat = df_tr.drop(columns=[c for c in drop_cols if c in df_tr.columns])
df_val_feat = df_val.drop(columns=[c for c in drop_cols if c in df_val.columns])
df_te_feat = df_te.drop(columns=[c for c in drop_cols if c in df_te.columns])

X_tr, X_val, X_te = engineer_features(df_tr_feat, df_val_feat, df_te_feat)

# Out-of-Fold Target Encoding
cat_cols_for_te = [
    c
    for c in X_tr.select_dtypes(include=["int64", "int32", "object"]).columns
    if 3 < X_tr[c].nunique() < 500 and c not in drop_cols
]

global_mean_log = y_tr_log.mean()

for col in cat_cols_for_te[:5]:
    X_tr[f"{col}_te"] = global_mean_log
    X_val[f"{col}_te"] = global_mean_log
    X_te[f"{col}_te"] = global_mean_log

    kf_te = KFold(n_splits=5, shuffle=True, random_state=42)
    y_s = pd.Series(y_tr_log, index=X_tr.index)
    for in_fold_idx, out_fold_idx in kf_te.split(X_tr):
        fold_map = (
            y_s.iloc[in_fold_idx].groupby(X_tr.iloc[in_fold_idx][col]).mean().to_dict()
        )
        X_tr.iloc[out_fold_idx, X_tr.columns.get_loc(f"{col}_te")] = (
            X_tr.iloc[out_fold_idx][col].map(fold_map).fillna(global_mean_log)
        )

    train_encoding_map = y_s.groupby(X_tr[col]).mean().to_dict()
    X_val[f"{col}_te"] = X_val[col].map(train_encoding_map).fillna(global_mean_log)
    X_te[f"{col}_te"] = X_te[col].map(train_encoding_map).fillna(global_mean_log)

# Sanitize feature names for GBDT compatibility
def sanitize_cols(cols):
    return [
        c.replace(":", "_")
        .replace(",", "_")
        .replace("{", "_")
        .replace("}", "_")
        .replace("[", "_")
        .replace("]", "_")
        .replace('"', "_")
        for c in cols
    ]

X_tr.columns = sanitize_cols(X_tr.columns)
X_val.columns = sanitize_cols(X_val.columns)
X_te.columns = sanitize_cols(X_te.columns)

final_features = list(X_tr.columns)

X_train_final = X_tr[final_features].copy()
X_val_final = X_val[final_features].copy()
X_test_final = X_te[final_features].copy()

# Leak-free median imputation for downstream GBDT and PyTorch models
imputer = SimpleImputer(strategy="median")
X_tr_imp = pd.DataFrame(imputer.fit_transform(X_train_final), columns=final_features)
X_val_imp = pd.DataFrame(imputer.transform(X_val_final), columns=final_features)
X_te_imp = pd.DataFrame(imputer.transform(X_test_final), columns=final_features)

print(f"Final Feature Matrix Shape: {X_tr_imp.shape}")


# ==========================================
# 2. MODEL DEFINITIONS (PYTORCH ARCHITECTURE)
# ==========================================
class GatedResidualBlock(nn.Module):
    def __init__(self, d_in, d_hidden, dropout_rate=0.2):
        super(GatedResidualBlock, self).__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.norm1 = nn.LayerNorm(d_hidden)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(d_hidden, d_in)
        self.gate = nn.Linear(d_in, d_in)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        residual = x
        out = self.fc1(x)
        out = self.norm1(out)
        out = self.act(out)
        out = self.dropout(out)
        out = self.fc2(out)
        g = torch.sigmoid(self.gate(x))
        return residual + g * out


class TabularGatedResNet(nn.Module):
    def __init__(self, num_features, hidden_dim=256, num_blocks=3, dropout_rate=0.2):
        super(TabularGatedResNet, self).__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
        )
        self.se_att = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim),
            nn.Sigmoid(),
        )
        self.blocks = nn.ModuleList(
            [
                GatedResidualBlock(hidden_dim, hidden_dim * 2, dropout_rate)
                for _ in range(num_blocks)
            ]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout_rate / 2.0),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        h = self.input_layer(x)
        h = h * self.se_att(h)
        for block in self.blocks:
            h = block(h)
        return self.head(h)


class CompositeRawLogRMSELoss(nn.Module):
    def __init__(self, raw_weight=0.05, max_raw_clip=1e6):
        super(CompositeRawLogRMSELoss, self).__init__()
        self.raw_weight = raw_weight
        self.max_raw_clip = max_raw_clip
        self.mse_log = nn.MSELoss()

    def forward(self, pred_log, target_log, target_raw=None):
        loss_log = self.mse_log(pred_log, target_log)
        if target_raw is not None and self.raw_weight > 0:
            pred_raw = torch.expm1(torch.clamp(pred_log, min=0, max=15))
            raw_diff = torch.clamp(
                pred_raw - target_raw, min=-self.max_raw_clip, max=self.max_raw_clip
            )
            loss_raw = torch.mean(raw_diff**2)
            scaled_raw_loss = loss_raw / (torch.mean(target_raw**2) + 1e-6)
            return loss_log + self.raw_weight * scaled_raw_loss
        return loss_log


# ==========================================
# 3. MULTI-MODEL TRAINING & EVALUATION
# ==========================================

# --- A1. LightGBM (RMSE Objective) ---
print("Training LightGBM (RMSE) model...")
lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.03,
    "num_leaves": 63,
    "max_depth": 8,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.5,
    "reg_lambda": 2.0,
    "extra_trees": True,
    "verbose": -1,
    "seed": 42,
    "n_jobs": -1,
}

train_ds_lgb = lgb.Dataset(X_tr_imp, label=y_tr_log)
val_ds_lgb = lgb.Dataset(X_val_imp, label=y_val_log, reference=train_ds_lgb)

lgb_model = lgb.train(
    lgb_params,
    train_ds_lgb,
    num_boost_round=1000,
    valid_sets=[val_ds_lgb],
    callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
)

lgb_val_raw = np.expm1(np.maximum(0, lgb_model.predict(X_val_imp)))
lgb_test_raw = np.expm1(np.maximum(0, lgb_model.predict(X_te_imp)))

# --- A2. LightGBM Tweedie (Heavy-Tail Objective) ---
print("Training LightGBM Tweedie model...")
lgb_tweedie_params = {
    "objective": "tweedie",
    "tweedie_variance_power": 1.5,
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.03,
    "num_leaves": 63,
    "max_depth": 8,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.5,
    "reg_lambda": 2.0,
    "verbose": -1,
    "seed": 42,
    "n_jobs": -1,
}

train_ds_lgb_tw = lgb.Dataset(X_tr_imp, label=y_tr_raw)
val_ds_lgb_tw = lgb.Dataset(X_val_imp, label=y_val_raw, reference=train_ds_lgb_tw)

lgb_tweedie_model = lgb.train(
    lgb_tweedie_params,
    train_ds_lgb_tw,
    num_boost_round=1000,
    valid_sets=[val_ds_lgb_tw],
    callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
)

lgb_tweedie_val_raw = np.maximum(0, lgb_tweedie_model.predict(X_val_imp))
lgb_tweedie_test_raw = np.maximum(0, lgb_tweedie_model.predict(X_te_imp))

# --- B. XGBoost ---
print("Training XGBoost model...")
xgb_params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "learning_rate": 0.03,
    "max_depth": 7,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "alpha": 0.1,
    "lambda": 1.0,
    "tree_method": "hist",
    "random_state": 42,
    "n_jobs": -1,
    "n_estimators": 1000,
    "early_stopping_rounds": 50,
}

xgb_model = xgb.XGBRegressor(**xgb_params)
xgb_model.fit(X_tr_imp, y_tr_log, eval_set=[(X_val_imp, y_val_log)], verbose=False)

xgb_val_raw = np.expm1(np.maximum(0, xgb_model.predict(X_val_imp)))
xgb_test_raw = np.expm1(np.maximum(0, xgb_model.predict(X_te_imp)))

# --- C. CatBoost ---
print("Training CatBoost model...")
cat_params = {
    "loss_function": "RMSE",
    "eval_metric": "RMSE",
    "learning_rate": 0.03,
    "depth": 7,
    "l2_leaf_reg": 5.0,
    "random_strength": 0.1,
    "bagging_temperature": 0.2,
    "random_seed": 42,
    "verbose": False,
    "iterations": 1000,
    "early_stopping_rounds": 50,
}

cat_model = CatBoostRegressor(**cat_params)
cat_model.fit(X_tr_imp, y_tr_log, eval_set=(X_val_imp, y_val_log), verbose=False)

cat_val_raw = np.expm1(np.maximum(0, cat_model.predict(X_val_imp)))
cat_test_raw = np.expm1(np.maximum(0, cat_model.predict(X_te_imp)))

# --- D. PyTorch TabularGatedResNet ---
print("Training PyTorch TabularGatedResNet...")
scaler = StandardScaler()
X_tr_scaled = scaler.fit_transform(X_tr_imp)
X_val_scaled = scaler.transform(X_val_imp)
X_te_scaled = scaler.transform(X_te_imp)

train_tensor_x = torch.tensor(X_tr_scaled, dtype=torch.float32)
train_tensor_y_log = torch.tensor(y_tr_log, dtype=torch.float32).unsqueeze(1)
train_tensor_y_raw = torch.tensor(y_tr_raw, dtype=torch.float32).unsqueeze(1)

val_tensor_x = torch.tensor(X_val_scaled, dtype=torch.float32)
val_tensor_y_log = torch.tensor(y_val_log, dtype=torch.float32).unsqueeze(1)
val_tensor_y_raw = torch.tensor(y_val_raw, dtype=torch.float32).unsqueeze(1)

test_tensor_x = torch.tensor(X_te_scaled, dtype=torch.float32)

train_dataset = TensorDataset(train_tensor_x, train_tensor_y_log, train_tensor_y_raw)
val_dataset = TensorDataset(val_tensor_x, val_tensor_y_log, val_tensor_y_raw)

train_loader = DataLoader(
    train_dataset, batch_size=256, shuffle=True, drop_last=False, num_workers=2
)
val_loader = DataLoader(
    val_dataset, batch_size=512, shuffle=False, drop_last=False, num_workers=2
)
test_loader = DataLoader(
    TensorDataset(test_tensor_x),
    batch_size=512,
    shuffle=False,
    drop_last=False,
    num_workers=2,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_features = X_tr_scaled.shape[1]

nn_model = TabularGatedResNet(
    num_features=num_features, hidden_dim=256, num_blocks=3, dropout_rate=0.2
).to(device)
criterion = CompositeRawLogRMSELoss(raw_weight=0.05).to(device)
optimizer = optim.AdamW(nn_model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40, eta_min=1e-6)

best_val_loss = float("inf")
best_model_path = "./working/best_nn_model.pt"
patience = 10
patience_counter = 0

for epoch in range(1, 41):
    nn_model.train()
    train_loss = 0.0
    train_count = 0
    for bx, by_log, by_raw in train_loader:
        bx, by_log, by_raw = bx.to(device), by_log.to(device), by_raw.to(device)
        optimizer.zero_grad()
        preds = nn_model(bx)
        loss = criterion(preds, by_log, by_raw)
        loss.backward()
        nn.utils.clip_grad_norm_(nn_model.parameters(), max_norm=1.0)
        optimizer.step()
        train_loss += loss.item() * len(bx)
        train_count += len(bx)

    scheduler.step()
    train_loss /= train_count

    nn_model.eval()
    val_loss = 0.0
    val_count = 0
    with torch.no_grad():
        for bx, by_log, by_raw in val_loader:
            bx, by_log, by_raw = bx.to(device), by_log.to(device), by_raw.to(device)
            preds = nn_model(bx)
            loss = criterion(preds, by_log, by_raw)
            val_loss += loss.item() * len(bx)
            val_count += len(bx)

    val_loss /= val_count
    print(
        f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(nn_model.state_dict(), best_model_path)
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch}")
            break

nn_model.load_state_dict(torch.load(best_model_path))
nn_model.eval()

nn_val_log_list = []
with torch.no_grad():
    for bx, _, _ in val_loader:
        preds = nn_model(bx.to(device))
        nn_val_log_list.append(preds.cpu().numpy())
nn_val_raw = np.expm1(np.maximum(0, np.vstack(nn_val_log_list).flatten()))

nn_test_log_list = []
with torch.no_grad():
    for (bx,) in test_loader:
        preds = nn_model(bx.to(device))
        nn_test_log_list.append(preds.cpu().numpy())
nn_test_raw = np.expm1(np.maximum(0, np.vstack(nn_test_log_list).flatten()))


# ==========================================
# 4. ENSEMBLE BLENDING (SLSQP OPTIMIZATION)
# ==========================================
print("Optimizing model ensemble weights using SLSQP...")
val_matrix = np.column_stack([lgb_val_raw, lgb_tweedie_val_raw, xgb_val_raw, cat_val_raw, nn_val_raw])
test_matrix = np.column_stack([lgb_test_raw, lgb_tweedie_test_raw, xgb_test_raw, cat_test_raw, nn_test_raw])


def ensemble_loss_func(weights):
    w = np.array(weights)
    pred = np.dot(val_matrix, w)
    return np.sqrt(mean_squared_error(y_val_raw, pred))


num_models = val_matrix.shape[1]
init_weights = [1.0 / num_models] * num_models
bounds = [(0, 1)] * num_models
constraints = {"type": "eq", "fun": lambda w: sum(w) - 1.0}

opt_res = minimize(
    ensemble_loss_func,
    init_weights,
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
)
best_weights = opt_res.x
print(f"Optimized Ensemble Weights: {best_weights}")

val_preds_final = np.dot(val_matrix, best_weights)
test_preds_final = np.maximum(0, np.dot(test_matrix, best_weights))


# ==========================================
# 5. SUBMISSION GENERATION & METRIC REPORT
# ==========================================
if "id" in test_raw.columns:
    test_ids = test_raw["id"].values
else:
    test_ids = np.arange(len(test_raw))

sub_df = pd.DataFrame({"id": test_ids, "gisacres": test_preds_final})
sub_df.to_csv("./submission/submission.csv", index=False)
print(
    f"Submission generated successfully at ./submission/submission.csv with shape {sub_df.shape}"
)

final_val_rmse = np.sqrt(mean_squared_error(y_val_raw, val_preds_final))
print(f"Final Validation Score: {final_val_rmse}")
