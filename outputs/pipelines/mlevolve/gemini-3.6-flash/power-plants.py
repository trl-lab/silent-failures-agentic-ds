import os
import warnings
import numpy as np
import pandas as pd
import scipy.optimize as opt
from scipy.optimize import minimize
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

warnings.filterwarnings("ignore")

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# ==============================================================================
# 1. Load Data & Preprocessing
# ==============================================================================

train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Identify test IDs
if "id" in test_df.columns:
    test_ids = test_df["id"].values
elif "ID" in test_df.columns:
    test_ids = test_df["ID"].values
else:
    test_ids = np.arange(len(test_df))

# Identify target column
target_col = None
for c in train_df.columns:
    if c.lower() == "namplt_cap":
        target_col = c
        break
if target_col is None:
    target_col = train_df.columns[-1]

train_len = len(train_df)
y_train_raw = train_df[target_col].values

# Combine datasets for feature engineering (excluding target)
df_all = pd.concat(
    [
        train_df.drop(columns=[target_col], errors="ignore"),
        test_df.drop(columns=[target_col], errors="ignore"),
    ],
    ignore_index=True,
)

# ==============================================================================
# 2. Geo-Spatial Feature Engineering
# ==============================================================================

lat_cols = [c for c in df_all.columns if "lat" in c.lower()]
lon_cols = [c for c in df_all.columns if "lon" in c.lower() or "lng" in c.lower()]

if lat_cols and lon_cols:
    lat_name = lat_cols[0]
    lon_name = lon_cols[0]

    df_all[lat_name] = df_all[lat_name].fillna(df_all[lat_name].median())
    df_all[lon_name] = df_all[lon_name].fillna(df_all[lon_name].median())

    # 3D Spherical Cartesian Coordinates
    lat_rad = np.radians(df_all[lat_name])
    lon_rad = np.radians(df_all[lon_name])
    df_all["geo_x"] = np.cos(lat_rad) * np.cos(lon_rad)
    df_all["geo_y"] = np.cos(lat_rad) * np.sin(lon_rad)
    df_all["geo_z"] = np.sin(lat_rad)

    # Multi-resolution K-Means geo-clustering and Spatial Nearest Neighbors
    geo_coords_train = df_all.iloc[:train_len][["geo_x", "geo_y", "geo_z"]].values
    geo_coords_all = df_all[["geo_x", "geo_y", "geo_z"]].values
    for k in [15, 40, 80]:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=5).fit(geo_coords_train)
        df_all[f"geo_cluster_{k}"] = kmeans.predict(geo_coords_all)

    nn = NearestNeighbors(n_neighbors=11, algorithm="kd_tree").fit(geo_coords_train)
    distances, _ = nn.kneighbors(geo_coords_all)
    df_all["dist_nn_1"] = distances[:, 1]
    df_all["dist_nn_5_mean"] = distances[:, 1:6].mean(axis=1)
    df_all["dist_nn_10_mean"] = distances[:, 1:11].mean(axis=1)
    df_all["dist_nn_10_std"] = distances[:, 1:11].std(axis=1)

# ==============================================================================
# 3. Categorical & Numerical Feature Engineering
# ==============================================================================

cat_cols = []
num_cols = []

for col in df_all.columns:
    if col in ["id", "ID"]:
        continue
    if df_all[col].dtype == "object" or df_all[col].dtype.name == "category":
        cat_cols.append(col)
        df_all[col] = df_all[col].astype(str).fillna("missing")
    elif np.issubdtype(df_all[col].dtype, np.number):
        num_cols.append(col)

# Interaction Features for Categoricals
if len(cat_cols) >= 2:
    base_cats = cat_cols[:4]
    for i in range(len(base_cats)):
        for j in range(i + 1, len(base_cats)):
            col1, col2 = base_cats[i], base_cats[j]
            inter_col = f"{col1}_{col2}"
            df_all[inter_col] = df_all[col1] + "_" + df_all[col2]
            if inter_col not in cat_cols:
                cat_cols.append(inter_col)

# Frequency Encodings
for col in cat_cols:
    freq = df_all[col].value_counts()
    df_all[f"{col}_freq"] = df_all[col].map(freq)

# Categorical Group Statistics across all categorical features (Computed using Train set statistics only)
if len(cat_cols) > 0 and len(num_cols) > 0:
    train_subset = df_all.iloc[:train_len]
    num_sub = [c for c in num_cols[:5] if "geo" not in c and "dist" not in c]
    if not num_sub:
        num_sub = num_cols[:3]
    for cat_c in cat_cols:
        for num_c in num_sub:
            grp = train_subset.groupby(cat_c)[num_c]
            mean_map = grp.mean()
            std_map = grp.std().fillna(0)
            min_map = grp.min()
            max_map = grp.max()
            median_map = grp.median()

            df_all[f"{cat_c}_{num_c}_mean"] = df_all[cat_c].map(mean_map)
            df_all[f"{cat_c}_{num_c}_std"] = df_all[cat_c].map(std_map).fillna(0)
            df_all[f"{cat_c}_{num_c}_min"] = df_all[cat_c].map(min_map)
            df_all[f"{cat_c}_{num_c}_max"] = df_all[cat_c].map(max_map)
            df_all[f"{cat_c}_{num_c}_median"] = df_all[cat_c].map(median_map)
            df_all[f"{cat_c}_{num_c}_diff"] = (
                df_all[num_c] - df_all[f"{cat_c}_{num_c}_mean"]
            )

# Split back to Train and Test sets
X_train_full = df_all.iloc[:train_len].copy().reset_index(drop=True)
X_test_full = df_all.iloc[train_len:].copy().reset_index(drop=True)

# Factorize Categorical Variables
for col in cat_cols:
    codes, _ = pd.factorize(df_all[col])
    X_train_full[col] = codes[:train_len]
    X_test_full[col] = codes[train_len:]

# Impute Numerical Missing Values with Train Medians
for col in X_train_full.columns:
    if X_train_full[col].isnull().any() or X_test_full[col].isnull().any():
        med = X_train_full[col].median()
        X_train_full[col] = X_train_full[col].fillna(med)
        X_test_full[col] = X_test_full[col].fillna(med)

# Feature Selection
valid_cols = [
    c
    for c in X_train_full.columns
    if X_train_full[c].nunique() > 1 and c not in ["id", "ID"]
]
X_train = X_train_full[valid_cols].copy()
X_test = X_test_full[valid_cols].copy()

# Out-of-fold Target Statistics (mean, median, std) on Raw Target
kf = KFold(n_splits=5, shuffle=True, random_state=42)

global_mean = y_train_raw.mean()
global_median = np.median(y_train_raw)

for c in cat_cols:
    target_enc_mean = np.zeros(len(X_train))
    target_enc_median = np.zeros(len(X_train))
    target_enc_std = np.zeros(len(X_train))

    for tr_idx, val_idx in kf.split(X_train):
        X_tr = X_train_full.iloc[tr_idx]
        y_tr_series = pd.Series(y_train_raw[tr_idx], index=X_tr.index)
        grp = y_tr_series.groupby(X_tr[c])
        mean_map = grp.mean()
        median_map = grp.median()
        std_map = grp.std().fillna(0)

        target_enc_mean[val_idx] = (
            X_train_full.iloc[val_idx][c].map(mean_map).fillna(global_mean)
        )
        target_enc_median[val_idx] = (
            X_train_full.iloc[val_idx][c].map(median_map).fillna(global_median)
        )
        target_enc_std[val_idx] = (
            X_train_full.iloc[val_idx][c].map(std_map).fillna(0)
        )

    full_series = pd.Series(y_train_raw, index=X_train_full.index)
    full_grp = full_series.groupby(X_train_full[c])
    full_mean_map = full_grp.mean()
    full_median_map = full_grp.median()
    full_std_map = full_grp.std().fillna(0)

    X_train[f"{c}_target_mean"] = target_enc_mean
    X_train[f"{c}_target_median"] = target_enc_median
    X_train[f"{c}_target_std"] = target_enc_std

    X_test[f"{c}_target_mean"] = X_test_full[c].map(full_mean_map).fillna(global_mean)
    X_test[f"{c}_target_median"] = X_test_full[c].map(full_median_map).fillna(global_median)
    X_test[f"{c}_target_std"] = X_test_full[c].map(full_std_map).fillna(0)

# ==============================================================================
# 4. Neural Network Architecture & Model Specifications
# ==============================================================================


class SwiGLU(nn.Module):
    """Gated Linear Unit with SiLU activation."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return x1 * F.silu(x2)


class TabularSwiGLUBlock(nn.Module):
    """Residual Tabular Block using LayerNorm, SwiGLU, and Dropout."""

    def __init__(self, in_features: int, hidden_dim: int, dropout_rate: float = 0.15):
        super().__init__()
        self.norm = nn.LayerNorm(in_features)
        self.fc1 = nn.Linear(in_features, hidden_dim * 2)
        self.swiglu = SwiGLU()
        self.fc2 = nn.Linear(hidden_dim, in_features)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.fc1(x)
        x = self.swiglu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return residual + x


class TabularSwiGLUNet(nn.Module):
    """Deep Tabular Neural Network with Residual Gated Linear Units."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_blocks: int = 3,
        dropout_rate: float = 0.15,
    ):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
        )
        self.blocks = nn.ModuleList(
            [
                TabularSwiGLUBlock(hidden_dim, hidden_dim * 2, dropout_rate)
                for _ in range(num_blocks)
            ]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x)


def create_torch_model(
    input_dim: int,
    hidden_dim: int = 256,
    num_blocks: int = 3,
    dropout_rate: float = 0.15,
) -> nn.Module:
    return TabularSwiGLUNet(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_blocks=num_blocks,
        dropout_rate=dropout_rate,
    )


def create_criterion(delta: float = 1.0) -> nn.Module:
    return nn.HuberLoss(delta=delta)


def create_optimizer(
    model: nn.Module, lr: float = 1e-3, weight_decay: float = 1e-4
) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def get_lgb_params(random_state: int = 42) -> dict:
    return {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "learning_rate": 0.025,
        "num_leaves": 45,
        "max_depth": 7,
        "feature_fraction": 0.75,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "min_child_samples": 20,
        "lambda_l1": 0.1,
        "lambda_l2": 1.0,
        "random_state": random_state,
        "verbosity": -1,
        "n_jobs": -1,
    }


def get_xgb_params(random_state: int = 42) -> dict:
    return {
        "n_estimators": 2000,
        "learning_rate": 0.025,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.75,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": random_state,
        "tree_method": "hist",
        "early_stopping_rounds": 50,
        "eval_metric": "rmse",
    }


def get_cat_params(random_state: int = 42) -> dict:
    return {
        "iterations": 2000,
        "learning_rate": 0.03,
        "depth": 6,
        "l2_leaf_reg": 3.0,
        "random_seed": random_state,
        "verbose": 0,
        "early_stopping_rounds": 50,
    }


# ==============================================================================
# 5. Cross-Validation Training & Model Ensembling
# ==============================================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

n_samples_train = len(X_train)
n_samples_test = len(X_test)
n_splits = kf.get_n_splits()

oof_preds_lgb = np.zeros(n_samples_train)
test_preds_lgb = np.zeros(n_samples_test)

oof_preds_xgb = np.zeros(n_samples_train)
test_preds_xgb = np.zeros(n_samples_test)

oof_preds_cat = np.zeros(n_samples_train)
test_preds_cat = np.zeros(n_samples_test)

oof_preds_nn = np.zeros(n_samples_train)
test_preds_nn = np.zeros(n_samples_test)

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_tr, y_tr = X_train.iloc[train_idx], y_train_raw[train_idx]
    X_va, y_va = X_train.iloc[val_idx], y_train_raw[val_idx]

    # 1. LightGBM Regressor
    lgb_train = lgb.Dataset(X_tr, label=y_tr)
    lgb_val = lgb.Dataset(X_va, label=y_va, reference=lgb_train)
    params_lgb = get_lgb_params(random_state=42 + fold)
    model_lgb = lgb.train(
        params_lgb,
        lgb_train,
        num_boost_round=2000,
        valid_sets=[lgb_val],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )
    oof_preds_lgb[val_idx] = model_lgb.predict(X_va)
    test_preds_lgb += model_lgb.predict(X_test) / n_splits

    # 2. XGBoost Regressor
    params_xgb = get_xgb_params(random_state=42 + fold)
    model_xgb = xgb.XGBRegressor(**params_xgb)
    model_xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    oof_preds_xgb[val_idx] = model_xgb.predict(X_va)
    test_preds_xgb += model_xgb.predict(X_test) / n_splits

    # 3. CatBoost Regressor
    params_cat = get_cat_params(random_state=42 + fold)
    model_cat = CatBoostRegressor(**params_cat)
    model_cat.fit(X_tr, y_tr, eval_set=(X_va, y_va))
    oof_preds_cat[val_idx] = model_cat.predict(X_va)
    test_preds_cat += model_cat.predict(X_test) / n_splits

    # 4. PyTorch TabularSwiGLUNet Regressor
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_va_scaled = scaler.transform(X_va)
    X_te_scaled = scaler.transform(X_test)

    train_dataset = TensorDataset(
        torch.tensor(X_tr_scaled, dtype=torch.float32),
        torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1),
    )
    val_dataset = TensorDataset(
        torch.tensor(X_va_scaled, dtype=torch.float32),
        torch.tensor(y_va, dtype=torch.float32).unsqueeze(1),
    )

    train_loader = DataLoader(
        train_dataset, batch_size=128, shuffle=True, drop_last=False
    )
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, drop_last=False)

    nn_model = create_torch_model(
        input_dim=X_tr.shape[1], hidden_dim=256, num_blocks=3, dropout_rate=0.15
    ).to(device)
    criterion = create_criterion(delta=1.0)
    optimizer = create_optimizer(nn_model, lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    best_val_loss = float("inf")
    best_weights = None
    patience_counter = 0
    max_epochs = 80

    for epoch in range(max_epochs):
        nn_model.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = nn_model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()

        nn_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                out = nn_model(bx)
                val_loss += criterion(out, by).item() * len(bx)
        val_loss /= len(X_va)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = {
                k: v.cpu().clone() for k, v in nn_model.state_dict().items()
            }
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 15:
                break

    if best_weights is not None:
        nn_model.load_state_dict({k: v.to(device) for k, v in best_weights.items()})

    nn_model.eval()
    with torch.no_grad():
        va_tensor = torch.tensor(X_va_scaled, dtype=torch.float32).to(device)
        te_tensor = torch.tensor(X_te_scaled, dtype=torch.float32).to(device)
        oof_preds_nn[val_idx] = nn_model(va_tensor).cpu().numpy().squeeze()
        test_preds_nn += nn_model(te_tensor).cpu().numpy().squeeze() / n_splits

# Non-negativity Clipping
oof_lgb_orig = np.maximum(0, oof_preds_lgb)
oof_xgb_orig = np.maximum(0, oof_preds_xgb)
oof_cat_orig = np.maximum(0, oof_preds_cat)
oof_nn_orig = np.maximum(0, oof_preds_nn)

test_lgb_orig = np.maximum(0, test_preds_lgb)
test_xgb_orig = np.maximum(0, test_preds_xgb)
test_cat_orig = np.maximum(0, test_preds_cat)
test_nn_orig = np.maximum(0, test_preds_nn)

# ==============================================================================
# 6. SLSQP Blend Optimization & Output Generation
# ==============================================================================


def rmse_objective(weights):
    w = np.array(weights)
    w = w / np.sum(w)
    blend = (
        w[0] * oof_lgb_orig
        + w[1] * oof_xgb_orig
        + w[2] * oof_cat_orig
        + w[3] * oof_nn_orig
    )
    return root_mean_squared_error(y_train_raw, blend)


initial_weights = [0.25, 0.25, 0.25, 0.25]
bounds = [(0, 1) for _ in range(4)]
res = minimize(rmse_objective, initial_weights, method="SLSQP", bounds=bounds)
w = res.x / np.sum(res.x)

oof_blend = (
    w[0] * oof_lgb_orig + w[1] * oof_xgb_orig + w[2] * oof_cat_orig + w[3] * oof_nn_orig
)
test_blend = (
    w[0] * test_lgb_orig
    + w[1] * test_xgb_orig
    + w[2] * test_cat_orig
    + w[3] * test_nn_orig
)

final_score = root_mean_squared_error(y_train_raw, oof_blend)

os.makedirs("./submission", exist_ok=True)
submission = pd.DataFrame({"id": test_ids, "NAMPLT_CAP": test_blend})
submission.to_csv("./submission/submission.csv", index=False)

print(f"Final Validation Score: {final_score}")