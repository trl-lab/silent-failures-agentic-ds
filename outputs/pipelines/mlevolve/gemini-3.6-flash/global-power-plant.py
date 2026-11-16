import os
import random
import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb

# Set global seed for exact reproducibility
SEED = 42


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


seed_everything(SEED)

# Create working and submission directories
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# 1. Load Raw Datasets
raw_train = pd.read_parquet("./input/train.parquet")
raw_test = pd.read_parquet("./input/test.parquet")

target_col = "capacity_mw"
y = raw_train[target_col].values
X_raw = raw_train.drop(columns=[target_col]).copy()
X_test_raw = raw_test.copy()
if target_col in X_test_raw.columns:
    X_test_raw = X_test_raw.drop(columns=[target_col])

# 2. Feature Engineering Pipeline
gen_cols = [
    c
    for c in X_raw.columns
    if "generation" in c.lower() and pd.api.types.is_numeric_dtype(X_raw[c])
]
cat_cols = X_raw.select_dtypes(include=["object", "category"]).columns.tolist()

for df in [X_raw, X_test_raw]:
    # Generation statistics
    if gen_cols:
        gen_data = df[gen_cols]
        df["gen_mean"] = gen_data.mean(axis=1)
        df["gen_std"] = gen_data.std(axis=1).fillna(0)
        df["gen_sum"] = gen_data.sum(axis=1)
        df["gen_max"] = gen_data.max(axis=1)
        df["gen_min"] = gen_data.min(axis=1)
        df["gen_nonzero_count"] = (gen_data > 0).sum(axis=1)
        df["gen_range"] = df["gen_max"] - df["gen_min"]
        if len(gen_cols) >= 2:
            df["gen_recent_vs_mean"] = gen_data[gen_cols[-1]] - df["gen_mean"]

    # Spatial transformations
    if "latitude" in df.columns and "longitude" in df.columns:
        lat = df["latitude"].fillna(X_raw["latitude"].median())
        lon = df["longitude"].fillna(X_raw["longitude"].median())
        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)
        df["spatial_x"] = np.cos(lat_rad) * np.cos(lon_rad)
        df["spatial_y"] = np.cos(lat_rad) * np.sin(lon_rad)
        df["spatial_z"] = np.sin(lat_rad)
        df["abs_latitude"] = np.abs(lat)

    # Temporal & Plant age variables
    if "commissioning_year" in df.columns:
        df["plant_age"] = 2024 - df["commissioning_year"]
        df["is_commissioning_missing"] = df["commissioning_year"].isna().astype(int)
        median_age = 2024 - X_raw["commissioning_year"].median()
        df["plant_age"] = df["plant_age"].fillna(median_age)

    # Missing feature counts
    df["num_missing_features"] = df.isna().sum(axis=1)

# Spatial KNN Density Metrics
if "latitude" in X_raw.columns and "longitude" in X_raw.columns:
    train_lat_lon = np.radians(
        X_raw[["latitude", "longitude"]]
        .fillna(X_raw[["latitude", "longitude"]].median())
        .values
    )
    nn_tree = NearestNeighbors(n_neighbors=10, algorithm="ball_tree")
    nn_tree.fit(train_lat_lon)
    for df in [X_raw, X_test_raw]:
        coords = np.radians(
            df[["latitude", "longitude"]]
            .fillna(X_raw[["latitude", "longitude"]].median())
            .values
        )
        distances, _ = nn_tree.kneighbors(coords)
        df["nn_dist_mean_5"] = distances[:, :5].mean(axis=1)
        df["nn_dist_mean_10"] = distances[:, :10].mean(axis=1)

# Group-Level Frequency & Summary Encoding
group_cols = [c for c in ["country", "primary_fuel", "owner"] if c in X_raw.columns]
for col in group_cols:
    freq_map = X_raw[col].value_counts().to_dict()
    for df in [X_raw, X_test_raw]:
        df[f"{col}_freq"] = df[col].map(freq_map).fillna(0)
    if "gen_sum" in X_raw.columns:
        group_stats = X_raw.groupby(col)["gen_sum"].mean().to_dict()
        for df in [X_raw, X_test_raw]:
            df[f"{col}_mean_gen_sum"] = df[col].map(group_stats).fillna(0)

# Categorical Ordinal Encoding
for col in cat_cols:
    X_raw[col] = X_raw[col].astype(str)
    X_test_raw[col] = X_test_raw[col].astype(str)
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X_raw[col] = encoder.fit_transform(X_raw[[col]])
    X_test_raw[col] = encoder.transform(X_test_raw[[col]])

# Impute remaining missing values with training medians
num_cols = X_raw.select_dtypes(include=[np.number]).columns.tolist()
train_medians = X_raw[num_cols].median()
X_raw[num_cols] = X_raw[num_cols].fillna(train_medians)
X_test_raw[num_cols] = X_test_raw[num_cols].fillna(train_medians)

X = X_raw.copy()
X_test = X_test_raw[X.columns.tolist()].copy()


# 3. Neural Network Architecture Definition
class TabularResNet(nn.Module):

    def __init__(self, input_dim, hidden_dim=256, dropout_rate=0.15):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
        )
        self.block1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
        )
        self.block2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
        )
        self.act = nn.SiLU()
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.act(x + self.block1(x))
        x = self.act(x + self.block2(x))
        return self.head(x)


# 4. Stratified 5-Fold Cross-Validation Setup
num_splits = 5
num_bins = min(10, len(np.unique(y)))
bins = pd.qcut(y, q=num_bins, labels=False, duplicates="drop")
skf = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=SEED)

oof_preds_lgb = np.zeros(len(X))
oof_preds_lgb_log = np.zeros(len(X))
oof_preds_cb = np.zeros(len(X))
oof_preds_xgb = np.zeros(len(X))
oof_preds_nn = np.zeros(len(X))

test_preds_lgb = np.zeros(len(X_test))
test_preds_lgb_log = np.zeros(len(X_test))
test_preds_cb = np.zeros(len(X_test))
test_preds_xgb = np.zeros(len(X_test))
test_preds_nn = np.zeros(len(X_test))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 5. Cross-Validation Training Loop
for fold, (train_idx, val_idx) in enumerate(skf.split(X, bins)):
    X_tr, y_tr = X.iloc[train_idx], y[train_idx]
    X_va, y_va = X.iloc[val_idx], y[val_idx]

    y_tr_log = np.log1p(np.maximum(0, y_tr))
    y_va_log = np.log1p(np.maximum(0, y_va))

    # --- Model 1: LightGBM Direct Target ---
    lgb_model = lgb.LGBMRegressor(
        n_estimators=1200,
        learning_rate=0.03,
        num_leaves=63,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=SEED + fold,
        n_jobs=-1,
    )
    lgb_model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="rmse",
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
    )
    oof_preds_lgb[val_idx] = np.maximum(0, lgb_model.predict(X_va))
    test_preds_lgb += np.maximum(0, lgb_model.predict(X_test)) / num_splits

    # --- Model 2: LightGBM Log-Transformed Target ---
    lgb_log_model = lgb.LGBMRegressor(
        n_estimators=1200,
        learning_rate=0.03,
        num_leaves=63,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=SEED + fold,
        n_jobs=-1,
    )
    lgb_log_model.fit(
        X_tr,
        y_tr_log,
        eval_set=[(X_va, y_va_log)],
        eval_metric="rmse",
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
    )
    val_p_log = np.expm1(lgb_log_model.predict(X_va))
    oof_preds_lgb_log[val_idx] = np.maximum(0, val_p_log)
    test_p_log = np.expm1(lgb_log_model.predict(X_test))
    test_preds_lgb_log += np.maximum(0, test_p_log) / num_splits

    # --- Model 3: CatBoost Regressor ---
    cb_model = cb.CatBoostRegressor(
        iterations=1200,
        learning_rate=0.03,
        depth=6,
        loss_function="RMSE",
        random_seed=SEED + fold,
        verbose=0,
    )
    cb_model.fit(
        X_tr,
        y_tr,
        eval_set=(X_va, y_va),
        early_stopping_rounds=50,
        verbose=0,
    )
    oof_preds_cb[val_idx] = np.maximum(0, cb_model.predict(X_va))
    test_preds_cb += np.maximum(0, cb_model.predict(X_test)) / num_splits

    # --- Model 4: XGBoost Regressor ---
    xgb_model = xgb.XGBRegressor(
        n_estimators=1200,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=SEED + fold,
        n_jobs=-1,
        early_stopping_rounds=50,
    )
    xgb_model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    oof_preds_xgb[val_idx] = np.maximum(0, xgb_model.predict(X_va))
    test_preds_xgb += np.maximum(0, xgb_model.predict(X_test)) / num_splits

    # --- Model 5: PyTorch TabularResNet ---
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_va_sc = scaler.transform(X_va)
    X_test_sc = scaler.transform(X_test)

    train_dataset = TensorDataset(
        torch.tensor(X_tr_sc, dtype=torch.float32),
        torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1),
    )
    val_dataset = TensorDataset(
        torch.tensor(X_va_sc, dtype=torch.float32),
        torch.tensor(y_va, dtype=torch.float32).unsqueeze(1),
    )

    train_loader = DataLoader(
        train_dataset, batch_size=256, shuffle=True, drop_last=False
    )
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)

    nn_model = TabularResNet(
        input_dim=X_tr_sc.shape[1], hidden_dim=256, dropout_rate=0.15
    ).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(nn_model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=35)

    best_val_loss = float("inf")
    best_nn_weights = None

    for epoch in range(35):
        nn_model.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(nn_model(bx), by)
            loss.backward()
            optimizer.step()
        scheduler.step()

        nn_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                preds = nn_model(bx)
                val_loss += criterion(preds, by).item() * len(by)
        val_loss /= len(val_dataset)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_nn_weights = {
                k: v.cpu().clone() for k, v in nn_model.state_dict().items()
            }

    nn_model.load_state_dict({k: v.to(device) for k, v in best_nn_weights.items()})
    nn_model.eval()

    with torch.no_grad():
        val_p_nn = (
            nn_model(torch.tensor(X_va_sc, dtype=torch.float32).to(device))
            .cpu()
            .numpy()
            .flatten()
        )
        test_p_nn = (
            nn_model(torch.tensor(X_test_sc, dtype=torch.float32).to(device))
            .cpu()
            .numpy()
            .flatten()
        )

    oof_preds_nn[val_idx] = np.maximum(0, val_p_nn)
    test_preds_nn += np.maximum(0, test_p_nn) / num_splits

# 6. Constrained Optimization for Ensemble Blending
oof_list = [
    oof_preds_lgb,
    oof_preds_lgb_log,
    oof_preds_cb,
    oof_preds_xgb,
    oof_preds_nn,
]
test_list = [
    test_preds_lgb,
    test_preds_lgb_log,
    test_preds_cb,
    test_preds_xgb,
    test_preds_nn,
]


def blend_objective(weights):
    weights = weights / np.sum(weights)
    blend = np.zeros_like(y)
    for w, p in zip(weights, oof_list):
        blend += w * p
    blend = np.maximum(0, blend)
    return np.sqrt(mean_squared_error(y, blend))


init_weights = np.ones(len(oof_list)) / len(oof_list)
bounds = [(0, 1) for _ in range(len(oof_list))]
constraints = {"type": "eq", "fun": lambda w: 1.0 - np.sum(w)}

res = minimize(
    blend_objective,
    init_weights,
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
)
blend_weights = res.x / np.sum(res.x)

# 7. Final Out-Of-Fold Evaluation & Test Submission Generation
final_oof_preds = np.zeros_like(y)
for w, p in zip(blend_weights, oof_list):
    final_oof_preds += w * p
final_oof_preds = np.maximum(0, final_oof_preds)

final_score = np.sqrt(mean_squared_error(y, final_oof_preds))

final_test_preds = np.zeros(len(X_test))
for w, p in zip(blend_weights, test_list):
    final_test_preds += w * p
final_test_preds = np.maximum(0, final_test_preds)

id_col = raw_test["id"] if "id" in raw_test.columns else np.arange(len(raw_test))
submission = pd.DataFrame({"id": id_col, "capacity_mw": final_test_preds})
submission.to_csv("./submission/submission.csv", index=False)

print(f"Final Validation Score: {final_score}")
