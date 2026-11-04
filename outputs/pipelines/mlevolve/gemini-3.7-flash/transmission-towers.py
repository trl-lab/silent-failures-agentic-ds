import gc
import os
from catboost import CatBoostRegressor
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb

# =========================================================================
# 0. Global Configurations and Reproducibility
# =========================================================================
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs("./submission", exist_ok=True)
os.makedirs("./working", exist_ok=True)

# =========================================================================
# 1. Load Data and Identify Column Schema
# =========================================================================
print("Loading datasets...")
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "SupStruc"
if target_col not in train_df.columns:
    for col in train_df.columns:
        if col.lower() == target_col.lower():
            target_col = col
            break

print(
    f"Train shape: {train_df.shape}, Test shape: {test_df.shape}, Target: {target_col}"
)

y_train = train_df[target_col].copy().astype(np.float32)
X_train_raw = train_df.drop(columns=[target_col]).copy()
X_test_raw = test_df.copy()

# Preserve or detect ID column
id_col = None
for candidate in ["id", "ID", "Id", "OBJECTID", "objectid", "index"]:
    if candidate in X_train_raw.columns:
        id_col = candidate
        break

all_features = [c for c in X_train_raw.columns if c != id_col]

lat_cols = [
    c
    for c in all_features
    if any(k in c.lower() for k in ["lat", "latitude", "y_coord", "y"])
]
lon_cols = [
    c
    for c in all_features
    if any(k in c.lower() for k in ["lon", "long", "longitude", "x_coord", "x"])
]

numeric_cols = []
categorical_cols = []

for c in all_features:
    if pd.api.types.is_numeric_dtype(X_train_raw[c]):
        numeric_cols.append(c)
    else:
        categorical_cols.append(c)

print(
    f"Detected {len(numeric_cols)} numeric features and {len(categorical_cols)} categorical features."
)


# =========================================================================
# 2. Geospatial Feature Engineering
# =========================================================================
def engineer_geospatial(df, lat_name=None, lon_name=None):
    df_out = df.copy()
    if lat_name and lon_name and lat_name in df.columns and lon_name in df.columns:
        lat = df[lat_name].fillna(df[lat_name].median()).values
        lon = df[lon_name].fillna(df[lon_name].median()).values

        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)

        # 3D Cartesian coordinates (Earth sphere approximation)
        df_out["geo_x"] = np.cos(lat_rad) * np.cos(lon_rad)
        df_out["geo_y"] = np.cos(lat_rad) * np.sin(lon_rad)
        df_out["geo_z"] = np.sin(lat_rad)

        # Coordinate Rotations for directional alignment
        for angle in [30, 45, 60]:
            rad = np.radians(angle)
            df_out[f"rot_x_{angle}"] = lon * np.cos(rad) - lat * np.sin(rad)
            df_out[f"rot_y_{angle}"] = lon * np.sin(rad) + lat * np.cos(rad)

        # Polar Coordinates
        df_out["geo_radius"] = np.sqrt(lat**2 + lon**2)
        df_out["geo_angle"] = np.arctan2(lat, lon)
    return df_out


lat_name = lat_cols[0] if len(lat_cols) > 0 else None
lon_name = lon_cols[0] if len(lon_cols) > 0 else None

X_train_geo = engineer_geospatial(X_train_raw, lat_name, lon_name)
X_test_geo = engineer_geospatial(X_test_raw, lat_name, lon_name)

# Spatial Clustering and Distance to Centers (fitted strictly on training set)
if lat_name and lon_name:
    train_coords = X_train_geo[[lat_name, lon_name]].fillna(0).values
    test_coords = X_test_geo[[lat_name, lon_name]].fillna(0).values

    n_clusters_list = [10, 25, 50]
    for k in n_clusters_list:
        kmeans = MiniBatchKMeans(n_clusters=k, random_state=SEED, batch_size=2048)
        kmeans.fit(train_coords)
        X_train_geo[f"cluster_{k}"] = kmeans.predict(train_coords)
        X_test_geo[f"cluster_{k}"] = kmeans.predict(test_coords)
        # Distance to cluster centers
        train_dist = kmeans.transform(train_coords)
        test_dist = kmeans.transform(test_coords)
        X_train_geo[f"dist_to_cluster_{k}_min"] = train_dist.min(axis=1)
        test_dist_min = test_dist.min(axis=1)
        X_test_geo[f"dist_to_cluster_{k}_min"] = test_dist_min

    # Spatial k-Nearest Neighbors density and coordinate aggregations
    knn = NearestNeighbors(n_neighbors=15, metric="euclidean", n_jobs=-1)
    knn.fit(train_coords)
    train_knn_dist, train_knn_idx = knn.kneighbors(train_coords)
    test_knn_dist, test_knn_idx = knn.kneighbors(test_coords)

    X_train_geo["knn_dist_mean_5"] = train_knn_dist[:, 1:6].mean(axis=1)
    X_train_geo["knn_dist_mean_15"] = train_knn_dist[:, 1:15].mean(axis=1)
    X_train_geo["knn_dist_std_15"] = train_knn_dist[:, 1:15].std(axis=1)

    X_test_geo["knn_dist_mean_5"] = test_knn_dist[:, 1:6].mean(axis=1)
    X_test_geo["knn_dist_mean_15"] = test_knn_dist[:, 1:15].mean(axis=1)
    X_test_geo["knn_dist_std_15"] = test_knn_dist[:, 1:15].std(axis=1)

# =========================================================================
# 3. Categorical & Missingness Transforms
# =========================================================================
num_feats = [
    c
    for c in X_train_geo.columns
    if pd.api.types.is_numeric_dtype(X_train_geo[c]) and c != id_col
]
cat_feats = [
    c
    for c in X_train_geo.columns
    if not pd.api.types.is_numeric_dtype(X_train_geo[c]) and c != id_col
]

# 2-way categorical interaction features
pair_feats = []
if len(cat_feats) >= 2:
    top_cats = sorted(cat_feats, key=lambda c: X_train_geo[c].nunique(), reverse=True)[:5]
    for i in range(len(top_cats)):
        for j in range(i + 1, len(top_cats)):
            c1, c2 = top_cats[i], top_cats[j]
            pair_name = f"{c1}_x_{c2}"
            pair_feats.append(pair_name)
            X_train_geo[pair_name] = X_train_geo[c1].astype(str) + "_" + X_train_geo[c2].astype(str)
            X_test_geo[pair_name] = X_test_geo[c1].astype(str) + "_" + X_test_geo[c2].astype(str)

all_cat_cols = cat_feats + pair_feats

# Hierarchical group aggregations (mean, std, min, max, relative ratios)
agg_numeric_cols = [c for c in num_feats if c not in [id_col, "fold"]][:8]
for grp_col in cat_feats[:6]:
    for num_col in agg_numeric_cols:
        grp_stats = X_train_geo.groupby(grp_col)[num_col].agg(["mean", "std", "min", "max"]).reset_index()
        grp_stats.columns = [grp_col, f"{grp_col}_{num_col}_mean", f"{grp_col}_{num_col}_std", f"{grp_col}_{num_col}_min", f"{grp_col}_{num_col}_max"]

        X_train_geo = X_train_geo.merge(grp_stats, on=grp_col, how="left")
        X_test_geo = X_test_geo.merge(grp_stats, on=grp_col, how="left")

        # Relative ratios / differences to group mean
        X_train_geo[f"{num_col}_diff_{grp_col}_mean"] = X_train_geo[num_col] - X_train_geo[f"{grp_col}_{num_col}_mean"]
        X_test_geo[f"{num_col}_diff_{grp_col}_mean"] = X_test_geo[num_col] - X_test_geo[f"{grp_col}_{num_col}_mean"]

# Frequency Encoding
for col in all_cat_cols:
    freq = X_train_geo[col].value_counts(normalize=True).to_dict()
    X_train_geo[f"{col}_freq"] = X_train_geo[col].map(freq).fillna(0).astype(np.float32)
    X_test_geo[f"{col}_freq"] = X_test_geo[col].map(freq).fillna(0).astype(np.float32)

# Ordinal Encoding
for col in all_cat_cols:
    X_train_geo[col] = X_train_geo[col].astype(str).fillna("__MISSING__")
    X_test_geo[col] = X_test_geo[col].astype(str).fillna("__MISSING__")

    oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X_train_geo[f"{col}_encoded"] = oe.fit_transform(X_train_geo[[col]]).astype(
        np.int32
    )
    X_test_geo[f"{col}_encoded"] = oe.transform(X_test_geo[[col]]).astype(np.int32)

# Missingness indicators
for col in num_feats:
    if col in X_train_geo.columns and X_train_geo[col].isnull().sum() > 0:
        X_train_geo[f"{col}_isna"] = X_train_geo[col].isnull().astype(np.float32)
        X_test_geo[f"{col}_isna"] = X_test_geo[col].isnull().astype(np.float32)

# =========================================================================
# 4. Out-of-Fold Target Encoding (Strictly Leak-Free on Singles & Pairs)
# =========================================================================
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
train_folds = np.zeros(len(X_train_geo), dtype=int)

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_geo, y_train)):
    train_folds[val_idx] = fold

X_train_geo["fold"] = train_folds

target_global_mean = float(y_train.mean())
smoothing = 10.0

for col in all_cat_cols:
    oof_target_enc = np.zeros(len(X_train_geo), dtype=np.float32)

    for fold in range(n_splits):
        idx_tr = np.where(train_folds != fold)[0]
        idx_val = np.where(train_folds == fold)[0]

        tr_cat = X_train_geo.iloc[idx_tr][col]
        tr_y = y_train.iloc[idx_tr]

        stats = tr_y.groupby(tr_cat).agg(["count", "mean"])
        stats["smooth"] = (
            stats["count"] * stats["mean"] + smoothing * target_global_mean
        ) / (stats["count"] + smoothing)
        encoding_dict = stats["smooth"].to_dict()

        oof_target_enc[idx_val] = (
            X_train_geo.iloc[idx_val][col]
            .map(encoding_dict)
            .fillna(target_global_mean)
            .values
        )

    X_train_geo[f"{col}_target_enc"] = oof_target_enc

    full_stats = y_train.groupby(X_train_geo[col]).agg(["count", "mean"])
    full_stats["smooth"] = (
        full_stats["count"] * full_stats["mean"] + smoothing * target_global_mean
    ) / (full_stats["count"] + smoothing)
    full_encoding = full_stats["smooth"].to_dict()
    X_test_geo[f"{col}_target_enc"] = (
        X_test_geo[col].map(full_encoding).fillna(target_global_mean).astype(np.float32)
    )

# Drop raw categorical strings
X_train_processed = X_train_geo.drop(columns=all_cat_cols).copy()
X_test_processed = X_test_geo.drop(columns=all_cat_cols).copy()


# =========================================================================
# 5. Model Architectures and Hyperparameter Configurations
# =========================================================================
class TabularResidualBlock(nn.Module):

    def __init__(self, hidden_dim: int, dropout_rate: float = 0.15):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class TabularResNet(nn.Module):

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = [256, 128, 64],
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, hidden_dims[0]),
            nn.SiLU(),
            nn.Dropout(p=dropout_rate),
        )
        self.res_block1 = TabularResidualBlock(
            hidden_dims[0], dropout_rate=dropout_rate
        )
        self.transition1 = nn.Sequential(
            nn.BatchNorm1d(hidden_dims[0]),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.SiLU(),
            nn.Dropout(p=dropout_rate),
        )
        self.res_block2 = TabularResidualBlock(
            hidden_dims[1], dropout_rate=dropout_rate
        )
        self.transition2 = nn.Sequential(
            nn.BatchNorm1d(hidden_dims[1]),
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.SiLU(),
            nn.Dropout(p=dropout_rate),
        )
        self.head = nn.Linear(hidden_dims[2], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(x)
        x = self.res_block1(x)
        x = self.transition1(x)
        x = self.res_block2(x)
        x = self.transition2(x)
        out = self.head(x)
        return out.squeeze(-1)


def get_lightgbm_params(seed: int = 42) -> dict:
    return {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "learning_rate": 0.03,
        "num_leaves": 63,
        "max_depth": 8,
        "min_child_samples": 20,
        "subsample": 0.85,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": seed,
        "n_estimators": 2500,
        "n_jobs": -1,
        "verbose": -1,
    }


def get_xgboost_params(seed: int = 42) -> dict:
    return {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "booster": "gbtree",
        "learning_rate": 0.03,
        "max_depth": 7,
        "min_child_weight": 3,
        "subsample": 0.85,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": seed,
        "n_estimators": 2500,
        "n_jobs": -1,
        "tree_method": "hist",
    }


def compute_rmse(y_true, y_pred):
    return root_mean_squared_error(y_true, y_pred)


# =========================================================================
# 6. Cross-Validation Training & Inference
# =========================================================================
exclude_cols = [target_col, "fold"]
if id_col is not None and id_col in X_train_processed.columns:
    exclude_cols.append(id_col)

feature_cols = [c for c in X_train_processed.columns if c not in exclude_cols]
n_features = len(feature_cols)
n_train = len(X_train_processed)
n_test = len(X_test_processed)

print(f"Training Setup: {n_features} features, {n_splits} folds, Device: {device}")

oof_lgb = np.zeros(n_train, dtype=np.float32)
oof_xgb = np.zeros(n_train, dtype=np.float32)
oof_nn = np.zeros(n_train, dtype=np.float32)

test_preds_lgb = np.zeros(n_test, dtype=np.float32)
test_preds_xgb = np.zeros(n_test, dtype=np.float32)
test_preds_nn = np.zeros(n_test, dtype=np.float32)

X_test_mat = X_test_processed[feature_cols].fillna(0).values.astype(np.float32)

for fold in range(n_splits):
    idx_tr = np.where(X_train_processed["fold"] != fold)[0]
    idx_val = np.where(X_train_processed["fold"] == fold)[0]

    X_tr_df = X_train_processed.iloc[idx_tr][feature_cols].fillna(0)
    y_tr = y_train.iloc[idx_tr].values.astype(np.float32)

    X_val_df = X_train_processed.iloc[idx_val][feature_cols].fillna(0)
    y_val = y_train.iloc[idx_val].values.astype(np.float32)

    X_tr_vals = X_tr_df.values.astype(np.float32)
    X_val_vals = X_val_df.values.astype(np.float32)

    # 1. Train LightGBM
    lgb_train_data = lgb.Dataset(X_tr_vals, label=y_tr)
    lgb_val_data = lgb.Dataset(X_val_vals, label=y_val, reference=lgb_train_data)
    lgb_params = get_lightgbm_params(seed=SEED + fold)

    lgb_model = lgb.train(
        lgb_params,
        lgb_train_data,
        valid_sets=[lgb_val_data],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )

    val_pred_lgb = lgb_model.predict(X_val_vals, num_iteration=lgb_model.best_iteration)
    oof_lgb[idx_val] = val_pred_lgb
    test_preds_lgb += (
        lgb_model.predict(X_test_mat, num_iteration=lgb_model.best_iteration) / n_splits
    )

    # 2. Train XGBoost
    dtrain = xgb.DMatrix(X_tr_vals, label=y_tr)
    dval = xgb.DMatrix(X_val_vals, label=y_val)
    dtest = xgb.DMatrix(X_test_mat)
    xgb_params = get_xgboost_params(seed=SEED + fold)

    xgb_model = xgb.train(
        xgb_params,
        dtrain,
        num_boost_round=2500,
        evals=[(dval, "val")],
        early_stopping_rounds=50,
        verbose_eval=False,
    )

    val_pred_xgb = xgb_model.predict(
        dval, iteration_range=(0, xgb_model.best_iteration + 1)
    )
    oof_xgb[idx_val] = val_pred_xgb
    test_preds_xgb += (
        xgb_model.predict(dtest, iteration_range=(0, xgb_model.best_iteration + 1))
        / n_splits
    )

    # 3. Train TabularResNet
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr_vals)
    X_val_scaled = scaler.transform(X_val_vals)
    X_test_scaled = scaler.transform(X_test_mat)

    train_dataset = TensorDataset(
        torch.tensor(X_tr_scaled, dtype=torch.float32),
        torch.tensor(y_tr, dtype=torch.float32),
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val_scaled, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32),
    )

    train_loader = DataLoader(
        train_dataset, batch_size=256, shuffle=True, drop_last=False
    )
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)

    nn_model = TabularResNet(
        input_dim=n_features, hidden_dims=[256, 128, 64], dropout_rate=0.15
    ).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        nn_model.parameters(), lr=2e-3, weight_decay=1e-4, eps=1e-8
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=25, eta_min=1e-6
    )

    best_val_loss = float("inf")
    best_nn_weights = None

    for epoch in range(25):
        nn_model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            out = nn_model(batch_x)
            loss = criterion(out, batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(nn_model.parameters(), max_norm=2.0)
            optimizer.step()
        scheduler.step()

        nn_model.eval()
        val_losses = []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                val_out = nn_model(batch_x)
                val_losses.append(criterion(val_out, batch_y).item())
        epoch_val_loss = np.mean(val_losses)

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_nn_weights = {k: v.cpu() for k, v in nn_model.state_dict().items()}

    nn_model.load_state_dict({k: v.to(device) for k, v in best_nn_weights.items()})
    nn_model.eval()

    with torch.no_grad():
        val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32).to(device)
        val_pred_nn = nn_model(val_tensor).cpu().numpy()
        oof_nn[idx_val] = val_pred_nn

        test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
        test_preds_nn += nn_model(test_tensor).cpu().numpy() / n_splits

    fold_lgb_rmse = compute_rmse(y_val, val_pred_lgb)
    fold_xgb_rmse = compute_rmse(y_val, val_pred_xgb)
    fold_nn_rmse = compute_rmse(y_val, val_pred_nn)
    print(
        f"Fold {fold+1}/{n_splits} | LGB RMSE: {fold_lgb_rmse:.4f} | XGB RMSE: {fold_xgb_rmse:.4f} | NN RMSE: {fold_nn_rmse:.4f}"
    )

    del lgb_model, xgb_model, nn_model, train_loader, val_loader
    gc.collect()

# =========================================================================
# 7. Ensemble Blending Optimization
# =========================================================================
oof_matrix = np.column_stack([oof_lgb, oof_xgb, oof_nn])
test_matrix = np.column_stack([test_preds_lgb, test_preds_xgb, test_preds_nn])


def blend_objective(weights):
    w = np.array(weights)
    pred = oof_matrix @ w
    return compute_rmse(y_train.values, pred)


initial_weights = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
bounds = [(0.0, 1.0) for _ in range(3)]
constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

opt_res = minimize(
    blend_objective,
    initial_weights,
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
    options={"maxiter": 500, "ftol": 1e-7},
)

opt_weights = opt_res.x if opt_res.success else np.array(initial_weights)
print(
    f"Optimal Weights: LGB={opt_weights[0]:.4f}, XGB={opt_weights[1]:.4f}, NN={opt_weights[2]:.4f}"
)

oof_ensemble = oof_matrix @ opt_weights
final_test_predictions = test_matrix @ opt_weights

# =========================================================================
# 8. Generate Submission & Final Validation Output
# =========================================================================
sub_df = pd.DataFrame(
    {
        "id": np.arange(len(final_test_predictions), dtype=int),
        "SupStruc": final_test_predictions,
    }
)
submission_path = "./submission/submission.csv"
sub_df.to_csv(submission_path, index=False)
print(f"Saved submission to {submission_path} with shape {sub_df.shape}")

final_score = compute_rmse(y_train.values, oof_ensemble)
print(f"Final Validation Score: {final_score}")