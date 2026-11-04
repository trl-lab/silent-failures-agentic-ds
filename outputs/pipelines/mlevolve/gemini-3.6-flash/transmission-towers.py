import json
import os
import warnings
import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb

warnings.filterwarnings("ignore")

os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# Set seeds for complete reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================================================================
# STEP 1: DATA LOADING AND FEATURE ENGINEERING
# ==============================================================================
print("Loading raw datasets...")
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "SupStruc"
if target_col not in train_df.columns:
    possible_targets = [
        c for c in train_df.columns if "sup" in c.lower() or "struc" in c.lower()
    ]
    if possible_targets:
        target_col = possible_targets[0]

lat_col, lon_col = None, None
for col in train_df.columns:
    c_lower = col.lower()
    if lat_col is None and c_lower in ["latitude", "lat", "y", "y_coord", "lat_deg"]:
        lat_col = col
    if lon_col is None and c_lower in [
        "longitude",
        "lon",
        "long",
        "x",
        "x_coord",
        "lon_deg",
    ]:
        lon_col = col

if lat_col is None or lon_col is None:
    for col in train_df.columns:
        if lat_col is None and "lat" in col.lower():
            lat_col = col
        if lon_col is None and ("lon" in col.lower() or "lng" in col.lower()):
            lon_col = col


def extract_spatial_features(df, reference_df=None):
    if reference_df is None:
        reference_df = df

    features = pd.DataFrame(index=df.index)

    if lat_col and lon_col:
        lat = np.radians(df[lat_col].fillna(df[lat_col].median()).values)
        lon = np.radians(df[lon_col].fillna(df[lon_col].median()).values)
        ref_lat = np.radians(
            reference_df[lat_col].fillna(reference_df[lat_col].median()).values
        )
        ref_lon = np.radians(
            reference_df[lon_col].fillna(reference_df[lon_col].median()).values
        )

        r = 6371.0
        x = r * np.cos(lat) * np.cos(lon)
        y = r * np.cos(lat) * np.sin(lon)
        z = r * np.sin(lat)

        ref_x = r * np.cos(ref_lat) * np.cos(ref_lon)
        ref_y = r * np.cos(ref_lat) * np.sin(ref_lon)
        ref_z = r * np.sin(ref_lat)

        features["coord_x"] = x
        features["coord_y"] = y
        features["coord_z"] = z

        tree = cKDTree(np.column_stack([ref_x, ref_y, ref_z]))

        k_list = [2, 4, 8, 16, 32]
        distances, indices = tree.query(np.column_stack([x, y, z]), k=max(k_list) + 1)

        for k in k_list:
            k_dist = distances[:, 1:k]
            features[f"knn_dist_mean_{k}"] = np.mean(k_dist, axis=1)
            features[f"knn_dist_std_{k}"] = np.std(k_dist, axis=1)
            features[f"knn_dist_min_{k}"] = np.min(k_dist, axis=1)
            features[f"knn_dist_max_{k}"] = np.max(k_dist, axis=1)

        for radius in [1.0, 5.0, 20.0, 50.0]:
            count = tree.query_ball_point(
                np.column_stack([x, y, z]), r=radius, return_length=True
            )
            features[f"density_r_{radius}km"] = count

        nn_idx = indices[:, 1]
        nn_lat = ref_lat[nn_idx]
        nn_lon = ref_lon[nn_idx]

        dlon = nn_lon - lon
        bearing_y = np.sin(dlon) * np.cos(nn_lat)
        bearing_x = np.cos(lat) * np.sin(nn_lat) - np.sin(lat) * np.cos(
            nn_lat
        ) * np.cos(dlon)
        bearing = np.arctan2(bearing_y, bearing_x)

        features["nearest_bearing_sin"] = np.sin(bearing)
        features["nearest_bearing_cos"] = np.cos(bearing)

        for scale in [0.01, 0.05, 0.2, 1.0]:
            lat_grid = (df[lat_col] / scale).round().astype(str)
            lon_grid = (df[lon_col] / scale).round().astype(str)
            features[f"spatial_grid_{scale}"] = lat_grid + "_" + lon_grid

    return features


print("Extracting spatial & topological features...")
train_spatial = extract_spatial_features(train_df, reference_df=train_df)
test_spatial = extract_spatial_features(test_df, reference_df=train_df)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
train_df["fold"] = -1
for fold, (trn_idx, val_idx) in enumerate(kf.split(train_df)):
    train_df.loc[val_idx, "fold"] = fold

ignore_cols = [target_col, "fold", "id", "ID", "index"]
feature_cols = [c for c in train_df.columns if c not in ignore_cols]

cat_cols, num_cols = [], []
for c in feature_cols:
    if (
        train_df[c].dtype == "object"
        or train_df[c].dtype.name == "category"
        or train_df[c].nunique() < 15
    ):
        cat_cols.append(c)
    else:
        num_cols.append(c)

spatial_grid_cols = [c for c in train_spatial.columns if c.startswith("spatial_grid_")]
spatial_num_cols = [
    c for c in train_spatial.columns if not c.startswith("spatial_grid_")
]

cat_cols.extend(spatial_grid_cols)
num_cols.extend(spatial_num_cols)

train_processed = pd.DataFrame(index=train_df.index)
test_processed = pd.DataFrame(index=test_df.index)

train_processed["fold"] = train_df["fold"]
train_processed[target_col] = train_df[target_col]

for c in num_cols:
    if c in train_df.columns:
        val_trn = train_df[c]
        val_tst = test_df[c]
    else:
        val_trn = train_spatial[c]
        val_tst = test_spatial[c]

    med = val_trn.median()
    train_processed[c] = val_trn.fillna(med)
    test_processed[c] = val_tst.fillna(med)

    if (
        train_processed[c].skew() > 3.0
        and (train_processed[c] >= 0).all()
        and (test_processed[c] >= 0).all()
    ):
        train_processed[f"{c}_log1p"] = np.log1p(train_processed[c])
        test_processed[f"{c}_log1p"] = np.log1p(test_processed[c])

for c in cat_cols:
    if c in train_df.columns:
        val_trn = train_df[c].astype(str).fillna("MISSING")
        val_tst = test_df[c].astype(str).fillna("MISSING")
    else:
        val_trn = train_spatial[c].astype(str).fillna("MISSING")
        val_tst = test_spatial[c].astype(str).fillna("MISSING")

    enc = OrdinalEncoder(
        handle_unknown="use_encoded_value", unknown_value=-1, dtype=np.int32
    )
    train_processed[f"{c}_ordinal"] = enc.fit_transform(
        val_trn.values.reshape(-1, 1)
    ).ravel()
    test_processed[f"{c}_ordinal"] = enc.transform(
        val_tst.values.reshape(-1, 1)
    ).ravel()

    freq = val_trn.value_counts(normalize=True).to_dict()
    train_processed[f"{c}_freq"] = val_trn.map(freq).fillna(0).astype(np.float32)
    test_processed[f"{c}_freq"] = val_tst.map(freq).fillna(0).astype(np.float32)

# Out-Of-Fold target encodings strictly on train folds
global_target_mean = train_df[target_col].mean()

for c in cat_cols:
    if c in train_df.columns:
        s_trn = train_df[c].astype(str).fillna("MISSING")
        s_tst = test_df[c].astype(str).fillna("MISSING")
    else:
        s_trn = train_spatial[c].astype(str).fillna("MISSING")
        s_tst = test_spatial[c].astype(str).fillna("MISSING")

    oof_col = f"{c}_target_enc"
    train_processed[oof_col] = np.nan

    for fold in range(5):
        val_mask = train_df["fold"] == fold
        trn_mask = ~val_mask

        stats = (
            train_df[trn_mask]
            .groupby(s_trn[trn_mask])[target_col]
            .agg(["mean", "count"])
        )
        smooth = 10
        smooth_target = (
            stats["count"] * stats["mean"] + smooth * global_target_mean
        ) / (stats["count"] + smooth)
        smooth_map = smooth_target.to_dict()

        train_processed.loc[val_mask, oof_col] = (
            s_trn[val_mask].map(smooth_map).fillna(global_target_mean)
        )

    full_stats = train_df.groupby(s_trn)[target_col].agg(["mean", "count"])
    full_smooth_target = (
        full_stats["count"] * full_stats["mean"] + smooth * global_target_mean
    ) / (full_stats["count"] + smooth)
    test_processed[oof_col] = s_tst.map(full_smooth_target.to_dict()).fillna(
        global_target_mean
    )

final_feature_cols = [
    c for c in train_processed.columns if c not in [target_col, "fold"]
]

# Add domain interaction features
if "knn_dist_mean_2" in final_feature_cols and "knn_dist_mean_32" in final_feature_cols:
    train_processed["knn_ratio_2_32"] = train_processed["knn_dist_mean_2"] / (
        train_processed["knn_dist_mean_32"] + 1e-5
    )
    test_processed["knn_ratio_2_32"] = test_processed["knn_dist_mean_2"] / (
        test_processed["knn_dist_mean_32"] + 1e-5
    )
    if "knn_ratio_2_32" not in final_feature_cols:
        final_feature_cols.append("knn_ratio_2_32")

if "density_r_1.0km" in final_feature_cols and "density_r_50.0km" in final_feature_cols:
    train_processed["density_ratio"] = (train_processed["density_r_1.0km"] + 1.0) / (
        train_processed["density_r_50.0km"] + 1.0
    )
    test_processed["density_ratio"] = (test_processed["density_r_1.0km"] + 1.0) / (
        test_processed["density_r_50.0km"] + 1.0
    )
    if "density_ratio" not in final_feature_cols:
        final_feature_cols.append("density_ratio")

# Clean inf/nan values
for c in final_feature_cols:
    med = train_processed[c].median() if train_processed[c].notna().any() else 0.0
    train_processed[c] = (
        train_processed[c].replace([np.inf, -np.inf], np.nan).fillna(med)
    )
    test_processed[c] = test_processed[c].replace([np.inf, -np.inf], np.nan).fillna(med)

y = train_processed[target_col].values
X = train_processed[final_feature_cols].values
X_test = test_processed[final_feature_cols].values

n_train = len(train_processed)
n_test = len(test_processed)
n_folds = 5


# ==============================================================================
# STEP 2: MODEL ARCHITECTURES & TRAINING
# ==============================================================================
class TabularResNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, 256), nn.LayerNorm(256), nn.GELU()
        )
        self.b1_fc1 = nn.Linear(256, 256)
        self.b1_norm1 = nn.LayerNorm(256)
        self.b1_act = nn.GELU()
        self.b1_drop = nn.Dropout(0.2)
        self.b1_fc2 = nn.Linear(256, 256)
        self.b1_norm2 = nn.LayerNorm(256)

        self.b2_fc1 = nn.Linear(256, 256)
        self.b2_norm1 = nn.LayerNorm(256)
        self.b2_act = nn.GELU()
        self.b2_drop = nn.Dropout(0.2)
        self.b2_fc2 = nn.Linear(256, 256)
        self.b2_norm2 = nn.LayerNorm(256)

        self.head = nn.Sequential(
            nn.GELU(), nn.Dropout(0.1), nn.Linear(256, 64), nn.GELU(), nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.input_layer(x)
        res1 = x
        out1 = self.b1_drop(self.b1_act(self.b1_norm1(self.b1_fc1(x))))
        out1 = self.b1_norm2(self.b1_fc2(out1))
        x = res1 + out1

        res2 = x
        out2 = self.b2_drop(self.b2_act(self.b2_norm1(self.b2_fc1(x))))
        out2 = self.b2_norm2(self.b2_fc2(out2))
        x = res2 + out2

        return self.head(x).squeeze(-1)


oof_lgb = np.zeros(n_train)
oof_xgb = np.zeros(n_train)
oof_cat = np.zeros(n_train)
oof_et = np.zeros(n_train)
oof_nn = np.zeros(n_train)

test_lgb = np.zeros(n_test)
test_xgb = np.zeros(n_test)
test_cat = np.zeros(n_test)
test_et = np.zeros(n_test)
test_nn = np.zeros(n_test)

print("Starting 5-fold cross validation for all 5 model architectures...")

for fold in range(n_folds):
    val_mask = train_processed["fold"].values == fold
    trn_mask = ~val_mask

    X_trn, y_trn = X[trn_mask], y[trn_mask]
    X_val, y_val = X[val_mask], y[val_mask]

    # 1. LightGBM
    model_lgb = lgb.LGBMRegressor(
        n_estimators=1200,
        learning_rate=0.03,
        num_leaves=45,
        subsample=0.75,
        colsample_bytree=0.75,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42 + fold,
        n_jobs=-1,
    )
    model_lgb.fit(
        X_trn,
        y_trn,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )
    oof_lgb[val_mask] = model_lgb.predict(X_val)
    test_lgb += model_lgb.predict(X_test) / n_folds

    # 2. XGBoost
    model_xgb = xgb.XGBRegressor(
        n_estimators=1200,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=0.1,
        reg_lambda=1.0,
        tree_method="hist",
        early_stopping_rounds=50,
        random_state=42 + fold,
        n_jobs=-1,
    )
    model_xgb.fit(X_trn, y_trn, eval_set=[(X_val, y_val)], verbose=False)
    oof_xgb[val_mask] = model_xgb.predict(X_val)
    test_xgb += model_xgb.predict(X_test) / n_folds

    # 3. CatBoost
    model_cat = cb.CatBoostRegressor(
        iterations=1000,
        learning_rate=0.035,
        depth=6,
        l2_leaf_reg=4.0,
        random_seed=42 + fold,
        early_stopping_rounds=50,
        verbose=0,
        thread_count=-1,
    )
    model_cat.fit(X_trn, y_trn, eval_set=(X_val, y_val), verbose=False)
    oof_cat[val_mask] = model_cat.predict(X_val)
    test_cat += model_cat.predict(X_test) / n_folds

    # 4. ExtraTrees
    model_et = ExtraTreesRegressor(
        n_estimators=250,
        max_depth=18,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42 + fold,
        n_jobs=-1,
    )
    model_et.fit(X_trn, y_trn)
    oof_et[val_mask] = model_et.predict(X_val)
    test_et += model_et.predict(X_test) / n_folds

    # 5. PyTorch Neural ResNet
    scaler = StandardScaler()
    X_trn_scaled = scaler.fit_transform(X_trn)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    trn_ds = TensorDataset(
        torch.tensor(X_trn_scaled, dtype=torch.float32),
        torch.tensor(y_trn, dtype=torch.float32),
    )
    trn_loader = DataLoader(trn_ds, batch_size=256, shuffle=True)

    model_nn = TabularResNet(input_dim=len(final_feature_cols)).to(device)
    optimizer = torch.optim.AdamW(model_nn.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25)
    criterion = nn.MSELoss()

    model_nn.train()
    for epoch in range(25):
        for bx, by in trn_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(model_nn(bx), by)
            loss.backward()
            optimizer.step()
        scheduler.step()

    model_nn.eval()
    with torch.no_grad():
        v_tensor = torch.tensor(X_val_scaled, dtype=torch.float32).to(device)
        oof_nn[val_mask] = model_nn(v_tensor).cpu().numpy()

        t_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
        test_nn += model_nn(t_tensor).cpu().numpy() / n_folds

    score_lgb = np.sqrt(mean_squared_error(y_val, oof_lgb[val_mask]))
    score_xgb = np.sqrt(mean_squared_error(y_val, oof_xgb[val_mask]))
    score_cat = np.sqrt(mean_squared_error(y_val, oof_cat[val_mask]))
    score_et = np.sqrt(mean_squared_error(y_val, oof_et[val_mask]))
    score_nn = np.sqrt(mean_squared_error(y_val, oof_nn[val_mask]))

    print(
        f"Fold {fold+1}/{n_folds} - LGB: {score_lgb:.4f} | XGB: {score_xgb:.4f} | Cat: {score_cat:.4f} | ET: {score_et:.4f} | NN: {score_nn:.4f}"
    )

# ==============================================================================
# STEP 3: ENSEMBLE BLENDING & FINAL SUBMISSION
# ==============================================================================
oof_matrix = np.column_stack([oof_lgb, oof_xgb, oof_cat, oof_et, oof_nn])
test_matrix = np.column_stack([test_lgb, test_xgb, test_cat, test_et, test_nn])


def blend_loss(weights, oof_preds, y_true):
    pred = np.dot(oof_preds, weights)
    return np.sqrt(mean_squared_error(y_true, pred))


num_models = oof_matrix.shape[1]
init_weights = np.ones(num_models) / num_models
bounds = [(0, 1) for _ in range(num_models)]
constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

opt_res = minimize(
    blend_loss,
    init_weights,
    args=(oof_matrix, y),
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
)
opt_weights = opt_res.x

oof_blend = np.dot(oof_matrix, opt_weights)
test_blend = np.dot(test_matrix, opt_weights)

min_y, max_y = y.min(), y.max()
test_blend = np.clip(test_blend, min_y, max_y)

final_score = np.sqrt(mean_squared_error(y, oof_blend))

submission_df = pd.DataFrame({"id": np.arange(len(test_blend)), "SupStruc": test_blend})
submission_df.to_csv("./submission/submission.csv", index=False)

print(f"Final Validation Score: {final_score}")
