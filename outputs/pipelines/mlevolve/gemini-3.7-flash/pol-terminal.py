import math
import os
import random
import re
import warnings
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb

warnings.filterwarnings("ignore")

# ==============================================================================
# 0. Global Setup & Seed Everything
# ==============================================================================
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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("./submission", exist_ok=True)
os.makedirs("./working", exist_ok=True)

# ==============================================================================
# 1. Load Datasets & Target Identification
# ==============================================================================
print("Loading train and test datasets...")
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

TARGET_COL = "CAPACITY"
if TARGET_COL not in train_df.columns:
    for c in train_df.columns:
        if c.upper() == "CAPACITY":
            train_df = train_df.rename(columns={c: TARGET_COL})
            break

# Identify ID column if present
id_col = None
for col in ["id", "ID", "Id", "index", "INDEX"]:
    if col in train_df.columns and col in test_df.columns:
        id_col = col
        break

train_ids = train_df[id_col].values if id_col is not None else np.arange(len(train_df))
test_ids = test_df[id_col].values if id_col is not None else np.arange(len(test_df))

y_train_raw = train_df[TARGET_COL].values.astype(np.float64)
all_cols = [c for c in train_df.columns if c != TARGET_COL and c != id_col]

# Coordinate detection
lat_col = None
lon_col = None
for c in all_cols:
    c_lower = c.lower()
    if c_lower in ["latitude", "lat", "y", "lat_deg", "point_y"]:
        lat_col = c
    elif c_lower in ["longitude", "lon", "long", "x", "lon_deg", "point_x"]:
        lon_col = c

HUBS = {
    "cushing_ok": (35.9828, -96.7678),
    "houston_ship_channel": (29.7604, -95.3698),
    "mont_belvieu_tx": (29.8538, -94.8774),
    "corpus_christi_tx": (27.8006, -97.3964),
    "port_arthur_tx": (29.8850, -93.9399),
    "st_james_la": (30.0163, -90.8498),
    "new_york_harbor_nj": (40.6318, -74.2440),
    "chicago_whiting_in": (41.6792, -87.4947),
    "long_beach_ca": (33.7701, -118.1937),
    "bakersfield_ca": (35.3733, -119.0187),
    "philadelphia_pa": (39.9526, -75.1652),
    "clove_nm": (32.8466, -103.2084),
}


def haversine_np(lat1, lon1, lat2, lon2):
    r = 6371.0
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    a = (
        np.sin(delta_phi / 2.0) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return r * c


# ==============================================================================
# 2. Strategic Feature Engineering
# ==============================================================================
def engineer_features(df, is_train=True, fit_objects=None):
    df_feat = pd.DataFrame(index=df.index)

    # A. Geospatial Features
    if lat_col and lon_col:
        lat = pd.to_numeric(df[lat_col], errors="coerce").fillna(
            df[lat_col].median() if is_train else fit_objects["lat_median"]
        )
        lon = pd.to_numeric(df[lon_col], errors="coerce").fillna(
            df[lon_col].median() if is_train else fit_objects["lon_median"]
        )

        df_feat["geo_lat"] = lat
        df_feat["geo_lon"] = lon

        rad_lat = np.radians(lat)
        rad_lon = np.radians(lon)
        df_feat["geo_x_3d"] = np.cos(rad_lat) * np.cos(rad_lon)
        df_feat["geo_y_3d"] = np.cos(rad_lat) * np.sin(rad_lon)
        df_feat["geo_z_3d"] = np.sin(rad_lat)

        for hub_name, (h_lat, h_lon) in HUBS.items():
            df_feat[f"dist_to_{hub_name}"] = haversine_np(
                lat.values, lon.values, h_lat, h_lon
            )

        dist_cols = [c for c in df_feat.columns if c.startswith("dist_to_")]
        df_feat["min_dist_to_any_hub"] = df_feat[dist_cols].min(axis=1)
        df_feat["mean_dist_to_hubs"] = df_feat[dist_cols].mean(axis=1)
        df_feat["std_dist_to_hubs"] = df_feat[dist_cols].std(axis=1)

    # B. Text and Domain Keyword Matching
    domain_keywords = [
        "crude",
        "oil",
        "petroleum",
        "gasoline",
        "diesel",
        "jet",
        "fuel",
        "distillate",
        "ethanol",
        "biodiesel",
        "asphalt",
        "ngl",
        "lpg",
        "propane",
        "butane",
        "marine",
        "dock",
        "wharf",
        "barge",
        "tanker",
        "rail",
        "pipeline",
        "rack",
        "terminal",
        "refinery",
        "storage",
        "bulk",
        "transmix",
        "chemical",
        "energy",
        "logistics",
    ]

    string_cols = [
        c for c in all_cols if df[c].dtype == "object" or df[c].dtype == "string"
    ]
    if string_cols:
        text_series = (
            df[string_cols].astype(str).fillna("").agg(" ".join, axis=1).str.lower()
        )

        for kw in domain_keywords:
            df_feat[f"kw_{kw}"] = text_series.apply(lambda s: 1 if kw in s else 0)

        df_feat["kw_total_matches"] = df_feat[
            [f"kw_{kw}" for kw in domain_keywords]
        ].sum(axis=1)
        df_feat["text_char_length"] = text_series.apply(len)
        df_feat["text_word_count"] = text_series.apply(lambda s: len(s.split()))

    # C. Processing Numeric Columns
    num_cols = [
        c
        for c in all_cols
        if pd.api.types.is_numeric_dtype(df[c]) and c not in [lat_col, lon_col]
    ]
    for col in num_cols:
        val = pd.to_numeric(df[col], errors="coerce")
        df_feat[f"num_{col}"] = val
        df_feat[f"num_{col}_isnan"] = val.isna().astype(int)

    # D. Categorical Representation
    cat_cols = [
        c for c in all_cols if c not in num_cols and c not in [lat_col, lon_col]
    ]
    for c in cat_cols:
        s = df[c].astype(str).fillna("__MISSING__")
        df_feat[f"cat_{c}"] = s

    return df_feat


lat_median = (
    pd.to_numeric(train_df[lat_col], errors="coerce").median() if lat_col else 0.0
)
lon_median = (
    pd.to_numeric(train_df[lon_col], errors="coerce").median() if lon_col else 0.0
)
fit_objects = {"lat_median": lat_median, "lon_median": lon_median}

train_raw_feats = engineer_features(train_df, is_train=True, fit_objects=fit_objects)
test_raw_feats = engineer_features(test_df, is_train=False, fit_objects=fit_objects)

# Spatial K-NN Density Features
if lat_col and lon_col:
    train_coords = np.radians(train_raw_feats[["geo_lat", "geo_lon"]].values)
    test_coords = np.radians(test_raw_feats[["geo_lat", "geo_lon"]].values)

    for k in [3, 7, 15]:
        knn_model = NearestNeighbors(
            n_neighbors=k + 1, metric="haversine", algorithm="ball_tree"
        )
        knn_model.fit(train_coords)

        train_dists, _ = knn_model.kneighbors(train_coords)
        train_raw_feats[f"knn_dist_mean_{k}"] = train_dists[:, 1:].mean(axis=1) * 6371.0
        train_raw_feats[f"knn_dist_min_{k}"] = train_dists[:, 1] * 6371.0
        train_raw_feats[f"knn_dist_max_{k}"] = train_dists[:, -1] * 6371.0

        test_dists, _ = knn_model.kneighbors(test_coords)
        test_raw_feats[f"knn_dist_mean_{k}"] = test_dists[:, :k].mean(axis=1) * 6371.0
        test_raw_feats[f"knn_dist_min_{k}"] = test_dists[:, 0] * 6371.0
        test_raw_feats[f"knn_dist_max_{k}"] = test_dists[:, k - 1] * 6371.0

# TF-IDF Features
string_cols = [
    c
    for c in all_cols
    if train_df[c].dtype == "object" or train_df[c].dtype == "string"
]
if string_cols:
    train_text = (
        train_df[string_cols].astype(str).fillna("").agg(" ".join, axis=1).str.lower()
    )
    test_text = (
        test_df[string_cols].astype(str).fillna("").agg(" ".join, axis=1).str.lower()
    )

    tfidf = TfidfVectorizer(
        max_features=40, stop_words="english", ngram_range=(1, 2), min_df=3
    )
    train_tfidf = tfidf.fit_transform(train_text).toarray()
    test_tfidf = tfidf.transform(test_text).toarray()

    tfidf_feature_names = [f"tfidf_{name}" for name in tfidf.get_feature_names_out()]
    for i, name in enumerate(tfidf_feature_names):
        train_raw_feats[name] = train_tfidf[:, i]
        test_raw_feats[name] = test_tfidf[:, i]

# Categorical Frequency and Label Encodings
cat_feature_cols = [c for c in train_raw_feats.columns if c.startswith("cat_")]
for col in cat_feature_cols:
    freq_map = train_raw_feats[col].value_counts().to_dict()
    train_raw_feats[f"{col}_freq"] = train_raw_feats[col].map(freq_map).fillna(1)
    test_raw_feats[f"{col}_freq"] = test_raw_feats[col].map(freq_map).fillna(0)

    le = LabelEncoder()
    all_uniques = list(
        set(train_raw_feats[col].astype(str)).union(
            set(test_raw_feats[col].astype(str))
        )
    )
    le.fit(all_uniques + ["__UNSEEN__"])
    train_raw_feats[f"{col}_le"] = le.transform(train_raw_feats[col].astype(str))
    test_raw_feats[f"{col}_le"] = le.transform(test_raw_feats[col].astype(str))

    train_raw_feats = train_raw_feats.drop(columns=[col])
    test_raw_feats = test_raw_feats.drop(columns=[col])

feature_names = [c for c in train_raw_feats.columns]
for c in feature_names:
    if c not in test_raw_feats.columns:
        test_raw_feats[c] = 0.0

test_raw_feats = test_raw_feats[feature_names]

col_medians = train_raw_feats.median(numeric_only=True).to_dict()
train_processed = train_raw_feats.fillna(col_medians).fillna(0)
test_processed = test_raw_feats.fillna(col_medians).fillna(0)

# Stratified K-Fold Setup
N_SPLITS = 5
log_y = np.log1p(np.clip(y_train_raw, 0, None))
num_bins = min(10, len(np.unique(log_y)))
y_binned = pd.qcut(log_y, q=num_bins, labels=False, duplicates="drop")

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
fold_assignments = np.zeros(len(train_df), dtype=int)

for fold, (train_idx, val_idx) in enumerate(skf.split(train_processed, y_binned)):
    fold_assignments[val_idx] = fold

# Leak-free Target Encodings
for cat_c in [c for c in train_processed.columns if c.endswith("_le")]:
    target_enc_train = np.zeros(len(train_processed), dtype=np.float64)
    target_enc_test = np.zeros(len(test_processed), dtype=np.float64)
    global_mean = y_train_raw.mean()

    for fold in range(N_SPLITS):
        tr_mask = fold_assignments != fold
        val_mask = fold_assignments == fold

        tr_df = pd.DataFrame(
            {
                "cat": train_processed.loc[tr_mask, cat_c],
                "target": y_train_raw[tr_mask],
            }
        )
        cat_stats = tr_df.groupby("cat")["target"].agg(["count", "mean"])
        smoothing = 10.0
        smooth_enc = (
            cat_stats["count"] * cat_stats["mean"] + smoothing * global_mean
        ) / (cat_stats["count"] + smoothing)
        smooth_map = smooth_enc.to_dict()

        target_enc_train[val_mask] = (
            train_processed.loc[val_mask, cat_c].map(smooth_map).fillna(global_mean)
        )
        target_enc_test += (
            test_processed[cat_c].map(smooth_map).fillna(global_mean).values / N_SPLITS
        )

    train_processed[f"{cat_c}_target_enc"] = target_enc_train
    test_processed[f"{cat_c}_target_enc"] = target_enc_test


# ==============================================================================
# 3. Model Architecture Definitions
# ==============================================================================
class GatedResidualBlock(nn.Module):

    def __init__(self, d_in: int, d_out: int, dropout_rate: float = 0.15):
        super().__init__()
        self.linear1 = nn.Linear(d_in, d_out * 2)
        self.norm1 = nn.LayerNorm(d_out * 2)
        self.linear2 = nn.Linear(d_out, d_out)
        self.norm2 = nn.LayerNorm(d_out)
        self.dropout = nn.Dropout(dropout_rate)
        self.shortcut = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        h = self.linear1(x)
        h = self.norm1(h)
        val, gate = torch.chunk(h, 2, dim=-1)
        h = val * torch.sigmoid(gate)
        h = self.dropout(h)
        h = self.linear2(h)
        h = self.norm2(h)
        h = F.gelu(h)
        h = self.dropout(h)
        return residual + h


class TabularGateNet(nn.Module):

    def __init__(
        self,
        num_features: int,
        hidden_dims: list = [256, 128, 64],
        dropout_rate: float = 0.15,
    ):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(num_features, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )
        blocks = []
        for in_dim, out_dim in zip(hidden_dims[:-1], hidden_dims[1:]):
            blocks.append(
                GatedResidualBlock(
                    d_in=in_dim, d_out=out_dim, dropout_rate=dropout_rate
                )
            )
        self.res_blocks = nn.ModuleList(blocks)
        self.head = nn.Sequential(
            nn.Linear(hidden_dims[-1], 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_layer(x)
        for block in self.res_blocks:
            h = block(h)
        out = self.head(h)
        return out.squeeze(-1)


# ==============================================================================
# 4. Cross-Validated Multi-Model Training Loop
# ==============================================================================
feature_cols = [c for c in train_processed.columns]
n_train = len(train_processed)
n_test = len(test_processed)
n_folds = N_SPLITS

oof_lgb = np.zeros(n_train, dtype=np.float64)
test_lgb = np.zeros(n_test, dtype=np.float64)

oof_xgb = np.zeros(n_train, dtype=np.float64)
test_xgb = np.zeros(n_test, dtype=np.float64)

oof_nn = np.zeros(n_train, dtype=np.float64)
test_nn = np.zeros(n_test, dtype=np.float64)

print(
    f"\nTraining on {n_train} samples, predicting on {n_test} test samples ({len(feature_cols)} features)..."
)

for fold in range(n_folds):
    tr_idx = np.where(fold_assignments != fold)[0]
    val_idx = np.where(fold_assignments == fold)[0]

    X_tr = train_processed.iloc[tr_idx][feature_cols].values.astype(np.float32)
    y_tr = y_train_raw[tr_idx]

    X_val = train_processed.iloc[val_idx][feature_cols].values.astype(np.float32)
    y_val = y_train_raw[val_idx]

    X_test_arr = test_processed[feature_cols].values.astype(np.float32)

    # 1. LightGBM
    lgb_model = lgb.LGBMRegressor(
        objective="regression",
        metric="rmse",
        boosting_type="gbdt",
        n_estimators=2000,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=7,
        min_child_samples=15,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=SEED + fold,
        n_jobs=-1,
        verbose=-1,
    )
    lgb_model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(50, verbose=False),
            lgb.log_evaluation(-1),
        ],
    )
    pred_val_lgb = lgb_model.predict(X_val)
    oof_lgb[val_idx] = pred_val_lgb
    test_lgb += lgb_model.predict(X_test_arr) / n_folds

    # 2. XGBoost
    xgb_model = xgb.XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        n_estimators=2000,
        learning_rate=0.03,
        max_depth=6,
        min_child_weight=3,
        subsample=0.85,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.5,
        random_state=SEED + fold,
        n_jobs=-1,
        tree_method="hist",
        early_stopping_rounds=50,
    )
    xgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    pred_val_xgb = xgb_model.predict(X_val)
    oof_xgb[val_idx] = pred_val_xgb
    test_xgb += xgb_model.predict(X_test_arr) / n_folds

    # 3. TabularGateNet
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test_arr)

    y_mean = y_tr.mean()
    y_std = y_tr.std() + 1e-6
    y_tr_norm = (y_tr - y_mean) / y_std

    train_dataset = TensorDataset(
        torch.tensor(X_tr_scaled, dtype=torch.float32),
        torch.tensor(y_tr_norm, dtype=torch.float32),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        drop_last=(len(X_tr) > 64),
        num_workers=0,
    )

    nn_model = TabularGateNet(
        num_features=X_tr_scaled.shape[1],
        hidden_dims=[256, 128, 64],
        dropout_rate=0.15,
    ).to(device)

    optimizer = AdamW(nn_model.parameters(), lr=1e-3, weight_decay=1e-4, eps=1e-8)
    scheduler = CosineAnnealingLR(optimizer, T_max=40, eta_min=1e-5)
    loss_fn = nn.MSELoss()

    best_val_rmse = float("inf")
    best_nn_val_preds = None
    best_nn_test_preds = None
    patience = 12
    patience_counter = 0

    for epoch in range(40):
        nn_model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            pred = nn_model(batch_x)
            loss = loss_fn(pred, batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(nn_model.parameters(), max_norm=2.0)
            optimizer.step()
        scheduler.step()

        nn_model.eval()
        with torch.no_grad():
            x_val_t = torch.tensor(X_val_scaled, dtype=torch.float32).to(device)
            raw_val_pred = (
                (nn_model(x_val_t).cpu().numpy() * y_std + y_mean)
                .astype(np.float64)
                .squeeze()
            )
            val_rmse_epoch = root_mean_squared_error(y_val, raw_val_pred)

            if val_rmse_epoch < best_val_rmse:
                best_val_rmse = val_rmse_epoch
                best_nn_val_preds = raw_val_pred
                x_test_t = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
                best_nn_test_preds = (
                    (nn_model(x_test_t).cpu().numpy() * y_std + y_mean)
                    .astype(np.float64)
                    .squeeze()
                )
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

    oof_nn[val_idx] = best_nn_val_preds
    test_nn += best_nn_test_preds / n_folds

    fold_lgb_score = root_mean_squared_error(y_val, pred_val_lgb)
    fold_xgb_score = root_mean_squared_error(y_val, pred_val_xgb)
    fold_nn_score = root_mean_squared_error(y_val, best_nn_val_preds)
    print(
        f"Fold {fold+1}/{n_folds} - LGB RMSE: {fold_lgb_score:.4f} | XGB RMSE: {fold_xgb_score:.4f} | NN RMSE: {fold_nn_score:.4f}"
    )

# ==============================================================================
# 5. Optimal Ensemble Blending & Evaluation
# ==============================================================================
lgb_total_rmse = root_mean_squared_error(y_train_raw, oof_lgb)
xgb_total_rmse = root_mean_squared_error(y_train_raw, oof_xgb)
nn_total_rmse = root_mean_squared_error(y_train_raw, oof_nn)
print(
    f"\nOOF Summary -> LightGBM: {lgb_total_rmse:.4f}, XGBoost: {xgb_total_rmse:.4f}, TabularGateNet: {nn_total_rmse:.4f}"
)


def blend_objective(weights):
    w1, w2, w3 = weights
    pred = w1 * oof_lgb + w2 * oof_xgb + w3 * oof_nn
    return root_mean_squared_error(y_train_raw, np.clip(pred, 0.0, None))


init_weights = [0.4, 0.4, 0.2]
bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]
constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

opt_res = minimize(
    blend_objective,
    init_weights,
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
)
w_lgb, w_xgb, w_nn = opt_res.x
print(
    f"Optimal Blend Weights -> LightGBM: {w_lgb:.4f}, XGBoost: {w_xgb:.4f}, TabularGateNet: {w_nn:.4f}"
)

final_oof_preds = np.clip(w_lgb * oof_lgb + w_xgb * oof_xgb + w_nn * oof_nn, 0.0, None)
final_test_preds = np.clip(
    w_lgb * test_lgb + w_xgb * test_xgb + w_nn * test_nn, 0.0, None
)

final_score = root_mean_squared_error(y_train_raw, final_oof_preds)

# ==============================================================================
# 6. Generate Submission
# ==============================================================================
sub_df = pd.DataFrame(
    {"id": test_ids.astype(int), "CAPACITY": final_test_preds.astype(np.float64)}
)
sub_path = "./submission/submission.csv"
sub_df.to_csv(sub_path, index=False)

assert sub_df.shape[0] == len(
    test_df
), f"Expected {len(test_df)} rows, found {len(sub_df)}"
assert list(sub_df.columns) == [
    "id",
    "CAPACITY",
], f"Incorrect columns: {sub_df.columns}"
assert not sub_df.isna().any().any(), "Submission contains NaN values!"
print(f"Saved submission to {sub_path} with {len(sub_df)} rows.")

print(f"Final Validation Score: {final_score}")
