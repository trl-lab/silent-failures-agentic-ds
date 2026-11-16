import os
import random
import re
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.sparse import hstack
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.ensemble import ExtraTreesRegressor
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor


# ---------------------------------------------------------
# 0. Global Configurations and Seed Setting
# ---------------------------------------------------------
def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


seed_everything(42)
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)


# ---------------------------------------------------------
# 1. Load Datasets and Identify Column Types
# ---------------------------------------------------------
train_df = pd.read_parquet("./input/train.parquet").reset_index(drop=True)
test_df = pd.read_parquet("./input/test.parquet").reset_index(drop=True)

TARGET_COL = "number_of_officers"
if TARGET_COL not in train_df.columns:
    for c in train_df.columns:
        if c.lower() == "number_of_officers":
            TARGET_COL = c
            break

ID_COL = None
for candidate in ["id", "ID", "Id", "OBJECTID", "objectid", "index"]:
    if candidate in test_df.columns:
        ID_COL = candidate
        break

if ID_COL is not None:
    test_ids = test_df[ID_COL].values
else:
    test_ids = np.arange(len(test_df), dtype=int)

cols = [c for c in train_df.columns if c != TARGET_COL]

lat_col = None
lon_col = None
for c in cols:
    c_lower = c.lower()
    if c_lower in ["latitude", "lat", "y", "lat_deg", "y_coord"]:
        lat_col = c
    elif c_lower in ["longitude", "lon", "long", "x", "lon_deg", "x_coord"]:
        lon_col = c

name_cols = [
    c for c in cols if any(k in c.lower() for k in ["name", "agency", "dept", "title"])
]
cat_cols = [
    c
    for c in cols
    if train_df[c].dtype == "object"
    or c.lower()
    in ["type", "state", "county", "countyfips", "status", "level", "zip", "source"]
]
num_cols = [c for c in cols if c not in cat_cols and c != TARGET_COL and c != ID_COL]


# ---------------------------------------------------------
# 2. Feature Engineering Helper Functions
# ---------------------------------------------------------
def extract_law_enforcement_keywords(series: pd.Series) -> pd.DataFrame:
    patterns = {
        "kw_police": r"\bpolice\b|\bpd\b",
        "kw_sheriff": r"\bsheriff\b|\bso\b|\bdeputy\b",
        "kw_marshal": r"\bmarshal\b",
        "kw_constable": r"\bconstable\b",
        "kw_patrol": r"\bpatrol\b|\btroop\b|\bhyp\b|\bsp\b",
        "kw_county": r"\bcounty\b|\bparish\b",
        "kw_dept": r"\bdept\b|\bdepartment\b|\bdivision\b|\bbureau\b",
        "kw_university": r"\buniversity\b|\bcollege\b|\bcampus\b|\bschool\b|\bisd\b",
        "kw_transit": r"\btransit\b|\bport\b|\bairport\b|\brailroad\b|\bbridge\b|\btpk\b",
        "kw_tribal": r"\btribal\b|\btribe\b|\bnation\b|\bindian\b|\breservation\b",
        "kw_state": r"\bstate\b|\bstatewide\b|\bcommonwealth\b",
        "kw_detention": r"\bdetention\b|\bcorrection\b|\bjail\b|\bprison\b",
        "kw_hospital": r"\bhospital\b|\bmedical\b|\bhealth\b",
        "kw_park": r"\bpark\b|\bconservation\b|\bwildlife\b|\bforest\b",
    }
    extracted = pd.DataFrame(index=series.index)
    clean_text = series.fillna("").astype(str).str.lower()
    for feat_name, pattern in patterns.items():
        extracted[feat_name] = clean_text.str.contains(pattern, regex=True).astype(
            np.int8
        )
    return extracted


def compute_spatial_features(df: pd.DataFrame, lat_c: str, lon_c: str) -> pd.DataFrame:
    feats = pd.DataFrame(index=df.index)
    if lat_c is not None and lon_c is not None:
        lat = pd.to_numeric(df[lat_c], errors="coerce").fillna(
            df[lat_c].median() if df[lat_c].notnull().any() else 37.0902
        )
        lon = pd.to_numeric(df[lon_c], errors="coerce").fillna(
            df[lon_c].median() if df[lon_c].notnull().any() else -95.7129
        )

        rad_lat = np.radians(lat)
        rad_lon = np.radians(lon)

        feats["coord_x"] = np.cos(rad_lat) * np.cos(rad_lon)
        feats["coord_y"] = np.cos(rad_lat) * np.sin(rad_lon)
        feats["coord_z"] = np.sin(rad_lat)
        feats["spatial_grid_100km"] = (
            lat.round(0).astype(str) + "_" + lon.round(0).astype(str)
        )
        feats["spatial_grid_10km"] = (
            lat.round(1).astype(str) + "_" + lon.round(1).astype(str)
        )
        feats["dist_to_center"] = np.sqrt(
            (lat - 39.8283) ** 2 + (lon - (-98.5795)) ** 2
        )
    return feats


def compute_text_meta_features(series: pd.Series, prefix: str = "") -> pd.DataFrame:
    s = series.fillna("").astype(str)
    meta = pd.DataFrame(index=series.index)
    meta[f"{prefix}_char_len"] = s.str.len()
    meta[f"{prefix}_word_count"] = s.str.split().str.len()
    meta[f"{prefix}_digit_count"] = s.str.count(r"\d")
    meta[f"{prefix}_upper_count"] = s.str.count(r"[A-Z]")
    return meta


# ---------------------------------------------------------
# 3. Apply Transformations to Feature Matrix
# ---------------------------------------------------------
train_processed = pd.DataFrame(index=train_df.index)
test_processed = pd.DataFrame(index=test_df.index)

for nc in num_cols:
    train_processed[f"{nc}_val"] = pd.to_numeric(train_df[nc], errors="coerce")
    test_processed[f"{nc}_val"] = pd.to_numeric(test_df[nc], errors="coerce")
    train_processed[f"{nc}_isnan"] = train_processed[f"{nc}_val"].isna().astype(np.int8)
    test_processed[f"{nc}_isnan"] = test_processed[f"{nc}_val"].isna().astype(np.int8)

if lat_col and lon_col:
    train_spatial = compute_spatial_features(train_df, lat_col, lon_col)
    test_spatial = compute_spatial_features(test_df, lat_col, lon_col)
    for c in train_spatial.columns:
        train_processed[c] = train_spatial[c]
        test_processed[c] = test_spatial[c]

primary_name_col = (
    name_cols[0] if len(name_cols) > 0 else (cat_cols[0] if len(cat_cols) > 0 else None)
)
if primary_name_col:
    kw_tr = extract_law_enforcement_keywords(train_df[primary_name_col])
    kw_te = extract_law_enforcement_keywords(test_df[primary_name_col])
    for col in kw_tr.columns:
        train_processed[col] = kw_tr[col]
        test_processed[col] = kw_te[col]

    meta_tr = compute_text_meta_features(
        train_df[primary_name_col], prefix=primary_name_col
    )
    meta_te = compute_text_meta_features(
        test_df[primary_name_col], prefix=primary_name_col
    )
    for col in meta_tr.columns:
        train_processed[col] = meta_tr[col]
        test_processed[col] = meta_te[col]

all_cat_candidates = list(
    set(
        cat_cols
        + [c for c in train_processed.columns if train_processed[c].dtype == "object"]
    )
)
for cc in all_cat_candidates:
    if cc in train_df.columns:
        tr_s = train_df[cc].fillna("MISSING").astype(str)
        te_s = test_df[cc].fillna("MISSING").astype(str)
    elif cc in train_processed.columns:
        tr_s = train_processed[cc].fillna("MISSING").astype(str)
        te_s = test_processed[cc].fillna("MISSING").astype(str)
    else:
        continue

    freq_map = tr_s.value_counts().to_dict()
    train_processed[f"{cc}_freq"] = tr_s.map(freq_map).fillna(1).astype(np.float32)
    test_processed[f"{cc}_freq"] = te_s.map(freq_map).fillna(1).astype(np.float32)

    le = LabelEncoder()
    combined = pd.concat([tr_s, te_s], axis=0).unique()
    le.fit(combined)
    train_processed[f"{cc}_label"] = le.transform(tr_s).astype(np.int32)
    test_processed[f"{cc}_label"] = le.transform(te_s).astype(np.int32)

obj_cols = [c for c in train_processed.columns if train_processed[c].dtype == "object"]
train_processed.drop(columns=obj_cols, inplace=True)
test_processed.drop(columns=obj_cols, inplace=True)

# ---------------------------------------------------------
# 4. Out-Of-Fold Encodings & Leak-Free Cross-Validation
# ---------------------------------------------------------
N_SPLITS = 5
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
y_train = (
    pd.to_numeric(train_df[TARGET_COL], errors="coerce")
    .fillna(0)
    .values.astype(np.float32)
)

N_SVD_COMPONENTS = 16
tfidf_cols = [f"svd_{i}" for i in range(N_SVD_COMPONENTS)]
for c in tfidf_cols:
    train_processed[c] = 0.0
    test_processed[c] = 0.0

if primary_name_col:
    train_text = train_df[primary_name_col].fillna("").astype(str).values
    test_text = test_df[primary_name_col].fillna("").astype(str).values

    tfidf_full = TfidfVectorizer(
        max_features=1000, ngram_range=(1, 2), token_pattern=r"(?u)\b\w+\b"
    )
    X_text_tr_full = tfidf_full.fit_transform(train_text)
    X_text_te_full = tfidf_full.transform(test_text)

    svd_full = TruncatedSVD(n_components=N_SVD_COMPONENTS, random_state=42)
    svd_full.fit(X_text_tr_full)
    svd_te_features = svd_full.transform(X_text_te_full)
    for i, c in enumerate(tfidf_cols):
        test_processed[c] = svd_te_features[:, i].astype(np.float32)

    for tr_idx, val_idx in kf.split(train_processed):
        tfidf_fold = TfidfVectorizer(
            max_features=1000, ngram_range=(1, 2), token_pattern=r"(?u)\b\w+\b"
        )
        X_tr = tfidf_fold.fit_transform(train_text[tr_idx])
        X_val = tfidf_fold.transform(train_text[val_idx])

        svd_fold = TruncatedSVD(n_components=N_SVD_COMPONENTS, random_state=42)
        svd_fold.fit(X_tr)
        svd_val = svd_fold.transform(X_val)

        for i, c in enumerate(tfidf_cols):
            train_processed.loc[val_idx, c] = svd_val[:, i].astype(np.float32)

key_cat_cols = [c for c in cat_cols if c in train_df.columns and c != ID_COL]
global_mean = float(y_train.mean())
SMOOTHING_WEIGHT = 10.0

for cc in key_cat_cols:
    te_feat_name = f"{cc}_oof_target_enc"
    train_processed[te_feat_name] = global_mean
    test_processed[te_feat_name] = global_mean

    counts = train_df.groupby(cc)[TARGET_COL].count()
    means = train_df.groupby(cc)[TARGET_COL].mean()
    smooth_full = (counts * means + SMOOTHING_WEIGHT * global_mean) / (
        counts + SMOOTHING_WEIGHT
    )
    test_processed[te_feat_name] = (
        test_df[cc].map(smooth_full).fillna(global_mean).astype(np.float32).values
    )

    for tr_idx, val_idx in kf.split(train_df):
        tr_subset = train_df.iloc[tr_idx]
        val_subset = train_df.iloc[val_idx]

        f_counts = tr_subset.groupby(cc)[TARGET_COL].count()
        f_means = tr_subset.groupby(cc)[TARGET_COL].mean()
        f_smooth = (f_counts * f_means + SMOOTHING_WEIGHT * global_mean) / (
            f_counts + SMOOTHING_WEIGHT
        )

        train_processed.loc[val_idx, te_feat_name] = (
            val_subset[cc].map(f_smooth).fillna(global_mean).astype(np.float32).values
        )

for col in train_processed.columns:
    med = train_processed[col].median()
    if np.isnan(med):
        med = 0.0
    train_processed[col] = train_processed[col].fillna(med).astype(np.float32)
    test_processed[col] = test_processed[col].fillna(med).astype(np.float32)

folds = list(kf.split(train_processed))
n_train = len(train_processed)
n_test = len(test_processed)


# ---------------------------------------------------------
# 5. Model Architecture Definitions
# ---------------------------------------------------------
class TabularResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout_rate: float = 0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim, dim),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class TabularResNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_blocks: int = 2,
        dropout_rate: float = 0.15,
        out_dim: int = 1,
    ):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )
        self.res_blocks = nn.ModuleList(
            [
                TabularResidualBlock(hidden_dim, dropout_rate=dropout_rate)
                for _ in range(num_blocks)
            ]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(hidden_dim // 2, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_layer(x)
        for block in self.res_blocks:
            h = block(h)
        out = self.head(h)
        return out.squeeze(-1)


def get_lgb_model(custom_params: dict = None) -> lgb.LGBMRegressor:
    default_params = {
        "objective": "regression",
        "metric": "rmse",
        "n_estimators": 2500,
        "learning_rate": 0.03,
        "num_leaves": 45,
        "max_depth": -1,
        "min_child_samples": 20,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }
    if custom_params:
        default_params.update(custom_params)
    return lgb.LGBMRegressor(**default_params)


def get_xgb_model(custom_params: dict = None) -> xgb.XGBRegressor:
    default_params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "n_estimators": 2500,
        "learning_rate": 0.03,
        "max_depth": 6,
        "min_child_weight": 3,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "tree_method": "hist",
        "random_state": 42,
        "n_jobs": -1,
    }
    if custom_params:
        default_params.update(custom_params)
    return xgb.XGBRegressor(**default_params)


def get_catboost_model(custom_params: dict = None) -> CatBoostRegressor:
    default_params = {
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "iterations": 2500,
        "learning_rate": 0.04,
        "depth": 6,
        "l2_leaf_reg": 3.0,
        "random_seed": 42,
        "verbose": False,
        "thread_count": -1,
    }
    if custom_params:
        default_params.update(custom_params)
    return CatBoostRegressor(**default_params)


def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2)))


# ---------------------------------------------------------
# 6. Cross-Validated Model Training
# ---------------------------------------------------------
oof_lgb = np.zeros(n_train, dtype=np.float32)
test_lgb = np.zeros(n_test, dtype=np.float32)

oof_xgb = np.zeros(n_train, dtype=np.float32)
test_xgb = np.zeros(n_test, dtype=np.float32)

oof_cat = np.zeros(n_train, dtype=np.float32)
test_cat = np.zeros(n_test, dtype=np.float32)

oof_nn = np.zeros(n_train, dtype=np.float32)
test_nn = np.zeros(n_test, dtype=np.float32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim = train_processed.shape[1]
BATCH_SIZE = 256
EPOCHS = 35

for fold, (tr_idx, val_idx) in enumerate(folds):
    X_tr, y_tr = train_processed.iloc[tr_idx], y_train[tr_idx]
    X_val, y_val = train_processed.iloc[val_idx], y_train[val_idx]

    # --- LightGBM ---
    lgb_model = get_lgb_model({"random_state": 42 + fold})
    lgb_model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )
    oof_lgb[val_idx] = np.clip(lgb_model.predict(X_val), 0, None)
    test_lgb += (np.clip(lgb_model.predict(test_processed), 0, None) / N_SPLITS).astype(
        np.float32
    )

    # --- XGBoost ---
    xgb_model = get_xgb_model({"early_stopping_rounds": 50, "random_state": 100 + fold})
    xgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    oof_xgb[val_idx] = np.clip(xgb_model.predict(X_val), 0, None)
    test_xgb += (np.clip(xgb_model.predict(test_processed), 0, None) / N_SPLITS).astype(
        np.float32
    )

    # --- CatBoost ---
    cat_model = get_catboost_model({"random_seed": 200 + fold})
    cat_model.fit(
        X_tr,
        y_tr,
        eval_set=(X_val, y_val),
        early_stopping_rounds=50,
        verbose=False,
    )
    oof_cat[val_idx] = np.clip(cat_model.predict(X_val), 0, None)
    test_cat += (np.clip(cat_model.predict(test_processed), 0, None) / N_SPLITS).astype(
        np.float32
    )

    # --- Tabular ResNet ---
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_val_scaled = scaler.transform(X_val)
    X_te_scaled = scaler.transform(test_processed)

    train_ds = TensorDataset(
        torch.tensor(X_tr_scaled, dtype=torch.float32),
        torch.tensor(y_tr, dtype=torch.float32),
    )
    val_ds = TensorDataset(
        torch.tensor(X_val_scaled, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32),
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    nn_model = TabularResNet(
        input_dim=input_dim, hidden_dim=128, num_blocks=2, dropout_rate=0.15
    ).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(nn_model.parameters(), lr=3e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6
    )

    best_val_loss = float("inf")
    best_weights = None

    for epoch in range(EPOCHS):
        nn_model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            pred = nn_model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(nn_model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        nn_model.eval()
        val_losses = []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                pred = nn_model(batch_x)
                val_losses.append(criterion(pred, batch_y).item())

        epoch_val_loss = np.mean(val_losses)
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_weights = {
                k: v.cpu().clone() for k, v in nn_model.state_dict().items()
            }

    nn_model.load_state_dict(best_weights)
    nn_model.eval()

    with torch.no_grad():
        val_t = torch.tensor(X_val_scaled, dtype=torch.float32).to(device)
        oof_nn[val_idx] = np.clip(nn_model(val_t).cpu().numpy(), 0, None)

        test_t = torch.tensor(X_te_scaled, dtype=torch.float32).to(device)
        test_nn += (np.clip(nn_model(test_t).cpu().numpy(), 0, None) / N_SPLITS).astype(
            np.float32
        )


# ---------------------------------------------------------
# 7. Constrained Ensemble Blending & Evaluation
# ---------------------------------------------------------
oof_matrix = np.column_stack([oof_lgb, oof_xgb, oof_cat, oof_nn])
test_matrix = np.column_stack([test_lgb, test_xgb, test_cat, test_nn])


def ensemble_loss(weights):
    w = np.array(weights)
    w = w / np.sum(w)
    pred = np.dot(oof_matrix, w)
    return calculate_rmse(y_train, pred)


init_weights = [0.35, 0.35, 0.20, 0.10]
bounds = [(0.0, 1.0) for _ in range(4)]
constraints = {"type": "eq", "fun": lambda w: 1.0 - np.sum(w)}

res = minimize(
    ensemble_loss,
    init_weights,
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
)
best_w = res.x / np.sum(res.x)

oof_ensemble = np.clip(np.dot(oof_matrix, best_w), 0, None)
final_val_rmse = calculate_rmse(y_train, oof_ensemble)


# ---------------------------------------------------------
# 8. Test Inference & Submission Generation
# ---------------------------------------------------------
test_predictions = np.clip(np.dot(test_matrix, best_w), 0, None)

submission = pd.DataFrame(
    {
        "id": np.arange(len(test_predictions), dtype=int),
        "number_of_officers": test_predictions,
    }
)

submission.to_csv("./submission/submission.csv", index=False)
print(f"Final Validation Score: {final_val_rmse}")