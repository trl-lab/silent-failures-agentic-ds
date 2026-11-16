import gc
import math
import os
import re
import warnings
from typing import Any, Dict, Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import RobustScaler, StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
import xgboost as xgb

try:
    import catboost as cb
except ImportError:
    cb = None

warnings.filterwarnings("ignore")

# Ensure required directories exist
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# ==============================================================================
# 1. Data Loading & Feature Engineering
# ==============================================================================

print("Loading datasets from ./input...")
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "REVENUE15"
has_target = target_col in train_df.columns

print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

y_train = train_df[target_col].copy() if has_target else None
train_df_features = train_df.drop(columns=[target_col], errors="ignore").copy()
test_df_features = test_df.copy()

train_n = len(train_df_features)
test_n = len(test_df_features)

feature_cols = [c for c in train_df_features.columns if c in test_df_features.columns]
train_df_features = train_df_features[feature_cols]
test_df_features = test_df_features[feature_cols]

df_all = pd.concat([train_df_features, test_df_features], axis=0, ignore_index=True)

# Comprehensive numeric parsing, type recovery, and column taxonomy
def clean_numeric_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    cleaned = (
        series.astype(str)
        .str.replace(r"[\$,%]", "", regex=True)
        .str.replace(r"\((.*?)\)", r"-\1", regex=True)
        .str.strip()
    )
    return pd.to_numeric(cleaned, errors="coerce")

num_cols = []
cat_cols = []
pure_text_cols = []

for c in df_all.columns:
    parsed_s = clean_numeric_series(df_all[c])
    parse_ratio = parsed_s.notna().mean()
    nunique_val = df_all[c].nunique(dropna=True)

    explicit_text = any(k in c.upper() for k in ["LEGALNAME", "COMMONNAME", "ALTNAME", "WEBURL", "ADDRESS", "CITY", "STREET", "DESC", "MISSION"])
    explicit_num = any(k in c.upper() for k in ["REV", "INC", "EXP", "POP", "EMP", "VOL", "LAT", "LON", "LONG", "YEAR", "FISYR", "AMT", "VAL", "ASSET", "LIAB"])

    if parse_ratio >= 0.30 and not explicit_text:
        num_cols.append(c)
        if nunique_val <= 30 and not explicit_num:
            cat_cols.append(c)
    elif nunique_val <= 250 and not explicit_text:
        cat_cols.append(c)
    else:
        pure_text_cols.append(c)

print(
    f"Identified {len(num_cols)} numeric, {len(cat_cols)} categorical, and {len(pure_text_cols)} text columns."
)


# Geospatial Feature Extraction & Spatial Clustering
def extract_geo_features(df):
    lat_col = next((c for c in df.columns if "LAT" in c.upper()), None)
    lon_col = next(
        (c for c in df.columns if "LON" in c.upper() or "LONG" in c.upper()), None
    )

    geo_df = pd.DataFrame(index=df.index)
    if lat_col and lon_col:
        lat = clean_numeric_series(df[lat_col]).fillna(39.8283).clip(18.0, 72.0)
        lon = clean_numeric_series(df[lon_col]).fillna(-98.5795).clip(-180.0, -65.0)

        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)

        geo_df["geo_x"] = np.cos(lat_rad) * np.cos(lon_rad)
        geo_df["geo_y"] = np.cos(lat_rad) * np.sin(lon_rad)
        geo_df["geo_z"] = np.sin(lat_rad)

        geo_df["geo_rot45_x"] = 0.7071 * lat + 0.7071 * lon
        geo_df["geo_rot45_y"] = -0.7071 * lat + 0.7071 * lon

        coords = np.column_stack([lat.values, lon.values])
        kmeans = MiniBatchKMeans(n_clusters=12, random_state=42, batch_size=1024, n_init="auto")
        geo_df["geo_cluster"] = kmeans.fit_predict(coords)
        cluster_dists = kmeans.transform(coords)
        for k in range(min(6, cluster_dists.shape[1])):
            geo_df[f"geo_cluster_dist_{k}"] = cluster_dists[:, k]

        hubs = {
            "dist_nyc": (40.7128, -74.0060),
            "dist_dc": (38.9072, -77.0369),
            "dist_chi": (41.8781, -87.6298),
            "dist_la": (34.0522, -118.2437),
            "dist_sf": (37.7749, -122.4194),
            "dist_hou": (29.7604, -95.3698),
            "dist_atl": (33.7490, -84.3880),
        }
        for name, (h_lat, h_lon) in hubs.items():
            dlat = np.radians(lat - h_lat)
            dlon = np.radians(lon - h_lon)
            a = (
                np.sin(dlat / 2.0) ** 2
                + np.cos(np.radians(h_lat)) * np.cos(lat_rad) * np.sin(dlon / 2.0) ** 2
            )
            c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
            geo_df[name] = 6371.0 * c

    return geo_df


geo_features = extract_geo_features(df_all)


# Text Domain Keywords & Metadata
def clean_text(val):
    if pd.isna(val):
        return ""
    text = str(val).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


combined_text = pd.Series("", index=df_all.index)
for col in pure_text_cols:
    cleaned = df_all[col].apply(clean_text)
    combined_text = combined_text + " " + cleaned

combined_text = combined_text.str.strip()

domain_keywords = [
    "foundation",
    "society",
    "historical",
    "association",
    "university",
    "children",
    "art",
    "memorial",
    "zoo",
    "botanic",
    "science",
    "natural",
    "center",
    "national",
    "institute",
    "preservation",
    "heritage",
]
keyword_features = pd.DataFrame(index=df_all.index)
for kw in domain_keywords:
    keyword_features[f"kw_{kw}"] = combined_text.str.contains(
        r"\b" + kw + r"\b", regex=True
    ).astype(int)

keyword_features["text_length"] = combined_text.str.len()
keyword_features["text_word_count"] = combined_text.str.split().apply(len)

# Categorical Encodings
cat_features = pd.DataFrame(index=df_all.index)
for col in cat_cols:
    val_series = df_all[col].astype(str).fillna("MISSING")
    freq = val_series.value_counts(normalize=True)
    cat_features[f"{col}_freq"] = val_series.map(freq).astype(float)
    cat_features[f"{col}_cat"] = pd.factorize(val_series)[0]

state_col = next(
    (c for c in cat_cols if "STATE" in c.upper() or "ST" in c.upper()), None
)
discip_col = next((c for c in cat_cols if "DISCIP" in c.upper()), None)
if state_col and discip_col:
    combo = df_all[state_col].astype(str) + "_" + df_all[discip_col].astype(str)
    cat_features["state_discip_freq"] = combo.map(
        combo.value_counts(normalize=True)
    ).astype(float)
    cat_features["state_discip_cat"] = pd.factorize(combo)[0]

# Numeric Cleaning & Cross-Feature Engineering
numeric_features = pd.DataFrame(index=df_all.index)
for col in num_cols:
    s = clean_numeric_series(df_all[col])
    numeric_features[f"{col}_isna"] = s.isna().astype(int)
    median_val = s.median()
    median_val = 0.0 if pd.isna(median_val) else float(median_val)
    imputed_s = s.fillna(median_val)
    numeric_features[f"{col}_val"] = imputed_s
    numeric_features[f"{col}_log1p"] = np.sign(imputed_s) * np.log1p(np.abs(imputed_s))

fin_cols = [c for c in num_cols if any(k in c.upper() for k in ["REV", "INC", "EXP", "AMT", "VAL", "ASSET", "LIAB"])]
if len(fin_cols) >= 2:
    val_matrix = numeric_features[[f"{c}_val" for c in fin_cols]].values
    numeric_features["fin_sum"] = np.sum(val_matrix, axis=1)
    numeric_features["fin_mean"] = np.mean(val_matrix, axis=1)
    numeric_features["fin_std"] = np.std(val_matrix, axis=1)
    numeric_features["fin_max"] = np.max(val_matrix, axis=1)
    numeric_features["fin_min"] = np.min(val_matrix, axis=1)

    top_fin = fin_cols[:min(5, len(fin_cols))]
    for i in range(len(top_fin)):
        for j in range(i + 1, len(top_fin)):
            c1, c2 = top_fin[i], top_fin[j]
            v1 = numeric_features[f"{c1}_val"]
            v2 = numeric_features[f"{c2}_val"]
            numeric_features[f"diff_{c1}_{c2}"] = v1 - v2
            numeric_features[f"ratio_{c1}_{c2}"] = (v1 + 1.0) / (v2.abs() + 1.0)

all_engineered = pd.concat(
    [numeric_features, cat_features, keyword_features, geo_features], axis=1
)

train_feat = all_engineered.iloc[:train_n].copy().reset_index(drop=True)
test_feat = all_engineered.iloc[train_n:].copy().reset_index(drop=True)

train_text = combined_text.iloc[:train_n].reset_index(drop=True)
test_text = combined_text.iloc[train_n:].reset_index(drop=True)

# TF-IDF Word + Char N-Grams + TruncatedSVD
print("Fitting word and character TF-IDF and SVD strictly on training text...")
from scipy.sparse import hstack

tfidf_w = TfidfVectorizer(
    max_features=20000, ngram_range=(1, 2), min_df=2, sublinear_tf=True
)
train_tfidf_w = tfidf_w.fit_transform(train_text)
test_tfidf_w = tfidf_w.transform(test_text)

tfidf_c = TfidfVectorizer(
    max_features=10000, analyzer="char_wb", ngram_range=(3, 5), min_df=4, sublinear_tf=True
)
train_tfidf_c = tfidf_c.fit_transform(train_text)
test_tfidf_c = tfidf_c.transform(test_text)

train_tfidf_all = hstack([train_tfidf_w, train_tfidf_c])
test_tfidf_all = hstack([test_tfidf_w, test_tfidf_c])

n_svd = min(36, train_tfidf_all.shape[1] - 1)
svd = TruncatedSVD(n_components=n_svd, random_state=42)
train_svd = svd.fit_transform(train_tfidf_all)
test_svd = svd.transform(test_tfidf_all)

for i in range(n_svd):
    train_feat[f"svd_text_{i}"] = train_svd[:, i]
    test_feat[f"svd_text_{i}"] = test_svd[:, i]

# 5-Fold Split Setup
n_splits = 5
num_bins = 10
try:
    y_binned = pd.qcut(y_train, q=num_bins, labels=False, duplicates="drop")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    folds = np.zeros(train_n, dtype=int)
    for fold_idx, (_, val_idx) in enumerate(skf.split(train_feat, y_binned)):
        folds[val_idx] = fold_idx
except Exception:
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    folds = np.zeros(train_n, dtype=int)
    for fold_idx, (_, val_idx) in enumerate(kf.split(train_feat)):
        folds[val_idx] = fold_idx

# Out-of-fold target encoding with additive smoothing
global_mean_y = float(y_train.mean())
smooth_m = 10.0
te_cols = [c for c in cat_cols if df_all[c].nunique(dropna=True) >= 3]

for c in te_cols:
    train_feat[f"{c}_te"] = np.nan
    test_s = df_all.iloc[train_n:][c].astype(str).fillna("MISSING")

    # Global target encoding for test set
    full_train_s = df_all.iloc[:train_n][c].astype(str).fillna("MISSING")
    df_temp = pd.DataFrame({"cat": full_train_s, "y": y_train.values})
    stats = df_temp.groupby("cat")["y"].agg(["count", "mean"])
    smooth_te = (stats["count"] * stats["mean"] + smooth_m * global_mean_y) / (stats["count"] + smooth_m)
    test_feat[f"{c}_te"] = test_s.map(smooth_te).fillna(global_mean_y).astype(float).values

    # Strictly leak-free fold-wise encoding for train set
    for fold in range(n_splits):
        val_idx = np.where(folds == fold)[0]
        tr_idx = np.where(folds != fold)[0]

        df_tr = pd.DataFrame({"cat": full_train_s.iloc[tr_idx], "y": y_train.iloc[tr_idx].values})
        tr_stats = df_tr.groupby("cat")["y"].agg(["count", "mean"])
        tr_smooth = (tr_stats["count"] * tr_stats["mean"] + smooth_m * global_mean_y) / (tr_stats["count"] + smooth_m)
        train_feat.loc[val_idx, f"{c}_te"] = full_train_s.iloc[val_idx].map(tr_smooth).fillna(global_mean_y).astype(float).values

train_feat["fold"] = folds
train_feat[target_col] = y_train.values


def sanitize_columns(df):
    clean_cols = [re.sub(r"[\[\]<>{}:,\"]+", "_", str(c)) for c in df.columns]
    df.columns = clean_cols
    return df


train_feat = sanitize_columns(train_feat)
test_feat = sanitize_columns(test_feat)


# ==============================================================================
# 2. Model Architecture & Model Builders
# ==============================================================================


class ResidualBlock(nn.Module):

    def __init__(self, hidden_dim: int, dropout_rate: float = 0.15):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.act1 = nn.Mish()
        self.dropout1 = nn.Dropout(dropout_rate)

        self.norm2 = nn.LayerNorm(hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.act2 = nn.Mish()
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.norm1(x)
        out = self.linear1(out)
        out = self.act1(out)
        out = self.dropout1(out)

        out = self.norm2(out)
        out = self.linear2(out)
        out = self.act2(out)
        out = self.dropout2(out)

        return residual + out


class TabularResNetRegressor(nn.Module):

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_blocks: int = 3,
        dropout_rate: float = 0.15,
        out_dim: int = 1,
    ):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.Mish(),
            nn.Dropout(dropout_rate),
        )

        self.blocks = nn.ModuleList(
            [
                ResidualBlock(hidden_dim, dropout_rate=dropout_rate)
                for _ in range(num_blocks)
            ]
        )

        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Mish(),
            nn.Dropout(dropout_rate / 2.0),
            nn.Linear(hidden_dim // 2, out_dim),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_layer(x)
        for block in self.blocks:
            h = block(h)
        out = self.head(h)
        return out.squeeze(-1)


def build_lightgbm_regressor(custom_params: Optional[Dict[str, Any]] = None):
    default_params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "n_estimators": 2500,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": 7,
        "min_child_samples": 20,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.75,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }
    if custom_params:
        default_params.update(custom_params)
    return lgb.LGBMRegressor(**default_params)


def build_xgboost_regressor(custom_params: Optional[Dict[str, Any]] = None):
    default_params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "n_estimators": 2500,
        "learning_rate": 0.03,
        "max_depth": 6,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.75,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": -1,
    }
    if custom_params:
        default_params.update(custom_params)
    return xgb.XGBRegressor(**default_params)


def build_catboost_regressor(custom_params: Optional[Dict[str, Any]] = None):
    if cb is None:
        return None
    default_params = {
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "iterations": 2500,
        "learning_rate": 0.03,
        "depth": 6,
        "l2_leaf_reg": 3.0,
        "random_seed": 42,
        "verbose": False,
        "thread_count": -1,
    }
    if custom_params:
        default_params.update(custom_params)
    return cb.CatBoostRegressor(**default_params)


class TabularTorchDataset(Dataset):

    def __init__(self, X_data, y_data=None):
        self.X = torch.tensor(X_data, dtype=torch.float32)
        self.y = (
            torch.tensor(y_data, dtype=torch.float32) if y_data is not None else None
        )

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]


# ==============================================================================
# 3. Cross-Validation & Ensembling
# ==============================================================================

ignore_cols = ["fold", target_col]
feature_names = [c for c in train_feat.columns if c not in ignore_cols]

X = train_feat[feature_names].values.astype(np.float32)
y = train_feat[target_col].values.astype(np.float32)
folds_arr = train_feat["fold"].values
X_test = test_feat[feature_names].values.astype(np.float32)

n_samples = len(X)
n_test = len(X_test)
n_folds = len(np.unique(folds_arr))
n_features = X.shape[1]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(
    f"Training setup: {n_samples} train samples, {n_test} test samples, {n_features} features, device: {device}"
)

oof_lgb = np.zeros(n_samples, dtype=np.float32)
oof_xgb = np.zeros(n_samples, dtype=np.float32)
oof_nn = np.zeros(n_samples, dtype=np.float32)

test_preds_lgb = np.zeros(n_test, dtype=np.float32)
test_preds_xgb = np.zeros(n_test, dtype=np.float32)
test_preds_nn = np.zeros(n_test, dtype=np.float32)

has_cb = cb is not None
if has_cb:
    oof_cb = np.zeros(n_samples, dtype=np.float32)
    test_preds_cb = np.zeros(n_test, dtype=np.float32)

for fold in range(n_folds):
    tr_idx = np.where(folds_arr != fold)[0]
    val_idx = np.where(folds_arr == fold)[0]

    X_train_f, y_train_f = X[tr_idx], y[tr_idx]
    X_val_f, y_val_f = X[val_idx], y[val_idx]

    # 1. LightGBM
    model_lgb = build_lightgbm_regressor()
    model_lgb.fit(
        X_train_f,
        y_train_f,
        eval_set=[(X_val_f, y_val_f)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )
    oof_lgb[val_idx] = model_lgb.predict(X_val_f)
    test_preds_lgb += model_lgb.predict(X_test) / n_folds

    # 2. XGBoost
    model_xgb = build_xgboost_regressor({"early_stopping_rounds": 50})
    model_xgb.fit(X_train_f, y_train_f, eval_set=[(X_val_f, y_val_f)], verbose=False)
    oof_xgb[val_idx] = model_xgb.predict(X_val_f)
    test_preds_xgb += model_xgb.predict(X_test) / n_folds

    # 3. CatBoost (if available)
    if has_cb:
        model_cb = build_catboost_regressor()
        model_cb.fit(
            X_train_f,
            y_train_f,
            eval_set=(X_val_f, y_val_f),
            early_stopping_rounds=50,
            verbose=False,
        )
        oof_cb[val_idx] = model_cb.predict(X_val_f)
        test_preds_cb += model_cb.predict(X_test) / n_folds

    # 4. TabularResNet (PyTorch)
    scaler = StandardScaler()
    X_tr_scaled = np.nan_to_num(scaler.fit_transform(X_train_f), nan=0.0)
    X_val_scaled = np.nan_to_num(scaler.transform(X_val_f), nan=0.0)
    X_test_scaled = np.nan_to_num(scaler.transform(X_test), nan=0.0)

    train_ds = TabularTorchDataset(X_tr_scaled, y_train_f)
    val_ds = TabularTorchDataset(X_val_scaled, y_val_f)
    test_ds = TabularTorchDataset(X_test_scaled)

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    nn_model = TabularResNetRegressor(
        input_dim=n_features, hidden_dim=128, num_blocks=2, dropout_rate=0.15
    ).to(device)
    criterion = nn.MSELoss()
    optimizer = AdamW(nn_model.parameters(), lr=3e-3, weight_decay=1e-4, eps=1e-8)
    scheduler = CosineAnnealingLR(optimizer, T_max=25, eta_min=1e-6)

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

        if scheduler:
            scheduler.step()

        nn_model.eval()
        val_losses = []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                out = nn_model(batch_x)
                val_losses.append(criterion(out, batch_y).item())
        avg_val_loss = np.mean(val_losses)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_nn_weights = {
                k: v.cpu().clone() for k, v in nn_model.state_dict().items()
            }

    nn_model.load_state_dict({k: v.to(device) for k, v in best_nn_weights.items()})
    nn_model.eval()

    val_preds_list = []
    with torch.no_grad():
        for batch_x, _ in val_loader:
            val_preds_list.append(nn_model(batch_x.to(device)).cpu().numpy())
    oof_nn[val_idx] = np.concatenate(val_preds_list)

    test_preds_list = []
    with torch.no_grad():
        for batch_x in test_loader:
            test_preds_list.append(nn_model(batch_x.to(device)).cpu().numpy())
    test_preds_nn += np.concatenate(test_preds_list) / n_folds

    fold_rmse_lgb = root_mean_squared_error(y_val_f, oof_lgb[val_idx])
    fold_rmse_xgb = root_mean_squared_error(y_val_f, oof_xgb[val_idx])
    print(
        f"Fold {fold+1}/{n_folds} Finished | LGB RMSE: {fold_rmse_lgb:.4f} | XGB RMSE: {fold_rmse_xgb:.4f}"
    )

    gc.collect()

# Ensemble Optimization & Meta-Stacking
model_names = ["LightGBM", "XGBoost", "ResNet"]
oof_matrix = np.column_stack([oof_lgb, oof_xgb, oof_nn])
test_matrix = np.column_stack([test_preds_lgb, test_preds_xgb, test_preds_nn])

if has_cb:
    model_names.append("CatBoost")
    oof_matrix = np.column_stack([oof_matrix, oof_cb])
    test_matrix = np.column_stack([test_matrix, test_preds_cb])

# Ridge Meta-model stacking
meta_ridge = Ridge(alpha=10.0, positive=True)
meta_ridge.fit(oof_matrix, y)
oof_ridge = np.clip(meta_ridge.predict(oof_matrix), 0, None)
test_ridge = np.clip(meta_ridge.predict(test_matrix), 0, None)
ridge_rmse = root_mean_squared_error(y, oof_ridge)

# SLSQP Bounded Non-negative Blending
def blend_objective(weights):
    weights = np.array(weights)
    weights = weights / (np.sum(weights) + 1e-12)
    pred = np.sum(oof_matrix * weights, axis=1)
    pred = np.clip(pred, 0, None)
    return root_mean_squared_error(y, pred)

n_models = oof_matrix.shape[1]
init_weights = np.ones(n_models) / n_models
bounds = [(0, 1) for _ in range(n_models)]
res = minimize(
    blend_objective,
    init_weights,
    method="SLSQP",
    bounds=bounds,
    constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
)

opt_weights = (
    res.x / np.sum(res.x)
    if res.success
    else np.ones(n_models, dtype=np.float32) / n_models
)
print(f"Optimal Ensemble Weights: {dict(zip(model_names, np.round(opt_weights, 4)))}")

oof_slsqp = np.clip(np.sum(oof_matrix * opt_weights, axis=1), 0, None)
test_slsqp = np.clip(np.sum(test_matrix * opt_weights, axis=1), 0, None)
slsqp_rmse = root_mean_squared_error(y, oof_slsqp)

oof_final = 0.5 * oof_slsqp + 0.5 * oof_ridge
test_final = 0.5 * test_slsqp + 0.5 * test_ridge
final_rmse = root_mean_squared_error(y, oof_final)

if slsqp_rmse < final_rmse and slsqp_rmse < ridge_rmse:
    oof_final = oof_slsqp
    test_final = test_slsqp
    final_rmse = slsqp_rmse
elif ridge_rmse < final_rmse and ridge_rmse < slsqp_rmse:
    oof_final = oof_ridge
    test_final = test_ridge
    final_rmse = ridge_rmse

# ==============================================================================
# 4. Generate Submission
# ==============================================================================

submission_df = pd.DataFrame(
    {"id": np.arange(len(test_final), dtype=int), "REVENUE15": test_final}
)
submission_path = "./submission/submission.csv"
submission_df.to_csv(submission_path, index=False)

print(
    f"Submission saved successfully to {submission_path} with shape {submission_df.shape}."
)
print(f"Final Validation Score: {final_rmse}")
