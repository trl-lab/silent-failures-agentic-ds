import json
import os
import re
import warnings
from catboost import CatBoostClassifier
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
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

# ---------------------------------------------------------
# 1. Configuration & Reproducibility
# ---------------------------------------------------------
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# ---------------------------------------------------------
# 2. Data Loading & Preprocessing
# ---------------------------------------------------------
train_path = "./input/train.parquet"
test_path = "./input/test.parquet"

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = "DEATHS_TOTAL_DESCRIPTION"

if target_col in train_df.columns:
    train_df = train_df.dropna(subset=[target_col]).reset_index(drop=True)

target_encoder = LabelEncoder()
train_df["target"] = target_encoder.fit_transform(train_df[target_col].astype(str))
target_classes = list(target_encoder.classes_)
num_classes = len(target_classes)

available_features = [
    c
    for c in train_df.columns
    if c in test_df.columns and c not in [target_col, "target"]
]


def find_col(df, candidates):
    cols_lower = {col.lower(): col for col in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None


def engineer_base_features(df):
    df_feat = df.copy()

    # Missingness indicators
    df_feat["missing_count"] = df.isna().sum(axis=1)
    df_feat["missing_ratio"] = df.isna().mean(axis=1)

    # Spatial features
    lat_col = find_col(df, ["latitude", "lat"])
    lon_col = find_col(df, ["longitude", "lon", "long"])
    elev_col = find_col(df, ["elevation", "elev", "altitude"])

    if lat_col and lon_col:
        lat = pd.to_numeric(df_feat[lat_col], errors="coerce").fillna(0.0)
        lon = pd.to_numeric(df_feat[lon_col], errors="coerce").fillna(0.0)

        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)

        df_feat["geo_x"] = np.cos(lat_rad) * np.cos(lon_rad)
        df_feat["geo_y"] = np.cos(lat_rad) * np.sin(lon_rad)
        df_feat["geo_z"] = np.sin(lat_rad)

        df_feat["abs_latitude"] = np.abs(lat)
        df_feat["is_northern_hemisphere"] = (lat >= 0).astype(int)
        df_feat["is_eastern_hemisphere"] = (lon >= 0).astype(int)

        is_ring_of_fire = (
            ((lat.between(-60, 60)) & (lon.between(100, 180)))
            | ((lat.between(-60, 60)) & (lon.between(-180, -60)))
        ).astype(int)
        df_feat["is_ring_of_fire"] = is_ring_of_fire

    if elev_col:
        elev = pd.to_numeric(df_feat[elev_col], errors="coerce").fillna(0.0)
        df_feat["is_submarine"] = (elev < 0).astype(int)
        df_feat["elev_log_abs"] = np.log1p(np.abs(elev))
        df_feat["elev_sign"] = np.sign(elev)

    # Temporal features
    year_col = find_col(df, ["year", "yr"])
    month_col = find_col(df, ["month", "mo"])

    yr = pd.Series(1900.0, index=df_feat.index)
    if year_col:
        yr = pd.to_numeric(df_feat[year_col], errors="coerce").fillna(1900.0)
        df_feat["year_clean"] = yr
        df_feat["era_ancient"] = (yr < 1500).astype(int)
        df_feat["era_early_modern"] = (yr.between(1500, 1800)).astype(int)
        df_feat["era_industrial"] = (yr.between(1800, 1950)).astype(int)
        df_feat["era_modern"] = (yr > 1950).astype(int)
        df_feat["century"] = np.floor(yr / 100.0)

        min_yr = -2000.0
        df_feat["year_shifted_log"] = np.log1p(np.maximum(0.0, yr - min_yr))
        df_feat["log_years_ago"] = np.log1p(np.maximum(0.0, 2024.0 - yr))

    if month_col:
        mo = pd.to_numeric(df_feat[month_col], errors="coerce").fillna(6.0)
        df_feat["month_sin"] = np.sin(2 * np.pi * mo / 12.0)
        df_feat["month_cos"] = np.cos(2 * np.pi * mo / 12.0)
        df_feat["month_is_missing"] = df_feat[month_col].isna().astype(int)

    # Physical severity proxies
    vei = pd.Series(-1.0, index=df_feat.index)
    vei_col = find_col(df, ["vei", "volcanic_explosivity_index"])
    if vei_col:
        vei = pd.to_numeric(df_feat[vei_col], errors="coerce").fillna(-1.0)
        df_feat["vei_clean"] = vei
        df_feat["vei_is_missing"] = (vei < 0).astype(int)
        df_feat["vei_energy_proxy"] = np.where(
            vei >= 0, np.power(10.0, np.clip(vei, 0, 8)), 0.0
        )

        if elev_col:
            elev = pd.to_numeric(df_feat[elev_col], errors="coerce").fillna(0.0)
            df_feat["vei_x_elev"] = np.maximum(0.0, vei) * np.log1p(np.abs(elev))

        # Era and reporting bias interactions
        if year_col:
            df_feat["vei_x_year_log"] = (
                np.maximum(0.0, vei) * df_feat["year_shifted_log"]
            )
            era_score = (
                1 * df_feat["era_ancient"]
                + 2 * df_feat["era_early_modern"]
                + 3 * df_feat["era_industrial"]
                + 4 * df_feat["era_modern"]
            )
            df_feat["era_vei_interaction"] = np.maximum(0.0, vei) * era_score
            df_feat["vei_over_log_years"] = np.maximum(0.0, vei) / (
                df_feat["log_years_ago"] + 1.0
            )

    # Associated event indicators
    has_tsu = pd.Series(0, index=df_feat.index)
    tsu_col = find_col(df, ["tsu", "tsunami", "flag_tsu"])
    if tsu_col:
        has_tsu = (
            df_feat[tsu_col].notna().astype(int) & (df_feat[tsu_col] != 0).astype(int)
        )
        df_feat["has_tsunami"] = has_tsu

    has_eq = pd.Series(0, index=df_feat.index)
    eq_col = find_col(df, ["eq", "earthquake", "flag_eq"])
    if eq_col:
        has_eq = (
            df_feat[eq_col].notna().astype(int) & (df_feat[eq_col] != 0).astype(int)
        )
        df_feat["has_earthquake"] = has_eq

    # Morphology keyword indicators
    type_col = find_col(df, ["type", "primary_volcano_type", "volcano_type"])
    is_strato = pd.Series(0, index=df_feat.index)
    is_pyro = pd.Series(0, index=df_feat.index)
    if type_col:
        type_str = df_feat[type_col].astype(str).str.lower()
        is_strato = type_str.str.contains("strato", regex=False).astype(int)
        is_pyro = type_str.str.contains("pyroclastic", regex=False).astype(int)
        df_feat["is_stratovolcano"] = is_strato
        df_feat["is_caldera"] = type_str.str.contains("caldera", regex=False).astype(
            int
        )
        df_feat["is_shield"] = type_str.str.contains("shield", regex=False).astype(int)
        df_feat["is_complex"] = type_str.str.contains("complex", regex=False).astype(
            int
        )
        df_feat["is_submarine_type"] = type_str.str.contains(
            "submarine", regex=False
        ).astype(int)
        df_feat["is_pyroclastic"] = is_pyro

    # Multi-hazard co-occurrence indicators
    df_feat["multi_hazard_count"] = (
        has_tsu + has_eq + (vei >= 3).astype(int)
    )
    df_feat["tsu_x_vei"] = has_tsu * np.maximum(0.0, vei)
    df_feat["eq_x_vei"] = has_eq * np.maximum(0.0, vei)
    df_feat["tsu_x_eq"] = has_tsu * has_eq
    df_feat["tsu_x_strato"] = has_tsu * is_strato
    df_feat["eq_x_strato"] = has_eq * is_strato

    # Advanced physical hazard severity indices
    vei_pos = np.maximum(0.0, vei)
    elev_val = pd.to_numeric(df_feat[elev_col], errors="coerce").fillna(0.0) if elev_col else pd.Series(0.0, index=df_feat.index)
    df_feat["hazard_composite_energy"] = (df_feat.get("vei_energy_proxy", 0.0) + 1.0) * (1.0 + 1.5 * has_tsu + 1.2 * has_eq)
    df_feat["tsunami_coastal_hazard"] = has_tsu * (elev_val < 800.0).astype(int)
    df_feat["pyroclastic_threat_index"] = is_pyro * (vei_pos + 1.0) + is_strato * (vei_pos + 1.0)
    df_feat["explosivity_density_interaction"] = vei_pos * (1.0 - df_feat["missing_ratio"])

    # Composite categorical interaction pairs
    country_col = find_col(df, ["country"])
    region_col = find_col(df, ["region"])
    tect_col = find_col(df, ["tectonic_settings", "tectonic_setting", "tectonic"])

    if country_col and type_col:
        df_feat["cat_country_x_type"] = (
            df_feat[country_col].astype(str) + "_" + df_feat[type_col].astype(str)
        )
    if region_col and tect_col:
        df_feat["cat_region_x_tect"] = (
            df_feat[region_col].astype(str) + "_" + df_feat[tect_col].astype(str)
        )
    if type_col and tect_col:
        df_feat["cat_type_x_tect"] = (
            df_feat[type_col].astype(str) + "_" + df_feat[tect_col].astype(str)
        )
    if country_col and tect_col:
        df_feat["cat_country_x_tect"] = (
            df_feat[country_col].astype(str) + "_" + df_feat[tect_col].astype(str)
        )

    return df_feat


train_processed = engineer_base_features(train_df[available_features])
test_processed = engineer_base_features(test_df[available_features])

# Spatial k-NN neighborhood density and proximity features (using 3D cartesian coordinates)
geo_cols = ["geo_x", "geo_y", "geo_z"]
if all(c in train_processed.columns for c in geo_cols):
    train_geo = train_processed[geo_cols].fillna(0.0).values
    test_geo = test_processed[geo_cols].fillna(0.0).values

    knn = NearestNeighbors(n_neighbors=16, metric="euclidean")
    knn.fit(train_geo)

    train_dists, _ = knn.kneighbors(train_geo, n_neighbors=16)
    test_dists, _ = knn.kneighbors(test_geo, n_neighbors=15)

    train_processed["knn_dist_1"] = train_dists[:, 1]
    train_processed["knn_dist_3"] = train_dists[:, 1:4].mean(axis=1)
    train_processed["knn_dist_5"] = train_dists[:, 1:6].mean(axis=1)
    train_processed["knn_dist_10"] = train_dists[:, 1:11].mean(axis=1)
    train_processed["knn_dist_15"] = train_dists[:, 1:16].mean(axis=1)
    train_processed["knn_density_5"] = 1.0 / (train_processed["knn_dist_5"] + 1e-4)
    train_processed["knn_density_10"] = 1.0 / (train_processed["knn_dist_10"] + 1e-4)

    test_processed["knn_dist_1"] = test_dists[:, 0]
    test_processed["knn_dist_3"] = test_dists[:, :3].mean(axis=1)
    test_processed["knn_dist_5"] = test_dists[:, :5].mean(axis=1)
    test_processed["knn_dist_10"] = test_dists[:, :10].mean(axis=1)
    test_processed["knn_dist_15"] = test_dists[:, :15].mean(axis=1)
    test_processed["knn_density_5"] = 1.0 / (test_processed["knn_dist_5"] + 1e-4)
    test_processed["knn_density_10"] = 1.0 / (test_processed["knn_dist_10"] + 1e-4)

    # Multi-scale KMeans spatial clustering
    for k_clusters in [8, 16]:
        kmeans = KMeans(n_clusters=k_clusters, random_state=SEED, n_init=10)
        train_km_dists = kmeans.fit_transform(train_geo)
        test_km_dists = kmeans.transform(test_geo)
        for c_i in range(k_clusters):
            train_processed[f"kmeans_{k_clusters}_dist_c{c_i}"] = train_km_dists[:, c_i]
            test_processed[f"kmeans_{k_clusters}_dist_c{c_i}"] = test_km_dists[:, c_i]
        train_processed[f"kmeans_{k_clusters}_min_dist"] = train_km_dists.min(axis=1)
        test_processed[f"kmeans_{k_clusters}_min_dist"] = test_km_dists.min(axis=1)

# Text TF-IDF + TruncatedSVD feature extraction
text_candidate_cols = [
    c for c in train_df.columns
    if c in test_df.columns and c not in [target_col, "target"] and train_df[c].dtype == "object"
]
if text_candidate_cols:
    train_text = train_df[text_candidate_cols].astype(str).fillna("").agg(" ".join, axis=1)
    test_text = test_df[text_candidate_cols].astype(str).fillna("").agg(" ".join, axis=1)

    tfidf = TfidfVectorizer(max_features=250, stop_words="english", ngram_range=(1, 2))
    train_tfidf = tfidf.fit_transform(train_text)
    test_tfidf = tfidf.transform(test_text)

    n_svd_comps = min(5, train_tfidf.shape[1])
    if n_svd_comps > 1:
        svd = TruncatedSVD(n_components=n_svd_comps, random_state=SEED)
        train_svd = svd.fit_transform(train_tfidf)
        test_svd = svd.transform(test_tfidf)
        for comp_idx in range(n_svd_comps):
            train_processed[f"text_svd_comp_{comp_idx}"] = train_svd[:, comp_idx]
            test_processed[f"text_svd_comp_{comp_idx}"] = test_svd[:, comp_idx]

common_cols = [c for c in train_processed.columns if c in test_processed.columns]
train_processed = train_processed[common_cols].copy()
test_processed = test_processed[common_cols].copy()

categorical_cols = []
numerical_cols = []
for c in common_cols:
    if (
        train_processed[c].dtype == "object"
        or train_processed[c].dtype.name == "category"
    ):
        categorical_cols.append(c)
    else:
        numerical_cols.append(c)

for col in categorical_cols:
    train_processed[col] = train_processed[col].astype(str).fillna("missing")
    test_processed[col] = test_processed[col].astype(str).fillna("missing")

    freq_map = train_processed[col].value_counts(normalize=True).to_dict()
    train_processed[f"{col}_freq"] = train_processed[col].map(freq_map).fillna(0.0)
    test_processed[f"{col}_freq"] = test_processed[col].map(freq_map).fillna(0.0)

    le = LabelEncoder()
    le.fit(train_processed[col].tolist() + ["<unknown>"])
    train_processed[f"{col}_cat_code"] = le.transform(train_processed[col])
    test_vals = test_processed[col].apply(
        lambda x: x if x in le.classes_ else "<unknown>"
    )
    test_processed[f"{col}_cat_code"] = le.transform(test_vals)

raw_cats_for_te = {
    c: (train_processed[c].copy(), test_processed[c].copy()) for c in categorical_cols
}

train_processed = train_processed.drop(columns=categorical_cols)
test_processed = test_processed.drop(columns=categorical_cols)

feature_columns = list(train_processed.columns)
train_medians = train_processed[feature_columns].median()
train_processed = train_processed.fillna(train_medians)
test_processed = test_processed.fillna(train_medians)

# ---------------------------------------------------------
# 3. Stratified K-Fold Setup & Out-of-Fold Target Encoding
# ---------------------------------------------------------
num_folds = 10
skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=SEED)
train_df["fold"] = -1

for fold_idx, (trn_idx, val_idx) in enumerate(
    skf.split(train_processed, train_df["target"])
):
    train_df.loc[val_idx, "fold"] = fold_idx

high_card_candidates = [
    c
    for c in categorical_cols
    if c.lower()
    in [
        "country",
        "region",
        "type",
        "volcano_type",
        "primary_volcano_type",
        "location",
        "tectonic_settings",
        "cat_country_x_type",
        "cat_region_x_tect",
        "cat_type_x_tect",
        "cat_country_x_tect",
    ]
    or c.startswith("cat_")
]

for cat_col in high_card_candidates:
    raw_train_cat, raw_test_cat = raw_cats_for_te[cat_col]

    for cl in range(num_classes):
        oof_col_name = f"te_{cat_col}_class_{cl}"
        train_processed[oof_col_name] = 0.0
        test_processed[oof_col_name] = 0.0
        global_prior = (train_df["target"] == cl).mean()

        for fold_idx in range(num_folds):
            trn_idx = train_df["fold"] != fold_idx
            val_idx = train_df["fold"] == fold_idx

            trn_group = (
                train_df.loc[trn_idx]
                .groupby(raw_train_cat.loc[trn_idx])["target"]
                .apply(lambda s: (s == cl).mean())
            )
            trn_counts = (
                train_df.loc[trn_idx].groupby(raw_train_cat.loc[trn_idx]).size()
            )
            smoothed = (trn_counts * trn_group + 10.0 * global_prior) / (
                trn_counts + 10.0
            )

            val_cats = raw_train_cat.loc[val_idx]
            train_processed.loc[val_idx, oof_col_name] = (
                val_cats.map(smoothed).fillna(global_prior).values
            )

        full_group = train_df.groupby(raw_train_cat)["target"].apply(
            lambda s: (s == cl).mean()
        )
        full_counts = train_df.groupby(raw_train_cat).size()
        full_smoothed = (full_counts * full_group + 10.0 * global_prior) / (
            full_counts + 10.0
        )
        test_processed[oof_col_name] = (
            raw_test_cat.map(full_smoothed).fillna(global_prior).values
        )

feature_cols = list(train_processed.columns)
train_processed["target"] = train_df["target"].values
train_processed["fold"] = train_df["fold"].values
X_test_raw = test_processed[feature_cols].values.astype(np.float32)


# ---------------------------------------------------------
# 4. Neural Network Architecture & Helpers
# ---------------------------------------------------------
class GatedTabularResidualBlock(nn.Module):

    def __init__(self, in_features: int, out_features: int, dropout_rate: float = 0.2):
        super().__init__()
        self.norm1 = nn.LayerNorm(in_features)
        self.fc1 = nn.Linear(in_features, out_features * 2)
        self.dropout = nn.Dropout(dropout_rate)
        self.norm2 = nn.LayerNorm(out_features)
        self.fc2 = nn.Linear(out_features, out_features)

        self.shortcut = (
            nn.Linear(in_features, out_features)
            if in_features != out_features
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.norm1(x)
        gate_val = self.fc1(out)
        val, gate = torch.chunk(gate_val, 2, dim=-1)
        out = F.silu(val) * torch.sigmoid(gate)
        out = self.dropout(out)
        out = self.norm2(out)
        out = self.fc2(out)
        return out + residual


class VolcanoTabularResNet(nn.Module):

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: list = [256, 128, 64],
        dropout_rate: float = 0.25,
    ):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.Mish(),
            nn.Dropout(dropout_rate * 0.5),
        )

        blocks = []
        for i in range(len(hidden_dims) - 1):
            blocks.append(
                GatedTabularResidualBlock(
                    in_features=hidden_dims[i],
                    out_features=hidden_dims[i + 1],
                    dropout_rate=dropout_rate,
                )
            )
        self.res_blocks = nn.Sequential(*blocks)

        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dims[-1]),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dims[-1], num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_layer(x)
        h = self.res_blocks(h)
        logits = self.head(h)
        return logits


class LabelSmoothingCrossEntropyLoss(nn.Module):

    def __init__(self, smoothing: float = 0.05):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = logits.size(-1)
        log_preds = F.log_softmax(logits, dim=-1)
        loss = -log_preds.sum(dim=-1).mean()
        nll = F.nll_loss(log_preds, targets)
        return (1.0 - self.smoothing) * nll + (self.smoothing / num_classes) * loss


def create_optimizer_and_scheduler(
    model: nn.Module,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    epochs: int = 35,
):
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "norm" in name or "bias" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer = AdamW(
        [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=lr,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    return optimizer, scheduler


def train_nn_model(X_train_s, y_train_f, X_val_s, y_val_f, X_test_s, n_classes, fold):
    train_dataset = TensorDataset(
        torch.tensor(X_train_s, dtype=torch.float32),
        torch.tensor(y_train_f, dtype=torch.long),
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val_s, dtype=torch.float32),
        torch.tensor(y_val_f, dtype=torch.long),
    )
    test_tensor = torch.tensor(X_test_s, dtype=torch.float32)

    train_loader = DataLoader(
        train_dataset, batch_size=64, shuffle=True, drop_last=False
    )
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

    model = VolcanoTabularResNet(
        input_dim=X_train_s.shape[1],
        num_classes=n_classes,
        hidden_dims=[256, 128, 64],
        dropout_rate=0.25,
    ).to(device)

    criterion = LabelSmoothingCrossEntropyLoss(smoothing=0.05)
    epochs = 35
    optimizer, scheduler = create_optimizer_and_scheduler(
        model, lr=1e-3, weight_decay=1e-4, epochs=epochs
    )

    best_val_loss = float("inf")
    best_weights = None

    for epoch in range(epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

        if scheduler:
            scheduler.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                val_loss += loss.item() * batch_x.size(0)
        val_loss /= len(val_dataset)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_weights)
    model.to(device)
    model.eval()

    with torch.no_grad():
        val_preds_list = []
        for batch_x, _ in val_loader:
            logits = model(batch_x.to(device))
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            val_preds_list.append(probs)
        val_probs = np.vstack(val_preds_list)

        test_loader = DataLoader(
            TensorDataset(test_tensor), batch_size=128, shuffle=False
        )
        test_preds_list = []
        for (batch_x,) in test_loader:
            logits = model(batch_x.to(device))
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            test_preds_list.append(probs)
        test_probs = np.vstack(test_preds_list)

    return val_probs, test_probs


# ---------------------------------------------------------
# 5. GBDT Hyperparameters
# ---------------------------------------------------------
def get_lgbm_params(n_classes: int, seed: int = 42) -> dict:
    return {
        "objective": "multiclass",
        "num_class": n_classes,
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
        "random_state": seed,
        "n_estimators": 1000,
        "n_jobs": -1,
        "verbose": -1,
    }


def get_xgb_params(n_classes: int, seed: int = 42) -> dict:
    return {
        "objective": "multi:softprob",
        "num_class": n_classes,
        "eval_metric": "mlogloss",
        "learning_rate": 0.03,
        "max_depth": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 2,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": seed,
        "n_estimators": 1000,
        "n_jobs": -1,
    }


def get_catboost_params(n_classes: int, seed: int = 42) -> dict:
    return {
        "loss_function": "MultiClass",
        "iterations": 800,
        "learning_rate": 0.035,
        "depth": 6,
        "l2_leaf_reg": 3.0,
        "random_seed": seed,
        "verbose": False,
        "thread_count": -1,
    }


# ---------------------------------------------------------
# 6. Stratified 10-Fold Training Loop
# ---------------------------------------------------------
oof_lgb = np.zeros((len(train_processed), num_classes), dtype=np.float32)
oof_xgb = np.zeros((len(train_processed), num_classes), dtype=np.float32)
oof_cat = np.zeros((len(train_processed), num_classes), dtype=np.float32)
oof_nn = np.zeros((len(train_processed), num_classes), dtype=np.float32)

test_lgb = np.zeros((len(test_processed), num_classes), dtype=np.float32)
test_xgb = np.zeros((len(test_processed), num_classes), dtype=np.float32)
test_cat = np.zeros((len(test_processed), num_classes), dtype=np.float32)
test_nn = np.zeros((len(test_processed), num_classes), dtype=np.float32)

for fold in range(num_folds):
    trn_mask = train_processed["fold"] != fold
    val_mask = train_processed["fold"] == fold

    X_train = train_processed.loc[trn_mask, feature_cols].values.astype(np.float32)
    y_train = train_processed.loc[trn_mask, "target"].values.astype(np.int64)

    X_val = train_processed.loc[val_mask, feature_cols].values.astype(np.float32)
    y_val = train_processed.loc[val_mask, "target"].values.astype(np.int64)

    # 6.1 LightGBM
    lgb_params = get_lgbm_params(n_classes=num_classes, seed=SEED + fold)
    clf_lgb = lgb.LGBMClassifier(**lgb_params)
    clf_lgb.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=-1),
        ],
    )
    oof_lgb[val_mask] = clf_lgb.predict_proba(X_val)
    test_lgb += clf_lgb.predict_proba(X_test_raw) / num_folds

    # 6.2 XGBoost
    xgb_params = get_xgb_params(n_classes=num_classes, seed=SEED + fold)
    clf_xgb = xgb.XGBClassifier(**xgb_params, early_stopping_rounds=50, verbosity=0)
    clf_xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    oof_xgb[val_mask] = clf_xgb.predict_proba(X_val)
    test_xgb += clf_xgb.predict_proba(X_test_raw) / num_folds

    # 6.3 CatBoost
    cat_params = get_catboost_params(n_classes=num_classes, seed=SEED + fold)
    clf_cat = CatBoostClassifier(**cat_params)
    clf_cat.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=50, verbose=False)
    oof_cat[val_mask] = clf_cat.predict_proba(X_val)
    test_cat += clf_cat.predict_proba(X_test_raw) / num_folds

    # 6.4 VolcanoTabularResNet
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test_raw)

    nn_val_prob, nn_test_prob = train_nn_model(
        X_train_scaled,
        y_train,
        X_val_scaled,
        y_val,
        X_test_scaled,
        num_classes,
        fold,
    )
    oof_nn[val_mask] = nn_val_prob
    test_nn += nn_test_prob / num_folds

# ---------------------------------------------------------
# 7. Constrained Simplex Blending & Evaluation
# ---------------------------------------------------------
y_true_all = train_processed["target"].values

def blend_objective(weights):
    w = np.array(weights)
    w = np.maximum(w, 0.0)
    total = np.sum(w)
    if total > 0:
        w = w / total
    else:
        w = np.ones_like(w) / len(w)
    blended = w[0] * oof_lgb + w[1] * oof_xgb + w[2] * oof_cat + w[3] * oof_nn
    eps = 1e-15
    blended = np.clip(blended, eps, 1.0 - eps)
    blended = blended / blended.sum(axis=1, keepdims=True)
    y_oh = np.eye(num_classes)[y_true_all]
    return -np.mean(np.sum(y_oh * np.log(blended), axis=1))

init_w = np.array([0.3, 0.3, 0.3, 0.1])
constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
bounds = [(0.0, 1.0)] * 4

opt_res = minimize(
    blend_objective,
    init_w,
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
)
opt_w = np.maximum(opt_res.x, 0.0)
opt_w = opt_w / np.sum(opt_w)

oof_probabilities = (
    opt_w[0] * oof_lgb + opt_w[1] * oof_xgb + opt_w[2] * oof_cat + opt_w[3] * oof_nn
)
test_probabilities = (
    opt_w[0] * test_lgb + opt_w[1] * test_xgb + opt_w[2] * test_cat + opt_w[3] * test_nn
)

oof_predictions = np.argmax(oof_probabilities, axis=1)
overall_cv_accuracy = accuracy_score(y_true_all, oof_predictions)

final_test_pred_indices = np.argmax(test_probabilities, axis=1)
final_test_predictions = target_encoder.inverse_transform(final_test_pred_indices)

submission_df = pd.DataFrame(
    {
        "id": np.arange(len(final_test_predictions)),
        "DEATHS_TOTAL_DESCRIPTION": final_test_predictions,
    }
)
submission_df.to_csv("./submission/submission.csv", index=False)

print(f"Final Validation Score: {overall_cv_accuracy}")
