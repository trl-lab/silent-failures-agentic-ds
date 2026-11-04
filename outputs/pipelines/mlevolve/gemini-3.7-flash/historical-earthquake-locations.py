import copy
import gc
import os
import random
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import BallTree, NearestNeighbors
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import lightgbm as lgb
import xgboost as xgb
import catboost as cb


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =====================================================================
# 1. Deep Tabular ResNet Architecture
# =====================================================================
class ResNetBlock(nn.Module):
    """Residual block tailored for dense tabular representations."""

    def __init__(self, in_features: int, out_features: int, dropout_rate: float = 0.2):
        super().__init__()
        self.linear1 = nn.Linear(in_features, out_features)
        self.bn1 = nn.BatchNorm1d(out_features)
        self.act1 = nn.SiLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear2 = nn.Linear(out_features, out_features)
        self.bn2 = nn.BatchNorm1d(out_features)
        self.act2 = nn.SiLU()

        if in_features != out_features:
            self.shortcut = nn.Sequential(
                nn.Linear(in_features, out_features), nn.BatchNorm1d(out_features)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.act1(self.bn1(self.linear1(x)))
        out = self.dropout(out)
        out = self.bn2(self.linear2(out))
        out = self.act2(out + residual)
        return out


class TabularResNet(nn.Module):
    """Deep Tabular ResNet with residual skip connections for multiclass damage classification."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: tuple = (256, 256, 128),
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.SiLU(),
            nn.Dropout(dropout_rate / 2),
        )

        blocks = []
        current_dim = hidden_dims[0]
        for next_dim in hidden_dims[1:]:
            blocks.append(ResNetBlock(current_dim, next_dim, dropout_rate=dropout_rate))
            current_dim = next_dim
        self.res_blocks = nn.Sequential(*blocks)

        self.head = nn.Sequential(
            nn.BatchNorm1d(current_dim),
            nn.Dropout(dropout_rate),
            nn.Linear(current_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(x)
        x = self.res_blocks(x)
        logits = self.head(x)
        return logits


def get_neural_model_and_optimizer(
    input_dim: int,
    num_classes: int,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    epochs: int = 40,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    model = TabularResNet(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dims=(256, 256, 128),
        dropout_rate=0.2,
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )
    return model, criterion, optimizer, scheduler


# =====================================================================
# 2. Main Execution Pipeline
# =====================================================================
def main():
    set_seed(42)
    os.makedirs("./working", exist_ok=True)
    os.makedirs("./submission", exist_ok=True)

    # 1. Load Data
    train_path = "./input/train.parquet"
    test_path = "./input/test.parquet"

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    target_col = "Damage Description (total)"
    if target_col not in train_df.columns:
        candidates = [
            c for c in train_df.columns if "damage" in c.lower() and "desc" in c.lower()
        ]
        target_col = candidates[0] if candidates else train_df.columns[-1]

    # Target processing
    y_raw = train_df[target_col].copy()
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_raw.fillna(-1).astype(str))
    num_classes = len(label_encoder.classes_)

    train_features = train_df.drop(columns=[target_col], errors="ignore").copy()
    test_features = test_df.copy()

    feature_cols = [
        col for col in train_features.columns if col in test_features.columns
    ]
    train_features = train_features[feature_cols]
    test_features = test_features[feature_cols]

    n_train = len(train_features)
    n_test = len(test_features)
    combined_df = pd.concat([train_features, test_features], axis=0, ignore_index=True)

    # 2. Seismological & Domain Feature Engineering
    engineered_df = pd.DataFrame(index=combined_df.index)
    col_map = {c.lower(): c for c in combined_df.columns}

    lat_col = next((col_map[k] for k in col_map if "latitude" in k or k == "lat"), None)
    lon_col = next(
        (col_map[k] for k in col_map if "longitude" in k or k == "lon" or "long" in k),
        None,
    )
    depth_col = next(
        (col_map[k] for k in col_map if "depth" in k or "focal" in k), None
    )
    mag_col = next(
        (
            col_map[k]
            for k in col_map
            if "eq_primary" in k or "magnitude" in k or "mag" in k
        ),
        None,
    )
    year_col = next((col_map[k] for k in col_map if "year" in k), None)
    month_col = next((col_map[k] for k in col_map if "month" in k), None)

    # Spatial: 3D Spherical coordinates
    if lat_col and lon_col:
        lat_rad = np.radians(
            pd.to_numeric(combined_df[lat_col], errors="coerce").fillna(0)
        )
        lon_rad = np.radians(
            pd.to_numeric(combined_df[lon_col], errors="coerce").fillna(0)
        )
        engineered_df["geo_x"] = np.cos(lat_rad) * np.cos(lon_rad)
        engineered_df["geo_y"] = np.cos(lat_rad) * np.sin(lon_rad)
        engineered_df["geo_z"] = np.sin(lat_rad)
        engineered_df["abs_latitude"] = np.abs(lat_rad)

    intensity_col = next(
        (col_map[k] for k in col_map if "intensity" in k or "mmi" in k), None
    )
    country_col = next((col_map[k] for k in col_map if "country" in k), None)

    # Seismology: Energy proxy, GMPE PGA proxy, empirical attenuation & depth dissipation
    calc_intensity = None
    if mag_col:
        mag = pd.to_numeric(combined_df[mag_col], errors="coerce").fillna(5.0)
        engineered_df["seismic_energy_log10"] = 4.8 + 1.5 * mag
        engineered_df["seismic_energy_proxy"] = np.power(
            10.0, np.clip(1.5 * (mag - 5.0), -10, 10)
        )

        if depth_col:
            depth = pd.to_numeric(combined_df[depth_col], errors="coerce").fillna(10.0)
            depth_clipped = np.clip(depth, 1.0, 1000.0)
            hypo_dist = np.sqrt(depth_clipped**2 + 100.0)
            engineered_df["depth_log"] = np.log1p(depth_clipped)
            engineered_df["hypo_dist"] = hypo_dist

            # Physics-grounded empirical ground motion attenuation: I_calc = 1.5 * M - 1.8 * ln(sqrt(h^2 + 100))
            calc_intensity = 1.5 * mag - 1.8 * np.log(hypo_dist)
            engineered_df["attenuated_calc_intensity"] = calc_intensity.astype(np.float32)
            engineered_df["energy_depth_dissipation_ratio"] = (
                (4.8 + 1.5 * mag) / np.log(depth_clipped + 10.0)
            ).astype(np.float32)
            engineered_df["surface_energy_proxy"] = (
                engineered_df["seismic_energy_proxy"] / (depth_clipped**1.5)
            ).astype(np.float32)
            # Empirical GMPE Peak Ground Acceleration (PGA) Proxy
            engineered_df["gmpe_pga_proxy"] = (
                np.power(10.0, np.clip(0.5 * mag, -5.0, 10.0)) / np.power(hypo_dist, 1.5)
            ).astype(np.float32)
            engineered_df["is_shallow_eq"] = (depth < 70).astype(int)
            engineered_df["is_deep_eq"] = (depth >= 300).astype(int)

        if intensity_col:
            intensity = pd.to_numeric(combined_df[intensity_col], errors="coerce").fillna(0)
            engineered_df["mag_intensity_discrepancy"] = (mag - intensity).astype(np.float32)
            if calc_intensity is not None:
                engineered_df["calc_vs_observed_intensity_diff"] = (
                    calc_intensity - intensity
                ).astype(np.float32)

    # Multi-indicator casualty/damage interactions & missingness patterns
    human_num_cols = [
        c for c in combined_df.columns
        if any(k in c.lower() for k in ["death", "injur", "miss"]) and "desc" not in c.lower()
    ]
    damage_num_cols = [
        c for c in combined_df.columns
        if any(k in c.lower() for k in ["house", "damage", "cost", "million", "$mil"]) and "desc" not in c.lower()
    ]
    all_indicator_cols = [
        c for c in combined_df.columns
        if any(k in c.lower() for k in ["death", "injur", "miss", "house", "damage", "cost", "million", "$mil"])
        and c != target_col
    ]

    total_human_count = np.zeros(len(combined_df), dtype=np.float32)
    for col in human_num_cols:
        val = pd.to_numeric(combined_df[col], errors="coerce").fillna(0).clip(lower=0)
        total_human_count += val.values.astype(np.float32)
        engineered_df[f"{col}_log1p"] = np.log1p(val).astype(np.float32)

    engineered_df["composite_human_impact"] = np.log1p(total_human_count).astype(np.float32)

    total_damage_count = np.zeros(len(combined_df), dtype=np.float32)
    for col in damage_num_cols:
        val = pd.to_numeric(combined_df[col], errors="coerce").fillna(0).clip(lower=0)
        total_damage_count += val.values.astype(np.float32)
        engineered_df[f"{col}_log1p"] = np.log1p(val).astype(np.float32)

    engineered_df["composite_damage_impact"] = np.log1p(total_damage_count).astype(np.float32)

    if all_indicator_cols:
        engineered_df["disaster_missing_indicator_count"] = (
            combined_df[all_indicator_cols].isna().sum(axis=1).astype(np.float32)
        )
        engineered_df["disaster_available_indicator_ratio"] = (
            combined_df[all_indicator_cols].notna().mean(axis=1).astype(np.float32)
        )

    # Hierarchical country-level aggregation statistics
    if country_col:
        country_series = combined_df[country_col].fillna("__UNKNOWN__").astype(str)
        country_counts = country_series.value_counts()
        engineered_df["country_event_count"] = (
            country_series.map(country_counts).fillna(1).astype(np.float32)
        )
        if mag_col:
            mag_num = pd.to_numeric(combined_df[mag_col], errors="coerce")
            engineered_df["country_mag_mean"] = (
                country_series.map(combined_df.groupby(country_series)[mag_col].mean())
                .fillna(mag_num.mean())
                .astype(np.float32)
            )
            engineered_df["country_mag_std"] = (
                country_series.map(combined_df.groupby(country_series)[mag_col].std())
                .fillna(0.0)
                .astype(np.float32)
            )
        if depth_col:
            depth_num = pd.to_numeric(combined_df[depth_col], errors="coerce")
            engineered_df["country_depth_mean"] = (
                country_series.map(combined_df.groupby(country_series)[depth_col].mean())
                .fillna(depth_num.mean())
                .astype(np.float32)
            )
            engineered_df["country_depth_std"] = (
                country_series.map(combined_df.groupby(country_series)[depth_col].std())
                .fillna(0.0)
                .astype(np.float32)
            )

    # Granular text representation via TF-IDF + TruncatedSVD
    text_cols = [
        c
        for c in combined_df.columns
        if any(k in c.lower() for k in ["location", "name", "region", "area"])
    ]
    if text_cols:
        text_data = (
            combined_df[text_cols].fillna("").astype(str).agg(" ".join, axis=1)
        )
        tfidf = TfidfVectorizer(
            ngram_range=(2, 4), analyzer="char_wb", min_df=2, max_features=5000
        )
        tfidf_mat = tfidf.fit_transform(text_data)
        svd = TruncatedSVD(n_components=12, random_state=42)
        svd_feats = svd.fit_transform(tfidf_mat)
        for comp_i in range(12):
            engineered_df[f"text_svd_{comp_i}"] = svd_feats[:, comp_i].astype(
                np.float32
            )

    # Temporal: Era segmentation
    if year_col:
        year = pd.to_numeric(combined_df[year_col], errors="coerce").fillna(1900)
        engineered_df["year_normalized"] = year
        engineered_df["is_ancient"] = (year < 1800).astype(int)
        engineered_df["is_early_instrumental"] = (
            (year >= 1800) & (year < 1964)
        ).astype(int)
        engineered_df["is_modern_wwssn"] = (year >= 1964).astype(int)

    if month_col:
        month = pd.to_numeric(combined_df[month_col], errors="coerce").fillna(1)
        engineered_df["month_sin"] = np.sin(2 * np.pi * month / 12.0)
        engineered_df["month_cos"] = np.cos(2 * np.pi * month / 12.0)

    # Missingness Indicators
    for col in combined_df.columns:
        if combined_df[col].isna().sum() > 0:
            engineered_df[f"{col}_isna"] = combined_df[col].isna().astype(np.float32)

    # Separate numeric and categorical features
    numeric_cols = []
    categorical_cols = []
    for col in combined_df.columns:
        if pd.api.types.is_numeric_dtype(combined_df[col]):
            numeric_cols.append(col)
        else:
            converted = pd.to_numeric(combined_df[col], errors="coerce")
            if converted.notna().sum() > 0.5 * len(combined_df):
                combined_df[col] = converted
                numeric_cols.append(col)
            else:
                categorical_cols.append(col)

    features_df = pd.concat([combined_df[numeric_cols], engineered_df], axis=1)

    # Categorical Frequency & Factorization
    for cat_col in categorical_cols:
        cat_series = combined_df[cat_col].fillna("__MISSING__").astype(str)
        freq = cat_series.value_counts(normalize=True).to_dict()
        features_df[f"{cat_col}_freq"] = cat_series.map(freq).astype(np.float32)
        cat_encoded, _ = pd.factorize(cat_series)
        features_df[f"{cat_col}_code"] = cat_encoded.astype(np.int32)

    X_train = features_df.iloc[:n_train].copy().reset_index(drop=True)
    X_test = features_df.iloc[n_train:].copy().reset_index(drop=True)

    # 3. Leakage-Free Out-of-Fold Target Encoding
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_splits = list(skf.split(X_train, y_train))

    high_card_cols = [c for c in categorical_cols if combined_df[c].nunique() > 3]
    for cat_col in high_card_cols:
        cat_series_train = (
            combined_df.iloc[:n_train][cat_col].fillna("__MISSING__").astype(str)
        )
        cat_series_test = (
            combined_df.iloc[n_train:][cat_col].fillna("__MISSING__").astype(str)
        )

        oof_target_enc = np.zeros((n_train, num_classes), dtype=np.float32)
        test_target_enc = np.zeros((n_test, num_classes), dtype=np.float32)
        global_priors = np.bincount(y_train, minlength=num_classes) / len(y_train)

        for tr_idx, val_idx in oof_splits:
            tr_cat = cat_series_train.iloc[tr_idx]
            tr_y = y_train[tr_idx]
            val_cat = cat_series_train.iloc[val_idx]

            for class_idx in range(num_classes):
                class_binary = (tr_y == class_idx).astype(float)
                stats = (
                    pd.DataFrame({"cat": tr_cat, "target": class_binary})
                    .groupby("cat")["target"]
                    .agg(["count", "mean"])
                )
                weight = 10.0
                smooth_enc = (
                    stats["count"] * stats["mean"] + weight * global_priors[class_idx]
                ) / (stats["count"] + weight)
                mapping = smooth_enc.to_dict()
                oof_target_enc[val_idx, class_idx] = (
                    val_cat.map(mapping).fillna(global_priors[class_idx]).values
                )

        for class_idx in range(num_classes):
            class_binary = (y_train == class_idx).astype(float)
            stats = (
                pd.DataFrame({"cat": cat_series_train, "target": class_binary})
                .groupby("cat")["target"]
                .agg(["count", "mean"])
            )
            weight = 10.0
            smooth_enc = (
                stats["count"] * stats["mean"] + weight * global_priors[class_idx]
            ) / (stats["count"] + weight)
            mapping = smooth_enc.to_dict()
            test_target_enc[:, class_idx] = (
                cat_series_test.map(mapping).fillna(global_priors[class_idx]).values
            )

            X_train[f"{cat_col}_te_class_{class_idx}"] = oof_target_enc[:, class_idx]
            X_test[f"{cat_col}_te_class_{class_idx}"] = test_target_enc[:, class_idx]

    # Spatial k-NN and Spherical Haversine neighborhood features (Leak-free CV computation)
    lat_arr = pd.to_numeric(combined_df[lat_col], errors="coerce").fillna(0.0).values if lat_col else np.zeros(len(combined_df))
    lon_arr = pd.to_numeric(combined_df[lon_col], errors="coerce").fillna(0.0).values if lon_col else np.zeros(len(combined_df))
    mag_arr = pd.to_numeric(combined_df[mag_col], errors="coerce").fillna(5.0).values if mag_col else np.full(len(combined_df), 5.0)

    train_lats, train_lons, train_mags = lat_arr[:n_train], lon_arr[:n_train], mag_arr[:n_train]
    test_lats, test_lons, test_mags = lat_arr[n_train:], lon_arr[n_train:], mag_arr[n_train:]

    train_coords_rad = np.radians(np.column_stack([train_lats, train_lons]))
    test_coords_rad = np.radians(np.column_stack([test_lats, test_lons]))

    r_earth_km = 6371.0
    radii_km = [250.0, 1000.0]

    for r_km in radii_km:
        r_rad = r_km / r_earth_km
        oof_count = np.zeros(n_train, dtype=np.float32)
        oof_mean_mag = np.zeros(n_train, dtype=np.float32)
        oof_max_mag = np.zeros(n_train, dtype=np.float32)

        for tr_idx, val_idx in oof_splits:
            tree_fold = BallTree(train_coords_rad[tr_idx], metric="haversine")
            ind_val = tree_fold.query_radius(train_coords_rad[val_idx], r=r_rad)
            tr_mags_fold = train_mags[tr_idx]

            for i_local, neighbors in enumerate(ind_val):
                idx_val = val_idx[i_local]
                n_count = len(neighbors)
                oof_count[idx_val] = n_count
                if n_count > 0:
                    nb_m = tr_mags_fold[neighbors]
                    oof_mean_mag[idx_val] = np.mean(nb_m)
                    oof_max_mag[idx_val] = np.max(nb_m)
                else:
                    oof_mean_mag[idx_val] = 5.0
                    oof_max_mag[idx_val] = 5.0

        tree_full = BallTree(train_coords_rad, metric="haversine")
        ind_test = tree_full.query_radius(test_coords_rad, r=r_rad)
        test_count = np.zeros(n_test, dtype=np.float32)
        test_mean_mag = np.zeros(n_test, dtype=np.float32)
        test_max_mag = np.zeros(n_test, dtype=np.float32)

        for i_test, neighbors in enumerate(ind_test):
            n_count = len(neighbors)
            test_count[i_test] = n_count
            if n_count > 0:
                nb_m = train_mags[neighbors]
                test_mean_mag[i_test] = np.mean(nb_m)
                test_max_mag[i_test] = np.max(nb_m)
            else:
                test_mean_mag[i_test] = 5.0
                test_max_mag[i_test] = 5.0

        tag = int(r_km)
        X_train[f"eq_count_{tag}km"] = oof_count
        X_train[f"eq_mean_mag_{tag}km"] = oof_mean_mag
        X_train[f"eq_max_mag_{tag}km"] = oof_max_mag
        X_test[f"eq_count_{tag}km"] = test_count
        X_test[f"eq_mean_mag_{tag}km"] = test_mean_mag
        X_test[f"eq_max_mag_{tag}km"] = test_max_mag

    if (
        "geo_x" in X_train.columns
        and "geo_y" in X_train.columns
        and "geo_z" in X_train.columns
    ):
        coords_train = X_train[["geo_x", "geo_y", "geo_z"]].fillna(0).values
        coords_test = X_test[["geo_x", "geo_y", "geo_z"]].fillna(0).values

        oof_dist_5 = np.zeros(n_train, dtype=np.float32)
        oof_dist_10 = np.zeros(n_train, dtype=np.float32)

        for tr_idx, val_idx in oof_splits:
            nn_fold = NearestNeighbors(n_neighbors=10, metric="euclidean", n_jobs=-1)
            nn_fold.fit(coords_train[tr_idx])
            val_dists, _ = nn_fold.kneighbors(coords_train[val_idx])
            oof_dist_5[val_idx] = val_dists[:, 4].astype(np.float32)
            oof_dist_10[val_idx] = val_dists[:, 9].astype(np.float32)

        nn_full = NearestNeighbors(n_neighbors=10, metric="euclidean", n_jobs=-1)
        nn_full.fit(coords_train)
        test_dists, _ = nn_full.kneighbors(coords_test)

        X_train["spatial_density_dist_5"] = oof_dist_5
        X_train["spatial_density_dist_10"] = oof_dist_10
        X_test["spatial_density_dist_5"] = test_dists[:, 4].astype(np.float32)
        X_test["spatial_density_dist_10"] = test_dists[:, 9].astype(np.float32)

    # Imputation & Scaling for Neural Network
    imputer = SimpleImputer(strategy="median")
    imputer.fit(X_train)
    X_tr_imp = imputer.transform(X_train)
    X_te_imp = imputer.transform(X_test)

    scaler = StandardScaler()
    scaler.fit(X_tr_imp)
    X_train_scaled = scaler.transform(X_tr_imp).astype(np.float32)
    X_test_scaled = scaler.transform(X_te_imp).astype(np.float32)

    # 4. Multi-Model Cross-Validation Protocol
    n_folds = len(oof_splits)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    oof_lgb = np.zeros((n_train, num_classes), dtype=np.float32)
    oof_xgb = np.zeros((n_train, num_classes), dtype=np.float32)
    oof_cat = np.zeros((n_train, num_classes), dtype=np.float32)
    oof_nn = np.zeros((n_train, num_classes), dtype=np.float32)

    test_lgb = np.zeros((n_test, num_classes), dtype=np.float32)
    test_xgb = np.zeros((n_test, num_classes), dtype=np.float32)
    test_cat = np.zeros((n_test, num_classes), dtype=np.float32)
    test_nn = np.zeros((n_test, num_classes), dtype=np.float32)

    for fold, (tr_idx, val_idx) in enumerate(oof_splits):
        X_tr_df, y_tr = X_train.iloc[tr_idx], y_train[tr_idx]
        X_val_df, y_val = X_train.iloc[val_idx], y_train[val_idx]

        X_tr_sc, X_val_sc = X_train_scaled[tr_idx], X_train_scaled[val_idx]

        # --- A. LightGBM ---
        lgb_clf = lgb.LGBMClassifier(
            objective="multiclass",
            num_class=num_classes,
            metric="multi_logloss",
            boosting_type="gbdt",
            n_estimators=1200,
            learning_rate=0.03,
            num_leaves=31,
            max_depth=7,
            subsample=0.8,
            colsample_bytree=0.75,
            reg_alpha=0.2,
            reg_lambda=1.5,
            min_child_samples=20,
            random_state=42 + fold,
            n_jobs=-1,
            verbose=-1,
        )
        lgb_callbacks = [
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0),
        ]
        lgb_clf.fit(
            X_tr_df, y_tr, eval_set=[(X_val_df, y_val)], callbacks=lgb_callbacks
        )

        oof_lgb[val_idx] = lgb_clf.predict_proba(X_val_df)
        test_lgb += lgb_clf.predict_proba(X_test) / n_folds

        # --- B. XGBoost ---
        xgb_clf = xgb.XGBClassifier(
            objective="multi:softprob",
            num_class=num_classes,
            eval_metric="mlogloss",
            n_estimators=1200,
            learning_rate=0.03,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.75,
            reg_alpha=0.2,
            reg_lambda=1.5,
            min_child_weight=2,
            random_state=42 + fold,
            n_jobs=-1,
            tree_method="hist",
            early_stopping_rounds=50,
        )
        xgb_clf.fit(X_tr_df, y_tr, eval_set=[(X_val_df, y_val)], verbose=False)

        oof_xgb[val_idx] = xgb_clf.predict_proba(X_val_df)
        test_xgb += xgb_clf.predict_proba(X_test) / n_folds

        # --- C. CatBoost ---
        cat_clf = cb.CatBoostClassifier(
            loss_function="MultiClass",
            eval_metric="MultiClass",
            iterations=1200,
            learning_rate=0.035,
            depth=6,
            l2_leaf_reg=4.0,
            random_seed=42 + fold,
            thread_count=-1,
            verbose=False,
            early_stopping_rounds=50,
        )
        cat_clf.fit(X_tr_df, y_tr, eval_set=(X_val_df, y_val), verbose=False)

        oof_cat[val_idx] = cat_clf.predict_proba(X_val_df)
        test_cat += cat_clf.predict_proba(X_test) / n_folds

        # --- D. Tabular ResNet ---
        input_dim = X_tr_sc.shape[1]
        nn_model, criterion, optimizer, scheduler = get_neural_model_and_optimizer(
            input_dim=input_dim,
            num_classes=num_classes,
            lr=1e-3,
            weight_decay=1e-4,
            epochs=40,
            device=device,
        )

        tr_dataset = TensorDataset(
            torch.tensor(X_tr_sc, dtype=torch.float32),
            torch.tensor(y_tr, dtype=torch.long),
        )
        val_dataset = TensorDataset(
            torch.tensor(X_val_sc, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.long),
        )

        tr_loader = DataLoader(
            tr_dataset, batch_size=128, shuffle=True, drop_last=False
        )
        val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

        best_val_loss = float("inf")
        best_nn_weights = None

        for epoch in range(40):
            nn_model.train()
            for batch_x, batch_y in tr_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                logits = nn_model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
            scheduler.step()

            nn_model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    logits = nn_model(batch_x)
                    val_loss += criterion(logits, batch_y).item() * len(batch_y)
            val_loss /= len(val_dataset)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_nn_weights = copy.deepcopy(nn_model.state_dict())

        nn_model.load_state_dict(best_nn_weights)
        nn_model.eval()
        with torch.no_grad():
            val_tensor = torch.tensor(X_val_sc, dtype=torch.float32).to(device)
            test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
            val_probs = torch.softmax(nn_model(val_tensor), dim=-1).cpu().numpy()
            test_probs = torch.softmax(nn_model(test_tensor), dim=-1).cpu().numpy()

        oof_nn[val_idx] = val_probs
        test_nn += test_probs / n_folds

        del nn_model, best_nn_weights
        gc.collect()

    # 5. Meta-Learner Stacking Layer & Nelder-Mead Optimization
    X_meta_train = np.hstack([oof_lgb, oof_xgb, oof_cat, oof_nn])
    X_meta_test = np.hstack([test_lgb, test_xgb, test_cat, test_nn])

    oof_meta = np.zeros((n_train, num_classes), dtype=np.float32)
    test_meta = np.zeros((n_test, num_classes), dtype=np.float32)

    meta_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for m_tr_idx, m_val_idx in meta_skf.split(X_meta_train, y_train):
        meta_lr = LogisticRegression(
            C=1.0,
            penalty="l2",
            solver="lbfgs",
            max_iter=1000,
            random_state=42,
        )
        meta_lr.fit(X_meta_train[m_tr_idx], y_train[m_tr_idx])
        oof_meta[m_val_idx] = meta_lr.predict_proba(X_meta_train[m_val_idx])
        test_meta += meta_lr.predict_proba(X_meta_test) / meta_skf.n_splits

    base_oofs = [oof_lgb, oof_xgb, oof_cat, oof_nn, oof_meta]
    base_tests = [test_lgb, test_xgb, test_cat, test_nn, test_meta]
    n_models = len(base_oofs)

    def ensemble_loss(weights):
        w = np.exp(weights) / np.sum(np.exp(weights))
        blend = sum(w[i] * base_oofs[i] for i in range(n_models))
        eps = 1e-15
        blend_clipped = np.clip(blend, eps, 1.0 - eps)
        blend_clipped /= blend_clipped.sum(axis=1, keepdims=True)
        return -np.mean(np.log(blend_clipped[np.arange(n_train), y_train]))

    init_weights = np.zeros(n_models)
    opt_res = minimize(
        ensemble_loss, init_weights, method="Nelder-Mead", options={"maxiter": 1000}
    )
    best_weights = np.exp(opt_res.x) / np.sum(np.exp(opt_res.x))

    oof_ensemble_probs = sum(best_weights[i] * base_oofs[i] for i in range(n_models))
    test_ensemble_probs = sum(best_weights[i] * base_tests[i] for i in range(n_models))

    # Metric-Aligned Direct Accuracy Optimization via Class Multipliers
    def acc_objective(cw, probs, targets):
        cw = np.asarray(cw)
        scaled_p = probs * cw
        preds = np.argmax(scaled_p, axis=1)
        return -accuracy_score(targets, preds)

    best_cw = np.ones(num_classes, dtype=np.float64)
    best_acc = accuracy_score(y_train, np.argmax(oof_ensemble_probs, axis=1))

    for restart_seed in [42, 101, 777, 2024]:
        np.random.seed(restart_seed)
        init_cw = np.ones(num_classes) + np.random.uniform(-0.15, 0.15, size=num_classes)
        opt_cw = minimize(
            acc_objective,
            init_cw,
            args=(oof_ensemble_probs, y_train),
            method="Nelder-Mead",
            options={"maxiter": 1000, "xatol": 1e-4, "fatol": 1e-4},
        )
        cur_acc = -opt_cw.fun
        if cur_acc > best_acc:
            best_acc = cur_acc
            best_cw = opt_cw.x

    # Coordinate search refinement
    for _ in range(3):
        for c in range(num_classes):
            for step in [-0.2, -0.1, -0.05, -0.02, -0.01, 0.01, 0.02, 0.05, 0.1, 0.2]:
                cand = best_cw.copy()
                cand[c] += step
                if np.min(cand) > 0:
                    cand_acc = -acc_objective(cand, oof_ensemble_probs, y_train)
                    if cand_acc > best_acc:
                        best_acc = cand_acc
                        best_cw = cand

    oof_pred_classes = np.argmax(oof_ensemble_probs * best_cw, axis=1)
    final_accuracy = accuracy_score(y_train, oof_pred_classes)

    # 6. Generate Submission
    test_pred_classes = np.argmax(test_ensemble_probs * best_cw, axis=1)
    test_pred_labels = label_encoder.inverse_transform(test_pred_classes)

    submission_df = pd.DataFrame(
        {
            "id": np.arange(len(test_pred_labels)),
            "Damage Description (total)": test_pred_labels,
        }
    )
    submission_df.to_csv("./submission/submission.csv", index=False)

    print(f"Final Validation Score: {final_accuracy:.6f}")


if __name__ == "__main__":
    main()