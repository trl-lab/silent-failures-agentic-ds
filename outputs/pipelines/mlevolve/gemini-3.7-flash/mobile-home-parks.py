import json
import math
import os
import random
import re
import sys
import warnings
from catboost import CatBoostClassifier
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.sparse import hstack
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)


# =========================================================================
# 1. Diversified Decision Forest Model Suite
# =========================================================================
def get_lgbm_classifier(
    num_classes: int,
    random_state: int = 42,
    n_estimators: int = 1200,
    learning_rate: float = 0.03,
) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="multiclass",
        num_class=num_classes,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=31,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        min_child_samples=20,
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
    )


def get_lgbm_extra_classifier(
    num_classes: int,
    random_state: int = 42,
    n_estimators: int = 1200,
    learning_rate: float = 0.03,
) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="multiclass",
        num_class=num_classes,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=45,
        max_depth=7,
        subsample=0.7,
        subsample_freq=1,
        colsample_bytree=0.6,
        extra_trees=True,
        reg_alpha=0.5,
        reg_lambda=2.0,
        min_child_samples=15,
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
    )


def get_xgb_classifier(
    num_classes: int,
    random_state: int = 42,
    n_estimators: int = 1200,
    learning_rate: float = 0.03,
) -> XGBClassifier:
    return XGBClassifier(
        objective="multi:softprob",
        num_class=num_classes,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=0.5,
        reg_lambda=2.0,
        min_child_weight=4,
        random_state=random_state,
        n_jobs=-1,
        tree_method="hist",
        early_stopping_rounds=50,
        eval_metric="mlogloss",
    )


def get_catboost_classifier(
    num_classes: int,
    random_state: int = 42,
    n_estimators: int = 1200,
    learning_rate: float = 0.03,
) -> CatBoostClassifier:
    return CatBoostClassifier(
        iterations=n_estimators,
        learning_rate=learning_rate,
        depth=6,
        l2_leaf_reg=5.0,
        loss_function="MultiClass",
        random_seed=random_state,
        verbose=0,
        early_stopping_rounds=50,
        thread_count=-1,
    )


def compute_spatial_target_features(
    train_coords_rad: np.ndarray,
    y_train: np.ndarray,
    query_coords_rad: np.ndarray,
    is_self: bool = False,
    ks=(7, 15, 30),
    sigmas_km=(5.0, 15.0, 50.0, 150.0, 500.0),
    num_classes: int = 10,
) -> np.ndarray:
    k_query = min(100, len(train_coords_rad))
    n_neighbors = min(k_query + 1 if is_self else k_query, len(train_coords_rad))
    nn_model = NearestNeighbors(
        n_neighbors=n_neighbors, metric="haversine", algorithm="ball_tree"
    )
    nn_model.fit(train_coords_rad)
    distances, indices = nn_model.kneighbors(query_coords_rad)

    if is_self:
        distances = distances[:, 1:]
        indices = indices[:, 1:]

    dist_km = distances * 6371.0088
    neighbor_labels = y_train[indices]
    class_priors = np.bincount(y_train, minlength=num_classes) / len(y_train)

    features = []

    # 1. Discrete top-K proportions
    for k in ks:
        if k <= neighbor_labels.shape[1]:
            sub_labels = neighbor_labels[:, :k]
            for c in range(num_classes):
                prop = (sub_labels == c).mean(axis=1, dtype=np.float32)
                features.append(prop)

    # 2. Gaussian kernel decay IDW across multi-scale bandwidths
    for sigma in sigmas_km:
        w = np.exp(-0.5 * (dist_km / sigma) ** 2)
        w_sum = w.sum(axis=1, keepdims=True) + 1e-6
        for c in range(num_classes):
            mask = (neighbor_labels == c).astype(np.float32)
            weighted_count = (w * mask).sum(axis=1, keepdims=True)
            smooth_m = 1.0
            prob = (weighted_count + smooth_m * class_priors[c]) / (w_sum + smooth_m)
            features.append(prob.ravel().astype(np.float32))

    # 3. Inverse distance weighted (1 / (d + 1))
    w_inv = 1.0 / (dist_km + 1.0)
    w_inv_sum = w_inv.sum(axis=1, keepdims=True) + 1e-6
    for c in range(num_classes):
        mask = (neighbor_labels == c).astype(np.float32)
        weighted_count = (w_inv * mask).sum(axis=1, keepdims=True)
        prob = (weighted_count + 1.0 * class_priors[c]) / (w_inv_sum + 1.0)
        features.append(prob.ravel().astype(np.float32))

    return np.column_stack(features).astype(np.float32)


def compute_oof_target_encoding(
    train_cats: pd.DataFrame,
    y_train: np.ndarray,
    val_cats: pd.DataFrame,
    test_cats: pd.DataFrame,
    cat_cols: list,
    num_classes: int,
    smoothing: float = 10.0,
    n_inner_splits: int = 5,
    seed: int = 42,
):
    n_tr = len(train_cats)
    n_va = len(val_cats)
    n_te = len(test_cats)

    class_priors = np.bincount(y_train, minlength=num_classes) / n_tr

    tr_encoded_list = []
    va_encoded_list = []
    te_encoded_list = []

    inner_kf = StratifiedKFold(
        n_splits=n_inner_splits, shuffle=True, random_state=seed
    )

    for col in cat_cols:
        col_tr = train_cats[col].values
        col_va = val_cats[col].values
        col_te = test_cats[col].values

        oof_tr_te = np.zeros((n_tr, num_classes), dtype=np.float32)
        for in_trn_idx, in_val_idx in inner_kf.split(train_cats, y_train):
            in_col_trn = col_tr[in_trn_idx]
            in_y_trn = y_train[in_trn_idx]
            in_col_val = col_tr[in_val_idx]

            df_in = pd.DataFrame({"cat": in_col_trn, "target": in_y_trn})
            counts = df_in["cat"].value_counts()
            for c in range(num_classes):
                c_counts = df_in[df_in["target"] == c]["cat"].value_counts()
                smoothed_map = (c_counts + smoothing * class_priors[c]) / (
                    counts + smoothing
                )
                smoothed_dict = smoothed_map.to_dict()
                oof_tr_te[in_val_idx, c] = (
                    pd.Series(in_col_val)
                    .map(smoothed_dict)
                    .fillna(class_priors[c])
                    .values
                )

        df_full = pd.DataFrame({"cat": col_tr, "target": y_train})
        full_counts = df_full["cat"].value_counts()

        va_te = np.zeros((n_va, num_classes), dtype=np.float32)
        te_te = np.zeros((n_te, num_classes), dtype=np.float32)

        for c in range(num_classes):
            c_counts = df_full[df_full["target"] == c]["cat"].value_counts()
            smoothed_map = (c_counts + smoothing * class_priors[c]) / (
                full_counts + smoothing
            )
            smoothed_dict = smoothed_map.to_dict()
            va_te[:, c] = (
                pd.Series(col_va).map(smoothed_dict).fillna(class_priors[c]).values
            )
            te_te[:, c] = (
                pd.Series(col_te).map(smoothed_dict).fillna(class_priors[c]).values
            )

        tr_encoded_list.append(oof_tr_te)
        va_encoded_list.append(va_te)
        te_encoded_list.append(te_te)

    trn_out = np.hstack(tr_encoded_list)
    val_out = np.hstack(va_encoded_list)
    test_out = np.hstack(te_encoded_list)
    return trn_out, val_out, test_out


# =========================================================================
# 2. Main End-to-End Pipeline
# =========================================================================
def main():
    set_seed(42)

    input_dir = "./input"
    working_dir = "./working"
    submission_dir = "./submission"
    os.makedirs(working_dir, exist_ok=True)
    os.makedirs(submission_dir, exist_ok=True)

    train_path = os.path.join(input_dir, "train.parquet")
    test_path = os.path.join(input_dir, "test.parquet")

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    target_col = "SIZE"
    if target_col not in train_df.columns:
        for c in train_df.columns:
            if c.upper() == "SIZE":
                target_col = c
                break

    # Target label encoding
    y_raw = train_df[target_col].astype(str)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_raw)
    classes = np.array(label_encoder.classes_)
    num_classes = len(classes)

    n_train = len(train_df)
    n_test = len(test_df)

    # -------------------------------------------------------------
    # Feature Engineering
    # -------------------------------------------------------------
    lat_col, lon_col = None, None
    for c in train_df.columns:
        if c.upper() in ["LATITUDE", "LAT", "Y", "POINT_Y"] and lat_col is None:
            lat_col = c
        elif (
            c.upper() in ["LONGITUDE", "LON", "LNG", "LONG", "X", "POINT_X"]
            and lon_col is None
        ):
            lon_col = c

    if lat_col and lon_col:
        for df in [train_df, test_df]:
            df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
            df[lon_col] = pd.to_numeric(df[lon_col], errors="coerce")
            invalid_coords = (
                (df[lat_col] < -90)
                | (df[lat_col] > 90)
                | (df[lon_col] < -180)
                | (df[lon_col] > 180)
                | ((df[lat_col] == 0) & (df[lon_col] == 0))
            )
            df.loc[invalid_coords, lat_col] = np.nan
            df.loc[invalid_coords, lon_col] = np.nan

        lat_median = train_df[lat_col].median()
        lon_median = train_df[lon_col].median()
        train_df[lat_col] = train_df[lat_col].fillna(lat_median)
        train_df[lon_col] = train_df[lon_col].fillna(lon_median)
        test_df[lat_col] = test_df[lat_col].fillna(lat_median)
        test_df[lon_col] = test_df[lon_col].fillna(lon_median)

    train_feat = pd.DataFrame(index=train_df.index)
    test_feat = pd.DataFrame(index=test_df.index)

    # Geospatial engineering
    if lat_col and lon_col:
        for df, feat in [(train_df, train_feat), (test_df, test_feat)]:
            lat_rad = np.radians(df[lat_col].values)
            lon_rad = np.radians(df[lon_col].values)
            feat["lat"] = df[lat_col].values
            feat["lon"] = df[lon_col].values
            feat["coord_x"] = np.cos(lat_rad) * np.cos(lon_rad)
            feat["coord_y"] = np.cos(lat_rad) * np.sin(lon_rad)
            feat["coord_z"] = np.sin(lat_rad)

            for angle in [15, 30, 45, 60, 75, 120, 135, 150]:
                rad = np.radians(angle)
                feat[f"rot_{angle}"] = df[lat_col].values * np.cos(rad) + df[
                    lon_col
                ].values * np.sin(rad)

        train_coords = train_df[[lat_col, lon_col]].values
        test_coords = test_df[[lat_col, lon_col]].values

        for n_clusters in [10, 25, 50]:
            kmeans = MiniBatchKMeans(
                n_clusters=n_clusters, random_state=42, batch_size=1024, n_init=3
            )
            kmeans.fit(train_coords)
            train_feat[f"kmeans_cluster_{n_clusters}"] = kmeans.predict(train_coords)
            test_feat[f"kmeans_cluster_{n_clusters}"] = kmeans.predict(test_coords)
            train_feat[f"kmeans_dist_min_{n_clusters}"] = kmeans.transform(
                train_coords
            ).min(axis=1)
            test_feat[f"kmeans_dist_min_{n_clusters}"] = kmeans.transform(
                test_coords
            ).min(axis=1)

        train_rads = np.radians(train_coords)
        test_rads = np.radians(test_coords)

        k_neighbors = 55
        nn = NearestNeighbors(
            n_neighbors=k_neighbors, metric="haversine", algorithm="ball_tree"
        )
        nn.fit(train_rads)

        train_distances, _ = nn.kneighbors(train_rads, n_neighbors=k_neighbors)
        train_distances_km = train_distances[:, 1:] * 6371.0088

        test_distances, _ = nn.kneighbors(test_rads, n_neighbors=k_neighbors - 1)
        test_distances_km = test_distances * 6371.0088

        for k in [1, 2, 3, 5, 10, 20, 50]:
            train_feat[f"knn_dist_{k}"] = train_distances_km[:, k - 1]
            test_feat[f"knn_dist_{k}"] = test_distances_km[:, k - 1]
            if k > 1:
                train_feat[f"knn_dist_ratio_{k}_1"] = train_distances_km[:, k - 1] / (
                    train_distances_km[:, 0] + 1e-3
                )
                test_feat[f"knn_dist_ratio_{k}_1"] = test_distances_km[:, k - 1] / (
                    test_distances_km[:, 0] + 1e-3
                )

        for k in [5, 10, 25, 50]:
            train_feat[f"knn_dist_mean_{k}"] = train_distances_km[:, :k].mean(axis=1)
            test_feat[f"knn_dist_mean_{k}"] = test_distances_km[:, :k].mean(axis=1)
            train_feat[f"knn_dist_std_{k}"] = train_distances_km[:, :k].std(axis=1)
            test_feat[f"knn_dist_std_{k}"] = test_distances_km[:, :k].std(axis=1)

        # Concentric spatial density rings
        rings = [(0.0, 2.0), (2.0, 5.0), (5.0, 15.0), (15.0, 50.0), (50.0, 150.0)]
        for r_min, r_max in rings:
            col_name = f"ring_density_{int(r_min)}_{int(r_max)}km"
            train_feat[col_name] = (
                (train_distances_km >= r_min) & (train_distances_km < r_max)
            ).sum(axis=1)
            test_feat[col_name] = (
                (test_distances_km >= r_min) & (test_distances_km < r_max)
            ).sum(axis=1)

        for radius_km in [2.0, 5.0, 15.0, 50.0, 100.0, 150.0]:
            train_feat[f"density_within_{int(radius_km)}km"] = (
                train_distances_km < radius_km
            ).sum(axis=1)
            test_feat[f"density_within_{int(radius_km)}km"] = (
                test_distances_km < radius_km
            ).sum(axis=1)

    # Street address number magnitude
    addr_col = next((c for c in train_df.columns if "ADDRESS" in c.upper()), None)
    if addr_col:
        s_tr_addr = train_df[addr_col].fillna("").astype(str)
        s_te_addr = test_df[addr_col].fillna("").astype(str)
        num_tr = pd.to_numeric(s_tr_addr.str.extract(r"(\d+)")[0], errors="coerce").fillna(0)
        num_te = pd.to_numeric(s_te_addr.str.extract(r"(\d+)")[0], errors="coerce").fillna(0)
        train_feat["addr_street_num_log"] = np.log1p(num_tr).values
        test_feat["addr_street_num_log"] = np.log1p(num_te).values
        train_feat["addr_has_suite"] = s_tr_addr.str.lower().str.contains(
            r"\b(ste|suite|apt|unit|spc|space|lot|#)\b", regex=True
        ).astype(int).values
        test_feat["addr_has_suite"] = s_te_addr.str.lower().str.contains(
            r"\b(ste|suite|apt|unit|spc|space|lot|#)\b", regex=True
        ).astype(int).values

    # Lexical and text features
    name_col = None
    for c in train_df.columns:
        if c.upper() == "NAME":
            name_col = c
            break

    domain_keywords = [
        "rv",
        "senior",
        "55+",
        "phase",
        "acres",
        "estates",
        "co-op",
        "coop",
        "resort",
        "village",
        "manor",
        "gardens",
        "court",
        "park",
        "club",
        "ranch",
        "haven",
        "shores",
        "terrace",
        "villa",
        "plaza",
        "country",
        "community",
        "heights",
        "pointe",
        "woods",
        "meadows",
        "hills",
        "grove",
        "trailer",
        "mobile",
        "mhp",
        "lake",
        "springs",
        "valley",
        "creek",
        "pines",
        "palms",
        "oaks",
        "island",
        "bay",
        "cove",
        "royal",
        "golden",
        "sun",
        "green",
        "vista",
        "view",
        "forest",
        "shadow",
        "ridge",
        "sand",
        "west",
        "north",
        "south",
        "east",
        "hidden",
        "timber",
        "pine",
        "crest",
        "san",
        "del",
        "los",
        "casa",
        "camp",
        "lodge",
        "landing",
        "crossing",
        "trace",
        "hacienda",
        "playa",
        "colonial",
        "subdivision",
        "square",
        "center",
    ]

    for col in train_df.columns:
        if col == target_col:
            continue
        if train_df[col].dtype == object or str(train_df[col].dtype) == "string":
            s_train = train_df[col].fillna("").astype(str)
            s_test = test_df[col].fillna("").astype(str)

            train_feat[f"{col}_char_len"] = s_train.str.len()
            test_feat[f"{col}_char_len"] = s_test.str.len()
            train_feat[f"{col}_word_count"] = s_train.apply(lambda x: len(x.split()))
            test_feat[f"{col}_word_count"] = s_test.apply(lambda x: len(x.split()))
            train_feat[f"{col}_has_digits"] = s_train.str.contains(
                r"\d", regex=True
            ).astype(int)
            test_feat[f"{col}_has_digits"] = s_test.str.contains(
                r"\d", regex=True
            ).astype(int)

    if name_col:
        s_train_name = train_df[name_col].fillna("").astype(str).str.lower()
        s_test_name = test_df[name_col].fillna("").astype(str).str.lower()

        for kw in domain_keywords:
            clean_kw = re.escape(kw)
            train_feat[f"kw_{kw}"] = s_train_name.str.contains(
                r"(?:\b|_)" + clean_kw + r"(?:\b|_)", regex=True
            ).astype(int)
            test_feat[f"kw_{kw}"] = s_test_name.str.contains(
                r"(?:\b|_)" + clean_kw + r"(?:\b|_)", regex=True
            ).astype(int)

        corpus_cols = [
            c
            for c in ["NAME", "ADDRESS", "CITY", "COUNTY"]
            if c in train_df.columns or c.lower() in train_df.columns
        ]

        if corpus_cols:
            train_corpus = train_df[corpus_cols[0]].fillna("").astype(str)
            test_corpus = test_df[corpus_cols[0]].fillna("").astype(str)
            for col in corpus_cols[1:]:
                train_corpus = train_corpus + " " + train_df[col].fillna("").astype(str)
                test_corpus = test_corpus + " " + test_df[col].fillna("").astype(str)

            word_tfidf = TfidfVectorizer(
                max_features=1000,
                ngram_range=(1, 2),
                min_df=2,
                token_pattern=r"(?u)\b\w+\b",
            )
            char_tfidf = TfidfVectorizer(
                max_features=1000,
                analyzer="char_wb",
                ngram_range=(3, 5),
                min_df=2,
            )
            word_train = word_tfidf.fit_transform(train_corpus)
            word_test = word_tfidf.transform(test_corpus)
            char_train = char_tfidf.fit_transform(train_corpus)
            char_test = char_tfidf.transform(test_corpus)

            tfidf_train = hstack([word_train, char_train])
            tfidf_test = hstack([word_test, char_test])

            n_svd = 40
            svd = TruncatedSVD(n_components=n_svd, random_state=42)
            svd_train = svd.fit_transform(tfidf_train)
            svd_test = svd.transform(tfidf_test)

            for i in range(n_svd):
                train_feat[f"tfidf_svd_{i}"] = svd_train[:, i]
                test_feat[f"tfidf_svd_{i}"] = svd_test[:, i]

    # Categorical and hierarchical encodings
    cat_candidates = [
        "STATE",
        "COUNTY",
        "CITY",
        "ZIP",
        "TYPE",
        "STATUS",
        "SOURCE",
        "VAL_METHOD",
        "COUNTYFIPS",
    ]
    cat_cols_found = [
        c for c in train_df.columns if c.upper() in cat_candidates and c != target_col
    ]

    for col in cat_cols_found:
        s_train = train_df[col].fillna("MISSING").astype(str)
        s_test = test_df[col].fillna("MISSING").astype(str)

        freq_map = s_train.value_counts(normalize=True).to_dict()
        train_feat[f"{col}_freq"] = s_train.map(freq_map).fillna(0.0).astype(float)
        test_feat[f"{col}_freq"] = s_test.map(freq_map).fillna(0.0).astype(float)

        unique_vals = list(s_train.unique())
        val2idx = {val: idx for idx, val in enumerate(unique_vals)}
        train_feat[f"{col}_cat"] = s_train.map(val2idx).fillna(-1).astype(int)
        test_feat[f"{col}_cat"] = s_test.map(val2idx).fillna(-1).astype(int)

    zip_col = next((c for c in train_df.columns if c.upper() == "ZIP"), None)
    if zip_col:
        zip_clean_tr = (
            train_df[zip_col]
            .fillna("")
            .astype(str)
            .str.extract(r"(\d{5})")[0]
            .fillna("00000")
        )
        zip_clean_te = (
            test_df[zip_col]
            .fillna("")
            .astype(str)
            .str.extract(r"(\d{5})")[0]
            .fillna("00000")
        )

        train_feat["zip_num"] = (
            pd.to_numeric(zip_clean_tr, errors="coerce").fillna(0).astype(float)
        )
        test_feat["zip_num"] = (
            pd.to_numeric(zip_clean_te, errors="coerce").fillna(0).astype(float)
        )

        zip3_tr = zip_clean_tr.str[:3]
        zip3_te = zip_clean_te.str[:3]
        zip3_freq = zip3_tr.value_counts(normalize=True).to_dict()
        train_feat["zip3_freq"] = zip3_tr.map(zip3_freq).fillna(0.0).astype(float)
        test_feat["zip3_freq"] = zip3_te.map(zip3_freq).fillna(0.0).astype(float)
        train_df["ZIP3"] = zip3_tr
        test_df["ZIP3"] = zip3_te

    # Domain signal extraction: Telephone area codes and Date decomposition
    phone_col = next((c for c in train_df.columns if "PHONE" in c.upper() or "TELEPHONE" in c.upper()), None)
    if phone_col:
        phone_tr = train_df[phone_col].fillna("").astype(str)
        phone_te = test_df[phone_col].fillna("").astype(str)
        area_tr = phone_tr.str.extract(r"(\d{3})")[0].fillna("-1")
        area_te = phone_te.str.extract(r"(\d{3})")[0].fillna("-1")
        train_feat["phone_has"] = (phone_tr.str.len() > 5).astype(float)
        test_feat["phone_has"] = (phone_te.str.len() > 5).astype(float)
        area_freq = area_tr.value_counts(normalize=True).to_dict()
        train_feat["phone_area_freq"] = area_tr.map(area_freq).fillna(0.0).astype(float)
        test_feat["phone_area_freq"] = area_te.map(area_freq).fillna(0.0).astype(float)
        train_df["AREA_CODE"] = area_tr
        test_df["AREA_CODE"] = area_te

    date_cols = [c for c in train_df.columns if "DATE" in c.upper()]
    for col in date_cols:
        s_tr = pd.to_datetime(train_df[col], errors="coerce")
        s_te = pd.to_datetime(test_df[col], errors="coerce")
        train_feat[f"{col}_year"] = s_tr.dt.year.fillna(-1).astype(float)
        test_feat[f"{col}_year"] = s_te.dt.year.fillna(-1).astype(float)
        train_feat[f"{col}_month"] = s_tr.dt.month.fillna(-1).astype(float)
        test_feat[f"{col}_month"] = s_te.dt.month.fillna(-1).astype(float)
        train_feat[f"{col}_age"] = (2025.0 - s_tr.dt.year).fillna(-1).astype(float)
        test_feat[f"{col}_age"] = (2025.0 - s_te.dt.year).fillna(-1).astype(float)
        train_feat[f"{col}_is_null"] = s_tr.isna().astype(float)
        test_feat[f"{col}_is_null"] = s_te.isna().astype(float)

    # Composite categorical interaction columns
    for col1, col2, new_col in [
        ("STATE", "TYPE", "STATE_TYPE"),
        ("COUNTY", "TYPE", "COUNTY_TYPE"),
        ("ZIP3", "TYPE", "ZIP3_TYPE"),
    ]:
        if col1 in train_df.columns and col2 in train_df.columns:
            train_df[new_col] = train_df[col1].astype(str) + "_" + train_df[col2].astype(str)
            test_df[new_col] = test_df[col1].astype(str) + "_" + test_df[col2].astype(str)
            s_tr = train_df[new_col]
            s_te = test_df[new_col]
            freq_map = s_tr.value_counts(normalize=True).to_dict()
            train_feat[f"{new_col}_freq"] = s_tr.map(freq_map).fillna(0.0).astype(float)
            test_feat[f"{new_col}_freq"] = s_te.map(freq_map).fillna(0.0).astype(float)

    te_candidate_cols = [
        "STATE",
        "COUNTY",
        "COUNTYFIPS",
        "CITY",
        "ZIP3",
        "VAL_METHOD",
        "TYPE",
        "STATUS",
        "AREA_CODE",
        "STATE_TYPE",
        "COUNTY_TYPE",
        "ZIP3_TYPE",
    ]
    te_cat_cols = [c for c in te_candidate_cols if c in train_df.columns]

    state_col = next((c for c in train_df.columns if c.upper() == "STATE"), None)
    if state_col and lat_col and lon_col:
        grp_stats = train_df.groupby(state_col)[[lat_col, lon_col]].agg(
            ["mean", "std", "count"]
        )
        grp_stats.columns = [f"{state_col}_{c}_{stat}" for c, stat in grp_stats.columns]

        train_merged = train_df[[state_col]].merge(grp_stats, on=state_col, how="left")
        test_merged = test_df[[state_col]].merge(grp_stats, on=state_col, how="left")

        for col_name in grp_stats.columns:
            train_feat[col_name] = train_merged[col_name].fillna(0.0).values
            test_feat[col_name] = test_merged[col_name].fillna(0.0).values

        dlat_tr = (
            train_df[lat_col].values
            - train_feat[f"{state_col}_{lat_col}_mean"].values
        )
        dlon_tr = (
            train_df[lon_col].values
            - train_feat[f"{state_col}_{lon_col}_mean"].values
        )
        dlat_te = (
            test_df[lat_col].values
            - test_feat[f"{state_col}_{lat_col}_mean"].values
        )
        dlon_te = (
            test_df[lon_col].values
            - test_feat[f"{state_col}_{lon_col}_mean"].values
        )

        train_feat[f"dist_to_{state_col}_centroid"] = np.sqrt(dlat_tr**2 + dlon_tr**2)
        test_feat[f"dist_to_{state_col}_centroid"] = np.sqrt(dlat_te**2 + dlon_te**2)
        train_feat[f"bearing_to_{state_col}_centroid"] = np.arctan2(dlon_tr, dlat_tr)
        test_feat[f"bearing_to_{state_col}_centroid"] = np.arctan2(dlon_te, dlat_te)

        for radius_km in [5, 15, 50, 100]:
            den_col = f"density_within_{radius_km}km"
            if den_col in train_feat.columns:
                temp_df = pd.DataFrame(
                    {"state": train_df[state_col], "den": train_feat[den_col]}
                )
                state_den_mean = temp_df.groupby("state")["den"].transform("mean")
                state_den_map = temp_df.groupby("state")["den"].mean().to_dict()
                global_den_mean = train_feat[den_col].mean()

                train_feat[f"rel_{den_col}_to_state"] = train_feat[den_col] / (
                    state_den_mean + 1e-3
                )
                test_den_mean = (
                    test_df[state_col].map(state_den_map).fillna(global_den_mean)
                )
                test_feat[f"rel_{den_col}_to_state"] = test_feat[den_col] / (
                    test_den_mean + 1e-3
                )

    for c in train_df.columns:
        if c == target_col or c in [lat_col, lon_col]:
            continue
        if pd.api.types.is_numeric_dtype(train_df[c]):
            med_val = train_df[c].median()
            train_feat[f"num_{c}"] = train_df[c].fillna(med_val).values
            test_feat[f"num_{c}"] = test_df[c].fillna(med_val).values

    train_feat = train_feat.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    test_feat = test_feat.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    feature_cols = list(train_feat.columns)
    X_train_base = train_feat[feature_cols].values.astype(np.float32)
    X_test_base = test_feat[feature_cols].values.astype(np.float32)

    # -------------------------------------------------------------
    # 3. Stratified 5-Fold Cross-Validation
    # -------------------------------------------------------------
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    oof_lgb = np.zeros((n_train, num_classes), dtype=np.float32)
    oof_lgb_extra = np.zeros((n_train, num_classes), dtype=np.float32)
    oof_xgb = np.zeros((n_train, num_classes), dtype=np.float32)
    oof_cat = np.zeros((n_train, num_classes), dtype=np.float32)

    test_preds_lgb = np.zeros((n_test, num_classes), dtype=np.float32)
    test_preds_lgb_extra = np.zeros((n_test, num_classes), dtype=np.float32)
    test_preds_xgb = np.zeros((n_test, num_classes), dtype=np.float32)
    test_preds_cat = np.zeros((n_test, num_classes), dtype=np.float32)

    has_rads = lat_col is not None and lon_col is not None

    for fold, (trn_idx, val_idx) in enumerate(skf.split(train_feat, y_encoded)):
        y_train_fold = y_encoded[trn_idx]
        y_val_fold = y_encoded[val_idx]

        if has_rads:
            trn_spatial = compute_spatial_target_features(
                train_rads[trn_idx],
                y_train_fold,
                train_rads[trn_idx],
                is_self=True,
                ks=(7, 15, 30),
                sigmas_km=(5.0, 15.0, 50.0, 150.0, 500.0),
                num_classes=num_classes,
            )
            val_spatial = compute_spatial_target_features(
                train_rads[trn_idx],
                y_train_fold,
                train_rads[val_idx],
                is_self=False,
                ks=(7, 15, 30),
                sigmas_km=(5.0, 15.0, 50.0, 150.0, 500.0),
                num_classes=num_classes,
            )
            test_spatial = compute_spatial_target_features(
                train_rads[trn_idx],
                y_train_fold,
                test_rads,
                is_self=False,
                ks=(7, 15, 30),
                sigmas_km=(5.0, 15.0, 50.0, 150.0, 500.0),
                num_classes=num_classes,
            )
        else:
            trn_spatial = np.zeros((len(trn_idx), 0), dtype=np.float32)
            val_spatial = np.zeros((len(val_idx), 0), dtype=np.float32)
            test_spatial = np.zeros((n_test, 0), dtype=np.float32)

        if te_cat_cols:
            trn_te, val_te, test_te = compute_oof_target_encoding(
                train_cats=train_df.iloc[trn_idx][te_cat_cols],
                y_train=y_train_fold,
                val_cats=train_df.iloc[val_idx][te_cat_cols],
                test_cats=test_df[te_cat_cols],
                cat_cols=te_cat_cols,
                num_classes=num_classes,
                smoothing=10.0,
                n_inner_splits=5,
                seed=42 + fold,
            )
            X_train_fold = np.hstack([X_train_base[trn_idx], trn_spatial, trn_te])
            X_val_fold = np.hstack([X_train_base[val_idx], val_spatial, val_te])
            X_test_fold = np.hstack([X_test_base, test_spatial, test_te])
        else:
            X_train_fold = np.hstack([X_train_base[trn_idx], trn_spatial])
            X_val_fold = np.hstack([X_train_base[val_idx], val_spatial])
            X_test_fold = np.hstack([X_test_base, test_spatial])

        fold_features = X_train_fold.shape[1]

        # (A) LightGBM Classifier
        lgb_model = get_lgbm_classifier(num_classes=num_classes, random_state=42 + fold)
        lgb_callbacks = [
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0),
        ]
        lgb_model.fit(
            X_train_fold,
            y_train_fold,
            eval_set=[(X_val_fold, y_val_fold)],
            callbacks=lgb_callbacks,
        )

        val_lgb_probs = lgb_model.predict_proba(X_val_fold)
        oof_lgb[val_idx] = val_lgb_probs
        test_preds_lgb += lgb_model.predict_proba(X_test_fold).astype(np.float32) / 5.0

        # (B) XGBoost Classifier
        xgb_model = get_xgb_classifier(num_classes=num_classes, random_state=42 + fold)
        xgb_model.fit(
            X_train_fold,
            y_train_fold,
            eval_set=[(X_val_fold, y_val_fold)],
            verbose=False,
        )

        val_xgb_probs = xgb_model.predict_proba(X_val_fold)
        oof_xgb[val_idx] = val_xgb_probs
        test_preds_xgb += xgb_model.predict_proba(X_test_fold).astype(np.float32) / 5.0

        # (C) CatBoost Classifier
        cat_model = get_catboost_classifier(
            num_classes=num_classes, random_state=42 + fold
        )
        cat_model.fit(
            X_train_fold,
            y_train_fold,
            eval_set=(X_val_fold, y_val_fold),
            verbose=False,
        )

        val_cat_probs = cat_model.predict_proba(X_val_fold)
        oof_cat[val_idx] = val_cat_probs
        test_preds_cat += cat_model.predict_proba(X_test_fold).astype(np.float32) / 5.0

        # (D) Extra-trees LightGBM Classifier
        lgb_extra_model = get_lgbm_extra_classifier(
            num_classes=num_classes, random_state=42 + fold
        )
        lgb_extra_callbacks = [
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0),
        ]
        lgb_extra_model.fit(
            X_train_fold,
            y_train_fold,
            eval_set=[(X_val_fold, y_val_fold)],
            callbacks=lgb_extra_callbacks,
        )

        val_lgb_extra_probs = lgb_extra_model.predict_proba(X_val_fold)
        oof_lgb_extra[val_idx] = val_lgb_extra_probs
        test_preds_lgb_extra += (
            lgb_extra_model.predict_proba(X_test_fold).astype(np.float32) / 5.0
        )

        fold_acc_lgb = accuracy_score(y_val_fold, val_lgb_probs.argmax(axis=1))
        fold_acc_lgb_extra = accuracy_score(
            y_val_fold, val_lgb_extra_probs.argmax(axis=1)
        )
        fold_acc_xgb = accuracy_score(y_val_fold, val_xgb_probs.argmax(axis=1))
        fold_acc_cat = accuracy_score(y_val_fold, val_cat_probs.argmax(axis=1))
        print(
            f"Fold {fold + 1}/5 - LGB: {fold_acc_lgb:.4f} | LGB_ET: {fold_acc_lgb_extra:.4f} | XGB: {fold_acc_xgb:.4f} | CAT: {fold_acc_cat:.4f}"
        )

    # -------------------------------------------------------------
    # 4. Ensemble Weight Calibration & Submission
    # -------------------------------------------------------------
    def obj_weights(raw_w):
        w = np.exp(raw_w - np.max(raw_w))
        w = w / np.sum(w)
        blended = (
            w[0] * oof_lgb
            + w[1] * oof_lgb_extra
            + w[2] * oof_xgb
            + w[3] * oof_cat
        )
        preds = blended.argmax(axis=1)
        return -accuracy_score(y_encoded, preds)

    best_w_raw = np.zeros(4)
    best_acc = -obj_weights(best_w_raw)

    for init_raw in [
        np.array([0.0, 0.0, 0.0, 0.0]),
        np.array([1.0, 1.0, 1.0, 1.0]),
        np.array([1.5, 1.0, 1.0, 1.5]),
        np.array([2.0, 1.0, 0.5, 1.5]),
        np.array([1.0, 1.5, 1.0, 1.0]),
    ]:
        res = minimize(
            obj_weights,
            init_raw,
            method="Nelder-Mead",
            options={"maxiter": 500},
        )
        if -res.fun > best_acc:
            best_acc = -res.fun
            best_w_raw = res.x

    opt_w_exp = np.exp(best_w_raw - np.max(best_w_raw))
    opt_weights = opt_w_exp / np.sum(opt_w_exp)
    print(
        f"Optimized Weights -> LGB: {opt_weights[0]:.4f}, LGB_ET: {opt_weights[1]:.4f}, XGB: {opt_weights[2]:.4f}, CAT: {opt_weights[3]:.4f}"
    )

    oof_blend_prob = (
        opt_weights[0] * oof_lgb
        + opt_weights[1] * oof_lgb_extra
        + opt_weights[2] * oof_xgb
        + opt_weights[3] * oof_cat
    )

    def obj_multipliers(raw_m):
        m = np.exp(raw_m)
        scaled_prob = oof_blend_prob * m
        preds = scaled_prob.argmax(axis=1)
        return -accuracy_score(y_encoded, preds)

    init_m = np.zeros(num_classes)
    res_m = minimize(
        obj_multipliers,
        init_m,
        method="Powell",
        options={"maxiter": 300},
    )
    best_multipliers = np.exp(res_m.x)

    calibrated_oof_preds = (oof_blend_prob * best_multipliers).argmax(axis=1)
    final_score = accuracy_score(y_encoded, calibrated_oof_preds)

    test_blend_prob = (
        opt_weights[0] * test_preds_lgb
        + opt_weights[1] * test_preds_lgb_extra
        + opt_weights[2] * test_preds_xgb
        + opt_weights[3] * test_preds_cat
    )
    test_pred_classes_idx = (test_blend_prob * best_multipliers).argmax(axis=1)
    test_pred_labels = classes[test_pred_classes_idx]

    submission_df = pd.DataFrame(
        {"id": np.arange(n_test, dtype=int), "SIZE": test_pred_labels}
    )

    submission_path = os.path.join(submission_dir, "submission.csv")
    submission_df.to_csv(submission_path, index=False)

    print(f"Final Validation Score: {final_score:.6f}")


if __name__ == "__main__":
    main()
