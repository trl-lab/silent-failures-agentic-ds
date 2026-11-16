import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

try:
    from catboost import CatBoostClassifier

    USE_CATBOOST = True
except ImportError:
    from sklearn.ensemble import HistGradientBoostingClassifier

    USE_CATBOOST = False

# Ensure working directory exists
os.makedirs("./working", exist_ok=True)

# 1. Load Data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "SIZE"
y_raw = train_df[target_col].values

# Label encode target
le = LabelEncoder()
y = le.fit_transform(y_raw)
num_classes = len(le.classes_)

X_tr_raw = train_df.drop(columns=[target_col], errors="ignore").copy()
X_te_raw = test_df.drop(columns=[target_col], errors="ignore").copy()


# 2. General Feature Engineering Pipeline
def engineer_features(tr, te):
    combined = pd.concat([tr, te], axis=0, ignore_index=True)

    lat_cols = [c for c in combined.columns if "lat" in c.lower()]
    lon_cols = [c for c in combined.columns if "lon" in c.lower()]

    lat_col = lat_cols[0] if lat_cols else combined.columns[0]
    lon_col = lon_cols[0] if lon_cols else combined.columns[1]

    lat_vals = combined[lat_col].fillna(combined[lat_col].median())
    lon_vals = combined[lon_col].fillna(combined[lon_col].median())

    lat_rad = np.radians(lat_vals)
    lon_rad = np.radians(lon_vals)

    # 3D Cartesian coordinates
    combined["x_3d"] = np.cos(lat_rad) * np.cos(lon_rad)
    combined["y_3d"] = np.cos(lat_rad) * np.sin(lon_rad)
    combined["z_3d"] = np.sin(lat_rad)

    # Mercator coordinates
    combined["mercator_x"] = lon_rad
    combined["mercator_y"] = np.log(np.tan(np.pi / 4.0 + lat_rad / 2.0) + 1e-8)

    # Rotated Coordinates
    for angle in [15, 30, 45, 60, 75]:
        rad = np.radians(angle)
        combined[f"rot_{angle}_lat"] = lat_vals * np.cos(rad) - lon_vals * np.sin(rad)
        combined[f"rot_{angle}_lon"] = lat_vals * np.sin(rad) + lon_vals * np.cos(rad)

    # Distance & Angle to Centroid
    mean_lat = lat_vals.mean()
    mean_lon = lon_vals.mean()
    combined["dist_centroid"] = np.sqrt(
        (lat_vals - mean_lat) ** 2 + (lon_vals - mean_lon) ** 2
    )
    combined["angle_centroid"] = np.arctan2(lat_vals - mean_lat, lon_vals - mean_lon)

    # Spatial K-Means Cluster Centroids
    coords = np.column_stack([lat_vals, lon_vals])
    kmeans = KMeans(n_clusters=15, random_state=42, n_init=10)
    kmeans.fit(coords[: len(tr)])
    combined["kmeans_cluster"] = kmeans.predict(coords)
    cluster_dists = kmeans.transform(coords)
    for k in range(15):
        combined[f"cluster_dist_{k}"] = cluster_dists[:, k]

    # Name Keywords & Length features
    name_cols = [c for c in combined.columns if "name" in c.lower()]
    if name_cols:
        name_col = name_cols[0]
        names_str = combined[name_col].astype(str).str.lower()
        keywords = [
            "estates",
            "village",
            "resort",
            "park",
            "court",
            "gardens",
            "mobile",
            "rv",
            "manor",
            "ranch",
            "community",
        ]
        for kw in keywords:
            combined[f"kw_{kw}"] = names_str.str.contains(kw).astype(int)
        combined["name_len"] = names_str.str.len()
        combined["name_words"] = names_str.str.split().str.len()

    # Categorical Frequency Encodings
    cat_cols = combined.select_dtypes(include=["object", "category"]).columns.tolist()
    for col in cat_cols:
        freq_map = combined[col].value_counts(normalize=True).to_dict()
        combined[f"{col}_freq"] = combined[col].map(freq_map).fillna(0)

    # Split back to train/test
    tr_feats = combined.iloc[: len(tr)].copy()
    te_feats = combined.iloc[len(tr) :].copy()

    # Spatial k-NN Density Features
    tr_coords = np.column_stack([lat_vals[: len(tr)], lon_vals[: len(tr)]])
    te_coords = np.column_stack([lat_vals[len(tr) :], lon_vals[len(tr) :]])

    nn = NearestNeighbors(n_neighbors=21, algorithm="kd_tree")
    nn.fit(tr_coords)

    dists_tr, _ = nn.kneighbors(tr_coords)
    dists_te, _ = nn.kneighbors(te_coords)

    for k in [3, 5, 10, 20]:
        tr_feats[f"knn_dist_{k}"] = dists_tr[:, k]
        te_feats[f"knn_dist_{k}"] = dists_te[:, k]

    # Keep numeric features
    num_cols = tr_feats.select_dtypes(include=[np.number]).columns.tolist()

    X_tr_out = tr_feats[num_cols].fillna(-999).replace([np.inf, -np.inf], -999)
    X_te_out = te_feats[num_cols].fillna(-999).replace([np.inf, -np.inf], -999)

    return X_tr_out, X_te_out, tr_coords, te_coords


X_train, X_test, train_coords, test_coords = engineer_features(X_tr_raw, X_te_raw)


# Helper function to compute out-of-fold spatial target distributions
def compute_knn_target_props(
    tr_coords,
    tr_y,
    val_coords,
    te_coords,
    k_list=[10, 25, 50],
    n_classes=num_classes,
):
    max_k = max(k_list)
    nn = NearestNeighbors(n_neighbors=max_k + 1, algorithm="kd_tree")
    nn.fit(tr_coords)

    _, indices_tr = nn.kneighbors(tr_coords)
    _, indices_val = nn.kneighbors(val_coords)
    _, indices_te = nn.kneighbors(te_coords)

    tr_dict, val_dict, te_dict = {}, {}, {}

    for k in k_list:
        tr_neigh_y = tr_y[indices_tr[:, 1 : k + 1]]
        val_neigh_y = tr_y[indices_val[:, :k]]
        te_neigh_y = tr_y[indices_te[:, :k]]

        for c in range(n_classes):
            tr_dict[f"knn_target_k{k}_c{c}"] = (tr_neigh_y == c).mean(axis=1)
            val_dict[f"knn_target_k{k}_c{c}"] = (val_neigh_y == c).mean(axis=1)
            te_dict[f"knn_target_k{k}_c{c}"] = (te_neigh_y == c).mean(axis=1)

    return (
        pd.DataFrame(tr_dict),
        pd.DataFrame(val_dict),
        pd.DataFrame(te_dict),
    )


# 3. Stratified 5-Fold Cross Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_probs = np.zeros((len(X_train), num_classes))
test_probs = np.zeros((len(X_test), num_classes))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y)):
    X_tr_f = X_train.iloc[train_idx].copy()
    y_tr_f = y[train_idx]

    X_val_f = X_train.iloc[val_idx].copy()
    y_val_f = y[val_idx]

    X_te_f = X_test.copy()

    # Compute fold-specific spatial k-NN target features
    tr_target_feats, val_target_feats, te_target_feats = compute_knn_target_props(
        train_coords[train_idx],
        y_tr_f,
        train_coords[val_idx],
        test_coords,
        k_list=[10, 25, 50],
        n_classes=num_classes,
    )

    tr_target_feats.index = X_tr_f.index
    val_target_feats.index = X_val_f.index
    te_target_feats.index = X_te_f.index

    X_tr_f = pd.concat([X_tr_f, tr_target_feats], axis=1)
    X_val_f = pd.concat([X_val_f, val_target_feats], axis=1)
    X_te_f = pd.concat([X_te_f, te_target_feats], axis=1)

    if USE_CATBOOST:
        clf = CatBoostClassifier(
            iterations=1000,
            learning_rate=0.06,
            depth=6,
            loss_function="MultiClass",
            verbose=0,
            random_seed=42 + fold,
        )
        clf.fit(X_tr_f, y_tr_f, eval_set=(X_val_f, y_val_f), early_stopping_rounds=50)
    else:
        clf = HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.06, max_depth=6, random_state=42 + fold
        )
        clf.fit(X_tr_f, y_tr_f)

    oof_probs[val_idx] = clf.predict_proba(X_val_f)
    test_probs += clf.predict_proba(X_te_f) / skf.n_splits

# 4. Out-of-fold Evaluation
oof_preds = np.argmax(oof_probs, axis=1)
oof_accuracy = accuracy_score(y, oof_preds)
print(f"Validation Accuracy: {oof_accuracy:.5f}")

# 5. Generate Submission File
test_preds = np.argmax(test_probs, axis=1)
test_labels = le.inverse_transform(test_preds)

submission = pd.DataFrame({"id": np.arange(len(test_df)), "SIZE": test_labels})

submission.to_csv("./working/submission.csv", index=False)
