import os
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold

# Ensure output directories exist
os.makedirs("./submission", exist_ok=True)
os.makedirs("./working", exist_ok=True)

print("Loading dataset from ./input...")
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Mark datasets for unified feature engineering
train_df["is_train"] = 1
test_df["is_train"] = 0

combined_df = pd.concat([train_df, test_df], ignore_index=True)

print("Engineering 3D spatial geometry and distance features...")

def load_and_merge_structures(df):
    cols_lower = {c.lower(): c for c in df.columns}
    coord_pairs = [
        ("x_0", "y_0", "z_0", "x_1", "y_1", "z_1"),
        ("x0", "y0", "z0", "x1", "y1", "z1"),
        ("x_atom0", "y_atom0", "z_atom0", "x_atom1", "y_atom1", "z_atom1"),
        ("x_atom_0", "y_atom_0", "z_atom_0", "x_atom_1", "y_atom_1", "z_atom_1"),
    ]
    for x0, y0, z0, x1, y1, z1 in coord_pairs:
        if all(k in cols_lower for k in [x0, y0, z0, x1, y1, z1]):
            return df.rename(columns={
                cols_lower[x0]: "x_0", cols_lower[y0]: "y_0", cols_lower[z0]: "z_0",
                cols_lower[x1]: "x_1", cols_lower[y1]: "y_1", cols_lower[z1]: "z_1"
            })

    struct_path = None
    for search_dir in ["./input", "./input/original", ".", "./working"]:
        if os.path.exists(search_dir):
            for root, _, files in os.walk(search_dir):
                for f in files:
                    if ("structure" in f.lower() or "molecule" in f.lower()) and f.endswith((".parquet", ".csv")):
                        struct_path = os.path.join(root, f)
                        break
                if struct_path:
                    break
        if struct_path:
            break

    if struct_path is not None:
        print(f"Loading structure data from {struct_path}...")
        structures = pd.read_parquet(struct_path) if struct_path.endswith(".parquet") else pd.read_csv(struct_path)
        s_cols = {c.lower(): c for c in structures.columns}
        mol_col = s_cols.get("molecule_name", s_cols.get("molecule"))
        idx_col = s_cols.get("atom_index", s_cols.get("atom_idx"))
        atom_col = s_cols.get("atom", s_cols.get("atom_type"))
        x_col, y_col, z_col = s_cols.get("x"), s_cols.get("y"), s_cols.get("z")

        if mol_col and idx_col and x_col and y_col and z_col:
            s0_cols = [mol_col, idx_col, x_col, y_col, z_col]
            s0_names = ["molecule_name", "atom_index_0", "x_0", "y_0", "z_0"]
            if atom_col:
                s0_cols.append(atom_col)
                s0_names.append("atom_0")
            s0 = structures[s0_cols].copy()
            s0.columns = s0_names

            s1_cols = [mol_col, idx_col, x_col, y_col, z_col]
            s1_names = ["molecule_name", "atom_index_1", "x_1", "y_1", "z_1"]
            if atom_col:
                s1_cols.append(atom_col)
                s1_names.append("atom_1")
            s1 = structures[s1_cols].copy()
            s1.columns = s1_names

            for col in ["x_0", "y_0", "z_0", "x_1", "y_1", "z_1", "atom_0", "atom_1"]:
                if col in df.columns:
                    df = df.drop(columns=[col])

            df = df.merge(s0, on=["molecule_name", "atom_index_0"], how="left")
            df = df.merge(s1, on=["molecule_name", "atom_index_1"], how="left")

    for col in ["x_0", "y_0", "z_0", "x_1", "y_1", "z_1"]:
        if col not in df.columns:
            df[col] = 0.0
    for col in ["atom_0", "atom_1"]:
        if col not in df.columns:
            df[col] = "C"

    return df

def extract_structures(df):
    struct_path = None
    for search_dir in ["./input", "./input/original", ".", "./working"]:
        if os.path.exists(search_dir):
            for root, _, files in os.walk(search_dir):
                for f in files:
                    if ("structure" in f.lower() or "molecule" in f.lower()) and f.endswith((".parquet", ".csv")):
                        struct_path = os.path.join(root, f)
                        break
                if struct_path:
                    break
        if struct_path:
            break

    if struct_path is not None:
        print(f"Loading structure data from {struct_path}...")
        structures = pd.read_parquet(struct_path) if struct_path.endswith(".parquet") else pd.read_csv(struct_path)
        s_cols = {c.lower(): c for c in structures.columns}
        mol_col = s_cols.get("molecule_name", s_cols.get("molecule"))
        idx_col = s_cols.get("atom_index", s_cols.get("atom_idx"))
        atom_col = s_cols.get("atom", s_cols.get("atom_type"))
        x_col, y_col, z_col = s_cols.get("x"), s_cols.get("y"), s_cols.get("z")
        res = structures[[mol_col, idx_col, atom_col, x_col, y_col, z_col]].copy()
        res.columns = ["molecule_name", "atom_index", "atom", "x", "y", "z"]
        return res
    else:
        s0 = df[["molecule_name", "atom_index_0", "atom_0", "x_0", "y_0", "z_0"]].rename(
            columns={"atom_index_0": "atom_index", "atom_0": "atom", "x_0": "x", "y_0": "y", "z_0": "z"}
        )
        s1 = df[["molecule_name", "atom_index_1", "atom_1", "x_1", "y_1", "z_1"]].rename(
            columns={"atom_index_1": "atom_index", "atom_1": "atom", "x_1": "x", "y_1": "y", "z_1": "z"}
        )
        structures = pd.concat([s0, s1], ignore_index=True).drop_duplicates(
            subset=["molecule_name", "atom_index"]
        ).reset_index(drop=True)
        return structures


def compute_neighbor_geometry_features(combined_df, structures):
    print("Computing species counts and nearest-neighbor geometry features...")

    # 1. Species counts per molecule
    species_counts = structures.groupby(["molecule_name", "atom"]).size().unstack(fill_value=0)
    species_counts.columns = [f"num_atom_{col}" for col in species_counts.columns]
    num_atoms = structures.groupby("molecule_name")["atom_index"].count().rename("num_atoms")

    mol_species = species_counts.join(num_atoms).reset_index()
    combined_df = combined_df.merge(mol_species, on="molecule_name", how="left")

    # 2. Compute 1st and 2nd nearest neighbors per atom
    s_pairs = structures.merge(structures, on="molecule_name", suffixes=("", "_nn"))
    s_pairs = s_pairs[s_pairs["atom_index"] != s_pairs["atom_index_nn"]].copy()

    dx_nn = s_pairs["x_nn"] - s_pairs["x"]
    dy_nn = s_pairs["y_nn"] - s_pairs["y"]
    dz_nn = s_pairs["z_nn"] - s_pairs["z"]
    s_pairs["dist_nn"] = np.sqrt(dx_nn**2 + dy_nn**2 + dz_nn**2)
    s_pairs["dx_nn"] = dx_nn
    s_pairs["dy_nn"] = dy_nn
    s_pairs["dz_nn"] = dz_nn

    s_pairs = s_pairs.sort_values(["molecule_name", "atom_index", "dist_nn"])

    nn1 = s_pairs.groupby(["molecule_name", "atom_index"]).first().reset_index()
    nn2 = s_pairs.groupby(["molecule_name", "atom_index"]).nth(1).reset_index()

    nn1_0 = nn1[["molecule_name", "atom_index", "dist_nn", "dx_nn", "dy_nn", "dz_nn", "atom_nn"]].copy()
    nn1_0.columns = ["molecule_name", "atom_index_0", "atom0_nn1_dist", "atom0_nn1_dx", "atom0_nn1_dy", "atom0_nn1_dz", "atom0_nn1_type"]

    nn2_0 = nn2[["molecule_name", "atom_index", "dist_nn"]].copy()
    nn2_0.columns = ["molecule_name", "atom_index_0", "atom0_nn2_dist"]

    nn1_1 = nn1[["molecule_name", "atom_index", "dist_nn", "dx_nn", "dy_nn", "dz_nn", "atom_nn"]].copy()
    nn1_1.columns = ["molecule_name", "atom_index_1", "atom1_nn1_dist", "atom1_nn1_dx", "atom1_nn1_dy", "atom1_nn1_dz", "atom1_nn1_type"]

    nn2_1 = nn2[["molecule_name", "atom_index", "dist_nn"]].copy()
    nn2_1.columns = ["molecule_name", "atom_index_1", "atom1_nn2_dist"]

    combined_df = combined_df.merge(nn1_0, on=["molecule_name", "atom_index_0"], how="left")
    combined_df = combined_df.merge(nn2_0, on=["molecule_name", "atom_index_0"], how="left")
    combined_df = combined_df.merge(nn1_1, on=["molecule_name", "atom_index_1"], how="left")
    combined_df = combined_df.merge(nn2_1, on=["molecule_name", "atom_index_1"], how="left")

    # 3. Compute cosine angles
    dx_01 = combined_df["x_1"] - combined_df["x_0"]
    dy_01 = combined_df["y_1"] - combined_df["y_0"]
    dz_01 = combined_df["z_1"] - combined_df["z_0"]
    dist_01 = combined_df["dist"] + 1e-6

    dot_0 = dx_01 * combined_df["atom0_nn1_dx"] + dy_01 * combined_df["atom0_nn1_dy"] + dz_01 * combined_df["atom0_nn1_dz"]
    combined_df["cos_bond_nn0"] = dot_0 / (dist_01 * combined_df["atom0_nn1_dist"] + 1e-6)

    dot_1 = (-dx_01) * combined_df["atom1_nn1_dx"] + (-dy_01) * combined_df["atom1_nn1_dy"] + (-dz_01) * combined_df["atom1_nn1_dz"]
    combined_df["cos_bond_nn1"] = dot_1 / (dist_01 * combined_df["atom1_nn1_dist"] + 1e-6)

    dot_01 = combined_df["atom0_nn1_dx"] * combined_df["atom1_nn1_dx"] + combined_df["atom0_nn1_dy"] * combined_df["atom1_nn1_dy"] + combined_df["atom0_nn1_dz"] * combined_df["atom1_nn1_dz"]
    combined_df["cos_nn0_nn1"] = dot_01 / (combined_df["atom0_nn1_dist"] * combined_df["atom1_nn1_dist"] + 1e-6)

    combined_df = combined_df.drop(columns=[
        "atom0_nn1_dx", "atom0_nn1_dy", "atom0_nn1_dz",
        "atom1_nn1_dx", "atom1_nn1_dy", "atom1_nn1_dz"
    ])

    return combined_df


def compute_midpoint_and_common_neighbor_features(df, structures):
    print("Computing common neighbor geometry and midpoint proximity features...")

    df["mid_x"] = (df["x_0"] + df["x_1"]) / 2.0
    df["mid_y"] = (df["y_0"] + df["y_1"]) / 2.0
    df["mid_z"] = (df["z_0"] + df["z_1"]) / 2.0

    pair_cols = ["molecule_name", "atom_index_0", "atom_index_1", "x_0", "y_0", "z_0", "x_1", "y_1", "z_1", "mid_x", "mid_y", "mid_z"]
    df_sub = df[pair_cols].copy()

    merged = df_sub.merge(
        structures[["molecule_name", "atom_index", "x", "y", "z"]].rename(
            columns={"atom_index": "atom_index_k", "x": "x_k", "y": "y_k", "z": "z_k"}
        ),
        on="molecule_name"
    )

    merged = merged[(merged["atom_index_k"] != merged["atom_index_0"]) & (merged["atom_index_k"] != merged["atom_index_1"])].copy()

    dx_mid = merged["x_k"] - merged["mid_x"]
    dy_mid = merged["y_k"] - merged["mid_y"]
    dz_mid = merged["z_k"] - merged["mid_z"]
    merged["dist_k_mid"] = np.sqrt(dx_mid**2 + dy_mid**2 + dz_mid**2)

    dx_0k = merged["x_0"] - merged["x_k"]
    dy_0k = merged["y_0"] - merged["y_k"]
    dz_0k = merged["z_0"] - merged["z_k"]
    d_0k = np.sqrt(dx_0k**2 + dy_0k**2 + dz_0k**2)

    dx_1k = merged["x_1"] - merged["x_k"]
    dy_1k = merged["y_1"] - merged["y_k"]
    dz_1k = merged["z_1"] - merged["z_k"]
    d_1k = np.sqrt(dx_1k**2 + dy_1k**2 + dz_1k**2)

    merged["d_0k"] = d_0k
    merged["d_1k"] = d_1k
    merged["sum_dist_k"] = d_0k + d_1k

    mid_aggs = merged.groupby(["molecule_name", "atom_index_0", "atom_index_1"])["dist_k_mid"].agg(
        mid_dist_min="min",
        mid_dist_mean="mean"
    ).reset_index()

    best_k_idx = merged.groupby(["molecule_name", "atom_index_0", "atom_index_1"])["sum_dist_k"].idxmin()
    best_k = merged.loc[best_k_idx].copy()

    dot_k = (best_k["x_0"] - best_k["x_k"]) * (best_k["x_1"] - best_k["x_k"]) + \
            (best_k["y_0"] - best_k["y_k"]) * (best_k["y_1"] - best_k["y_k"]) + \
            (best_k["z_0"] - best_k["z_k"]) * (best_k["z_1"] - best_k["z_k"])
    best_k["common_neighbor_cos_angle"] = dot_k / (best_k["d_0k"] * best_k["d_1k"] + 1e-6)

    best_k_features = best_k[["molecule_name", "atom_index_0", "atom_index_1", "sum_dist_k", "d_0k", "d_1k", "common_neighbor_cos_angle"]].rename(
        columns={"sum_dist_k": "common_neighbor_dist_sum", "d_0k": "cn_dist_0", "d_1k": "cn_dist_1"}
    )

    df = df.merge(mid_aggs, on=["molecule_name", "atom_index_0", "atom_index_1"], how="left")
    df = df.merge(best_k_features, on=["molecule_name", "atom_index_0", "atom_index_1"], how="left")

    df = df.drop(columns=["mid_x", "mid_y", "mid_z"])
    return df


combined_df = load_and_merge_structures(combined_df)
structures_table = extract_structures(combined_df)

# 1. Coordinate displacements and Euclidean distance
dx = combined_df["x_0"] - combined_df["x_1"]
dy = combined_df["y_0"] - combined_df["y_1"]
dz = combined_df["z_0"] - combined_df["z_1"]

dist = np.sqrt(dx**2 + dy**2 + dz**2)
combined_df["dist"] = dist
combined_df["abs_dx"] = np.abs(dx)
combined_df["abs_dy"] = np.abs(dy)
combined_df["abs_dz"] = np.abs(dz)

# 2. Non-linear power-law potential proxies
combined_df["inv_dist"] = 1.0 / (dist + 1e-6)
combined_df["inv_dist_sq"] = combined_df["inv_dist"] ** 2
combined_df["inv_dist_cube"] = combined_df["inv_dist"] ** 3
combined_df["inv_dist_6"] = combined_df["inv_dist"] ** 6

# 3. Bond count feature
combined_df["num_bonds"] = pd.to_numeric(combined_df["type"].astype(str).str[0], errors="coerce").fillna(1).astype(int)

# 4. Molecular centroid radial features
molecule_centroids = combined_df.groupby("molecule_name")[
    ["x_0", "y_0", "z_0"]
].transform("mean")

combined_df["r0_mid"] = np.sqrt(
    (combined_df["x_0"] - molecule_centroids["x_0"]) ** 2
    + (combined_df["y_0"] - molecule_centroids["y_0"]) ** 2
    + (combined_df["z_0"] - molecule_centroids["z_0"]) ** 2
)
combined_df["r1_mid"] = np.sqrt(
    (combined_df["x_1"] - molecule_centroids["x_0"]) ** 2
    + (combined_df["y_1"] - molecule_centroids["y_0"]) ** 2
    + (combined_df["z_1"] - molecule_centroids["z_0"]) ** 2
)

dot_prod = (
    (combined_df["x_0"] - molecule_centroids["x_0"])
    * (combined_df["x_1"] - molecule_centroids["x_0"])
    + (combined_df["y_0"] - molecule_centroids["y_0"])
    * (combined_df["y_1"] - molecule_centroids["y_0"])
    + (combined_df["z_0"] - molecule_centroids["z_0"])
    * (combined_df["z_1"] - molecule_centroids["z_0"])
)
combined_df["cos_centroid_angle"] = dot_prod / (
    combined_df["r0_mid"] * combined_df["r1_mid"] + 1e-6
)

print("Computing contextual group aggregations...")

# 5. Molecule level distance statistics
mol_dist = (
    combined_df.groupby("molecule_name")["dist"]
    .agg(["mean", "min", "max", "std"])
    .reset_index()
)
mol_dist.columns = [
    "molecule_name",
    "mol_dist_mean",
    "mol_dist_min",
    "mol_dist_max",
    "mol_dist_std",
]
combined_df = combined_df.merge(mol_dist, on="molecule_name", how="left")

# 6. Molecule x Type aggregations
mol_type_dist = (
    combined_df.groupby(["molecule_name", "type"], observed=False)["dist"]
    .agg(["mean", "min", "max"])
    .reset_index()
)
mol_type_dist.columns = [
    "molecule_name",
    "type",
    "mol_type_dist_mean",
    "mol_type_dist_min",
    "mol_type_dist_max",
]
combined_df = combined_df.merge(mol_type_dist, on=["molecule_name", "type"], how="left")

# 7. Atom 0 and Atom 1 aggregations
atom0_agg = (
    combined_df.groupby(["molecule_name", "atom_index_0"])["dist"]
    .agg(["count", "mean", "min", "max"])
    .reset_index()
)
atom0_agg.columns = [
    "molecule_name",
    "atom_index_0",
    "atom0_count",
    "atom0_dist_mean",
    "atom0_dist_min",
    "atom0_dist_max",
]
combined_df = combined_df.merge(
    atom0_agg, on=["molecule_name", "atom_index_0"], how="left"
)

atom1_agg = (
    combined_df.groupby(["molecule_name", "atom_index_1"])["dist"]
    .agg(["count", "mean", "min", "max"])
    .reset_index()
)
atom1_agg.columns = [
    "molecule_name",
    "atom_index_1",
    "atom1_count",
    "atom1_dist_mean",
    "atom1_dist_min",
    "atom1_dist_max",
]
combined_df = combined_df.merge(
    atom1_agg, on=["molecule_name", "atom_index_1"], how="left"
)

# 8. Type normalized global distance differences (placeholder for post-split computation)
combined_df["dist_to_type_mean"] = 0.0
combined_df["dist_to_type_ratio"] = 0.0

# 9. Nearest-neighbor atomic geometry & species counts
combined_df = compute_neighbor_geometry_features(combined_df, structures_table)

# 10. Midpoint spatial proximity & common neighbor geometry features
combined_df = compute_midpoint_and_common_neighbor_features(combined_df, structures_table)

# Separate back into train and test sets
train_processed = (
    combined_df[combined_df["is_train"] == 1]
    .drop(columns=["is_train"])
    .reset_index(drop=True)
)
test_processed = (
    combined_df[combined_df["is_train"] == 0]
    .drop(columns=["is_train"])
    .reset_index(drop=True)
)

print("Splitting training data using GroupKFold on molecule_name...")
gkf = GroupKFold(n_splits=5)
train_idx, val_idx = next(
    gkf.split(train_processed, groups=train_processed["molecule_name"])
)

train_split = train_processed.iloc[train_idx].reset_index(drop=True)
val_split = train_processed.iloc[val_idx].reset_index(drop=True)

# Compute type-level distance statistics strictly on training split to prevent leakage
type_stats = (
    train_split.groupby("type", observed=False)["dist"]
    .agg(["mean", "std"])
    .reset_index()
)
type_stats.columns = ["type", "type_dist_mean_global", "type_dist_std_global"]
type_map = type_stats.set_index("type")["type_dist_mean_global"].to_dict()

for df in [train_split, val_split, test_processed]:
    mean_dist = df["type"].map(type_map)
    df["dist_to_type_mean"] = df["dist"] - mean_dist
    df["dist_to_type_ratio"] = df["dist"] / (mean_dist + 1e-6)

target_col = "scalar_coupling_constant"
drop_cols = ["id", "molecule_name", "type", target_col]
feature_cols = [c for c in train_split.columns if c not in drop_cols]

# Cast categorical and object columns
for c in feature_cols:
    if train_split[c].dtype == "object" or c in ["atom_0", "atom_1"]:
        train_split[c] = train_split[c].astype("category")
        val_split[c] = val_split[c].astype("category")
        test_processed[c] = test_processed[c].astype("category")


def get_type_lgb_params(coupling_type: str) -> dict:
    base_params = {
        "objective": "regression_l1",
        "metric": "mae",
        "boosting_type": "gbdt",
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }
    configs = {
        "1JHC": {
            "num_leaves": 255,
            "max_depth": 12,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.7,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "min_child_samples": 30,
        },
        "2JHC": {
            "num_leaves": 127,
            "max_depth": 10,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.2,
            "reg_lambda": 1.0,
            "min_child_samples": 20,
        },
        "3JHC": {
            "num_leaves": 127,
            "max_depth": 10,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.2,
            "reg_lambda": 1.0,
            "min_child_samples": 20,
        },
        "1JHN": {
            "num_leaves": 127,
            "max_depth": 10,
            "learning_rate": 0.03,
            "subsample": 0.85,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 0.5,
            "min_child_samples": 15,
        },
        "2JHN": {
            "num_leaves": 63,
            "max_depth": 8,
            "learning_rate": 0.03,
            "subsample": 0.85,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.5,
            "reg_lambda": 1.0,
            "min_child_samples": 15,
        },
        "3JHN": {
            "num_leaves": 63,
            "max_depth": 8,
            "learning_rate": 0.03,
            "subsample": 0.85,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.5,
            "reg_lambda": 1.0,
            "min_child_samples": 15,
        },
        "2HH": {
            "num_leaves": 127,
            "max_depth": 10,
            "learning_rate": 0.04,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.2,
            "reg_lambda": 1.0,
            "min_child_samples": 20,
        },
        "3HH": {
            "num_leaves": 127,
            "max_depth": 10,
            "learning_rate": 0.04,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.2,
            "reg_lambda": 1.0,
            "min_child_samples": 20,
        },
    }
    params = base_params.copy()
    params.update(configs.get(coupling_type, configs["2JHC"]))
    return params


val_preds = np.zeros(len(val_split), dtype=np.float32)
test_preds = np.zeros(len(test_processed), dtype=np.float32)
unique_types = np.unique(train_split["type"])

print("Training type-stratified LightGBM models...")

for coupling_type in unique_types:
    tr_mask = train_split["type"] == coupling_type
    val_mask = val_split["type"] == coupling_type
    test_mask = test_processed["type"] == coupling_type

    X_tr = train_split.loc[tr_mask, feature_cols]
    y_tr = train_split.loc[tr_mask, target_col]

    X_val = val_split.loc[val_mask, feature_cols]
    y_val = val_split.loc[val_mask, target_col]

    X_test = test_processed.loc[test_mask, feature_cols]

    dtrain = lgb.Dataset(X_tr, label=y_tr)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

    params = get_type_lgb_params(str(coupling_type))

    model = lgb.train(
        params,
        dtrain,
        valid_sets=[dtrain, dval],
        valid_names=["train", "val"],
        num_boost_round=6000,
        callbacks=[
            lgb.early_stopping(stopping_rounds=100, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )

    type_val_pred = model.predict(X_val, num_iteration=model.best_iteration)
    type_test_pred = model.predict(X_test, num_iteration=model.best_iteration)

    val_preds[val_mask.values] = type_val_pred
    test_preds[test_mask.values] = type_test_pred

    type_mae = mean_absolute_error(y_val, type_val_pred)
    type_log_mae = np.log(max(type_mae, 1e-9))
    print(
        f"Type {str(coupling_type):<5} | Best Iter: {model.best_iteration:<5} | Val MAE: {type_mae:.4f} | Log MAE: {type_log_mae:.4f}"
    )

overall_mae = float(mean_absolute_error(val_split[target_col].values, val_preds))

submission_df = pd.DataFrame(
    {"id": test_processed["id"].astype(int), target_col: test_preds}
)
submission_df.to_csv("./submission/submission.csv", index=False)

assert len(submission_df) == len(test_processed), "Submission row count mismatch!"
assert not submission_df[target_col].isnull().any(), "Submission contains NaNs!"

print("Submission saved successfully to ./submission/submission.csv")
print(f"Final Validation Score: {overall_mae}")